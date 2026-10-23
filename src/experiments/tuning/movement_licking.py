"""Descriptive running-bout and licking profiles, with temporal support audits."""
import json
import warnings

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.io import resolve
from src.data.recording import Recording
from src.data.task import load_task, recording_path, spike_counts, source_fingerprint
from src.experiments.tuning.running_correlation import bin_count, common_lag_mask
from src.experiments.tuning.running_examples import segments, smooth_segments


def mean_finite(values, axis=0):
    n = np.isfinite(values).sum(axis=axis)
    return np.divide(np.nansum(values, axis=axis), n,
                     out=np.full(np.shape(n), np.nan, dtype=float), where=n > 0)


def safe_quantile(values, q, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanquantile(values, q, axis=axis)


def correlations(x, y, mask):
    """Pearson correlations of x with rows of y over shared valid samples."""
    use = mask & np.isfinite(x)
    if use.sum() < 3:
        return np.full(len(y), np.nan)
    a = np.asarray(x[use], float)
    b = np.asarray(y[:, use], float)
    a -= a.mean()
    b -= b.mean(axis=1, keepdims=True)
    den = np.sqrt(np.dot(a, a) * np.sum(b * b, axis=1))
    return np.divide(b @ a, den, out=np.full(len(y), np.nan), where=den > 1e-10)


def mean_rate(y, mask):
    return np.asarray(y[:, mask].mean(axis=1), float) if mask.any() else np.full(len(y), np.nan)


def moving_average(values, width):
    """Centered boxcar; validity of the entire support is checked separately."""
    if width % 2 != 1:
        raise ValueError('Centered boxcar must contain an odd number of bins')
    half = width // 2
    arr = np.atleast_2d(values)
    prefix = np.pad(np.cumsum(arr, axis=1, dtype=float), ((0, 0), (1, 0)))
    result = np.full(arr.shape, np.nan)
    result[:, half:arr.shape[1] - half] = (prefix[:, width:] - prefix[:, :-width]) / width
    return result[0] if values.ndim == 1 else result


def bootstrap_weights(n, repeats, block, rng):
    weights = np.zeros((repeats, n), float)
    if n == 0:
        return weights
    block = min(block, n)
    for i in range(repeats):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        indices = (starts[:, None] + np.arange(block)).ravel()[:n]
        weights[i] = np.bincount(indices, minlength=n)
    return weights


def mean_uncertainty(values, weights):
    """Values are observation x neuron; bands are descriptive resampling spread."""
    valid = np.isfinite(values)
    num = weights @ np.nan_to_num(values)
    den = weights @ valid.astype(float)
    sampled = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    low, high = safe_quantile(sampled, [.025, .975])
    supported = valid.sum(axis=0) >= 5
    low[~supported] = np.nan
    high[~supported] = np.nan
    return low, high


def delete_one_range(values):
    valid = np.isfinite(values)
    total, n = np.nansum(values, axis=0), valid.sum(axis=0)
    denominator = n[None, :] - valid
    remaining = np.divide(total[None, :] - np.nan_to_num(values), denominator,
                          out=np.full_like(values, np.nan), where=denominator > 0)
    return safe_quantile(remaining, [0, 1])


def running_bouts(speed, valid, cfg, dt, factor=1.):
    smoothed = smooth_segments(speed, valid, float(cfg.detection_sigma_s) / dt)
    high, low = float(cfg.onset_cm_s) * factor, float(cfg.offset_cm_s) * factor
    running = np.zeros(len(speed), bool)
    # Enter only above high, exit below low; missing samples end a bout.
    active = False
    for i, value in enumerate(smoothed):
        if not np.isfinite(value) or not valid[i]:
            active = False
        elif active:
            active = value >= low
        else:
            active = value >= high
        running[i] = active
    raw = segments(running)
    gap = bin_count(cfg.merge_gap_s, dt, 'merge_gap_s')
    for (_, b), (c, _) in zip(raw, raw[1:]):
        if c - b <= gap and valid[b:c].all():
            running[b:c] = True
    records = []
    for number, (a, b) in enumerate(segments(running)):
        duration = (b - a) * dt
        records.append(dict(bout_id=number, start_bin=a, stop_bin=b,
                            onset_s=a * dt, offset_s=b * dt, duration_s=duration,
                            coverage=float(valid[a:b].mean()),
                            mean_speed_cm_s=float(np.nanmean(speed[a:b])),
                            speed_sd_cm_s=float(np.nanstd(speed[a:b])),
                            peak_speed_cm_s=float(np.nanmax(speed[a:b])),
                            eligible=duration >= float(cfg.min_duration_s)
                            and valid[a:b].mean() >= float(cfg.min_bout_coverage)))
    audit = pd.DataFrame(records)
    if audit.empty:
        raise ValueError('No detected running bouts')
    return audit.loc[audit.eligible].copy().reset_index(drop=True), audit, smoothed


def bout_features(y, speed, valid, bouts, cfg, dt):
    shape = (len(bouts), len(y))
    result = {k: np.full(shape, np.nan) for k in
              ['baseline_hz', 'running_hz', 'delta_hz', 'modulation', 'within_bout_r',
               'early_delta_hz', 'late_delta_hz', 'late_minus_early_hz']}
    baseline_a, baseline_b = [bin_count(v, dt, 'baseline') for v in cfg.baseline_s]
    baseline_samples = []
    for j, bout in bouts.iterrows():
        a, b = int(bout.start_bin), int(bout.stop_bin)
        mask = np.zeros(len(speed), bool)
        mask[a:b] = valid[a:b]
        run = mean_rate(y, mask)
        result['running_hz'][j] = run
        baseline = np.zeros(len(speed), bool)
        if a + baseline_a >= 0:
            lo, hi = a + baseline_a, a + baseline_b
            baseline[lo:hi] = valid[lo:hi] & (speed[lo:hi] <= float(cfg.offset_cm_s))
        baseline_samples.append(int(baseline.sum()))
        if baseline.sum() * dt >= float(cfg.min_baseline_s):
            base = mean_rate(y, baseline)
            delta = run - base
            result['baseline_hz'][j], result['delta_hz'][j] = base, delta
            result['modulation'][j] = np.divide(delta, run + base,
                                                out=np.full(len(y), np.nan), where=run + base > 0)
            third = max(1, (b - a) // 3)
            early = y[:, a:a + third][:, valid[a:a + third]].mean(axis=1)
            late = y[:, b - third:b][:, valid[b - third:b]].mean(axis=1)
            result['early_delta_hz'][j] = early - base
            result['late_delta_hz'][j] = late - base
            result['late_minus_early_hz'][j] = late - early
        if bout.speed_sd_cm_s >= float(cfg.min_speed_sd_cm_s):
            result['within_bout_r'][j] = correlations(speed, y, mask)
    bouts['baseline_samples'] = baseline_samples
    bouts['baseline_available'] = np.asarray(baseline_samples) * dt >= float(cfg.min_baseline_s)
    return result


def bout_lag_correlations(y, speed, valid, bouts, lags, dt, minimum_speed_sd):
    """Secondary lag comparison: every pair remains inside the same bout."""
    offsets = [bin_count(lag, dt, 'running lag') for lag in lags]
    result = np.full((len(bouts), len(y), len(lags)), np.nan)
    for j, bout in bouts.iterrows():
        a, b = int(bout.start_bin), int(bout.stop_bin)
        common = common_lag_mask(valid[a:b], offsets)
        anchor = a + np.flatnonzero(common)
        if len(anchor) < 3 or np.std(speed[anchor]) < minimum_speed_sd:
            continue
        for k, offset in enumerate(offsets):
            result[j, :, k] = correlations(speed[anchor], y[:, anchor + offset],
                                            np.ones(len(anchor), bool))
    return result


def prepare(cfg):
    info, units, trials = load_task(cfg)
    units = units.loc[units.ccf_acronym.isin(list(cfg.regions))
                      & units.decoder_label.isin(list(cfg.sorting))].copy()
    units['region_order'] = units.ccf_acronym.map({r: i for i, r in enumerate(cfg.regions)})
    units = units.sort_values(['region_order', 'depth', 'id'], ascending=[True, False, True]).reset_index(drop=True)
    if not units.global_unit_id.is_unique or not units.global_unit_id.str.startswith(f'{cfg.animal}/{cfg.session}/').all():
        raise ValueError('Unit identities do not match requested session')
    dt = float(cfg.bin_size_s)
    duration = float(info['duration_s'])
    n = int(np.floor(duration / dt))
    edges = np.arange(n + 1) * dt
    time = (edges[:-1] + edges[1:]) / 2
    valid = np.ones(n, bool)
    valid &= edges[:-1] >= trials.loc[trials.usable, 'start_time'].min()
    padding = info['excluded_trial_padding_s']
    for trial in trials.loc[~trials.usable].itertuples():
        valid &= ((edges[1:] <= trial.start_time - padding['before'])
                  | (edges[:-1] >= trial.stop_time + padding['after']))
    recording = Recording(recording_path(cfg))
    timestamps, data = recording.signal('speed')
    timestamps = timestamps - info['clock']['origin_s']
    data = np.asarray(data, float).ravel()
    if len(data) != len(timestamps) or not np.all(np.diff(timestamps) > 0):
        raise ValueError('Invalid velocity clock')
    vi = np.floor(timestamps / dt).astype(int)
    keep = (vi >= 0) & (vi < n) & np.isfinite(data)
    counts = np.bincount(vi[keep], minlength=n)
    speed = np.divide(np.bincount(vi[keep], weights=abs(data[keep]), minlength=n), counts,
                      out=np.full(n, np.nan), where=counts > 0)
    expected = dt / np.median(np.diff(timestamps))
    speed_valid = valid & (counts >= int(np.ceil(.8 * expected - 1e-7)))
    licks = []
    for side in ['left', 'right']:
        events = recording.events(f'{side}_lick') - info['clock']['origin_s']
        licks.append(np.histogram(events[np.isfinite(events)], edges)[0])
    y = spike_counts(recording, units, edges).astype(np.float32) / dt
    recording.close()
    units.insert(0, 'row', np.arange(len(units)))
    units['eligible_spikes'] = np.rint(y[:, valid].sum(axis=1) * dt).astype(int)
    units['mean_rate_hz'] = y[:, valid].mean(axis=1)
    units['low_spike_support'] = units.eligible_spikes < int(cfg.validation.minimum_spikes)
    return info, units, time, y, speed, np.asarray(licks), valid, speed_valid


def behavior_profile(x, y, valid, bins, dt):
    labels = np.digitize(x, bins) - 1
    rates = np.full((len(y), len(bins) - 1), np.nan)
    exposure = np.zeros(len(bins) - 1)
    for j in range(len(bins) - 1):
        mask = valid & np.isfinite(x) & (labels == j)
        exposure[j] = mask.sum() * dt
        rates[:, j] = mean_rate(y, mask)
    return rates, exposure


def lick_statistics(x, y, valid, blocks):
    stats = []
    for mask in blocks:
        use = valid & mask
        xx, yy = x[use], np.asarray(y[:, use], float)
        n = np.full(len(y), use.sum())
        stats.append(np.stack([n, np.full(len(y), xx.sum()), yy.sum(axis=1),
                              np.full(len(y), np.dot(xx, xx)), (yy * yy).sum(axis=1), yy @ xx]))
    return np.asarray(stats)


def r_from_stats(stats):
    n, sx, sy, sxx, syy, sxy = np.moveaxis(stats, -2, 0)
    n = np.maximum(n, 1)
    vx, vy = sxx - sx * sx / n, syy - sy * sy / n
    ok = (vx > 1e-9 * np.maximum(sxx, 1)) & (vy > 1e-9 * np.maximum(syy, 1)) & (n >= 3)
    return np.divide(sxy - sx * sy / n, np.sqrt(np.maximum(vx * vy, 0)),
                     out=np.full_like(sy, np.nan), where=ok)


def run(cfg):
    from src.cli import require_compute
    from src.experiments.tuning.movement_licking_plot import plot
    require_compute('movement_licking', cfg)
    out = resolve(cfg.output)
    if cfg.stage == 'plot':
        plot(out)
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output is not empty; use a fresh output directory')
    info, units, time, y, speed, lick_counts, valid, speed_valid = prepare(cfg)
    dt = float(cfg.bin_size_s)
    duration = float(info['duration_s'])
    rng = np.random.default_rng(int(cfg.validation.seed))
    repeats = int(cfg.validation.bootstrap_repetitions)
    print(f'Prepared {len(units)} units and {len(time)} task bins', flush=True)
    bouts, audit, detected_speed = running_bouts(speed, speed_valid, cfg.running, dt)
    effects = bout_features(y, speed, speed_valid, bouts, cfg.running, dt)
    print(f'Detected {len(bouts)} bouts; {int(bouts.baseline_available.sum())} with quiet baseline', flush=True)
    features = units.copy()
    fields = ['baseline_hz', 'running_hz', 'delta_hz', 'modulation', 'within_bout_r',
              'early_delta_hz', 'late_delta_hz', 'late_minus_early_hz']
    weights = bootstrap_weights(len(bouts), repeats, int(cfg.validation.bootstrap_consecutive_bouts), rng)
    for name in fields:
        values = effects[name]
        features['run_' + name] = mean_finite(values)
        features['run_' + name + '_n_bouts'] = np.isfinite(values).sum(axis=0)
    features['run_delta_median_hz'] = safe_quantile(effects['delta_hz'], .5)
    features['run_paired_running_hz'] = mean_finite(np.where(
        np.isfinite(effects['baseline_hz']), effects['running_hz'], np.nan))
    features['run_delta_positive_fraction'] = np.divide((effects['delta_hz'] > 0).sum(axis=0),
        np.isfinite(effects['delta_hz']).sum(axis=0), out=np.full(len(units), np.nan),
        where=np.isfinite(effects['delta_hz']).sum(axis=0) > 0)
    for name in ['delta_hz', 'within_bout_r', 'late_delta_hz', 'late_minus_early_hz']:
        low, high = mean_uncertainty(effects[name], weights)
        features[f'run_{name}_resample_low'], features[f'run_{name}_resample_high'] = low, high
        low, high = delete_one_range(effects[name])
        features[f'run_{name}_deletion_low'], features[f'run_{name}_deletion_high'] = low, high
    features['speed_whole_session_r'] = correlations(speed, y, speed_valid)
    duration_r = np.full(len(units), np.nan)
    for j in range(len(units)):
        ok = np.isfinite(effects['delta_hz'][:, j])
        if ok.sum() >= int(cfg.validation.minimum_bouts):
            duration_r[j] = correlations(bouts.duration_s.to_numpy(), effects['delta_hz'][:, j][None, :], ok)[0]
    features['bout_duration_delta_r'] = duration_r
    run_lagged = bout_lag_correlations(y, speed, speed_valid, bouts, cfg.lags_s, dt,
                                      float(cfg.running.min_speed_sd_cm_s))
    for k, lag in enumerate(cfg.lags_s):
        features[f'run_within_bout_r_common_lag_{lag:+g}s'] = mean_finite(run_lagged[:, :, k])
        features[f'run_within_bout_r_common_lag_{lag:+g}s_n_bouts'] = np.isfinite(run_lagged[:, :, k]).sum(axis=0)

    # Temporal summaries are a description of changing responses, not pass/fail tests.
    epoch_rows = []
    for e, label in enumerate(['early', 'middle', 'late']):
        pick = (bouts.onset_s >= duration * e / 3) & (bouts.onset_s < duration * (e + 1) / 3)
        part = units[['global_unit_id', 'id']].copy()
        part['epoch'] = label
        part['n_bouts'] = int(pick.sum())
        for name in ['delta_hz', 'within_bout_r', 'modulation']:
            part['run_' + name] = mean_finite(effects[name][pick])
            part['run_' + name + '_n_bouts'] = np.isfinite(effects[name][pick]).sum(axis=0)
        epoch_rows.append(part)
    epoch_table = pd.concat(epoch_rows, ignore_index=True)

    # Onset curves and duration-normalized phases retain equal bout weights.
    offsets = np.arange(bin_count(cfg.running.onset_window_s[0], dt, 'onset window'),
                        bin_count(cfg.running.onset_window_s[1], dt, 'onset window'))
    onset = np.full((len(bouts), len(units), len(offsets)), np.nan, np.float32)
    phase = np.full((len(bouts), len(units), int(cfg.running.phase_bins)), np.nan, np.float32)
    sigma = float(cfg.running.onset_smoothing_sigma_s) / dt
    for j, bout in bouts.iterrows():
        if not bout.baseline_available:
            continue
        a, b = int(bout.start_bin), int(bout.stop_bin)
        indices = a + offsets
        inside = (indices >= 0) & (indices < len(time))
        # Mask neighboring bouts in peri-event displays rather than count them twice.
        if j > 0:
            inside &= indices >= int(bouts.iloc[j - 1].stop_bin)
        if j + 1 < len(bouts):
            inside &= indices < int(bouts.iloc[j + 1].start_bin)
        indices_safe = indices.clip(0, len(time) - 1)
        inside &= speed_valid[indices_safe]
        for u in range(len(units)):
            curve = smooth_segments(y[u, indices_safe].astype(float), inside, sigma)
            onset[j, u] = curve - effects['baseline_hz'][j, u]
        breaks = np.linspace(a, b, int(cfg.running.phase_bins) + 1).astype(int)
        for k, (lo, hi) in enumerate(zip(breaks, breaks[1:])):
            if hi > lo and speed_valid[lo:hi].any():
                phase[j, :, k] = y[:, lo:hi][:, speed_valid[lo:hi]].mean(axis=1) - effects['baseline_hz'][j]
    onset_mean = mean_finite(onset)
    phase_mean = mean_finite(phase)
    post = offsets >= 0
    for j in range(len(units)):
        curve = onset_mean[j, post]
        if (np.isfinite(curve).any() and np.nanmax(abs(curve)) > 1e-8
                and features.run_delta_hz_n_bouts.iloc[j] >= cfg.validation.minimum_bouts):
            k = int(np.nanargmax(abs(curve)))
            features.loc[j, 'run_onset_peak_delta_hz'] = curve[k]
            features.loc[j, 'run_onset_peak_time_s'] = offsets[post][k] * dt

    sensitivity = []
    for factor in cfg.running.sensitivity_factors:
        if float(factor) == 1:
            delta, nb = effects['delta_hz'], len(bouts)
        else:
            alternative, _, _ = running_bouts(speed, speed_valid, cfg.running, dt, float(factor))
            alternate_cfg = OmegaConf.create(OmegaConf.to_container(cfg.running))
            alternate_cfg.offset_cm_s = float(cfg.running.offset_cm_s) * float(factor)
            delta = bout_features(y, speed, speed_valid, alternative, alternate_cfg, dt)['delta_hz']
            nb = len(alternative)
        part = units[['global_unit_id', 'id']].copy()
        part['threshold_factor'], part['n_detected_bouts'] = float(factor), nb
        part['run_delta_hz'] = mean_finite(delta)
        part['n_supported_bouts'] = np.isfinite(delta).sum(axis=0)
        sensitivity.append(part)

    width = bin_count(cfg.lick_rate_window_s, dt, 'lick_rate_window_s')
    half = width // 2
    lags = np.asarray(cfg.lags_s, float)
    lag_bins = [bin_count(x, dt, 'lag') for x in lags]
    common = common_lag_mask(valid, [min(lag_bins) - half, max(lag_bins) + half])
    lick_rates = np.asarray([moving_average(counts / dt, width) for counts in lick_counts])
    smoothed_y = moving_average(y, width)
    block_width = float(cfg.validation.time_block_s)
    blocks = [(time >= a) & (time < a + block_width) for a in np.arange(0, duration, block_width)]
    keep_blocks = [i for i, m in enumerate(blocks) if (common & m).sum() * dt >= block_width * cfg.validation.min_block_coverage]
    supported_blocks = [blocks[i] for i in keep_blocks]
    if not supported_blocks:
        raise ValueError('No valid licking uncertainty blocks')
    lick_weights = bootstrap_weights(len(supported_blocks), repeats, 1, rng)
    local_tables, lick_curves, lick_exposure, lick_resamples = [], [], [], []
    lick_edges = np.asarray([-.1, .1, 2.1, 4.1, 6.1, 8.1, 10.1, np.inf])
    lick_lags = np.full((2, len(units), len(lags)), np.nan)
    for side, x in enumerate(lick_rates):
        name = ['left', 'right'][side]
        for k, lag in enumerate(lag_bins):
            anchor = np.flatnonzero(common)
            lick_lags[side, :, k] = correlations(x[anchor], smoothed_y[:, anchor + lag], np.ones(len(anchor), bool))
        zero = np.flatnonzero(lags == 0)
        if len(zero) != 1:
            raise ValueError('Exactly one zero lag required')
        features[f'{name}_lick_r'] = lick_lags[side, :, zero[0]]
        for k, lag in enumerate(lags):
            features[f'{name}_lick_r_lag_{lag:+g}s'] = lick_lags[side, :, k]
        stats = lick_statistics(x, smoothed_y, common, supported_blocks)
        resampled = np.einsum('rb,bsu->rsu', lick_weights, stats, optimize=True)
        rr = r_from_stats(resampled)
        low, high = safe_quantile(rr, [.025, .975])
        features[f'{name}_lick_r_resample_low'], features[f'{name}_lick_r_resample_high'] = low, high
        features[f'{name}_lick_r_block_supported'] = r_from_stats(stats.sum(axis=0))
        deletion = r_from_stats(stats.sum(axis=0)[None, :, :] - stats)
        low, high = safe_quantile(deletion, [0, 1])
        features[f'{name}_lick_r_deletion_low'], features[f'{name}_lick_r_deletion_high'] = low, high
        lick_resamples.append(rr.astype(np.float32))
        local = r_from_stats(stats)
        for j, original_block in enumerate(keep_blocks):
            part = units[['global_unit_id', 'id']].copy()
            part['side'], part['block'] = name, original_block
            part['center_s'] = (original_block + .5) * block_width
            part['r'] = local[j]
            part['paired_bins'] = int((common & blocks[original_block]).sum())
            local_tables.append(part)
        for e, label in enumerate(['early', 'middle', 'late']):
            mask = common & (time >= duration * e / 3) & (time < duration * (e + 1) / 3)
            epoch_table.loc[epoch_table.epoch.eq(label), f'{name}_lick_r'] = correlations(x, smoothed_y, mask)
            epoch_table.loc[epoch_table.epoch.eq(label), 'lick_paired_bins'] = int(mask.sum())
        curve, exposure = behavior_profile(x, y, common, lick_edges, dt)
        lick_curves.append(curve)
        lick_exposure.append(exposure)
        positive = common & (x > .1)
        features[f'{name}_lick_positive_bin_fraction'] = positive.sum() / common.sum()
    speed_edges = np.asarray([0, .1, .5, 1, 2, 4, 6, 8, np.inf])
    speed_curve, speed_exposure = behavior_profile(speed, y, speed_valid, speed_edges, dt)
    print('Computed running responses, threshold sensitivity, and licking profiles', flush=True)
    long_bouts = []
    for j, bout in bouts.iterrows():
        part = units[['global_unit_id', 'id']].copy()
        for field in ['bout_id', 'onset_s', 'duration_s']:
            part[field] = bout[field]
        for field in fields:
            part[field] = effects[field][j]
        for k, lag in enumerate(cfg.lags_s):
            part[f'within_bout_r_common_lag_{lag:+g}s'] = run_lagged[j, :, k]
        long_bouts.append(part)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / 'config.yaml')
    features.to_csv(out / 'features.csv', index=False)
    units.to_csv(out / 'units.csv', index=False)
    bouts.to_csv(out / 'running_bouts.csv', index=False)
    audit.to_csv(out / 'running_bout_audit.csv', index=False)
    pd.concat(long_bouts, ignore_index=True).to_csv(out / 'unit_bout_features.csv', index=False)
    epoch_table.to_csv(out / 'epoch_features.csv', index=False)
    pd.concat(sensitivity, ignore_index=True).to_csv(out / 'running_threshold_sensitivity.csv', index=False)
    pd.concat(local_tables, ignore_index=True).to_csv(out / 'lick_block_correlations.csv', index=False)
    np.savez_compressed(out / 'profiles.npz', global_unit_id=units.global_unit_id.to_numpy(str),
        onset_time_s=offsets * dt, onset_delta_hz=onset_mean, onset_per_bout_delta_hz=onset,
        onset_support=np.isfinite(onset).sum(axis=0), phase=np.linspace(0, 1, int(cfg.running.phase_bins), endpoint=False),
        phase_delta_hz=phase_mean, phase_per_bout_delta_hz=phase,
        run_lag_s=np.asarray(cfg.lags_s), run_per_bout_lag_r=run_lagged,
        lick_rate_hz=np.asarray([0, 2, 4, 6, 8, 10, 12]), lick_rate_edges=lick_edges,
        lick_firing_hz=np.asarray(lick_curves), lick_exposure_s=np.asarray(lick_exposure),
        lick_lag_s=lags, lick_correlations=lick_lags, lick_resampled_r=np.asarray(lick_resamples),
        speed_edges_cm_s=speed_edges, speed_firing_hz=speed_curve, speed_exposure_s=speed_exposure)
    np.savez_compressed(out / 'behavior.npz', time_s=time, speed_cm_s=speed, detection_speed_cm_s=detected_speed,
                        lick_rate_hz=lick_rates, lick_counts=lick_counts, task_valid=valid,
                        speed_valid=speed_valid, lick_common_valid=common)
    quiet = speed[speed_valid & (speed <= .25)]
    median = float(np.median(quiet))
    method = dict(recording_id=info['recording_id'], animal=cfg.animal, session=cfg.session, source=info['recording_h5'],
        n_units=len(units), n_bouts=len(bouts), n_baseline_bouts=int(bouts.baseline_available.sum()),
        duration_s=duration, eligible_task_s=float(valid.sum() * dt), eligible_speed_s=float(speed_valid.sum() * dt),
        licking_paired_s=float(common.sum() * dt), lick_events=[int(c[valid].sum()) for c in lick_counts],
        quiet_speed_median=median, quiet_speed_robust_sd=float(1.4826 * np.median(abs(quiet - median))),
        primary='Running bouts and quiet-baseline responses; left/right lick-rate correlations and profiles',
        running='Hysteresis 1 cm/s onset, 0.5 cm/s offset; speed smoothed sigma 0.2 s; merge <=1 s; minimum 2 s; no maximum bout duration',
        baseline='Quiet samples <=0.5 cm/s in -5 to -1 s before each bout; require 2 s; missing baseline does not exclude within-bout speed correlation',
        within_bout='Pearson r on unsmoothed 100-ms firing/speed bins within each bout; speed SD >=0.25 cm/s; arithmetic mean gives equal weight to supported bouts',
        running_lags='Secondary -0.5/0/+0.5-s correlations use identical anchor samples that keep all lagged pairs within each bout; speed SD >=0.25 cm/s on those anchors; no broad display smoothing',
        persistence='Compare first and last thirds of each bout against the same quiet baseline; also retain 20-bin phase curves',
        licking='All detected left/right licks including choice licks; centered 500-ms boxcar rates and firing, evaluated every 100 ms',
        lag='corr(behavior(t), firing(t+lag)); common valid support for -0.5/0/+0.5 s; zero lag primary',
        profiles='Mean observed firing in each behavior range; zero activity kept separate; exposure saved; final lick-rate category is >10 Hz, labeled 12 only as a plotting location',
        validation='Data coverage, 400 descriptive paired resamples, deletion influence, early/middle/late comparisons, and running threshold sensitivity',
        uncertainty='Running resamples groups of three consecutive bouts; licking resamples complete 60-s blocks. Percentile bands are descriptive, assume enough local stability, and are not significance tests or guaranteed confidence intervals under nonstationarity.',
        uncertainty_scope='Lick resampling and deletion use complete blocks with >=80% paired coverage; block-supported point estimate is saved separately from full-session correlation',
        onset='All baseline-supported bouts; neighboring qualifying bouts masked; sigma 0.2 s for display; signed peak over 0-10 s is descriptive and uncertain for flat responses',
        eligibility='Continuous task recordings, including ordinary trial boundaries; ambiguous trial intervals with saved padding excluded; no interpolation across true missing movement',
        independence='Overlapping features may reflect shared task or movement structure; no unique encoding or causal interpretation',
        heldout='No held-out acceptance gate or binary significance classification',
        sorting=units.decoder_label.value_counts().to_dict(),
        recording_source=source_fingerprint(recording_path(cfg)))
    (out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    plot(out)
    print(f'Completed {out}', flush=True)
    return out
