"""Review an existing movement/licking analysis without altering its features."""
import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.io import resolve
from src.experiments.tuning.movement_licking import (
    bootstrap_weights, correlations, lick_statistics, mean_finite,
    mean_uncertainty, moving_average, prepare, r_from_stats, safe_quantile,
)
from src.experiments.tuning.running_correlation import common_lag_mask


def sign_supported(value, low, high):
    return (np.sign(value) * low > 0) & (np.sign(value) * high > 0)


def run(cfg):
    from src.cli import require_compute
    require_compute('movement_licking_review', cfg)
    source, out = resolve(cfg.source), resolve(cfg.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Review output must be empty')
    original = OmegaConf.load(source / 'config.yaml')
    f = pd.read_csv(source / 'features.csv')
    e = pd.read_csv(source / 'epoch_features.csv')
    bouts = pd.read_csv(source / 'running_bouts.csv')
    per_bout = pd.read_csv(source / 'unit_bout_features.csv')
    thresholds = pd.read_csv(source / 'running_threshold_sensitivity.csv')
    info, units, time, y, speed, lick_counts, valid, speed_valid = prepare(original)
    if not np.array_equal(units.global_unit_id, f.global_unit_id):
        raise ValueError('Unit order changed')
    if not np.array_equal(units.eligible_spikes, f.eligible_spikes):
        raise ValueError('Input spike counts changed')
    dt, duration = float(original.bin_size_s), float(info['duration_s'])
    if not np.isclose(dt, .1):
        raise ValueError('This review requires the original 100-ms analysis grid')
    rng = np.random.default_rng(int(cfg.seed))
    repeats = int(cfg.resamples)
    review = f.copy()
    review['primary_neuron'] = f.decoder_label.eq('sua') & f.default_qc & ~f.low_spike_support
    quality_groups = {
        'QC single units': review.primary_neuron,
        'Other single units': f.decoder_label.eq('sua') & ~review.primary_neuron,
        'QC multiunits': f.decoder_label.eq('mua') & f.default_qc & ~f.low_spike_support,
        'Other multiunits': f.decoder_label.eq('mua') & ~(f.default_qc & ~f.low_spike_support),
    }
    identities = f.global_unit_id
    effects = {}
    for field in ['delta_hz', 'within_bout_r', 'late_delta_hz', 'late_minus_early_hz']:
        effects[field] = per_bout.pivot(index='bout_id', columns='global_unit_id', values=field).reindex(
            index=bouts.bout_id, columns=identities).to_numpy()
        np.testing.assert_allclose(mean_finite(effects[field]), f['run_' + field], atol=1e-7, equal_nan=True)

    # Vary the resampling grouping; these ranges are not null-hypothesis tests.
    resampling_rows = []
    for group in [1, 3, 5]:
        weights = bootstrap_weights(len(bouts), repeats, group, rng)
        for field, values in effects.items():
            low, high = mean_uncertainty(values, weights)
            part = pd.DataFrame({'global_unit_id': identities, 'feature': 'run_' + field,
                                 'group': group, 'group_units': 'consecutive bouts',
                                 'low': low, 'high': high})
            resampling_rows.append(part)
            review[f'run_{field}_resample_group{group}_same_sign'] = sign_supported(f['run_' + field], low, high)
    print('Reviewed bout resampling and original feature arithmetic', flush=True)

    # Inspect spike support within bouts, and compare 100-ms with 500-ms bins.
    spike_counts = np.zeros((len(bouts), len(f)), int)
    rebin_r = np.full((2, len(bouts), len(f)), np.nan)
    run_mask = np.zeros(len(time), bool)
    baseline_mask = np.zeros(len(time), bool)
    baseline_offsets = [int(round(float(value)/dt)) for value in original.running.baseline_s]
    for j, bout in bouts.iterrows():
        a, b = int(bout.start_bin), int(bout.stop_bin)
        run_mask[a:b] = speed_valid[a:b]
        spike_counts[j] = np.rint(y[:, a:b].sum(axis=1) * dt).astype(int)
        if bout.baseline_available:
            lo, hi = a + baseline_offsets[0], a + baseline_offsets[1]
            baseline_mask[lo:hi] |= speed_valid[lo:hi] & (speed[lo:hi] <= float(original.running.offset_cm_s))
        n = ((b - a) // 5) * 5
        for k, factor in enumerate([1, 5]):
            x = speed[a:a+n].reshape(-1, factor).mean(axis=1)
            firing = y[:, a:a+n].reshape(len(f), -1, factor).mean(axis=2)
            if len(x) >= 3 and np.std(x) >= .25:
                rebin_r[k, j] = correlations(x, firing, np.ones(len(x), bool))
    review['run_spikes'] = np.rint(y[:, run_mask].sum(axis=1) * dt).astype(int)
    review['quiet_baseline_spikes_unique'] = np.rint(y[:, baseline_mask].sum(axis=1) * dt).astype(int)
    for minimum in [5, 10]:
        supported = np.where(spike_counts >= minimum, effects['within_bout_r'], np.nan)
        review[f'run_r_min{minimum}spikes'] = mean_finite(supported)
        review[f'run_r_min{minimum}spikes_n_bouts'] = np.isfinite(supported).sum(axis=0)
    review['run_r_rebin100ms'] = mean_finite(rebin_r[0])
    review['run_r_rebin500ms'] = mean_finite(rebin_r[1])
    review['run_r_rebin_same_sign'] = np.sign(review.run_r_rebin500ms) == np.sign(f.run_within_bout_r)
    alternatives = thresholds.pivot(index='global_unit_id', columns='threshold_factor', values='run_delta_hz').reindex(identities)
    review['run_threshold_same_sign'] = (np.sign(alternatives.to_numpy()) == np.sign(f.run_delta_hz.to_numpy()[:, None])).all(axis=1)
    review['run_threshold_max_change_hz'] = abs(alternatives.to_numpy() - f.run_delta_hz.to_numpy()[:, None]).max(axis=1)

    # Lick-window and resampling-block sensitivity on fixed, common valid support.
    common = common_lag_mask(valid, [-8, 8])  # widest 700-ms window plus +/-500-ms lag
    smoothing_rows, centered_rows = [], []
    occupancy = []
    for width in [3, 5, 7]:
        yy = moving_average(y, width)
        for side, counts in zip(['left', 'right'], lick_counts):
            xx = moving_average(counts / dt, width)
            rr = correlations(xx, yy, common)
            review[f'{side}_lick_r_window{width*100}ms'] = rr
            smoothing_rows.append(pd.DataFrame({'global_unit_id': identities, 'side': side,
                                                'window_ms': width*100, 'r': rr}))
            if width != 5:
                continue
            for length in [30, 60, 120]:
                masks = [(time >= start) & (time < start + length) for start in np.arange(0, duration, length)]
                masks = [mask for mask in masks if np.sum(mask & common) * dt >= .8 * length]
                stats = lick_statistics(xx, yy, common, masks)
                weights = bootstrap_weights(len(masks), repeats, 1, rng)
                samples = r_from_stats(np.einsum('rb,bsu->rsu', weights, stats, optimize=True))
                low, high = safe_quantile(samples, [.025, .975])
                resampling_rows.append(pd.DataFrame({'global_unit_id': identities,
                    'feature': side + '_lick_r', 'group': length, 'group_units': 'seconds', 'low': low, 'high': high}))
                review[f'{side}_lick_resample_{length}s_same_sign'] = sign_supported(f[side + '_lick_r'], low, high)
                review[f'{side}_lick_{length}s_point'] = r_from_stats(stats.sum(axis=0))
                if length == 60:
                    # Remove each block's mean from both signals: diagnostic of slower cofluctuation.
                    n, sx, sy, sxx, syy, sxy = np.moveaxis(stats, -2, 0)
                    vx, vy = (sxx-sx*sx/n).sum(axis=0), (syy-sy*sy/n).sum(axis=0)
                    cov = (sxy-sx*sy/n).sum(axis=0)
                    centered = np.divide(cov, np.sqrt(np.maximum(vx*vy, 0)),
                        out=np.full(len(f), np.nan), where=(vx > 1e-8) & (vy > 1e-8))
                    review[f'{side}_lick_r_within60s'] = centered
                    local = r_from_stats(stats)
                    review[f'{side}_lick_defined_60s_blocks'] = np.isfinite(local).sum(axis=0)
                    review[f'{side}_lick_fraction_blocks_same_sign'] = mean_finite(
                        np.where(np.isfinite(local), np.sign(local) == np.sign(f[side+'_lick_r'].to_numpy()), np.nan))
                    centered_rows.append(dict(side=side, eligible_blocks=len(masks),
                        blocks_with_licking=int(sum(np.any(xx[common & mask] > 0) for mask in masks))))
            for epoch, label in enumerate(['early', 'middle', 'late']):
                mask = common & (time >= duration*epoch/3) & (time < duration*(epoch+1)/3)
                occupancy.append(dict(side=side, epoch=label, seconds=float(mask.sum()*dt),
                    lick_events=int(counts[mask].sum()), active_lick_s=float((mask & (xx > 0)).sum()*dt)))
    print('Reviewed lick-rate windows, time blocks, and within-minute associations', flush=True)

    # Temporal changes are kept as information, not used to reject changing neurons.
    epoch_rates = []
    for epoch, label in enumerate(['early', 'middle', 'late']):
        mask = valid & (time >= duration*epoch/3) & (time < duration*(epoch+1)/3)
        part = pd.DataFrame({'global_unit_id': identities, 'epoch': label,
                            'mean_rate_hz': y[:, mask].mean(axis=1),
                            'spikes': np.rint(y[:, mask].sum(axis=1)*dt).astype(int)})
        epoch_rates.append(part)
    epoch_rates = pd.concat(epoch_rates, ignore_index=True)
    for name in ['run_delta_hz', 'run_within_bout_r', 'left_lick_r', 'right_lick_r']:
        values = e.pivot(index='global_unit_id', columns='epoch', values=name).reindex(identities)
        review[name + '_early'] = values.early.to_numpy()
        review[name + '_late'] = values.late.to_numpy()
        review[name + '_early_late_same_sign'] = (np.sign(values.early.to_numpy()) == np.sign(values.late.to_numpy())) & values.early.notna().to_numpy() & values.late.notna().to_numpy()
        if 'lick' in name:
            review[name + '_all_epochs_same_sign'] = (np.sign(values[['early', 'middle', 'late']].to_numpy()) == np.sign(f[name].to_numpy()[:, None])).all(axis=1)

    # Peak timing and bout duration are secondary features: quantify their instability.
    with np.load(source / 'profiles.npz') as p:
        curves = p['onset_per_bout_delta_hz']
        post = p['onset_time_s'] >= 0
        onset_t = p['onset_time_s'][post]
    primary_rows = np.flatnonzero(review.primary_neuron)
    selected = curves[:, primary_rows][:, :, post].reshape(len(bouts), -1)
    weights = bootstrap_weights(len(bouts), repeats, 5, rng)
    num = weights @ np.nan_to_num(selected)
    den = weights @ np.isfinite(selected).astype(float)
    sampled_curves = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0).reshape(repeats, len(primary_rows), len(onset_t))
    peak = np.argmax(np.where(np.isfinite(sampled_curves), abs(sampled_curves), -np.inf), axis=2)
    peak_times = onset_t[peak]
    low, high = safe_quantile(peak_times, [.025, .975])
    review.loc[primary_rows, 'onset_peak_time_resample_low_s'] = low
    review.loc[primary_rows, 'onset_peak_time_resample_high_s'] = high
    review.loc[primary_rows, 'onset_peak_time_resample_span_s'] = high-low
    delta = effects['delta_hz']
    finite = np.isfinite(delta)
    x = bouts.duration_s.to_numpy()[:, None]
    safe_delta = np.nan_to_num(delta)
    stats = np.stack([finite, x*finite, safe_delta, x*x*finite, safe_delta*safe_delta, x*safe_delta], axis=1).astype(float)
    rr = r_from_stats(np.einsum('rb,bsu->rsu', weights, stats, optimize=True))
    low, high = safe_quantile(rr, [.025, .975])
    review['duration_r_resample_low'], review['duration_r_resample_high'] = low, high

    # Transparent descriptive screens, not tuned/untuned labels or p-value claims.
    run_sign = sign_supported(f.run_delta_hz, f.run_delta_hz_deletion_low, f.run_delta_hz_deletion_high)
    for group in [3, 5]:
        run_sign &= review[f'run_delta_hz_resample_group{group}_same_sign']
    review['run_response_checks'] = run_sign & review.run_threshold_same_sign & f.run_delta_hz_n_bouts.ge(10)
    speed_sign = sign_supported(f.run_within_bout_r, f.run_within_bout_r_deletion_low, f.run_within_bout_r_deletion_high)
    for group in [3, 5]:
        speed_sign &= review[f'run_within_bout_r_resample_group{group}_same_sign']
    review['speed_checks'] = speed_sign & review.run_r_rebin_same_sign & review.run_r_min5spikes_n_bouts.ge(10) & (np.sign(review.run_r_min5spikes) == np.sign(f.run_within_bout_r))
    for side in ['left', 'right']:
        name = side + '_lick_r'
        check = sign_supported(f[name], f[name+'_deletion_low'], f[name+'_deletion_high'])
        for length in [30, 60, 120]:
            check &= review[f'{side}_lick_resample_{length}s_same_sign']
        for width in [300, 500, 700]:
            check &= np.sign(review[f'{side}_lick_r_window{width}ms']) == np.sign(f[name])
        review[side+'_lick_checks'] = check & review[f'{side}_lick_defined_60s_blocks'].ge(10)
    screen_rows = []
    features = [('Running response', 'run_delta_hz', 'run_response_checks', [.5, 1., 2.]),
                ('Within-bout speed', 'run_within_bout_r', 'speed_checks', [.1, .2, .3]),
                ('Left licking', 'left_lick_r', 'left_lick_checks', [.1, .2, .3]),
                ('Right licking', 'right_lick_r', 'right_lick_checks', [.1, .2, .3])]
    for label, field, checks, cutoffs in features:
        for i, cutoff in enumerate(cutoffs):
            magnitude = f[field].abs().ge(cutoff)
            candidate = magnitude & review[checks]
            if i == 1:
                review[field+'_candidate'] = candidate
            for group, mask in quality_groups.items():
                screen_rows.append(dict(feature=label, cutoff=cutoff, group=group, n_units=int(mask.sum()),
                    magnitude_only=int((mask & magnitude).sum()), after_checks=int((mask & candidate).sum())))
    screens = pd.DataFrame(screen_rows)
    candidates = review[[field+'_candidate' for _,field,_,_ in features]].any(axis=1)
    review['any_candidate'] = candidates
    q = review.primary_neuron
    main_counts = screens.loc[screens.group.eq('QC single units') &
        ((screens.feature.eq('Running response') & screens.cutoff.eq(1)) |
         (~screens.feature.eq('Running response') & screens.cutoff.eq(.2)))]
    summary = dict(n_units=len(f), qc_single_units=int(q.sum()),
        qc_single_regions=f.loc[q].ccf_acronym.value_counts().to_dict(),
        qc_multiunits=int(quality_groups['QC multiunits'].sum()),
        total_single_units=int(f.decoder_label.eq('sua').sum()),
        total_multiunits=int(f.decoder_label.eq('mua').sum()),
        low_spike_units=int(f.low_spike_support.sum()),
        task_minutes=float(valid.sum()*dt/60), running_seconds=float(run_mask.sum()*dt),
        quiet_baseline_unique_seconds=float(baseline_mask.sum()*dt), n_bouts=len(bouts),
        baseline_bouts=int(bouts.baseline_available.sum()),
        baseline_bouts_by_epoch=e.groupby('epoch').run_delta_hz_n_bouts.first().to_dict(),
        lick_time_blocks=centered_rows, lick_occupancy=occupancy,
        candidate_counts=main_counts.to_dict(orient='records'),
        primary_neurons_any_candidate=int((q & candidates).sum()),
        primary_neurons_any_lick_candidate=int((q & (review.left_lick_r_candidate | review.right_lick_r_candidate)).sum()),
        candidate_ids={label: f.loc[q & review[field+'_candidate'], 'id'].tolist() for label,field,_,_ in features},
        temporal_sign_agreement={label: dict(n_candidates=int((q & review[field+'_candidate']).sum()),
            n_same_early_late=int((q & review[field+'_candidate'] & review[field+'_early_late_same_sign']).sum()))
            for label,field,_,_ in features},
        primary_peak_span_median_s=float(review.loc[q, 'onset_peak_time_resample_span_s'].median()),
        warning='Candidate screens are descriptive and cutoff-dependent; resampling bands are not calibrated significance tests or evidence of a stable trait across sessions. Temporal changes are reported, not disqualifying.',
        thresholds='Main screen: absolute run delta >=1 Hz, or absolute Pearson r >=0.2. No binary tuning label replaces continuous features.',
        checks='Same direction in deletion and resampling bands (running groups 3/5 bouts, licking 30/60/120 s). Running threshold signs must agree. Speed requires >=10 bouts with >=5 spikes and agreeing direction after 500-ms rebinning. Licking requires >=10 defined minute-block correlations and agreeing direction at 300/500/700-ms windows.',
        settings=OmegaConf.to_container(cfg, resolve=True))
    out.mkdir(parents=True, exist_ok=True)
    review.to_csv(out/'unit_review.csv', index=False)
    candidate_fields = ['id', 'ccf_acronym', 'default_qc', 'decoder_label', 'run_delta_hz',
        'run_modulation', 'run_within_bout_r', 'left_lick_r', 'right_lick_r',
        'run_delta_hz_candidate', 'run_within_bout_r_candidate', 'left_lick_r_candidate',
        'right_lick_r_candidate', 'run_delta_hz_early', 'run_delta_hz_late',
        'onset_peak_time_resample_span_s']
    review.loc[q & candidates, candidate_fields].to_csv(out/'candidate_neurons.csv', index=False)
    screens.to_csv(out/'candidate_counts.csv', index=False)
    pd.concat(resampling_rows).to_csv(out/'resampling_sensitivity.csv', index=False)
    pd.concat(smoothing_rows).to_csv(out/'lick_window_sensitivity.csv', index=False)
    epoch_rates.to_csv(out/'epoch_firing_rates.csv', index=False)
    pd.DataFrame(occupancy).to_csv(out/'lick_support_by_epoch.csv', index=False)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(review, main_counts, screens, out)
    print(json.dumps(summary, indent=2), flush=True)
    return out


def plot(review, main, screens, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.common.plotting import STYLE
    q = review.primary_neuron
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
        colors = ['#347e78', '#6688a4', '#9873b4', '#cf9258']
        ax = axes[0, 0]
        positions = np.arange(4)
        ax.barh(positions+.17, main.magnitude_only, height=.3, color='#d8e1e8', label='Size cutoff')
        ax.barh(positions-.17, main.after_checks, height=.3, color=colors, label='+ sensitivity checks')
        for k, row in enumerate(main.itertuples()):
            ax.text(row.after_checks+.5, k-.17, str(row.after_checks), va='center', fontsize=10)
        ax.set_yticks(positions, ['Run response', 'Speed within runs', 'Left licking', 'Right licking'])
        ax.invert_yaxis()
        ax.set(xlim=(0, 40), xlabel='Single neurons', title=f'{int(q.sum())} quality-passing single neurons')
        ax.legend(fontsize=8, loc='lower right')
        ax = axes[0, 1]
        for i, feature in enumerate(['Within-bout speed', 'Left licking', 'Right licking']):
            sub = screens.loc[screens.group.eq('QC single units') & screens.feature.eq(feature)]
            ax.plot(sub.cutoff, sub.after_checks, '-o', color=colors[i+1], lw=1.5, label=feature)
        ax.set(xlabel='Correlation magnitude cutoff', ylabel='Candidates after checks',
               title='Counts depend on the chosen cutoff', xticks=[.1, .2, .3], ylim=(0, None))
        ax.legend(fontsize=8)
        ax = axes[1, 0]
        use = q & review.run_delta_hz_candidate
        ax.scatter(review.loc[q, 'run_delta_hz_early'], review.loc[q, 'run_delta_hz_late'], color='#cbd5e1', s=22)
        ax.scatter(review.loc[use, 'run_delta_hz_early'], review.loc[use, 'run_delta_hz_late'], color=colors[0], s=28)
        ax.axhline(0, color='#94a3b8', lw=.7)
        ax.axvline(0, color='#94a3b8', lw=.7)
        ax.plot([-7, 50], [-7, 50], '--', color='#94a3b8', lw=.7)
        for uid in [382, 237, 132]:
            row = review.loc[review.id.eq(uid)].iloc[0]
            ax.annotate(str(uid), (row.run_delta_hz_early, row.run_delta_hz_late),
                        xytext=(5, 3), textcoords='offset points', fontsize=9)
        ax.set(xlabel='Early running − quiet (Hz)', ylabel='Late running − quiet (Hz)',
               title='9 early / 19 late baseline-supported bouts')
        ax = axes[1, 1]
        for side, color in [('left', colors[2]), ('right', colors[3])]:
            name = side+'_lick_r'
            ax.scatter(review.loc[q,name], review.loc[q,side+'_lick_r_within60s'],
                       s=25, alpha=.75, color=color, label=side.capitalize())
        ax.plot([-.25,.5], [-.25,.5], '--', color='#94a3b8', lw=.7)
        ax.axhline(0, color='#94a3b8', lw=.7)
        ax.axvline(0, color='#94a3b8', lw=.7)
        ax.set(xlabel='Whole-session licking correlation', ylabel='Correlation after removing minute means',
               title='Check for slow cofluctuation')
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.tick_params(length=0)
            ax.grid(alpha=.12, lw=.6)
            ax.set_axisbelow(True)
        fig.suptitle('Movement and licking · feature review', fontsize=15)
        fig.supxlabel('Descriptive candidates · run change ≥1 Hz in magnitude; correlations ≥0.2 · no significance labels',
                      fontsize=9, color='#64748b')
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out/f'feature_review.{suffix}', dpi=180, bbox_inches='tight')
        plt.close(fig)
