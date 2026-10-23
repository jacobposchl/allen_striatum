"""Time-resolved firing/speed correlations from audited behavioral bins."""
import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.io import resolve
from src.data.recording import Recording
from src.data.task import load_task, recording_path, task_grid, spike_counts, source_fingerprint
from src.common.plotting import STYLE


def bin_count(seconds, dt, name):
    value = int(round(float(seconds) / dt))
    if not np.isclose(value * dt, seconds, atol=1e-8):
        raise ValueError(f'{name} must be an integer multiple of {dt:g} s')
    return value


def interval_sums(values, starts, stops):
    prefix = np.r_[0., np.cumsum(values, dtype=np.float64)]
    return prefix[stops] - prefix[starts]


def common_lag_mask(valid, offsets):
    """Require every bin between the extreme lags, including the anchor."""
    lo, hi = min(0, min(offsets)), max(0, max(offsets))
    index = np.arange(len(valid))
    start, stop = index + lo, index + hi + 1
    inside = (start >= 0) & (stop <= len(valid))
    prefix = np.r_[0, np.cumsum(valid, dtype=np.int64)]
    count = prefix[np.clip(stop, 0, len(valid))] - prefix[np.clip(start, 0, len(valid))]
    return inside & (count == hi - lo + 1)


def rolling_correlation(x, y, valid, starts, stops, min_samples):
    """Pairwise Pearson r in complete windows; undefined values stay NaN."""
    mask = valid & np.isfinite(x) & np.isfinite(y)
    x = np.where(mask, x, 0.)
    y = np.where(mask, y, 0.)
    n = interval_sums(mask, starts, stops)
    sx = interval_sums(x, starts, stops)
    sy = interval_sums(y, starts, stops)
    sxx = interval_sums(x * x, starts, stops)
    syy = interval_sums(y * y, starts, stops)
    sxy = interval_sums(x * y, starts, stops)
    safe_n = np.maximum(n, 1)
    vx, vy = sxx - sx * sx / safe_n, syy - sy * sy / safe_n
    tolerance_x = 1e-10 * np.maximum(sxx, 1.)
    tolerance_y = 1e-10 * np.maximum(syy, 1.)
    usable = (n >= min_samples) & (vx > tolerance_x) & (vy > tolerance_y)
    result = np.full(len(starts), np.nan)
    result[usable] = ((sxy - sx * sy / safe_n)[usable]
                      / np.sqrt(vx[usable] * vy[usable]))
    return np.clip(result, -1., 1.)


def lagged(values, offset):
    """At time t return values[t + offset], without circular wrapping."""
    shifted = np.full(len(values), np.nan)
    if offset > 0:
        shifted[:-offset] = values[offset:]
    elif offset < 0:
        shifted[-offset:] = values[:offset]
    else:
        shifted[:] = values
    return shifted


def plot(cfg, out, units, correlations, centers, lags, duration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    edges = np.r_[centers - float(cfg.step_s) / 2, centers[-1] + float(cfg.step_s) / 2] / 60
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e5e7eb')
    regions = units.ccf_acronym.to_numpy()
    boundaries = np.flatnonzero(regions[1:] != regions[:-1]) + 1
    group_edges = np.r_[0, boundaries, len(units)]
    ticks = (group_edges[:-1] + group_edges[1:]) / 2
    labels = [regions[a] for a in group_edges[:-1]]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(lags), figsize=(5 * len(lags), 8.2),
                                 sharex=True, sharey=True, layout='constrained', squeeze=False)
        for ax, values, lag in zip(axes[0], correlations, lags):
            mesh = ax.pcolormesh(edges, np.arange(len(units) + 1),
                                 np.ma.masked_invalid(values), cmap=cmap,
                                 vmin=-1, vmax=1, shading='flat', rasterized=True)
            ax.set_facecolor('#e5e7eb')
            ax.set(xlim=(0, duration / 60), ylim=(len(units), 0),
                   xlabel='Session time (min)', title=f'Lag {lag:+g} s' if lag else 'Lag 0 s')
            ax.set_yticks(ticks, labels)
            ax.tick_params(axis='both', length=0)
            for boundary in boundaries:
                ax.axhline(boundary, color='white', lw=.7)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0, 0].set_ylabel(f'Units · n = {len(units)}')
        colorbar = fig.colorbar(mesh, ax=list(axes[0]), shrink=.65, pad=.015, fraction=.025)
        colorbar.set_label('Pearson r')
        colorbar.set_ticks([-1, -.5, 0, .5, 1])
        colorbar.outline.set_visible(False)
        fig.suptitle('Firing rate × running speed', fontsize=15)
        fig.supxlabel(f'{float(cfg.window_s):g} s windows · positive lag: firing follows running · gray: unavailable',
                      fontsize=10, color='#64748B')
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'running_correlation.{suffix}', dpi=220, bbox_inches='tight')
        plt.close(fig)


def run(cfg):
    # Protect direct calls as well as the shared command entry point.
    from src.cli import require_compute
    require_compute('running_correlation', cfg)
    out = resolve(cfg.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory is not empty; choose a new output for a changed analysis')
    info, units, trials = load_task(cfg)
    source = recording_path(cfg)
    with Recording(source) as recording:
        metadata = task_grid(recording, trials, info)
        raw_counts = spike_counts(recording, units, metadata['edges'])
    info['bin_size'] = .02
    units['count_row'] = np.arange(len(units))
    if not units.global_unit_id.is_unique:
        raise ValueError('Duplicate unit identities')
    prefix = f'{cfg.animal}/{cfg.session}/'
    if not units.global_unit_id.str.startswith(prefix).all():
        raise ValueError('Configured animal/session do not match prepared unit identities')
    units = units.loc[units.ccf_acronym.isin(list(cfg.regions))
                      & units.decoder_label.isin(list(cfg.sorting))].copy()
    units['region_order'] = units.ccf_acronym.map({r: i for i, r in enumerate(cfg.regions)})
    units = units.sort_values(['region_order', 'depth', 'id'],
                             ascending=[True, bool(cfg.depth_ascending), True],
                             kind='stable').reset_index(drop=True)
    if units.empty:
        raise ValueError('No eligible units')
    units.insert(0, 'heatmap_row', np.arange(len(units)))
    edges = metadata['edges']
    speed = metadata['speed']
    valid = metadata['usable'] & np.isfinite(speed)
    raw_dt = float(info['bin_size'])
    if not np.allclose(np.diff(edges), raw_dt) or len(speed) != len(edges) - 1:
        raise ValueError('Inconsistent source time grid')
    if raw_counts.shape != (int(info['n_units']), len(speed)):
        raise ValueError('Counts do not match source unit/time metadata')
    dt = float(cfg.bin_size_s)
    factor = bin_count(dt, raw_dt, 'bin_size_s')
    width = bin_count(cfg.window_s, dt, 'window_s')
    step = bin_count(cfg.step_s, dt, 'step_s')
    if factor < 1 or width < 2 or step < 1 or not 0 < cfg.min_valid_fraction <= 1:
        raise ValueError('Invalid bin/window/coverage settings')
    lags = np.asarray(cfg.lags_s, float)
    if not len(lags) or len(np.unique(lags)) != len(lags):
        raise ValueError('Supply distinct lags')
    offsets = [bin_count(lag, dt, 'lag') for lag in lags]
    length = len(speed) // factor
    used = length * factor
    if length < width or max(map(abs, offsets)) >= length:
        raise ValueError('Recording too short for window or lags')
    # Complete source bins only; never rebin across an excluded sub-bin.
    valid = valid[:used].reshape(length, factor).all(axis=1)
    speed = np.nan_to_num(speed[:used]).reshape(length, factor).mean(axis=1)
    valid = common_lag_mask(valid, offsets)
    starts = np.arange(0, length - width + 1, step)
    stops = starts + width
    centers = edges[0] + (starts + width / 2) * dt
    n_valid = interval_sums(valid, starts, stops).astype(int)
    minimum = max(3, int(np.ceil(width * float(cfg.min_valid_fraction))))
    correlations = np.full((len(lags), len(units), len(starts)), np.nan, np.float32)
    print(f'{len(units)} units; {len(starts)} windows; lags {lags.tolist()}', flush=True)
    for row, unit in units.iterrows():
        counts = np.asarray(raw_counts[int(unit.count_row), :used], dtype=np.float64)
        firing = counts.reshape(length, factor).sum(axis=1) / dt
        for j, offset in enumerate(offsets):
            correlations[j, row] = rolling_correlation(
                speed, lagged(firing, offset), valid, starts, stops, minimum)
    units['available_window_fraction'] = np.isfinite(correlations).mean(axis=(0, 2))
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / 'config.yaml')
    units.to_csv(out / 'units.csv', index=False)
    pd.DataFrame({'center_s': centers, 'start_s': edges[0] + starts * dt,
                  'stop_s': edges[0] + stops * dt, 'paired_bins': n_valid,
                  'coverage': n_valid / width}).to_csv(out / 'windows.csv', index=False)
    np.savez_compressed(out / 'running_correlation.npz', correlation=correlations,
                        global_unit_id=units.global_unit_id.to_numpy(str),
                        center_s=centers, lag_s=lags, paired_bins=n_valid)
    method = dict(
        recording_id=info['recording_id'], animal=cfg.animal, session=cfg.session, source=str(source), n_units=len(units),
        source_counts_shape=list(raw_counts.shape), shape=list(correlations.shape),
        sorting=units.decoder_label.value_counts().to_dict(),
        regions=units.ccf_acronym.value_counts().to_dict(),
        correlation='Pearson r between speed(t) and firing(t + lag) within each rolling window',
        lag_sign='Positive means firing follows running; no circular wrapping',
        row_order='Configured region order, source depth, original unit ID; identical in all panels',
        rate='Spike counts summed into 100-ms bins and divided by duration; no additional smoothing'
             if dt == .1 else f'Spike counts summed into {dt:g}-s bins; no additional smoothing',
        windows=dict(width_s=float(cfg.window_s), step_s=float(cfg.step_s),
                     minimum_coverage=float(cfg.min_valid_fraction)),
        masks='Directly audited eligible task bins, complete rebinned bins, and eligibility at every intermediate lag; common pairs across lag panels',
        missing='Gray: insufficient paired coverage, constant speed/firing, or no complete centered window',
        inherited_speed='HDF5 absolute speed in cm/s; source preparation interpolated up to four missing 20-ms speed bins',
        source_task_duration_s=float(info['duration_s']),
        available_fraction=float(np.isfinite(correlations).mean()),
        scope='All selected SUA/MUA, including units without completed GLM fits; no GLM coefficients used',
        interpretation='Descriptive local associations; overlapping windows and lags are dependent; no significance or independent encoding claim',
        recording_source=source_fingerprint(source))
    (out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    plot(cfg, out, units, correlations, centers, lags, length * dt)
    print(f'Saved {out / "running_correlation.png"}', flush=True)
    return out
