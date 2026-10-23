"""Late-session running traces alongside the saved zero-lag heatmap."""
import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d

from src.data.io import resolve
from src.data.recording import Recording
from src.data.task import load_task, recording_path, task_grid, spike_counts, source_fingerprint
from src.experiments.tuning.running_correlation import bin_count
from src.common.plotting import STYLE


def segments(valid):
    return np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2)


def smooth_segments(values, valid, sigma_bins):
    """Normalized Gaussian smoothing within each observed segment only."""
    result = np.full(len(values), np.nan)
    for a, b in segments(valid & np.isfinite(values)):
        if b - a < 3:
            continue
        numerator = gaussian_filter1d(values[a:b], sigma_bins, mode='constant', truncate=3.)
        denominator = gaussian_filter1d(np.ones(b - a), sigma_bins, mode='constant', truncate=3.)
        result[a:b] = numerator / denominator
    return result


def acceleration(velocity, dt, margin):
    result = np.full(len(velocity), np.nan)
    for a, b in segments(np.isfinite(velocity)):
        if b - a <= 2 * margin + 2:
            continue
        derivative = np.gradient(velocity[a:b], dt)
        result[a + margin:b - margin] = derivative[margin:-margin]
    return result


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    method = json.loads((out / 'method.json').read_text())
    selected = pd.read_csv(out / 'selected_units.csv')
    units = pd.read_csv(out / 'heatmap_units.csv')
    with np.load(out / 'running_examples.npz') as z:
        corr, centers, time = z['correlation'], z['window_center_s'], z['time_s']
        velocity, accel, firing = z['velocity_cm_s'], z['acceleration_cm_s2'], z['firing_rate_hz']
    lo, hi = np.asarray(method['display_interval_s']) / 60
    times = time / 60
    step = method['heatmap_step_s']
    x_edges = np.r_[centers - step / 2, centers[-1] + step / 2] / 60
    colors = ['#7856a4', '#d58b32', '#247d88']
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e5e7eb')
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(17, 9.3), layout='constrained')
        layout = fig.add_gridspec(1, 2, width_ratios=[1, 1.75], wspace=.055)
        left = layout[0].subgridspec(2, 1, height_ratios=[24, 1], hspace=.06)
        ax = fig.add_subplot(left[0])
        mesh = ax.pcolormesh(x_edges, np.arange(len(units) + 1), np.ma.masked_invalid(corr),
                            cmap=cmap, vmin=-1, vmax=1, shading='flat', rasterized=True)
        regions = units.ccf_acronym.to_numpy()
        boundaries = np.flatnonzero(regions[1:] != regions[:-1]) + 1
        group_edges = np.r_[0, boundaries, len(units)]
        ax.set_yticks((group_edges[:-1] + group_edges[1:]) / 2,
                      [regions[i] for i in group_edges[:-1]])
        for boundary in boundaries:
            ax.axhline(boundary, color='white', lw=.8)
        for (_, unit), color in zip(selected.iterrows(), colors):
            ax.scatter([1.012], [unit.heatmap_row + .5], transform=ax.get_yaxis_transform(),
                       marker='<', s=48, color=color, clip_on=False, zorder=5)
        ax.set(xlim=(lo, hi), ylim=(len(units), 0), xlabel='Session time (min)',
               ylabel=f'Units · n = {len(units)}', title='Zero lag · firing vs speed')
        ax.set_facecolor('#e5e7eb')
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbax = fig.add_subplot(left[1])
        cb = fig.colorbar(mesh, cax=cbax, orientation='horizontal')
        cb.set_label('Pearson r', labelpad=3)
        cb.set_ticks([-1, -.5, 0, .5, 1])
        cb.outline.set_visible(False)

        right = layout[1].subgridspec(5, 1, hspace=.12)
        axes = []
        traces = [velocity, accel, *firing]
        trace_colors = ['#2d806f', '#627a92', *colors]
        labels = ['Velocity (cm/s)', 'Acceleration (cm/s²)', 'Firing (Hz)', 'Firing (Hz)', 'Firing (Hz)']
        for i, (trace, color, label) in enumerate(zip(traces, trace_colors, labels)):
            axis = fig.add_subplot(right[i], sharex=axes[0] if axes else None)
            axes.append(axis)
            axis.plot(times, trace, color=color, lw=.75, rasterized=True)
            axis.fill_between(times, 0, trace, color=color, alpha=.11, linewidth=0, rasterized=True)
            axis.axhline(0, color='#cbd5e1', lw=.6, zorder=0)
            axis.set(xlim=(lo, hi), ylabel=label)
            axis.yaxis.set_major_locator(MaxNLocator(3, min_n_ticks=2))
            axis.grid(axis='x', color='#eef1f5', lw=.6)
            axis.tick_params(axis='both', length=0)
            if i < 4:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel('Session time (min)')
            if i >= 2:
                unit = selected.iloc[i - 2]
                axis.set_ylim(bottom=0)
                axis.set_title(f'Unit {int(unit.id)} · {unit.ccf_acronym} · median r = {unit.median_r:.2f}',
                               loc='left', fontsize=11, color=color, pad=4)
        fig.suptitle('Running and neural activity · final 30 minutes', fontsize=15)
        fig.supxlabel(f'120 s correlations · traces: Gaussian σ = {method["smoothing_sigma_s"]:g} s · gray/blank: unavailable',
                      fontsize=10, color='#64748B')
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'running_examples.{suffix}', dpi=210, bbox_inches='tight')
        plt.close(fig)


def run(cfg):
    from src.cli import require_compute
    require_compute('running_examples', cfg)
    root, out = resolve(cfg.correlations), resolve(cfg.output)
    if cfg.stage == 'plot':
        plot(out)
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output is not empty; use a new output or stage=plot')
    saved_method = json.loads((root / 'method.json').read_text())
    correlation_cfg = OmegaConf.load(root / 'config.yaml')
    correlation_cfg.recording = saved_method['recording_id']
    correlation_cfg.recording_root = cfg.get('recording_root') or correlation_cfg.get('recording_root')
    info, count_units, trials = load_task(correlation_cfg)
    source = recording_path(correlation_cfg)
    if saved_method.get('recording_source') != source_fingerprint(source):
        raise ValueError('Correlation output does not match the selected Brainsets recording')
    recording = Recording(source)
    metadata = task_grid(recording, trials, info)
    info['bin_size'] = .02
    units = pd.read_csv(root / 'units.csv')
    duration = float(cfg.duration_s)
    end = float(info['duration_s'])
    start = end - duration
    if start < 0 or duration <= 0 or cfg.smoothing_sigma_s <= 0:
        raise ValueError('Invalid display duration or smoothing')
    width = float(correlation_cfg.window_s)
    with np.load(root / 'running_correlation.npz') as z:
        if not np.array_equal(z['global_unit_id'], units.global_unit_id.to_numpy(str)):
            raise ValueError('Heatmap identity/order mismatch')
        zero = np.flatnonzero(np.isclose(z['lag_s'], 0))
        if len(zero) != 1:
            raise ValueError('Exactly one zero-lag panel is required')
        centers = z['center_s']
        keep = (centers - width / 2 >= start) & (centers + width / 2 <= end)
        centers = centers[keep]
        corr = z['correlation'][zero[0]][:, keep]
    if not len(centers):
        raise ValueError('No complete correlation windows in requested interval')
    ranked = units.copy()
    ranked['median_r'] = [np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan
                          for row in corr]
    ranked['window_coverage'] = np.isfinite(corr).mean(axis=1)
    eligible = (ranked.decoder_label.eq('sua') & ranked.default_qc.eq(True)
                & ranked.window_coverage.ge(float(cfg.min_window_coverage))
                & ranked.median_r.ge(float(cfg.min_median_r)))
    ranked['selection_eligible'] = eligible
    selected = ranked.loc[eligible].sort_values(['median_r', 'id'], ascending=[False, True]).head(3).copy()
    if len(selected) != 3:
        raise ValueError('Fewer than three supported quality-passing SUA examples; inspect ranking or revise criteria explicitly')
    selected.insert(0, 'example_rank', np.arange(1, 4))

    raw_dt = float(info['bin_size'])
    dt = float(cfg.bin_size_s)
    factor = bin_count(dt, raw_dt, 'bin_size_s')
    edges = metadata['edges']
    source_valid = metadata['usable'] & np.isfinite(metadata['speed'])
    if factor < 1 or not np.allclose(np.diff(edges), raw_dt):
        raise ValueError('Inconsistent time grid')
    length = len(source_valid) // factor
    used = length * factor
    valid = source_valid[:used].reshape(length, factor).all(axis=1)
    time = edges[0] + (np.arange(length) + .5) * dt
    if not np.array_equal(count_units.iloc[selected.count_row].global_unit_id,
                          selected.global_unit_id):
        raise ValueError('Selected unit identities do not match spike-count rows')

    timestamps, measured = recording.signal('speed')
    timestamps = timestamps - info['clock']['origin_s']
    measured = np.asarray(measured, float).reshape(-1)
    velocity_info = recording.signal_info('speed')
    if len(measured) != len(timestamps) or not np.all(np.diff(timestamps) > 0):
        raise ValueError('Invalid velocity sample clock')
    index = np.floor((timestamps - edges[0]) / dt).astype(int)
    good = (index >= 0) & (index < length) & np.isfinite(measured)
    samples = np.bincount(index[good], minlength=length)
    sums = np.bincount(index[good], weights=measured[good], minlength=length)
    native_dt = float(np.median(np.diff(timestamps)))
    valid &= samples >= int(np.ceil(.8 * dt / native_dt - 1e-7))
    velocity = np.divide(sums, samples, out=np.full(length, np.nan), where=samples > 0)
    sigma = float(cfg.smoothing_sigma_s) / dt
    velocity_smooth = smooth_segments(velocity, valid, sigma)
    margin = max(1, int(np.ceil(sigma)))
    accel = acceleration(velocity_smooth, dt, margin)
    counts = spike_counts(recording, count_units, edges)
    recording.close()
    firing = []
    for _, unit in selected.iterrows():
        rate = np.asarray(counts[int(unit.count_row), :used], float).reshape(length, factor).sum(axis=1) / dt
        firing.append(smooth_segments(rate, valid, sigma))
    firing = np.asarray(firing)
    display = (time >= start) & (time <= end)
    finite_velocity = measured[(timestamps >= start) & (timestamps <= end) & np.isfinite(measured)]

    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / 'config.yaml')
    selected.to_csv(out / 'selected_units.csv', index=False)
    ranked.to_csv(out / 'selection_audit.csv', index=False)
    units.to_csv(out / 'heatmap_units.csv', index=False)
    np.savez_compressed(out / 'running_examples.npz', correlation=corr, window_center_s=centers,
                        heatmap_unit_id=units.global_unit_id.to_numpy(str),
                        selected_unit_id=selected.global_unit_id.to_numpy(str), time_s=time[display],
                        velocity_cm_s=velocity_smooth[display], acceleration_cm_s2=accel[display],
                        firing_rate_hz=firing[:, display], velocity_unsmoothed_cm_s=velocity[display],
                        valid=valid[display])
    method = dict(
        recording_id=info['recording_id'], animal=saved_method['animal'], session=saved_method['session'],
        display_interval_s=[start, end], heatmap_window_s=width,
        heatmap_step_s=float(correlation_cfg.step_s), heatmap_windows=len(centers),
        heatmap='Existing zero-lag firing-versus-absolute-speed correlation; only windows fully contained in displayed interval',
        row_order='Unchanged from saved population heatmap; colored markers identify selected units',
        selection=f'Top three positive median correlations among default-QC-passing SUA, coverage >= {cfg.min_window_coverage}, median r >= {cfg.min_median_r}',
        selection_scope='Selected and displayed in the same interval; illustrative strong associations, not independent validation',
        selected=selected[['id', 'global_unit_id', 'ccf_acronym', 'median_r', 'window_coverage']].to_dict('records'),
        velocity='Signed source linear velocity, conversion and offset applied; source data units verified as cm/s',
        velocity_description=velocity_info.get('description'),
        negative_velocity_fraction=float((finite_velocity < 0).mean()),
        acceleration='Time derivative of smoothed signed velocity; cm/s²; one smoothing sigma removed at each segment boundary',
        smoothing_sigma_s=float(cfg.smoothing_sigma_s),
        smoothing='Normalized Gaussian, truncated at 3 sigma, independently within contiguous eligible segments; no bridging of gaps',
        bin_size_s=dt, firing='Observed spike counts, rebinned and smoothed for display; not GLM predictions',
        masks='Audited source task mask, complete rebinned source bins, at least 80% of native velocity samples per bin',
        sources=dict(correlations=str(root), recording_h5=info['recording_h5']),
        recording_source=source_fingerprint(source))
    (out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    plot(out)
    print(selected[['id', 'ccf_acronym', 'median_r']].to_string(index=False), flush=True)
    print(f'Saved {out / "running_examples.png"}', flush=True)
    return out
