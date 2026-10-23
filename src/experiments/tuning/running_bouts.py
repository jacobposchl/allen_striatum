"""Bout-onset population PSTH and five individual running examples."""
import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.io import resolve
from src.data.recording import Recording
from src.data.task import load_task, recording_path, spike_counts, source_fingerprint
from src.experiments.tuning.running_correlation import bin_count
from src.experiments.tuning.running_examples import segments, smooth_segments
from src.common.plotting import STYLE


def detect_bouts(time, velocity, eligible, cfg, dt):
    # Neural responses never enter detection or selection.
    speed = smooth_segments(abs(velocity), np.isfinite(velocity), float(cfg.detection_sigma_s) / dt)
    running = np.isfinite(speed) & (speed >= float(cfg.threshold_cm_s))
    runs = segments(running)
    gap = bin_count(cfg.merge_gap_s, dt, 'merge_gap_s')
    for (_, b), (c, _) in zip(runs, runs[1:]):
        if c - b <= gap and np.isfinite(speed[b:c]).all():
            running[b:c] = True
    runs = segments(running)
    before = bin_count(cfg.before_s, dt, 'before_s')
    after = bin_count(cfg.after_s, dt, 'after_s')
    records = []
    for j, (a, b) in enumerate(runs):
        duration = (b - a) * dt
        previous_end = runs[j - 1, 1] if j else 0
        next_start = runs[j + 1, 0] if j + 1 < len(runs) else len(time)
        complete = a >= before and a + after <= len(time)
        coverage = eligible[a - before:a + after].mean() if complete else 0.
        duration_ok = float(cfg.min_duration_s) <= duration <= float(cfg.max_duration_s)
        isolated = previous_end <= a - before and next_start >= a + after
        qualified = complete and duration_ok and isolated and coverage >= float(cfg.min_coverage)
        records.append(dict(candidate=j, onset_index=a, offset_index=b, onset_s=float(time[a]),
                            duration_s=float(duration), peak_speed_cm_s=float(np.nanmax(speed[a:b])),
                            coverage=float(coverage), complete=complete, duration_ok=duration_ok,
                            isolated=isolated, eligible=qualified))
    audit = pd.DataFrame(records)
    if audit.empty:
        raise ValueError('No running bouts found')
    selected = audit.loc[audit.eligible].head(int(cfg.n_bouts)).copy()
    if len(selected) != int(cfg.n_bouts):
        raise ValueError('Too few eligible bouts; inspect detection settings explicitly')
    selected.insert(0, 'bout', np.arange(1, len(selected) + 1))
    audit['selected'] = audit.candidate.isin(selected.candidate)
    return selected, audit


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    units = pd.read_csv(out / 'units.csv')
    selected = pd.read_csv(out / 'selected_units.csv')
    bouts = pd.read_csv(out / 'bouts.csv')
    method = json.loads((out / 'method.json').read_text())
    with np.load(out / 'running_bouts.npz') as z:
        time = z['relative_time_s']
        heatmap = z['row_scaled_delta_rate']
        firing, velocity = z['example_firing_hz'], z['velocity_cm_s']
    dt = method['bin_size_s']
    edges = np.r_[time - dt / 2, time[-1] + dt / 2]
    colors = ['#7856a4', '#d58b32', '#247d88']
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e5e7eb')
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(18.2, 8.4), layout='constrained')
        layout = fig.add_gridspec(1, 2, width_ratios=[1, 2.65], wspace=.07)
        left = layout[0].subgridspec(2, 1, height_ratios=[23, 1], hspace=.06)
        ax = fig.add_subplot(left[0])
        mesh = ax.pcolormesh(edges, np.arange(len(units) + 1), np.ma.masked_invalid(heatmap),
                            vmin=-1, vmax=1, cmap=cmap, rasterized=True, shading='flat')
        regions = units.ccf_acronym.to_numpy()
        boundaries = np.flatnonzero(regions[1:] != regions[:-1]) + 1
        groups = np.r_[0, boundaries, len(units)]
        ax.set_yticks((groups[:-1] + groups[1:]) / 2, [regions[i] for i in groups[:-1]])
        for boundary in boundaries:
            ax.axhline(boundary, color='white', lw=.7)
        ax.axvline(0, color='#475569', lw=.8, ls='--')
        for (_, unit), color in zip(selected.iterrows(), colors):
            ax.scatter(1.015, unit.heatmap_row + .5, transform=ax.get_yaxis_transform(),
                       marker='<', color=color, s=48, clip_on=False)
        ax.set(xlim=(-method['before_s'], method['after_s']), ylim=(len(units), 0),
               title='Run-onset PSTH · 5 bouts', ylabel=f'Units · n = {len(units)}',
               xlabel='Time from run onset (s)')
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(mesh, cax=fig.add_subplot(left[1]), orientation='horizontal')
        colorbar.set_ticks([-1, 0, 1])
        colorbar.set_label('Row-scaled Δ firing')
        colorbar.outline.set_visible(False)

        right = layout[1].subgridspec(4, len(bouts), wspace=.06, hspace=.2)
        all_traces = np.concatenate([velocity[None, :, :], firing], axis=0)
        trace_colors = ['#2d806f', *colors]
        row_labels = ['Velocity\n(cm/s)', *[f'Unit {int(u.id)}\nFiring (Hz)' for _, u in selected.iterrows()]]
        axes = []
        for i, (traces, color, label) in enumerate(zip(all_traces, trace_colors, row_labels)):
            finite = traces[np.isfinite(traces)]
            lo = min(0, float(finite.min())) if i == 0 else 0.
            hi = max(float(finite.max()), 1.)
            limits = (lo - .05 * (hi - lo) if i == 0 else 0, hi * 1.08)
            row_axes = []
            for j, (_, bout) in enumerate(bouts.iterrows()):
                axis = fig.add_subplot(right[i, j])
                axis.plot(time, traces[j], color=color, lw=1.15)
                axis.axvline(0, color='#94a3b8', ls='--', lw=.7)
                axis.axvline(bout.duration_s, color='#cbd5e1', ls=':', lw=.8)
                axis.set(xlim=(-method['before_s'], method['after_s']), ylim=limits,
                         xticks=[0, 10, 20])
                axis.yaxis.set_major_locator(MaxNLocator(3, min_n_ticks=2))
                axis.tick_params(length=0, labelsize=9)
                if j == 0:
                    axis.set_ylabel(label, color=color, labelpad=7)
                else:
                    axis.tick_params(labelleft=False)
                if i == 0:
                    axis.set_title(f'Bout {int(bout.bout)} · {bout.onset_s / 60:.1f} min', fontsize=10.5)
                if i < 3:
                    axis.tick_params(labelbottom=False)
                elif j == len(bouts) // 2:
                    axis.set_xlabel('Time from run onset (s)')
                row_axes.append(axis)
            axes.append(row_axes)
        fig.suptitle('Five running bouts · units 382, 237 and 132', fontsize=15)
        fig.supxlabel('Dashed: onset · dotted: offset · traces: Gaussian σ = 1 s',
                      fontsize=10, color='#64748B')
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'running_bouts.{suffix}', dpi=210, bbox_inches='tight')
        plt.close(fig)


def run(cfg):
    from src.cli import require_compute
    require_compute('running_bouts', cfg)
    root, out = resolve(cfg.examples), resolve(cfg.output)
    if cfg.stage == 'plot':
        plot(out)
        return out
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output is not empty; choose another output or stage=plot')
    old_method = json.loads((root / 'method.json').read_text())
    example_cfg = OmegaConf.load(root / 'config.yaml')
    correlation_cfg = OmegaConf.load(resolve(example_cfg.correlations) / 'config.yaml')
    correlation_cfg.recording = old_method['recording_id']
    correlation_cfg.recording_root = cfg.get('recording_root') or correlation_cfg.get('recording_root')
    info, count_units, trials = load_task(correlation_cfg)
    if old_method.get('recording_source') != source_fingerprint(recording_path(correlation_cfg)):
        raise ValueError('Example output does not match the selected Brainsets recording')
    info['bin_size'] = .02
    units = pd.read_csv(root / 'heatmap_units.csv')
    selected_units = pd.read_csv(root / 'selected_units.csv')
    dt = float(old_method['bin_size_s'])
    sigma = float(old_method['smoothing_sigma_s']) / dt
    before = bin_count(cfg.before_s, dt, 'before_s')
    after = bin_count(cfg.after_s, dt, 'after_s')
    if not (cfg.before_s > 0 and cfg.after_s > cfg.max_duration_s
            and cfg.threshold_cm_s > 0 and cfg.min_duration_s > 0):
        raise ValueError('Invalid bout/window settings')
    with np.load(root / 'running_examples.npz') as z:
        time, raw_velocity = z['time_s'], z['velocity_unsmoothed_cm_s']
        if not np.array_equal(z['selected_unit_id'], selected_units.global_unit_id.to_numpy(str)):
            raise ValueError('Selected-unit identity mismatch')
    # Continuous observed task traces do not require a bin to fall wholly
    # inside one trial. Keep actual excluded trials/padding and missing data.
    valid = np.isfinite(raw_velocity)
    padding = info['excluded_trial_padding_s']
    for trial in trials.loc[~trials.usable].itertuples():
        valid &= ((time < trial.start_time - padding['before'])
                  | (time > trial.stop_time + padding['after']))
    velocity = smooth_segments(raw_velocity, valid, sigma)
    bouts, audit = detect_bouts(time, raw_velocity, valid, cfg, dt)
    indices = np.stack([int(b.onset_index) + np.arange(-before, after) for _, b in bouts.iterrows()])
    rel = np.arange(-before, after) * dt
    velocity_bouts = velocity[indices]

    raw_dt = float(info['bin_size'])
    factor = bin_count(dt, raw_dt, 'bin_size_s')
    edges = np.arange(int(np.ceil(info['duration_s'] / raw_dt)) + 1) * raw_dt
    with Recording(recording_path(correlation_cfg)) as recording:
        raw_counts = spike_counts(recording, count_units, edges)
    if not np.array_equal(count_units.iloc[units.count_row].global_unit_id,
                          units.global_unit_id):
        raise ValueError('Population unit identity mismatch')
    # Source grid begins at zero; verify before mapping saved bin centers.
    if not np.isclose(edges[0], 0):
        raise ValueError('Expected source time origin zero')
    local_origin = int(round(time[0] / dt - .5))
    if not np.isclose((local_origin + .5) * dt, time[0]):
        raise ValueError('Saved trace bins do not align with source counts')
    margin = int(np.ceil(3 * sigma))
    population = np.full((len(units), len(bouts), len(rel)), np.nan, np.float32)
    for j, (_, bout) in enumerate(bouts.iterrows()):
        a, b = int(bout.onset_index) - before - margin, int(bout.onset_index) + after + margin
        if a < 0 or b > len(time):
            raise ValueError('Insufficient smoothing context')
        raw_a, raw_b = (local_origin + a) * factor, (local_origin + b) * factor
        for row, unit in units.iterrows():
            rate = np.asarray(raw_counts[int(unit.count_row), raw_a:raw_b], float)
            rate = rate.reshape(b - a, factor).sum(axis=1) / dt
            smoothed = smooth_segments(rate, valid[a:b], sigma)
            population[row, j] = smoothed[margin:-margin]
    # Extract examples from these exact population traces, preserving identity.
    selected_rows = selected_units.heatmap_row.to_numpy(int)
    if not np.array_equal(units.iloc[selected_rows].global_unit_id, selected_units.global_unit_id):
        raise ValueError('Population and example unit identities disagree')
    example_bouts = population[selected_rows]
    support = np.isfinite(population).sum(axis=1)
    mean = np.divide(np.nansum(population, axis=1), support,
                     out=np.full((len(units), len(rel)), np.nan), where=support >= int(cfg.min_bouts_per_bin))
    base = (rel >= float(cfg.baseline_s[0])) & (rel < float(cfg.baseline_s[1]))
    if not base.any():
        raise ValueError('Baseline is outside displayed window')
    base_support = np.isfinite(mean[:, base]).sum(axis=1)
    baseline = np.divide(np.nansum(mean[:, base], axis=1), base_support,
                         out=np.full(len(units), np.nan), where=base_support >= .8 * base.sum())
    delta = mean - baseline[:, None]
    maximum = np.max(np.where(np.isfinite(delta), abs(delta), 0), axis=1)
    scaled = np.divide(delta, maximum[:, None], out=np.full_like(delta, np.nan),
                       where=maximum[:, None] > 1e-8)
    units['baseline_hz'] = baseline
    units['peak_absolute_delta_hz'] = maximum
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / 'config.yaml')
    units.to_csv(out / 'units.csv', index=False)
    selected_units.to_csv(out / 'selected_units.csv', index=False)
    bouts.to_csv(out / 'bouts.csv', index=False)
    audit.to_csv(out / 'bout_selection_audit.csv', index=False)
    np.savez_compressed(out / 'running_bouts.npz', relative_time_s=rel,
                        unit_id=units.global_unit_id.to_numpy(str),
                        selected_unit_id=selected_units.global_unit_id.to_numpy(str),
                        bout_onset_s=bouts.onset_s.to_numpy(), bout_duration_s=bouts.duration_s.to_numpy(),
                        per_bout_firing_hz=population, mean_firing_hz=mean,
                        baseline_hz=baseline, delta_firing_hz=delta, row_scaled_delta_rate=scaled,
                        contributing_bouts=support, velocity_cm_s=velocity_bouts,
                        example_firing_hz=example_bouts)
    method = dict(
        source=str(root), recording_id=info['recording_id'], recording_h5=info['recording_h5'], bin_size_s=dt,
        before_s=float(cfg.before_s), after_s=float(cfg.after_s),
        selection=f'First {cfg.n_bouts} qualifying isolated bouts in chronological order; behavior only',
        detection=dict(speed_threshold_cm_s=float(cfg.threshold_cm_s), sigma_s=float(cfg.detection_sigma_s),
                       merge_gap_s=float(cfg.merge_gap_s), minimum_duration_s=float(cfg.min_duration_s),
                       maximum_duration_s=float(cfg.max_duration_s), minimum_coverage=float(cfg.min_coverage)),
        isolation='No other detected run within the displayed -5 to +20 s window; complete recording context',
        unit_selection='Same three previously selected default-QC-passing SUA; no reselection by bout response',
        smoothing='Same sigma=1 s Gaussian, independently within eligible segments; no filling across missing intervals',
        eligibility='Continuous task recording; missing velocity and invalid trial intervals with original padding excluded; ordinary trial boundaries retained',
        psth='Equal-bout mean observed firing at each onset-relative bin; at least three of five bouts required',
        baseline_s=list(cfg.baseline_s),
        heatmap='Subtract each unit mean pre-onset rate, divide by its largest absolute mean change; raw Hz also saved',
        heatmap_interpretation='Response shape; row scaling does not measure effect size or significance; flat/unsupported rows are missing',
        traces='Signed velocity and observed firing in Hz; shared y scale for each row across bouts; lines only',
        scope='Five behavior-selected examples from the final 30 minutes; descriptive, not independent validation',
        n_bouts=len(bouts), n_units=len(units))
    (out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    plot(out)
    print(bouts[['bout', 'onset_s', 'duration_s', 'coverage']].to_string(index=False), flush=True)
    print(f'Saved {out / "running_bouts.png"}', flush=True)
    return out
