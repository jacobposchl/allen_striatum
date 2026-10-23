"""Unsmoothened, task-aligned traces for selected saved pairwise associations."""
import json
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.experiments.neighbors.neighbor_pairs import correlation
from src.data.io import resolve
from src.data.recording import Recording, resolve_recording
from src.embeddings.windows import RecordingExport
from src.common.plotting import STYLE


def task_data(cfg, starts, duration, origin):
    recording = Recording(resolve_recording(cfg.recording, root=cfg.get('recording_root')))
    exported = RecordingExport(cfg.export)
    if recording.identity != exported.source_identity:
        recording.close()
        raise ValueError("Recording changed since embedding export; re-export before plotting")
    table = recording.trials
    fields = ['id', 'start_time', 'stop_time', 'goCue_start_time', 'reward_outcome_time',
              'animal_response', 'reward_probabilityL', 'reward_probabilityR',
              'rewarded_historyL', 'rewarded_historyR']
    trials = table[fields].copy()
    for key in ['start_time', 'stop_time', 'goCue_start_time', 'reward_outcome_time']:
        trials[key] -= origin
    trials['rewarded'] = trials.rewarded_historyL.astype(bool) | trials.rewarded_historyR.astype(bool)
    probabilities = trials[['reward_probabilityL', 'reward_probabilityR']].to_numpy()
    changed = np.flatnonzero(np.any(np.diff(probabilities, axis=0) != 0, axis=1)) + 1
    transitions = trials.start_time.to_numpy()[changed]
    edges = np.r_[starts, starts[-1] + duration]
    behavior = pd.DataFrame({'start_session_s': starts, 'stop_session_s': edges[1:]})
    for side in ['left', 'right']:
        times = recording.events(f'{side}_lick') - origin
        behavior[f'{side}_licks_per_s'] = np.histogram(times, edges)[0] / duration
    times, speed = recording.signal('speed')
    times = times - origin
    speed = np.abs(np.asarray(speed).reshape(-1))
    if len(speed) != len(times):
        raise ValueError("Expected one running-speed value per timestamp")
    recording.close()
    good = np.isfinite(times) & np.isfinite(speed)
    totals = np.histogram(times[good], edges, weights=speed[good])[0]
    samples = np.histogram(times[good], edges)[0]
    behavior['speed_cm_s'] = np.divide(totals, samples, out=np.full_like(totals, np.nan), where=samples > 0)
    behavior['speed_samples'] = samples
    return trials, behavior, transitions


def draw_task(axes, trials, behavior, edges, transitions):
    from matplotlib.lines import Line2D
    left_color, right_color = '#3078B5', '#D27A22'
    for ax in axes:
        for time in transitions:
            if edges[0] <= time < edges[-1]:
                ax.axvline(time / 60, color='#94A3B8', lw=.55, alpha=.3, zorder=0)
    for side, color in [('L', left_color), ('R', right_color)]:
        axes[0].step(trials.start_time / 60, trials[f'reward_probability{side}'], where='post',
                     color=color, lw=1.5, label='Left' if side == 'L' else 'Right')
    axes[0].set(ylim=(0, .8), yticks=[.1, .4, .7])
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    local = trials.loc[(trials.stop_time > edges[0]) & (trials.start_time < edges[-1])]
    axes[1].plot(local.goCue_start_time / 60, np.full(len(local), 2), '|', ms=3, color='#64748B', alpha=.7)
    for response, y, color in [(0, 1, left_color), (1, 0, right_color), (2, -1, '#64748B')]:
        for rewarded in [False, True]:
            selected = local.loc[local.animal_response.eq(response) & local.rewarded.eq(rewarded)]
            axes[1].scatter(selected.reward_outcome_time / 60, np.full(len(selected), y), s=9,
                            facecolors=color if rewarded else 'white', edgecolors=color, lw=.65, zorder=3)
    axes[1].set(yticks=[-1, 0, 1, 2], yticklabels=['None', 'Right', 'Left', 'Cue'], ylim=(-1.5, 2.5))
    axes[1].legend(handles=[Line2D([], [], marker='o', ls='', color='#475569', markersize=4, label='Reward'),
                            Line2D([], [], marker='o', ls='', color='#475569', markerfacecolor='white', markersize=4, label='No reward')],
                   loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    for side, color in [('left', left_color), ('right', right_color)]:
        axes[2].stairs(behavior[f'{side}_licks_per_s'], edges / 60, color=color, lw=.9, baseline=None)
    axes[3].stairs(behavior.speed_cm_s, edges / 60, color='#475569', lw=.9, baseline=None)
    for label, color in [('D1', '#7161A5'), ('D2', '#218B87')]:
        axes[4].stairs(behavior[f'available_{label}'], edges / 60, color=color, lw=.9, baseline=None, label=label)
    axes[4].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    for ax, label in zip(axes[:5], ['Reward\nprobability', 'Trial events', 'Licks / s', 'Speed\n(cm/s)', 'Available\nlabeled units']):
        ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=20, fontsize=9)


def figure(pairs, modes, trials, behavior, edges, transitions, output, title):
    import matplotlib.pyplot as plt
    colors = ['#7451AA', '#179B83']
    trace_rows = [(pair, mode) for pair in pairs for mode in modes]
    n = 5 + len(trace_rows)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n, 1, figsize=(18, 4.8 + 1.35 * len(trace_rows)), sharex=True,
                                 gridspec_kw={'height_ratios': [.85, 1.05, .65, .65, .65] + [1.25] * len(trace_rows)})
        fig.subplots_adjust(left=.095, right=.87, bottom=.065, top=.91, hspace=.65)
        for ax in axes:
            ax.set_xlim(edges[0] / 60, edges[-1] / 60)
            ax.tick_params(length=0, labelsize=8)
            ax.grid(axis='y', color='#E5E9EF', lw=.5)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
        draw_task(axes, trials, behavior, edges, transitions)
        for ax, (pair, mode) in zip(axes[5:], trace_rows):
            values = pair[mode]
            for side, color in enumerate(colors):
                # Step edges preserve the full 10-second support; NaNs create gaps.
                ax.stairs(values[side], edges / 60, baseline=None, color=color, lw=1, alpha=.85,
                          label=pair['names'][side])
            if mode == 'raw':
                ax.set(ylim=(-.04, 1.04), yticks=[0, .5, 1])
                ylabel, r = 'Same-class\nfraction', pair['raw_r']
            else:
                ax.set(ylim=(-1, 1), yticks=[-1, 0, 1])
                ax.axhline(0, color='#94A3B8', lw=.6)
                ylabel, r = 'Class-centered\nscore', pair['class_centered_r']
            ax.set_ylabel(ylabel, rotation=0, ha='right', va='center', labelpad=20, fontsize=9)
            ax.set_title(f"{pair['names'][0]} ↔ {pair['names'][1]}", loc='left', fontsize=10, pad=5)
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.08), fontsize=8)
            ax.text(1.02, .22, f"r = {r:+.3f}\nn = {pair['n_joint_windows']}", transform=ax.transAxes, fontsize=9, va='top')
            for time in transitions:
                if edges[0] <= time < edges[-1]:
                    ax.axvline(time / 60, color='#94A3B8', lw=.55, alpha=.3, zorder=0)
        axes[-1].set_xlabel('Time in session (min)', labelpad=8)
        axes[-1].set_xticks(np.arange(0, edges[-1] / 60, 5))
        fig.suptitle(f'745502 · Dec 10 · Sep 12 model\n{title}', x=.095, y=.98, ha='left', fontsize=15, linespacing=1.5)
        fig.text(.095, .018, '10-second scores · No smoothing · Gaps = unavailable · Faint vertical lines = probability changes', color='#64748B', fontsize=9)
        for suffix in ['png', 'pdf']:
            fig.savefig(output.with_suffix('.' + suffix), dpi=220, bbox_inches='tight', pad_inches=.18)
        plt.close(fig)


def run(cfg):
    import matplotlib
    matplotlib.use('Agg')
    source, output, export = resolve(cfg.source), resolve(cfg.output), resolve(cfg.export)
    if output.exists():
        raise FileExistsError(output)
    data = np.load(source / 'neighbors.npz', allow_pickle=False)
    units = pd.read_csv(source / 'units.csv')
    saved_pairs = pd.read_csv(resolve(cfg.pair_results) / 'pairs.csv')
    assert np.array_equal(units.global_unit_id.to_numpy(str), data['global_unit_id'])
    starts, duration = data['start_session_s'], float(data['duration_s'])
    scores, labels = data['same_class_fraction'], units.label.to_numpy(str)
    exported = RecordingExport(cfg.export)
    saved = json.loads((source / 'metadata.json').read_text())
    if saved.get('export_manifest_sha256') != exported.manifest_sha256:
        raise ValueError('Neighbor scores do not match the selected HDF5 embedding export')
    if not np.allclose(np.diff(starts), duration, rtol=0, atol=1e-6):
        raise ValueError('This paired trace layout requires a contiguous selection of windows')
    selected_views = data['view_index'].astype(int)
    if not np.allclose(exported.starts_session[selected_views], starts, atol=1e-6, rtol=0):
        raise ValueError('Neighbor window times differ from the export schedule')
    offsets = exported.starts_source[selected_views] - starts
    origin = float(offsets[0])
    if not np.isfinite(offsets).all() or not np.allclose(offsets, origin, atol=1e-6, rtol=0):
        raise ValueError('Source-to-session clock alignment is unavailable')
    trials, behavior, transitions = task_data(cfg, starts, duration, origin)
    edges = np.r_[starts, starts[-1] + duration]
    for label in ['D1', 'D2']:
        behavior[f'available_{label}'] = data['vector_valid'][labels == label].sum(axis=0)
    pairs, frames = [], []
    for specification in cfg.pairs:
        indices, names = [], []
        for label, unit in specification:
            found = np.flatnonzero(units.label.eq(str(label)) & units.unit_id.astype(str).eq(str(unit)))
            if len(found) != 1:
                raise ValueError(f'Unit is not uniquely identifiable: {label} {unit}')
            indices.append(int(found[0]))
            names.append(f'{label} {unit}')
        i, j = indices
        matched = saved_pairs.loc[((saved_pairs.source_row_a == i) & (saved_pairs.source_row_b == j)) |
                                  ((saved_pairs.source_row_a == j) & (saved_pairs.source_row_b == i))]
        if len(matched) != 1:
            raise ValueError('Requested pair was not in the >75% full-session analysis')
        saved = matched.iloc[0]
        means = []
        for row in indices:
            reference = (labels == labels[row]) & ~np.isin(np.arange(len(units)), indices)
            means.append(np.nanmean(scores[reference], axis=0))
        raw = scores[indices].copy()
        adjusted = raw - np.asarray(means)
        joint = np.isfinite(raw).all(axis=0) & np.isfinite(adjusted).all(axis=0)
        raw_r = correlation(*raw[:, joint])
        adjusted_r = correlation(*adjusted[:, joint])
        np.testing.assert_allclose([raw_r, adjusted_r], [saved.raw_r, saved.class_centered_r], atol=1e-12)
        assert joint.sum() == saved.n_joint_windows
        pair = dict(names=names, source_rows=indices, raw=raw, adjusted=adjusted,
                    raw_r=raw_r, class_centered_r=adjusted_r, n_joint_windows=int(joint.sum()))
        pairs.append(pair)
        frames.append(pd.DataFrame(dict(pair=' / '.join(names), start_session_s=starts, stop_session_s=edges[1:],
                                        raw_a=raw[0], raw_b=raw[1], class_mean_a=means[0], class_mean_b=means[1],
                                        adjusted_a=adjusted[0], adjusted_b=adjusted[1], joint_available=joint)))
    output.mkdir(parents=True)
    pd.concat(frames, ignore_index=True).to_csv(output / 'score_traces.csv', index=False)
    trials.loc[(trials.stop_time > edges[0]) & (trials.start_time < edges[-1])].to_csv(output / 'trials.csv', index=False)
    behavior.to_csv(output / 'behavior_windows.csv', index=False)
    metadata = dict(recording_source_identity=exported.source_identity, export_manifest_sha256=exported.manifest_sha256,
                    source=str(source), pair_results=str(resolve(cfg.pair_results)), recording_h5=str(resolve_recording(cfg.recording, root=cfg.get('recording_root'))),
                    selection='Four pairs highlighted in the preceding full-session pairwise comparison; exploratory selection on these data',
                    coverage_s=[float(edges[0]), float(edges[-1])], n_windows=len(starts),
                    source_minus_session_s=origin, clock_validation='Validated every exported window',
                    adjustment='Subtract available same-class mean excluding both compared neurons; reference pool includes all originally labeled neurons',
                    missing='Each trace retains its own availability; correlations use joint valid windows only',
                    smoothing='None', original_knn_scores_unchanged=True,
                    validation='Every displayed correlation and overlap count matches saved pairwise results',
                    pairs=[{k:v for k,v in pair.items() if k not in ['raw', 'adjusted']} for pair in pairs])
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    OmegaConf.save(cfg, output / 'config.yaml')
    figure(pairs, ['raw'], trials, behavior, edges, transitions, output / 'raw_pair_traces', 'Selected pairs · Raw scores')
    figure(pairs, ['adjusted'], trials, behavior, edges, transitions, output / 'adjusted_pair_traces', 'Selected pairs · Class-wide fluctuations removed')
    for pair in pairs:
        name = '_'.join(pair['names']).replace(' ', '_')
        figure([pair], ['raw', 'adjusted'], trials, behavior, edges, transitions, output / name, 'Selected pair · Raw and class-centered scores')
    print(json.dumps(metadata, indent=2), flush=True)
    print(output, flush=True)
