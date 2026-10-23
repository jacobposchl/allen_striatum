"""Task-aligned view of saved neighbor scores; no neighbor recomputation.

Run on an existing CPU allocation:
python -m src.experiments.neighbors.plots.task_neighbors --config /path/to/settings.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.io import resolve
from src.data.recording import Recording, resolve_recording
from src.embeddings.windows import RecordingExport
from src.common.plotting import STYLE


def run(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    source, export = resolve(cfg.source), resolve(cfg.export)
    output = resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    data = np.load(source / 'neighbors.npz', allow_pickle=False)
    units = pd.read_csv(source / 'units.csv')
    assert np.array_equal(units.global_unit_id.to_numpy(str), data['global_unit_id'])
    starts = data['start_session_s']
    duration = float(data['duration_s'])
    exported = RecordingExport(cfg.export)
    saved = json.loads((source / 'metadata.json').read_text())
    if saved.get('export_manifest_sha256') != exported.manifest_sha256:
        raise ValueError('Neighbor scores do not match the selected HDF5 embedding export')
    offsets = exported.starts_source - exported.starts_session
    origin = float(offsets[0])
    if not np.isfinite(offsets).all() or not np.allclose(offsets, origin, atol=1e-6, rtol=0):
        raise ValueError('Source-to-session clock alignment is unavailable')
    recording = Recording(resolve_recording(cfg.recording, root=cfg.get('recording_root')))
    if recording.identity != exported.source_identity:
        recording.close()
        raise ValueError("Recording changed since export; regenerate embeddings")
    table = recording.trials
    keys = ['id', 'start_time', 'stop_time', 'goCue_start_time', 'reward_outcome_time',
            'animal_response', 'rewarded_historyL', 'rewarded_historyR',
            'reward_probabilityL', 'reward_probabilityR', 'bait_left', 'bait_right']
    trials = table[keys].copy()
    for key in ['start_time', 'stop_time', 'goCue_start_time', 'reward_outcome_time']:
        trials[key] -= origin
    trials['rewarded'] = trials.rewarded_historyL.astype(bool) | trials.rewarded_historyR.astype(bool)
    probabilities = trials[['reward_probabilityL', 'reward_probabilityR']].to_numpy()
    changes = np.flatnonzero(np.any(np.diff(probabilities, axis=0) != 0, axis=1)) + 1
    change_times = trials.start_time.to_numpy()[changes]
    # Pick the earliest isolated reversal, using behavior alone, before reading scores.
    candidates = []
    for index, time in zip(changes, change_times):
        left = round(time / duration) * duration - float(cfg.duration_s) / 2
        right = left + float(cfg.duration_s)
        reversal = np.prod(probabilities[[index - 1, index], 0] - probabilities[[index - 1, index], 1]) < 0
        isolated = np.sum((change_times >= left) & (change_times < right)) == 1
        local = starts[(starts >= left) & (starts + duration <= right)]
        contiguous = len(local) > 0 and np.allclose(np.diff(local), duration, rtol=0, atol=1e-6)
        complete = len(local) * duration == float(cfg.duration_s)
        if reversal and isolated and contiguous and complete and left >= starts[0] and right <= starts[-1] + duration:
            candidates.append((index, time, left, right))
    if not candidates:
        raise ValueError('No isolated reward-probability reversal in embedding coverage')
    index, transition, left, right = candidates[0]
    columns = (starts >= left) & (starts + duration <= right)
    selected_starts = starts[columns]
    assert len(selected_starts) * duration == float(cfg.duration_s)
    views = data['view_index'][columns].astype(int)
    if not np.allclose(exported.starts_session[views], selected_starts, atol=1e-6, rtol=0):
        raise ValueError('Neighbor window times disagree with the export schedule')
    counts = data['present'][:, columns].sum(axis=1)
    keep = counts / len(selected_starts) > float(cfg.min_presence_fraction)
    if not keep.any():
        raise ValueError('No neurons meet the presence threshold')
    selected_units = units.loc[keep].reset_index(drop=True)
    agreement = data['same_class_fraction'][keep][:, columns]
    edges = np.r_[selected_starts, right]
    centers = (edges[:-1] + edges[1:]) / 2
    behavior = pd.DataFrame({'start_session_s': selected_starts, 'stop_session_s': edges[1:]})
    for side in ['left', 'right']:
        timestamps = recording.events(f'{side}_lick') - origin
        behavior[f'{side}_licks_per_s'] = np.histogram(timestamps, edges)[0] / duration
    timestamps, speed = recording.signal('speed')
    timestamps = timestamps - origin
    speed = np.abs(np.asarray(speed).reshape(-1))
    if len(speed) != len(timestamps):
        raise ValueError("Expected one running-speed value per timestamp")
    recording.close()
    good = np.isfinite(timestamps) & np.isfinite(speed)
    sums = np.histogram(timestamps[good], edges, weights=speed[good])[0]
    samples = np.histogram(timestamps[good], edges)[0]
    behavior['speed_cm_s'] = np.divide(sums, samples, out=np.full(len(samples), np.nan), where=samples > 0)
    behavior['speed_samples'] = samples
    for label in ['D1', 'D2']:
        behavior[f'available_{label}'] = data['vector_valid'][units.label.eq(label)][:, columns].sum(axis=0)
    shown_trials = trials.loc[(trials.stop_time > left) & (trials.start_time < right)].copy()
    output.mkdir(parents=True)
    audit = units.copy()
    audit['available_windows_in_interval'] = counts
    audit['available_fraction_in_interval'] = counts / len(selected_starts)
    audit['retained'] = keep
    audit.to_csv(output / 'availability.csv', index=False)
    selected_units.to_csv(output / 'units.csv', index=False)
    shown_trials.to_csv(output / 'trials.csv', index=False)
    behavior.to_csv(output / 'behavior_windows.csv', index=False)
    np.savez_compressed(output / 'agreement.npz', same_class_fraction=agreement,
                        global_unit_id=selected_units.global_unit_id.to_numpy(str),
                        start_session_s=selected_starts, duration_s=duration)
    metadata = dict(selection='Earliest isolated reversal of the higher-probability side; behavior only',
                    source=str(source), recording_h5=str(resolve_recording(cfg.recording, root=cfg.get('recording_root'))), source_minus_session_s=origin,
                    clock_validation='Exported source/session offset identical in every displayed window',
                    interval_s=[left, right], transition_s=transition,
                    transition_trial_id=int(trials.iloc[index].id),
                    transition_anchor='Start of first trial with changed programmed probabilities',
                    probabilities_before=probabilities[index - 1].tolist(), probabilities_after=probabilities[index].tolist(),
                    presence_threshold=float(cfg.min_presence_fraction), threshold_operator='>',
                    presence_scope='Selected interval only', n_windows=len(selected_starts),
                    minimum_windows=int(np.floor(float(cfg.min_presence_fraction) * len(selected_starts)) + 1),
                    retained_units=int(keep.sum()), class_counts=selected_units.label.value_counts().to_dict(),
                    neighbor_pool='Unchanged: all original available labeled units per window, self excluded',
                    available_neighbor_track='Counts include each query when present; its own kNN excludes self',
                    speed='Mean absolute linear velocity per 10-second window, cm/s',
                    no_response_trials=int(shown_trials.animal_response.eq(2).sum()))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    OmegaConf.save(cfg, output / 'config.yaml')

    left_color, right_color = '#3078B5', '#D27A22'
    d1_color, d2_color = '#7161A5', '#218B87'
    heat_height = max(5.5, len(selected_units) * .085)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, heat_height + 5.8))
        grid = fig.add_gridspec(6, 1, height_ratios=[.9, 1.05, .65, .65, .65, heat_height],
                               left=.12, right=.79, bottom=.055, top=.94, hspace=.23)
        axes = []
        for row in range(6):
            axes.append(fig.add_subplot(grid[row], sharex=axes[0] if axes else None))
        for ax in axes:
            ax.set_xlim(left / 60, right / 60)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(axis='both', length=0)
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
            ax.axvline(transition / 60, color='#475569', linestyle='--', linewidth=.9, alpha=.7)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E5E9EF', linewidth=.5)
        for side, color in [('L', left_color), ('R', right_color)]:
            axes[0].step(trials.start_time / 60, trials[f'reward_probability{side}'], where='post', color=color, lw=2, label='Left' if side == 'L' else 'Right')
        axes[0].set_ylim(0, .8)
        axes[0].set_yticks([.1, .4, .7])
        axes[0].set_ylabel('Reward\nprobability', rotation=0, ha='right', va='center', labelpad=16)
        axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
        axes[0].set_title('745502 · Dec 10 · Sep 12 model\nReward-probability switch', loc='left', fontsize=15, pad=18)
        trial_ax = axes[1]
        trial_ax.plot(shown_trials.goCue_start_time / 60, np.full(len(shown_trials), 2), '|', color='#64748B', ms=7)
        for response, y, color in [(0, 1, left_color), (1, 0, right_color), (2, -1, '#64748B')]:
            for reward in [False, True]:
                chosen = shown_trials.loc[shown_trials.animal_response.eq(response) & shown_trials.rewarded.eq(reward)]
                trial_ax.scatter(chosen.reward_outcome_time / 60, np.full(len(chosen), y), s=24,
                                 facecolors=color if reward else 'white', edgecolors=color, linewidths=1.1, zorder=3)
        no_response = shown_trials.animal_response.eq(2).any()
        trial_ax.set_yticks([-1, 0, 1, 2] if no_response else [0, 1, 2],
                           ['None', 'Right', 'Left', 'Cue'] if no_response else ['Right', 'Left', 'Cue'])
        trial_ax.set_ylim(-1.5 if no_response else -.5, 2.5)
        trial_ax.set_ylabel('Trial events', rotation=0, ha='right', va='center', labelpad=16)
        trial_ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#475569', label='Reward'),
                                 Line2D([], [], marker='o', ls='', markerfacecolor='white', color='#475569', label='No reward')],
                        loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
        for side, color in [('left', left_color), ('right', right_color)]:
            axes[2].stairs(behavior[f'{side}_licks_per_s'], edges / 60, color=color, linewidth=1.5)
        axes[2].set_ylabel('Licks / s', rotation=0, ha='right', va='center', labelpad=16)
        axes[3].stairs(behavior.speed_cm_s, edges / 60, color='#475569', linewidth=1.5)
        axes[3].set_ylabel('Speed\n(cm/s)', rotation=0, ha='right', va='center', labelpad=16)
        for label, color in [('D1', d1_color), ('D2', d2_color)]:
            axes[4].stairs(behavior[f'available_{label}'], edges / 60, color=color, linewidth=1.5, label=label)
        axes[4].set_ylabel('Available\nlabeled units', rotation=0, ha='right', va='center', labelpad=16)
        axes[4].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
        cmap = LinearSegmentedColormap.from_list('agreement', ['#8B1026', '#F7F5EF', '#00552E'], N=11)
        cmap.set_bad('#BCC3CB')
        heat = axes[5].imshow(np.ma.masked_invalid(agreement), aspect='auto', interpolation='nearest',
                              cmap=cmap, norm=BoundaryNorm(np.linspace(-.05, 1.05, 12), 11),
                              extent=(left / 60, right / 60, len(selected_units) - .5, -.5))
        axes[5].set_yticks(np.arange(len(selected_units)), [f'{r.label} · {r.unit_id}' for r in selected_units.itertuples()], fontsize=6.5)
        boundary = np.flatnonzero(selected_units.label.to_numpy()[1:] != selected_units.label.to_numpy()[:-1]) + .5
        for y in boundary:
            axes[5].axhline(y, color='#334155', lw=1)
        axes[5].axvline(transition / 60, color='#334155', ls='--', lw=.9, alpha=.65)
        axes[5].set_xlabel('Time in session (min)', labelpad=10)
        axes[5].set_ylabel('Labeled unit', labelpad=12)
        axes[5].set_xticks(np.arange(left, right + 1, 30) / 60,
                          [f'{int(s//60)}:{int(s%60):02d}' for s in np.arange(left, right + 1, 30)])
        box = axes[5].get_position()
        color_ax = fig.add_axes([.83, box.y0 + box.height * .57, .015, box.height * .35])
        bar = fig.colorbar(heat, cax=color_ax, ticks=[0, .5, 1])
        bar.ax.set_yticklabels(['0 · opposite', '5 · mixed', '10 · same'])
        bar.set_label('Same-class neighbors (of 10)', labelpad=10)
        bar.outline.set_visible(False)
        axes[5].legend(handles=[Patch(facecolor='#BCC3CB', label='Unavailable')], loc='upper left',
                       bbox_to_anchor=(1.04, .48), frameon=False, fontsize=9)
        fig.text(.83, box.y0 + box.height * .34,
                 f'{len(selected_units)} neurons\nAvailable >75%\nof this interval\n\n10-second windows\nOriginal neighbor pool',
                 fontsize=9, va='top', linespacing=1.5)
        for suffix in ['png', 'pdf']:
            fig.savefig(output / f'task_aligned_heatmap.{suffix}', dpi=220, bbox_inches='tight', pad_inches=.18)
        plt.close(fig)
    print(json.dumps(metadata, indent=2), flush=True)
    print(output / 'task_aligned_heatmap.png', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(OmegaConf.load(Path(args.config)))
