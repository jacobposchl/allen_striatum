"""Observed population PSTHs, exact event-relative bins, and reusable trial counts."""
import html
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from omegaconf import OmegaConf

from src.common.analysis import sha256, write_json
from src.data.io import resolve
from src.data.recording import Recording
from src.data.task import recording_path, audit_trials, source_fingerprint
from src.common.plotting import STYLE, COLORS


def read_units(recording, cfg):
    frame = recording.units.copy()
    frame['included'] = frame.decoder_label.isin(list(cfg.sorting))
    frame['region'] = frame.ccf_acronym.fillna('Unassigned').astype(str)
    frame['probe'] = frame.device_name.fillna('Unknown probe').astype(str)
    included = frame[frame.included].copy()
    # Stored probe coordinates can run opposite to anatomical depth; configure the direction.
    order = included.groupby('region').depth.median().sort_values(
        ascending=bool(cfg.depth_ascending), kind='stable', na_position='last').index.tolist()
    included['region_order'] = included.region.map({r: i for i, r in enumerate(order)})
    included = included.sort_values(['region_order', 'probe', 'depth', 'id'],
        ascending=[True, True, bool(cfg.depth_ascending), True],
        kind='stable', na_position='last').reset_index(drop=True)
    included['plot_row'] = np.arange(len(included))
    assert included.global_unit_id.is_unique
    return frame, included


def run(cfg):
    out = resolve(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    recording = Recording(recording_path(cfg))
    trials, info = audit_trials(recording, allow_no_opto=bool(cfg.get('allow_no_opto', False)))
    if str(cfg.animal) != str(info['animal']) or str(cfg.session) != str(info['session']):
        raise ValueError('Configured panel identity does not match the selected recording')
    audit, units = read_units(recording, cfg)
    audit.to_csv(out / 'unit_inclusion.csv', index=False)
    trials['source_trial_row'] = np.arange(len(trials))
    trials['window_start_s'] = trials.goCue_start_time - float(cfg.before_s)
    trials['window_stop_s'] = trials.goCue_start_time + float(cfg.after_s)
    trials['psth_included'] = (trials.usable.eq(True)
        & np.isfinite(trials.goCue_start_time)
        & trials.window_start_s.ge(trials.start_time)
        & trials.window_stop_s.le(trials.stop_time)
        & trials.window_start_s.ge(0)
        & trials.window_stop_s.le(float(info['opto_start_s'])))
    trials['psth_exclusion'] = np.where(~trials.usable.eq(True), 'not_in_valid_behavior_trial_audit',
        np.where(~trials.psth_included, 'full_window_not_inside_trial_or_recording', ''))
    trials.to_csv(out / 'trial_inclusion.csv', index=False)
    selected = trials[trials.psth_included].sort_values('goCue_start_time').reset_index(drop=True)
    assert len(selected) > 0 and selected.id.is_unique
    selected['trial_column'] = np.arange(len(selected))
    selected['choice_lag_s'] = selected.choice_time - selected.goCue_start_time
    selected.to_csv(out / 'trials.csv', index=False)
    dt = float(cfg.bin_size_s)
    n_bins = round((float(cfg.before_s)+float(cfg.after_s))/dt)
    assert np.isclose(n_bins*dt, float(cfg.before_s)+float(cfg.after_s))
    edges = -float(cfg.before_s) + np.arange(n_bins+1)*dt
    centers = .5*(edges[1:]+edges[:-1])
    origin = float(info['clock']['origin_s'])
    # Work in the original spike clock; no approximation to the cached 20-ms count grid.
    absolute_edges = selected.goCue_start_time.to_numpy()[:, None] + origin + edges
    counts = open_memmap(out / 'trial_counts.npy', mode='w+', dtype=np.uint16,
                         shape=(len(units), len(selected), n_bins))
    psth = np.empty((len(units), n_bins), float)
    for row, source_row in enumerate(units.source_unit_row):
        spikes = recording.spikes(int(source_row))
        assert np.isfinite(spikes).all() and np.all(np.diff(spikes) >= 0)
        positions = np.searchsorted(spikes, absolute_edges, side='left')
        values = np.diff(positions, axis=1)
        assert values.min() >= 0 and values.max() <= np.iinfo(np.uint16).max
        counts[row] = values
        psth[row] = values.mean(axis=0)/dt
        if row % 100 == 0: print(f'Binned {row+1}/{len(units)} units', flush=True)
    counts.flush()
    baseline_mask = (centers >= float(cfg.baseline_s[0])) & (centers < float(cfg.baseline_s[1]))
    assert baseline_mask.any()
    baseline = psth[:, baseline_mask].mean(axis=1)
    mean = psth.mean(axis=1)
    relative = np.divide(psth-baseline[:, None], mean[:, None],
                         out=np.full_like(psth, np.nan), where=mean[:, None] > 0)
    units['baseline_rate_hz'] = baseline
    units['display_window_mean_rate_hz'] = mean
    units['peak_psth_hz'] = psth.max(axis=1)
    units['spikes_in_aligned_windows'] = np.asarray(counts.sum(axis=(1, 2)), dtype=np.int64)
    units.to_csv(out / 'units.csv', index=False)
    region = units.groupby('region', sort=False).agg(n_units=('id', 'size'),
        first_row=('plot_row', 'min'), last_row=('plot_row', 'max'),
        depth_min=('depth', 'min'), depth_max=('depth', 'max'))
    for label in ['D1', 'D2', 'Unlabeled']:
        region[label] = units.label.eq(label).groupby(units.region).sum().reindex(region.index, fill_value=0)
    region['sua'] = units.decoder_label.eq('sua').groupby(units.region).sum().reindex(region.index)
    region['mua'] = units.decoder_label.eq('mua').groupby(units.region).sum().reindex(region.index)
    region.to_csv(out / 'regions.csv')
    # Preserve side-specific and total observed licking, without removing choice licks.
    behavior = pd.DataFrame({'time_s': centers})
    for side in ['left', 'right']:
        licks = recording.events(f'{side}_lick')
        behavior[side+'_licks_hz'] = np.diff(np.searchsorted(licks, absolute_edges, side='left'), axis=1).mean(axis=0)/dt
    recording.close()
    behavior['licks_hz'] = behavior.left_licks_hz + behavior.right_licks_hz
    behavior.to_csv(out / 'aligned_behavior.csv', index=False)
    np.savez_compressed(out / 'population_psth.npz', global_unit_id=units.global_unit_id.to_numpy(str),
        trial_id=selected.id.to_numpy(), bin_edges_s=edges, bin_centers_s=centers,
        firing_rate_hz=psth, relative_change=relative, baseline_rate_hz=baseline, normalization_rate_hz=mean)
    limits = dict(rate_hz=max(float(np.quantile(psth, float(cfg.color_quantile))), 1e-8),
                  relative_change=max(float(np.nanquantile(np.abs(relative), float(cfg.color_quantile))), 1e-8))
    figures = plot(cfg, out, units, region, selected, edges, psth, relative, behavior, limits)
    choices = selected[selected.animal_response.isin([0, 1])]
    summary = dict(animal=str(info['animal']), session=str(info['session']), recording=str(info['recording']),
        raw_units=len(audit), included_units=len(units), sorting=units.decoder_label.value_counts().to_dict(),
        labels=units.label.value_counts().to_dict(), regions=region.reset_index().to_dict('records'),
        stored_trials=len(trials), eligible_behavior_trials=int(trials.usable.eq(True).sum()),
        included_trials=len(selected), rewarded_trials=int(choices.rewarded.eq(1).sum()),
        omitted_trials=int(choices.rewarded.eq(0).sum()), no_response_trials=int(selected.animal_response.eq(2).sum()),
        choice_lag_quantiles_s=choices.choice_lag_s.quantile([.1, .5, .9]).to_dict(),
        bin_size_s=dt, bin_count=n_bins, window_s=[float(edges[0]), float(edges[-1])],
        color_limits=limits, silent_units=int((mean == 0).sum()), probes=units.probe.unique().tolist(),
        depth_ascending=bool(cfg.depth_ascending), depth_axis_label=str(cfg.depth_axis_label))
    write_json(out / 'summary.json', summary)
    method = dict(recording_id=info['recording_id'], recording_h5=info['recording_h5'], origin_s=origin,
        units='All raw units classified sua or mua; includes all region annotations and units without embeddings or GLM fits. Noise excluded. No default_qc or firing-rate threshold.',
        trials='Reuse validated pre-optotagging trial metadata, independent of GLM fit success. Keep only trials whose entire fixed cue-relative window lies within that trial. Include no-response trials in overall PSTH; no reward/side balancing.',
        binning='Count original spike timestamps in exact half-open 50-ms bins relative to each go cue. Average counts across selected trials and divide by bin duration. No temporal smoothing, time warping, or GLM prediction.',
        ordering=f'Regions ordered by median stored depth; within each region, probe then stored depth then unit ID. Stored depth is {"ascending" if cfg.depth_ascending else "descending"} from top to bottom, corresponding to {cfg.depth_axis_label}. Depth is the source probe coordinate, not CCF dorsoventral position; for December 10 it increases toward cortex, as verified against CCF DV. Small region labels use leader lines to preserve legibility.',
        display='Raw firing rate in Hz is primary. Companion: subtract each unit\'s mean -1 to -0.2 s baseline and divide by that unit\'s mean across the complete displayed PSTH. This reveals relative response shape, not statistical significance; silent units are gray in the companion. Colors saturate at configured pooled quantiles; exact values retained.',
        region_views='Additional configured region views only filter displayed units; trials, normalization, color limits, region colors, and relative row order are unchanged. units_<view>.csv maps displayed plot_row to source_plot_row in the full population arrays. All color bars have rectangular ends.',
        labels='Original recorded predicted_cell_type annotations; untagged retained as unlabeled. Mixed single-unit and multi-unit labels are preserved in units.csv; a tag on a multiunit is not an isolated-neuron identity.',
        cache='trial_counts.npy axes: units.csv row order × trials.csv chronological order × population_psth.npz bin_centers_s. uint16 spike counts; divide by 0.05 for single-trial Hz. Intended for subsequent individual-neuron chronological heatmaps.',
        source_hashes={str(Path(__file__)): sha256(Path(__file__))}, recording_source=source_fingerprint(recording.path))
    write_json(out / 'method.json', method)
    OmegaConf.save(cfg, out / 'config.yaml')
    write_json(out / 'integrity_checks.json', dict(unique_unit_ids=True, unique_trial_ids=True,
        chronological_trials=bool(selected.goCue_start_time.is_monotonic_increasing),
        exact_bin_width=bool(np.allclose(np.diff(edges), dt)), raw_spikes_sorted=True,
        complete_trial_windows=True, counts_match_psth=bool(np.allclose(np.asarray(counts.mean(axis=1))/dt, psth)),
        depth_order_matches_config=all(
            g.depth.dropna().is_monotonic_increasing if cfg.depth_ascending else g.depth.dropna().is_monotonic_decreasing
            for _, g in units.groupby(['region', 'probe']))))
    text = ['<!doctype html><meta charset="utf-8"><title>Population trial-aligned firing</title>',
        '<style>body{font:16px system-ui;line-height:1.5;color:#243247;max-width:1200px;margin:36px auto;padding:0 24px}img{width:100%}table{border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #ddd}</style>',
        f'<h1>{html.escape(str(cfg.animal))} · {html.escape(str(cfg.session)[:10])}</h1>',
        f'<p>{len(units)} non-noise units, {len(selected)} trials, exact {1000*dt:g} ms bins. One row per unit, grouped by recorded region and ordered {html.escape(str(cfg.depth_axis_label))} from top to bottom. The stored probe coordinate is {"ascending" if cfg.depth_ascending else "descending"} within each region. All {len(units)} units are retained regardless of embedding availability, GLM fit, or default QC flag. Recorded tags: {summary["labels"]}. Sorting: {summary["sorting"]}.</p>',
        '<p>The primary heatmap shows observed trial-averaged firing in Hz. The companion shows baseline-subtracted firing divided by each unit’s mean rate in the displayed interval, which makes responses in slower units visible. This normalization does not establish reliability. The narrow strip marks recorded cell-type labels; gray means unlabeled. Region divider lines mark the actual boundaries; leader lines identify very small groups. There are no individual unit IDs on the y-axis.</p>',
        f'<p>Alignment is to the go cue, without time warping. Choice/outcome follows at a median {1000*summary["choice_lag_quantiles_s"][.5]:.0f} ms (10–90%: {1000*summary["choice_lag_quantiles_s"][.1]:.0f}–{1000*summary["choice_lag_quantiles_s"][.9]:.0f} ms). Thus cue-aligned activity can include action, outcome, and licking contributions. The trace above each heatmap shows mean licking on the same trials.</p>',
        f'<p>All outcomes and choices are pooled with their observed trial frequencies: {summary["rewarded_trials"]} rewarded, {summary["omitted_trials"]} unrewarded choices, and {summary["no_response_trials"]} no-response trials. {int(trials.usable.sum())-len(selected)} otherwise eligible trials lack the full display window. All data precede optotagging. Color limits use the {100*float(cfg.color_quantile):g}th percentile; exact, unclipped values are saved.</p>',
        region.to_html(border=0),
        '<p>Additional region views only filter displayed units and retain the same trials, normalization, color limits, and depth order. Their units tables map displayed plot_row to source_plot_row in the full population cache.</p>']
    for figure in figures:
        text.append(f'<p><a href="{figure}">{figure}</a></p><img src="{figure}">')
    text.append('<p><a href="units.csv">Unit identities and row order</a> · <a href="trials.csv">Included trials</a> · <a href="population_psth.npz">PSTH arrays</a> · <a href="trial_counts.npy">Chronological trial counts</a> · <a href="method.json">Methods</a> · <a href="figures_and_tables.zip">Figures and summary tables</a></p>')
    (out/'index.html').write_text('\n'.join(text)+'\n')
    with zipfile.ZipFile(out/'figures_and_tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.suffix in ['.png', '.csv', '.npy', '.npz', '.json', '.yaml', '.html']:
                archive.write(p, p.name)
    print(json.dumps(summary, indent=2), flush=True)
    print('Completed:', out, flush=True)


def plot(cfg, out, units, regions, trials, edges, psth, relative, behavior, limits):
    figures = _plot_view(cfg, out, units, regions, trials, edges, psth, relative, behavior, limits)
    for name, selected_regions in cfg.get('region_views', {}).items():
        mask = units.region.isin(list(selected_regions)).to_numpy()
        if not mask.any():
            raise ValueError(f'No units in region view {name}: {selected_regions}')
        subset = units.loc[mask].copy().reset_index(drop=True)
        subset['source_plot_row'] = subset.plot_row
        subset['plot_row'] = np.arange(len(subset))
        subset.to_csv(out / f'units_{name}.csv', index=False)
        subset_regions = regions.loc[subset.region.drop_duplicates()].copy()
        subset_regions['last_row'] = subset_regions.n_units.cumsum() - 1
        subset_regions['first_row'] = subset_regions.last_row + 1 - subset_regions.n_units
        subset_regions.to_csv(out / f'regions_{name}.csv')
        # Preserve the full-population color limits, normalization, trials, and row order.
        figures.extend(_plot_view(cfg, out, subset, subset_regions, trials, edges,
            psth[mask], relative[mask], behavior, limits, suffix=f'_{name}',
            **cfg.get('region_view_style', {}).get(name, {})))
    return figures


def _plot_view(cfg, out, units, regions, trials, edges, psth, relative, behavior, limits,
               suffix='', show_depth_label=True, shared_lick_trace=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    n = len(units)
    labels = list(dict.fromkeys(units.label))
    palette = {label: COLORS.get(label, '#8B5CF6') for label in labels}
    palette['Unlabeled'] = '#D4D9E1'
    strips = np.array([to_rgba(palette[x]) for x in units.label])[:, None, :]
    region_indices = units.groupby('region', sort=False).region_order.first().reindex(regions.index).to_numpy(int)
    region_colors = plt.get_cmap('tab10')(region_indices % 10)
    centers = .5*(regions.first_row.to_numpy()+regions.last_row.to_numpy()+1)
    label_y = centers.copy()
    gap = max(12, n*.035)
    for j in range(1, len(label_y)):
        label_y[j] = max(label_y[j], label_y[j-1]+gap)
    if len(label_y) and label_y[-1] > n-4:
        label_y[-1] = n-4
        for j in range(len(label_y)-2, -1, -1):
            label_y[j] = min(label_y[j], label_y[j+1]-gap)
    figures = []
    with plt.rc_context(STYLE):
        for mode in ['overview', 'rate_hz', 'relative_change']:
            panels = ['rate_hz', 'relative_change'] if mode == 'overview' else [mode]
            fig = plt.figure(figsize=(13.5 if len(panels)==2 else 8, 10), layout='constrained')
            grid = fig.add_gridspec(3, 2+len(panels), height_ratios=[1.3 if shared_lick_trace else .9, 8, .3],
                                   width_ratios=[1.3, .12]+[5]*len(panels))
            region_ax = fig.add_subplot(grid[1, 0])
            region_ax.set(xlim=(0, 1), ylim=(n, 0)); region_ax.axis('off')
            if show_depth_label:
                region_ax.text(-.30, .5, str(cfg.depth_axis_label), transform=region_ax.transAxes,
                               rotation=270, ha='center', va='center', fontsize=9, color='#64748B')
            for j, ((name, r), y) in enumerate(zip(regions.iterrows(), label_y)):
                a, b = r.first_row, r.last_row+1
                region_ax.plot([.96, .96], [a, b], color=region_colors[j], lw=4, solid_capstyle='butt')
                region_ax.plot([.80, .92], [y, (a+b)/2], color='#9AA7B8', lw=.7)
                region_ax.text(.76, y, f'{name} · {int(r.n_units)}', ha='right', va='center', fontsize=10)
            tag_ax = fig.add_subplot(grid[1, 1])
            tag_ax.imshow(strips, aspect='auto', interpolation='nearest', extent=(0, 1, n, 0))
            tag_ax.axis('off'); tag_ax.set_title('Tag', fontsize=9, pad=6)
            if shared_lick_trace:
                top_grid = grid[0, 2:].subgridspec(1, 3, width_ratios=[1, 2, 1])
                top = fig.add_subplot(top_grid[0, 1])
                top.plot(behavior.time_s, behavior.licks_hz, color='#475569', lw=1.3)
                top.axvline(0, color='#94A3B8', lw=.8, ls='--')
                top.set(xlim=(edges[0], edges[-1]), xticks=[edges[0], 0, edges[-1]])
                top.set_ylabel('Licks/s', fontsize=9)
                top.set_xlabel('Time from go cue (s)', fontsize=9)
                top.tick_params(labelsize=9)
            for col, panel in enumerate(panels, start=2):
                ax = fig.add_subplot(grid[1, col])
                color_ax = fig.add_subplot(grid[2, col])
                if not shared_lick_trace:
                    top = fig.add_subplot(grid[0, col], sharex=ax)
                    top.plot(behavior.time_s, behavior.licks_hz, color='#475569', lw=1.3)
                    top.axvline(0, color='#94A3B8', lw=.8, ls='--')
                    top.set_ylabel('Licks/s', fontsize=9)
                    top.tick_params(axis='x', bottom=False, labelbottom=False)
                    top.spines['bottom'].set_visible(False)
                if panel == 'rate_hz':
                    values, cmap, low, high = psth, plt.get_cmap('magma').copy(), 0, limits[panel]
                    title, color_label = 'Observed firing', 'Firing rate (Hz)'
                else:
                    values, cmap, low, high = relative, plt.get_cmap('RdBu_r').copy(), -limits[panel], limits[panel]
                    title, color_label = 'Within-neuron response', 'Δ firing / unit mean'
                cmap.set_bad('#D4D9E1')
                im = ax.imshow(values, aspect='auto', interpolation='nearest', extent=(edges[0], edges[-1], n, 0),
                               cmap=cmap, vmin=low, vmax=high, rasterized=True)
                ax.axvline(0, color='#C9D2DF', lw=.8, ls='--')
                for boundary in regions.last_row.iloc[:-1]+1:
                    ax.axhline(boundary, color='white', lw=.7)
                ax.set(xlabel='Time from go cue (s)', yticks=[], xlim=(edges[0], edges[-1]))
                (ax if shared_lick_trace else top).set_title(title, fontsize=12)
                for spine in ax.spines.values(): spine.set_visible(False)
                fig.colorbar(im, cax=color_ax, orientation='horizontal', extend='neither', label=color_label)
            fig.suptitle(f'{cfg.animal} · {str(cfg.session)[:10]} · {len(units)} units · {len(trials)} trials · {1000*float(cfg.bin_size_s):g} millisecond bins', fontsize=13)
            handles = [Patch(color=palette[label], label=label) for label in ['D1', 'D2', 'Unlabeled'] if label in palette]
            fig.legend(handles=handles, loc='outside lower center', ncol=len(handles), fontsize=9)
            name = f'population_psth_{mode}{suffix}.png'
            fig.savefig(out/name, dpi=220, bbox_inches='tight')
            plt.close(fig); figures.append(name)
    return figures
