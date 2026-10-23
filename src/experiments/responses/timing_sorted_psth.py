"""Observed response shapes sorted by peak timing, with anatomy and tag strips."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.common.analysis import sha256, write_json
from src.data.io import resolve
from src.common.plotting import COLORS, STYLE


def run(cfg):
    source, out = resolve(cfg.source), resolve(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    source_cfg = OmegaConf.load(source/'config.yaml')
    all_units = pd.read_csv(source/'units.csv')
    mask = all_units.region.isin(list(cfg.regions)).to_numpy()
    units = all_units.loc[mask].copy().reset_index(drop=True)
    units['source_plot_row'] = units.plot_row
    with np.load(source/'population_psth.npz') as cached:
        assert np.array_equal(all_units.global_unit_id.to_numpy(str), cached['global_unit_id'])
        centers, edges = cached['bin_centers_s'], cached['bin_edges_s']
        psth = cached['firing_rate_hz'][mask]
        trial_ids = cached['trial_id']
    low, high = map(float, cfg.window_s)
    window = (centers >= low) & (centers < high)
    baseline_mask = (centers >= float(source_cfg.baseline_s[0])) & (centers < float(source_cfg.baseline_s[1]))
    assert window.any() and baseline_mask.any()
    indices = np.flatnonzero(window)
    display_edges = edges[indices[0]:indices[-1]+2]
    assert np.allclose(display_edges[[0, -1]], [low, high])
    rates = psth[:, window]
    baseline = psth[:, baseline_mask].mean(1)
    changes = rates-baseline[:, None]
    scale = np.abs(changes).max(1)
    no_peak = np.ptp(rates, axis=1) < 1e-12
    peak_bin = rates.argmax(1)
    normalized = np.divide(changes, scale[:, None], out=np.full_like(changes, np.nan),
                           where=scale[:, None] > 0)
    normalized[no_peak] = np.nan
    units['cache_subset_row'] = np.arange(len(units))
    units['peak_time_s'] = np.where(no_peak, np.nan, centers[window][peak_bin])
    units['no_identifiable_peak'] = no_peak
    units['baseline_rate_hz'] = baseline
    units['normalization_scale_hz'] = scale
    # Break equal-bin ties independently of anatomy or recorded cell label, so
    # annotation blocks are not manufactured by a depth- or region-based tie rule.
    units['tie_order'] = np.random.default_rng(int(cfg.tie_seed)).permutation(len(units))
    units = units.sort_values(['peak_time_s', 'tie_order'], na_position='last').reset_index(drop=True)
    units['plot_row'] = np.arange(len(units))
    order = units.cache_subset_row.to_numpy(int)
    values = normalized[order]
    assert units.global_unit_id.is_unique
    assert units.peak_time_s.dropna().is_monotonic_increasing
    assert np.nanmax(np.abs(values)) <= 1+1e-12
    units.to_csv(out/'units.csv', index=False)
    np.savez_compressed(out/'response_shapes.npz', global_unit_id=units.global_unit_id.to_numpy(str),
        bin_edges_s=display_edges, bin_centers_s=centers[window], trial_id=trial_ids,
        firing_rate_hz=rates[order], change_from_baseline_hz=changes[order],
        normalized_response=values, normalization_scale_hz=scale[order])
    plot(cfg, out, units, values, display_edges, len(trial_ids))
    summary = dict(animal=str(cfg.animal), session=str(cfg.session), n_units=len(units),
        n_trials=len(trial_ids), regions=units.region.value_counts().to_dict(),
        labels=units.label.value_counts().to_dict(), n_without_peak=int(no_peak.sum()),
        bin_size_s=float(np.diff(display_edges)[0]), window_s=[low, high],
        baseline_s=list(source_cfg.baseline_s))
    write_json(out/'summary.json', summary)
    method = dict(
        source=str(source),
        selection='All raw non-noise CP and ACB units from the population overview, including SUA and MUA. No response, label, QC, embedding, or GLM-fit filter.',
        response='Reuse observed trial-averaged spike rates in 50-ms bins. Subtract each unit\'s mean -1 to -0.2 s pre-cue firing; divide by its maximum absolute deviation from that baseline within 0–1 s. This shows response shape, not absolute magnitude or reliability, and can amplify weak/noisy variation. No smoothing or new fits.',
        ordering='Both regions are interleaved in one matrix sorted by time of maximum firing in 0–1 s. Equal peak bins use a fixed random tie order independent of region and tag. Non-flat tied maxima use the earliest bin. Flat units have no identifiable peak, are retained at the bottom, and are masked gray. The visible early-to-late progression is imposed by sorting; it is not evidence of a discovered sequence or discrete clusters.',
        annotation='Left strip: CP purple (#9467bd), ACB brown (#8c564b), matching the original population plot. Adjacent strip: original recorded D1 blue, D2 orange, unlabeled gray. Tags on MUA are retained as recorded and do not establish an isolated-neuron identity.',
        row_mapping='units.csv plot_row matches all saved array rows. source_plot_row identifies the original full-population cache row.',
        source_hashes={str(p):sha256(p) for p in [Path(__file__), source/'units.csv', source/'population_psth.npz', source/'config.yaml']})
    write_json(out/'method.json', method)
    OmegaConf.save(cfg, out/'config.yaml')
    report = ['<!doctype html><meta charset="utf-8"><title>Timing-sorted CP and ACB responses</title>',
        '<style>body{font:16px system-ui;line-height:1.5;max-width:1000px;margin:32px auto;color:#243247;padding:0 24px}img{max-width:100%}</style>',
        '<h1>Timing-sorted CP and ACB responses</h1>',
        f'<p>{len(units)} units, {len(trial_ids)} trials, 50-ms bins. Both regions share a single peak-time ordering; region and recorded cell-type annotations are shown in the two left strips.</p>',
        '<p>Colors show baseline change scaled separately for each unit. They emphasize shape, not response strength or reliability. The six units with no identifiable peak are gray at the bottom. Sorting imposes the timing progression; it does not establish clusters.</p>',
        '<img src="timing_sorted_psth_cp_acb.png">',
        '<p><a href="timing_sorted_psth_cp_acb.png">PNG</a> · <a href="timing_sorted_psth_cp_acb.pdf">PDF</a> · <a href="units.csv">Unit identities and row order</a> · <a href="response_shapes.npz">Response arrays</a> · <a href="method.json">Methods</a> · <a href="figures_and_tables.zip">Download</a></p>']
    (out/'index.html').write_text('\n'.join(report)+'\n')
    with zipfile.ZipFile(out/'figures_and_tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.suffix in ['.png', '.pdf', '.csv', '.npz', '.json', '.yaml', '.html']:
                archive.write(p, p.name)
    print(json.dumps(summary, indent=2), flush=True)
    print('Completed:', out, flush=True)


def plot(cfg, out, units, values, edges, n_trials):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    region_colors = {'CP': '#9467BD', 'ACB': '#8C564B'}
    tag_colors = {**COLORS, 'Unlabeled': '#D4D9E1'}
    n = len(units)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.5, 9.5), layout='constrained')
        grid = fig.add_gridspec(2, 3, width_ratios=[.2, .2, 5], height_ratios=[8, .23])
        for col, (field, palette, title) in enumerate([
                ('region', region_colors, 'Area'), ('label', tag_colors, 'Tag')]):
            ax = fig.add_subplot(grid[0, col])
            strip = np.array([to_rgba(palette[label]) for label in units[field]])[:, None]
            ax.imshow(strip, aspect='auto', interpolation='nearest', extent=(0, 1, n, 0))
            ax.set(xticks=[], yticks=[])
            ax.set_title(title, fontsize=9, pad=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
        ax = fig.add_subplot(grid[0, 2])
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#D4D9E1')
        im = ax.imshow(values, aspect='auto', interpolation='nearest', cmap=cmap, vmin=-1, vmax=1,
                       extent=(edges[0], edges[-1], n, 0), rasterized=True)
        ax.set(xlabel='Time from go cue (s)', xticks=np.linspace(edges[0], edges[-1], 6), yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        color_ax = fig.add_subplot(grid[1, 2])
        fig.colorbar(im, cax=color_ax, orientation='horizontal', extend='neither', ticks=[-1, 0, 1],
                     label='Normalized change from baseline')
        fig.suptitle('Responses ordered by peak time', fontsize=14)
        ax.set_title(f'{cfg.animal} · {str(cfg.session)[:10]} · {n} units · 50 millisecond bins', fontsize=9, pad=8)
        handles = [Patch(color=color, label=region) for region, color in region_colors.items()]
        handles += [Patch(color=tag_colors[label], label=label) for label in ['D1', 'D2', 'Unlabeled']]
        fig.legend(handles=handles, loc='outside lower center', ncol=5, fontsize=9, columnspacing=1.4)
        for extension in ['png', 'pdf']:
            fig.savefig(out/f'timing_sorted_psth_cp_acb.{extension}', dpi=240, bbox_inches='tight')
        plt.close(fig)
