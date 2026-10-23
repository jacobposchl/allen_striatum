"""Compact figures for descriptive movement/licking feature inspection."""
import json

import numpy as np
import pandas as pd

from src.common.plotting import STYLE


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = pd.read_csv(out / 'features.csv')
    bouts = pd.read_csv(out / 'running_bouts.csv')
    per_bout = pd.read_csv(out / 'unit_bout_features.csv')
    local = pd.read_csv(out / 'lick_block_correlations.csv')
    method = json.loads((out / 'method.json').read_text())
    with np.load(out / 'profiles.npz') as z:
        p = {k: z[k] for k in ['onset_time_s', 'onset_delta_hz', 'phase', 'phase_delta_hz',
                               'lick_rate_hz', 'lick_firing_hz', 'lick_exposure_s']}
    with np.load(out / 'behavior.npz') as z:
        t, speed, lick_rates, valid = z['time_s'], z['detection_speed_cm_s'], z['lick_rate_hz'], z['task_valid']
    region_colors = {'CP': '#7856a4', 'ACB': '#b47740'}
    examples = [int(np.flatnonzero(f.id.to_numpy() == uid)[0]) for uid in [382, 237, 132]
                if uid in f.id.to_numpy()]
    colors = ['#7856a4', '#d58b32', '#247d88']

    def tidy(ax):
        ax.tick_params(length=0)
        ax.grid(alpha=.13, lw=.6)
        ax.set_axisbelow(True)

    def scatter(ax, x, y, legend=False):
        for region, color in region_colors.items():
            for sorting, marker in [('sua', 'o'), ('mua', 'x')]:
                use = f.ccf_acronym.eq(region) & f.decoder_label.eq(sorting) & np.isfinite(x) & np.isfinite(y)
                ax.scatter(np.asarray(x)[use], np.asarray(y)[use], color=color,
                           marker=marker, s=22, alpha=.65, linewidths=.6,
                           label=f'{region} · {sorting.upper()}')
        if legend:
            ax.legend(fontsize=8, ncol=2)
        tidy(ax)

    def save(fig, name):
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'{name}.{suffix}', dpi=190, bbox_inches='tight')
        plt.close(fig)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), layout='constrained')
        ax = axes[0, 0]
        ax.plot(t / 60, speed, color='#2d806f', lw=.55)
        ax.plot(bouts.onset_s / 60, np.full(len(bouts), -.4), '|', color='#233248', ms=6)
        ax.set(xlim=(0, method['duration_s'] / 60), xlabel='Session time (min)', ylabel='Speed (cm/s)',
               title=f'{len(bouts)} running bouts · {method["n_baseline_bouts"]} with quiet baseline')
        tidy(ax)
        ax = axes[0, 1]
        x, y = f.run_baseline_hz, f.run_paired_running_hz
        scatter(ax, x, y, legend=True)
        maxrate = float(max(np.nanmax(x), np.nanmax(y))) * 1.1
        ax.plot([0, maxrate], [0, maxrate], color='#94a3b8', lw=.8, ls='--')
        ax.set(xscale='symlog', yscale='symlog', xlim=(0, maxrate), ylim=(0, maxrate),
               xlabel='Quiet rate (Hz)', ylabel='Running rate (Hz)', title='Paired bout / baseline rates')
        ticks = [0, 1, 10, 100]
        ax.set_xticks(ticks, ['0', '1', '10', '100'])
        ax.set_yticks(ticks, ['0', '1', '10', '100'])
        ax = axes[1, 0]
        for row, color in zip(examples, colors):
            ax.plot(p['onset_time_s'], p['onset_delta_hz'][row], color=color, lw=1.6,
                    label=f'Unit {int(f.iloc[row].id)}')
        ax.axvline(0, color='#94a3b8', lw=.8, ls='--')
        ax.axhline(0, color='#94a3b8', lw=.6)
        ax.set(xlabel='Time from run onset (s)', ylabel='Δ firing (Hz)', title='All supported bouts · same example units')
        ax.legend(fontsize=9)
        tidy(ax)
        ax = axes[1, 1]
        scatter(ax, f.run_delta_hz, f.run_within_bout_r)
        ax.axvline(0, color='#94a3b8', lw=.6)
        ax.axhline(0, color='#94a3b8', lw=.6)
        ax.set(xscale='symlog', xlabel='Running − quiet (Hz)', ylabel='Within-bout speed correlation',
               title='Running response versus speed tracking', ylim=(-1, 1))
        fig.suptitle('Movement profile · 745502 · December 10', fontsize=15)
        save(fig, 'running_features')

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), layout='constrained')
        ax = axes[0, 0]
        block = np.floor(t / 60).astype(int)
        centers = (np.arange(block.max() + 1) + .5)
        for side, color, name in [(0, '#3879b9', 'Left'), (1, '#d28a43', 'Right')]:
            means = [np.nanmean(lick_rates[side, (block == k) & valid]) for k in range(block.max() + 1)]
            ax.plot(centers, means, color=color, lw=1.4, label=name)
        ax.set(xlabel='Session time (min)', ylabel='Mean lick rate (Hz)', title='Licking across the session')
        ax.legend(fontsize=9)
        tidy(ax)
        ax = axes[0, 1]
        scatter(ax, f.left_lick_r, f.right_lick_r, legend=True)
        ax.axvline(0, color='#94a3b8', lw=.6)
        ax.axhline(0, color='#94a3b8', lw=.6)
        ax.set(xlabel='Left-lick correlation', ylabel='Right-lick correlation',
               title='Zero lag · 500-ms rate estimates', xlim=(-.5, .8), ylim=(-.5, .8))
        candidates = f.loc[f.decoder_label.eq('sua') & f.default_qc.eq(True) & ~f.low_spike_support].copy()
        candidates['strength'] = candidates[['left_lick_r', 'right_lick_r']].abs().max(axis=1)
        selected = candidates.nlargest(3, 'strength')
        selected[['global_unit_id', 'id', 'ccf_acronym', 'left_lick_r', 'right_lick_r']].to_csv(
            out / 'lick_profile_examples.csv', index=False)
        for side, ax in enumerate(axes[1]):
            exposure = p['lick_exposure_s'][side]
            for (row, unit), color in zip(selected.iterrows(), colors):
                response = p['lick_firing_hz'][side, row] - p['lick_firing_hz'][side, row, 0]
                response[exposure < 5.] = np.nan
                ax.plot(p['lick_rate_hz'], response, '-o', color=color, lw=1.4, ms=4,
                        label=f'Unit {int(unit.id)}')
            ax.axhline(0, color='#94a3b8', lw=.6)
            ax.set(xlabel=f'{["Left", "Right"][side]} lick rate (Hz)', ylabel='Firing − no licking (Hz)',
                   title=f'{["Left", "Right"][side]}-lick response profiles')
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12], ['0', '2', '4', '6', '8', '10', '>10'])
            ax.legend(fontsize=9)
            tidy(ax)
        fig.suptitle('Licking profile · 745502 · December 10', fontsize=15)
        save(fig, 'licking_features')

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 8), layout='constrained', sharey=True)
        run = per_bout.pivot(index='global_unit_id', columns='bout_id', values='modulation').reindex(f.global_unit_id)
        arrays = [run.to_numpy()]
        full_blocks = np.arange(int(np.ceil(method['duration_s'] / 60)))
        for side in ['left', 'right']:
            values = local.loc[local.side.eq(side)].pivot(index='global_unit_id', columns='block', values='r')
            arrays.append(values.reindex(index=f.global_unit_id, columns=full_blocks).to_numpy())
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#e5e7eb')
        for j, (ax, values, title) in enumerate(zip(axes, arrays,
                ['Run modulation', 'Left licking · Pearson r', 'Right licking · Pearson r'])):
            x = np.arange(values.shape[1] + 1)
            mesh = ax.pcolormesh(x, np.arange(len(f) + 1), np.ma.masked_invalid(values),
                                 cmap=cmap, vmin=-1, vmax=1, rasterized=True, shading='flat')
            boundary = int(f.ccf_acronym.eq('CP').sum())
            ax.axhline(boundary, color='white', lw=.8)
            ax.set(ylim=(len(f), 0), title=title,
                   xlabel='Bout (chronological)' if j == 0 else 'Session time (min)')
            ax.set_yticks([boundary / 2, (boundary + len(f)) / 2], ['CP', 'ACB'])
            ax.tick_params(length=0)
        axes[0].set_ylabel(f'Units · n = {len(f)}')
        bar = fig.colorbar(mesh, ax=list(axes), fraction=.025, pad=.015, shrink=.65)
        bar.set_ticks([-1, -.5, 0, .5, 1])
        bar.outline.set_visible(False)
        fig.suptitle('Feature variation within the session', fontsize=15)
        fig.supxlabel('Run modulation = (running − quiet) / (running + quiet) · gray: unsupported',
                      fontsize=10, color='#64748B')
        save(fig, 'temporal_features')

    html = ('<!doctype html><meta charset="utf-8"><title>Movement and licking</title>'
            '<style>body{max-width:1300px;margin:35px auto;font:16px system-ui;color:#233248}img{width:100%}</style>'
            '<h1>Movement and licking · 745502 · December 10</h1>'
            '<p>Descriptive profiles; correlations and resampling ranges are not significance tests.</p>'
            + ''.join(f'<img src="{name}.png">' for name in ['running_features', 'licking_features', 'temporal_features'])
            + '<p><a href="features.csv">Per-unit features</a> · <a href="method.json">Methods</a> · '
              '<a href="epoch_features.csv">Temporal summaries</a> · <a href="running_threshold_sensitivity.csv">Threshold sensitivity</a></p>')
    (out / 'index.html').write_text(html)
