"""Population-level quantitative review of trial-response cross-validation."""
import json

import numpy as np
import pandas as pd

from src.common.plotting import STYLE


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    units = pd.read_csv(out / 'unit_summary.csv')
    folds = pd.read_csv(out / 'fold_scores.csv')
    groups = pd.read_csv(out / 'subblock_scores.csv')
    audit = pd.read_csv(out / 'subblock_audit.csv')
    summary = json.loads((out / 'summary.json').read_text())
    metric = 'full_vs_common_bits_per_spike'
    order = units.sort_values(metric, ascending=False, na_position='last').global_unit_id
    fold_matrix = folds.loc[folds.scope.eq('choice')].pivot(index='global_unit_id', columns='fold', values=metric).reindex(order).to_numpy()
    group_matrix = groups.loc[groups.scope.eq('choice')].pivot(index='global_unit_id', columns='subblock', values=metric).reindex(order).to_numpy()
    finite = np.r_[fold_matrix.ravel(), group_matrix.ravel()]
    finite = finite[np.isfinite(finite)]
    scale = max(float(np.quantile(abs(finite), .97)), .01) if len(finite) else 1.
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e5e7eb')
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.5), layout='constrained')
        for label, color, marker in [('sua', '#326d9b', 'o'), ('mua', '#8b95a4', 'x')]:
            pick = units.decoder_label.eq(label) & units.cv_complete.eq(True)
            axes[0, 0].scatter(units.loc[pick, 'full_vs_constant_bits_per_spike'], units.loc[pick, metric],
                               s=17, alpha=.65, color=color, marker=marker, label=label.upper())
            axes[0, 1].scatter(units.loc[pick, metric], units.loc[pick, 'curve_relative_pair_rms'],
                               s=17, alpha=.65, color=color, marker=marker)
        axes[0, 0].set(title='Prediction', xlabel='Gain vs constant rate (bits/spike)',
                       ylabel='Gain vs shared response (bits/spike)')
        axes[0, 1].set(title='Curve stability', xlabel='Gain vs shared response (bits/spike)',
                       ylabel='Relative difference across fits')
        axes[0, 0].axhline(0, color='#a1a8b3', lw=.7, ls='--')
        for ax in axes[0]:
            ax.axvline(0, color='#a1a8b3', lw=.7, ls='--')
            ax.grid(alpha=.12, lw=.5)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
        axes[0, 0].legend(frameon=False, fontsize=9)
        axes[1, 0].imshow(fold_matrix, aspect='auto', origin='upper', cmap=cmap, vmin=-scale, vmax=scale,
                           interpolation='nearest', extent=[-.5, 2.5, len(units), 0])
        axes[1, 0].set(title='Across folds', xlabel='Tested sub-block', ylabel=f'Neurons · n = {len(units)}',
                       xticks=[0, 1, 2], xticklabels=['A', 'B', 'C'])
        edges = np.r_[audit.start_s.to_numpy(), audit.stop_s.iloc[-1]] / 60
        mesh = axes[1, 1].pcolormesh(edges, np.arange(len(units) + 1), group_matrix,
                                     cmap=cmap, vmin=-scale, vmax=scale, rasterized=True)
        axes[1, 1].set_ylim(len(units), 0)
        axes[1, 1].set(title='Across session', xlabel='Session time (min)', yticks=[])
        for ax in axes[1]:
            ax.tick_params(length=0)
        bar = fig.colorbar(mesh, ax=list(axes[1]), fraction=.025, pad=.025, extend='both')
        bar.set_label('Gain vs shared response (bits/spike)')
        bar.outline.set_visible(False)
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'validation_overview.{suffix}', dpi=190, bbox_inches='tight')
        plt.close(fig)
    (out / 'plot_settings.json').write_text(json.dumps(dict(
        heatmap_limit_bits_per_spike=scale, heatmap_limit_quantile=.97,
        row_order='Descending pooled choice-window gain versus shared-event reference; missing last',
        missing_color='gray', heatmap_values='held-out Poisson likelihood gain per test spike',
        curve_stability='Mean pairwise RMS Hz difference divided by RMS of mean predicted condition curves',
        positive_values='Descriptive improvement, not significance or a tuned-neuron threshold'), indent=2) + '\n')
    (out / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Trial-response validation</title>'
        '<style>body{max-width:1200px;margin:36px auto;padding:0 20px;font:16px system-ui;color:#263444}'
        'img{width:100%}a{color:#326d9b}p{line-height:1.5}</style>'
        '<h1>Trial-response validation</h1>'
        f'<p>745502 · December 10, 2024 · {summary["n_choice_trials"]} choices · '
        f'{summary["cv_complete"]}/{summary["n_units"]} neurons with complete validation fits.</p>'
        '<p>Nine consecutive trial blocks, three sub-blocks each. Three held-out folds; fixed regularization. '
        'Scores refer to choice-response windows. The shared-response reference ignores choice side and outcome.</p>'
        '<img src="validation_overview.png" alt="Prediction, curve stability, fold and session profiles">'
        '<p>Heatmaps share the same neuron order and color scale; gray means unavailable. '
        'Color limits clip the outer 3% of absolute heatmap values. '
        'Lower curve difference indicates greater agreement across fits. '
        'Positive prediction gain is not a significance threshold; poor local prediction does not identify its cause.</p>'
        '<p><a href="unit_summary.csv">Neuron summary</a> · <a href="fold_scores.csv">Fold scores</a> · '
        '<a href="subblock_scores.csv">Session scores</a> · <a href="curve_stability.csv">Condition-curve stability</a> · '
        '<a href="fold_audit.csv">Trial support</a> · <a href="method.json">Methods</a></p>')
