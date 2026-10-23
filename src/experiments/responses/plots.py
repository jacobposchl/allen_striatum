"""Compact response-fit diagnostics and a map of fitted response shapes."""
import numpy as np
import pandas as pd

from src.common.plotting import STYLE


def plot(cfg, out, units, trials, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    with np.load(out / 'fits.npz') as z:
        times, post, basis = z['time_s'], z['post_bins'], z['basis']
        observed, fitted = z['observed_rate_hz'], z['full_fitted_rate_hz']
        test_observed, test_fitted = z['heldout_observed_rate_hz'], z['heldout_predicted_rate_hz']
        explained = z['pca_explained_variance']
    fold_metrics = pd.read_csv(out / 'validation_folds.csv')
    colors = {'CP': '#DC3C44', 'ACB': '#2563EB'}
    def save(fig, name):
        for extension in ['png', 'pdf']:
            fig.savefig(out / f'{name}.{extension}', dpi=220, bbox_inches='tight')
        plt.close(fig)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.4), layout='constrained')
        for b in range(basis.shape[1]):
            ax.plot(times[post], basis[post, b], lw=2, label=f'{b+1}')
        ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel='Time from go cue (s)', ylabel='Basis weight')
        ax.legend(title='Basis', ncol=5, loc='upper right')
        save(fig, 'temporal_basis')

        denominator = units.display_window_mean_rate_hz.to_numpy(float)
        denominator[denominator <= 0] = np.nan
        baseline = units.baseline_rate_hz.to_numpy(float)
        fig = plt.figure(figsize=(10, 8.7), layout='constrained')
        gs = fig.add_gridspec(1, 4, width_ratios=[.025, .025, 1, 1], wspace=.035)
        region_axis, tag_axis = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        region_axis.imshow(units.region.map({'CP':0, 'ACB':1}).to_numpy()[:,None], aspect='auto',
            cmap=ListedColormap(['#9467BD', '#8C564B']), vmin=0, vmax=1, interpolation='nearest')
        region_axis.set(xticks=[], yticks=[np.flatnonzero(units.region.eq(r)).mean() for r in ['CP','ACB']],
                        yticklabels=[f'{r} · {units.region.eq(r).sum()}' for r in ['CP','ACB']])
        region_axis.tick_params(axis='y', length=0, pad=6)
        tag_axis.imshow(units.label.map({'Unlabeled':0,'D1':1,'D2':2}).to_numpy()[:,None], aspect='auto',
            cmap=ListedColormap(['#D4D9E1','#2563EB','#D97706']), vmin=0, vmax=2, interpolation='nearest')
        tag_axis.set(xticks=[], yticks=[])
        for ax in [region_axis, tag_axis]:
            for spine in ax.spines.values():
                spine.set_visible(False)
        heat_axes = []
        for col, (array, title) in enumerate([(observed, 'Observed'), (fitted, 'Full-data fit')], start=2):
            ax = fig.add_subplot(gs[0, col])
            response = (array[:, post] - baseline[:, None]) / denominator[:, None]
            cmap = plt.get_cmap('RdBu_r').copy()
            cmap.set_bad('#E5E7EB')
            image = ax.imshow(response, extent=(0,1,len(units)-.5,-.5), aspect='auto', interpolation='nearest',
                              cmap=cmap, vmin=-3.8, vmax=3.8)
            ax.set(title=title, xlabel='Time from go cue (s)', yticks=[], xticks=[0,.5,1])
            ax.axhline(units.region.eq('CP').sum()-.5, color='#94A3B8', lw=.6)
            heat_axes.append(ax)
        fig.colorbar(image, ax=heat_axes, location='bottom', shrink=.72, aspect=55, pad=.035,
                     label='Δ firing / unit mean')
        fig.suptitle('Five-basis go-cue model · 20 millisecond bins', fontsize=13)
        fig.legend(handles=[Line2D([],[],marker='s',ls='',color=color,label=label)
            for label,color in [('D1','#2563EB'),('D2','#D97706'),('Unlabeled','#D4D9E1')]],
            loc='outside lower center',ncol=3)
        save(fig, 'fit_overview')

        valid = units.response_pc1.notna()
        shown = units.loc[valid]
        fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.6), sharex=True, sharey=True, layout='constrained')
        for region in ['CP','ACB']:
            group = shown.loc[shown.region.eq(region)]
            axes[0].scatter(group.response_pc1, group.response_pc2, s=25,
                color=colors[region], alpha=.2 + .7*group.heldout_psth_correlation_median.fillna(0).clip(0,1).to_numpy(),
                edgecolors='white', linewidths=.3, label=f'{region} · {len(group)}')
        axes[0].legend(loc='best')
        axes[0].set_title('Region')
        points = axes[1].scatter(shown.response_pc1, shown.response_pc2, s=25,
            c=shown.heldout_psth_correlation_median.fillna(0), cmap='coolwarm', vmin=-1, vmax=1,
            edgecolors='white', linewidths=.3)
        axes[1].set_title('Shape repeatability')
        bar = fig.colorbar(points, ax=axes[1], ticks=[-1,0,1], shrink=.8)
        bar.set_label('Median held-out correlation')
        for ax in axes:
            ax.set(xlabel=f'Response PC1 ({explained[0]*100:.1f}%)', ylabel=f'Response PC2 ({explained[1]*100:.1f}%)')
        fig.suptitle('Fitted response shapes · one point per unit', fontsize=13)
        save(fig, 'response_space')

        # Two supported examples near each region's median reliability, plus one model limitation.
        selected = []
        for region in ['CP','ACB']:
            group = units.loc[units.region.eq(region) & units.positive_likelihood_folds.eq(3) & units.full_post_spikes.ge(100)]
            if group.empty:
                group = units.loc[units.region.eq(region) & units.full_post_spikes.ge(100)]
            row = (group.heldout_psth_correlation_median - group.heldout_psth_correlation_median.median()).abs().idxmin()
            selected.append((int(row), f'{region}: supported response'))
        failure = units.loc[units.full_post_spikes.ge(100) & units.observed_period_correlation_median.ge(.6)
                            & ~units.index.isin([r for r,_ in selected])]
        if failure.empty:
            failure = units.loc[units.full_post_spikes.ge(100) & ~units.index.isin([r for r,_ in selected])]
        row = int(failure.full_psth_r2.idxmin())
        selected.append((row, 'Example of a fit limitation'))
        units.iloc[[r for r,_ in selected]].assign(example_role=[label for _,label in selected]).to_csv(out/'example_units.csv', index=False)
        fig, axes = plt.subplots(3,3,figsize=(11.5,8.1),sharex=True,layout='constrained')
        for row, (index, description) in enumerate(selected):
            unit = units.iloc[index]
            ymax = max(test_observed[index,:,post].max(), test_fitted[index,:,post].max()) * 1.1
            for fold, ax in enumerate(axes[row]):
                ax.plot(times[post], test_observed[index,fold,post], color='#94A3B8', lw=1.1)
                ax.plot(times[post], test_fitted[index,fold,post], color='#DD6B20', lw=2)
                score = fold_metrics.loc[fold_metrics.global_unit_id.eq(unit.global_unit_id) & fold_metrics.fold.eq(fold)].iloc[0]
                ax.set(xlim=(0,1), ylim=(0,ymax), xticks=[0,.5,1])
                ax.set_title(f'{["Early","Middle","Late"][fold]} · r={score.heldout_psth_correlation:.2f} · R²={score.heldout_psth_r2:.2f}', fontsize=10)
                if row == 2:
                    ax.set_xlabel('Time from go cue (s)')
                if fold == 0:
                    ax.set_ylabel(f'{unit.region} unit {int(unit.id)}\nFiring rate (Hz)')
        fig.legend(handles=[Line2D([],[],color='#94A3B8',lw=1.5,label='Observed held-out trials'),
            Line2D([],[],color='#DD6B20',lw=2,label='Predicted from other trials')], loc='outside upper center', ncol=2)
        save(fig, 'validation_examples')

    (out/'index.html').write_text('\n'.join([
        '<!doctype html><meta charset="utf-8"><title>Five-basis go-cue response model</title>',
        '<style>body{font:16px system-ui;line-height:1.5;max-width:1150px;margin:32px auto;color:#243247;padding:0 24px}img{max-width:100%}td,th{padding:6px}</style>',
        '<h1>December 10 · five-basis response model</h1>',
        '<p>All 318 CP/ACB units; 526 trials; 20-ms bins. Fit a separate baseline and five temporal basis coefficients per unit. Three entire chronological trial thirds are held out in turn. Full-data coefficients and validation fits are stored separately.</p>',
        '<img src="fit_overview.png">',
        '<p>Rows retain depth order within region. Cell tags: blue D1, orange D2, gray unlabeled. Both panels use the same observed baseline, normalization and colors. This is a full-data fit overview; held-out prediction is shown separately for each period below, so averaging across periods cannot conceal rate changes.</p>',
        '<img src="validation_examples.png">',
        '<p>Top two rows: supported examples near their region’s median held-out correlation. Bottom row: a reproducible observed response poorly captured by five functions. Each prediction is fitted using only the other two trial thirds.</p>',
        '<img src="response_space.png">',
        '<p>PCA uses baseline-subtracted, magnitude-normalized full-data fitted curves. Every dot is a unit; these are not NuCLR embeddings or clusters. More opaque region-colored points have higher median held-out shape correlation. Missing correlations are displayed as zero support. Correlation measures temporal shape and can remain high despite large firing-rate errors; inspect held-out R² separately.</p>',
        pd.DataFrame(summary['scopes']).drop(columns=['positive_fold_counts']).to_html(index=False),
        '<p><a href="full_coefficients.csv">Full-data coefficients</a> · <a href="units.csv">Unit metrics and coordinates</a> · <a href="validation_folds.csv">Validation by fold</a> · <a href="fits.npz">Fitted curves and coefficients</a> · <a href="method.json">Methods</a> · <a href="temporal_basis.png">Five basis functions</a></p>'])+'\n')
