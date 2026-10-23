"""Exploratory within-recording comparison of existing communities and GLMs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution, rankdata

GROUPS = ['Full', 'Cue', 'Choice / outcome', 'Choice side', 'Outcome',
          'Licking', 'Movement', 'Reward probability', 'History']


def bh(values):
    values = np.asarray(values, float)
    result = np.full(len(values), np.nan)
    good = np.flatnonzero(np.isfinite(values))
    order = good[np.argsort(values[good])]
    result[order] = np.minimum(1, np.minimum.accumulate(
        (values[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return result


def association(frame, feature, adjusted):
    """Nested rank-outcome regression; descriptive F tests, not replication."""
    columns = ['cluster', feature]
    if adjusted:
        columns += ['analysis_firing_rate_hz', 'available_fraction', 'depth',
                    'decoder_label', 'label', 'default_qc']
    d = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    y = rankdata(d[feature].to_numpy())
    base = np.ones((len(d), 1))
    if adjusted:
        rate = np.log1p(d.analysis_firing_rate_hz.to_numpy())
        numeric = np.column_stack([rate, rate**2, d.available_fraction, d.depth])
        numeric = (numeric - numeric.mean(0)) / np.maximum(numeric.std(0), 1e-12)
        categorical = pd.get_dummies(d[['decoder_label', 'label', 'default_qc']].astype(str),
                                     drop_first=True, dtype=float).to_numpy()
        base = np.column_stack([base, numeric, categorical])
    full = np.column_stack([base, pd.get_dummies(d.cluster, drop_first=True, dtype=float)])
    rank0, rank1 = np.linalg.matrix_rank(base), np.linalg.matrix_rank(full)
    df1, df2 = rank1 - rank0, len(d) - rank1
    rss0 = np.square(y - base @ np.linalg.lstsq(base, y, rcond=None)[0]).sum()
    rss1 = np.square(y - full @ np.linalg.lstsq(full, y, rcond=None)[0]).sum()
    if df1 <= 0 or df2 <= 0 or rss0 <= 1e-12:
        return dict(n=len(d), partial_r2=np.nan, p=np.nan, df1=df1, df2=df2)
    effect = max(0., 1 - rss1 / rss0)
    statistic = max(0., (rss0 - rss1) / df1) / max(rss1 / df2, 1e-12)
    return dict(n=len(d), partial_r2=effect, p=float(f_distribution.sf(statistic, df1, df2)),
                df1=df1, df2=df2)


def run(cfg):
    from omegaconf import OmegaConf
    from src.experiments.clustering.plots import PALETTE, style
    import matplotlib.pyplot as plt

    source, glm, out = Path(cfg.clusters), Path(cfg.glm), Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / 'config.yaml')
    c = pd.read_csv(source / 'units.csv')
    c = c.loc[c.cluster.ge(0)].copy()
    n_clustered = len(c)
    status = pd.read_csv(glm / 'unit_fit_status.csv')
    d = c.merge(status[['global_unit_id', 'unit_id', 'fitted_folds', 'fit_status']],
                on='global_unit_id', how='left', suffixes=('', '_glm'), validate='one_to_one')
    assert d.unit_id.eq(d.unit_id_glm).all(), 'Brainsets source unit identity mismatch'
    d['cluster_display'] = d.cluster.map(lambda c: f'C{int(c)+1}')
    d.to_csv(out / 'cohort_audit.csv', index=False)
    d = d.loc[d.fitted_folds.eq(3)].copy()
    reliability = pd.read_csv(glm / 'reliability.csv')
    r = reliability.merge(d[['global_unit_id', 'cluster', 'cluster_display']],
                          on='global_unit_id', validate='many_to_one')
    assert r.n_folds.eq(3).all() and r.all_converged.all()
    gains = r.pivot(index='global_unit_id', columns='group', values='heldout_gain')
    d = d.merge(gains, on='global_unit_id', validate='one_to_one')
    curves = {}
    for uid in d.unit_id:
        with np.load(glm / 'curves' / f'unit_{uid:04d}.npz') as z:
            curves[int(uid)] = {k: z[k].copy() for k in ['lags', 'kernel_names', 'kernels',
                'speed', 'movement', 'probability', 'probability_effect', 'history']}
    d['movement_lograte_1cms'] = [np.interp(1, curves[u]['speed'],
        curves[u]['movement'].mean(0)) for u in d.unit_id]
    d.to_csv(out / 'unit_features.csv', index=False)
    r.to_csv(out / 'reliability_by_cluster.csv', index=False)
    scopes = {'All': np.ones(len(d), bool), 'SUA': d.decoder_label.eq('sua'),
              'QC': d.default_qc.eq(True),
              'QC SUA': d.default_qc.eq(True) & d.decoder_label.eq('sua'),
              'Coverage >75%': d.available_fraction.gt(.75)}
    scopes = {name: d.loc[mask].copy() for name, mask in scopes.items()}
    results, summaries = [], []
    features = GROUPS + ['movement_lograte_1cms']
    for scope, subset in scopes.items():
        counts = subset.cluster.value_counts()
        usable = subset[subset.cluster.isin(counts[counts.ge(cfg.minimum_cluster_units)].index)]
        for feature in features:
            for adjusted in [False, True]:
                results.append(dict(scope=scope, feature=feature, adjusted=adjusted,
                    clusters=','.join(f'C{int(c)+1}' for c in sorted(usable.cluster.unique())),
                    **association(usable, feature, adjusted)))
        for cluster, part in subset.groupby('cluster'):
            for feature in features:
                x = part[feature].to_numpy()
                consistent = r.loc[r.global_unit_id.isin(part.global_unit_id) & r.group.eq(feature),
                                   'consistent_gain']
                summaries.append(dict(scope=scope, cluster=cluster, cluster_display=f'C{cluster+1}',
                    feature=feature, n=len(part), mean=float(np.mean(x)), median=float(np.median(x)),
                    q25=float(np.quantile(x, .25)), q75=float(np.quantile(x, .75)),
                    consistent_n=int(consistent.sum()) if len(consistent) else np.nan))
    tests = pd.DataFrame(results)
    tests['q'] = tests.groupby(['scope', 'adjusted']).p.transform(lambda x: bh(x.to_numpy()))
    tests.to_csv(out / 'associations.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'cluster_glm_summary.csv', index=False)
    composition = d.groupby('cluster').agg(n=('unit_id', 'size'),
        sua=('decoder_label', lambda x: x.eq('sua').sum()), qc=('default_qc', 'sum'),
        d1=('label', lambda x: x.eq('D1').sum()), d2=('label', lambda x: x.eq('D2').sum()),
        median_rate=('analysis_firing_rate_hz', 'median'),
        median_coverage=('available_fraction', 'median'), median_depth=('depth', 'median'))
    composition.to_csv(out / 'cluster_composition.csv')
    style()
    clusters = sorted(d.cluster.unique())
    color = {c: PALETTE[c % len(PALETTE)] for c in clusters}
    def save(fig, name):
        fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)
    def dots(ax, frame, feature, scale=1):
        rng = np.random.default_rng(0)
        for c in clusters:
            y = frame.loc[frame.cluster.eq(c), feature].to_numpy() * scale
            if not len(y):
                continue
            ax.scatter(c + rng.uniform(-.19, .19, len(y)), y, color=color[c], s=12,
                       alpha=.5, linewidths=0)
            ax.plot([c-.24, c+.24], [np.median(y)]*2, color='#273345', lw=2)
        ax.axhline(0, color='#AEB9C5', lw=.7, zorder=0)
        ax.set_xticks(clusters, [f'C{c+1}' for c in clusters])
        ax.grid(axis='y', alpha=.15)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, feature in zip(axes.flat, GROUPS):
        dots(ax, d, feature, 100)
        ax.set_title(feature)
        ax.set_ylabel('Held-out deviance gain (%)')
    fig.suptitle('Dec 10 · GLM associations by embedding cluster', fontsize=15)
    fig.text(.5, .01, '192 complete fits · dots: units · bars: medians · Full compares against slow drift',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .96))
    save(fig, 'glm_gains')

    example = next(iter(curves.values()))
    curve_specs = [(name, example['lags'], lambda z, i=i: z['kernels'].mean(0)[i], 'Time (s)')
                   for i, name in enumerate(['Cue', 'Choice / outcome common', 'Left − right choice',
                       'Reward − omission', 'Side × outcome', 'Left lick', 'Right lick'])]
    curve_specs += [('Movement', example['speed'], lambda z: z['movement'].mean(0), 'Speed (cm/s)'),
        ('Left reward probability', example['probability'], lambda z: z['probability_effect'].mean(0)[0], 'Probability'),
        ('Right reward probability', example['probability'], lambda z: z['probability_effect'].mean(0)[1], 'Probability'),
        ('Previous trial', np.arange(3), lambda z: z['history'].mean(0), 'Previous trial contrast')]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    curve_rows = []
    for ax, (name, x, getter, xlabel) in zip(axes.flat, curve_specs):
        for c, part in d.groupby('cluster'):
            y = np.stack([getter(curves[u]) for u in part.unit_id])
            median = np.median(y, axis=0)
            ax.plot(x, median, color=color[c], label=f'C{c+1} · {len(part)}', lw=1.8)
            for xx, mm, lo, hi in zip(x, median, np.quantile(y, .25, axis=0), np.quantile(y, .75, axis=0)):
                curve_rows.append(dict(cluster=c, curve=name, x=xx, median=mm, q25=lo, q75=hi, n=len(y)))
        ax.axhline(0, color='#AEB9C5', lw=.7)
        ax.set_title(name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Log rate change')
        if name == 'Previous trial':
            ax.set_xticks([0, 1, 2], ['Response', 'L − R', 'Reward − omit'], fontsize=8)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, frameon=False, loc='center')
    fig.suptitle('Full GLM response curves · cluster medians', fontsize=15)
    fig.text(.5, .01, 'Average fits within each unit, then median across units · no responder filtering · curves are conditional associations', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .96))
    save(fig, 'response_curves')
    pd.DataFrame(curve_rows).to_csv(out / 'response_curve_summary.csv', index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, scope in enumerate(['All', 'SUA', 'QC SUA']):
        subset = scopes[scope]
        counts = subset.cluster.value_counts()
        subset = subset[subset.cluster.isin(counts[counts.ge(cfg.minimum_cluster_units)].index)]
        dots(axes[0, column], subset, 'Movement', 100)
        axes[0, column].set_title(f'{scope} · {len(subset)} units')
        axes[0, column].set_ylabel('Movement deviance gain (%)')
        for c, part in subset.groupby('cluster'):
            y = np.stack([curves[u]['movement'].mean(0) for u in part.unit_id])
            show = example['speed'] <= 1.15
            axes[1, column].plot(example['speed'][show], np.median(y, axis=0)[show],
                color=color[c], label=f'C{c+1} · {len(part)}', lw=2)
        axes[1, column].axhline(0, color='#AEB9C5', lw=.7)
        axes[1, column].set(xlim=(0, 1.15), xlabel='Speed (cm/s)', ylabel='Log rate change')
        axes[1, column].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle('Movement association · recording-quality checks', fontsize=15)
    fig.text(.5, .01, 'Lower panels: common speed range (0–95th percentile), median curves · clusters with ≥5 units per subset', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .96))
    save(fig, 'movement_quality')
    paths = [source/'units.csv', glm/'reliability.csv', glm/'unit_fit_status.csv', glm/'fit_source.json']
    method = dict(join='one-to-one global_unit_id; Brainsets source unit IDs agree',
        clustered_units=n_clustered, complete_fits=len(d), scopes={k: len(v) for k, v in scopes.items()},
        cluster_display='CSV cluster 0 is C1; colors and names preserved from clustering figures',
        glm_scope='0–4516.16 s', embedding_scope='0–4270 s',
        primary='Complete, converged three-fold fits; retain negative gains and all responders/nonresponders',
        tests='Nested rank-outcome OLS F tests; BH over 9 gains plus movement at 1 cm/s separately per scope and adjustment',
        controls='log1p(export-span rate), its square, embedding availability, depth, sorting, recorded label, source QC',
        minimum_cluster_units=int(cfg.minimum_cluster_units),
        curve_summary='Mean across folds per unit, then median across units; not confidence intervals',
        limitations=['Exploratory single-recording unit-level tests; correlated neurons weaken nominal p-value interpretation',
            'Scopes are sensitivity checks, not independent replications; BH does not span all scopes',
            'Cluster membership and GLM use the same spike recordings; not an independent prediction test',
            'Rate adjustment is a parametric sensitivity check and cannot remove all spike-statistics effects',
            'Cluster boundaries have moderate stability; no discrete biological identity established',
            'MUA are not necessarily single neurons; source QC and rate differ sharply among clusters',
            'Near-simultaneous choice/outcome estimates are conditional associations, not isolated causal responses'],
        hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'method.json').write_text(json.dumps(method, indent=2))
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Dec 10 cluster–GLM comparison</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 25px;color:#253247;line-height:1.5}img{width:100%}a{color:#3976B9}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:left}</style><h1>Dec 10 · embedding clusters and GLMs</h1>'
    page += f'<p>{len(d)} complete GLM fits among {n_clustered} clustered units. C1–C6 match the original clustering figures.</p>'
    page += '<p>Exploratory within-recording associations. Held-out GLM gains assess prediction within units; cluster comparisons are not independent validation. Movement is modeled jointly with task features and slow drift.</p>'
    for name, title in [('glm_gains', 'Held-out feature gains'), ('response_curves', 'Full conditional response curves'), ('movement_quality', 'Movement and unit quality')]:
        page += f'<h2>{title}</h2><a href="{name}.png"><img src="{name}.png"></a>'
    page += '<p><a href="associations.csv">Association tests</a> · <a href="cluster_glm_summary.csv">Cluster summaries</a> · <a href="unit_features.csv">Unit features</a> · <a href="cohort_audit.csv">Inclusion audit</a> · <a href="method.json">Methods and limitations</a></p>'
    page += '<p>Adjusted comparisons account for firing rate, availability, depth, sorting, recorded label and QC. Tests exclude clusters with fewer than five units in the relevant subset. Subsets are sensitivity checks; nominal p/q values assume independent units.</p>'
    page += composition.to_html(float_format=lambda x: f'{x:.3g}') + '</html>'
    (out/'index.html').write_text(page)
    print(composition.to_string())
    print(tests.loc[tests.scope.isin(['All', 'SUA', 'QC SUA'])].to_string(index=False))

