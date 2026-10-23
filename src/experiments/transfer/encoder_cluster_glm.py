"""Compare label-free encoder clusters against the frozen GLM cache.

This analysis never fits a GLM or uses GLM information to match clusters.
All computations, including plotting, must run on an existing compute allocation.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import f as f_distribution, rankdata

from src.experiments.clustering.cluster_glm import bh

GAINS = {
    'Full': 'gain_full', 'Cue': 'gain_cue',
    'Choice / outcome': 'gain_choice_common', 'Choice side': 'gain_choice_side',
    'Outcome': 'gain_outcome', 'Licking': 'gain_licking',
    'Movement': 'gain_movement', 'Reward probability': 'gain_probability',
    'History': 'gain_history',
}
KERNELS = ['cue', 'choice_common', 'choice_side', 'outcome', 'side_outcome',
           'left_lick', 'right_lick']
FEATURES = list(GAINS.values()) + ['kernel_' + k for k in KERNELS]
SHARED = [k for k in FEATURES if k != 'gain_movement']
LABELS = dict(zip(FEATURES, ['Full', 'Cue', 'Choice', 'Side', 'Outcome', 'Licking',
    'Movement', 'Probability', 'History', 'Cue effect', 'Choice effect',
    'Side effect', 'Outcome effect', 'Side × outcome', 'Left lick effect',
    'Right lick effect']))
ENCODERS = ['historical', 'passive_pretrained', 'finetuned']
NAMES = {'historical': 'Sep 12', 'passive_pretrained': 'Passive pretrained',
         'finetuned': 'Fine-tuned'}
PHYSIO = ['analysis_firing_rate_hz', 'analysis_isi_cv', 'analysis_frac_isi_lt_2ms',
          'half_width_ms', 'peak_to_valley_ms', 'depth', 'available_fraction',
          'decoder_label', 'default_qc', 'label', 'ccf_acronym']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def targets(cfg):
    """Publish reusable unit targets without requiring clustering outputs."""
    root, out = Path(cfg.glm), Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(root / 'unit_index.csv', dtype={'animal': str})
    units = units.loc[units.animal.eq(str(cfg.animal))].copy()
    rel = pd.read_csv(root / 'reliability.csv')
    good = rel.n_folds.eq(3) & rel.all_converged.eq(True)
    rel = rel.loc[good & rel.global_unit_id.isin(units.global_unit_id)]
    gains = rel.pivot(index='global_unit_id', columns='group', values='heldout_gain').rename(columns=GAINS)
    valid_full = set(rel.loc[rel.group.eq('Full'), 'global_unit_id'])
    d = units[['global_unit_id', 'session', 'id', 'final_status', 'final_file']].merge(
        gains, on='global_unit_id', how='left', validate='one_to_one')
    d['analysis_included'] = d.final_status.eq('fit') & d.global_unit_id.isin(valid_full)
    for feature in FEATURES:
        if feature not in d:
            d[feature] = np.nan
    curve_cache = {}
    for i, row in d.loc[d.analysis_included].iterrows():
        with np.load(root / row.final_file, allow_pickle=False) as z:
            curve_cache[row.global_unit_id] = {k: z[k].copy() for k in z.files}
            lags = z['lags_s']
            names = list(z['kernel_names'])
            for name in KERNELS:
                window = ((lags >= -.10001) & (lags <= .10001) if 'lick' in name
                          else (lags >= -.00001) & (lags <= 1.00001))
                d.loc[i, 'kernel_' + name] = z['kernels'][names.index(name), window].mean()
    d.to_csv(out / 'glm_targets.csv', index=False)
    definitions = {'gains': {v: k for k, v in GAINS.items()},
        'gain_definition': 'Held-out drop-group deviance gain; Full is versus drift baseline. Three converged CV folds required.',
        'kernels': {f'kernel_{k}': {'source': k, 'interval_s': [-.1, .1] if 'lick' in k else [0, 1],
                                  'units': 'mean final-fit log-rate effect'} for k in KERNELS},
        'curves': 'Full final-fit kernels are retained in the original frozen cache; summary intervals are descriptive and do not replace them.',
        'primary_inclusion': 'Converged final fit and three converged Full CV folds; no positive-gain selection.',
        'unit_index_sha256': sha256(root / 'unit_index.csv'),
        'reliability_sha256': sha256(root / 'reliability.csv')}
    (out / 'target_definitions.json').write_text(json.dumps(definitions, indent=2))
    print(f'Published GLM targets: {len(d)} indexed, {d.analysis_included.sum()} eligible', flush=True)
    return d, curve_cache


def basis(x):
    """An orthonormal basis that handles constant/collinear covariates."""
    if x.shape[1] == 0:
        return np.zeros((len(x), 0))
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    keep = s > max(x.shape) * np.finfo(float).eps * max(s[0], 1.)
    return u[:, keep]


def covariates(frame, adjustment):
    if adjustment == 'Raw':
        return np.ones((len(frame), 1))
    rate = np.log1p(frame.analysis_firing_rate_hz.to_numpy(float))
    if adjustment == 'Flexible physiology':
        from sklearn.preprocessing import SplineTransformer
        spline = SplineTransformer(n_knots=5, degree=3, include_bias=False)
        columns = list(spline.fit_transform(rate[:, None]).T)
    else:
        columns = [rate, rate ** 2]
    columns += [frame.available_fraction.to_numpy(float), frame.depth.to_numpy(float)]
    if adjustment in ['Physiology', 'Flexible physiology']:
        columns += [np.log1p(frame.analysis_isi_cv.to_numpy(float)),
            frame.analysis_frac_isi_lt_2ms.to_numpy(float),
            frame.half_width_ms.to_numpy(float), frame.peak_to_valley_ms.to_numpy(float)]
    x = np.column_stack(columns)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)
    cats = pd.get_dummies(frame[['decoder_label', 'default_qc']].astype(str),
                          drop_first=True, dtype=float).to_numpy()
    return np.column_stack([np.ones(len(frame)), x, cats])


def required(adjustment):
    if adjustment == 'Raw':
        return []
    cols = ['analysis_firing_rate_hz', 'available_fraction', 'depth',
            'decoder_label', 'default_qc']
    if adjustment in ['Physiology', 'Flexible physiology']:
        cols += ['analysis_isi_cv', 'analysis_frac_isi_lt_2ms',
                 'half_width_ms', 'peak_to_valley_ms']
    return cols


def association(frame, feature, adjustment, cfg):
    d = frame.dropna(subset=[feature] + required(adjustment)).copy()
    counts = d.cluster.value_counts()
    d = d.loc[d.cluster.isin(counts[counts.ge(int(cfg.minimum_cluster_units))].index)]
    result = dict(n=len(d), clusters=d.cluster.nunique(), partial_r2=np.nan,
                  p_f=np.nan, p_permutation=np.nan, df_cluster=0, df_residual=0)
    if len(d) < 10 or d.cluster.nunique() < 2:
        return result
    y = rankdata(d[feature].to_numpy(float))
    q0 = basis(covariates(d, adjustment))
    cluster = pd.get_dummies(d.cluster, drop_first=True, dtype=float).to_numpy()
    cluster -= q0 @ (q0.T @ cluster)
    qc = basis(cluster)
    resid = y - q0 @ (q0.T @ y)
    ss0 = resid @ resid
    explained = np.square(qc.T @ resid).sum()
    ss1 = max(ss0 - explained, 1e-12)
    df1, df2 = qc.shape[1], len(d) - q0.shape[1] - qc.shape[1]
    if df1 < 1 or df2 < 3 or ss0 < 1e-12:
        return result
    effect = min(1., max(0., explained / ss0))
    statistic = (explained / df1) / (ss1 / df2)
    # Freedman-Lane residual permutations, restricted to SUA/MUA strata.
    rng = np.random.default_rng(int(cfg.seed))
    indices = np.tile(np.arange(len(d))[:, None], (1, int(cfg.permutations)))
    for _, positions in d.reset_index(drop=True).groupby('decoder_label').indices.items():
        indices[positions] = np.column_stack([rng.permutation(positions) for _ in range(int(cfg.permutations))])
    perm = resid[indices]
    perm -= q0 @ (q0.T @ perm)
    null = np.square(qc.T @ perm).sum(0) / np.maximum(np.square(perm).sum(0), 1e-12)
    result.update(partial_r2=effect, p_f=float(f_distribution.sf(statistic, df1, df2)),
        p_permutation=float((1 + np.sum(null >= effect - 1e-12)) / (len(null) + 1)),
        df_cluster=df1, df_residual=df2)
    return result


def profile(frame, cluster_ids, adjustment):
    """Within-recording functional contrasts on comparable percentile scales."""
    matrix = np.full((len(cluster_ids), len(SHARED)), np.nan)
    for j, feature in enumerate(SHARED):
        d = frame.dropna(subset=[feature] + required(adjustment))
        if len(d) < 5:
            continue
        y = rankdata(d[feature].to_numpy(float)) / len(d)
        q = basis(covariates(d, adjustment))
        residual = y - q @ (q.T @ y)
        means = pd.Series(residual, index=d.index).groupby(d.cluster).mean()
        matrix[:, j] = [means.get(c, np.nan) for c in cluster_ids]
    return matrix


def similarity(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    a, b = a[ok], b[ok]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else np.nan


def matching(frame, arrays, cfg):
    out = Path(cfg.output)
    matches, validations, profiles, matrix_rows = [], [], [], []
    first, second = map(str, cfg.sessions)
    meta = pd.read_csv(Path(cfg.source) / 'units.csv')
    index = {uid: i for i, uid in enumerate(meta.global_unit_id)}
    for encoder in ENCODERS:
        d = frame.loc[frame.encoder.eq(encoder)].copy()
        # Mapping uses all clustered units, including those without eligible GLMs.
        groups = [d.loc[d.session.eq(s)] for s in [first, second]]
        ids = [sorted(p.cluster.unique()) for p in groups]
        vectors = [arrays[encoder][[index[u] for u in p.global_unit_id]].astype(float) for p in groups]
        vectors = [x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12) for x in vectors]
        centroids = [np.stack([x[p.cluster.to_numpy() == c].mean(0) for c in cs])
                     for p, x, cs in zip(groups, vectors, ids)]
        centered = [c - x.mean(0) for c, x in zip(centroids, vectors)]
        physcols = ['analysis_firing_rate_hz', 'analysis_isi_cv', 'analysis_frac_isi_lt_2ms',
                    'half_width_ms', 'peak_to_valley_ms']
        phys = d[physcols].to_numpy(float)
        phys[:, :2] = np.log1p(phys[:, :2])
        median = np.nanmedian(phys, axis=0)
        scale = np.maximum(np.nanquantile(phys, .75, axis=0) - np.nanquantile(phys, .25, axis=0), 1e-6)
        d.loc[:, physcols] = (phys - median) / scale
        pcentroids = [np.stack([d.loc[d.session.eq(s) & d.cluster.eq(c), physcols].median().to_numpy()
                                for c in cs]) for s, cs in zip([first, second], ids)]
        distances = {
            'Embedding centroid': cdist(*centroids, metric='cosine'),
            'Centered centroid': cdist(*centered, metric='cosine'),
            'Physiology': cdist(*pcentroids, metric='euclidean'),
        }
        for method, cost in distances.items():
            assert np.isfinite(cost).all(), (encoder, method)
            row_ids, col_ids = linear_sum_assignment(cost)
            for i in range(len(ids[0])):
                for j in range(len(ids[1])):
                    matrix_rows.append(dict(encoder=encoder, method=method,
                        cluster_a=ids[0][i], cluster_b=ids[1][j], distance=cost[i, j],
                        matched=bool(((row_ids == i) & (col_ids == j)).any())))
            for i, j in zip(row_ids, col_ids):
                alternatives = np.delete(cost[i], j)
                matches.append(dict(encoder=encoder, method=method, session_a=first,
                    session_b=second, cluster_a=ids[0][i], cluster_b=ids[1][j],
                    n_a=int(groups[0].cluster.eq(ids[0][i]).sum()),
                    n_b=int(groups[1].cluster.eq(ids[1][j]).sum()), distance=float(cost[i, j]),
                    glm_n_a=int((groups[0].cluster.eq(ids[0][i]) & groups[0].analysis_included).sum()),
                    glm_n_b=int((groups[1].cluster.eq(ids[1][j]) & groups[1].analysis_included).sum()),
                    sua_n_a=int((groups[0].cluster.eq(ids[0][i]) & groups[0].decoder_label.eq('sua') & groups[0].analysis_included).sum()),
                    sua_n_b=int((groups[1].cluster.eq(ids[1][j]) & groups[1].decoder_label.eq('sua') & groups[1].analysis_included).sum()),
                    row_margin=float(alternatives.min() - cost[i, j]) if len(alternatives) else np.nan,
                    mutual_nearest=bool(np.argmin(cost[i]) == j and np.argmin(cost[:, j]) == i)))
            for scope in ['All', 'SUA']:
                valid = frame.loc[frame.encoder.eq(encoder) & frame.analysis_included.fillna(False)].copy()
                if scope == 'SUA':
                    valid = valid.loc[valid.decoder_label.eq('sua')]
                for adjustment in ['Raw', 'Physiology']:
                    pa, pb = [profile(valid.loc[valid.session.eq(s)], cs, adjustment)
                              for s, cs in zip([first, second], ids)]
                    for i, j in zip(row_ids, col_ids):
                        for f, va, vb in zip(SHARED, pa[i], pb[j]):
                            profiles.append(dict(encoder=encoder, method=method, scope=scope,
                                adjustment=adjustment, cluster_a=ids[0][i], cluster_b=ids[1][j],
                                feature=f, value_a=va, value_b=vb))
                    observed = similarity(pa[row_ids], pb[col_ids])
                    n = len(pb)
                    if n <= 8:
                        perms = np.array(list(itertools.permutations(range(n))), dtype=int)
                        exact = True
                    else:
                        rng = np.random.default_rng(int(cfg.seed))
                        perms = np.stack([rng.permutation(n) for _ in range(int(cfg.matching_permutations))])
                        exact = False
                    null = np.array([similarity(pa[row_ids], pb[p[col_ids]]) for p in perms])
                    null = null[np.isfinite(null)]
                    if not np.isfinite(observed) or not len(null):
                        pvalue = np.nan
                    elif exact:
                        pvalue = np.mean(null >= observed - 1e-12)
                    else:
                        pvalue = (1 + np.sum(null >= observed - 1e-12)) / (1 + len(null))
                    validations.append(dict(encoder=encoder, method=method, scope=scope,
                        adjustment=adjustment, pairs=len(row_ids), profile_cosine=observed,
                        null_mean=np.mean(null) if len(null) else np.nan, p=pvalue,
                        permutations=len(null), exact=exact))
    pd.DataFrame(matches).to_csv(out / 'independent_cluster_matches.csv', index=False)
    pd.DataFrame(matrix_rows).to_csv(out / 'matching_distances.csv', index=False)
    pd.DataFrame(profiles).to_csv(out / 'matched_functional_profiles.csv', index=False)
    valid = pd.DataFrame(validations)
    valid['q'] = bh(valid.p.to_numpy())
    valid.to_csv(out / 'cross_session_replication.csv', index=False)
    return pd.DataFrame(matches), valid


def plots(data, associations, curves, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.experiments.clustering.plots import PALETTE, style
    style()
    out = Path(cfg.output)
    def save(fig, path):
        fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
    for scope, adjustment in itertools.product(['All', 'SUA'], ['Physiology', 'Flexible physiology']):
        fig, ax = plt.subplots(figsize=(13, 4.8))
        sub = associations.loc[associations.scope.eq(scope) & associations.adjustment.eq(adjustment)]
        rows = [(s, e) for s in map(str, cfg.sessions) for e in ENCODERS]
        mat = sub.pivot(index=['session', 'encoder'], columns='feature', values='partial_r2').reindex(index=rows, columns=FEATURES)
        q = sub.pivot(index=['session', 'encoder'], columns='feature', values='q_permutation').reindex(index=rows, columns=FEATURES)
        cmap = plt.get_cmap('Blues').copy(); cmap.set_bad('#eeeeee')
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=max(.3, np.nanmax(mat.to_numpy())))
        for i, j in zip(*np.where(q.to_numpy() < .05)):
            ax.text(j, i, '•', ha='center', va='center', color='#e47728', fontsize=14)
        ax.set_xticks(range(len(FEATURES)), [LABELS[f] for f in FEATURES], rotation=45, ha='right')
        ax.set_yticks(range(len(rows)), [f'{s[5:10]} · {NAMES[e]}' for s, e in rows])
        ax.set_title(f'Cluster–GLM association · {scope} · {adjustment.lower()} adjusted')
        fig.colorbar(im, ax=ax, label='Partial R²', shrink=.8)
        fig.text(.16, .015, 'Orange dot: exploratory q < 0.05', fontsize=9)
        suffix = '_flexible' if adjustment == 'Flexible physiology' else ''
        fig.tight_layout(rect=(0, .04, 1, 1)); save(fig, out / f'associations_{scope.lower()}{suffix}')
    curve_rows = []
    for session in map(str, cfg.sessions):
        for scope in ['All', 'SUA']:
            fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex='col', sharey='col')
            for row, encoder in enumerate(ENCODERS):
                d = data.loc[data.session.eq(session) & data.encoder.eq(encoder) & data.analysis_included.fillna(False)]
                if scope == 'SUA':
                    d = d.loc[d.decoder_label.eq('sua')]
                for column, kernel in enumerate(['cue', 'left_lick', 'right_lick']):
                    ax = axes[row, column]
                    for cluster, part in d.groupby('cluster'):
                        if len(part) < int(cfg.minimum_cluster_units):
                            continue
                        items = [curves[u] for u in part.global_unit_id]
                        lags = items[0]['lags_s']
                        z = np.stack([item['kernels'][list(item['kernel_names']).index(kernel)] for item in items])
                        median = np.median(z, axis=0)
                        mask = np.ones(len(lags), bool) if kernel == 'cue' else (lags >= -.4) & (lags <= .6)
                        ax.plot(lags[mask], median[mask], color=PALETTE[int(cluster) % len(PALETTE)],
                                lw=1.8, label=f'C{int(cluster)+1} · {len(part)}')
                        for t, m, low, high in zip(lags, median, np.quantile(z, .25, axis=0), np.quantile(z, .75, axis=0)):
                            curve_rows.append(dict(session=session, encoder=encoder, scope=scope,
                                cluster=int(cluster), kernel=kernel, time_s=t, median=m, q25=low,
                                q75=high, n=len(part)))
                    ax.axhline(0, color='#bcc4cb', lw=.7)
                    ax.axvline(0, color='#bcc4cb', lw=.7, ls=':')
                    if row == 0:
                        ax.set_title(kernel.replace('_', ' ').capitalize())
                    if column == 0:
                        ax.set_ylabel(NAMES[encoder] + '\nLog-rate effect')
                    if row == 2:
                        ax.set_xlabel('Time (s)')
                    if column == 2:
                        ax.legend(frameon=False, fontsize=7, loc='best', ncols=2)
            fig.suptitle(f'{session[:10]} · {scope} · final GLM kernels')
            fig.tight_layout(); save(fig, out / f'kernels_{session[:10]}_{scope.lower()}')
    pd.DataFrame(curve_rows).to_csv(out / 'cluster_kernel_curves.csv', index=False)
    matched = pd.read_csv(out / 'matched_functional_profiles.csv')
    for adjustment in ['Raw', 'Physiology']:
        d = matched.loc[matched.method.eq('Embedding centroid') & matched.scope.eq('All') & matched.adjustment.eq(adjustment)]
        rows = list(d[['encoder', 'cluster_a', 'cluster_b']].drop_duplicates().itertuples(index=False, name=None))
        a = d.pivot(index=['encoder', 'cluster_a', 'cluster_b'], columns='feature', values='value_a').reindex(index=rows, columns=SHARED)
        b = d.pivot(index=['encoder', 'cluster_a', 'cluster_b'], columns='feature', values='value_b').reindex(index=rows, columns=SHARED)
        vmax = max(.1, np.nanmax(np.abs(np.r_[a.to_numpy(), b.to_numpy()])))
        fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5, len(rows) * .27)), sharey=True)
        for ax, matrix, date in zip(axes, [a, b], ['December 10', 'December 18']):
            im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_xticks(range(len(SHARED)), [LABELS[f] for f in SHARED], rotation=60, ha='right', fontsize=8)
            ax.set_yticks(range(len(rows)), [f'{NAMES[e]} · C{int(c)+1} → C{int(k)+1}' for e, c, k in rows], fontsize=8)
            ax.set_title(date)
        fig.suptitle(f'Embedding-matched groups · {adjustment.lower()} functional profiles')
        fig.colorbar(im, ax=axes, label='Mean centered percentile', fraction=.025, pad=.025)
        fig.subplots_adjust(left=.22, right=.86, bottom=.25, top=.88, wspace=.08)
        save(fig, out / f'cross_session_profiles_{adjustment.lower()}')


def run(cfg):
    from omegaconf import OmegaConf
    out, source = Path(cfg.output), Path(cfg.source)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / 'config.yaml')
    target, curves = targets(cfg)
    meta = pd.read_csv(source / 'units.csv', dtype={'animal': str})
    clusters = pd.read_csv(source / 'clusters.csv')
    assert not clusters.duplicated(['encoder', 'global_unit_id']).any()
    assert meta.global_unit_id.is_unique
    phys = pd.read_csv(cfg.physiology)
    phys = phys[['global_unit_id'] + PHYSIO]
    assert phys.global_unit_id.is_unique
    # Source metadata wins for availability if the common-window analysis exports it.
    keep = [c for c in meta if c not in PHYSIO and c not in ['cluster', 'encoder']]
    d = clusters[['encoder', 'global_unit_id', 'cluster']].merge(meta[keep], on='global_unit_id', validate='many_to_one')
    d = d.merge(phys, on='global_unit_id', validate='many_to_one', how='left')
    if 'available_fraction' in meta:
        availability = meta.set_index('global_unit_id').available_fraction
        d['available_fraction'] = d.global_unit_id.map(availability)
    d = d.merge(target[['global_unit_id', 'analysis_included', 'final_file'] + FEATURES],
                on='global_unit_id', validate='many_to_one', how='left')
    d['analysis_included'] = d.analysis_included.fillna(False).astype(bool)
    d = d.loc[d.session.isin(list(map(str, cfg.sessions))) & d.cluster.ge(0)].copy()
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.to_csv(out / 'unit_features.csv', index=False)
    audit = d.groupby(['encoder', 'session']).agg(clustered=('global_unit_id', 'size'),
        glm_eligible=('analysis_included', 'sum'), physiology_complete=('analysis_isi_cv', 'count')).reset_index()
    audit.to_csv(out / 'coverage.csv', index=False)
    tests, summaries = [], []
    for (encoder, session), part in d.loc[d.analysis_included].groupby(['encoder', 'session']):
        print(f'GLM associations: {encoder} {session} n={len(part)}', flush=True)
        for scope in ['All', 'SUA']:
            subset = part if scope == 'All' else part.loc[part.decoder_label.eq('sua')]
            for feature in FEATURES:
                for adjustment in ['Raw', 'Rate + quality', 'Physiology', 'Flexible physiology']:
                    tests.append(dict(encoder=encoder, session=session, scope=scope, feature=feature,
                        adjustment=adjustment, **association(subset, feature, adjustment, cfg)))
                for cluster, group in subset.groupby('cluster'):
                    values = group[feature].dropna()
                    summaries.append(dict(encoder=encoder, session=session, scope=scope, feature=feature,
                        cluster=int(cluster), n=len(values), median=values.median(),
                        q25=values.quantile(.25), q75=values.quantile(.75)))
    tests = pd.DataFrame(tests)
    tests['q_permutation'] = tests.groupby(['scope', 'adjustment']).p_permutation.transform(lambda x: bh(x.to_numpy()))
    tests['q_f'] = tests.groupby(['scope', 'adjustment']).p_f.transform(lambda x: bh(x.to_numpy()))
    tests.to_csv(out / 'associations.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'cluster_glm_summary.csv', index=False)
    composition = d.groupby(['encoder', 'session', 'cluster']).agg(n=('global_unit_id', 'size'),
        glm_eligible=('analysis_included', 'sum'), sua=('decoder_label', lambda x: x.eq('sua').sum()),
        d1=('label', lambda x: x.eq('D1').sum()), d2=('label', lambda x: x.eq('D2').sum()),
        median_rate_hz=('analysis_firing_rate_hz', 'median'), median_isi_cv=('analysis_isi_cv', 'median'),
        median_half_width_ms=('half_width_ms', 'median'), median_peak_to_valley_ms=('peak_to_valley_ms', 'median')).reset_index()
    composition.to_csv(out / 'cluster_composition.csv', index=False)
    with np.load(source / 'embeddings.npz', allow_pickle=False) as z:
        assert np.array_equal(z['global_unit_id'].astype(str), meta.global_unit_id.to_numpy(str))
        arrays = {k: z[k].copy() for k in ENCODERS}
    matching(d, arrays, cfg)
    plots(d, tests, curves, cfg)
    method = dict(
        association='Nested regressions of ranked GLM features on cluster identity, restricted to clusters with at least five units per scope. Freedman-Lane residual permutations within SUA/MUA strata; 999 permutations.',
        multiple_testing='BH across encoders × sessions × 16 targets separately for each scope and adjustment. Cross-session alignment BH across all 36 matching/scope/adjustment comparisons.',
        controls='Rate + quality: log1p rate and its square, availability, depth, sorting and QC. Physiology additionally log1p ISI CV, fraction ISI <2ms, waveform half-width and peak-to-valley. Flexible physiology replaces the quadratic log-rate term with a cubic spline having five evenly spaced log-rate knots. No cell-type labels used as controls or to choose groups.',
        physiology_source='Existing frozen Sep12 per-recording clustering physiology. Spike statistics describe its full exported span; any different window spans are documented by encoder-comparison audit.',
        cross_session_matching='Maximum total similarity one-to-one Hungarian assignment using same-encoder cosine centroids of unit-normalized vectors; centered-centroid and robust-scaled-physiology sensitivities. All units used for matching, eligible GLM units for subsequent evaluation. GLM information never used to select assignments.',
        cross_session_validation='Cosine of flattened cluster mean within-recording rank residual profiles across 15 shared targets. Compare against permutations of all December18 cluster identities; exhaustive for up to eight clusters. Does not assume matched groups truly correspond.',
        limitations=['Same animal; no independent-animal replication.',
            'Encoders and clusters saw these recordings; GLM validation does not make the cluster analysis independent.',
            'Unit-level approximate permutation inference ignores possible population dependence and post-selection of recordings.',
            'December18 has no movement covariate and no anatomical labels; negative or weak feature gains must not be interpreted as established encoding.',
            'Forced matches can be ambiguous when session identity dominates; inspect mutual nearest status and distance margins.',
            'Cross-session profile permutation assumes cluster profiles exchangeable under the null despite unequal cluster sizes; it is an exploratory reassignment benchmark. Some SUA profiles have few units, and all profile sample sizes are saved.',
            'Historical exports included noise units in population attention; the new exports did not. Old/new comparisons therefore also change population context. Passive-pretrained versus fine-tuned exports share context.',
            'Kernel summaries are descriptive selected intervals; full final kernels remain available.',
            'SUA subset retains labels learned with all units; it is a sensitivity check, not a separate clustering.'],
        source_hashes={str(p): sha256(p) for p in [source / 'units.csv', source / 'clusters.csv', Path(cfg.glm) / 'unit_index.csv', Path(cfg.glm) / 'reliability.csv', Path(cfg.physiology)]})
    (out / 'method.json').write_text(json.dumps(method, indent=2))
    (out / 'checks.json').write_text(json.dumps(dict(unique_cluster_ids=True, unique_metadata_ids=True,
        embedding_row_identity_verified=True, no_glm_refits=True, coverage=audit.to_dict('records')), indent=2))
    print('Complete: encoder cluster–GLM analysis', flush=True)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/experiments/transfer/configs/encoder_cluster_glm.yaml')
    parser.add_argument('--targets-only', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    targets(cfg) if args.targets_only else run(cfg)
