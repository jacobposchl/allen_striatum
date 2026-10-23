"""Unsupervised session coordinates colored by cached, individual GLM effects."""
import html
import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from src.common.analysis import sha256, write_json
from src.embeddings.io import read_table
from src.data.io import resolve
from src.common.plotting import STYLE


def run(cfg):
    out = resolve(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    root, feature_root = resolve(cfg.source), resolve(cfg.features)
    coordinate_path = resolve(cfg.coordinates)
    coordinates = read_table(coordinate_path)
    coordinates = coordinates[coordinates.session.eq(str(cfg.session))].copy().reset_index(drop=True)
    assert coordinates.global_unit_id.is_unique and len(coordinates) >= 4
    provenance = json.loads(coordinate_path.with_suffix('.meta.json').read_text())
    source = json.loads((root / f'{cfg.encoder}_source.json').read_text())
    for key in ['encoder_signature', 'data_fingerprint']:
        assert provenance[key] == source[key], f'Mismatched {key}'
    assert provenance['fit_scope'] == 'independent session'
    with np.load(root / 'embeddings.npz', allow_pickle=False) as archive:
        ix = pd.Index(archive['global_unit_id'].astype(str)).get_indexer(coordinates.global_unit_id)
        assert (ix >= 0).all()
        x = archive[str(cfg.encoder)][ix].astype(float)
    assert np.isfinite(x).all() and (np.linalg.norm(x, axis=1) > 0).all()
    # Preserve angular geometry; do not variance-standardize encoder dimensions.
    design = normalize(x)
    pca = PCA(n_components=min(int(cfg.pca_components), *design.shape), svd_solver='full')
    scores = pca.fit_transform(design)
    for j in range(scores.shape[1]):
        coordinates[f'pc_{j+1}'] = scores[:, j]
    np.savez_compressed(out / 'pca_model.npz', components=pca.components_, mean=pca.mean_,
                        variance_fraction=pca.explained_variance_ratio_,
                        global_unit_id=coordinates.global_unit_id.to_numpy(str), scores=scores)
    pd.DataFrame({'component': np.arange(1, scores.shape[1]+1),
                  'variance_fraction': pca.explained_variance_ratio_}).to_csv(out / 'pca_variance.csv', index=False)
    specs = {s['name']: s for s in json.loads((feature_root / 'feature_definitions.json').read_text())}
    reliability = read_table(feature_root / 'unit_features.csv')
    selected = list(cfg.selected_features)
    maps, counts, associations, long, archive_out = {}, [], [], [], {}
    with np.load(feature_root / 'features.npz', allow_pickle=False) as archive:
        feature_ix = pd.Index(archive['global_unit_id'].astype(str)).get_indexer(coordinates.global_unit_id)
        matched = feature_ix >= 0
        for name in selected:
            spec = specs[name]
            assert spec['kind'] == 'kernel'
            lag = archive[name + '_axis']
            kernel = archive[name][feature_ix[matched]]
            folds = archive[name + '_validation'][feature_ix[matched]]
            info = reliability[reliability.feature.eq(name)].set_index('global_unit_id').reindex(coordinates.global_unit_id)
            group_supported = info.parent_group_supported.eq(True).to_numpy(dtype=bool)
            windows = [('summary', float(cfg.summary_s[0]), float(cfg.summary_s[1]))]
            windows += [(f'window_{j}', float(w.start), float(w.stop)) for j, w in enumerate(cfg.temporal_windows)]
            values, stable = {}, {}
            for label, start, stop in windows:
                # Inclusive support matches the existing cached GLM summaries.
                mask = (lag >= start-1e-8) & (lag <= stop+1e-8)
                assert mask.any()
                value = np.full(len(coordinates), np.nan)
                value[matched] = kernel[:, mask].mean(1)
                agree = np.zeros(len(coordinates), bool)
                agree[matched] = (np.sign(folds[:, :, mask].mean(2)) == np.sign(value[matched, None])).all(1)
                values[label], stable[label] = value, group_supported & agree
                coordinates[name + '_' + label] = value
                coordinates[name + '_' + label + '_supported_stable'] = stable[label]
                for i, gid in enumerate(coordinates.global_unit_id):
                    long.append(dict(global_unit_id=gid, feature=name, window=label, start_s=start, stop_s=stop,
                        value=value[i], parent_group_supported=bool(group_supported[i]),
                        fold_sign_consistent=bool(agree[i]), matched_glm=bool(matched[i])))
            assert np.allclose(values['summary'][matched], info.value.to_numpy(float)[matched])
            limit = max(float(np.quantile(np.abs(np.concatenate([v[matched] for v in values.values()])),
                                           float(cfg.color_quantile))), 1e-8)
            maps[name] = dict(spec=spec, values=values, stable=stable, limit=limit)
            counts.append(dict(feature=name, label=spec['label'], n_matched=int(matched.sum()),
                n_supported=int(group_supported.sum()), n_supported_stable=int(stable['summary'].sum()),
                color_limit=limit, n_summary_saturated=int((np.abs(values['summary'][matched]) > limit).sum())))
            for j in range(scores.shape[1]):
                for scope, mask in [('All', matched), ('SUA', matched & coordinates.decoder_label.eq('sua').to_numpy())]:
                    associations.append(dict(feature=name, component=j+1, scope=scope, n_units=int(mask.sum()),
                        spearman=float(spearmanr(scores[mask, j], values['summary'][mask]).statistic)))
            archive_out[name], archive_out[name + '_axis'], archive_out[name + '_validation'] = kernel, lag, folds
    coordinates.to_csv(out / 'coordinates_and_features.csv', index=False)
    pd.DataFrame(long).to_csv(out / 'feature_windows.csv', index=False)
    pd.DataFrame(counts).to_csv(out / 'feature_coverage.csv', index=False)
    pd.DataFrame(associations).to_csv(out / 'pc_feature_associations.csv', index=False)
    np.savez_compressed(out / 'kernels.npz', global_unit_id=coordinates.loc[matched, 'global_unit_id'].to_numpy(str), **archive_out)
    figures = plot(cfg, out, coordinates, maps, pca.explained_variance_ratio_)
    method = dict(session=str(cfg.session), encoder=str(cfg.encoder), n_units=len(coordinates),
        n_matched_glm=int(matched.sum()), n_sua=int((matched & coordinates.decoder_label.eq('sua')).sum()),
        geometry='Cached independent-session cosine UMAP; PCA on L2-normalized embeddings of the same units, centered without dimension-wise variance scaling. Neither projection uses GLM features.',
        pca_variance_fraction=pca.explained_variance_ratio_.tolist(),
        coloring='Full-recording conditional GLM kernels, inclusive window means. Blue negative, red positive, white zero. Each feature has its own symmetric color range, shared between projections and time windows. Range is the configured absolute-value quantile across all displayed windows; saturation is recorded in feature_coverage.csv.',
        reliability='Outlined points have supported parent predictor group and matching summary sign in all three validation fits. This is a descriptive flag, not a confidence interval or a separate test of the exact contrast. Gray points lack a matched GLM feature in the selected cache; they are not zero effects.',
        interpretation='Exploratory visualization, not prediction or a significance test. UMAP distances and apparent gaps do not establish functional separation. PC-feature correlations are descriptive discovery statistics. Full kernels and validation kernels are preserved.',
        source_hashes={str(p): sha256(p) for p in [Path(__file__), coordinate_path, coordinate_path.with_suffix('.meta.json'), root / 'embeddings.npz', feature_root / 'features.npz', feature_root / 'unit_features.csv']})
    write_json(out / 'method.json', method)
    OmegaConf.save(cfg, out / 'config.yaml')
    body = ['<!doctype html><meta charset="utf-8"><title>Within-session GLM embedding maps</title>',
        '<style>body{font:16px system-ui;line-height:1.5;color:#243247;max-width:1100px;margin:36px auto;padding:0 24px}img{width:100%}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}</style>',
        f'<h1>{html.escape(str(cfg.session))}: individual GLM effects</h1>',
        f'<p>{len(coordinates)} units in both projections; {matched.sum()} with matched GLM features. The UMAP and PCA are unsupervised and fixed across every coloring. PC1 and PC2 retain {100*pca.explained_variance_ratio_[:2].sum():.1f}% of normalized embedding variance.</p>',
        '<p>Colors show full-recording conditional log-rate effects, not firing rates or model predictions. Primary maps average 0–1 s relative to the corresponding event: go cue for the cue kernel, registered choice/outcome for the other three. Right − left is the outcome-averaged choice contrast; reward − omission averages choice sides. Pre/early/late panels show how this summary can hide temporal differences.</p>',
        '<p>Blue: negative. Red: positive. White: zero. Gray: no matched feature. Outlines: supported parent group and consistent sign across all three validation fits. This flag does not establish the reliability of every individual coefficient. Color limits vary by feature, but are fixed between UMAP/PCA and time windows for that feature; extreme values may saturate.</p>',
        pd.DataFrame(counts).to_html(index=False, border=0, float_format=lambda v: f'{v:.3f}'),
        '<p>These are descriptive views. Separation in UMAP is not evidence of a discrete cell type, and no gradient in PC1/PC2 does not rule out structure in later dimensions. No alignment, cluster changes, GLM refitting, or encoder training was performed.</p>']
    for name in figures:
        body.append(f'<p><a href="{name}">{html.escape(name)}</a></p><img src="{name}">')
    body.append('<p><a href="coordinates_and_features.csv">Neuron coordinates and effects</a> · <a href="kernels.npz">Full kernels</a> · <a href="pc_feature_associations.csv">Descriptive PC associations</a> · <a href="method.json">Methods</a> · <a href="figures_and_tables.zip">Download</a></p>')
    (out / 'index.html').write_text('\n'.join(body) + '\n')
    write_json(out / 'integrity_checks.json', dict(unique_ids=True, provenance_matches=True,
        pca_reconstruction_matches=bool(np.allclose(pca.transform(design), scores)),
        primary_colors_match_existing_feature_cache=True, same_coordinates_all_features=True))
    with zipfile.ZipFile(out / 'figures_and_tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.suffix in ['.png', '.csv', '.json', '.yaml', '.html', '.npz']:
                archive.write(path, path.name)
    print(json.dumps({k: method[k] for k in ['session', 'n_units', 'n_matched_glm', 'n_sua', 'pca_variance_fraction']}, indent=2), flush=True)
    print('Completed:', out, flush=True)


def plot(cfg, out, frame, maps, variance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    figures = []
    def panel(ax, projection, value, stable, limit):
        columns = ['umap_1', 'umap_2'] if projection == 'UMAP' else ['pc_1', 'pc_2']
        xy = frame[columns].to_numpy()
        valid = np.isfinite(value)
        ax.scatter(*xy[~valid].T, color='#CCD3DC', s=18, linewidths=0, zorder=1)
        dots = ax.scatter(*xy[valid].T, c=value[valid], cmap='RdBu_r', norm=Normalize(-limit, limit),
            s=28, linewidths=0, alpha=.95, zorder=2)
        ax.scatter(*xy[stable].T, facecolors='none', edgecolors='#27364B', s=29, linewidths=.55, zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        for spine in ax.spines.values(): spine.set_visible(False)
        if projection == 'PCA':
            ax.set_xlabel(f'PC1 · {variance[0]:.1%}')
            ax.set_ylabel(f'PC2 · {variance[1]:.1%}')
        return dots
    def save(fig, name):
        fig.savefig(out / name, dpi=200, bbox_inches='tight')
        plt.close(fig)
        figures.append(name)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(maps), 2, figsize=(9, 2.65*len(maps)), layout='constrained', squeeze=False)
        for row, (name, item) in enumerate(maps.items()):
            for col, projection in enumerate(['UMAP', 'PCA']):
                dots = panel(axes[row, col], projection, item['values']['summary'], item['stable']['summary'], item['limit'])
                axes[row, col].set_title(item['spec']['label'] + ' · ' + projection, fontsize=10)
            fig.colorbar(dots, ax=axes[row, :], shrink=.75, pad=.025, extend='both', label='Log-rate effect')
        session_label = date.fromisoformat(str(cfg.session)[:10]).strftime('%B %d').replace(' 0', ' ')
        fig.suptitle(session_label + ' · GLM effects, 0–1 s', fontsize=13)
        save(fig, 'glm_umap_pca_overview.png')
        for name, item in maps.items():
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
            for ax, projection in zip(axes, ['UMAP', 'PCA']):
                dots = panel(ax, projection, item['values']['summary'], item['stable']['summary'], item['limit'])
                ax.set_title(projection)
            fig.colorbar(dots, ax=axes, shrink=.75, extend='both', label='Log-rate effect')
            fig.suptitle(item['spec']['label'] + ' · 0–1 s', fontsize=13)
            save(fig, name + '_umap_pca.png')
            fig, axes = plt.subplots(2, len(cfg.temporal_windows), figsize=(12, 6.5), layout='constrained', squeeze=False)
            for row, projection in enumerate(['UMAP', 'PCA']):
                for col, window in enumerate(cfg.temporal_windows):
                    key = f'window_{col}'
                    dots = panel(axes[row, col], projection, item['values'][key], item['stable'][key], item['limit'])
                    axes[row, col].set_title(str(window.label), fontsize=10)
                axes[row, 0].set_ylabel(projection + (f' · PC2 {variance[1]:.1%}' if row else ''))
            fig.colorbar(dots, ax=axes.ravel().tolist(), shrink=.7, extend='both', label='Log-rate effect')
            fig.suptitle(item['spec']['label'] + ' · time windows', fontsize=13)
            save(fig, name + '_temporal_maps.png')
    return figures
