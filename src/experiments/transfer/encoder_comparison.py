"""Matched-unit unsupervised geometry and identity probes across three encoders."""
import itertools
import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, pairwise_distances, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.impute import SimpleImputer

from src.experiments.clustering.clustering import communities, bh
from src.embeddings.io import read_table
from src.data.io import resolve
from src.embeddings.joint_umap import cached_means
from src.common.plotting import STYLE

MODELS = ['historical', 'passive_pretrained', 'finetuned']
NAMES = {'historical': 'Active pretrained (Sep 12)', 'passive_pretrained': 'Passive pretrained', 'finetuned': 'Fine-tuned'}


def json_write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def baseline(frame):
    cols = ['analysis_firing_rate_hz', 'analysis_isi_cv', 'analysis_frac_isi_lt_2ms',
            'analysis_frac_isi_lt_10ms', 'analysis_frac_isi_gt_500ms',
            'peak_to_valley_ms', 'half_width_ms', 'depth', 'available_fraction']
    array = frame[cols].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    array[:, :2] = np.log1p(array[:, :2])
    return np.column_stack([array, frame.decoder_label.eq('mua').to_numpy(float)])


def probe(values, y, name, task, frame, seed):
    prediction = np.empty(len(y), dtype=object)
    rows, folds = [], np.full(len(y), -1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(cv.split(values, y)):
        model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                              LogisticRegression(C=1., class_weight='balanced', max_iter=3000))
        model.fit(values[train], y[train])
        prediction[test] = model.predict(values[test])
        folds[test] = fold
        rows.append(dict(encoder=name, task=task, fold=fold,
                         balanced_accuracy=balanced_accuracy_score(y[test], prediction[test]), n_test=len(test)))
    records = frame[['global_unit_id', 'session']].copy()
    records['encoder'], records['task'], records['truth'] = name, task, y
    records['prediction'], records['fold'] = prediction, folds
    return rows, records, dict(encoder=name, task=task, n_units=len(y),
        balanced_accuracy=balanced_accuracy_score(y, prediction), chance=1/len(np.unique(y)))


def run(cfg):
    output = resolve(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
    sources, matrices, frames = {}, {}, {}
    historical = resolve(cfg.historical)
    old = read_table(cfg.historical_clusters)
    old = old[old.animal.eq('745502') & old.decoder_label.isin(['sua', 'mua'])].copy()
    frame = read_table(historical / 'unit_metadata.csv')
    with np.load(historical / 'mean_embeddings.npz', allow_pickle=False) as data:
        assert np.array_equal(frame.global_unit_id.astype(str), data['global_unit_id'])
        matrices['historical'] = data['embeddings']
    frames['historical'] = frame
    sources['historical'] = json.loads((historical / 'mean_embeddings.meta.json').read_text())
    for name in MODELS[1:]:
        matrices[name], frames[name], sources[name] = cached_means(cfg[name], output, name)
    common = sorted(set(old.global_unit_id).intersection(*(set(f.global_unit_id) for f in frames.values())))
    unit_frame = old.set_index('global_unit_id').loc[common].reset_index()
    audit = []
    for name in MODELS:
        positions = pd.Series(np.arange(len(frames[name])), index=frames[name].global_unit_id).loc[common].to_numpy()
        metadata = frames[name].iloc[positions].reset_index(drop=True)
        matrices[name] = matrices[name][positions].astype(float)
        assert np.isfinite(matrices[name]).all()
        for key in ['animal', 'session', 'recording', 'unit_id', 'label', 'decoder_label', 'source_unit_row']:
            assert np.array_equal(metadata[key].astype(str), unit_frame[key].astype(str)), (name, key)
        unit_frame[f'n_windows_{name}'] = metadata.n_windows.to_numpy()
        for session, group in metadata.groupby('session'):
            audit.append(dict(encoder=name, session=session, n_units=len(group),
                              dimensions=matrices[name].shape[1], n_windows_min=int(group.n_windows.min()),
                              n_windows_max=int(group.n_windows.max())))
    unit_frame.to_csv(output / 'units.csv', index=False)
    availability_equal = all(np.array_equal(unit_frame.n_windows_historical, unit_frame[f'n_windows_{name}']) for name in MODELS[1:])
    if not availability_equal:
        raise ValueError('Per-unit window counts differ across encoders; match windows before comparison')
    json_write(output / 'checks.json', dict(matched_unit_identities=True,
        matched_labels_sorting_and_source_rows=True, finite_embeddings=True,
        identical_per_unit_window_counts=availability_equal))
    np.savez_compressed(output / 'embeddings.npz', **matrices,
                        global_unit_id=unit_frame.global_unit_id.to_numpy(dtype=str))
    pd.DataFrame(audit).to_csv(output / 'coverage.csv', index=False)
    json_write(output / 'sources.json', sources)
    clusters, stability, summaries, comparisons, old_check = [], [], [], [], []
    rng = np.random.default_rng(int(cfg.seed))
    for session, group in unit_frame.groupby('session', sort=True):
        positions, n = group.index.to_numpy(), len(group)
        print(f'Clustering and comparing {session}: {n} matched units', flush=True)
        distances, labels, kernels = {}, {}, {}
        for name in MODELS:
            values = matrices[name][positions]
            distance = pairwise_distances(values, metric='cosine')
            np.fill_diagonal(distance, 0)
            label, modularity = communities(distance, int(cfg.neighbors), int(cfg.seed), 1.)
            records = group[['global_unit_id', 'animal', 'session', 'recording']].copy()
            records['encoder'], records['cluster'] = name, label
            clusters.append(records)
            for k, seed in itertools.product([10, 15, 25], [0, 1, 2]):
                alternative, _ = communities(distance, k, seed, 1.)
                stability.append(dict(encoder=name, session=session, k=k, seed=seed,
                    ari=adjusted_rand_score(label, alternative), n_clusters=len(np.unique(alternative))))
            sua = group.decoder_label.eq('sua').to_numpy()
            sua_labels, _ = communities(distance[np.ix_(sua, sua)], int(cfg.neighbors), int(cfg.seed), 1.)
            summaries.append(dict(encoder=name, session=session, n_units=n, n_clusters=len(np.unique(label)),
                silhouette=silhouette_score(distance, label, metric='precomputed'), modularity=modularity,
                sua_recluster_ari=adjusted_rand_score(label[sua], sua_labels)))
            centered = normalize(values) - normalize(values).mean(0)
            kernels[name] = centered @ centered.T
            distances[name], labels[name] = distance, label
        old_check.append(dict(session=session, ari_to_saved_historical=adjusted_rand_score(group.cluster, labels['historical'])))
        upper = np.triu_indices(n, 1)
        for first, second in itertools.combinations(MODELS, 2):
            d1, d2, g1, g2 = distances[first], distances[second], kernels[first], kernels[second]
            ranks1, ranks2 = stats.rankdata(d1[upper]), stats.rankdata(d2[upper])
            rankmat = np.zeros_like(d2)
            rankmat[upper] = ranks2
            rankmat += rankmat.T
            def corr(a, b):
                return float(np.corrcoef(a, b)[0, 1])
            spearman = corr(ranks1, ranks2)
            cka = float(np.sum(g1 * g2) / np.sqrt(np.sum(g1*g1)*np.sum(g2*g2)))
            neighbors = []
            for d in (d1, d2):
                temporary = d.copy()
                np.fill_diagonal(temporary, np.inf)
                neighbors.append(np.argsort(temporary, axis=1)[:, :10])
            overlap = float(np.mean([len(set(a) & set(b))/10 for a, b in zip(*neighbors)]))
            ari = adjusted_rand_score(labels[first], labels[second])
            null_cka, null_dist, null_ari = [], [], []
            for _ in range(int(cfg.permutations)):
                permutation = rng.permutation(n)
                null_cka.append(np.sum(g1*g2[np.ix_(permutation, permutation)]) / np.sqrt(np.sum(g1*g1)*np.sum(g2*g2)))
                null_dist.append(corr(ranks1, rankmat[np.ix_(permutation, permutation)][upper]))
                null_ari.append(adjusted_rand_score(labels[first], labels[second][permutation]))
            record = dict(session=session, first=first, second=second, n_units=n,
                distance_spearman=spearman, linear_cka=cka, knn10_overlap=overlap,
                knn10_chance=10/(n-1), cluster_ari=ari,
                cka_null_mean=float(np.mean(null_cka)))
            for metric, value, null in [('distance', spearman, null_dist), ('cka', cka, null_cka), ('ari', ari, null_ari)]:
                record[f'{metric}_p'] = (1+np.sum(np.asarray(null) >= value))/(1+len(null))
            comparisons.append(record)
            contingency = pd.crosstab(labels[first], labels[second])
            contingency.to_csv(output / f'overlap_{session[:10]}_{first}_{second}.csv')
    pd.concat(clusters, ignore_index=True).to_csv(output / 'clusters.csv', index=False)
    pd.DataFrame(stability).to_csv(output / 'cluster_stability.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'cluster_summary.csv', index=False)
    pd.DataFrame(old_check).to_csv(output / 'historical_cluster_check.csv', index=False)
    comparison = pd.DataFrame(comparisons)
    for metric in ['distance', 'cka', 'ari']:
        comparison[f'{metric}_q'] = bh(comparison[f'{metric}_p'])
    comparison.to_csv(output / 'geometry.csv', index=False)
    print('Clustering outputs ready; fitting small identity probes', flush=True)
    folds, predictions, scores = [], [], []
    features = dict(matrices, physiology=baseline(unit_frame))
    tasks = [('session', np.ones(len(unit_frame), bool), unit_frame.session.to_numpy(str)),
             ('D1_D2_Dec10', (unit_frame.session.str.startswith('2024-12-10') & unit_frame.label.isin(['D1','D2'])).to_numpy(), unit_frame.label.to_numpy(str))]
    for task, keep, y in tasks:
        for name, values in features.items():
            rows, records, score = probe(values[keep], y[keep], name, task, unit_frame.loc[keep].reset_index(drop=True), int(cfg.seed))
            folds.extend(rows)
            predictions.append(records)
            scores.append(score)
    pd.DataFrame(folds).to_csv(output / 'probe_folds.csv', index=False)
    pd.concat(predictions).to_csv(output / 'probe_predictions.csv', index=False)
    pd.DataFrame(scores).to_csv(output / 'probe_scores.csv', index=False)
    plot(comparison, pd.DataFrame(scores), output)
    partition_plot(output)
    pca_physiology(unit_frame, matrices).to_csv(output / 'pca_physiology.csv', index=False)
    json_write(output / 'method.json', dict(n_units=len(unit_frame), models=MODELS,
        clustering='Per-session cosine kNN union graph, Louvain k15/resolution1/seed0; labels and GLM not used',
        comparison='Same original recorded units across encoders; dimension differs 48 versus128',
        historical_confound='Historical exports included noise units in the attention population, whereas new exports exclude noise; historical/new differences also change training data and dimensions. Passive-pretrained versus fine-tuned shares preparation and input population.',
        probe='Fixed C=1 balanced logistic regression; 5 identical stratified unit folds; train-only imputation and standardization',
        scope='Exploratory encoder-trained recordings; probes evaluate held-out units only',
        null='199 within-session unit-identity permutations; FDR separately per metric across nine comparisons',
        historical_cluster_check=old_check))
    print('Geometry and probes complete', flush=True)


def pca_physiology(frame, matrices):
    from sklearn.decomposition import PCA
    rows = []
    for encoder, values in matrices.items():
        for session, group in frame.groupby('session'):
            model = PCA(n_components=5)
            scores = model.fit_transform(normalize(values[group.index]))
            for axis in range(5):
                record = dict(encoder=encoder, session=session, axis=axis+1,
                              variance_fraction=model.explained_variance_ratio_[axis])
                for feature in ['analysis_firing_rate_hz', 'analysis_isi_cv', 'peak_to_valley_ms', 'half_width_ms', 'depth']:
                    y = pd.to_numeric(group[feature], errors='coerce').to_numpy()
                    keep = np.isfinite(y)
                    record[feature+'_spearman'] = stats.spearmanr(scores[keep,axis], y[keep]).statistic
                rows.append(record)
    return pd.DataFrame(rows)


def plot(comparison, scores, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout='constrained')
        comparisons = list(itertools.combinations(MODELS, 2))
        sessions = sorted(comparison.session.unique())
        labels = ['Old / passive', 'Old / fine-tuned', 'Passive / fine-tuned']
        for ax, metric, title in zip(axes, ['distance_spearman','knn10_overlap','cluster_ari'], ['Distance correlation','Neighbor overlap','Cluster agreement (ARI)']):
            values = np.array([[comparison[(comparison.session==session)&(comparison['first']==a)&(comparison.second==b)][metric].iloc[0] for a,b in comparisons] for session in sessions])
            ax.imshow(values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            for (row, col), value in np.ndenumerate(values):
                ax.text(col,row,f'{value:.2f}',ha='center',va='center',color='white' if value>.6 else '#1E293B')
            ax.set(xticks=range(3),xticklabels=labels,yticks=range(3),yticklabels=[s[5:10] for s in sessions],title=title)
            ax.tick_params(axis='x',rotation=35)
        fig.savefig(output/'geometry.png',dpi=200)
        plt.close(fig)
        fig, axes = plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
        for ax, task, title in zip(axes, ['session','D1_D2_Dec10'], ['Session decoding','D1 / D2 · December 10']):
            subset=scores[scores.task.eq(task)]
            ax.bar(range(len(subset)),subset.balanced_accuracy,color=['#64748B','#3B82F6','#0D9488','#D97706'])
            ax.axhline(subset.chance.iloc[0],color='#94A3B8',linestyle='--')
            ax.set(xticks=range(len(subset)),xticklabels=['Old','Passive','Fine-tuned','Physiology'],ylim=(0,1),title=title,ylabel='Balanced accuracy')
        fig.savefig(output/'probes.png',dpi=200)
        plt.close(fig)


def partition_plot(output):
    """Display membership changes on one fixed reference layout per session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment
    from src.embeddings.suite import SESSION_COLORS
    clusters = read_table(output / 'clusters.csv')
    sessions = sorted(clusters.session.unique())
    reference = resolve('results/experiments/umap/finetuned_active_745502_9048859/per_session')
    assignments = []
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3,3,figsize=(11,10),layout='constrained')
        for col, session in enumerate(sessions):
            coords = read_table(reference / f'745502_{session}_coordinates.csv').set_index('global_unit_id')
            fine = clusters[(clusters.session==session)&(clusters.encoder=='finetuned')].set_index('global_unit_id').cluster
            for row, encoder in enumerate(MODELS):
                source = clusters[(clusters.session==session)&(clusters.encoder==encoder)].set_index('global_unit_id').cluster
                source = source.loc[fine.index]
                table = pd.crosstab(source, fine)
                a,b = linear_sum_assignment(-table.to_numpy())
                colors = {int(table.index[i]): int(table.columns[j]) for i,j in zip(a,b)}
                extra = max(fine)+1
                for label in sorted(source.unique()):
                    if label not in colors:
                        colors[label] = extra
                        extra += 1
                    assignments.append(dict(session=session,encoder=encoder,cluster=int(label),display_color_index=colors[label]))
                xy = coords.loc[fine.index, ['umap_1','umap_2']].to_numpy()
                ax=axes[row,col]
                ax.scatter(xy[:,0],xy[:,1],s=12,c=[SESSION_COLORS[colors[label]%len(SESSION_COLORS)] for label in source],linewidths=0)
                ax.set_aspect('equal',adjustable='datalim')
                ax.set_axis_off()
                ax.set_title(f'{NAMES[encoder]} · {session[5:10]}',fontsize=10)
        fig.suptitle('Cluster membership · fixed fine-tuned UMAP coordinates',fontsize=13)
        fig.savefig(output/'cluster_partitions.png',dpi=200)
        plt.close(fig)
    pd.DataFrame(assignments).to_csv(output/'partition_display_colors.csv',index=False)
