"""Per-recording kNN communities, geometry checks and descriptive physiology.

Only embeddings define communities. Feature associations are exploratory, and
neither community membership nor waveform/rate summaries identify cell type.
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_samples

from src.data.features import spike_features
from src.data.intervals import merge
from src.embeddings.windows import RecordingExport
from src.data.io import resolve
from src.data.recording import Recording, resolve_recording


FEATURES = ['analysis_firing_rate_hz', 'peak_to_valley_ms', 'half_width_ms',
            'analysis_isi_cv', 'analysis_frac_isi_lt_10ms',
            'analysis_frac_isi_gt_500ms', 'analysis_frac_isi_lt_2ms',
            'depth', 'available_fraction']


def bh(values):
    values = np.asarray(values, float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    result[order] = np.minimum(1, np.minimum.accumulate(
        (values[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return result


def communities(distance, k, seed, resolution):
    """Union kNN graph with a fixed recording-wide distance bandwidth."""
    n = len(distance)
    k = min(int(k), n - 1)
    d = distance.copy()
    np.fill_diagonal(d, np.inf)
    nearest = np.argsort(d, axis=1, kind='stable')[:, :k]
    scale = max(float(np.median(d[np.arange(n), nearest[:, -1]])), 1e-12)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in nearest[i]:
            graph.add_edge(i, int(j), weight=float(np.exp(-distance[i, j] / scale)))
    groups = nx.community.louvain_communities(graph, weight='weight', resolution=resolution, seed=seed)
    # Size ordering is deterministic and never uses external features or labels.
    groups = sorted(groups, key=lambda c: (-len(c), min(c)))
    labels = np.empty(n, int)
    for label, members in enumerate(groups):
        labels[list(members)] = label
    modularity = nx.community.modularity(graph, groups, weight='weight', resolution=resolution)
    return labels, float(modularity)


def extract_features(frame, cfg):
    animal, session, recording = [str(frame.iloc[0][k]) for k in ['animal', 'session', 'recording']]
    export = RecordingExport(cfg.export, f'{animal}/{session}/{recording}')
    intervals = merge(np.column_stack([export.starts_source, export.starts_source + export.duration]))
    if not len(intervals):
        raise ValueError(f'No scheduled windows for {export.key}')
    path = resolve_recording(f'{animal}/{session}/{recording}', root=resolve(cfg.data_root))
    source = Recording(path)
    if source.identity != export.source_identity:
        source.close()
        raise ValueError("Recording changed since embedding export; re-export before analyzing physiology")
    units = source.units
    rows = frame.source_unit_row.to_numpy(int)
    assert np.array_equal(units['id'].to_numpy()[rows].astype(str), frame.unit_id.to_numpy(str))
    names = ['peak_to_valley', 'half_width', 'peak_trough_ratio', 'snr', 'amplitude',
             'ccf_acronym', 'ccf_ap', 'ccf_dv', 'ccf_ml', 'rp_contamination',
             'isi_violations_ratio', 'presence_ratio', 'default_qc', 'decoder_label']
    for name in names:
        if name in units and units[name].ndim == 1:
            values = units[name].to_numpy()[rows]
            if values.dtype.kind in 'SO':
                values = [v.decode() if isinstance(v, bytes) else v for v in values]
            if name in ['decoder_label', 'default_qc'] and name in frame:
                assert np.array_equal(np.asarray(values).astype(str), frame[name].to_numpy(str))
            frame[name] = values
        elif name not in frame:
            frame[name] = np.nan
    features = []
    for row in rows:
        times = source.spikes(row)
        assert np.all(np.diff(times) >= 0)
        features.append(spike_features(times, intervals))
    source.close()
    for key in features[0]:
        frame['analysis_' + key] = [f[key] for f in features]
    frame['peak_to_valley_ms'] = pd.to_numeric(frame.peak_to_valley, errors='coerce') * 1000
    frame['half_width_ms'] = pd.to_numeric(frame.half_width, errors='coerce') * 1000
    frame['available_fraction'] = frame.n_windows / len(export.starts_source)
    assert frame.available_fraction.between(0, 1).all()
    return frame, dict(feature_schema='brainsets_h5_v2', recording_id=export.key,
                       recording_identity=export.source_identity, export_manifest_sha256=export.manifest_sha256,
                       n_windows=len(export.starts_source), intervals_source_s=intervals.tolist(),
                       waveform_source='Brainsets stored scalar metrics',
                       spike_feature_scope='All scheduled export windows, including silent windows; ISIs never span excluded gaps')



def association(frame, feature, scope):
    mask = np.isfinite(pd.to_numeric(frame[feature], errors='coerce'))
    if scope == 'SUA only':
        mask &= frame.decoder_label.eq('sua')
    f = frame.loc[mask]
    groups = [g[feature].to_numpy(float) for _, g in f.groupby('cluster') if len(g) >= 3]
    result = dict(feature=feature, scope=scope, n=sum(map(len, groups)), groups=len(groups),
                  p=np.nan, epsilon_squared=np.nan)
    if len(groups) >= 2 and len(np.unique(np.concatenate(groups))) > 1:
        h, p = stats.kruskal(*groups)
        result.update(p=float(p), epsilon_squared=float(max(0, (h - len(groups) + 1) /
                                                              (sum(map(len, groups)) - len(groups)))))
    return result


def adjusted_waveform_test(frame, feature):
    """Exploratory rank-outcome regression controlling rate, sorting, depth, coverage.

    This is an approximate F test, not a mechanistic or cell-type validation.
    """
    columns = [feature, 'analysis_firing_rate_hz', 'depth', 'available_fraction']
    good = np.isfinite(frame[columns].to_numpy(float)).all(axis=1)
    f = frame.loc[good]
    result = dict(feature=feature, n=len(f), p=np.nan, partial_r2=np.nan)
    if len(f) < 20 or f.cluster.nunique() < 2:
        return result
    rate = np.log1p(f.analysis_firing_rate_hz.to_numpy())
    nuisance = np.column_stack([np.ones(len(f)), rate, rate**2,
                                f.depth, f.available_fraction, f.decoder_label.eq('mua').astype(float)])
    full = np.column_stack([nuisance, pd.get_dummies(f.cluster, drop_first=True, dtype=float)])
    y = stats.rankdata(f[feature])
    r0, r1 = np.linalg.matrix_rank(nuisance), np.linalg.matrix_rank(full)
    sse0 = np.sum((y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0])**2)
    sse1 = np.sum((y - full @ np.linalg.lstsq(full, y, rcond=None)[0])**2)
    if r1 > r0 and len(y) > r1 and sse1 > 0 and sse0 > 0:
        value = max(0, sse0 - sse1) / (r1 - r0) / (sse1 / (len(y) - r1))
        result.update(p=float(stats.f.sf(value, r1 - r0, len(y) - r1)),
                      partial_r2=float(max(0, 1 - sse1 / sse0)))
    return result


def run(cfg):
    source, output = resolve(cfg.source), resolve(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.csv').exists():
        raise FileExistsError(f'Completed results already exist: {output}')
    if (output / 'config.yaml').exists():
        prior = OmegaConf.to_container(OmegaConf.load(output / 'config.yaml'), resolve=True)
        if prior != OmegaConf.to_container(cfg, resolve=True):
            raise ValueError('Partial output uses different settings; choose a new output directory')
    OmegaConf.save(cfg, output / 'config.yaml')
    source_info = json.loads((source / 'mean_embeddings.meta.json').read_text())
    frame = pd.read_csv(source / 'unit_metadata.csv', dtype={'animal': str, 'unit_id': str})
    with np.load(source / 'mean_embeddings.npz', allow_pickle=False) as saved:
        embeddings = saved['embeddings'].astype(float)
        assert np.array_equal(saved['global_unit_id'], frame.global_unit_id.to_numpy(str))
    assert not frame.global_unit_id.duplicated().any()
    assert np.isfinite(embeddings).all() and (np.linalg.norm(embeddings, axis=1) > 0).all()
    all_units, summaries, tests, adjusted, sensitivity, provenance, cluster_stability = [], [], [], [], [], [], []
    for (animal, session, recording), group in frame.groupby(['animal', 'session', 'recording'], sort=True):
        key = f'{animal}_{session}_{recording}'
        directory = output / 'recordings' / key
        directory.mkdir(parents=True, exist_ok=True)
        print(f'Processing {key}: {len(group)} exported units', flush=True)
        cached = directory / 'features.csv'
        # Recompute recording-dependent physiology from the selected HDF5 input;
        # historical features.csv files are outputs, never a data-loading shortcut.
        window_export = RecordingExport(cfg.export, f'{animal}/{session}/{recording}')
        if source_info.get('export_manifest_sha256') != window_export.manifest_sha256:
            raise ValueError('Saved mean embeddings come from a different export; regenerate them from the selected HDF5 export')
        f, audit = extract_features(group.copy(), cfg)
        f.to_csv(cached, index=False)
        (directory / 'feature_source.json').write_text(json.dumps(audit, indent=2))
        provenance.append(dict(key=key, **audit))
        coords = pd.read_csv(source / 'per_session' / f'{animal}_{session}_coordinates.csv')
        f = f.merge(coords[['global_unit_id', 'umap_1', 'umap_2']], on='global_unit_id', validate='one_to_one', how='left')
        assert f[['umap_1', 'umap_2']].notna().all().all()
        keep = f.decoder_label.isin(['sua', 'mua']).to_numpy()
        f['included'] = keep
        f['cluster'] = -1
        f['silhouette'] = np.nan
        f['sua_cluster'] = -1
        vectors = embeddings[group.index.to_numpy()][keep]
        distance = pairwise_distances(vectors, metric='cosine')
        np.fill_diagonal(distance, 0)
        labels, modularity = communities(distance, int(cfg.neighbors), int(cfg.seeds[0]), float(cfg.resolution))
        f.loc[keep, 'cluster'] = labels
        sil = silhouette_samples(distance, labels, metric='precomputed') if 1 < len(np.unique(labels)) < len(labels) else np.zeros(len(labels))
        f.loc[keep, 'silhouette'] = sil
        aris, alternatives = [], []
        for k in cfg.neighbor_sensitivity:
            for seed in cfg.seeds:
                alternative, _ = communities(distance, int(k), int(seed), float(cfg.resolution))
                ari = float(adjusted_rand_score(labels, alternative))
                sensitivity.append(dict(key=key, k=int(k), seed=int(seed), n_clusters=len(np.unique(alternative)), ari=ari))
                if int(k) != cfg.neighbors or int(seed) != cfg.seeds[0]:
                    aris.append(ari)
                    alternatives.append(alternative)
        sua = f.loc[keep, 'decoder_label'].eq('sua').to_numpy()
        sua_ari = np.nan
        if sua.sum() >= 20:
            sua_labels, _ = communities(distance[np.ix_(sua, sua)], int(cfg.neighbors), int(cfg.seeds[0]), float(cfg.resolution))
            f.loc[f.decoder_label.eq('sua'), 'sua_cluster'] = sua_labels
            sua_ari = float(adjusted_rand_score(labels[sua], sua_labels))
        for cluster in np.unique(labels):
            members = labels == cluster
            scores = [max(np.sum(members & (alt == c)) / np.sum(members | (alt == c))
                          for c in np.unique(alt)) for alt in alternatives]
            sua_members = members[sua]
            sua_jaccard = np.nan
            if sua.sum() >= 20 and sua_members.any():
                sua_jaccard = max(np.sum(sua_members & (sua_labels == c)) /
                                  np.sum(sua_members | (sua_labels == c)) for c in np.unique(sua_labels))
            cluster_stability.append(dict(key=key, cluster=int(cluster), median_jaccard=float(np.median(scores)),
                                          min_jaccard=float(min(scores)), sua_recluster_jaccard=float(sua_jaccard)))
        counts = np.bincount(labels)
        summary = dict(key=key, animal=animal, session=session, recording=recording,
                       n_exported=len(f), n_included=int(keep.sum()), n_noise=int(f.decoder_label.eq('noise').sum()),
                       n_sua=int(sua.sum()), n_clusters=len(counts), smallest_cluster=int(counts.min()),
                       silhouette=float(sil.mean()), stability_median_ari=float(np.median(aris)),
                       stability_min_ari=float(min(aris)), sua_recluster_ari=sua_ari, modularity=modularity)
        summary['separated_stable'] = bool(summary['silhouette'] >= cfg.separation_threshold and
                                         summary['stability_median_ari'] >= cfg.stability_threshold and
                                         counts.min() >= cfg.min_cluster_size and len(counts) >= 2)
        summaries.append(summary)
        inc = f.loc[keep]
        for feature in FEATURES:
            for scope in ['SUA + MUA', 'SUA only']:
                tests.append(dict(key=key, **association(inc, feature, scope)))
        for feature in ['peak_to_valley_ms', 'half_width_ms']:
            adjusted.append(dict(key=key, **adjusted_waveform_test(inc, feature)))
        f['key'] = key
        f.to_csv(directory / 'units.csv', index=False)
        np.savez_compressed(directory / 'geometry.npz', global_unit_id=f.loc[keep, 'global_unit_id'].to_numpy(str),
                            cosine_distance=distance, cluster=labels)
        all_units.append(f)
        print(f'  {len(counts)} groups; silhouette={sil.mean():.3f}; stability={np.median(aris):.3f}', flush=True)
    combined, summary = pd.concat(all_units, ignore_index=True), pd.DataFrame(summaries)
    test_table, adjusted_table = pd.DataFrame(tests), pd.DataFrame(adjusted)
    test_table['q'] = bh(test_table.p)
    adjusted_table['q'] = bh(adjusted_table.p)
    # Failures and missing features remain explicit rather than silently disappearing.
    combined.to_csv(output / 'units.csv', index=False)
    test_table.to_csv(output / 'feature_associations.csv', index=False)
    adjusted_table.to_csv(output / 'adjusted_waveform_associations.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(output / 'sensitivity.csv', index=False)
    pd.DataFrame(cluster_stability).to_csv(output / 'cluster_stability.csv', index=False)
    profiles = []
    for (key, cluster), g in combined.loc[combined.included].groupby(['key', 'cluster']):
        row = dict(key=key, cluster=int(cluster), n=len(g), n_sua=int(g.decoder_label.eq('sua').sum()),
                   n_mua=int(g.decoder_label.eq('mua').sum()), n_D1=int(g.label.eq('D1').sum()),
                   n_D2=int(g.label.eq('D2').sum()), silhouette=float(g.silhouette.mean()))
        for feature in FEATURES:
            row[feature + '_median'] = float(g[feature].median())
            row[feature + '_n'] = int(g[feature].notna().sum())
        profiles.append(row)
    pd.DataFrame(profiles).to_csv(output / 'cluster_profiles.csv', index=False)
    method = dict(model='Sep 12 active pretrained model', dimensions=int(embeddings.shape[1]),
                  geometry='Cosine distance between mean available-window embeddings, independently per recording',
                  graph='Symmetric union kNN, exp(-distance / median kth-neighbor distance), Louvain',
                  primary='SUA and MUA; noise and unknown sorting excluded; no availability filter',
                  clustering_uses='Embeddings only; no labels, spike summaries, waveforms or UMAP coordinates',
                  features='Rates and ISIs recomputed over scheduled export intervals, without crossing gaps; canonical waveform scalars',
                  display='Previously saved session UMAP, which includes noise units in its geometry; only included units displayed',
                  separation_rule=dict(silhouette=float(cfg.separation_threshold), median_ari=float(cfg.stability_threshold),
                                       minimum_cluster_size=int(cfg.min_cluster_size)),
                  rule_interpretation='Operational descriptive screen, not a statistical test for discrete cell types',
                  feature_tests='Kruskal-Wallis, groups with >=3 units; BH across all available recording x feature x scope tests',
                  waveform_adjustment='Rank-outcome OLS F test: cluster contribution beyond log1p(rate), its square, depth, coverage and sorting; BH across both waveform metrics and all recordings',
                  caveats=['Graph communities may partition a continuum.', 'Rates and ISIs derive from the encoder input and are not independent validation.',
                           'Waveform width is not a cell-type label.', 'MUA groups need not be individual neurons.',
                           'No cross-recording cluster identity is assumed.', 'Stability is across settings and seeds, not time or recordings.',
                           'ISI estimates with few spikes can be unreliable; eligible spike counts are saved.'],
                  source_provenance=json.loads((source / 'mean_embeddings.meta.json').read_text()), recordings=provenance)
    (output / 'method.json').write_text(json.dumps(method, indent=2))
    from src.experiments.clustering.plots import plot, bundle
    plot(combined, summary, test_table, output)
    summary.to_csv(output / 'summary.csv', index=False)
    bundle(output, summary)
    print(summary.to_string(index=False), flush=True)
