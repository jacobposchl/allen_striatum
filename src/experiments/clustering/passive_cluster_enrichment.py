"""Label-blind passive communities, descriptive ChAT concentration and label enrichment.

Unknown labels never serve as confirmed negatives. Enrichment inference is restricted
to recordings containing both ChAT and D1/D2 labels; cluster selection is never based
on labels. ChAT-dominant groups used in physiological displays are post-hoc summaries.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import fisher_exact, rankdata
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_score

from src.experiments.clustering.clustering import bh, communities
from src.embeddings.io import export_manifest, provenance, read_table
from src.data.io import resolve
from src.common.plotting import STYLE


FEATURES = ['firing_rate_hz', 'isi_cv', 'frac_isi_lt_10ms', 'frac_isi_gt_500ms',
            'waveform_duration_ms', 'half_width_ms', 'depth', 'available_fraction']


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def pretrain_means(source, output, frame):
    """Aggregate fixed-identity exports without iterating over metadata rows."""
    manifest_path, manifest = export_manifest(source)
    identity = provenance(manifest_path, manifest)
    path = output / 'pretrain_means.npz'
    if path.exists():
        saved = json.loads((output / 'pretrain_source.json').read_text())
        assert saved['export_manifest_sha256'] == identity['export_manifest_sha256']
        with np.load(path, allow_pickle=False) as f:
            assert np.array_equal(f['global_unit_id'], frame.global_unit_id.to_numpy(str))
            return f['embeddings'], saved
    ids = frame.global_unit_id.to_numpy(str)
    lookup = pd.Index(ids)
    sums, counts = None, np.zeros(len(frame), np.int64)
    seen = set()
    for batch_index, relative in enumerate(manifest['files']):
        path_batch = (manifest_path.parent / relative).resolve()
        assert path_batch.is_relative_to(manifest_path.parent.resolve()) and relative not in seen
        seen.add(relative)
        with np.load(path_batch, allow_pickle=False) as f:
            assert str(f['encoder_signature'].item()) == identity['encoder_signature']
            positions = lookup.get_indexer(f['global_unit_id'].astype(str))
            assert (positions >= 0).all() and len(np.unique(positions)) == len(positions)
            vectors = f['pooled']
            assert vectors.ndim == 2 and np.isfinite(vectors).all()
            if sums is None:
                sums = np.zeros((len(frame), vectors.shape[1]), np.float64)
            for name in ['animal', 'session', 'recording', 'label', 'decoder_label', 'source_unit_row']:
                assert np.array_equal(f[name].astype(str), frame.iloc[positions][name].to_numpy(str)), name
            sums[positions] += vectors
            counts[positions] += 1
        if batch_index % 1000 == 0:
            print(f'Pretrain averaging: {batch_index + 1}/{len(manifest["files"])} batches', flush=True)
    assert np.array_equal(counts, frame.n_windows.to_numpy(int))
    assert (counts > 0).all()
    values = (sums / counts[:, None]).astype(np.float32)
    np.savez_compressed(path, embeddings=values, global_unit_id=ids, n_windows=counts)
    frame.to_csv(output / 'pretrain_units.csv', index=False)
    dump(output / 'pretrain_source.json', identity)
    return values, identity


def label_summary(frame, labels, encoder, key, scope):
    f = frame.copy()
    f['cluster'] = labels
    chat = f.label.eq('ChAT')
    msn = f.label.isin(['D1', 'D2'])
    eligible = int(chat.sum()) > 0 and int(msn.sum()) > 0
    rows = []
    for cluster, group in f.groupby('cluster'):
        mask = f.cluster.eq(cluster)
        a, b = int((mask & chat).sum()), int((mask & msn).sum())
        c, d = int((~mask & chat).sum()), int((~mask & msn).sum())
        odds, p = fisher_exact([[a, b], [c, d]], alternative='greater') if eligible else (np.nan, np.nan)
        row = dict(encoder=encoder, key=key, scope=scope, cluster=int(cluster), n=len(group),
                   n_sua=int(group.decoder_label.eq('sua').sum()), n_chat=a, n_d1=int(group.label.eq('D1').sum()),
                   n_d2=int(group.label.eq('D2').sum()), n_other_label=int((~group.label.isin(['Unlabeled', 'ChAT', 'D1', 'D2'])).sum()),
                   n_unknown=int(group.label.eq('Unlabeled').sum()), n_chat_recording=int(chat.sum()),
                   n_msn_recording=int(msn.sum()), enrichment_eligible=eligible,
                   chat_recall=a / int(chat.sum()) if chat.any() else np.nan,
                   chat_fraction_of_known_chat_msn=a / (a + b) if a + b else np.nan,
                   odds_ratio=odds, p=p)
        for feature in FEATURES:
            row[feature + '_median'] = pd.to_numeric(group[feature], errors='coerce').median()
        rows.append(row)
    return rows


def run(cfg):
    import matplotlib
    matplotlib.use('Agg')
    output = resolve(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.yaml').exists():
        assert OmegaConf.to_container(OmegaConf.load(output / 'config.yaml')) == OmegaConf.to_container(cfg)
    OmegaConf.save(cfg, output / 'config.yaml')
    cache = resolve(cfg.finetune_cache)
    frame = read_table(cache / 'passive_units.csv')
    with np.load(cache / 'passive_means.npz', allow_pickle=False) as f:
        assert np.array_equal(f['global_unit_id'], frame.global_unit_id.to_numpy(str))
        ft = f['embeddings']
    assert not frame.global_unit_id.duplicated().any()
    assert frame.decoder_label.isin(['sua', 'mua']).all()
    pt, pt_source = pretrain_means(cfg.pretrain_export, output, frame)
    assert pt.shape == ft.shape and np.isfinite(pt).all() and np.isfinite(ft).all()
    frame['half_width_ms'] = pd.to_numeric(frame.half_width, errors='coerce') * 1000
    # The exported prestimulus windows start at zero and cover complete 10-second windows.
    # Per-recording manifest file counts are authoritative for coverage.
    manifest_path, manifest = export_manifest(cfg.pretrain_export)
    recording_windows = {}
    for relative in manifest['files']:
        parts = Path(relative).parts
        record_key = '/'.join(parts[:-1])
        recording_windows[record_key] = recording_windows.get(record_key, 0) + 1
    record_paths = frame.animal.astype(str) + '/' + frame.session.astype(str) + '/' + frame.recording.astype(str)
    assert set(record_paths).issubset(recording_windows)
    frame['available_fraction'] = frame.n_windows.to_numpy() / record_paths.map(recording_windows).to_numpy()
    assert frame.available_fraction.between(0, 1).all()
    frame['key'] = record_paths
    all_units, summaries, cluster_rows, sensitivities, physio, comparisons = [], [], [], [], [], []
    assigned = {}
    for encoder, matrix in [('pretrain', pt), ('finetune', ft)]:
        for key, group in frame.groupby('key', sort=True):
            print(f'{encoder}: {key} ({len(group)} units)', flush=True)
            index = group.index.to_numpy()
            distance = pairwise_distances(matrix[index], metric='cosine')
            distance = np.maximum(distance, 0)
            np.fill_diagonal(distance, 0)
            labels, mod = communities(distance, int(cfg.neighbors), int(cfg.seeds[0]), float(cfg.resolution))
            assigned[(encoder, key)] = labels
            alternatives = []
            for k in cfg.neighbor_sensitivity:
                for seed in cfg.seeds:
                    alt, _ = communities(distance, int(k), int(seed), float(cfg.resolution))
                    ari = adjusted_rand_score(labels, alt)
                    sensitivities.append(dict(encoder=encoder, key=key, k=int(k), seed=int(seed), ari=ari,
                                              n_clusters=int(len(np.unique(alt)))))
                    if k != cfg.neighbors or seed != cfg.seeds[0]:
                        alternatives.append(alt)
            f = group.copy()
            f['encoder'], f['cluster'] = encoder, labels
            f['sua_cluster'] = -1
            sua = f.decoder_label.eq('sua').to_numpy()
            cluster_rows.extend(label_summary(f, labels, encoder, key, 'All units'))
            cluster_rows.extend(label_summary(f[sua], labels[sua], encoder, key, 'SUA labels in all-unit clusters'))
            sua_ari = np.nan
            if sua.sum() >= int(cfg.min_sua_recluster):
                sua_labels, _ = communities(distance[np.ix_(sua, sua)], int(cfg.neighbors), int(cfg.seeds[0]), float(cfg.resolution))
                f.loc[sua, 'sua_cluster'] = sua_labels
                cluster_rows.extend(label_summary(f[sua], sua_labels, encoder, key, 'SUA reclustered'))
                sua_ari = adjusted_rand_score(labels[sua], sua_labels)
            sil = silhouette_score(distance, labels, metric='precomputed') if 1 < len(np.unique(labels)) < len(labels) else np.nan
            row = dict(encoder=encoder, key=key, animal=str(f.iloc[0].animal), session=str(f.iloc[0].session),
                       n_units=len(f), n_sua=int(sua.sum()), n_chat=int(f.label.eq('ChAT').sum()),
                       n_chat_sua=int((f.label.eq('ChAT') & f.decoder_label.eq('sua')).sum()),
                       n_msn=int(f.label.isin(['D1', 'D2']).sum()), n_clusters=len(np.unique(labels)), modularity=mod,
                       silhouette=sil, median_stability_ari=float(np.median([adjusted_rand_score(labels, x) for x in alternatives])),
                       sua_recluster_ari=sua_ari)
            summaries.append(row)
            if f.label.eq('ChAT').any():
                # Selection explicitly post-hoc; this is not evidence that clusters identify ChAT.
                chat_counts = f.loc[f.label.eq('ChAT')].cluster.value_counts()
                top = int(chat_counts.sort_index().idxmax())
                f['most_chat_cluster'] = f.cluster.eq(top)
                for scope, select in [('All units', np.ones(len(f), bool)), ('SUA', sua)]:
                    local = f.loc[select]
                    for feature in FEATURES:
                        a = pd.to_numeric(local.loc[local.cluster.eq(top), feature], errors='coerce').dropna().to_numpy()
                        b = pd.to_numeric(local.loc[~local.cluster.eq(top), feature], errors='coerce').dropna().to_numpy()
                        effect = np.nan
                        if len(a) and len(b):
                            ranks = rankdata(np.r_[a, b])
                            u = ranks[:len(a)].sum() - len(a)*(len(a)+1)/2
                            effect = 2*u/(len(a)*len(b))-1
                        physio.append(dict(encoder=encoder, key=key, scope=scope, feature=feature, cluster=top,
                                           n_chat_in_cluster=int(chat_counts[top]), n_chat_recording=int(chat_counts.sum()),
                                           n_cluster=len(a), n_rest=len(b), cluster_median=np.median(a) if len(a) else np.nan,
                                           rest_median=np.median(b) if len(b) else np.nan, rank_biserial=effect))
            all_units.append(f)
    for key, f in frame.groupby('key', sort=True):
        a, b = assigned[('pretrain', key)], assigned[('finetune', key)]
        for scope, mask in [('All units', np.ones(len(f), bool)), ('SUA', f.decoder_label.eq('sua').to_numpy())]:
            comparisons.append(dict(key=key, scope=scope, n=int(mask.sum()), n_chat=int(f.label.eq('ChAT').sum()),
                                    ari=adjusted_rand_score(a[mask], b[mask]) if mask.sum() > 1 else np.nan))
    units, summary, clusters = pd.concat(all_units, ignore_index=True), pd.DataFrame(summaries), pd.DataFrame(cluster_rows)
    # One correction family across both encoders, all recordings and all scopes.
    clusters['q'] = bh(clusters.p.to_numpy())
    for name, table in [('units', units), ('recordings', summary), ('clusters', clusters),
                        ('sensitivity', pd.DataFrame(sensitivities)), ('physiology_contrasts', pd.DataFrame(physio)),
                        ('pretrain_finetune_agreement', pd.DataFrame(comparisons))]:
        table.to_csv(output / f'{name}.csv', index=False)
    label_inventory = frame.groupby(['animal', 'session', 'recording', 'label', 'decoder_label']).size().rename('n').reset_index()
    label_inventory.to_csv(output / 'label_inventory.csv', index=False)
    source_ft = json.loads((cache / 'passive_source.json').read_text())
    method = {'sources': {'pretrain': pt_source, 'finetune': source_ft}, 'n_units': len(frame),
              'n_recordings': int(frame.key.nunique()), 'n_chat': int(frame.label.eq('ChAT').sum()),
              'n_chat_sua': int((frame.label.eq('ChAT') & frame.decoder_label.eq('sua')).sum()),
              'n_chat_recordings': int(frame.loc[frame.label.eq('ChAT'), 'key'].nunique()),
              'clustering': 'Original-dimensional cosine distance; union 15-nearest-neighbor graph; exponential weights; Louvain resolution 1, seed 0; independent per recording and encoder',
              'sensitivity': 'k 10,15,20 crossed with seeds 0,1,2; additionally recluster SUA when at least 20 available',
              'inference': 'One-sided Fisher ChAT-vs-D1/D2 enrichment, only within recordings containing both labels. Unknowns and other cell types never negative labels. BH across all eligible encoder/recording/cluster/scope tests.',
              'physiology': 'Post-hoc cluster with most ChAT labels vs remaining units of its recording; descriptive rank-biserial effects, no inferential tests or claimed cell-type assignment. Tie broken by cluster index.',
              'scope': 'Means of identical exported 10-second windows; same 9219 unit identities; no labels or GLM used in clustering. All available exports (SUA+MUA).',
              'limitations': ['Sparse labels and many single-label recordings limit enrichment inference.', 'Unknown units can be ChAT; cluster purity cannot be inferred.', 'Physiology may partly reflect firing properties used to train the encoder.', 'Recordings are not independent biological replicates when animals recur.', 'Label enrichment is exploratory; encoders saw these recordings during self-supervised pretraining.']}
    dump(output / 'method.json', method)
    dump(output / 'checks.json', {'finite_embeddings': True, 'unique_unit_ids': True, 'identical_pretrain_finetune_units': True,
                                 'identical_available_window_counts': True, 'export_encoder_signatures_checked': True,
                                 'unknown_units_excluded_from_fisher_tables': True, 'all_unit_cluster_counts_match': bool(clusters[clusters.scope.eq('All units')].n.sum() == 2*len(frame))})
    make_figures(output, summary, clusters, pd.DataFrame(physio), pd.DataFrame(comparisons))
    print(json.dumps({k: method[k] for k in ['n_units', 'n_recordings', 'n_chat', 'n_chat_sua', 'n_chat_recordings']}, indent=2), flush=True)
    return method


def short_key(key):
    parts = key.split('/')
    return parts[0] + ' · ' + re.search(r'\d{4}-\d{2}-\d{2}', parts[1]).group()


def make_figures(output, summary, clusters, physio, comparisons):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {'pretrain': '#6366F1', 'finetune': '#0D9488'}
    chat_keys = summary.loc[(summary.encoder.eq('pretrain')) & summary.n_chat.gt(0)].sort_values(['n_chat', 'key'], ascending=[False, True]).key.tolist()
    display_labels = [short_key(key) for key in chat_keys]
    duplicate = pd.Series(display_labels).duplicated(keep=False).to_numpy()
    for position in np.flatnonzero(duplicate):
        clock = re.search(r'_(\d{2})-(\d{2})-\d{2}$', chat_keys[position].split('/')[1])
        display_labels[position] += f' {clock.group(1)}:{clock.group(2)}'
    assert len(set(display_labels)) == len(display_labels)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.8), gridspec_kw={'width_ratios': [1.5, 1]}, sharey=True)
        for encoder, shift in [('pretrain', -.14), ('finetune', .14)]:
            local = clusters[(clusters.encoder.eq(encoder)) & clusters.scope.eq('All units')]
            for y, key in enumerate(chat_keys):
                row = local[local.key.eq(key)].sort_values(['n_chat', 'cluster'], ascending=[False, True]).iloc[0]
                axes[0].scatter(row.chat_recall, y+shift, color=colors[encoder], s=40)
                axes[0].plot([0, row.chat_recall], [y+shift, y+shift], color=colors[encoder], alpha=.25)
                axes[1].scatter(row.n/summary[(summary.encoder.eq(encoder)) & summary.key.eq(key)].iloc[0].n_units, y+shift, color=colors[encoder], s=40)
                axes[0].text(1.04, y+shift, f'{int(row.n_chat)}/{int(row.n_chat_recording)}', va='center', fontsize=8, color=colors[encoder])
        axes[0].set(yticks=range(len(chat_keys)), yticklabels=display_labels, xlim=(-.03, 1.2), xlabel='Fraction of labeled ChAT', title='Cluster containing the most ChAT labels')
        axes[0].invert_yaxis()
        axes[0].set_xticks([0,.25,.5,.75,1])
        axes[1].set(xlim=(0,1), xlabel='Fraction of all units', title='Size of that cluster')
        axes[1].legend(handles=[Line2D([], [], marker='o', linestyle='', color=colors[x], label=x.title()) for x in colors], loc='lower right')
        for ax in axes:
            ax.grid(axis='x', alpha=.15)
        fig.tight_layout()
        fig.savefig(output/'chat_concentration.png', dpi=220)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        all_agree = comparisons[comparisons.scope.eq('All units')]
        axes[0].hist(all_agree.ari, bins=np.linspace(-.05, 1, 16), color='#64748B', edgecolor='white')
        axes[0].set(xlabel='Pretrain–finetune cluster ARI', ylabel='Recordings', title='Group agreement')
        for i, metric in enumerate(['median_stability_ari', 'silhouette'], 1):
            pivot = summary.pivot(index='key', columns='encoder', values=metric)
            axes[i].scatter(pivot.pretrain, pivot.finetune, color='#94A3B8', s=16)
            keys = pivot.index.intersection(chat_keys)
            axes[i].scatter(pivot.loc[keys,'pretrain'],pivot.loc[keys,'finetune'],color='#059669',s=28,label='Has ChAT labels')
            lo,hi = float(np.nanmin(pivot.to_numpy())),float(np.nanmax(pivot.to_numpy()))
            axes[i].plot([lo,hi],[lo,hi],color='#CBD5E1',ls='--')
            axes[i].set(xlabel='Pretrain',ylabel='Fine-tune',title='Seed / k stability' if i==1 else 'Cluster separation')
        axes[2].legend(loc='upper left',fontsize=8)
        fig.tight_layout()
        fig.savefig(output/'cluster_reproducibility.png',dpi=220)
        plt.close(fig)
        features = ['firing_rate_hz','isi_cv','waveform_duration_ms','frac_isi_gt_500ms','depth']
        labels = ['Rate','ISI CV','Waveform duration','Long ISIs','Depth']
        fig, axes = plt.subplots(1,2,figsize=(10.5,6.8),sharey=True)
        for ax,encoder in zip(axes,['pretrain','finetune']):
            p = physio[(physio.encoder.eq(encoder)) & physio.scope.eq('All units')].pivot(index='key',columns='feature',values='rank_biserial').reindex(index=chat_keys,columns=features)
            im=ax.imshow(p.to_numpy(),vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
            ax.set(xticks=range(len(features)),xticklabels=labels,yticks=range(len(chat_keys)),yticklabels=display_labels,title=encoder.title())
            ax.tick_params(axis='x',rotation=40)
        fig.subplots_adjust(left=.30,right=.85,bottom=.22,top=.94,wspace=.12)
        colorbar_axis = fig.add_axes([.89,.35,.018,.45])
        fig.colorbar(im,cax=colorbar_axis,label='Rank effect: group vs rest')
        fig.savefig(output/'chat_cluster_physiology.png',dpi=220,bbox_inches='tight')
        plt.close(fig)
    figures=['chat_concentration.png','cluster_reproducibility.png','chat_cluster_physiology.png']
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Passive communities</title><body style="font:16px system-ui;max-width:1300px;margin:35px auto"><h1>Passive embedding communities</h1><p>Clusters were fit without labels. ChAT-containing groups are post-hoc descriptive selections, not inferred ChAT identities. Unknown labels never serve as negative examples.</p>'+''.join(f'<p><img style="max-width:100%" src="{x}"></p>' for x in figures)+'</body>')


if __name__ == '__main__':
    run(OmegaConf.load('src/experiments/clustering/configs/passive_cluster_enrichment.yaml'))
