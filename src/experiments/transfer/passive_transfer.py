"""Reference-group correspondence and descriptive passive behavior analyses.

Target cell labels never choose the score, radius, preprocessing, or candidates.
The reference was selected in prior exploratory analysis; this is retrospective
label-withheld matching, not independent validation on newly collected animals.
"""
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize, SplineTransformer

from src.experiments.clustering.clustering import bh
from src.embeddings.io import read_table
from src.data.recording import Recording, resolve_recording
from src.data.io import resolve
from src.common.plotting import STYLE


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_reference(values, frame, reference, reference_cluster, mode, quantile):
    """Fixed source cluster centroids; source-only leave-one-out acceptance radius."""
    transformed = normalize(values)
    if mode == 'session_centered':
        centered = transformed.copy()
        for _, group in frame.groupby('key', sort=True):
            centered[group.index] -= centered[group.index].mean(0)
        transformed = normalize(centered)
    ref = frame.key.eq(reference).to_numpy()
    member = ref & frame.cluster.eq(reference_cluster).to_numpy()
    cluster_ids = sorted(frame.loc[ref, 'cluster'].unique())
    centroids = normalize(np.stack([transformed[ref & frame.cluster.eq(c).to_numpy()].mean(0) for c in cluster_ids]))
    anchor = cluster_ids.index(reference_cluster)
    distances = cdist(transformed, centroids, metric='cosine')
    inside = transformed[member]
    loo = normalize(inside.sum(0)[None, :] - inside)
    source_distances = 1 - np.sum(inside * loo, axis=1)
    radius = max(float(np.quantile(source_distances, quantile)), 1e-8)
    positive = distances[:, anchor]
    negative = np.min(np.delete(distances, anchor, axis=1), axis=1)
    score = np.minimum(radius - positive, negative - positive) / radius
    return score, positive, negative, dict(reference_cluster=int(reference_cluster),
        radius=radius, n_reference_members=int(member.sum()), source_cluster_ids=[int(c) for c in cluster_ids],
        rule='candidate iff within source 95% leave-one-out radius and nearer reference centroid than every other source-cluster centroid')


def rate_reference(frame, reference, cluster, quantile):
    rate = np.log1p(frame.firing_rate_hz.to_numpy(float))
    source = frame.key.eq(reference)
    inside = rate[source & frame.cluster.eq(cluster)]
    centers = frame.loc[source].assign(log_rate=rate[source]).groupby('cluster').log_rate.mean()
    anchor = centers.loc[cluster]
    loo = (inside.sum()-inside)/(len(inside)-1)
    radius = max(float(np.quantile(np.abs(inside-loo), quantile)), 1e-8)
    positive = np.abs(rate-anchor)
    negative = np.min(np.abs(rate[:,None]-centers.drop(cluster).to_numpy()[None,:]),axis=1)
    return np.minimum(radius-positive,negative-positive)/radius


def evaluate_scores(scored, cfg):
    """Within-session ranking, with target-session label counts preserved in nulls."""
    summaries, local = [], []
    rng = np.random.default_rng(int(cfg.seed))
    anchor_animal = str(cfg.reference).split('/')[0]
    for (encoder, method), f in scored.groupby(['encoder','method'],sort=True):
        for scope in ['Other sessions', 'Other animals', 'Other animals SUA']:
            part = f[f.key.ne(cfg.reference) & f.label.isin(['ChAT','D1','D2'])].copy()
            if scope.startswith('Other animals'):
                part = part[part.animal.ne(anchor_animal)]
            if scope.endswith('SUA'):
                part = part[part.decoder_label.eq('sua')]
            arrays = []
            for key, group in part.groupby('key',sort=True):
                y = group.label.eq('ChAT').to_numpy()
                score = group.score.to_numpy(float)
                if y.any() and (~y).any():
                    auc = roc_auc_score(y, score)
                    arrays.append((y, score))
                    local.append(dict(encoder=encoder,method=method,scope=scope,key=key,
                        n_chat=int(y.sum()),n_msn=int((~y).sum()),auc=auc,
                        chat_selected=int((y & group.candidate.to_numpy()).sum()),
                        msn_selected=int((~y & group.candidate.to_numpy()).sum())))
            if arrays:
                observed = float(np.mean([roc_auc_score(y,s) for y,s in arrays]))
                null = []
                # Mann-Whitney AUC calculated from ranks avoids repeated estimator overhead.
                ranks = [(y,stats.rankdata(s)) for y,s in arrays]
                for _ in range(int(cfg.permutations)):
                    aucs=[]
                    for y,r in ranks:
                        perm = rng.permutation(y)
                        n1,n0=int(y.sum()),int((~y).sum())
                        aucs.append((r[perm].sum()-n1*(n1+1)/2)/(n1*n0))
                    null.append(np.mean(aucs))
                p=(1+np.sum(np.asarray(null)>=observed))/(1+len(null))
            else:
                observed,p=np.nan,np.nan
            chat=part.label.eq('ChAT')
            msn=part.label.isin(['D1','D2'])
            summaries.append(dict(encoder=encoder,method=method,scope=scope,
                n_recordings=part.key.nunique(),n_mixed_label_recordings=len(arrays),
                n_chat=int(chat.sum()),n_msn=int(msn.sum()),
                chat_selected=int((chat & part.candidate).sum()),msn_selected=int((msn & part.candidate).sum()),
                chat_recall=float(part.loc[chat,'candidate'].mean()) if chat.any() else np.nan,
                msn_selected_fraction=float(part.loc[msn,'candidate'].mean()) if msn.any() else np.nan,
                mean_within_session_auc=observed,p_permutation=p))
    summary=pd.DataFrame(summaries)
    summary['q']=bh(summary.p_permutation)
    return summary,pd.DataFrame(local)


def matching(cfg, out):
    source=resolve(cfg.source)
    all_units=read_table(source/'units.csv')
    matrices={}
    with np.load(source/'pretrain_means.npz',allow_pickle=False) as z:
        ids=z['global_unit_id'].astype(str)
        matrices['pretrain']=z['embeddings'].astype(float)
    with np.load(resolve(cfg.finetune_cache)/'passive_means.npz',allow_pickle=False) as z:
        assert np.array_equal(ids,z['global_unit_id'].astype(str))
        matrices['finetune']=z['embeddings'].astype(float)
    scored,anchors=[],{}
    for encoder in ['pretrain','finetune']:
        f=all_units[all_units.encoder.eq(encoder)].set_index('global_unit_id').loc[ids].reset_index()
        f['animal']=f.animal.astype(str)
        ref=f[f.key.eq(cfg.reference)]
        # This reference was chosen in the preceding analysis. No target labels below influence scoring.
        anchor=int(ref.loc[ref.label.eq('ChAT')].cluster.value_counts().sort_index().idxmax())
        for method in ['raw_embedding','session_centered']:
            score,dpos,dneg,info=score_reference(matrices[encoder],f,str(cfg.reference),anchor,method,float(cfg.radius_quantile))
            part=f.copy()
            part['method'],part['score'],part['distance_reference'],part['distance_other']=method,score,dpos,dneg
            part['candidate']=score>0
            scored.append(part)
            anchors[f'{encoder}/{method}']=info
        part=f.copy()
        part['method']='rate_only'
        part['score']=rate_reference(f,str(cfg.reference),anchor,float(cfg.radius_quantile))
        part['candidate']=part.score>0
        scored.append(part)
    scored=pd.concat(scored,ignore_index=True)
    scored.to_csv(out/'all_scores.csv',index=False)
    summary,local=evaluate_scores(scored,cfg)
    summary.to_csv(out/'label_evaluation.csv',index=False)
    local.to_csv(out/'label_evaluation_by_session.csv',index=False)
    primary=scored[scored.method.eq('raw_embedding')]
    decision=primary.pivot(index='global_unit_id',columns='encoder',values='candidate')
    candidates=primary[primary.encoder.eq('finetune')].copy().set_index('global_unit_id')
    candidates['candidate_pretrain']=decision.pretrain
    candidates['candidate_finetune']=decision.finetune
    candidates['candidate_consensus']=decision.all(axis=1)
    candidates['candidate_either']=decision.any(axis=1)
    candidates=candidates.reset_index()
    candidates.to_csv(out/'candidates.csv',index=False)
    groups=primary.groupby(['encoder','key','cluster']).agg(n=('global_unit_id','size'),
        n_candidates=('candidate','sum'),median_score=('score','median'),
        n_chat=('label',lambda s:int(s.eq('ChAT').sum())),n_msn=('label',lambda s:int(s.isin(['D1','D2']).sum()))).reset_index()
    groups.to_csv(out/'target_cluster_matches.csv',index=False)
    inventory=candidates.groupby(['animal','session','key']).agg(n_units=('global_unit_id','size'),
        n_consensus=('candidate_consensus','sum'),n_pretrain=('candidate_pretrain','sum'),n_finetune=('candidate_finetune','sum'),
        n_chat=('label',lambda s:int(s.eq('ChAT').sum()))).reset_index()
    inventory.to_csv(out/'candidate_inventory.csv',index=False)
    features=['firing_rate_hz','isi_cv','frac_isi_lt_10ms','frac_isi_gt_500ms','waveform_duration_ms','half_width_ms','depth']
    physiology=[]
    for key,g in candidates.groupby('key'):
        for scope in ['All units','SUA']:
            part=g if scope=='All units' else g[g.decoder_label.eq('sua')]
            for feature in features:
                a=pd.to_numeric(part.loc[part.candidate_consensus,feature],errors='coerce').dropna().to_numpy()
                b=pd.to_numeric(part.loc[~part.candidate_consensus,feature],errors='coerce').dropna().to_numpy()
                effect=2*stats.mannwhitneyu(a,b).statistic/(len(a)*len(b))-1 if len(a) and len(b) else np.nan
                physiology.append(dict(key=key,scope=scope,feature=feature,n_candidates=len(a),n_rest=len(b),
                    candidate_median=float(np.median(a)) if len(a) else np.nan,
                    rest_median=float(np.median(b)) if len(b) else np.nan,rank_effect=effect))
    pd.DataFrame(physiology).to_csv(out/'candidate_physiology.csv',index=False)
    write_json(out/'reference.json',anchors)
    write_json(out/'matching_method.json',dict(reference=str(cfg.reference),
        candidate_rule='Positive minimum of radius margin and nearest-other-source-cluster margin; 95% source-only leave-one-out cosine radius',
        primary='Raw normalized encoder space, consensus of pretrained and fine-tuned decisions; no target labels or behavior choose candidates',
        sensitivities='Session-centering fixed in advance; source-only rate prototype baseline; these do not alter primary candidates',
        evaluation='ChAT versus D1/D2 labels only. Mean AUC gives each mixed-label target recording equal weight; label permutations stay within recordings. BH across all encoder/method/scope evaluations.',
        label_limits='Unknowns are unknown. Reference was selected post hoc in prior analysis, so held-out refers to matching decisions, not independent discovery. Whole source animal excluded in main transfer summary.',
        source_hashes={str(p):digest(p) for p in [source/'units.csv',source/'pretrain_means.npz',resolve(cfg.finetune_cache)/'passive_means.npz']}))
    print(summary.to_string(index=False),flush=True)
    return candidates


def binned_signal(recording, name, edges, origin):
    if not recording.has_signal(name):
        return np.full(len(edges)-1,np.nan),dict(status='unavailable',samples=0,units=None)
    t,x=recording.signal(name)
    t=t-origin
    x=np.asarray(x).reshape(-1)
    if len(t)!=len(x):
        raise ValueError(f"Expected one {name} value per timestamp")
    good=np.isfinite(t)&np.isfinite(x)
    t,x=t[good],x[good]
    assert np.all(np.diff(t)>=0)
    keep=(t>=edges[0])&(t<edges[-1])
    j=np.searchsorted(edges,t[keep],side='right')-1
    count=np.bincount(j,minlength=len(edges)-1)
    total=np.bincount(j,weights=x[keep],minlength=len(edges)-1)
    values=np.divide(total,count,out=np.full(len(count),np.nan),where=count>0)
    units=recording.signal_info(name).get('unit','unspecified')
    return values,dict(status='available' if keep.any() else 'outside_eligible_interval',samples=int(keep.sum()),
        units=str(units),valid_bin_fraction=float(np.isfinite(values).mean()))


def qr_basis(values):
    u,s,_=np.linalg.svd(values,full_matrices=False)
    return u[:,s>max(values.shape)*np.finfo(float).eps*max(s[0],1)]


def corr_columns(a,b):
    a=a-a.mean(0)
    b=b-b.mean(0)
    return np.sum(a*b,axis=0)/np.maximum(np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0),1e-12)


def control_pairs(group):
    """Without-replacement same-sort controls within a factor of two in mean rate."""
    source=group[group.candidate_consensus].sort_values('firing_rate_hz',ascending=False)
    available=set(group.index[~group.candidate_consensus])
    pairs=[]
    for index,row in source.iterrows():
        choices=group.loc[sorted(available)]
        choices=choices[choices.decoder_label.eq(row.decoder_label)]
        if not len(choices) or row.firing_rate_hz<=0: continue
        distance=np.abs(np.log(choices.firing_rate_hz.to_numpy()/row.firing_rate_hz))
        if distance.min()>np.log(2): continue
        other=choices.index[np.argmin(distance)]
        available.remove(other)
        pairs.append((index,other))
    return pairs


def behavior(cfg,out,candidates):
    audit,metrics,events,matched,curve_rows,population=[],[],[],[],[],[]
    cached=out/'behavior_cache'
    cached.mkdir(exist_ok=True)
    for number,(key,g) in enumerate(candidates.groupby('key',sort=True),1):
        g=g.reset_index(drop=True)
        session=Recording(resolve_recording(key,root=resolve(cfg.data_root)))
        try:
            print(f'Behavior {number}/79: {key}',flush=True)
            eligible=session.intervals('pre_stimulus')
            if not len(eligible):
                raise ValueError(f'No prestimulus intervals for {key}')
            origin=float(eligible[0,0])
            eligible=eligible-origin
            positions=pd.Index(session.units.global_unit_id).get_indexer(g.global_unit_id)
            assert (positions>=0).all()
            # Separate eligible intervals; invalidate any bins bridging excluded periods.
            step=float(cfg.bin_s)
            edges=np.arange(0,float(eligible[-1,1])+step*.01,step)
            centers=(edges[:-1]+edges[1:])/2
            valid=np.zeros(len(centers),bool)
            for lo,hi in eligible:
                valid|=(edges[:-1]>=lo)&(edges[1:]<=hi)
            counts=np.empty((len(centers),len(g)),float)
            spikes=[]
            for j,position in enumerate(positions):
                train=session.spikes(int(session.units.iloc[position].source_unit_row))-origin
                counts[:,j]=np.histogram(train,edges)[0]
                spikes.append(train)
            signals={}
            source=session.path
            for name in ['running','whisking']:
                signal,info=binned_signal(session,name,edges,origin)
                signals[name]=signal
                row=dict(key=key,animal=str(g.animal.iloc[0]),signal=name,eligible_seconds=float(valid.sum()*step),
                         source=str(source),n_units=len(g),n_candidates=int(g.candidate_consensus.sum()),**info)
                event_name=name+'_onsets'
                times=session.events(event_name)-origin if session.has_events(event_name) else np.array([])
                selected=[]
                low,high=map(float,cfg.onset_window_s)
                for event in times:
                    if any(event+low>=lo and event+high<=hi for lo,hi in eligible):
                        if not selected or event-selected[-1]>=high-low:
                            selected.append(float(event))
                signals[name+'_onsets']=np.asarray(selected)
                row['n_onsets_total']=len(times)
                row['n_onsets_eligible']=len(selected)
                audit.append(row)
            z=np.sqrt(counts+3/8)
            time_basis=SplineTransformer(n_knots=5,degree=3,include_bias=False).fit_transform(centers[:,None])
            drift=np.column_stack([np.ones(len(centers)),time_basis])
            drift_q=qr_basis(drift[valid])
            drift_residual=z[valid]-drift_q@(drift_q.T@z[valid])
            original={name:np.full(len(g),np.nan) for name in ['running','whisking']}
            conditional={name:np.full(len(g),np.nan) for name in ['running','whisking']}
            for name in ['running','whisking']:
                speed=signals[name]
                use=valid&np.isfinite(speed)
                if use.sum()<100 or np.std(speed[use])<1e-10: continue
                x=stats.rankdata(speed[use])[:,None]
                y=np.apply_along_axis(stats.rankdata,0,counts[use])
                original[name]=corr_columns(y,np.repeat(x,len(g),axis=1))
                cov=drift[use]
                other='whisking' if name=='running' else 'running'
                # Add the other movement trace only when sufficiently observed in this comparison.
                if np.isfinite(signals[other][use]).all() and np.std(signals[other][use])>1e-10:
                    cov=np.column_stack([cov,stats.rankdata(signals[other][use])])
                q=qr_basis(cov)
                ry=y-q@(q.T@y)
                rx=x-q@(q.T@x)
                conditional[name]=corr_columns(ry,np.repeat(rx,len(g),axis=1))
            # Population coupling uses disjoint candidate/control/reference sets.
            pairs=control_pairs(g)
            pair_indices=np.asarray(pairs,int).reshape(-1,2)
            for a,b in pairs:
                matched.append(dict(key=key,candidate=g.global_unit_id[a],control=g.global_unit_id[b],
                    candidate_rate=float(g.firing_rate_hz[a]),control_rate=float(g.firing_rate_hz[b]),sorting=g.decoder_label[a]))
            use=valid.copy()
            usable=[]
            for name in ['running','whisking']:
                if np.isfinite(signals[name][valid]).mean()>=.9 and np.nanstd(signals[name][valid])>1e-10:
                    use &= np.isfinite(signals[name])
                    usable.append(name)
            cov=drift[use]
            for name in usable:
                # Flexible movement adjustment; direction labels remain source-defined.
                ranks=stats.rankdata(signals[name][use])[:,None]/use.sum()
                cov=np.column_stack([cov,SplineTransformer(n_knots=4,degree=3,include_bias=False).fit_transform(ranks)])
            q=qr_basis(cov)
            residual=z[use]-q@(q.T@z[use])
            scaled=residual/np.maximum(residual.std(0),1e-12)
            reference=~g.candidate_consensus.to_numpy()
            if len(pair_indices): reference[pair_indices[:,1]]=False
            population_r=np.full(len(g),np.nan)
            if reference.sum()>=int(cfg.minimum_reference_units):
                population_mean=scaled[:,reference].mean(1)
                population_r=corr_columns(scaled,np.repeat(population_mean[:,None],len(g),axis=1))
                # For units contributing to reference, remove self before computing their statistic.
                for j in np.flatnonzero(reference):
                    other=(scaled[:,reference].sum(1)-scaled[:,j])/(reference.sum()-1)
                    population_r[j]=corr_columns(scaled[:,[j]],other[:,None])[0]
            for cluster,other_group in g[~g.candidate_consensus].groupby('cluster'):
                if len(other_group)<2 or not g.candidate_consensus.any(): continue
                a=scaled[:,g.candidate_consensus.to_numpy()].mean(1)
                b=scaled[:,other_group.index].mean(1)
                population.append(dict(key=key,other_cluster=int(cluster),n_candidates=int(g.candidate_consensus.sum()),
                    n_reference=len(other_group),adjusted_correlation=float(corr_columns(a[:,None],b[:,None])[0]),
                    available_behavior_controls=','.join(usable)))
            unit=g[['global_unit_id','animal','session','key','label','decoder_label','candidate_consensus','firing_rate_hz']].copy()
            for name in ['running','whisking']:
                unit[name+'_correlation']=original[name]
                unit[name+'_drift_adjusted_correlation']=conditional[name]
            unit['population_adjusted_correlation']=population_r
            # Event responses use all units and identical eligible source onsets.
            event_arrays={}
            for name in ['running','whisking']:
                starts=signals[name+'_onsets']
                if len(starts)<int(cfg.minimum_events): continue
                lag_edges=np.arange(float(cfg.onset_window_s[0]),float(cfg.onset_window_s[1])+float(cfg.onset_bin_s)*.01,float(cfg.onset_bin_s))
                lags=(lag_edges[:-1]+lag_edges[1:])/2
                rates=np.zeros((len(g),len(lags)))
                deltas=np.zeros(len(g))
                for j,train in enumerate(spikes):
                    for event in starts:
                        lo,hi=np.searchsorted(train,[event+lag_edges[0],event+lag_edges[-1]],side='left')
                        rates[j]+=np.histogram(train[lo:hi]-event,lag_edges)[0]
                    rates[j]/=len(starts)*float(cfg.onset_bin_s)
                base=(lags>=-2)&(lags<-.5)
                response=(lags>=0)&(lags<1)
                deltas=rates[:,response].mean(1)-rates[:,base].mean(1)
                normalized=(rates-rates[:,base].mean(1)[:,None])/np.maximum(g.firing_rate_hz.to_numpy()[:,None],1e-6)
                unit[name+'_onset_delta_hz']=deltas
                unit[name+'_onset_delta_normalized']=deltas/np.maximum(g.firing_rate_hz.to_numpy(),1e-6)
                unit[name+'_n_onsets']=len(starts)
                event_arrays[name+'_rate_hz']=rates
                event_arrays[name+'_normalized']=normalized
                event_arrays[name+'_lags']=lags
                for t in starts: events.append(dict(key=key,signal=name,onset_local_s=t,onset_source_s=t+origin))
                selections={'Candidate':np.flatnonzero(g.candidate_consensus),'Other':np.flatnonzero(~g.candidate_consensus)}
                if len(pair_indices):
                    selections.update({'Matched candidate':pair_indices[:,0],'Matched control':pair_indices[:,1]})
                for label,indices in selections.items():
                    if not len(indices): continue
                    for lag,val in zip(lags,np.median(normalized[indices],axis=0)):
                        curve_rows.append(dict(key=key,animal=str(g.animal.iloc[0]),signal=name,group=label,lag_s=lag,
                                               median_normalized_rate=val,n_units=len(indices),n_events=len(starts)))
            metrics.append(unit)
            np.savez_compressed(cached/(g.animal.iloc[0]+'_'+g.session.iloc[0]+'.npz'),
                                global_unit_id=g.global_unit_id.to_numpy(dtype=str),**event_arrays)
        finally:
            session.close()
    pd.DataFrame(audit).to_csv(out/'behavior_availability.csv',index=False)
    pd.concat(metrics,ignore_index=True).to_csv(out/'unit_behavior.csv',index=False)
    pd.DataFrame(matched).to_csv(out/'rate_matched_pairs.csv',index=False)
    pd.DataFrame(events).to_csv(out/'eligible_onsets.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(out/'session_onset_curves.csv',index=False)
    pd.DataFrame(population).to_csv(out/'population_cluster_correlations.csv',index=False)
    write_json(out/'behavior_method.json',dict(
        epochs='Only canonical HDF5 prestimulus intervals; source behavior timestamps shifted by the recorded origin, without inferred offsets',
        scores='Candidates fixed by embedding reference matching before reading behavior; target labels never choose candidates',
        correlations='0.5s spike counts and binned recorded speed, rank correlations with cubic time spline adjustment; other movement adjusted when fully available',
        onsets='Source annotated onsets, complete -2..3s eligible windows, >=5s apart, >=5 events per recording; response0..1s minus baseline-2..-0.5s; curves normalized by session mean rate',
        controls='Same-recording/same-sort controls within factor2 of rate, nearest lograte without replacement; unmatched candidates retained in unpaired descriptions',
        coupling='Square-root counts residualized for slow time and available movement splines; population excludes all candidates and matched controls; reference members use leave-one-out population',
        limitations='Same spikes used to derive embeddings and functional summaries; exploratory associations, not independent validation or synaptic connectivity. Empty behavior arrays are missing, never inactivity. Speed units reported from source metadata when present.'))


def summarize_and_plot(cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    summary=pd.read_csv(out/'label_evaluation.csv')
    unit=read_table(out/'unit_behavior.csv')
    pairs=pd.read_csv(out/'rate_matched_pairs.csv')
    anchor_animal=str(cfg.reference).split('/')[0]
    features=[name for name in unit if name.endswith(('_drift_adjusted_correlation','_onset_delta_normalized'))]+['population_adjusted_correlation']
    differences=[]
    indexed=unit.set_index('global_unit_id')
    quality=pd.read_csv(out/'recording_quality.csv')
    clean_keys=set(quality.loc[~quality.population_gap_flag,'key'])
    scopes={'All recordings':pairs, 'Gap-free recordings':pairs[pairs.key.isin(clean_keys)],
            'Gap-free SUA':pairs[pairs.key.isin(clean_keys)&pairs.sorting.eq('sua')]}
    for scope,selected in scopes.items():
        for key,group in selected.groupby('key'):
            if str(key).split('/')[0]==anchor_animal: continue
            for feature in features:
                a=indexed.loc[group.candidate,feature].to_numpy(float)
                b=indexed.loc[group.control,feature].to_numpy(float)
                good=np.isfinite(a)&np.isfinite(b)
                if not good.any(): continue
                differences.append(dict(scope=scope,key=key,animal=str(key).split('/')[0],feature=feature,
                    n_pairs=int(good.sum()),candidate_median=float(np.median(a[good])),control_median=float(np.median(b[good])),
                    median_paired_difference=float(np.median(a[good]-b[good]))))
    differences=pd.DataFrame(differences)
    differences.to_csv(out/'matched_behavior_by_session.csv',index=False)
    aggregate=[]
    rng=np.random.default_rng(int(cfg.seed))
    for (scope,feature),g in differences.groupby(['scope','feature']):
        # Animal is the aggregation level so repeat recordings do not count as independent mice.
        animal=g.groupby('animal').median_paired_difference.mean()
        x=animal.to_numpy()
        if len(x)>=3:
            bootstrap=x[rng.integers(len(x),size=(2000,len(x)))].mean(1)
            lo,hi=np.quantile(bootstrap,[.025,.975])
            p=stats.binomtest(int((x>0).sum()),int((x!=0).sum()),.5).pvalue if (x!=0).any() else 1.
        else: lo,hi,p=np.nan,np.nan,np.nan
        aggregate.append(dict(scope=scope,feature=feature,n_recordings=len(g),n_animals=len(x),n_pairs=int(g.n_pairs.sum()),
            mean_animal_difference=float(x.mean()),ci_low=lo,ci_high=hi,p_sign=p,
            n_animals_positive=int((x>0).sum())))
    aggregate=pd.DataFrame(aggregate)
    aggregate['q_sign']=aggregate.groupby('scope').p_sign.transform(bh)
    aggregate.to_csv(out/'matched_behavior_summary.csv',index=False)
    curves=pd.read_csv(out/'session_onset_curves.csv')
    curves=curves[curves.animal.astype(str).ne(anchor_animal)&curves.key.isin(clean_keys)]
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
        for ax,encoder in zip(axes,['pretrain','finetune']):
            d=summary[(summary.encoder==encoder)&(summary.scope=='Other animals')].set_index('method').loc[['raw_embedding','session_centered','rate_only']]
            ax.bar(range(3),d.mean_within_session_auc,color=['#0D9488','#64748B','#D97706'])
            ax.axhline(.5,ls='--',c='#94A3B8')
            ax.set(xticks=range(3),xticklabels=['Original space','Session centered','Rate prototype'],ylim=(0,1),title=encoder.title(),ylabel='Within-recording ChAT / MSN AUC')
        fig.savefig(out/'label_transfer.png',dpi=220)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
        for ax,signal in zip(axes,['running','whisking']):
            for group,color in [('Matched candidate','#059669'),('Matched control','#64748B')]:
                d=curves[(curves.signal==signal)&(curves.group==group)]
                if not len(d): continue
                animal=d.groupby(['animal','lag_s']).median_normalized_rate.mean().unstack('lag_s')
                mean=animal.mean(0)
                sem=animal.std(0)/np.sqrt(len(animal))
                ax.plot(mean.index,mean,color=color,label=group.replace('Matched ',''))
                if len(animal)>1: ax.fill_between(mean.index,mean-sem,mean+sem,color=color,alpha=.16)
            ax.axvline(0,color='#94A3B8',lw=.8)
            ax.axhline(0,color='#CBD5E1',lw=.7)
            ax.set(title=signal.title()+' onset',xlabel='Time (s)',ylabel='Rate change / mean rate')
            handles,_=ax.get_legend_handles_labels()
            if handles: ax.legend()
            else: ax.text(.5,.5,'Insufficient recorded events',ha='center',va='center',transform=ax.transAxes)
        fig.savefig(out/'matched_onset_responses.png',dpi=220)
        plt.close(fig)
        physiology=pd.read_csv(out/'candidate_physiology.csv')
        fig,axes=plt.subplots(1,4,figsize=(12,3.8),layout='constrained')
        for ax,feature,label in zip(axes,['firing_rate_hz','isi_cv','waveform_duration_ms','frac_isi_gt_500ms'],['Rate (Hz)','ISI CV','Waveform duration (ms)','Long-ISI fraction']):
            d=physiology[(physiology.scope=='All units')&(physiology.feature==feature)&(physiology.n_candidates>0)]
            d=d[~d.key.str.startswith(anchor_animal+'/')&d.key.isin(clean_keys)]
            ax.scatter(d.rest_median,d.candidate_median,s=18,c='#059669',alpha=.7)
            lo=min(d.rest_median.min(),d.candidate_median.min())
            hi=max(d.rest_median.max(),d.candidate_median.max())
            ax.plot([lo,hi],[lo,hi],ls='--',c='#CBD5E1')
            ax.set(xlabel='Other units',ylabel='Candidates',title=label)
        fig.savefig(out/'candidate_physiology.png',dpi=220)
        plt.close(fig)
        labels={'population_adjusted_correlation':'Population coupling',
                'running_drift_adjusted_correlation':'Running correlation',
                'whisking_drift_adjusted_correlation':'Whisking correlation'}
        fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
        for ax,(feature,label) in zip(axes,labels.items()):
            for y,(scope,color) in enumerate([('Gap-free recordings','#059669'),('Gap-free SUA','#2563EB')]):
                row=aggregate[(aggregate.scope==scope)&(aggregate.feature==feature)].iloc[0]
                ax.errorbar(row.mean_animal_difference,y,
                            xerr=[[row.mean_animal_difference-row.ci_low],[row.ci_high-row.mean_animal_difference]],
                            fmt='o',c=color,capsize=3)
            ax.axvline(0,c='#94A3B8',ls='--')
            ax.set(yticks=[0,1],yticklabels=['All units','Single units'],ylim=(-.5,1.5),title=label,
                   xlabel='Candidate − control')
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.savefig(out/'matched_behavior_effects.png',dpi=220)
        plt.close(fig)
    availability=pd.read_csv(out/'behavior_availability.csv')
    write_json(out/'summary.json',dict(n_units=len(unit),n_candidates=int(unit.candidate_consensus.sum()),
        n_recordings=unit.key.nunique(),n_recordings_with_candidates=unit[unit.candidate_consensus].key.nunique(),
        other_animal_candidates=int(unit[unit.animal.ne(anchor_animal)].candidate_consensus.sum()),
        n_gap_flagged_recordings=int(quality.population_gap_flag.sum()),
        behavior_inventory=availability.groupby(['signal','status']).size().reset_index(name='recordings').to_dict('records'),
        matched_behavior=aggregate.to_dict('records')))
    print(aggregate.to_string(index=False),flush=True)


def audit(cfg,out,candidates):
    """Reconcile identities, source clocks, event counts and prepared physiology."""
    from src.data.features import spike_features
    assert candidates.global_unit_id.is_unique
    assert np.array_equal(candidates.candidate_consensus,candidates.candidate_pretrain&candidates.candidate_finetune)
    pairs=pd.read_csv(out/'rate_matched_pairs.csv')
    indexed=candidates.set_index('global_unit_id')
    assert pairs.candidate.is_unique and pairs.control.is_unique
    assert not set(pairs.candidate)&set(pairs.control)
    assert indexed.loc[pairs.candidate,'candidate_consensus'].all()
    assert not indexed.loc[pairs.control,'candidate_consensus'].any()
    assert np.array_equal(indexed.loc[pairs.candidate,'key'],indexed.loc[pairs.control,'key'])
    assert np.array_equal(indexed.loc[pairs.candidate,'decoder_label'],indexed.loc[pairs.control,'decoder_label'])
    assert np.abs(np.log(pairs.candidate_rate/pairs.control_rate)).max()<=np.log(2)+1e-10
    events=pd.read_csv(out/'eligible_onsets.csv')
    quality=[]
    n_checked=0
    event_checks=0
    for key,g in candidates.groupby('key',sort=True):
        recording=Recording(resolve_recording(key,root=resolve(cfg.data_root)))
        try:
            eligible=recording.intervals('pre_stimulus')
            if not len(eligible):
                raise ValueError(f'No prestimulus intervals for {key}')
            origin=float(eligible[0,0])
            eligible=eligible-origin
            positions=pd.Index(recording.units.global_unit_id).get_indexer(g.global_unit_id)
            assert (positions>=0).all()
            edges=np.arange(0,float(eligible[-1,1])+float(cfg.bin_s)*.01,float(cfg.bin_s))
            pooled=np.zeros(len(edges)-1,dtype=np.int64)
            with np.load(out/'behavior_cache'/(str(g.animal.iloc[0])+'_'+g.session.iloc[0]+'.npz'),allow_pickle=False) as cached:
                assert np.array_equal(cached['global_unit_id'],g.global_unit_id.to_numpy(str))
                for local,(position,(_,row)) in enumerate(zip(positions,g.iterrows())):
                    train=recording.spikes(int(recording.units.iloc[position].source_unit_row))-origin
                    assert np.isfinite(train).all() and np.all(np.diff(train)>=0)
                    actual=spike_features(train,eligible)
                    for feature in ['firing_rate_hz','isi_cv','frac_isi_gt_500ms','frac_isi_lt_10ms']:
                        assert np.isclose(actual[feature],row[feature],rtol=1e-6,atol=1e-9,equal_nan=True),(key,feature)
                    pooled+=np.histogram(train,edges)[0]
                    n_checked+=1
                    if local<2:
                        for signal in ['running','whisking']:
                            if signal+'_rate_hz' not in cached: continue
                            e=events[(events.key==key)&(events.signal==signal)]
                            starts=e.onset_local_s.to_numpy()
                            assert np.allclose(e.onset_source_s-starts,origin)
                            assert np.all(np.diff(starts)>=5-1e-8)
                            assert all(any(t-2>=lo and t+3<=hi for lo,hi in eligible) for t in starts)
                            lag_edges=np.arange(-2,3.001,float(cfg.onset_bin_s))
                            independent=np.diff(np.searchsorted(train,starts[:,None]+lag_edges[None,:],side='left'),axis=1).mean(0)/float(cfg.onset_bin_s)
                            assert np.allclose(independent,cached[signal+'_rate_hz'][local],atol=1e-8)
                            event_checks+=1
            max_gap=0.
            for lo,hi in eligible:
                valid=(edges[:-1]>=lo)&(edges[1:]<=hi)
                part=pooled[valid]
                nonzero=np.flatnonzero(part)
                if len(nonzero)>1:
                    max_gap=max(max_gap,float((np.diff(nonzero)-1).max()*float(cfg.bin_s)))
            quality.append(dict(key=key,animal=str(g.animal.iloc[0]),n_units=len(g),
                eligible_seconds=float(np.diff(eligible,axis=1).sum()),
                origin_s=float(origin),
                maximum_internal_population_silence_s=max_gap,population_gap_flag=max_gap>=10,
                n_candidates=int(g.candidate_consensus.sum())))
        finally: recording.close()
    quality=pd.DataFrame(quality)
    quality.to_csv(out/'recording_quality.csv',index=False)
    write_json(out/'integrity_checks.json',dict(status='passed',unique_unit_ids=len(candidates),
        physiology_and_monotonic_spikes_checked=n_checked,independent_onset_curves_checked=event_checks,
        matched_pairs_checked=len(pairs),gap_flagged_recordings=int(quality.population_gap_flag.sum()),
        gap_rule='>=10 seconds of internal silence across every exported unit in a recording; maximum half-second-bin gap. No clock correction inferred.',
        checks=['Consensus intersection reconstructed','Controls disjoint, same recording/sorting, within factor2 of rate',
                'All unit rates and ISI features independently recomputed from eligible spikes',
                'Event curves recounted with searchsorted; eligible windows and source-clock offsets checked']))
    print('Audit passed. Gap-flagged recordings:',quality[quality.population_gap_flag].key.tolist(),flush=True)


def report(cfg,out,candidates):
    """Animal-block uncertainty, consensus label audit, and a self-contained gallery."""
    import zipfile
    anchor_animal=str(cfg.reference).split('/')[0]
    local=pd.read_csv(out/'label_evaluation_by_session.csv')
    rng=np.random.default_rng(int(cfg.seed))
    uncertainty=[]
    for encoder in ['pretrain','finetune']:
        data=local[(local.encoder==encoder)&local.scope.eq('Other animals')]
        pivot=data.pivot(index='key',columns='method',values='auc').dropna()
        animals=np.asarray([k.split('/')[0] for k in pivot.index])
        unique=np.unique(animals)
        draws=rng.integers(len(unique),size=(3000,len(unique)))
        for method in ['raw_embedding','session_centered','rate_only']:
            values=pivot[method].to_numpy()
            baseline=pivot.rate_only.to_numpy()
            estimates=[]
            differences=[]
            for draw in draws:
                ids=np.concatenate([np.flatnonzero(animals==unique[j]) for j in draw])
                estimates.append(values[ids].mean())
                differences.append((values[ids]-baseline[ids]).mean())
            low,high=np.quantile(estimates,[.025,.975])
            dlo,dhi=np.quantile(differences,[.025,.975])
            uncertainty.append(dict(encoder=encoder,method=method,n_recordings=len(pivot),n_animals=len(unique),
                auc=float(values.mean()),auc_ci_low=low,auc_ci_high=high,
                delta_vs_rate=float((values-baseline).mean()),delta_ci_low=dlo,delta_ci_high=dhi))
    uncertainty=pd.DataFrame(uncertainty)
    uncertainty.to_csv(out/'label_transfer_uncertainty.csv',index=False)
    # Additional simple diagnostics: directions and missing-ISI fill use source only.
    # These compare a monotonic physiological axis with the nonmonotonic prototype baseline.
    reference=candidates[candidates.key.eq(cfg.reference)]
    source_group=reference.cluster.eq(json.loads((out/'reference.json').read_text())['finetune/raw_embedding']['reference_cluster'])
    baseline_rows=[]
    target=candidates[candidates.animal.ne(anchor_animal)&candidates.label.isin(['ChAT','D1','D2'])]
    baseline_methods={}
    for feature,title in [('firing_rate_hz','Rate ranking'),('isi_cv','ISI CV ranking')]:
        sign=float(np.sign(reference.loc[source_group,feature].median()-reference.loc[~source_group,feature].median()))
        fill=float(reference[feature].median())
        baseline_methods[feature]=dict(title=title,direction=sign,source_fill=fill,
                                     target_missing=int(target[feature].isna().sum()))
        for key,g in target.groupby('key'):
            y=g.label.eq('ChAT')
            if y.nunique()!=2: continue
            baseline_rows.append(dict(method=title,key=key,auc=roc_auc_score(y,sign*g[feature].fillna(fill))))
    for encoder,title in [('pretrain','Pretrained embedding'),('finetune','Fine-tuned embedding')]:
        for row in local[(local.encoder==encoder)&local.scope.eq('Other animals')&local.method.eq('raw_embedding')].itertuples():
            baseline_rows.append(dict(method=title,key=row.key,auc=row.auc))
    baseline_local=pd.DataFrame(baseline_rows)
    baseline_local.to_csv(out/'simple_baselines_by_session.csv',index=False)
    pivot=baseline_local.pivot(index='key',columns='method',values='auc').dropna()
    animals=np.asarray([key.split('/')[0] for key in pivot.index])
    unique=np.unique(animals)
    indexes=[np.concatenate([np.flatnonzero(animals==unique[j]) for j in draw])
             for draw in rng.integers(len(unique),size=(3000,len(unique)))]
    baseline_summary=[]
    for method in pivot:
        values=pivot[method].to_numpy()
        low,high=np.quantile([values[ix].mean() for ix in indexes],[.025,.975])
        baseline_summary.append(dict(method=method,auc=float(values.mean()),ci_low=low,ci_high=high,
                                     n_recordings=len(pivot),n_animals=len(unique)))
    baseline_summary=pd.DataFrame(baseline_summary)
    baseline_summary.to_csv(out/'simple_baselines.csv',index=False)
    contrasts=[]
    for encoder in ['Pretrained embedding','Fine-tuned embedding']:
        for comparator in ['Rate ranking','ISI CV ranking']:
            delta=(pivot[encoder]-pivot[comparator]).to_numpy()
            low,high=np.quantile([delta[ix].mean() for ix in indexes],[.025,.975])
            contrasts.append(dict(encoder=encoder,comparator=comparator,mean_auc_difference=float(delta.mean()),ci_low=low,ci_high=high))
    pd.DataFrame(contrasts).to_csv(out/'simple_baseline_contrasts.csv',index=False)
    write_json(out/'simple_baseline_method.json',dict(methods=baseline_methods,
        rationale='Added after initial prototype comparison to distinguish rate-value similarity from monotonic rate/ISI ranking; diagnostic, not a new selection rule',
        missing='Missing target ISI CV is filled using the entire source-recording median; no target labels or target fitting',
        uncertainty='3000 animal-block bootstrap samples, retaining all recordings of each resampled animal'))
    import matplotlib.pyplot as plt
    with plt.rc_context(STYLE):
        order=['Pretrained embedding','Fine-tuned embedding','Rate ranking','ISI CV ranking']
        fig,ax=plt.subplots(figsize=(7.4,3.6),layout='constrained')
        for y,method in enumerate(order):
            row=baseline_summary.set_index('method').loc[method]
            ax.errorbar(row.auc,y,xerr=[[row.auc-row.ci_low],[row.ci_high-row.auc]],fmt='o',
                        c='#059669' if y<2 else '#64748B',capsize=3)
        ax.axvline(.5,c='#CBD5E1',ls='--')
        ax.set(yticks=range(4),yticklabels=order,ylim=(3.5,-.5),xlim=(.25,1.02),
               xlabel='Within-recording ChAT / MSN AUC',title='Other animals')
        fig.savefig(out/'label_baselines.png',dpi=220)
        plt.close(fig)
    inventory=[]
    for scope,part in [('Source recording',candidates[candidates.key.eq(cfg.reference)]),
                       ('Other animals',candidates[candidates.animal.ne(anchor_animal)]),
                       ('Other animals SUA',candidates[candidates.animal.ne(anchor_animal)&candidates.decoder_label.eq('sua')])]:
        for label,g in part.groupby('label'):
            inventory.append(dict(scope=scope,label=label,n_total=len(g),n_selected=int(g.candidate_consensus.sum()),
                                  n_recordings=g.key.nunique(),n_selected_recordings=g[g.candidate_consensus].key.nunique()))
    inventory=pd.DataFrame(inventory)
    inventory.to_csv(out/'consensus_label_inventory.csv',index=False)
    physiology=pd.read_csv(out/'candidate_physiology.csv')
    quality=pd.read_csv(out/'recording_quality.csv')
    clean=set(quality.loc[~quality.population_gap_flag,'key'])
    physiology=physiology[physiology.key.isin(clean)&~physiology.key.str.startswith(anchor_animal+'/')]
    phys=[]
    for (scope,feature),g in physiology.groupby(['scope','feature']):
        g=g[(g.n_candidates>0)&(g.n_rest>0)].dropna(subset=['rank_effect'])
        phys.append(dict(scope=scope,feature=feature,n_recordings=len(g),
                         median_recording_candidate_value=float(g.candidate_median.median()),
                         median_recording_other_value=float(g.rest_median.median()),
                         n_recordings_candidate_higher=int((g.rank_effect>0).sum())))
    pd.DataFrame(phys).to_csv(out/'physiology_summary.csv',index=False)
    effects=pd.read_csv(out/'matched_behavior_summary.csv')
    availability=pd.read_csv(out/'behavior_availability.csv')
    availability['gap_free']=availability.key.isin(clean)
    coverage=availability.groupby(['signal','gap_free']).agg(recordings=('key','size'),
                traces_present=('status',lambda s:int(s.eq('available').sum())),
                sufficient_onsets=('n_onsets_eligible',lambda s:int((s>=int(cfg.minimum_events)).sum()))).reset_index()
    coverage.to_csv(out/'behavior_coverage_summary.csv',index=False)
    retained=candidates[candidates.key.isin(clean)&candidates.animal.ne(anchor_animal)]
    picked=retained[retained.candidate_consensus]
    full=json.loads((out/'summary.json').read_text())
    full.update(gap_free_other_animal_candidates=len(picked),gap_free_other_animal_recordings=picked.key.nunique(),
                gap_free_other_animals=picked.animal.nunique(),gap_free_other_animal_sua_candidates=int(picked.decoder_label.eq('sua').sum()),
                label_uncertainty=uncertainty.to_dict('records'),consensus_labels=inventory.to_dict('records'),
                simple_baselines=baseline_summary.to_dict('records'),simple_baseline_contrasts=contrasts)
    write_json(out/'summary.json',full)
    (out/'index.html').write_text('''<!doctype html><html><head><meta charset="utf-8"><title>Passive reference-group transfer</title>
<style>body{font:16px system-ui;color:#243247;background:#f6f8fb;margin:0;padding:32px}main{max-width:1100px;margin:auto}section{background:white;padding:24px;margin:20px 0;border-radius:14px}img{width:100%}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#047857}p{line-height:1.6}</style></head><body><main>
<h1>Passive reference-group transfer</h1>
<section><h2>Scope and interpretation</h2>
<p>9,219 units, 79 passive recordings. The May 15, 2023 reference from animal 655572 was selected in earlier exploratory work. Each encoder's independently clustered source group is matched by cosine distance in the existing common embedding space. Candidates must be nearer its centroid than every other source-cluster centroid and within its source-only 95th-percentile leave-one-out radius. The behavioral cohort is the intersection of pretrained and fine-tuned decisions. Target labels and behavior do not choose matches. Unknown labels never count as confirmed negatives.</p>
<p>All target animals were already included in self-supervised pretraining. “Other animals” excludes the source animal from label evaluation and behavioral aggregation, but is not new-animal encoder validation. The source was chosen post hoc, and these are retrospective exploratory results. The two encoders are closely related, so consensus is a stability criterion, not independent replication.</p>
<p>This analysis covers passive-to-passive correspondence and passive behavior. It does not establish passive-to-active correspondence or assign verified ChAT identities.</p></section>
<section><h2>Label transfer</h2><img src="label_baselines.png"><img src="label_transfer.png">
<p>AUC is calculated within each recording with both ChAT and D1/D2 labels, then averaged across recordings. Bootstrap intervals below resample whole animals. Session centering is a sensitivity analysis; it does not change the main candidates. The rate-only comparison uses the same source-centroid/radius logic.</p>'''+uncertainty.round(3).to_html(index=False,border=0)+
'''<p>Simple monotonic rate and ISI baselines were added as diagnostics after the prototype comparison. Their direction and missing-value handling use only the source recording. A rate prototype scores proximity to the source group's rate range; a monotonic rate ranking scores higher rate continuously. Those are different baselines. Intervals resample six animals and are necessarily imprecise.</p>'''+baseline_summary.round(3).to_html(index=False,border=0)+pd.DataFrame(contrasts).round(3).to_html(index=False,border=0)+
'''<p>Consensus selection counts:</p>'''+inventory.to_html(index=False,border=0)+
'''<p>The selected set contains labeled MSNs and other types. Its full-population ChAT precision cannot be inferred from the sparse, selectively obtained labels.</p></section>
<section><h2>Physiology</h2><img src="candidate_physiology.png"><p>Each point is one recording outside the source animal, excluding recordings flagged for a long population-wide gap. The diagonal indicates equal candidate and other-unit medians. Firing rate and ISI are derived from the same spike trains that produced the embeddings; they characterize the group but do not independently validate cell type.</p></section>
<section><h2>Movement and population activity</h2><img src="matched_behavior_effects.png"><img src="matched_onset_responses.png">
<p>Controls come from the same recording and sorting category, are matched without replacement, and have mean rates within a factor of two. Behavioral comparisons exclude the source animal. Session median paired differences are averaged within animal; intervals resample animals. Five feature comparisons are BH-adjusted separately within each scope. The onset curves average session-wise median normalized responses within animal, then across animals; shading is one standard error across animals. Onset comparisons use 0–1 s versus −2 to −0.5 s. At least five annotated, nonoverlapping events are required. Missing whisking arrays indicate missing data, not inactivity.</p>
<p>Running/whisking correlations control a slow time spline and the other movement when fully observed. Population coupling controls slow drift and available movement splines, using a reference population disjoint from all candidates and their matched controls. It is a cofluctuation measure, not evidence of synaptic connections. Only measured behavior can be controlled. Embedding selection and behavioral measures use the same recording, so these are functional descriptions rather than independent predictive validation.</p>'''+effects.round(4).to_html(index=False,border=0)+
'''</section><section><h2>Data quality and saved outputs</h2><p>During audit, some source files showed long population-wide spike gaps inside their annotated prestimulus interval. The quality sensitivity excludes entire recordings with at least 10 s of internal silence across every exported unit; no timing correction is inferred. Full-data results remain available. This criterion was added after inspecting the first results.</p>'''+quality[quality.population_gap_flag][['key','maximum_internal_population_silence_s','n_candidates']].to_html(index=False,border=0)+coverage.to_html(index=False,border=0)+
'''<p><a href="candidates.csv">All unit assignments and physiology</a> · <a href="unit_behavior.csv">Per-unit functional properties</a> · <a href="target_cluster_matches.csv">Target cluster correspondence</a> · <a href="population_cluster_correlations.csv">Candidate-to-cluster coupling</a> · <a href="behavior_availability.csv">Behavior availability</a> · <a href="integrity_checks.json">Validation</a> · <a href="figures_and_summary_tables.zip">Figures and summary tables</a></p></section></main></body></html>''')
    with zipfile.ZipFile(out/'figures_and_summary_tables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.suffix in ['.png','.html','.json','.yaml'] or p.name in ['candidates.csv','unit_behavior.csv',
                    'target_cluster_matches.csv','behavior_availability.csv','label_evaluation.csv','label_evaluation_by_session.csv',
                    'label_transfer_uncertainty.csv','consensus_label_inventory.csv','candidate_inventory.csv',
                    'simple_baselines.csv','simple_baselines_by_session.csv','simple_baseline_contrasts.csv',
                    'matched_behavior_summary.csv','matched_behavior_by_session.csv','behavior_coverage_summary.csv',
                    'physiology_summary.csv','recording_quality.csv','rate_matched_pairs.csv','population_cluster_correlations.csv']:
                archive.write(p,p.name)
    print('Report:',out/'index.html',flush=True)


def run(cfg):
    out=resolve(cfg.output)
    out.mkdir(parents=True,exist_ok=True)
    saved=out/'config.yaml'
    if saved.exists():
        previous=OmegaConf.load(saved)
        previous.stage=cfg.stage
        if OmegaConf.to_container(previous,resolve=True)!=OmegaConf.to_container(cfg,resolve=True):
            raise ValueError('Existing output has different configuration')
    OmegaConf.save(cfg,saved,resolve=True)
    if cfg.stage in ['all','match']:
        candidates=matching(cfg,out)
    else:
        candidates=read_table(out/'candidates.csv')
    if cfg.stage in ['all','behavior']:
        behavior(cfg,out,candidates)
        audit(cfg,out,candidates)
        summarize_and_plot(cfg,out)
        report(cfg,out,candidates)
    if cfg.stage=='audit':
        audit(cfg,out,candidates)
        summarize_and_plot(cfg,out)
        report(cfg,out,candidates)
    if cfg.stage=='plot':
        summarize_and_plot(cfg,out)
        report(cfg,out,candidates)
    print(f'Completed passive analysis stage {cfg.stage}: {out}',flush=True)
