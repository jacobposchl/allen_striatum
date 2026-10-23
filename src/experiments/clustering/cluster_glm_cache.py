"""Explore saved cluster membership and final GLMs without refitting either."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, f as f_distribution

from src.experiments.clustering.cluster_glm import GROUPS, bh


def association(frame, feature, adjustment):
    numeric, categorical = [], []
    d = frame.copy()
    if adjustment != 'Raw':
        d['log_rate'] = np.log1p(d.analysis_firing_rate_hz)
        d['log_rate_squared'] = d.log_rate**2
        numeric = ['log_rate', 'log_rate_squared', 'available_fraction', 'depth']
        categorical = ['decoder_label', 'label', 'default_qc']
    if adjustment == 'Spike features':
        numeric += ['analysis_isi_cv', 'analysis_frac_isi_lt_2ms', 'half_width_ms', 'peak_to_valley_ms']
    d = d[['cluster', feature] + numeric + categorical].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3 or d.cluster.nunique() < 2:
        return dict(n=len(d), clusters=d.cluster.nunique(), partial_r2=np.nan, p=np.nan)
    y = rankdata(d[feature])
    base = np.ones((len(d), 1))
    if numeric:
        x = d[numeric].to_numpy(float)
        x = (x-x.mean(0))/np.maximum(x.std(0), 1e-12)
        cat = pd.get_dummies(d[categorical].astype(str), drop_first=True, dtype=float)
        base = np.column_stack([base, x, cat])
    full = np.column_stack([base, pd.get_dummies(d.cluster, drop_first=True, dtype=float)])
    r0, r1 = np.linalg.matrix_rank(base), np.linalg.matrix_rank(full)
    df1, df2 = r1-r0, len(d)-r1
    rss0 = np.square(y-base@np.linalg.lstsq(base,y,rcond=None)[0]).sum()
    rss1 = np.square(y-full@np.linalg.lstsq(full,y,rcond=None)[0]).sum()
    if df1 <= 0 or df2 <= 0 or rss0 <= 1e-12:
        return dict(n=len(d), clusters=d.cluster.nunique(), partial_r2=np.nan, p=np.nan)
    effect = max(0., 1-rss1/rss0)
    statistic = max(0., (rss0-rss1)/df1)/max(rss1/df2,1e-12)
    return dict(n=len(d), clusters=d.cluster.nunique(), partial_r2=effect,
                p=float(f_distribution.sf(statistic,df1,df2)))


def run(cfg):
    from omegaconf import OmegaConf
    from src.experiments.clustering.plots import PALETTE, style
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    source, root, out = Path(cfg.clusters), Path(cfg.glm), Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out/'config.yaml')
    records = pd.read_csv(root/'recordings.csv')
    missing = records.missing_fields.fillna('').str.split(';').map(set)
    records['trials_available'] = missing.map(lambda x: 'intervals/trials' not in x)
    records['licking_available'] = missing.map(lambda x: not any('lick_time' in k for k in x))
    records['reward_events_available'] = missing.map(lambda x: not any('reward_delivery_time' in k for k in x))
    records['behavioral_data_available'] = records[['trials_available','licking_available','reward_events_available']].all(axis=1)
    records['glm_required_fields_available'] = records.missing_fields.fillna('').eq('')
    records['units_available'] = records.status.ne('missing_units')
    records['glm_cache_complete'] = records.status.eq('complete')
    records['key'] = records.animal.astype(str)+'_'+records.session+'_'+records.recording
    records.to_csv(root/'recording_availability.csv', index=False)
    records.to_csv(out/'recording_availability.csv', index=False)
    c = pd.read_csv(source/'units.csv')
    c = c.loc[c.cluster.ge(0)].copy()
    u = pd.read_csv(root/'unit_index.csv')
    assert c.global_unit_id.is_unique and u.global_unit_id.is_unique
    columns = ['global_unit_id','id','final_status','final_file','schema_variant','validation_folds']
    audit = c.merge(u[columns], on='global_unit_id', how='left', validate='one_to_one')
    audit = audit.merge(records[['key','behavioral_data_available','units_available','status','movement_available','anatomy_available']], on='key', how='left', validate='many_to_one')
    assert audit.loc[audit.id.notna(),'unit_id'].eq(audit.loc[audit.id.notna(),'id']).all()
    reliability = pd.read_csv(root/'reliability.csv')
    valid = reliability.loc[reliability.n_folds.eq(3) & reliability.all_converged.eq(True)].copy()
    full_ids = set(valid.loc[valid.group.eq('Full'),'global_unit_id'])
    audit['analysis_included'] = audit.final_status.eq('fit') & audit.global_unit_id.isin(full_ids)
    audit.to_csv(out/'cohort_audit.csv', index=False)
    d = audit.loc[audit.analysis_included].copy()
    gains = valid.pivot(index='global_unit_id', columns='group', values='heldout_gain')
    d = d.merge(gains, on='global_unit_id', validate='one_to_one')
    for feature in GROUPS:
        if feature not in d: d[feature] = np.nan
    d['cluster_display'] = d.cluster.map(lambda c: f'C{int(c)+1}')
    d.to_csv(out/'unit_features.csv', index=False)
    valid.merge(d[['global_unit_id','key','cluster']],on='global_unit_id',validate='many_to_one').to_csv(out/'reliability_by_cluster.csv',index=False)
    coverage = audit.groupby('key').agg(clustered=('unit_id','size'),final_fits=('final_status',lambda x:x.eq('fit').sum()),analyzed=('analysis_included','sum')).reset_index()
    coverage = coverage.merge(records[['key','status','behavioral_data_available']],on='key',validate='one_to_one')
    coverage.to_csv(out/'coverage.csv',index=False)
    tests, summaries, compositions, curve_rows = [], [], [], []
    style()
    def save(fig, path):
        fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
        fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
        plt.close(fig)
    for key, part in d.groupby('key',sort=True):
        directory=out/'recordings'/key; directory.mkdir(parents=True,exist_ok=True)
        title=f'{part.animal.iloc[0]} · {part.session.iloc[0][:10]}'
        scopes={'All':part,'SUA':part.loc[part.decoder_label.eq('sua')],
                'QC SUA':part.loc[part.decoder_label.eq('sua') & part.default_qc.eq(True)],
                'Coverage >75%':part.loc[part.available_fraction.gt(.75)]}
        for scope, subset in scopes.items():
            for feature in GROUPS:
                s=subset.dropna(subset=[feature])
                counts=s.cluster.value_counts()
                s=s.loc[s.cluster.isin(counts[counts.ge(cfg.minimum_cluster_units)].index)]
                for adjustment in ['Raw','Rate + quality','Spike features']:
                    tests.append(dict(key=key,scope=scope,feature=feature,adjustment=adjustment,
                                      **association(s,feature,adjustment)))
                ranks=s[feature].rank(pct=True)
                for cluster,p in s.groupby('cluster'):
                    rel=valid.loc[valid.group.eq(feature)&valid.global_unit_id.isin(p.global_unit_id)]
                    summaries.append(dict(key=key,scope=scope,cluster=int(cluster),feature=feature,n=len(p),
                        median=p[feature].median(),q25=p[feature].quantile(.25),q75=p[feature].quantile(.75),
                        mean_percentile=ranks.loc[p.index].mean(),consistent_fraction=rel.consistent_gain.mean()))
        curves={}
        for row in part.itertuples():
            with np.load(root/row.final_file) as z: curves[row.global_unit_id]={k:z[k] for k in z.files}
        for cluster,p in part.groupby('cluster'):
            compositions.append(dict(key=key,cluster=int(cluster),n=len(p),n_sua=p.decoder_label.eq('sua').sum(),
                n_qc_sua=(p.decoder_label.eq('sua')&p.default_qc.eq(True)).sum(),
                n_d1=p.label.eq('D1').sum(),n_d2=p.label.eq('D2').sum(),
                median_rate=p.analysis_firing_rate_hz.median(),median_isi_cv=p.analysis_isi_cv.median(),
                median_half_width_ms=p.half_width_ms.median(),median_depth=p.depth.median()))
        clusters=sorted(part.cluster.unique())
        fig,axes=plt.subplots(3,3,figsize=(11,8.3))
        rng=np.random.default_rng(0)
        for ax,feature in zip(axes.flat,GROUPS):
            for i,cluster in enumerate(clusters):
                y=part.loc[part.cluster.eq(cluster),feature].dropna().to_numpy()*100
                if len(y):
                    ax.scatter(i+rng.uniform(-.17,.17,len(y)),y,s=10,alpha=.5,color=PALETTE[int(cluster)%len(PALETTE)],linewidths=0)
                    ax.plot([i-.22,i+.22],[np.median(y)]*2,color='#273345',lw=2)
            ax.set_title(feature); ax.set_xticks(range(len(clusters)),[f'C{int(c)+1}' for c in clusters])
            ax.axhline(0,color='#b9c1ca',lw=.6); ax.set_ylabel('Held-out gain (%)')
            if part[feature].notna().sum()==0: ax.text(.5,.5,'Not recorded',ha='center',transform=ax.transAxes)
        fig.suptitle(title);fig.tight_layout();save(fig,directory/'feature_gains')
        example=next(iter(curves.values()));lags=example['lags_s']
        names=['Cue','Choice / outcome','Left − right choice','Reward − omission','Side × outcome','Left lick','Right lick']
        specs=[(name,lags,lambda z,i=i:z['kernels'][i],'Time (s)') for i,name in enumerate(names)]
        specs += [('Left reward probability',example['probability'],lambda z:z['probability_effect'][0],'Probability'),
                  ('Right reward probability',example['probability'],lambda z:z['probability_effect'][1],'Probability'),
                  ('History',np.arange(3),lambda z:z['history'],'Contrast')]
        if 'movement' in example: specs.append(('Movement',example['speed_cm_s'],lambda z:z['movement'],'Speed (cm/s)'))
        fig,axes=plt.subplots(3,4,figsize=(13,8))
        for ax,(name,x,getter,xlabel) in zip(axes.flat,specs):
            for cluster,p in part.groupby('cluster'):
                if len(p)<cfg.minimum_cluster_units: continue
                y=np.stack([getter(curves[uid]) for uid in p.global_unit_id])
                median=np.median(y,axis=0)
                ax.plot(x,median,color=PALETTE[int(cluster)%len(PALETTE)],label=f'C{int(cluster)+1} · {len(p)}',lw=1.8)
                for xx,mm,lo,hi in zip(x,median,np.quantile(y,.25,axis=0),np.quantile(y,.75,axis=0)):
                    curve_rows.append(dict(key=key,cluster=int(cluster),feature=name,x=xx,median=mm,q25=lo,q75=hi,n=len(p)))
            ax.set(title=name,xlabel=xlabel,ylabel='Log rate effect');ax.axhline(0,color='#b9c1ca',lw=.6)
            if name=='History':ax.set_xticks([0,1,2],['Response','Side','Outcome'],fontsize=8)
        for ax in list(axes.flat)[len(specs):]:ax.axis('off')
        handles,labels=axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].legend(handles,labels,loc='center',frameon=False)
        fig.suptitle(title+' · final-fit median kernels');fig.tight_layout();save(fig,directory/'response_curves')
    tests=pd.DataFrame(tests)
    tests['q']=tests.groupby(['scope','adjustment']).p.transform(lambda p:bh(p.to_numpy()))
    tests.to_csv(out/'associations.csv',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(out/'cluster_glm_summary.csv',index=False)
    composition=pd.DataFrame(compositions).merge(pd.read_csv(source/'cluster_stability.csv'),on=['key','cluster'],how='left',validate='one_to_one')
    composition.to_csv(out/'cluster_composition.csv',index=False)
    pd.DataFrame(curve_rows).to_csv(out/'response_curve_summary.csv',index=False)
    # A descriptive follow-up to the omnibus screen: opposite lick kernels.
    focus=d.loc[d.key.eq(cfg.focus_recording) & d.cluster.isin(cfg.focus_clusters)]
    sign_rows=[]
    if len(focus):
        fig,axes=plt.subplots(2,3,figsize=(10.5,6),sharex='col',sharey='col')
        focus_curves={}
        for row in focus.itertuples():
            path=root/row.final_file;directory=path.parent.parent
            info=json.loads((directory/'prepared/design_info.json').read_text())
            with np.load(path) as z:focus_curves[row.global_unit_id]={k:z[k] for k in z.files}
            with np.load(directory/'validation/units'/path.name) as z:coef=z['coefficients']
            with np.load(directory/'basis.npz') as z:
                lag=z['lags_s'];window=(lag>=-.1)&(lag<=.1)
                for feature in ['left_lick','right_lick']:
                    fold_curve=coef[:,info['slices'][feature]]@z[feature].T
                    fold_mean=fold_curve[:,window].mean(axis=1)
                    final=focus_curves[row.global_unit_id]
                    final_mean=final['kernels'][list(final['kernel_names']).index(feature),window].mean()
                    sign_rows.append(dict(global_unit_id=row.global_unit_id,key=row.key,cluster=row.cluster,
                        sorting=row.decoder_label,feature=feature,final_mean_log_effect=final_mean,
                        same_sign_all_folds=bool((np.sign(fold_mean)==np.sign(final_mean)).all())))
        for axes_row,scope in zip(axes,['All','SUA']):
            subset=focus if scope=='All' else focus.loc[focus.decoder_label.eq('sua')]
            for ax,index,label in zip(axes_row,[0,5,6],['Cue','Left lick','Right lick']):
                for cluster,part in subset.groupby('cluster'):
                    curves=[focus_curves[uid] for uid in part.global_unit_id]
                    lag=curves[0]['lags_s'];mask=np.ones(len(lag),bool) if index==0 else (lag>=-.4)&(lag<=.6)
                    median=np.median(np.stack([z['kernels'][index] for z in curves]),axis=0)
                    ax.plot(lag[mask],median[mask],label=f'C{int(cluster)+1} · {len(part)}',color=PALETTE[int(cluster)%len(PALETTE)],lw=2)
                ax.axhline(0,color='#b9c1ca',lw=.7);ax.axvline(0,color='#b9c1ca',lw=.7,ls=':')
                ax.set_title(f'{label} · {scope}');ax.set_xlabel('Time (s)')
                ax.legend(frameon=False,fontsize=8)
            axes_row[0].set_ylabel('Median log rate effect')
        fig.suptitle('March 31 · contrasting cluster responses');fig.tight_layout();save(fig,out/'march31_kernel_contrasts')
        pd.DataFrame(sign_rows).to_csv(out/'kernel_sign_consistency.csv',index=False)
    keys=sorted(d.key.unique())
    short={k:f'{k.split("_")[0]} · {k.split("_")[1]}' for k in keys}
    fig,axes=plt.subplots(1,2,figsize=(13,4.2),sharey=True)
    norm=Normalize(0,.5)
    for ax,scope in zip(axes,['All','SUA']):
        t=tests.loc[tests.scope.eq(scope)&tests.adjustment.eq('Rate + quality')]
        matrix=t.pivot(index='key',columns='feature',values='partial_r2').reindex(index=keys,columns=GROUPS)
        qs=t.pivot(index='key',columns='feature',values='q').reindex(index=keys,columns=GROUPS)
        cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#ededed')
        im=ax.imshow(matrix,cmap=cmap,norm=norm,aspect='auto')
        for i,j in zip(*np.where(qs.to_numpy()<.05)):ax.text(j,i,'•',ha='center',va='center',color='#e47128',fontsize=14)
        ax.set_xticks(range(len(GROUPS)),['Full','Cue','Choice /\noutcome','Side','Outcome','Licks','Movement','Probability','History'],rotation=45,ha='right')
        ax.set_yticks(range(len(keys)),[short[k] for k in keys]);ax.set_title(scope)
    fig.suptitle('Cluster–GLM association after rate and quality controls')
    fig.subplots_adjust(left=.16,right=.82,bottom=.25,top=.83,wspace=.12)
    fig.colorbar(im,cax=fig.add_axes([.85,.27,.015,.53]),label='Partial R² (ranked held-out gain)')
    fig.text(.16,.02,'Orange dot: exploratory q < 0.05 · gray: unavailable · cluster identities are recording-specific',fontsize=9)
    save(fig,out/'associations_overview')
    s=summary.loc[summary.scope.eq('All')]
    matrix=s.pivot(index=['key','cluster'],columns='feature',values='mean_percentile').reindex(columns=GROUPS)
    labels=[f'{short[key]} · C{cluster+1}' for key,cluster in matrix.index]
    fig,ax=plt.subplots(figsize=(10,max(5,.28*len(matrix))))
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#ededed')
    im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap=cmap)
    ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.set_xticks(range(len(GROUPS)),GROUPS,rotation=40,ha='right')
    ax.set_title('Functional profiles of existing clusters')
    fig.colorbar(im,ax=ax,label='Mean within-recording percentile of held-out gain',shrink=.7)
    fig.tight_layout();save(fig,out/'cluster_profiles')
    method=dict(join='Unique animal/session/recording/unit identity; Brainsets source unit IDs verified',
        primary='Converged final fit and three converged Full CV folds; each feature also requires three converged folds; no positive-gain filter',
        final_curves='Median of final full-data kernels across units; no fold averaging; descriptive quartiles saved',
        reliability='Held-out drop-group deviance gain, Full versus drift baseline; not coefficient confidence',
        comparisons='Within-recording nested OLS on ranked feature gains, minimum five units per cluster; BH across all recordings and features separately for each scope and adjustment',
        adjustments='Rate + quality: log1p(rate), squared log1p(rate), availability, depth, sorting, recorded label and QC. Spike features additionally controls ISI CV, fraction ISI <2ms, half-width and peak-to-valley.',
        cross_recording='Repeated cluster–function associations, not aligned cluster identities or independent animal-level replication',
        limitations=['Clusters and GLMs derive from the same recordings; exploratory unit-level tests assume independent units',
            'Four recordings have no movement covariate; December 18 lacks anatomical region labels',
            'SUA/QC subsets are sensitivity checks, not independent replication; cluster boundaries can be unstable',
            'Sparse cluster overlap on April 2 and April 9; cluster numbers and colors only identify within-recording groups',
            'Conditional task associations do not establish cell type, causality, reward prediction error or dopamine signals'],
        analyzed_units=len(d),recordings=len(keys),
        focus='Post-screen descriptive March 31 C1/C3/C6 kernel comparison; sign consistency checks mean lick kernel from -0.1 to +0.1 s across CV folds, not confidence intervals',
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),out/'config.yaml',source/'units.csv',root/'unit_index.csv',root/'reliability.csv',root/'schema.json']})
    (out/'method.json').write_text(json.dumps(method,indent=2)+'\n')
    page='<html><meta charset="utf-8"><title>Cluster–GLM comparison</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;color:#253247}img{max-width:100%}td,th{padding:6px;text-align:left}</style><h1>Existing clusters and GLM function</h1>'
    page+=f'<p>{len(d)} units across {len(keys)} recordings. Cluster identities are local to each recording. Exploratory associations; four recordings lack movement measurements.</p>'
    page+='<p><a href="recording_availability.csv">Recording availability</a> · <a href="associations.csv">Associations</a> · <a href="unit_features.csv">Unit features</a> · <a href="method.json">Methods</a></p>'
    page+='<img src="associations_overview.png"><img src="cluster_profiles.png">'
    if len(focus): page+='<h2>March 31 · descriptive kernel contrast</h2><img src="march31_kernel_contrasts.png">'
    for key in keys:page+=f'<h2>{short[key]}</h2><img src="recordings/{key}/feature_gains.png"><img src="recordings/{key}/response_curves.png">'
    (out/'index.html').write_text(page+'</html>')
    print(coverage.loc[coverage.analyzed.gt(0)].to_string(index=False))
    print(tests.loc[tests.scope.eq('All')&tests.adjustment.eq('Rate + quality')].sort_values('q').to_string(index=False))
