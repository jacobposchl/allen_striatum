"""Observed task responses, held-out predictions, and physiology-matched contrasts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import sparse

from src.data.recording import Recording, resolve_recording

EVENT_LABELS = {'cue':'Cue','quiet_cue':'Cue after quiet','bout_start':'Licking onset',
                'bout_end':'Licking offset','choice_reward':'Rewarded choice',
                'choice_omission':'Unrewarded choice','bout_start_away':'Onset away from cue/choice'}
PRIMARY = ['cue','bout_start','bout_end']


def nearest_distance(times, references):
    if not len(references): return np.full(len(times),np.inf)
    refs=np.sort(references);at=np.searchsorted(refs,times)
    return np.minimum(abs(times-refs[np.clip(at,0,len(refs)-1)]),abs(times-refs[np.clip(at-1,0,len(refs)-1)]))


def events_from_recording(info, trials, cfg):
    origin=info['clock']['origin_s']
    with Recording(resolve_recording(info['recording_id'], root=cfg.get('recording_root'))) as recording:
        sides=[recording.events(f'{side}_lick')-origin for side in ['left','right']]
    # Detect bouts on the full stream so truncation cannot create false starts/ends.
    licks=np.unique(np.concatenate(sides))
    starts=np.r_[0,np.flatnonzero(np.diff(licks)>cfg.bout_gap_s)+1] if len(licks) else np.array([],dtype=int)
    ends=np.r_[starts[1:]-1,len(licks)-1] if len(licks) else np.array([],dtype=int)
    bouts=pd.DataFrame(dict(start=licks[starts],end=licks[ends],n_licks=ends-starts+1))
    bouts['duration']=bouts.end-bouts.start
    bouts['eligible_definition']=(bouts.n_licks>=cfg.bout_min_licks)&(bouts.duration>=cfg.bout_min_duration_s)
    # First/last observed bouts have unknown quiet intervals beyond recording bounds.
    bouts.loc[bouts.index.isin([0,len(bouts)-1]),'eligible_definition']=False
    t=trials.loc[trials.usable].copy();cue=t.goCue_start_time.to_numpy()
    chosen=t.loc[t.animal_response.isin([0,1])]
    bouts['cue_distance']=nearest_distance(bouts.start.to_numpy(),cue)
    bouts['choice_distance']=nearest_distance(bouts.start.to_numpy(),chosen.choice_time.to_numpy())
    selected=bouts.loc[bouts.eligible_definition]
    previous=np.searchsorted(licks,cue,side='left')-1
    quiet=((previous<0)|(cue-licks[previous.clip(0)]>cfg.bout_gap_s)) if len(licks) else np.ones(len(cue),bool)
    events=dict(cue=cue,quiet_cue=cue[quiet],bout_start=selected.start.to_numpy(),bout_end=selected.end.to_numpy(),
        choice_reward=chosen.loc[chosen.rewarded.eq(1),'reward_outcome_time'].to_numpy(),
        choice_omission=chosen.loc[chosen.rewarded.eq(0),'reward_outcome_time'].to_numpy(),
        bout_start_away=selected.loc[selected.cue_distance.gt(2)&selected.choice_distance.gt(2),'start'].to_numpy())
    early_licks=np.searchsorted(licks,cue+2)-np.searchsorted(licks,cue)
    events['cue_with_lick']=cue[early_licks>0]
    events['cue_no_lick']=cue[early_licks==0]
    return events,bouts,licks


def event_indices(times, meta, common, fold, cfg, offset=False):
    before=cfg.offset_before_s if offset else cfg.before_s;after=cfg.offset_after_s if offset else cfg.after_s
    dt=float(np.diff(meta['edges'][:2])[0]);offsets=np.arange(-round(before/dt),round(after/dt))
    index=np.rint(np.asarray(times)[:,None]/dt-.5+offsets).astype(int)
    bounded=(index.min(axis=1)>=0)&(index.max(axis=1)<len(common))
    selected=np.flatnonzero(bounded);idx=index[bounded]
    keep=common[idx].all(axis=1)&(fold[idx]==fold[idx[:,round(before/dt)],None]).all(axis=1) if len(idx) else np.zeros(0,bool)
    selected=selected[keep];idx=idx[keep]
    return idx,selected,offsets*dt


def match_units(frame, reference, target, cfg, rich):
    names=['analysis_firing_rate_hz']
    if rich:names+=['analysis_isi_cv','peak_to_valley_ms','half_width_ms','depth','available_fraction']
    data=frame.dropna(subset=names).copy();a=data.loc[data.cluster.eq(reference)];b=data.loc[data.cluster.eq(target)]
    if not len(a) or not len(b):return pd.DataFrame(),pd.DataFrame()
    values=data[names].to_numpy(float);values[:,0]=np.log(values[:,0])
    if rich:values[:,1]=np.log(values[:,1])
    standardized=pd.DataFrame((values-values.mean(0))/np.maximum(values.std(0),1e-8),index=data.index,columns=names)
    delta=standardized.loc[b.index].to_numpy()[:,None,:]-standardized.loc[a.index].to_numpy()[None,:,:]
    cost=np.sqrt(np.mean(delta**2,axis=2));valid=(b.decoder_label.to_numpy()[:,None]==a.decoder_label.to_numpy())&(b.ccf_acronym.to_numpy()[:,None]==a.ccf_acronym.to_numpy())
    ratio=b.analysis_firing_rate_hz.to_numpy()[:,None]/a.analysis_firing_rate_hz.to_numpy()
    valid&=(ratio>=1/cfg.match_rate_ratio)&(ratio<=cfg.match_rate_ratio)
    if rich:
        ratio=b.analysis_isi_cv.to_numpy()[:,None]/a.analysis_isi_cv.to_numpy()
        valid&=(ratio>=1/cfg.match_isi_ratio)&(ratio<=cfg.match_isi_ratio)
        for field,limit in [('peak_to_valley_ms',cfg.match_peak_valley_ms),('half_width_ms',cfg.match_half_width_ms),('depth',cfg.match_depth),('available_fraction',cfg.match_availability)]:
            valid&=abs(b[field].to_numpy()[:,None]-a[field].to_numpy())<=limit
    # Expensive dummy matches maximize valid match count before minimizing distance.
    costs=np.column_stack([np.where(valid,cost,1e6),np.full((len(b),len(b)),1000.)])
    rr,cc=linear_sum_assignment(costs);pairs=[]
    for i,j in zip(rr,cc):
        if j<len(a) and valid[i,j]:pairs.append(dict(reference=a.iloc[j].global_unit_id,target=b.iloc[i].global_unit_id,distance=cost[i,j],sorting=b.iloc[i].decoder_label,region=b.iloc[i].ccf_acronym))
    pairs=pd.DataFrame(pairs);balance=[]
    if len(pairs):
        lookup=data.set_index('global_unit_id')
        for name in names:
            balance.append(dict(feature=name,reference_before=a[name].median(),target_before=b[name].median(),
                reference_matched=lookup.loc[pairs.reference,name].median(),target_matched=lookup.loc[pairs.target,name].median()))
    return pairs,pd.DataFrame(balance)


def analyze_recording(cfg, spec, data, output):
    from src.experiments.clustering.plots import style,PALETTE
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.preprocessing import SplineTransformer
    root=Path(cfg.glm);out=output/spec.key;out.mkdir(parents=True,exist_ok=True)
    d=data.loc[data.key.eq(spec.key)&data.cluster.isin(spec.clusters)].sort_values(['cluster','unit_id']).reset_index(drop=True)
    assert len(d) and d.global_unit_id.is_unique
    first=root/d.final_file.iloc[0];recording=first.parent.parent;prep=recording/'prepared'
    info=json.loads((prep/'design_info.json').read_text());meta=dict(np.load(prep/'design_metadata.npz'));trials=pd.read_csv(prep/'trials.csv')
    units=pd.read_csv(prep/'units.csv');counts=np.load(prep/'counts.npy',mmap_mode='r');lookup={uid:i for i,uid in enumerate(units.global_unit_id)}
    dt=info['bin_size'];predictions=[];fold=None;common=meta['usable'].copy();coef=[]
    for row in d.itertuples():
        file=recording/'validation/units'/Path(row.final_file).name
        with np.load(file) as z:
            if fold is None:fold=z['fold'].copy()
            else:assert np.array_equal(fold,z['fold'])
            predictions.append(z['prediction']);coef.append(z['coefficients'])
            common&=np.isfinite(z['prediction'])
    y=np.asarray(counts[[lookup[uid] for uid in d.global_unit_id]],float);pred=np.stack(predictions)
    assert np.all(fold[common]>=0)
    mean_rate=y[:,meta['usable']].mean(axis=1)/dt;assert (mean_rate>0).all()
    d['usable_mean_rate_hz']=mean_rate;d.to_csv(out/'units.csv',index=False)
    events,bouts,licks=events_from_recording(info,trials,cfg);bouts.to_csv(out/'lick_bouts.csv',index=False)
    sources={'observed':y,'heldout':pred}
    # December 10 only: model-based comparison at zero recorded speed, not causal intervention.
    if info.get('movement_available',True):
        x=sparse.load_npz(prep/'design.npz');idx=info['slices']['movement'];movement=x[:,idx].toarray()
        spline=SplineTransformer(degree=2,knots=np.array(info['movement_knots_cm_s'])[:,None],include_bias=False,extrapolation='linear').fit(np.array([[0.],[40.]]))
        relative=movement-spline.transform(np.array([[0.]]))[0];fixed=pred.copy()
        for i,c in enumerate(coef):
            for f in range(3):
                m=fold==f;effect=relative[m]@c[f,idx]
                fixed[i,m]=pred[i,m]*np.exp(np.clip(-effect,-30,30))
        sources['heldout_speed_zero']=fixed
        del x,movement,relative
    stride=round(cfg.plot_bin_s/dt);n_bins=round((cfg.before_s+cfg.after_s)/dt)
    assert n_bins%stride==0 and abs(stride*dt-cfg.plot_bin_s)<1e-8
    # The alignment grid has spacing dt and ≤dt/2 event-to-bin-center error.
    lag=(np.arange(-round(cfg.before_s/dt),round(cfg.after_s/dt))*dt).reshape(-1,stride).mean(axis=1)
    lick_counts=np.histogram(licks,meta['edges'])[0]/dt
    curves={};metrics=[];event_audit=[];behavior_rows=[];event_selection=[]
    for event,times in events.items():
        ix,chosen,fine_lag=event_indices(times,meta,common,fold,cfg,offset=event=='bout_end')
        lag=fine_lag.reshape(-1,stride).mean(axis=1)
        event_audit.append(dict(event=event,total=len(times),retained=len(ix)))
        for j,time in enumerate(times):event_selection.append(dict(event=event,time_s=time,retained=bool(j in set(chosen))))
        if len(ix)<cfg.min_events:continue
        event_fold=fold[ix[:,np.argmin(abs(fine_lag))]]
        base=list(cfg.offset_baseline_s if event=='bout_end' else cfg.baseline_s)
        response=list(cfg.offset_response_s if event=='bout_end' else cfg.response_s)
        base_mask=(lag>=base[0])&(lag<base[1]);resp_mask=(lag>=response[0])&(lag<response[1])
        group={'lag_s':lag}
        for source,array in sources.items():
            sampled=array[:,ix].reshape(len(d),len(ix),len(lag),stride).mean(axis=3)/dt
            psth=sampled.mean(axis=1);baseline=psth[:,base_mask].mean(axis=1)
            group[source]=psth;group[source+'_normalized']=(psth-baseline[:,None])/mean_rate[:,None]
            for i,row in enumerate(d.itertuples()):
                for f in [-1,0,1,2]:
                    selected=np.ones(len(ix),bool) if f==-1 else event_fold==f
                    if selected.sum()<cfg.min_events:continue
                    pre=sampled[i,selected][:,base_mask].mean();post=sampled[i,selected][:,resp_mask].mean()
                    metrics.append(dict(global_unit_id=row.global_unit_id,unit_id=row.unit_id,cluster=row.cluster,sorting=row.decoder_label,
                        qc=row.default_qc,coverage=row.available_fraction,event=event,source=source,fold=f,n_events=int(selected.sum()),
                        baseline_hz=pre,response_hz=post,delta_hz=post-pre,normalized_delta=(post-pre)/mean_rate[i]))
        curves[event]=group
        for name,values in [('licking_hz',lick_counts),('speed_cm_s',meta['speed'])]:
            if not np.isfinite(values[ix]).any():continue
            avg=np.mean(values[ix].reshape(len(ix),len(lag),stride),axis=(0,2))
            for xx,value in zip(lag,avg):behavior_rows.append(dict(event=event,feature=name,time_s=xx,value=value,n_events=len(ix)))
    pd.DataFrame(event_audit).to_csv(out/'event_counts.csv',index=False);pd.DataFrame(event_selection).to_csv(out/'event_selection.csv',index=False)
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'unit_responses.csv',index=False);behavior=pd.DataFrame(behavior_rows);behavior.to_csv(out/'aligned_behavior.csv',index=False)
    # Compare quiet-before, three relative bout phases, and quiet-after on the same bouts.
    phase_rows=[];phase_values={name:[] for name in sources};phase_bouts=[]
    centers=(meta['edges'][:-1]+meta['edges'][1:])/2
    for bout in bouts.loc[bouts.eligible_definition].itertuples():
        start,end=bout.start,bout.end;duration=end-start
        lo,hi=np.searchsorted(centers,[start-1,end+.8])
        if lo<0 or hi>=len(common) or hi<=lo or not common[lo:hi].all() or len(np.unique(fold[lo:hi]))!=1:continue
        bounds=[(start-1,start-.2),(start,start+.2*duration),(start+.2*duration,start+.8*duration),(start+.8*duration,end),(end+.2,end+.8)]
        ranges=[np.arange(*np.searchsorted(centers,[a,b])) for a,b in bounds]
        if any(not len(ix) for ix in ranges):continue
        phase_bouts.append(dict(bout_index=bout.Index,start=start,end=end,duration=duration,fold=int(fold[lo])))
        for name,array in sources.items():phase_values[name].append(np.stack([array[:,ix].mean(axis=1)/dt for ix in ranges],axis=1))
    phase_names=['Before','Early','Middle','Late','After']
    for source,values in phase_values.items():
        if not values:continue
        mean=np.stack(values).mean(axis=0)
        for i,row in enumerate(d.itertuples()):
            for phase,value in zip(phase_names,mean[i]):phase_rows.append(dict(global_unit_id=row.global_unit_id,cluster=row.cluster,sorting=row.decoder_label,phase=phase,source=source,rate_hz=value,normalized_change=(value-mean[i,0])/mean_rate[i],n_bouts=len(values)))
    phases=pd.DataFrame(phase_rows);phases.to_csv(out/'bout_phase_rates.csv',index=False);pd.DataFrame(phase_bouts).to_csv(out/'bout_phase_events.csv',index=False)
    np.savez_compressed(out/'aligned_responses.npz',global_unit_id=d.global_unit_id.to_numpy(str),**{e+'__'+s:v for e,g in curves.items() for s,v in g.items()})
    summaries=[]
    for scope,ids in [('All',d.global_unit_id),('SUA',d.loc[d.decoder_label.eq('sua'),'global_unit_id']),('QC SUA',d.loc[d.decoder_label.eq('sua')&d.default_qc.eq(True),'global_unit_id']),('Coverage >75%',d.loc[d.available_fraction.gt(.75),'global_unit_id'])]:
        sub=metrics.loc[metrics.global_unit_id.isin(ids)]
        for (cluster,event,source,f),g in sub.groupby(['cluster','event','source','fold']):
            summaries.append(dict(scope=scope,cluster=cluster,event=event,source=source,fold=f,n_units=len(g),n_events=g.n_events.iloc[0],median_delta_hz=g.delta_hz.median(),median_normalized_delta=g.normalized_delta.median(),positive_fraction=g.delta_hz.gt(0).mean()))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'cluster_responses.csv',index=False)
    # Fold sign stability is descriptive; folds share the recording and cluster discovery.
    stable=metrics.loc[metrics.source.eq('observed')&metrics.fold.ge(0)].groupby(['global_unit_id','cluster','sorting','event']).agg(n_folds=('fold','size'),positive_folds=('delta_hz',lambda x:x.gt(0).sum()),negative_folds=('delta_hz',lambda x:x.lt(0).sum())).reset_index()
    stable['same_sign_all_folds']=stable.n_folds.eq(3)&(stable.positive_folds.eq(3)|stable.negative_folds.eq(3));stable.to_csv(out/'fold_sign_consistency.csv',index=False)
    match_results=[];match_summary=[]
    for target in list(spec.clusters)[1:]:
        for rich in [False,True]:
            mode='Rate + physiology' if rich else 'Rate only';pairs,balance=match_units(d,int(spec.clusters[0]),int(target),cfg,rich)
            prefix=f'C{int(spec.clusters[0])+1}_C{int(target)+1}_'+('physiology' if rich else 'rate')
            pairs.to_csv(out/(prefix+'_pairs.csv'),index=False);balance.to_csv(out/(prefix+'_balance.csv'),index=False)
            match_summary.append(dict(reference_cluster=int(spec.clusters[0]),target_cluster=int(target),mode=mode,n_pairs=len(pairs),n_sua_pairs=int(pairs.sorting.eq('sua').sum()) if len(pairs) else 0))
            if not len(pairs):continue
            for (event,source),m in metrics.loc[metrics.fold.eq(-1)].groupby(['event','source']):
                lookup_metric=m.set_index('global_unit_id')
                for pair in pairs.itertuples():
                    match_results.append(dict(reference_cluster=int(spec.clusters[0]),target_cluster=int(target),mode=mode,event=event,source=source,
                        reference=pair.reference,target=pair.target,sorting=pair.sorting,
                        reference_response=lookup_metric.loc[pair.reference,'normalized_delta'],target_response=lookup_metric.loc[pair.target,'normalized_delta']))
    pd.DataFrame(match_results).to_csv(out/'matched_responses.csv',index=False);pd.DataFrame(match_summary).to_csv(out/'matching_summary.csv',index=False)
    style()
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    title=f'{d.animal.iloc[0]} · {d.session.iloc[0][:10]}'
    for scope,mask in [('all',np.ones(len(d),bool)),('sua',d.decoder_label.eq('sua').to_numpy())]:
        fig,axes=plt.subplots(len(spec.clusters),2,figsize=(10,2.6*len(spec.clusters)),squeeze=False,sharex=True)
        for row,cluster in enumerate(spec.clusters):
            selected=mask&d.cluster.eq(cluster).to_numpy()
            for col,(conditions,labels) in enumerate([(['cue_with_lick','cue_no_lick'],['Lick within 2 s','No lick within 2 s']),(['choice_reward','choice_omission'],['Reward','Omission'])]):
                ax=axes[row,col]
                for event,label,ls,color in zip(conditions,labels,['-','--'],[PALETTE[cluster],'#657483']):
                    if event not in curves or not selected.any():continue
                    g=curves[event];n=next(x['retained'] for x in event_audit if x['event']==event)
                    ax.plot(g['lag_s'],np.median(g['observed_normalized'][selected],axis=0),color=color,ls=ls,lw=1.8,label=f'{label} · {n}')
                ax.set(title=f'C{cluster+1} · '+('Cue' if col==0 else 'Choice / outcome'),xlabel='Time (s)',ylabel='Δ rate / session mean')
                ax.axhline(0,color='#bbc3cd',lw=.7);ax.axvline(0,color='#bbc3cd',lw=.7,ls=':');ax.legend(frameon=False,fontsize=8)
        fig.suptitle(title+' · observed trial conditions · '+scope.upper());fig.tight_layout();save(fig,'trial_conditions_'+scope)
    if len(phases):
        fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
        for ax,scope in zip(axes,['All','SUA']):
            p=phases if scope=='All' else phases.loc[phases.sorting.eq('sua')]
            for cluster in spec.clusters:
                for source,ls in [('observed','-'),('heldout','--')]:
                    g=p.loc[p.cluster.eq(cluster)&p.source.eq(source)]
                    line=g.groupby('phase').normalized_change.median().reindex(phase_names)
                    ax.plot(range(5),line,color=PALETTE[cluster],ls=ls,marker='o',ms=3,label=f'C{cluster+1}' if source=='observed' else None)
            ax.axhline(0,color='#bbb',lw=.7);ax.set_xticks(range(5),phase_names);ax.set_title(scope);ax.legend(frameon=False)
        axes[0].set_ylabel('Δ rate / session mean');fig.suptitle(title+f' · {len(phase_bouts)} complete bouts');fig.tight_layout();save(fig,'bout_phases')
    for scope,mask in [('All',np.ones(len(d),bool)),('SUA',d.decoder_label.eq('sua').to_numpy())]:
        fig,axes=plt.subplots(2,3,figsize=(12,6),gridspec_kw={'height_ratios':[2,1]})
        for column,event in enumerate(PRIMARY):
            ax=axes[0,column]
            if event not in curves:ax.set_title(EVENT_LABELS[event]+' · insufficient events');continue
            g=curves[event];lag=g['lag_s'];n=next(x['retained'] for x in event_audit if x['event']==event)
            for cluster in spec.clusters:
                selected=mask&d.cluster.eq(cluster).to_numpy()
                if not selected.any():continue
                for source,ls in [('observed','-'),('heldout','--')]:
                    ax.plot(lag,np.median(g[source+'_normalized'][selected],axis=0),color=PALETTE[cluster],ls=ls,lw=1.7,label=f'C{cluster+1} · {selected.sum()}' if source=='observed' else None)
            ax.axhline(0,color='#bbc3cd',lw=.7);ax.axvline(0,color='#bbc3cd',lw=.7,ls=':')
            ax.set(title=f'{EVENT_LABELS[event]} · {n} events',ylabel='Δ rate / session mean');ax.legend(frameon=False,fontsize=8)
            b=behavior.loc[behavior.event.eq(event)];bax=axes[1,column]
            l=b.loc[b.feature.eq('licking_hz')];bax.plot(l.time_s,l.value,color='#687583');bax.set(ylabel='Licks/s',xlabel='Time (s)')
            speed=b.loc[b.feature.eq('speed_cm_s')]
            if len(speed):
                other=bax.twinx();other.plot(speed.time_s,speed.value,color='#299d8f');other.set_ylabel('Speed (cm/s)',color='#299d8f')
            bax.axvline(0,color='#bbc3cd',lw=.7,ls=':')
        fig.suptitle(title+' · '+scope);fig.text(.5,.015,'Solid: observed · dashed: held-out prediction · offset change is relative to pre-offset activity',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.95));save(fig,'responses_'+scope.lower())
    for scope,mask in [('all',np.ones(len(d),bool)),('sua',d.decoder_label.eq('sua').to_numpy())]:
        selection=np.flatnonzero(mask);fig,axes=plt.subplots(1,3,figsize=(12,max(4,.22*len(selection))),sharey=True)
        for ax,event in zip(axes,PRIMARY):
            if event not in curves:continue
            lag=curves[event]['lag_s']
            im=ax.imshow(curves[event]['observed_normalized'][selection],aspect='auto',cmap='RdBu_r',vmin=-1.5,vmax=1.5,extent=[lag[0]-.05,lag[-1]+.05,len(selection)-.5,-.5])
            ax.axvline(0,color='#555',lw=.7,ls=':');ax.set(title=EVENT_LABELS[event],xlabel='Time (s)')
        axes[0].set_yticks(range(len(selection)),[f'C{int(d.iloc[i].cluster)+1} · {int(d.iloc[i].unit_id)}' for i in selection],fontsize=7)
        fig.suptitle(title+' · individual '+scope.upper()+' units')
        fig.subplots_adjust(left=.13,right=.87,bottom=.1,top=.9,wspace=.1)
        fig.colorbar(im,cax=fig.add_axes([.9,.2,.015,.6]),label='Δ rate / session mean (clipped ±1.5)');save(fig,'unit_heatmap_'+scope)
    gallery=out/'single_units';gallery.mkdir(exist_ok=True)
    for i,row in d.loc[d.decoder_label.eq('sua')].iterrows():
        fig,axes=plt.subplots(1,3,figsize=(11,3))
        for ax,event in zip(axes,PRIMARY):
            if event not in curves:continue
            lag=curves[event]['lag_s']
            for source,color,ls in [('observed','#344454','-'),('heldout',PALETTE[int(row.cluster)],'--')]:ax.plot(lag,curves[event][source][i],color=color,ls=ls,lw=1.4,label=source)
            ax.axvline(0,color='#bbb',lw=.7,ls=':');ax.set(title=EVENT_LABELS[event],xlabel='Time (s)',ylabel='Hz')
        axes[0].legend(frameon=False,fontsize=8);fig.suptitle(f'C{int(row.cluster)+1} · unit {int(row.unit_id)} · {row.ccf_acronym}');fig.tight_layout();fig.savefig(gallery/f'unit_{int(row.unit_id):04d}.png',dpi=130);plt.close(fig)
    checks=dict(units=len(d),single_units=int(d.decoder_label.eq('sua').sum()),clock_source=info['clock'],common_test_bins=int(common.sum()),
        aligned_events=event_audit,complete_bouts=len(phase_bouts),all_windows_heldout_and_single_fold=True,identity_checks='passed',movement_available='heldout_speed_zero' in sources)
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(title, 'events:',event_audit,flush=True)
    print(summary.loc[summary.scope.isin(['All','SUA'])&summary.fold.eq(-1)&summary.source.eq('observed')&summary.event.isin(PRIMARY)].to_string(index=False),flush=True)
    print(pd.DataFrame(match_summary).to_string(index=False),flush=True)
    return checks


def run(cfg):
    from omegaconf import OmegaConf
    out=Path(cfg.output);out.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,out/'config.yaml')
    data=pd.read_csv(Path(cfg.comparison)/'unit_features.csv');checks={}
    for spec in cfg.recordings:checks[spec.key]=analyze_recording(cfg,spec,data,out)
    method=dict(primary='Saved cluster memberships and converged three-fold GLMs; no new fits. Common held-out event windows, same events for every unit in a recording.',
        observed='20-ms spike counts averaged into 100-ms PSTHs; full window must be usable, held out and in a single CV fold. Exact event time alignment rounded to nearest count-bin center (≤10 ms).',
        normalization='Per-unit change from local baseline divided by that unit\'s mean rate across all eligible GLM bins; then median across units. Not a z score, baseline-rate percentage, or isolated conditional kernel.',
        bout_definition='Both lick sides combined, including registered choice licks; gap >1 s separates bouts; ≥3 licks and ≥1 s duration. First and last detected bouts excluded because surrounding quiet is unknown.',
        response_windows='Cue/onset/choice: baseline -1 to -0.2 s, response 0 to 0.8 s. Offset: baseline -0.8 to -0.2 s, response 0.2 to 0.8 s.',
        offset_window='Offset plots use -0.8 to +0.8 s because longer windows cross unmodeled intertrial gaps and retain only 6 March 31 / 4 December 10 bouts. Other plots use -2 to +3 s. Selection is based on coverage, not neural responses.',
        cue_subsets='Cue with/without any detected lick in the following 2 s; descriptive behavioral conditioning, not randomized trial groups.',
        bout_phases='Same complete held-out bouts: before -1 to -0.2 s; first 20%, middle 60%, last 20% of bout duration; after +0.2 to +0.8 s. Equal bout weighting, then median across units.',
        matching='Without-replacement assignment, exact sorting and region; rate ratio ≤2. Rich matching additionally requires ISI-CV ratio ≤2, peak-valley difference ≤0.2 ms, half-width difference ≤0.08 ms, stored depth difference ≤400, availability difference ≤0.25. Supports limited-overlap conclusions only; no significance tests.',
        movement='December 10 only: held-out model prediction with movement spline set to its value at speed zero, keeping other predictors fixed. Descriptive model adjustment, not an observed or causal intervention.',
        limitations=['Same recordings used for embedding/cluster discovery; held-out GLM predictions do not make this an independent cluster validation',
            'PSTHs combine overlapping cue, choice, licking and reward effects; events and units are correlated',
            'March 31 lacks movement measurements; bout cessation is not equivalent to behavioral disengagement',
            'Matching may retain few units; unmeasured physiology remains uncontrolled',
            'Offset PSTHs need not measure recovery to an absolute baseline; check raw Hz traces'],
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),out/'config.yaml',Path(cfg.comparison)/'unit_features.csv',Path(cfg.glm)/'schema.json']})
    (out/'method.json').write_text(json.dumps(method,indent=2)+'\n');(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    page='<html><meta charset="utf-8"><title>Observed cluster task responses</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto}img{width:100%}a{color:#3976b9}</style><h1>Observed and held-out cluster responses</h1><p>Exploratory within-recording analysis. Solid lines: observed spikes; dashed: held-out GLM predictions. Curves are changes from local baseline divided by session mean firing rate.</p><p><a href="method.json">Methods and limitations</a></p>'
    for spec in cfg.recordings:
        page+=f'<h2>{spec.key}</h2><img src="{spec.key}/responses_all.png"><img src="{spec.key}/responses_sua.png"><img src="{spec.key}/bout_phases.png"><img src="{spec.key}/trial_conditions_all.png"><img src="{spec.key}/trial_conditions_sua.png"><img src="{spec.key}/unit_heatmap_sua.png"><p><a href="{spec.key}/unit_responses.csv">Unit responses</a> · <a href="{spec.key}/matching_summary.csv">Matching coverage</a> · <a href="{spec.key}/matched_responses.csv">Matched comparisons</a> · <a href="{spec.key}/event_counts.csv">Event coverage</a></p>'
        for path in sorted((out/spec.key/'single_units').glob('*.png')):page+=f'<details><summary>{path.stem}</summary><img src="{path.relative_to(out)}"></details>'
    (out/'index.html').write_text(page+'</html>')
