"""Feature-by-feature GLM axes and bidirectional transfer in a fixed encoder space."""
import html
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

from src.common.analysis import correlation, sha256, write_json
from src.experiments.transfer.active_cue_axes import fit_axes, project, predict, kernel_metrics, neighbors
from src.embeddings.io import read_table
from src.data.io import resolve
from src.common.plotting import STYLE


def definitions():
    specs=[]
    def kernel(name,label,weights,groups,derived=False):
        lick='lick' in name
        specs.append(dict(name=name,label=label,kind='kernel',weights=weights,groups=groups,
            support_s=[-.4,.6] if lick else [-1,2],summary_s=[-.1,.1] if lick else [0,1],
            derived=derived,units='Conditional log-rate contrast' if name not in ['cue','choice_common','left_lick','right_lick'] else 'Conditional log-rate effect'))
    kernel('cue','Go cue',{'cue':1},['Cue'])
    kernel('choice_common','Common choice / outcome',{'choice_common':1},['Choice / outcome'])
    kernel('right_minus_left','Right − left',{'choice_side':-1},['Choice side'])
    kernel('outcome','Reward − omission',{'outcome':1},['Outcome'])
    kernel('side_outcome','Side × outcome',{'side_outcome':-1},['Choice side','Outcome'])
    kernel('left_lick','Left lick',{'left_lick':1},['Licking'])
    kernel('right_lick','Right lick',{'right_lick':1},['Licking'])
    for name,label,group,index,multiplier in [
        ('probability_left','Left reward probability','probability',0,.1),
        ('probability_right','Right reward probability','probability',1,.1),
        ('history_response','Previous valid response','history',0,1),
        ('history_side','Previous right − left','history',1,-1),
        ('history_outcome','Previous reward − omission','history',2,1)]:
        specs.append(dict(name=name,label=label,kind='scalar',coefficient_group=group,index=index,multiplier=multiplier,
            groups=['Reward probability' if group=='probability' else 'History'],derived=False,
            units='Log-rate effect per +0.10 programmed probability' if group=='probability' else 'Conditional log-rate contrast'))
    kernel('side_reward','Right − left · rewarded',{'choice_side':-1,'side_outcome':-.5},['Choice side'],True)
    kernel('side_omission','Right − left · unrewarded',{'choice_side':-1,'side_outcome':.5},['Choice side'],True)
    kernel('outcome_left','Reward − omission · left',{'outcome':1,'side_outcome':.5},['Outcome'],True)
    kernel('outcome_right','Reward − omission · right',{'outcome':1,'side_outcome':-.5},['Outcome'],True)
    return specs


def load(cfg,out,specs):
    root,glm=resolve(cfg.source),resolve(cfg.glm)
    units,targets=read_table(root/'units.csv'),read_table(cfg.targets)
    assert units.global_unit_id.is_unique and targets.global_unit_id.is_unique
    targets=targets[targets.analysis_included.astype(str).str.lower().eq('true')]
    frame=units.drop(columns=['cluster','sua_cluster','umap_1','umap_2'],errors='ignore').merge(
        targets[['global_unit_id','final_file']+[c for c in targets if c.startswith('gain_')]],on='global_unit_id',validate='one_to_one')
    frame=frame[frame.session.isin(cfg.sessions)].reset_index(drop=True)
    memberships=read_table(root/'clusters.csv')
    membership=memberships[memberships.encoder.eq(cfg.encoder)].set_index('global_unit_id').cluster
    frame['cluster']=frame.global_unit_id.map(membership).astype(int)
    frame['cluster_name']='C'+(frame.cluster+1).astype(str)
    reliability=read_table(glm/'reliability.csv')
    reliability['supported']=reliability.heldout_gain.gt(0)&reliability.positive_folds.ge(2)&reliability.all_converged.eq(True)
    supported=reliability.pivot(index='global_unit_id',columns='group',values='supported').reindex(frame.global_unit_id)
    with np.load(root/'embeddings.npz',allow_pickle=False) as z:
        positions=pd.Index(z['global_unit_id'].astype(str)).get_indexer(frame.global_unit_id)
        assert (positions>=0).all()
        x=z[str(cfg.encoder)][positions].astype(float)
    records={s['name']:[] for s in specs}
    cvs={s['name']:[] for s in specs}
    cached={}
    for row in frame.itertuples():
        path=glm/row.final_file
        recording=path.parent.parent
        if recording not in cached:
            info=json.loads((recording/'prepared/design_info.json').read_text())
            with np.load(recording/'basis.npz',allow_pickle=False) as z:
                basis={key:z[key] for key in ['lags_s','cue','choice_common','choice_side','outcome','side_outcome','left_lick','right_lick']}
            cached[recording]=info,basis
        info,basis=cached[recording]
        lag=basis['lags_s']
        with np.load(path,allow_pickle=False) as z:
            coefficients=z['coefficients']
            raw={key:z['kernels'][list(z['kernel_names']).index(key)] for key in basis if key!='lags_s'}
            for key in raw:assert np.allclose(raw[key],basis[key]@coefficients[info['slices'][key]])
        with np.load(recording/'validation/units'/path.name,allow_pickle=False) as z:
            cv_coef=z['coefficients']
        for spec in specs:
            name=spec['name']
            if spec['kind']=='kernel':
                mask=(lag>=spec['support_s'][0]-1e-8)&(lag<=spec['support_s'][1]+1e-8)
                records[name].append(sum(weight*raw[key] for key,weight in spec['weights'].items())[mask])
                cvs[name].append(sum(weight*(cv_coef[:,info['slices'][key]]@basis[key].T) for key,weight in spec['weights'].items())[:,mask])
            else:
                index=info['slices'][spec['coefficient_group']][spec['index']]
                records[name].append(np.atleast_1d(coefficients[index]*spec['multiplier']))
                cvs[name].append(cv_coef[:,[index]]*spec['multiplier'])
    data,archive={},dict(global_unit_id=frame.global_unit_id.to_numpy(str))
    long=[]
    for spec in specs:
        name=spec['name']
        y,cv=np.stack(records[name]),np.stack(cvs[name])
        assert np.isfinite(y).all() and np.isfinite(cv).all()
        axis=lag[(lag>=spec['support_s'][0]-1e-8)&(lag<=spec['support_s'][1]+1e-8)] if spec['kind']=='kernel' else np.asarray([0.])
        window=(axis>=spec['summary_s'][0]-1e-8)&(axis<=spec['summary_s'][1]+1e-8) if spec['kind']=='kernel' else np.ones(1,bool)
        reliable=supported[spec['groups']].fillna(False).all(axis=1).to_numpy(bool)
        scalar=y[:,window].mean(1)
        cv_scalar=cv[:,:,window].mean(2)
        data[name]=dict(y=y,cv=cv,axis=axis,window=window,supported=reliable,spec=spec)
        archive[name]=y
        archive[name+'_validation']=cv
        archive[name+'_axis']=axis
        for i,gid in enumerate(frame.global_unit_id):
            long.append(dict(global_unit_id=gid,feature=name,value=scalar[i],parent_group_supported=bool(reliable[i]),
                cv_min=float(cv_scalar[i].min()),cv_max=float(cv_scalar[i].max()),
                cv_sign_agreement=float((np.sign(cv_scalar[i])==np.sign(scalar[i])).mean())))
    np.savez_compressed(out/'features.npz',**archive)
    pd.DataFrame(long).to_csv(out/'unit_features.csv',index=False)
    frame.to_csv(out/'units.csv',index=False)
    write_json(out/'feature_definitions.json',specs)
    availability=read_table(glm/'recording_availability.csv')
    availability[availability.animal.astype(str).eq('745502')].to_csv(out/'recording_availability.csv',index=False)
    write_json(out/'excluded_features.json',dict(movement='Unavailable in December 18; cannot test transfer across this pair.',
        drift='Session nuisance basis, not treated as a functional response target.',
        rpe='Not present in the cached GLM; reward outcome is not relabeled as prediction error.',
        validation_gains='Reliability descriptors, not full-recording response coefficients.'))
    return frame,x,data


class Uncertainty:
    def __init__(self,cfg):
        self.cfg=cfg
        self.weights={}
        self.permutations={}

    def interval(self,y,pred,window):
        n=len(y)
        if n not in self.weights:
            rng=np.random.default_rng(int(self.cfg.seed)+77)
            self.weights[n]=rng.multinomial(n,np.ones(n)/n,size=int(self.cfg.bootstrap_samples)).astype(float)
        w=self.weights[n]
        total=w@np.square(y).sum(1)-np.square(w@y).sum(1)/n
        error=w@np.square(y-pred).sum(1)
        r2=1-error/np.maximum(total,1e-15)
        a,b=y[:,window].mean(1),pred[:,window].mean(1)
        aa=w@(a*a)-np.square(w@a)/n
        bb=w@(b*b)-np.square(w@b)/n
        numerator=w@(a*b)-(w@a)*(w@b)/n
        denom=np.sqrt(np.maximum(aa*bb,0))
        good=denom>1e-12
        r=numerator[good]/denom[good]
        lo,hi=np.quantile(r,[.025,.975]) if len(r) else [np.nan,np.nan]
        low,high=np.quantile(r2,[.025,.975])
        return dict(r2_ci_low=low,r2_ci_high=high,summary_r_ci_low=lo,summary_r_ci_high=hi)

    def permutation(self,y,pred):
        n=len(y)
        if n not in self.permutations:
            rng=np.random.default_rng(int(self.cfg.seed)+119)
            self.permutations[n]=np.stack([rng.permutation(n) for _ in range(int(self.cfg.permutations))])
        a,b=y-y.mean(0),pred-pred.mean(0)
        denom=np.linalg.norm(a)*np.linalg.norm(b)
        if denom<1e-12:return np.nan,np.zeros(int(self.cfg.permutations))
        cross=b@a.T
        observed=np.trace(cross)/denom
        null=cross[np.arange(n)[None,:],self.permutations[n]].sum(1)/denom
        return float((1+(null>=observed-1e-14).sum())/(len(null)+1)),null


def bh(p):
    p=np.asarray(p,float)
    result=np.full(len(p),np.nan)
    valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid])]
    result[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return result


def run_models(cfg,out,frame,x,data,specs):
    uncertainty=Uncertainty(cfg)
    rows,sensitivity,coordinates,assignments,associations,quartiles=[],[],[],[],[],[]
    nulls={}
    snapshots={}
    for directory in ['models','predictions','figures']:(out/directory).mkdir(exist_ok=True)
    for direction,(source_session,target_session) in enumerate([list(cfg.sessions),list(cfg.sessions)[::-1]]):
        sm=frame.session.eq(source_session).to_numpy()
        tm=frame.session.eq(target_session).to_numpy()
        sf,tf=frame[sm].copy(),frame[tm].copy()
        xs,xt=x[sm],x[tm]
        splits=list(KFold(int(cfg.folds),shuffle=True,random_state=int(cfg.seed)).split(xs))
        folds=np.full(len(sf),-1,int)
        source_neighbors=[]
        for fold,(tr,te) in enumerate(splits):
            assert not set(tr)&set(te)
            folds[te]=fold
            source_neighbors.append(neighbors(normalize(xs[tr]),normalize(xs[te]),int(cfg.neighbors),'cosine'))
        target_neighbors=neighbors(normalize(xs),normalize(xt),int(cfg.neighbors),'cosine')
        for gid,fold in zip(sf.global_unit_id,folds):assignments.append(dict(source_session=source_session,global_unit_id=gid,fold=int(fold)))
        for spec in specs:
            name=spec['name']
            item=data[name]
            ys,yt=item['y'][sm],item['y'][tm]
            window=item['window']
            components=int(cfg.kernel_components if spec['kind']=='kernel' else cfg.scalar_components)
            oof=np.zeros_like(ys)
            oof_one=np.zeros_like(ys)
            knn_oof=np.zeros_like(ys)
            source_mean=np.zeros_like(ys)
            for fold,(tr,te) in enumerate(splits):
                state=fit_axes(xs[tr],ys[tr],components,window)
                s=project(xs[te],state)
                oof[te]=predict(s,state,components)
                oof_one[te]=predict(s,state,1)
                knn_oof[te]=ys[tr][source_neighbors[fold]].mean(1)
                source_mean[te]=ys[tr].mean(0)
            state=fit_axes(xs,ys,components,window)
            source_scores,target_scores=project(xs,state),project(xt,state)
            target_pred=predict(target_scores,state,components)
            target_one=predict(target_scores,state,1)
            target_knn=ys[target_neighbors].mean(1)
            target_mean=np.tile(ys.mean(0),(len(yt),1))
            model_name=f'd{direction}_{name}'
            np.savez_compressed(out/'models'/f'{model_name}.npz',**state,training_ids=sf.global_unit_id.to_numpy(str),feature_axis=item['axis'])
            with np.load(out/'models'/f'{model_name}.npz',allow_pickle=False) as z:
                restored={key:z[key] for key in z.files}
            assert np.allclose(predict(project(xt,restored),restored,components),target_pred)
            snapshots[(direction,name)]=dict(source_frame=sf,target_frame=tf,source_scores=source_scores,target_scores=target_scores,
                source_truth=ys,target_truth=yt,source_prediction=oof,target_prediction=target_pred,state=state)
            for evaluation,part,truth,estimates,baseline,reliable in [
                ('source_heldout',sf,ys,{'feature_axes':oof,'original_neighbors':knn_oof},source_mean,item['supported'][sm]),
                ('other_session',tf,yt,{'feature_axes':target_pred,'original_neighbors':target_knn},target_mean,item['supported'][tm])]:
                assert all(np.isfinite(v).all() for v in estimates.values())
                np.savez_compressed(out/'predictions'/f'{model_name}_{evaluation}.npz',global_unit_id=part.global_unit_id.to_numpy(str),
                    truth=truth,**estimates,source_mean=baseline,fold=folds if evaluation=='source_heldout' else np.full(len(part),-1))
                for readout,prediction in estimates.items():
                    pvalue=np.nan
                    if evaluation=='other_session':
                        pvalue,null=uncertainty.permutation(truth,prediction)
                        nulls[f'{model_name}_{readout}']=null
                    for scope,mask in [('All',np.ones(len(part),bool)),('SUA',part.decoder_label.eq('sua').to_numpy()),('Parent group supported',reliable)]:
                        if mask.sum()<10:continue
                        a,b=truth[mask],prediction[mask]
                        stats=kernel_metrics(a,b)
                        ref=np.square(a-baseline[mask]).sum()
                        rows.append(dict(feature=name,label=spec['label'],kind=spec['kind'],derived=spec['derived'],
                            source_session=source_session,target_session=target_session,direction=direction,
                            evaluation=evaluation,readout=readout,components=components if readout=='feature_axes' else 0,
                            scope=scope,n_units=int(mask.sum()),**stats,**uncertainty.interval(a,b,window),
                            summary_r=correlation(a[:,window].mean(1),b[:,window].mean(1)),
                            summary_spearman=float(spearmanr(a[:,window].mean(1),b[:,window].mean(1)).statistic),
                            source_mean_skill=1-np.square(a-b).sum()/ref if ref>0 else np.nan,
                            permutation_p=pvalue if scope=='All' else np.nan))
            if components>1:
                for evaluation,truth,estimate in [('source_heldout',ys,oof_one),('other_session',yt,target_one)]:
                    sensitivity.append(dict(feature=name,direction=direction,evaluation=evaluation,components=1,
                        **kernel_metrics(truth,estimate),summary_r=correlation(truth[:,window].mean(1),estimate[:,window].mean(1))))
            for role,part,scores,truth,mask in [('source',sf,source_scores,ys,sm),('target',tf,target_scores,yt,tm)]:
                temp=part[['global_unit_id','session','unit_id','label','decoder_label','cluster_name']].copy()
                temp['feature'],temp['direction'],temp['role']=name,direction,role
                temp['axis_1']=scores[:,0]
                temp['axis_2']=scores[:,1] if components>1 else np.nan
                temp['observed_summary']=truth[:,window].mean(1)
                temp['parent_group_supported']=item['supported'][mask]
                coordinates.append(temp)
                for prop in ['analysis_firing_rate_hz','analysis_isi_cv','peak_to_valley_ms','half_width_ms','ccf_ap','ccf_dv','ccf_ml']:
                    values=pd.to_numeric(part[prop],errors='coerce').to_numpy()
                    valid=np.isfinite(values)
                    if valid.sum()>10 and np.std(values[valid])>0:
                        associations.append(dict(feature=name,direction=direction,role=role,property=prop,n_units=int(valid.sum()),
                            axis_1_spearman=float(spearmanr(scores[valid,0],values[valid]).statistic)))
                for end,select in [('low',scores[:,0]<=np.quantile(scores[:,0],.25)),('high',scores[:,0]>=np.quantile(scores[:,0],.75))]:
                    g=part[select]
                    quartiles.append(dict(feature=name,direction=direction,role=role,session=part.session.iloc[0],end=end,n_units=len(g),
                        n_sua=int(g.decoder_label.eq('sua').sum()),n_d1=int(g.label.eq('D1').sum()),n_d2=int(g.label.eq('D2').sum()),
                        response_median=float(np.median(truth[select][:,window].mean(1))),
                        rate_median=g.analysis_firing_rate_hz.median(),clusters=g.cluster_name.value_counts().to_json()))
            print(f'Completed {source_session[5:10]} → {target_session[5:10]}: {name}',flush=True)
    metrics=pd.DataFrame(rows)
    test=metrics.evaluation.eq('other_session')&metrics.scope.eq('All')
    metrics.loc[test,'permutation_q']=bh(metrics.loc[test,'permutation_p'])
    assert test.sum()==len(specs)*2*2
    metrics.to_csv(out/'metrics.csv',index=False)
    np.savez_compressed(out/'permutation_null.npz',**nulls)
    pd.DataFrame(sensitivity).to_csv(out/'one_axis_sensitivity.csv',index=False)
    pd.DataFrame(assignments).to_csv(out/'source_folds.csv',index=False)
    pd.concat(coordinates,ignore_index=True).to_csv(out/'coordinates.csv',index=False)
    pd.DataFrame(associations).to_csv(out/'axis_physiology_anatomy.csv',index=False)
    pd.DataFrame(quartiles).to_csv(out/'axis_quartile_descriptions.csv',index=False)
    metrics=augment_summary_metrics(cfg,out,frame,data,metrics,specs)
    summary=summarize(metrics,specs)
    summary.to_csv(out/'replication_summary.csv',index=False)
    return metrics,summary,snapshots


def augment_summary_metrics(cfg,out,frame,data,metrics,specs):
    uncertainty=Uncertainty(cfg)
    cache={}
    for idx,row in metrics.iterrows():
        key=(int(row.direction),row.feature,row.evaluation)
        if key not in cache:
            with np.load(out/'predictions'/f'd{key[0]}_{key[1]}_{key[2]}.npz',allow_pickle=False) as z:
                cache[key]={name:z[name] for name in z.files}
        saved=cache[key]
        pos=pd.Index(frame.global_unit_id).get_indexer(saved['global_unit_id'])
        assert (pos>=0).all()
        mask=np.ones(len(pos),bool) if row.scope=='All' else (frame.iloc[pos].decoder_label.eq('sua').to_numpy()
            if row.scope=='SUA' else data[row.feature]['supported'][pos])
        window=data[row.feature]['window']
        a=saved['truth'][mask][:,window].mean(1)[:,None]
        b=saved[row.readout][mask][:,window].mean(1)[:,None]
        stats=kernel_metrics(a,b)
        interval=uncertainty.interval(a,b,np.ones(1,bool))
        metrics.loc[idx,'summary_r2']=stats['kernel_r2']
        metrics.loc[idx,'summary_r2_ci_low']=interval['r2_ci_low']
        metrics.loc[idx,'summary_r2_ci_high']=interval['r2_ci_high']
        if row.evaluation=='other_session' and row.scope=='All':
            metrics.loc[idx,'summary_permutation_p']=uncertainty.permutation(a,b)[0]
    primary=metrics.evaluation.eq('other_session')&metrics.scope.eq('All')
    secondary=primary&metrics.kind.eq('kernel')
    combined=np.r_[metrics.loc[primary,'permutation_p'],metrics.loc[secondary,'summary_permutation_p']]
    q=bh(combined)
    metrics.loc[primary,'permutation_q']=q[:primary.sum()]
    metrics.loc[secondary,'summary_permutation_q']=q[primary.sum():]
    metrics.loc[primary&metrics.kind.eq('scalar'),'summary_permutation_q']=metrics.loc[primary&metrics.kind.eq('scalar'),'permutation_q']
    assert len(combined)==108
    metrics.to_csv(out/'metrics.csv',index=False)
    return metrics


def summarize(metrics,specs):
    test=metrics.evaluation.eq('other_session')&metrics.scope.eq('All')
    summary=[]
    for readout in ['feature_axes','original_neighbors']:
        for spec in specs:
            g=metrics[test&metrics.readout.eq(readout)&metrics.feature.eq(spec['name'])].sort_values('direction')
            significant=(g.permutation_q.lt(.05)&g.kernel_pattern_r.gt(0)).to_numpy()
            positive=g.kernel_r2.gt(0).to_numpy()
            status=('Association + positive accuracy both ways' if significant.all() and positive.all()
                else 'Association both ways; mixed accuracy' if significant.all()
                else 'Association one way' if significant.any() else 'No clear association')
            summary_significant=(g.summary_permutation_q.lt(.05)&g.summary_r.gt(0)).to_numpy()
            summary_positive=g.summary_r2.gt(0).to_numpy()
            summary_status=('Association + positive accuracy both ways' if summary_significant.all() and summary_positive.all()
                else 'Association both ways; mixed accuracy' if summary_significant.all()
                else 'Association one way' if summary_significant.any() else 'No clear association')
            row=dict(feature=spec['name'],label=spec['label'],readout=readout,status=status,
                     summary_status=summary_status,derived=spec['derived'])
            for direction in [0,1]:
                r=g.iloc[direction]
                row.update({f'{key}_d{direction}':r[key] for key in ['kernel_r2','kernel_pattern_r','summary_r','summary_r2',
                    'permutation_q','summary_permutation_q','r2_ci_low','r2_ci_high']})
            summary.append(row)
    summary=pd.DataFrame(summary)
    return summary


def plot(cfg,out,specs,data,metrics,snapshots):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[]
    def save(fig,name):
        fig.savefig(out/'figures'/name,dpi=180,bbox_inches='tight')
        plt.close(fig)
        files.append(name)
    names=[s['name'] for s in specs]
    labels=[s['label']+(' †' if s['derived'] else '') for s in specs]
    with plt.rc_context(STYLE):
        for readout,filename in [('feature_axes','feature_axis_transfer.png'),('original_neighbors','embedding_neighbor_transfer.png')]:
            fig,axes=plt.subplots(1,3,figsize=(11,8),layout='constrained')
            for ax,(column,title,limit) in zip(axes,[('kernel_r2','Full-feature R²',.3),('kernel_pattern_r','Full-feature correlation',.65),('summary_r','Mean-effect correlation',.65)]):
                selected=metrics[metrics.readout.eq(readout)&metrics.scope.eq('All')&metrics.evaluation.eq('other_session')]
                matrix=selected.pivot(index='feature',columns='direction',values=column).loc[names,[0,1]].to_numpy()
                im=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
                for row,name in enumerate(names):
                    for direction in [0,1]:
                        value=matrix[row,direction]
                        record=selected[selected.feature.eq(name)&selected.direction.eq(direction)].iloc[0]
                        q=record.summary_permutation_q if column=='summary_r' else record.permutation_q
                        suffix='*' if column!='kernel_r2' and q<.05 else ''
                        ax.text(direction,row,f'{value:.2f}{suffix}',ha='center',va='center',fontsize=9,color='white' if abs(value)>limit*.7 else '#243247')
                ax.set(xticks=[0,1],xticklabels=['Dec 10 → 18','Dec 18 → 10'],yticks=range(len(names)),yticklabels=labels if ax==axes[0] else [],title=title)
                ax.tick_params(axis='both',length=0)
                ax.axhline(11.5,color='white',lw=2)
                fig.colorbar(im,ax=ax,shrink=.45,pad=.02)
            save(fig,filename)

        fig,axes=plt.subplots(1,2,figsize=(8.5,7.6),layout='constrained',sharey=True)
        for ax,(evaluation,title) in zip(axes,[('source_heldout','Held-out source neurons'),('other_session','Frozen → other session')]):
            for direction,color,shift in [(0,'#3478B8',-.11),(1,'#C56C42',.11)]:
                g=metrics[(metrics.readout=='feature_axes')&(metrics.scope=='All')&(metrics.evaluation==evaluation)&(metrics.direction==direction)].set_index('feature').loc[names]
                y=np.arange(len(names))+shift
                ax.hlines(y,g.r2_ci_low,g.r2_ci_high,color=color,lw=1.1)
                ax.scatter(g.kernel_r2,y,color=color,s=23,label='Dec 10 source' if direction==0 else 'Dec 18 source')
            ax.axvline(0,color='#94A3B8',lw=.8)
            ax.set(yticks=range(len(names)),yticklabels=labels,title=title,xlabel='Full-feature R² · 95% interval')
        axes[0].invert_yaxis()
        axes[0].legend(fontsize=8)
        save(fig,'source_and_transfer_accuracy.png')

        for spec in specs:
            name=spec['name']
            item=data[name]
            fig,axes=plt.subplots(2,2,figsize=(8.6,5.8),layout='constrained',sharex=True,sharey=True)
            for direction in [0,1]:
                snap=snapshots[(direction,name)]
                for col,role in enumerate(['source','target']):
                    ax=axes[direction,col]
                    score=snap[role+'_scores'][:,0]
                    truth=snap[role+'_truth']
                    title=snap[role+'_frame'].session.iloc[0][5:10].replace('-','/')
                    ax.set_title(title+(' · discovery' if role=='source' else ' · frozen projection'))
                    if spec['kind']=='kernel':
                        for label,selected,color in [('Low axis 1',score<=np.quantile(score,.25),'#3478B8'),('High axis 1',score>=np.quantile(score,.75),'#C56C42')]:
                            q=np.quantile(truth[selected],[.25,.5,.75],axis=0)
                            ax.plot(item['axis'],q[1],color=color,lw=1.8,label=label)
                            ax.fill_between(item['axis'],q[0],q[2],color=color,alpha=.14,lw=0)
                        ax.axvline(0,color='#CBD5E1',lw=.8)
                        ax.axhline(0,color='#CBD5E1',lw=.8)
                        ax.set(xlabel='Event time (s)' if direction else '',ylabel='GLM log-rate effect' if col==0 else '')
                    else:
                        ax.scatter(score,truth[:,0],s=17,color='#3478B8',alpha=.55)
                        ax.set(xlabel='Feature axis · source SD' if direction else '',ylabel='GLM effect' if col==0 else '')
            if spec['kind']=='kernel':axes[0,0].legend(fontsize=8)
            fig.suptitle(spec['label'],fontsize=12)
            save(fig,f'{name}.png')
    return files


def report(cfg,out,specs,metrics,summary,files,frame):
    method=dict(feature_count=len(specs),independent_design_terms=12,derived_contrasts=4,
        encoder=str(cfg.encoder),sessions=list(cfg.sessions),animal='745502',
        axes='Independent PLS model for each feature. Two components for each full event kernel; one for scalar probability/history coefficients. Unit-normalized embeddings, source-only standardization and axes. One-component kernel sensitivity retained; primary dimension is fixed without target tuning.',
        validation='Five-fold held-out source neurons, refitting scaling and axes each fold. Full-source model frozen before target evaluation. Both transfer directions are analyzed; they are not independent biological replicates.',
        neighbors='Ten nearest source neurons by cosine distance in original embeddings; their full feature curves or scalar values predict target neurons. Source validation excludes held-out neurons from neighbor candidates.',
        metrics='Full-feature R2 uses between-neuron variation about each evaluation session mean at each lag; scalar R2 for non-temporal coefficients. Pattern correlation centers each lag across neurons. Summary correlation uses 0..1s for events or -0.1..0.1s for licks; scalar terms use their coefficient-defined effect.',
        multiplicity='999 whole-neuron feature-row permutations with frozen target predictions. Benjamini-Hochberg jointly over 108 distinct tests: 64 full-feature tests (16 features × 2 directions × 2 readouts) plus 44 prespecified-window kernel-summary tests (11 kernels × 2 directions × 2 readouts). Scalar summaries duplicate their full feature and are not counted twice. Summary significance was added after inspecting full-kernel results, using the already fixed windows; this remains exploratory. Both primary and derived contrasts are included. P-values assume exchangeable neurons, not animal-level replication.',
        uncertainty='1000 neuron-bootstrap intervals conditional on fitted predictions, within each recording; shared fluctuations and uncertainty in fitted source maps are not fully captured.',
        reliability='Parent drop-group positive average gain, at least two positive folds, and convergence; all specified parent groups must pass. These flags apply to groups, not significance of each individual coefficient. Feature-specific CV sign agreement retained separately.',
        limitations=['Only two GLM-covered sessions in these matched fine-tuned exports; December 11 lacks behavioral data.',
            'Same animal, both sessions included in encoder training.','Source/target GLMs differ because movement is unavailable in December 18.',
            'Both single units and multiunits included; SUA evaluation subsets do not refit models or remove MUA source neighbors.',
            'Event kernels are conditional associations, including pre-event terms, not causal effects.',
            'History and probability are non-temporal trial covariates; history contrasts are zero after invalid previous responses.',
            'Derived choice/outcome contrasts are algebraically dependent, not four independent discoveries.',
            'Other coefficients from the same GLM are not independent functional measurements.',
            'No physiology-baseline superiority requirement or residualization was applied.'],
        source_hashes={str(p):sha256(p) for p in [Path(__file__),Path('src/experiments/transfer/active_cue_axes.py'),resolve(cfg.source)/'embeddings.npz',resolve(cfg.targets)]})
    write_json(out/'method.json',method)
    primary=summary[summary.readout.eq('feature_axes')]
    coverage=frame.groupby('session').agg(n_units=('global_unit_id','size'),n_sua=('decoder_label',lambda s:int(s.eq('sua').sum()))).reset_index()
    coverage.to_csv(out/'coverage.csv',index=False)
    write_json(out/'summary.json',dict(coverage=coverage.to_dict('records'),features=primary.to_dict('records')))
    display=primary[['label','status','summary_status','kernel_r2_d0','kernel_r2_d1','summary_r_d0','summary_r_d1']]
    lookup=primary.set_index('feature')
    outcome,probability,history=lookup.loc['outcome'],lookup.loc['probability_left'],lookup.loc['history_side']
    findings=(f'<p><b>Main findings:</b> the average 0–1 s reward-minus-omission effect has transfer correlations '
        f'{outcome.summary_r_d0:.2f} (Dec 10 → 18) and {outcome.summary_r_d1:.2f} (reverse). '
        'Both remain positive in single-unit evaluations. This concerns early response strength: the much later outcome response differs substantially between sessions, and full-curve predictions fail. '
        f'Left reward-probability effects show smaller recurring correlations ({probability.summary_r_d0:.2f}, {probability.summary_r_d1:.2f}), '
        f'as do previous-choice effects ({history.summary_r_d0:.2f}, {history.summary_r_d1:.2f}). '
        'These associations survive the exploratory joint multiplicity correction.</p>'
        '<p>Licking shows recurring temporal organization, especially for the left lick. A short average around the lick misses this structure. '
        'Absolute prediction is asymmetric: left-lick full-kernel R² is positive for Dec 10 → 18 and negative in reverse. '
        'Ordinary right-versus-left choice preference, rewarded-trial side preference, and previous valid-response effects do not show positive transfer. '
        'Unrewarded-trial side preference has a weak recurring association; this is a derived contrast, not an independent feature discovery.</p>'
        '<p>No full temporal kernel has positive transfer R² in both directions. The useful result is recurring functional organization with imperfect transfer of response shape and magnitude. '
        'Early outcome response is the clearest next feature to inspect with original event-aligned firing, cluster membership, recorded labels, and anatomy. '
        'Programmed reward probability is not an inferred expectation or reward prediction error.</p>')
    body=['<!doctype html><meta charset="utf-8"><title>Individual GLM features across sessions</title>',
        '<style>body{font:16px system-ui;line-height:1.5;color:#243247;max-width:1150px;margin:36px auto;padding:0 24px}img{max-width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border-bottom:1px solid #ddd}a{color:#226ba5}p{max-width:1050px}</style>',
        '<h1>Individual GLM features across sessions</h1>',
        '<p>Animal 745502 · December 10 and 18 · fine-tuned embeddings. Each GLM feature gets its own source-fitted axes. Both transfer directions are shown. Original-embedding neighbor prediction is a complementary view.</p>',
        '<p>Full temporal kernels remain separate. Probability and trial history are scalar coefficients, not temporal kernels. Movement is excluded because December 18 lacks it; December 11 cannot enter because behavioral data are absent.</p>',
        findings,coverage.to_html(index=False,border=0),
        '<p><b>Reading the results:</b> positive R² means better absolute prediction than the evaluation session’s own mean feature. Positive correlation with negative R² indicates an association without accurate calibration. Asterisks mark exploratory q&lt;0.05 after all 108 distinct full-feature and fixed-window comparisons. † marks algebraically derived contrasts. Source discovery figures are descriptive; source held-out metrics and frozen target predictions provide validation.</p>',
        display.to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}'),
        '<p>Columns ending d0 mean Dec 10 → 18; d1 means Dec 18 → 10. These are two sessions of one animal already seen by the encoder. The reverse analysis adds a consistency check, not independent replication. Neuron-level intervals and permutation tests do not establish animal-level generalization.</p>']
    for name in files:
        body.append(f'<img src="figures/{name}" alt="{html.escape(name)}">')
        if name.endswith('transfer.png'):
            body.append('<p>Colors saturate at the displayed limits; numeric values are untruncated. Correlations for both full features and fixed-window summaries are included in the joint multiplicity correction. Kernel metrics concern variation between neurons, not explained spike-count variance.</p>')
        elif name not in ['source_and_transfer_accuracy.png']:
            body.append('<p>Source axes are fit to this feature only. Curves: median and middle 50% of low/high first-axis quartiles defined within each session. Scalars: individual unit effects. Rows reverse the training direction; column 2 always uses frozen source axes.</p>')
    body.append('<h2>Reusable outputs</h2><ul>')
    for name in ['replication_summary.csv','metrics.csv','coordinates.csv','unit_features.csv','axis_quartile_descriptions.csv',
                 'axis_physiology_anatomy.csv','one_axis_sensitivity.csv','feature_definitions.json','method.json','features.npz',
                 'recording_availability.csv','excluded_features.json','figures_and_tables.zip']:
        body.append(f'<li><a href="{name}">{name}</a></li>')
    body.append('</ul>')
    (out/'index.html').write_text('\n'.join(body)+'\n')
    write_json(out/'integrity_checks.json',dict(n_units=len(frame),unique_unit_ids=bool(frame.global_unit_id.is_unique),
        source_folds_disjoint=True,source_predictions_complete=True,serialized_models_checked=32,
        full_kernel_coefficient_reconstruction=True,n_target_multiplicity_tests=108))
    with zipfile.ZipFile(out/'figures_and_tables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ['.csv','.png','.json','.html','.yaml','.npz']:
                archive.write(path,path.relative_to(out))


def run(cfg):
    out=resolve(cfg.output)
    out.mkdir(parents=True,exist_ok=True)
    specs=definitions()
    if cfg.get('stage','all')=='report':
        frame=read_table(out/'units.csv')
        features=read_table(out/'unit_features.csv')
        with np.load(out/'features.npz',allow_pickle=False) as z:
            assert list(z['global_unit_id'])==frame.global_unit_id.tolist()
            data={}
            for spec in specs:
                name=spec['name']
                axis=z[name+'_axis']
                window=(axis>=spec['summary_s'][0]-1e-8)&(axis<=spec['summary_s'][1]+1e-8) if spec['kind']=='kernel' else np.ones(1,bool)
                reliable=features[features.feature.eq(name)].set_index('global_unit_id').loc[frame.global_unit_id].parent_group_supported.to_numpy(bool)
                data[name]=dict(spec=spec,y=z[name],cv=z[name+'_validation'],axis=axis,window=window,supported=reliable)
        metrics=augment_summary_metrics(cfg,out,frame,data,read_table(out/'metrics.csv'),specs)
        summary=summarize(metrics,specs)
        summary.to_csv(out/'replication_summary.csv',index=False)
        coords=read_table(out/'coordinates.csv')
        snapshots={}
        for direction in [0,1]:
            for spec in specs:
                name=spec['name']
                snap={}
                with np.load(out/'models'/f'd{direction}_{name}.npz',allow_pickle=False) as z:
                    snap['state']={key:z[key] for key in z.files}
                for role,evaluation in [('source','source_heldout'),('target','other_session')]:
                    with np.load(out/'predictions'/f'd{direction}_{name}_{evaluation}.npz',allow_pickle=False) as z:
                        part=frame.set_index('global_unit_id',drop=False).loc[z['global_unit_id']].reset_index(drop=True)
                        snap[role+'_frame']=part
                        snap[role+'_truth'],snap[role+'_prediction']=z['truth'],z['feature_axes']
                    sub=coords[(coords.feature==name)&(coords.direction==direction)&(coords.role==role)].set_index('global_unit_id').loc[part.global_unit_id]
                    snap[role+'_scores']=sub[['axis_1','axis_2']].to_numpy()
                snapshots[(direction,name)]=snap
        files=plot(cfg,out,specs,data,metrics,snapshots)
        report(cfg,out,specs,metrics,summary,files,frame)
        print('Refreshed saved results:',out,flush=True)
        return
    OmegaConf.save(cfg,out/'config.yaml')
    frame,x,data=load(cfg,out,specs)
    print(f'Loaded {len(frame)} units, {len(specs)} individual features.',flush=True)
    metrics,summary,snapshots=run_models(cfg,out,frame,x,data,specs)
    files=plot(cfg,out,specs,data,metrics,snapshots)
    report(cfg,out,specs,metrics,summary,files,frame)
    print(summary[summary.readout.eq('feature_axes')].to_string(index=False),flush=True)
    print('Completed:',out,flush=True)
