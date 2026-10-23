"""Fixed-cluster choice responses and source-only across-recording readouts.

The primary target is the outcome-averaged right-minus-left GLM contrast.
No cell labels, physiology controls, target GLM fitting, or encoder updates.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, normalize

from src.embeddings.io import read_table
from src.data.io import resolve
from src.common.analysis import correlation, sha256, write_json
from src.common.plotting import STYLE

CONTRASTS = ['right_minus_left', 'right_minus_left_reward', 'right_minus_left_omission']
TITLES = ['Outcome averaged', 'Rewarded', 'Unrewarded']


def load_targets(cfg, units, out):
    targets=read_table(cfg.targets)
    assert units.global_unit_id.is_unique and targets.global_unit_id.is_unique
    targets=targets[targets.analysis_included.astype(str).str.lower().eq('true')]
    columns=['global_unit_id','final_file','gain_full','gain_choice_side']
    joined=units.merge(targets[columns],on='global_unit_id',validate='one_to_one')
    joined=joined[joined.session.isin([cfg.source_session,cfg.target_session])].reset_index(drop=True)
    reliability=read_table(resolve(cfg.glm)/'reliability.csv')
    reliability=reliability[reliability.group.eq('Choice side')]
    reliability=reliability.rename(columns={'n_folds':'choice_cv_folds','positive_folds':'choice_positive_folds',
                                          'all_converged':'choice_cv_converged'})
    joined=joined.merge(reliability[['global_unit_id','choice_cv_folds','choice_positive_folds','choice_cv_converged']],
                        on='global_unit_id',how='left',validate='one_to_one')
    final,cv,kernels=[],[],[]
    schema={}
    for row in joined.itertuples():
        path=resolve(cfg.glm)/row.final_file
        recording=path.parent.parent
        if recording not in schema:
            info=json.loads((recording/'prepared/design_info.json').read_text())
            assert 'side (+0.5 left)' in info['choice_outcome']
            with np.load(recording/'basis.npz',allow_pickle=False) as z:
                schema[recording]=(info,z['choice_side'].copy(),z['side_outcome'].copy())
        info,side_basis,interaction_basis=schema[recording]
        with np.load(path,allow_pickle=False) as z:
            lag=z['lags_s']
            names=list(z['kernel_names'])
            side=z['kernels'][names.index('choice_side')]
            interaction=z['kernels'][names.index('side_outcome')]
            contrast=np.stack([-side,-side-.5*interaction,-side+.5*interaction])
            assert np.allclose(contrast[1],z['kernels'][names.index('right_reward')]-z['kernels'][names.index('left_reward')])
            assert np.allclose(contrast[2],z['kernels'][names.index('right_omission')]-z['kernels'][names.index('left_omission')])
            final.append(contrast)
            kernels.append(z['kernels'][:7].copy())
            assert np.allclose(side,side_basis@z['coefficients'][info['slices']['choice_side']])
        with np.load(recording/'validation/units'/path.name,allow_pickle=False) as z:
            coefficients=z['coefficients']
            side=coefficients[:,info['slices']['choice_side']]@side_basis.T
            interaction=coefficients[:,info['slices']['side_outcome']]@interaction_basis.T
            cv.append(np.stack([-side,-side-.5*interaction,-side+.5*interaction],axis=1))
    final,cv=np.stack(final),np.stack(cv)
    assert np.isfinite(final).all() and np.isfinite(cv).all()
    window=(lag>=float(cfg.effect_window_s[0])-1e-8)&(lag<=float(cfg.effect_window_s[1])+1e-8)
    effects=final[:,:,window].mean(2)
    cv_effects=cv[:,:,:,window].mean(3)
    for j,name in enumerate(CONTRASTS):
        joined[name]=effects[:,j]
        joined[name+'_cv_min']=cv_effects[:,:,j].min(1)
        joined[name+'_cv_max']=cv_effects[:,:,j].max(1)
        joined[name+'_cv_sign_agreement']=(np.sign(cv_effects[:,:,j])==np.sign(effects[:,[j]])).mean(1)
    joined['choice_supported']=joined.gain_choice_side.gt(0)&joined.choice_positive_folds.ge(2)&joined.choice_cv_converged.eq(True)
    joined['direction_supported']=joined.choice_supported&joined.right_minus_left_cv_sign_agreement.eq(1)
    joined.to_csv(out/'unit_features.csv',index=False)
    np.savez_compressed(out/'observed_curves.npz',global_unit_id=joined.global_unit_id.to_numpy(str),lags_s=lag,
                        contrast_names=np.asarray(CONTRASTS),curves=final,validation_curves=cv,
                        original_kernel_names=np.asarray(['cue','choice_common','choice_side','outcome','side_outcome','left_lick','right_lick']),
                        original_kernels=np.stack(kernels))
    write_json(out/'target_definition.json',dict(primary='right_minus_left',window_s=list(cfg.effect_window_s),
        units='Mean conditional log-rate contrast, right minus left, equally averaging reward and omission conditions',
        formula='-choice_side; reward: -choice_side - 0.5*side_outcome; omission: -choice_side + 0.5*side_outcome',
        coding='Side +0.5 left/-0.5 right; outcome +0.5 reward/-0.5 omission. Registration time is the choice/outcome event.',
        reliability='choice_supported: positive held-out Choice side gain in >=2/3 folds and all folds converged; direction_supported additionally requires all three validation mean contrasts to agree with final-fit sign. Descriptive flags, not significance or posterior probabilities.',
        inclusion='Converged full fit and three converged Full validation folds. No positive-gain or sign selection for primary training/evaluation.',
        limitations=['Predicts a neuron response property, not individual-trial mouse choices.','A post-registration side contrast is not evidence of a causal choice computation.',
                    'Source GLM includes movement; target GLM omits unavailable movement.','All selected recordings contributed to representation training.',
                    'Final GLM and embedding windows overlap and need not have identical temporal coverage.','Outcome-related responses are not labeled as reward prediction errors.']))
    return joined,final,cv,lag,window


def transform_fit(x,space):
    base=normalize(x)
    center=base.mean(0) if space=='session_centered' else np.zeros(x.shape[1])
    transformed=normalize(base-center) if space=='session_centered' else base
    scaler=StandardScaler().fit(transformed)
    return scaler.transform(transformed),(center,scaler)


def transform_apply(x,state,space,session_center=None):
    center,scaler=state
    base=normalize(x)
    if space=='session_centered':
        base=normalize(base-(center if session_center is None else session_center))
    return scaler.transform(base)


def tune(x,y,space,cfg,seed):
    splits=list(KFold(int(cfg.inner_folds),shuffle=True,random_state=seed).split(x))
    rows=[]
    for penalty in cfg.ridge_penalties:
        errors=[]
        for tr,te in splits:
            train,state=transform_fit(x[tr],space)
            test=transform_apply(x[te],state,space)
            model=Ridge(alpha=float(penalty)*len(tr),solver='svd').fit(train,y[tr])
            errors.extend(np.square(y[te]-model.predict(test)))
        rows.append(dict(penalty=float(penalty),mse=float(np.mean(errors))))
    # Prefer the stronger penalty only for exact ties.
    best=min(rows,key=lambda r:(r['mse'],-r['penalty']))['penalty']
    return best,rows


def centroid_transfer(source_values,target_values,clusters,source_responses,source_indices,space,quantile):
    """Whole-source unlabeled geometry; source GLM responses only for prototypes."""
    source=normalize(source_values)
    target=normalize(target_values)
    if space=='session_centered':
        source=normalize(source-source.mean(0))
        target=normalize(target-target.mean(0))
    ids=np.unique(clusters)
    centroids=normalize(np.stack([source[clusters==c].mean(0) for c in ids]))
    radii=[]
    profiles=[]
    for c in ids:
        members=source[clusters==c]
        loo=normalize(members.sum(0)[None,:]-members)
        radii.append(float(np.quantile(1-np.sum(members*loo,axis=1),quantile)))
        available=clusters[source_indices]==c
        profiles.append(source_responses[available].mean(0) if available.any() else np.full(source_responses.shape[1:],np.nan))
    distance=np.maximum(cdist(target,centroids,'cosine'),0)
    closest=distance.argmin(1)
    within=distance[np.arange(len(target)),closest]<=np.asarray(radii)[closest]
    return np.stack(profiles)[closest],ids[closest],within,distance[np.arange(len(target)),closest],np.asarray(radii)[closest]


def summarize_predictions(frame,cfg):
    rows=[]
    rng=np.random.default_rng(int(cfg.seed))
    for keys,g in frame.groupby(['encoder','space','evaluation','readout','contrast'],sort=True):
        selections={'All':np.ones(len(g),bool),'SUA':g.decoder_label.eq('sua').to_numpy(),
                    'Direction supported':g.direction_supported.to_numpy(bool)}
        if keys[3]=='cluster_profile': selections['Within source radius']=g.within_source_radius.to_numpy(bool)
        for scope,mask in selections.items():
            part=g.loc[mask].dropna(subset=['truth','prediction'])
            a,b=part.truth.to_numpy(),part.prediction.to_numpy()
            if len(a)<5: continue
            r=correlation(a,b)
            draws=rng.integers(len(a),size=(int(cfg.bootstrap_samples),len(a)))
            aa=a[draws]-a[draws].mean(1,keepdims=True)
            bb=b[draws]-b[draws].mean(1,keepdims=True)
            boot=np.sum(aa*bb,axis=1)/np.maximum(np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1),1e-12)
            low,high=np.quantile(boot,[.025,.975])
            variance=np.square(a-a.mean()).sum()
            rows.append(dict(zip(['encoder','space','evaluation','readout','contrast'],keys),scope=scope,n_units=len(a),
                pearson=r,pearson_ci_low=low,pearson_ci_high=high,
                spearman=float(spearmanr(a,b).statistic) if np.std(b)>1e-12 else np.nan,
                r2=1-float(np.square(a-b).sum()/variance) if variance>0 else np.nan,
                rmse=float(np.sqrt(np.mean(np.square(a-b)))),mean_bias=float(np.mean(b-a))))
    return pd.DataFrame(rows)


def plot_and_report(cfg,out,features,curves,cv,lag,window,snapshots,predictions,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator
    primary=snapshots[(str(cfg.primary_encoder),'original')]
    palette=plt.get_cmap('tab10').colors
    figures=[]
    def save(fig,name):
        fig.savefig(out/name,dpi=190,bbox_inches='tight')
        plt.close(fig)
        figures.append(name)
    cluster_rows=[]
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
        jitter=np.random.default_rng(int(cfg.seed))
        for column,(which,title) in enumerate([('source','December 10'),('target','December 18')]):
            frame=primary[which+'_frame']
            cls=primary[which+'_clusters']
            ids=np.unique(cls)
            values=[primary[which+'_truth'][:,0,window].mean(1),100*frame.gain_choice_side.to_numpy()]
            for row,metric in enumerate(values):
                ax=axes[row,column]
                for c in ids:
                    y=metric[cls==c];y=y[np.isfinite(y)]
                    x=int(c)+jitter.uniform(-.17,.17,len(y))
                    ax.scatter(x,y,s=11,c=[palette[int(c)%10]],alpha=.45,edgecolors='none')
                    if len(y):
                        q1,med,q3=np.quantile(y,[.25,.5,.75])
                        ax.plot([c,c],[q1,q3],c='#334155',lw=1.5)
                        ax.plot([c-.16,c+.16],[med,med],c='#0F172A',lw=2)
                ax.axhline(0,c='#CBD5E1',lw=.8)
                ax.set(xticks=ids,xticklabels=[f'C{int(c)+1}' for c in ids],
                       ylabel='Right − left log-rate effect' if row==0 else 'Choice-side gain (%)')
                if row==0:ax.set_title(title)
                else:ax.set_yscale('symlog',linthresh=.01)
        save(fig,'cluster_choice_strength_and_direction.png')
        for which,title in [('source','December 10'),('target','December 18')]:
            frame=primary[which+'_frame']
            cls=primary[which+'_clusters']
            truth=primary[which+'_truth']
            ids=np.unique(cls)
            fig,axes=plt.subplots(len(ids),3,figsize=(10.5,1.65*len(ids)),sharex=True,squeeze=False,layout='constrained')
            for row,c in enumerate(ids):
                part=truth[cls==c]
                low,median,high=np.quantile(part,[.25,.5,.75],axis=0)
                lower=min(0,float(low.min()))
                upper=max(0,float(high.max()))
                pad=max((upper-lower)*.08,.03)
                for column in range(3):
                    ax=axes[row,column]
                    ax.plot(lag,median[column],c=palette[int(c)%10],lw=1.7)
                    ax.fill_between(lag,low[column],high[column],color=palette[int(c)%10],alpha=.18)
                    ax.axhline(0,c='#CBD5E1',lw=.7)
                    ax.axvline(0,c='#94A3B8',lw=.7)
                    ax.set_ylim(lower-pad,upper+pad)
                    ax.yaxis.set_major_locator(MaxNLocator(3))
                    if row==0: ax.set_title(TITLES[column])
                    if column==0: ax.set_ylabel(f'C{int(c)+1} · n={len(part)}')
                    if row==len(ids)-1: ax.set_xlabel('Time from choice (s)')
                means=part[:,:,window].mean(2)
                for j,name in enumerate(CONTRASTS):
                    group=frame.loc[cls==c]
                    cluster_rows.append(dict(session=str(group.session.iloc[0]),cluster=int(c),contrast=name,n=len(group),
                        median=float(np.median(means[:,j])),n_positive=int((means[:,j]>0).sum()),
                        n_negative=int((means[:,j]<0).sum()),n_direction_supported=int(group.direction_supported.sum()),
                        n_sua=int(group.decoder_label.eq('sua').sum()),n_d1=int(group.label.eq('D1').sum()),n_d2=int(group.label.eq('D2').sum())))
            fig.suptitle(title+' · right − left',fontsize=13)
            save(fig,f'cluster_choice_kernels_{which}.png')
            effect=truth[:,0,window].mean(1)
            order=np.lexsort((effect,cls))
            limit=max(float(np.quantile(np.abs(truth[:,0]),.97)),.05)
            fig,axes=plt.subplots(1,2,figsize=(8,6.6),gridspec_kw={'width_ratios':[12,1]},layout='constrained')
            im=axes[0].imshow(truth[order,0],aspect='auto',origin='upper',cmap='RdBu_r',vmin=-limit,vmax=limit,
                              extent=[lag[0],lag[-1],len(frame)-.5,-.5])
            starts=np.r_[0,np.flatnonzero(np.diff(cls[order]))+1]
            ends=np.r_[starts[1:],len(order)]
            for end in ends[:-1]: axes[0].axhline(end-.5,c='white',lw=.8)
            axes[0].set(yticks=(starts+ends-1)/2,yticklabels=[f'C{int(cls[order[s]])+1}' for s in starts],
                        xlabel='Time from choice (s)',title=title+' · individual neurons')
            axes[0].axvline(0,c='#64748B',lw=.8)
            axes[1].imshow(frame.direction_supported.to_numpy()[order,None],aspect='auto',origin='upper',
                           cmap=matplotlib.colors.ListedColormap(['#E2E8F0','#059669']),vmin=0,vmax=1)
            axes[1].set(xticks=[],yticks=[],title='Supported',frame_on=False)
            fig.colorbar(im,ax=axes[0],label='Right − left log-rate effect',shrink=.7,extend='both')
            save(fig,f'choice_neurons_{which}.png')
        fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
        for ax,which,title in zip(axes,['source','target'],['December 10 · held-out units','December 18 · frozen readout']):
            truth=primary[which+'_truth'][:,0,window].mean(1)
            estimate=primary[which+'_prediction'][:,0,window].mean(1)
            cls=primary[which+'_clusters']
            ax.scatter(estimate,truth,s=17,c=[palette[int(c)%10] for c in cls],alpha=.6,edgecolors='none')
            low=min(truth.min(),estimate.min()); high=max(truth.max(),estimate.max())
            ax.plot([low,high],[low,high],ls='--',c='#CBD5E1',lw=1)
            ax.axhline(0,c='#E2E8F0',lw=.7);ax.axvline(0,c='#E2E8F0',lw=.7)
            ax.set(title=title,xlabel='Predicted right − left',ylabel='GLM right − left')
            ax.text(.04,.94,f'r = {correlation(truth,estimate):.2f}',transform=ax.transAxes,va='top')
        save(fig,'choice_transfer.png')
        # Shared PCA coordinates are fitted on source embeddings only, not GLM responses.
        projection=PCA(n_components=2,svd_solver='full').fit(primary['source_design'])
        source_xy=projection.transform(primary['source_design']);target_xy=projection.transform(primary['target_design'])
        maximum=max(float(np.quantile(np.abs(curves[:,0,window].mean(1)),.95)),.05)
        norm=TwoSlopeNorm(vmin=-maximum,vcenter=0,vmax=maximum)
        fig,axes=plt.subplots(2,3,figsize=(11,6.4),layout='constrained')
        for row,(which,xy) in enumerate([('source',source_xy),('target',target_xy)]):
            cls=primary[which+'_clusters']
            truth=primary[which+'_truth'][:,0,window].mean(1)
            estimate=primary[which+'_prediction'][:,0,window].mean(1)
            axes[row,0].scatter(xy[:,0],xy[:,1],s=17,c=[palette[int(c)%10] for c in cls],alpha=.8,edgecolors='none')
            for col,values in [(1,truth),(2,estimate)]:
                im=axes[row,col].scatter(xy[:,0],xy[:,1],s=17,c=values,cmap='RdBu_r',norm=norm,edgecolors='none')
            for col in range(3):
                axes[row,col].set(xlabel='Source PC1',ylabel='Source PC2')
                axes[row,col].set_title(('December 10' if row==0 else 'December 18')+' · '+['clusters','GLM effect','prediction'][col])
        fig.colorbar(im,ax=axes[:,1:],label='Right − left log-rate effect',shrink=.7,extend='both')
        save(fig,'embedding_choice_views.png')
        fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
        for ax,which,title in zip(axes,['source','target'],['December 10 · held-out units','December 18 · frozen readout']):
            estimate=primary[which+'_prediction'][:,0,window].mean(1)
            truth=primary[which+'_truth'][:,0]
            # Quartiles use predictions only, never observed response values.
            lo,hi=np.quantile(estimate,[.25,.75])
            for mask,label,color in [(estimate<=lo,'Lower predicted contrast','#2563EB'),(estimate>=hi,'Higher predicted contrast','#DC2626')]:
                median=np.median(truth[mask],axis=0)
                ax.plot(lag,median,c=color,label=f'{label} · n={mask.sum()}')
            ax.axvline(0,c='#94A3B8',lw=.7);ax.axhline(0,c='#CBD5E1',lw=.7)
            ax.set(title=title,xlabel='Time from choice (s)',ylabel='Median GLM right − left')
            ax.legend(fontsize=8)
        save(fig,'predicted_groups_observed_kernels.png')
    cluster_rows=pd.DataFrame(cluster_rows)
    cluster_rows.to_csv(out/'primary_cluster_description.csv',index=False)
    primary_predictions=predictions[(predictions.encoder==cfg.primary_encoder)&predictions.space.eq('original')&
                                   predictions.readout.eq('ridge')&predictions.contrast.eq(CONTRASTS[0])].copy()
    primary_predictions=primary_predictions.merge(features[['global_unit_id','gain_choice_side','right_minus_left_cv_min',
                    'right_minus_left_cv_max','ccf_acronym','ccf_ap','ccf_dv','ccf_ml']],on='global_unit_id',validate='one_to_one')
    primary_predictions['absolute_effect']=primary_predictions.truth.abs()
    primary_predictions.sort_values(['direction_supported','absolute_effect'],ascending=False).to_csv(out/'neurons_to_inspect.csv',index=False)
    coverage=features.groupby('session').agg(n=('global_unit_id','size'),choice_supported=('choice_supported','sum'),
               direction_supported=('direction_supported','sum'),n_sua=('decoder_label',lambda s:int(s.eq('sua').sum())))
    coverage.to_csv(out/'coverage.csv')
    primary_metrics=metrics[(metrics.encoder==cfg.primary_encoder)&metrics.space.eq('original')&metrics.contrast.eq(CONTRASTS[0])]
    write_json(out/'summary.json',dict(primary_encoder=str(cfg.primary_encoder),source=str(cfg.source_session),target=str(cfg.target_session),
        source_neurons=int(features.session.eq(cfg.source_session).sum()),target_neurons=int(features.session.eq(cfg.target_session).sum()),
        primary_metrics=primary_metrics.to_dict('records'),coverage=coverage.reset_index().to_dict('records'),figures=figures))
    page='''<!doctype html><html><head><meta charset="utf-8"><title>Active choice-response organization</title>
<style>body{font:16px system-ui;background:#f6f8fb;color:#243247;padding:32px}main{max-width:1150px;margin:auto}section{background:white;padding:24px;border-radius:14px;margin:22px 0}img{max-width:100%}p{line-height:1.6}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}a{color:#047857}</style></head><body><main>
<h1>Active clusters and choice responses</h1><section><p>The first direction in the analysis plan connects fixed unsupervised clusters to task responses. This pilot uses the fine-tuned encoder as primary and the two other saved encoders as sensitivities. All analyses use existing GLM fits and embeddings; no neuron labels or physiological measurements define the functional mapping.</p>
<p>The target is the mean 0–1 s conditional right-minus-left log-rate contrast around registered choice, equally averaging reward and omission conditions. Positive values indicate a larger modeled right-side effect, conditional on the other GLM predictors. This describes a neuron's response property; it does not decode trial choices or establish a causal decision signal. Rewarded and unrewarded contrasts and their full curves remain separate.</p>'''+coverage.to_html(border=0)+'''</section>
<section><h2>Clusters and individual neurons</h2><img src="cluster_choice_strength_and_direction.png"><p>Dots are individual neurons; black marks show medians and interquartile ranges. The gain axis uses a symmetric log scale. Choice-side gain measures the held-out contribution of both side and side-by-outcome terms. A neuron can therefore carry choice-related information without a consistent outcome-averaged side preference. Cluster IDs are local to each recording. Curves show the median and across-unit interquartile range, not confidence intervals. Individual-neuron heatmaps show every included unit, ordered by cluster and mean contrast, with display colors clipped at the 97th percentile of absolute effects. Green “Supported” marks a positive held-out side gain in at least two folds and a final effect direction reproduced in all three validation fits; this is a descriptive reliability flag.</p>
<img src="cluster_choice_kernels_source.png"><img src="choice_neurons_source.png">
<img src="cluster_choice_kernels_target.png"><img src="choice_neurons_target.png"></section>
<section><h2>Source-to-target prediction</h2><p>Source: December 10. Target: December 18, the same animal (745502). Ridge scaling and regularization use source data only. December 10 predictions use nested held-out-unit folds; December 18 predictions use a mapping fitted to all eligible source neurons. The second readout transfers source-cluster response profiles through nearest centroid matches, flagging target units outside a source-only 95th-percentile radius. The target's GLM values never choose matches, transforms, or penalties.</p>
<img src="choice_transfer.png"><img src="predicted_groups_observed_kernels.png">
<p>Higher/lower groups are the top/bottom prediction quartiles within each recording, defined without its observed GLM contrasts. These plots describe the observed full kernels behind scalar prediction results; they are not independent additional tests. Table intervals bootstrap neurons, conditional on these recordings; they are not animal-level confidence intervals and may be optimistic when neurons covary.</p>'''+primary_metrics.round(4).to_html(index=False,border=0)+'''
<p>R² measures absolute prediction accuracy relative to the target's mean; correlation measures ordering. A positive correlation with negative R² can indicate weak ordering with poor calibration. Neither metric is a requirement that a cluster outperform simple physiology to be scientifically interesting.</p></section>
<section><h2>Embedding views</h2><img src="embedding_choice_views.png"><p>Both recordings use the same source-fitted PCA axes of the high-dimensional readout input. Panels use their own axis limits. Two-dimensional projection can omit functional structure. Predictions are made in the full embedding representation. Equal cluster colors across recordings do not establish group identity.</p></section>
<section><h2>Scope and limitations</h2><p>This is a retrospective across-session readout test in one animal. The encoder was trained using both recordings, and earlier analyses inspected these data. Embeddings and GLMs use overlapping neural recordings; source held-out units are not independent representation training data. The source GLM includes a movement covariate that is unavailable in the target GLM. Session centering is a predefined sensitivity using unlabeled target embeddings; its results must remain separate from the original-space result. No target-GLM alignment or physiological residualization is performed. A reward prediction error would require a separately defined expectation model and GLM predictor; no RPE claim is made here.</p>
<p><a href="neurons_to_inspect.csv">Individual neurons and reliability</a> · <a href="primary_cluster_description.csv">Cluster descriptions</a> · <a href="metrics.csv">All readout and encoder results</a> · <a href="cluster_units.csv">Cluster membership and metadata</a> · <a href="predictions.csv">All scalar predictions</a> · <a href="target_definition.json">Feature definition</a> · <a href="method.json">Methods</a> · <a href="integrity_checks.json">Validation</a> · <a href="figures_and_tables.zip">Download figures and tables</a></p></section></main></body></html>'''
    (out/'index.html').write_text(page)


def validate(cfg,out,snapshots,features,window,predictions):
    units=read_table(resolve(cfg.source)/'units.csv')
    all_ids=pd.Index(units.global_unit_id)
    with np.load(resolve(cfg.source)/'embeddings.npz',allow_pickle=False) as z:
        assert np.array_equal(z['global_unit_id'].astype(str),all_ids.to_numpy(str))
        for (encoder,space),snapshot in snapshots.items():
            with np.load(out/'models'/f'{encoder}_{space}.npz',allow_pickle=False) as saved:
                train_ids=set(saved['training_ids'])
                assert train_ids==set(snapshot['source_frame'].global_unit_id)
                assert not train_ids&set(snapshot['target_frame'].global_unit_id)
                pos=all_ids.get_indexer(snapshot['target_frame'].global_unit_id)
                assert (pos>=0).all()
                # Independent reconstruction directly from the serialized readout.
                x=z[encoder][pos].astype(float)
                x=x/np.linalg.norm(x,axis=1,keepdims=True)
                if space=='session_centered':
                    x=x-saved['target_center']
                    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
                x=(x-saved['scale_mean'])/saved['scale']
                estimate=(x@saved['coefficients'].T+saved['intercept']).reshape(snapshot['target_prediction'].shape)
                assert np.allclose(estimate,snapshot['target_prediction'],rtol=1e-7,atol=1e-9)
                target_predictions=predictions[(predictions.encoder==encoder)&predictions.space.eq(space)&
                                               predictions.evaluation.eq('Other session')&predictions.readout.eq('ridge')]
                for j,name in enumerate(CONTRASTS):
                    part=target_predictions[target_predictions.contrast.eq(name)].set_index('global_unit_id').loc[snapshot['target_frame'].global_unit_id]
                    assert np.allclose(part.prediction,estimate[:,j,window].mean(1))
                    assert np.allclose(part.truth,snapshot['target_truth'][:,j,window].mean(1))
    fold_table=pd.read_csv(out/'source_folds.csv')
    assert fold_table.overlap.eq(0).all()
    assert features.global_unit_id.is_unique
    expected=len(features)*len(cfg.encoders)*len(cfg.spaces)*2*3
    assert len(predictions)==expected
    assert not predictions.duplicated(['global_unit_id','encoder','space','readout','contrast']).any()
    write_json(out/'integrity_checks.json',dict(status='passed',n_unique_neurons=len(features),n_prediction_rows=len(predictions),
        n_serialized_models_reconstructed=len(snapshots),n_source_folds_checked=len(fold_table),
        checks=['Embedding row identity checked before and after fitting','Full contrast signs checked against explicit right/left derived GLM curves',
                'Final side kernels reconstructed from original-unit coefficients and basis','Every serialized model contains source-only training IDs',
                'All target readout predictions independently reconstructed from saved coefficients and transforms',
                'All scalar predictions reconciled with saved full time courses','Outer source training/test indices disjoint',
                'Target labels, anatomy, physiology and GLM values are not inputs to fitting, penalty selection or embedding correspondence']))
    with zipfile.ZipFile(out/'figures_and_tables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.suffix in ['.png','.html','.csv','.json','.yaml']:archive.write(path,path.name)


def run(cfg):
    out=resolve(cfg.output)
    out.mkdir(parents=True,exist_ok=True)
    saved=out/'config.yaml'
    if saved.exists(): assert OmegaConf.to_container(OmegaConf.load(saved))==OmegaConf.to_container(cfg)
    OmegaConf.save(cfg,saved,resolve=True)
    source=resolve(cfg.source)
    units=read_table(source/'units.csv').reset_index(drop=True)
    units['_embedding_row']=np.arange(len(units))
    with np.load(source/'embeddings.npz',allow_pickle=False) as z:
        assert np.array_equal(z['global_unit_id'].astype(str),units.global_unit_id.to_numpy(str))
        matrices={encoder:z[encoder].astype(float) for encoder in cfg.encoders}
    memberships=read_table(source/'clusters.csv')
    features,curves,cv,lag,window=load_targets(cfg,units,out)
    source_mask=features.session.eq(cfg.source_session).to_numpy()
    target_mask=features.session.eq(cfg.target_session).to_numpy()
    source_frame=features[source_mask].reset_index(drop=True)
    target_frame=features[target_mask].reset_index(drop=True)
    assert not set(source_frame.global_unit_id)&set(target_frame.global_unit_id)
    y_source=curves[source_mask]
    y_target=curves[target_mask]
    flat_source=y_source.reshape(len(y_source),-1)
    scalar=y_source[:,0,window].mean(1)
    predictions,fold_rows,tuning_rows,model_rows,cluster_summary=[],[],[],[],[]
    snapshots={}
    cluster_tables=[]
    out.joinpath('models').mkdir(exist_ok=True)
    out.joinpath('predictions').mkdir(exist_ok=True)
    for encoder in cfg.encoders:
        membership=memberships[memberships.encoder.eq(encoder)].set_index('global_unit_id').cluster
        cls=features.global_unit_id.map(membership).to_numpy(int)
        assert (cls>=0).all()
        cluster_tables.append(features.assign(encoder=encoder,cluster=cls))
        for session,g in features.assign(cluster=cls).groupby('session'):
            for c,group in g.groupby('cluster'):
                for contrast in CONTRASTS:
                    cluster_summary.append(dict(encoder=encoder,session=session,cluster=int(c),contrast=contrast,n_units=len(group),
                        median=float(group[contrast].median()),mean=float(group[contrast].mean()),
                        q25=float(group[contrast].quantile(.25)),q75=float(group[contrast].quantile(.75)),
                        n_choice_supported=int(group.choice_supported.sum()),n_direction_supported=int(group.direction_supported.sum()),
                        median_choice_gain=float(group.gain_choice_side.median()),n_sua=int(group.decoder_label.eq('sua').sum())))
        x_source=matrices[encoder][source_frame._embedding_row]
        x_target=matrices[encoder][target_frame._embedding_row]
        all_source=units[units.session.eq(cfg.source_session)]
        all_target=units[units.session.eq(cfg.target_session)]
        all_source_x=matrices[encoder][all_source._embedding_row]
        all_target_x=matrices[encoder][all_target._embedding_row]
        all_source_clusters=all_source.global_unit_id.map(membership).to_numpy(int)
        source_positions=pd.Index(all_source.global_unit_id).get_indexer(source_frame.global_unit_id)
        target_positions=pd.Index(all_target.global_unit_id).get_indexer(target_frame.global_unit_id)
        assert (source_positions>=0).all() and (target_positions>=0).all()
        for space in cfg.spaces:
            print(f'Choice transfer: {encoder}, {space}',flush=True)
            oof_ridge=np.full_like(flat_source,np.nan)
            oof_cluster=np.full_like(flat_source,np.nan)
            folds=np.full(len(source_frame),-1,int)
            for fold,(tr,te) in enumerate(KFold(int(cfg.folds),shuffle=True,random_state=int(cfg.seed)).split(x_source)):
                penalty,detail=tune(x_source[tr],scalar[tr],space,cfg,int(cfg.seed)+fold+1)
                train,state=transform_fit(x_source[tr],space)
                test=transform_apply(x_source[te],state,space)
                model=Ridge(alpha=penalty*len(tr),solver='svd').fit(train,flat_source[tr])
                oof_ridge[te]=model.predict(test)
                for c in np.unique(cls[source_mask]):
                    members=tr[cls[source_mask][tr]==c]
                    test_members=te[cls[source_mask][te]==c]
                    oof_cluster[test_members]=flat_source[members].mean(0) if len(members) else flat_source[tr].mean(0)
                folds[te]=fold
                fold_rows.append(dict(encoder=encoder,space=space,fold=fold,n_train=len(tr),n_test=len(te),
                                      overlap=len(set(tr)&set(te)),penalty=penalty))
                for row in detail:tuning_rows.append(dict(encoder=encoder,space=space,fit=f'outer_{fold}',**row))
            assert np.isfinite(oof_ridge).all() and np.isfinite(oof_cluster).all() and (folds>=0).all()
            penalty,detail=tune(x_source,scalar,space,cfg,int(cfg.seed)+101)
            for row in detail:tuning_rows.append(dict(encoder=encoder,space=space,fit='source_full',**row))
            train,state=transform_fit(x_source,space)
            target_center=normalize(all_target_x).mean(0) if space=='session_centered' else None
            target_design=transform_apply(x_target,state,space,target_center)
            model=Ridge(alpha=penalty*len(x_source),solver='svd').fit(train,flat_source)
            target_ridge=model.predict(target_design).reshape(y_target.shape)
            profiles,matches,supported,distance,radii=centroid_transfer(all_source_x,all_target_x,all_source_clusters,
                 y_source,source_positions,space,float(cfg.source_radius_quantile))
            profiles=profiles[target_positions]
            matches,supported,distance,radii=[v[target_positions] for v in [matches,supported,distance,radii]]
            model_rows.append(dict(encoder=encoder,space=space,penalty=penalty,n_source=len(x_source),n_target=len(x_target),
                                   n_target_within_source_radius=int(supported.sum())))
            np.savez_compressed(out/'models'/f'{encoder}_{space}.npz',coefficients=model.coef_,intercept=model.intercept_,
                center=state[0],scale_mean=state[1].mean_,scale=state[1].scale_,
                target_center=target_center if target_center is not None else np.zeros(x_target.shape[1]),
                training_ids=source_frame.global_unit_id.to_numpy(str),penalty=np.asarray(penalty))
            snapshot=dict(source_frame=source_frame,target_frame=target_frame,source_clusters=cls[source_mask],target_clusters=cls[target_mask],
                source_truth=y_source,target_truth=y_target,source_prediction=oof_ridge.reshape(y_source.shape),
                target_prediction=target_ridge,target_cluster_prediction=profiles,source_design=train,target_design=target_design,
                matched_source_cluster=matches,supported=supported)
            snapshots[(encoder,space)]=snapshot
            for evaluation,frame,truth,ridge,prototype,clustering in [
                ('Source held-out units',source_frame,y_source,oof_ridge.reshape(y_source.shape),oof_cluster.reshape(y_source.shape),cls[source_mask]),
                ('Other session',target_frame,y_target,target_ridge,profiles,cls[target_mask])]:
                np.savez_compressed(out/'predictions'/f'{encoder}_{space}_{"source" if evaluation.startswith("Source") else "target"}.npz',
                    global_unit_id=frame.global_unit_id.to_numpy(str),lags_s=lag,truth=truth,ridge=ridge,cluster_profile=prototype,
                    clusters=clustering,fold=folds if evaluation.startswith('Source') else np.full(len(frame),-1),
                    matched_source_cluster=matches if evaluation=='Other session' else clustering,
                    within_source_radius=supported if evaluation=='Other session' else np.ones(len(frame),bool))
                for readout,estimate in [('ridge',ridge),('cluster_profile',prototype)]:
                    for j,name in enumerate(CONTRASTS):
                        part=frame[['global_unit_id','unit_id','animal','session','label','decoder_label','direction_supported','choice_supported']].copy()
                        part['encoder'],part['space'],part['evaluation'],part['readout'],part['contrast']=encoder,space,evaluation,readout,name
                        part['cluster']=clustering
                        part['truth']=truth[:,j,window].mean(1)
                        part['prediction']=estimate[:,j,window].mean(1)
                        part['within_source_radius']=supported if evaluation=='Other session' else True
                        part['matched_source_cluster']=matches if evaluation=='Other session' else clustering
                        part['fold']=folds if evaluation.startswith('Source') else -1
                        predictions.append(part)
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_csv(out/'predictions.csv',index=False)
    metrics=summarize_predictions(predictions,cfg)
    metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'source_folds.csv',index=False)
    pd.DataFrame(tuning_rows).to_csv(out/'source_tuning.csv',index=False)
    pd.DataFrame(model_rows).to_csv(out/'models.csv',index=False)
    pd.DataFrame(cluster_summary).to_csv(out/'cluster_features.csv',index=False)
    pd.concat(cluster_tables,ignore_index=True).to_csv(out/'cluster_units.csv',index=False)
    write_json(out/'method.json',dict(primary_encoder=str(cfg.primary_encoder),primary_space='original',primary_readout='ridge',
        primary_target='right_minus_left',direction=f'{cfg.source_session} to {cfg.target_session}',
        readout='Unit-normalized embeddings, source-fitted standardization, ridge regression. Penalty selected only by source 3-fold MSE for primary 0..1s contrast; same penalty predicts all three full contrast curves.',
        source_validation='Nested 5-fold held-out-unit predictions. Embeddings and fixed clusters were learned using these recordings; source GLM response values excluded from each readout test fold.',
        cluster_transfer='Nearest cosine source cluster centroid. Profile uses source GLM values only. Target membership has source-only 95th-percentile leave-one-out radius flag; no forced matches in supported subset.',
        centered_sensitivity='Subtract source-training mean or entire unlabeled target-session mean from normalized vectors, renormalize, then source-scale. Target centering is transductive and uses no GLM values. Cluster centroids use whole-source unlabeled geometry.',
        uncertainty='1000 neuron bootstrap intervals, conditional on these recordings and fitted mappings. Not animal-level intervals; correlated neurons can make intervals optimistic.',
        scope='No baseline-superiority criterion, physiology residualization, new GLM fitting, encoder updates, or target-GLM alignment.',
        source_hashes={str(p):sha256(p) for p in [source/'units.csv',source/'embeddings.npz',source/'clusters.csv',resolve(cfg.targets),resolve(cfg.glm)/'schema.json',Path(__file__)]}))
    plot_and_report(cfg,out,features,curves,cv,lag,window,snapshots,predictions,metrics)
    validate(cfg,out,snapshots,features,window,predictions)
    print(metrics[(metrics.encoder==cfg.primary_encoder)&(metrics.space=='original')&(metrics.contrast==CONTRASTS[0])].to_string(index=False),flush=True)
    print('Completed:',out,flush=True)
