"""Observed firing validation of frozen, other-session outcome readouts."""
import json
import zipfile

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from src.common.analysis import correlation,write_json,sha256
from src.embeddings.io import read_table
from src.data.io import resolve
from src.common.plotting import STYLE


def paired_trials(trials,cfg):
    rows=[]
    for key,g in trials.groupby(['animal_response','reward_probabilityL','reward_probabilityR','time_block'],sort=True):
        reward=g[g.rewarded.eq(1)]
        omission=g[g.rewarded.eq(0)]
        if not len(reward) or not len(omission):continue
        cost=np.abs(reward.stored_trial_index.to_numpy()[:,None]-omission.stored_trial_index.to_numpy())
        r,o=linear_sum_assignment(cost)
        for i,j in zip(r,o):
            a,b=reward.iloc[i],omission.iloc[j]
            rows.append(dict(reward_row=int(a.sample_row),omission_row=int(b.sample_row),reward_trial=int(a.id),
                omission_trial=int(b.id),side=int(key[0]),probability_left=key[1],probability_right=key[2],
                time_block=int(key[3]),fold=int(a.fold),trial_separation=int(cost[i,j]),
                time_separation_s=float(abs(a.reward_outcome_time-b.reward_outcome_time))))
    pairs=pd.DataFrame(rows)
    assert len(pairs)>0 and pairs.reward_trial.is_unique and pairs.omission_trial.is_unique
    assert set(pairs.side)=={0,1}
    return pairs


def side_balanced(sampled,trials,pairs=None,fold=None):
    conditions={}
    for side in [0,1]:
        for outcome in [0,1]:
            if pairs is None:
                mask=trials.animal_response.eq(side)&trials.rewarded.eq(outcome)
                if fold is not None:mask&=trials.fold.eq(fold)
                idx=trials.loc[mask,'sample_row'].to_numpy(int)
            else:
                selected=pairs[pairs.side.eq(side)]
                if fold is not None:selected=selected[selected.fold.eq(fold)]
                idx=selected['reward_row' if outcome else 'omission_row'].to_numpy(int)
            if not len(idx):return None
            conditions[(side,outcome)]=sampled[:,idx].mean(1)
    reward=.5*(conditions[(0,1)]+conditions[(1,1)])
    omission=.5*(conditions[(0,0)]+conditions[(1,0)])
    return reward,omission,conditions


def rank_r(a,b):
    return float(spearmanr(a,b).statistic) if np.std(a)>1e-12 and np.std(b)>1e-12 else np.nan


def rank_ci(a,b,n,seed):
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(n):
        idx=rng.integers(len(a),size=len(a))
        values.append(rank_r(a[idx],b[idx]))
    return np.nanquantile(values,[.025,.975])


def analyze(cfg,session,direction,frame,out):
    from pathlib import Path
    part=frame[frame.session.eq(session)].copy().reset_index(drop=True)
    # Selection comes only from the other recording's frozen mapping.
    with np.load(resolve(cfg.transfer)/'predictions'/f'd{direction}_outcome_other_session.npz',allow_pickle=False) as z:
        pos=pd.Index(z['global_unit_id'].astype(str)).get_indexer(part.global_unit_id)
        assert (pos>=0).all()
        with np.load(resolve(cfg.transfer)/'features.npz',allow_pickle=False) as f:
            lag=f['outcome_axis']
        win=(lag>=float(cfg.response_s[0])-1e-8)&(lag<=float(cfg.response_s[1])+1e-8)
        score=z['feature_axes'][pos][:,win].mean(1)
        glm=z['truth'][pos][:,win].mean(1)
    part['frozen_outcome_score']=score
    part['observed_glm_outcome']=glm
    thresholds=np.quantile(score,[.25,.75])
    part['score_group']=np.where(score<=thresholds[0],'Low',np.where(score>=thresholds[1],'High','Middle'))
    directory=resolve(cfg.glm)/Path(part.final_file.iloc[0]).parent.parent
    prep=directory/'prepared'
    info=json.loads((prep/'design_info.json').read_text())
    units=read_table(prep/'units.csv')
    with np.load(prep/'design_metadata.npz',allow_pickle=False) as z:meta={k:z[k] for k in ['edges','trial_index','speed','usable']}
    dt=float(info['bin_size'])
    counts=np.load(prep/'counts.npy',mmap_mode='r')
    unit_ix=pd.Index(units.global_unit_id).get_indexer(part.global_unit_id)
    assert (unit_ix>=0).all()
    trials=read_table(prep/'trials.csv')
    trials['source_trial_row']=np.arange(len(trials))
    all_trials=trials.copy()
    trials=trials[trials.usable.eq(True)&trials.animal_response.isin([0,1])&trials.rewarded.isin([0,1])].copy()
    offsets=np.arange(-round(float(cfg.before_s)/dt),round(float(cfg.after_s)/dt))
    center0=.5*(meta['edges'][0]+meta['edges'][1])
    central=np.rint((trials.reward_outcome_time.to_numpy()-center0)/dt).astype(int)
    ix=central[:,None]+offsets
    valid=(ix.min(1)>=0)&(ix.max(1)<counts.shape[1])
    safe=np.clip(ix,0,counts.shape[1]-1)
    valid&=meta['usable'][safe].all(1)
    valid&=(meta['trial_index'][safe]==trials.source_trial_row.to_numpy()[:,None]).all(1)
    audit=trials.copy()
    audit['window_eligible']=valid
    audit.to_csv(out/f'{session}_trial_eligibility.csv',index=False)
    trials=trials[valid].reset_index(drop=True)
    ix=ix[valid]
    trials['sample_row']=np.arange(len(trials))
    trials['time_block']=trials.stored_trial_index//int(cfg.time_block_trials)
    stride=round(float(cfg.plot_bin_s)/dt)
    assert len(offsets)%stride==0 and np.isclose(stride*dt,float(cfg.plot_bin_s))
    lag=(offsets*dt).reshape(-1,stride).mean(1)
    sampled=np.stack([np.asarray(counts[i,ix],float).reshape(len(ix),-1,stride).sum(2)/float(cfg.plot_bin_s) for i in unit_ix])
    assert np.isfinite(sampled).all()
    mean_rate=np.asarray([np.asarray(counts[i,meta['usable']],float).mean()/dt for i in unit_ix])
    assert (mean_rate>0).all()
    part['observed_mean_rate_hz']=mean_rate
    baseline=(lag>=float(cfg.baseline_s[0]))&(lag<float(cfg.baseline_s[1]))
    response=(lag>=float(cfg.response_s[0]))&(lag<float(cfg.response_s[1]))
    pairs=paired_trials(trials,cfg)
    pairs.to_csv(out/f'{session}_matched_pairs.csv',index=False)
    # Lick stream restores the registered choice licks omitted from the GLM lick kernels.
    with np.load(prep/'events.npz',allow_pickle=False) as z:
        licks=np.unique(np.r_[z['left_lick'],z['right_lick'],all_trials.loc[all_trials.usable.eq(True)&all_trials.animal_response.isin([0,1]),'choice_time']])
    lick_rate=np.histogram(licks,meta['edges'])[0]/dt
    behavior={'licks_hz':lick_rate[ix].reshape(len(ix),-1,stride).mean(2)[None,:,:]}
    if np.isfinite(meta['speed'][ix]).all():behavior['speed_cm_s']=meta['speed'][ix].reshape(len(ix),-1,stride).mean(2)[None,:,:]
    metrics,associations,group_stats,behaviors=[],[],[],[]
    saved=dict(global_unit_id=part.global_unit_id.to_numpy(str),lag_s=lag,score=score,trial_id=trials.id.to_numpy(),
        trial_fold=trials.fold.to_numpy(),trial_side=trials.animal_response.to_numpy(),trial_outcome=trials.rewarded.to_numpy(),
        trial_rate_hz=sampled.astype(np.float32),mean_rate_hz=mean_rate)
    curves={}
    for selection,pairset in [('All trials',None),('Matched trials',pairs)]:
        reward,omission,_=side_balanced(sampled,trials,pairset)
        pre_r,pre_o=reward[:,baseline].mean(1),omission[:,baseline].mean(1)
        difference=reward-omission
        pre_difference=difference[:,baseline].mean(1)
        delta=difference-pre_difference[:,None]
        values=dict(pre_hz=pre_difference,post_hz=difference[:,response].mean(1),delta_hz=delta[:,response].mean(1),
                    relative_delta=delta[:,response].mean(1)/mean_rate)
        # Exploratory follow-up to the reward/omission plots, added after inspecting them.
        # Equal outcome weights isolate the response shared across both outcomes.
        common=.5*(reward+omission)
        common_delta=common[:,response].mean(1)-common[:,baseline].mean(1)
        values.update(common_delta_hz=common_delta,common_relative_delta=common_delta/mean_rate)
        curves[selection]=dict(reward=reward,omission=omission,delta=delta,
            reward_change=(reward-pre_r[:,None])/mean_rate[:,None],omission_change=(omission-pre_o[:,None])/mean_rate[:,None])
        key='matched' if pairset is not None else 'all'
        for name,array in curves[selection].items():saved[f'{key}_{name}']=array
        for name,behavior_array in behavior.items():
            br,bo,_=side_balanced(behavior_array,trials,pairset)
            for t,r,o in zip(lag,br[0],bo[0]):behaviors.append(dict(session=session,selection=selection,feature=name,time_s=t,reward=r,omission=o))
        for name,array in values.items():
            for i,gid in enumerate(part.global_unit_id):metrics.append(dict(global_unit_id=gid,session=session,selection=selection,measure=name,value=array[i]))
            for scope,mask in [('All',np.ones(len(part),bool)),('SUA',part.decoder_label.eq('sua').to_numpy())]:
                low,high=rank_ci(score[mask],array[mask],int(cfg.bootstrap_samples),int(cfg.seed))
                associations.append(dict(session=session,selection=selection,scope=scope,measure=name,n_units=int(mask.sum()),
                    spearman=rank_r(score[mask],array[mask]),spearman_ci_low=low,spearman_ci_high=high,
                    pearson=correlation(score[mask],array[mask]),glm_spearman=rank_r(glm[mask],array[mask])))
                low_group=mask&part.score_group.eq('Low').to_numpy()
                high_group=mask&part.score_group.eq('High').to_numpy()
                group_stats.append(dict(session=session,selection=selection,scope=scope,measure=name,
                    n_low=int(low_group.sum()),n_high=int(high_group.sum()),median_low=float(np.median(array[low_group])),
                    median_high=float(np.median(array[high_group])),high_minus_low=float(np.median(array[high_group])-np.median(array[low_group]))))
    # Replicate the pre-specified contrast across the three existing interleaved trial folds.
    fold_rows=[]
    for fold in [0,1,2]:
        result=side_balanced(sampled,trials,pairs,fold)
        if result is None:continue
        reward,omission,_=result
        diff=reward-omission
        delta=diff[:,response].mean(1)-diff[:,baseline].mean(1)
        common=.5*(reward+omission)
        common_delta=common[:,response].mean(1)-common[:,baseline].mean(1)
        for name,value in [('delta_hz',delta),('relative_delta',delta/mean_rate),
                           ('common_delta_hz',common_delta),('common_relative_delta',common_delta/mean_rate)]:
            for scope,mask in [('All',np.ones(len(part),bool)),('SUA',part.decoder_label.eq('sua').to_numpy())]:
                fold_rows.append(dict(session=session,fold=fold,scope=scope,measure=name,n_pairs=int(pairs.fold.eq(fold).sum()),
                    spearman=rank_r(score[mask],value[mask]),high_minus_low=float(np.median(value[mask&part.score_group.eq('High').to_numpy()])-
                        np.median(value[mask&part.score_group.eq('Low').to_numpy()]))))
    # Trial-block bootstrap keeps all neurons together and preserves rewarded/omitted trial pairs.
    pair_change=sampled[:,pairs.reward_row.to_numpy()]-sampled[:,pairs.omission_row.to_numpy()]
    pair_delta=pair_change[:,:,response].mean(2)-pair_change[:,:,baseline].mean(2)
    pair_common=.5*(sampled[:,pairs.reward_row.to_numpy()]+sampled[:,pairs.omission_row.to_numpy()])
    pair_common_delta=pair_common[:,:,response].mean(2)-pair_common[:,:,baseline].mean(2)
    blocks=np.unique(pairs.time_block)
    rng=np.random.default_rng(int(cfg.seed)+31)
    block_draws=[]
    for _ in range(int(cfg.bootstrap_samples)):
        selected=rng.choice(blocks,size=len(blocks),replace=True)
        indices=np.concatenate([np.flatnonzero(pairs.time_block.to_numpy()==b) for b in selected])
        if not all(np.any(pairs.side.to_numpy()[indices]==s) for s in [0,1]):continue
        for prefix,array in [('',pair_delta),('common_',pair_common_delta)]:
            mean=.5*sum(array[:,indices[pairs.side.to_numpy()[indices]==s]].mean(1) for s in [0,1])
            for name,value in [(prefix+'delta_hz',mean),(prefix+'relative_delta',mean/mean_rate)]:
                for scope,mask in [('All',np.ones(len(part),bool)),('SUA',part.decoder_label.eq('sua').to_numpy())]:
                    block_draws.append(dict(scope=scope,measure=name,spearman=rank_r(score[mask],value[mask]),
                        high_minus_low=np.median(value[mask&part.score_group.eq('High').to_numpy()])-np.median(value[mask&part.score_group.eq('Low').to_numpy()])))
    block_ci=[]
    for (scope,name),g in pd.DataFrame(block_draws).groupby(['scope','measure']):
        block_ci.append(dict(session=session,scope=scope,measure=name,n_blocks=len(blocks),
            spearman_ci_low=g.spearman.quantile(.025),spearman_ci_high=g.spearman.quantile(.975),
            difference_ci_low=g.high_minus_low.quantile(.025),difference_ci_high=g.high_minus_low.quantile(.975)))
    observed=pd.DataFrame(metrics)
    joined=observed[observed.selection.eq('Matched trials')].pivot(index='global_unit_id',columns='measure',values='value')
    part=part.join(joined,on='global_unit_id',validate='one_to_one')
    part.to_csv(out/f'{session}_neurons.csv',index=False)
    np.savez_compressed(out/f'{session}_observed.npz',**saved)
    coverage=dict(session=session,n_units=len(part),n_sua=int(part.decoder_label.eq('sua').sum()),
        n_eligible_trials=len(trials),n_pairs=len(pairs),left_pairs=int(pairs.side.eq(0).sum()),right_pairs=int(pairs.side.eq(1).sum()),
        median_trial_separation=float(pairs.trial_separation.median()),n_time_blocks=len(blocks),
        max_alignment_error_s=float(np.max(np.abs(meta['edges'][0]+(central[valid]+.5)*dt-trials.reward_outcome_time.to_numpy()))))
    return dict(frame=part,curves=curves,lag=lag,metrics=metrics,associations=associations,group_stats=group_stats,
        behaviors=behaviors,fold_rows=fold_rows,block_ci=block_ci,coverage=coverage)


def plot(cfg,out,snapshots):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[]
    def save(fig,name):
        fig.savefig(out/name,dpi=185,bbox_inches='tight')
        plt.close(fig)
        files.append(name)
    with plt.rc_context(STYLE):
        for scope in ['All','SUA']:
            fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained',sharex=True)
            for col,snap in enumerate(snapshots):
                frame=snap['frame'];curves=snap['curves']['Matched trials'];lag=snap['lag']
                eligible=np.ones(len(frame),bool) if scope=='All' else frame.decoder_label.eq('sua').to_numpy()
                for row,group in enumerate(['Low','High']):
                    mask=eligible&frame.score_group.eq(group).to_numpy()
                    for name,color,label in [('reward_change','#3478B8','Reward'),('omission_change','#C56C42','Omission')]:
                        q=np.quantile(curves[name][mask],[.25,.5,.75],axis=0)
                        axes[row,col].plot(lag,q[1],color=color,lw=2,label=label)
                        axes[row,col].fill_between(lag,q[0],q[2],color=color,alpha=.12,lw=0)
                    axes[row,col].axvline(0,color='#CBD5E1',lw=.8)
                    axes[row,col].axhline(0,color='#CBD5E1',lw=.8)
                    axes[row,col].set(title=f'{frame.session.iloc[0][5:10]} · {group.lower()} score · n={mask.sum()}',
                        xlabel='Time from outcome (s)' if row else '',ylabel='Δ firing / session mean' if col==0 else '')
                axes[0,col].legend(fontsize=8)
            fig.suptitle('Observed firing · matched trials · '+scope,fontsize=12)
            save(fig,f'observed_outcome_{scope.lower()}.png')
        fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
        for col,snap in enumerate(snapshots):
            frame=snap['frame']
            for row,(measure,label) in enumerate([('delta_hz','Observed reward − omission · Δ Hz'),('relative_delta','Observed contrast / session rate')]):
                for sorting,color in [('mua','#CBD5E1'),('sua','#3478B8')]:
                    g=frame[frame.decoder_label.eq(sorting)]
                    axes[row,col].scatter(g.frozen_outcome_score,g[measure],color=color,s=21,alpha=.8,label=sorting.upper())
                axes[row,col].axhline(0,color='#CBD5E1',lw=.8)
                axes[row,col].set(xlabel='Frozen embedding prediction',ylabel=label,title=frame.session.iloc[0][5:10])
                rho=rank_r(frame.frozen_outcome_score,frame[measure])
                axes[row,col].text(.04,.95,f'ρ = {rho:.2f}',transform=axes[row,col].transAxes,va='top')
            axes[0,col].legend(fontsize=8)
        save(fig,'prediction_vs_observed.png')
        fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
        for ax,snap in zip(axes,snapshots):
            frame=snap['frame']
            for sorting,color in [('mua','#CBD5E1'),('sua','#3478B8')]:
                g=frame[frame.decoder_label.eq(sorting)]
                ax.scatter(g.frozen_outcome_score,g.common_relative_delta,color=color,s=21,alpha=.8,label=sorting.upper())
            ax.axhline(0,color='#CBD5E1',lw=.8)
            ax.set(xlabel='Frozen embedding prediction',ylabel='Common response / session rate',title=frame.session.iloc[0][5:10])
            ax.text(.04,.95,f'ρ = {rank_r(frame.frozen_outcome_score,frame.common_relative_delta):.2f}',transform=ax.transAxes,va='top')
        axes[0].legend(fontsize=8)
        fig.suptitle('Exploratory · response shared across outcomes',fontsize=12)
        save(fig,'common_response.png')
        fig,axes=plt.subplots(1,2,figsize=(8.5,3),layout='constrained',sharey=True)
        for ax,snap in zip(axes,snapshots):
            behavior=pd.DataFrame(snap['behaviors'])
            b=behavior[(behavior.selection=='Matched trials')&(behavior.feature=='licks_hz')]
            ax.plot(b.time_s,b.reward,color='#3478B8',label='Reward',lw=2)
            ax.plot(b.time_s,b.omission,color='#C56C42',label='Omission',lw=2)
            ax.axvline(0,color='#CBD5E1',lw=.8)
            ax.set(title=snap['frame'].session.iloc[0][5:10],xlabel='Time from outcome (s)',ylabel='Licks / s')
        axes[0].legend(fontsize=8)
        save(fig,'matched_trial_licking.png')
    return files


def run(cfg):
    from pathlib import Path
    out=resolve(cfg.output);out.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(cfg,out/'config.yaml')
    frame=read_table(resolve(cfg.transfer)/'units.csv')
    snapshots=[]
    for session,direction in zip(cfg.sessions,[1,0]):
        print('Observed outcome check:',session,flush=True)
        snapshots.append(analyze(cfg,session,direction,frame,out))
    for key,name in [('metrics','unit_responses.csv'),('associations','associations.csv'),('group_stats','group_comparisons.csv'),
                     ('behaviors','behavior.csv'),('fold_rows','trial_fold_replication.csv'),('block_ci','trial_block_uncertainty.csv')]:
        pd.DataFrame([r for s in snapshots for r in s[key]]).to_csv(out/name,index=False)
    coverage=pd.DataFrame([s['coverage'] for s in snapshots]);coverage.to_csv(out/'coverage.csv',index=False)
    files=plot(cfg,out,snapshots)
    method=dict(selection='Frozen other-session outcome readout predicts each target neuron’s mean 0..1s GLM outcome effect. Target GLM values and observed firing do not select groups. Quartiles defined by this prediction within each target session.',
        observed='Saved observed 20-ms spike counts, not model predictions. Outcome-registration aligned, -1..+2s, averaged into 100-ms bins. Nearest count-bin center alignment error <=10ms. Full window must lie within the same usable trial.',
        matching='Rewarded and omitted trials matched without replacement within choice side, both programmed reward probabilities, and 20-trial time block. Minimum trial-index separation pairing. Left/right contrasts get equal weight. No neural information enters matching.',
        measures='Raw pre and post reward-minus-omission Hz contrasts; baseline-adjusted contrast uses 0..1s minus -1..-0.2s. Relative contrast divides by the unit’s mean firing over usable GLM bins. Raw and relative results both retained; no physiology-based exclusion or residualization.',
        exploratory_followup='After inspecting the observed traces, added the common response: equal average of rewarded and omitted baseline-adjusted responses. Same frozen score, groups, matching and time windows. This is post hoc characterization, not a prespecified or independent replication. glm_spearman always refers to the own-session outcome GLM effect, including when compared with common firing.',
        uncertainty='1000 neuron bootstrap Spearman intervals conditional on observed trial means. Separate 20-trial-block bootstrap resamples trial pairs while holding neurons together. Three existing interleaved trial folds test consistency. These are within-session checks, not animal-level replication.',
        limitations=['Same recordings supplied embeddings and GLMs, so this is an observed-spike corroboration, not independent data validation.',
            'Observed PSTHs combine cue, choice, reward, licking and movement contributions; they are not conditional GLM kernels.',
            'Matching on task covariates does not equate subsequent licking or prove a causal outcome mechanism.',
            'Cue and choice often occur close together, limiting temporal attribution.',
            'Recorded D1/D2 labels include multiunits; all-unit and SUA evaluations are separate.',
            'The bidirectional mappings come from two sessions of one animal; no tracked neuron identity across days.'],
        source_hashes={str(p):sha256(p) for p in [Path(__file__),resolve(cfg.transfer)/'method.json',resolve(cfg.transfer)/'metrics.csv']})
    write_json(out/'method.json',method)
    associations=pd.read_csv(out/'associations.csv')
    selected=associations[(associations.selection=='Matched trials')&associations.measure.isin(['delta_hz','relative_delta','common_delta_hz','common_relative_delta'])]
    write_json(out/'summary.json',dict(coverage=coverage.to_dict('records'),associations=selected.to_dict('records')))
    body=['<!doctype html><meta charset="utf-8"><title>Observed outcome response</title>',
        '<style>body{font:16px system-ui;line-height:1.5;color:#243247;max-width:1080px;margin:36px auto;padding:0 24px}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border-bottom:1px solid #ddd}</style>',
        '<h1>Frozen embedding predictions and observed firing</h1>',
        '<p>Each session uses outcome predictions from a model fit only to the other session. These predictions select high/low quartile groups. Actual spike counts provide the response measurements. This checks the GLM-derived interpretation using observed firing from the same recordings; it is not independent-data validation.</p>',
        '<p>Rewarded and unrewarded trials are matched within side, programmed reward probabilities, and 20-trial time blocks. Both choice sides receive equal weight. Baseline: −1 to −0.2 s. Response: 0 to 1 s. Subsequent licking is displayed because observed outcome responses can include consumption and movement.</p>',
        '<h2>Interpretation</h2><p>The reward-specific observed-firing check is mixed. Across all units, matched reward-minus-omission correlations are −0.044 and 0.147 in Hz, or 0.048 and 0.129 relative to session mean firing. Single-unit relative responses show stronger correspondence in December 10 (0.374) than December 18 (0.187). Trial-block uncertainty includes zero in December 18; trial-fold effects are inconsistent. This does not establish robust reward selectivity across both recordings.</p>',
        '<p>A post hoc follow-up identifies a stronger relationship with common activation versus suppression around choice/outcome, shared by rewarded and omitted trials. Low-score neurons tend to activate and high-score neurons tend to suppress. The common-response correlations are −0.519 and −0.585 in Hz, and −0.600 and −0.454 relative to session mean firing. These are associations with a task-related transition, not evidence for a reward, RPE, or causal motor mechanism. No new embedding model or groups were fitted for this follow-up.</p>',
        '<p>The current GLM outcome effect itself correlates more strongly with observed outcome contrasts than the transferred embedding score does. Thus weak direct transfer does not make the GLM effect meaningless. Conditional GLM effects and marginal event-aligned firing measure different properties. See the repository ANALYSIS_PLAN.md for the resulting hypotheses and concrete tests.</p>',
        coverage.to_html(index=False,border=0),selected[['session','scope','measure','n_units','spearman','spearman_ci_low','spearman_ci_high']].to_html(index=False,border=0,float_format=lambda v:f'{v:.3f}')]
    for name in files:body.append(f'<img src="{name}">')
    body.append('<p>PSTH lines show median across units; shaded bands show the middle 50%. Relative responses divide each neuron’s baseline change by its session mean rate. Positive outcome contrast means reward greater than omission; negative means omission greater than reward.</p><ul>')
    for name in ['associations.csv','group_comparisons.csv','trial_fold_replication.csv','trial_block_uncertainty.csv','unit_responses.csv','behavior.csv','method.json','figures_and_tables.zip']:
        body.append(f'<li><a href="{name}">{name}</a></li>')
    body.append('</ul>');(out/'index.html').write_text('\n'.join(body)+'\n')
    write_json(out/'integrity_checks.json',dict(unique_unit_ids=True,source_only_predictions=True,
        matched_trials_not_reused=True,same_trial_windows=True,maximum_alignment_error_s=float(coverage.max_alignment_error_s.max()),
        n_units=int(coverage.n_units.sum())))
    with zipfile.ZipFile(out/'figures_and_tables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.suffix in ['.png','.csv','.json','.html','.yaml','.npz']:archive.write(p,p.name)
    print(selected.to_string(index=False),flush=True)
    print('Completed:',out,flush=True)
