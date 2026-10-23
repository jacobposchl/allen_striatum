"""Behavior-only, full-session design for an exploratory blocked-CV GLM."""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import SplineTransformer
from src.data.recording import Recording
from src.data.task import recording_path, audit_trials, selected_units, binned_speed, source_fingerprint
from src.glm.design import event_design, temporal_basis


KERNELS = [('cue', -1., 2., 7), ('choice_common', -1., 2., 7),
           ('choice_side', -1., 2., 7), ('outcome', -1., 2., 7),
           ('side_outcome', -1., 2., 7), ('left_lick', -.4, .6, 6),
           ('right_lick', -.4, .6, 6)]


def guarded(mask, centers, radius):
    """Erode each continuous selected interval, leaving a gap at every boundary."""
    indices = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    out = np.zeros(len(mask), bool)
    for a, b in indices.reshape(-1, 2):
        out[a:b] = (centers[a:b] >= centers[a] + radius) & (centers[a:b] <= centers[b-1] - radius)
    return out


def prepare(cfg, output):
    path = recording_path(cfg)
    recording = Recording(path)
    trials, task_info = audit_trials(recording, int(cfg.block_trials), strict_reward_timing=True, allow_no_opto=bool(cfg.get('allow_no_opto', False)))
    clock = task_info['clock']; origin = clock['origin_s']
    opto = task_info['opto_start_s']; extra_deliveries = task_info['extra_reward_deliveries']
    sides = [recording.events(f'{side}_lick') - origin for side in ['left', 'right']]
    valid = trials.usable
    stop = float(trials.loc[valid,'stop_time'].max())
    dt = float(cfg.bin_size); edges = np.arange(int(np.ceil(stop/dt))+1)*dt; centers=(edges[1:]+edges[:-1])/2
    ti = np.searchsorted(trials.start_time.to_numpy(),edges[:-1],side='right')-1
    safe=ti.clip(0)
    usable=(ti>=0)&valid.to_numpy()[safe]&(edges[1:]<=trials.stop_time.to_numpy()[safe])
    for row in trials.loc[~valid].itertuples():
        usable &= (centers<row.start_time-1)|(centers>row.stop_time+2)
    t=trials.loc[valid]; chosen=t.loc[t.animal_response.isin([0,1])]
    events={'cue': t.goCue_start_time.to_numpy(), 'choice_common': chosen.reward_outcome_time.to_numpy()}
    sign_side=np.where(chosen.animal_response.eq(0),.5,-.5)
    sign_reward=np.where(chosen.rewarded.eq(1),.5,-.5)
    weights={'choice_side': sign_side, 'outcome':sign_reward, 'side_outcome':sign_side*sign_reward}
    for name in weights: events[name]=events['choice_common']
    # Registered choice lick is excluded from the separate lick kernels.
    for side,name in enumerate(['left_lick','right_lick']):
        licks=sides[side]; remove=np.zeros(len(licks),bool)
        for value in chosen.loc[chosen.animal_response.eq(side),'choice_time']:
            remove[np.argmin(abs(licks-value))]=True
        events[name]=licks[~remove]
    # Reward delivery follows registration by 1.5–4.5 ms, below the 20-ms bins.
    # Retain actual delivery times for observed/predicted event-aligned displays.
    delays=[]
    for side,name in enumerate(['left','right']):
        delivered=recording.events(f'{name}_reward')-origin
        c=chosen.loc[chosen.animal_response.eq(side)&chosen.rewarded.eq(1)]
        actual=np.array([delivered[np.argmin(abs(delivered-x))] for x in c.outcome_time])
        if len(actual) and np.max(abs(actual-c.outcome_time.to_numpy()))>.02:
            raise ValueError('Reward timing no longer supports the shared choice/outcome anchor')
        delays.extend((actual-c.outcome_time.to_numpy()).tolist())
        events[name+'_reward_delivery']=actual
        events[name+'_reward']=chosen.loc[chosen.animal_response.eq(side)&chosen.rewarded.eq(1),'outcome_time'].to_numpy()
        events[name+'_omission']=chosen.loc[chosen.animal_response.eq(side)&chosen.rewarded.eq(0),'outcome_time'].to_numpy()
    blocks=[];columns=[];slices={};position=0
    for name,start,end,count in KERNELS:
        if name in weights:
            block=np.zeros((len(centers),count))
            for value in np.unique(weights[name]):
                block+=value*event_design(edges,events[name][weights[name]==value],start,end,count)
        else: block=event_design(edges,events[name],start,end,count)
        blocks.append(sparse.csr_matrix(block));slices[name]=list(range(position,position+count));position+=count
        columns.extend([f'{name}_{i}' for i in range(count)])
    speed_bin = binned_speed(recording, edges, interpolate=4)
    usable &= np.isfinite(speed_bin)
    movement=SplineTransformer(degree=2,knots=np.array([0,1,5,15,40])[:,None],include_bias=False,extrapolation='linear')
    movement.fit(np.array([[0.],[40.]]))
    covariates={'movement':movement.transform(np.nan_to_num(speed_bin)[:,None]),
                'probability':trials[['reward_probabilityL','reward_probabilityR']].to_numpy()[safe],
                'history':trials[['previous_response','previous_side_contrast','previous_outcome_contrast']].to_numpy()[safe]}
    knots=np.r_[0.,np.arange(float(cfg.trend_spacing_s),edges[-1],float(cfg.trend_spacing_s)),edges[-1]]
    covariates['drift']=SplineTransformer(degree=3,knots=knots[:,None],include_bias=False).fit_transform(centers[:,None])
    for name,values in covariates.items():
        count=values.shape[1]; blocks.append(sparse.csr_matrix(values));slices[name]=list(range(position,position+count));position+=count
        columns.extend([f'{name}_{i}' for i in range(count)])
    x=sparse.hstack(blocks,format='csr');x.eliminate_zeros()
    groups={'Cue':slices['cue'], 'Choice / outcome':sum([slices[k] for k in ['choice_common','choice_side','outcome','side_outcome']],[]),
            'Choice side':slices['choice_side']+slices['side_outcome'], 'Outcome':slices['outcome']+slices['side_outcome'],
            'Licking':slices['left_lick']+slices['right_lick'], 'Movement':slices['movement'],
            'Reward probability':slices['probability'],'History':slices['history']}
    masks={};fold=trials.fold.to_numpy()[safe];block_id=trials.block.to_numpy()[safe]
    # Erode assignment boundaries, not tiny gaps between successive trials.
    coverage=(edges[:-1]>=trials.start_time.iloc[0])&(edges[1:]<=stop)
    for f in range(3):
        train=usable&guarded(coverage&(fold!=f),centers,float(cfg.guard_s))
        test=usable&guarded(coverage&(fold==f),centers,float(cfg.guard_s))
        inner_val=((block_id//3)%4==0)
        fit=usable&guarded(coverage&(fold!=f)&~inner_val,centers,float(cfg.guard_s))
        val=usable&guarded(coverage&(fold!=f)&inner_val,centers,float(cfg.guard_s))
        for role,m in [('train',train),('test',test),('fit',fit),('validation',val)]:
            masks[f'{role}_{f}']=m
            if m.sum()<100: raise ValueError('Insufficient split coverage')
        assert not np.any(train&test) and not np.any(fit&val)
        assert np.all((fit|val)<=train)
    units, _ = selected_units(recording, cfg.regions, cfg.sorting,
                              allow_missing_anatomy=bool(cfg.get('allow_missing_anatomy', False)))
    anatomy_available = bool(units.anatomy_available.any())
    units.to_csv(output/'unit_audit.csv',index=False); units=units.loc[units.included].reset_index(drop=True)
    units.to_csv(output/'units.csv',index=False);trials.to_csv(output/'trials.csv',index=False)
    counts=np.zeros((len(units),len(centers)),dtype=np.uint16)
    for i,row in enumerate(units.source_unit_row):
        times=recording.spikes(int(row))-origin
        y=np.histogram(times[(times>=0)&(times<edges[-1])],edges)[0]
        if y.max()>np.iinfo(np.uint16).max: raise ValueError('Spike count overflow')
        counts[i]=y
    np.save(output/'counts.npy',counts);sparse.save_npz(output/'design.npz',x)
    np.savez_compressed(output/'design_metadata.npz',edges=edges,trial_index=ti,speed=speed_bin,usable=usable,**masks)
    np.savez_compressed(output/'events.npz',**events)
    info=dict(clock=clock,recording_id=str(recording.info['id']),recording_h5=str(path),recording_source=source_fingerprint(path),duration_s=stop,opto_start_s=opto,stored_trials=len(trials),usable_trials=int(valid.sum()),
              n_units=len(units),sorting=units.decoder_label.value_counts().to_dict(),labels=units.predicted_cell_type.value_counts().to_dict(),
              extra_reward_deliveries=extra_deliveries,excluded_trial_ids=trials.loc[~valid,'id'].tolist(),
              excluded_trial_padding_s=dict(before=1,after=2),
              columns=columns,slices=slices,groups=groups,kernels=KERNELS,bin_size=dt,
              reward_delay_s_quantiles=np.quantile(delays,[0,.5,1]).tolist(),
              split_bin_counts={k:int(v.sum()) for k,v in masks.items()},
              model='Poisson log link, ridge penalty, 20-ms counts; no spike-history predictor',
              choice_outcome='Factorial common, side (+0.5 left), outcome (+0.5 reward), and interaction kernels at registration. Actual reward is <20 ms later. Individual side/outcome curves are derived, not independently fitted overlapping kernels.',
              lick_scope='Side-specific detected licks excluding the matched registered choice lick',
              history_scope='Previous stored trial: valid response indicator, side contrast (+0.5 left/-0.5 right), outcome contrast (+0.5 reward/-0.5 omission); zero contrasts without a valid previous response. Held constant throughout current trial.',
              validation='Three folds of interleaved 20-trial blocks, 3-s guards on each side. Training-only inner block validation for full and drift-baseline ridge strength; drop-group models reuse selected full-model alpha.',
              uncertainty='Three-fold curve spread and positive held-out gains are descriptive reliability, not confidence intervals or significance',
              movement_knots_cm_s=[0,1,5,15,40],sorting_policy='All non-noise SUA/MUA mapped to ACB/CP; default_qc preserved, not required',
              observed_speed_quantiles_cm_s=np.quantile(speed_bin[usable],[0,.05,.5,.95,.99,1]).tolist(),
              scope='All usable behavioral trials before optotagging; independent of embedding availability')
    (output/'design_info.json').write_text(json.dumps(info,indent=2)+'\n')
    recording.close()
    print(f'Prepared {len(units)} units, {valid.sum()} trials, {x.shape[1]} predictors',flush=True)
