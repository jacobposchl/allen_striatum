"""Full-curve unit gallery and descriptive, held-out reliability summaries."""
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import SplineTransformer
from src.glm.design import temporal_basis

COLOR='#187f82'
LABELS={'cue':'Go cue','choice_common':'Choice / outcome · common','choice_side':'Left − right','outcome':'Reward − omission',
        'side_outcome':'Side × outcome','left_lick':'Left licking','right_lick':'Right licking'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#cbd2d8','axes.labelcolor':'#42505c','xtick.color':'#63717b','ytick.color':'#63717b',
                         'axes.titleweight':'normal','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})


def curve(ax,x,y,title,color=COLOR):
    if np.isfinite(y).any():
        for values in y: ax.plot(x,values,color=color,alpha=.22,lw=.8)
        ax.fill_between(x,np.nanmin(y,axis=0),np.nanmax(y,axis=0),color=color,alpha=.12,lw=0)
        ax.plot(x,np.nanmean(y,axis=0),color=color,lw=1.7)
    ax.axhline(0,color='#b7c0c7',lw=.7);ax.set_title(title,loc='left',fontsize=10)
    ax.tick_params(labelsize=8)


def kernel_values(coef,info):
    lags=np.arange(-50,101)*.02
    vals={name:coef[:,info['slices'][name]]@temporal_basis(lags,start,stop,count).T for name,start,stop,count in info['kernels']}
    for side,s in [('left',.5),('right',-.5)]:
        for outcome,r in [('reward',.5),('omission',-.5)]:
            vals[side+'_'+outcome]=vals['choice_common']+s*vals['choice_side']+r*vals['outcome']+s*r*vals['side_outcome']
    speed=np.linspace(0,info['observed_speed_quantiles_cm_s'][-1],161)
    movement=SplineTransformer(degree=2,knots=np.array(info['movement_knots_cm_s'])[:,None],include_bias=False,extrapolation='linear').fit(np.array([[0.],[40.]]))
    basis=movement.transform(speed[:,None]);mv=coef[:,info['slices']['movement']]@(basis-basis[0]).T
    probability=np.array([.1,.4,.7]);pv=coef[:,info['slices']['probability'],None]*(probability-.4)[None,None,:]
    return lags,vals,speed,mv,probability,pv


def kernel_figure(unit,coef,info,path):
    lags,vals,speed,mv,probability,pv=kernel_values(coef,info)
    fig,axes=plt.subplots(4,3,figsize=(12.4,10.2),layout='constrained');axes=axes.ravel()
    for ax,name in zip(axes,list(LABELS)):
        spec=next(k for k in info['kernels'] if k[0]==name);sel=(lags>=spec[1])&(lags<=spec[2])
        curve(ax,lags[sel],vals[name][:,sel],LABELS[name]);ax.axvline(0,color='#d4dbe0',lw=.8,ls=':');ax.set_xlabel('Time from event (s)')
    curve(axes[7],speed,mv,'Speed · relative to rest');axes[7].set_xlabel('Speed (cm/s)')
    for side,color in [(0,COLOR),(1,'#b06539')]:
        y=pv[:,side]
        axes[8].plot(probability,np.nanmean(y,axis=0),color=color,marker='o',ms=3,label=['Left','Right'][side])
        axes[8].fill_between(probability,np.nanmin(y,axis=0),np.nanmax(y,axis=0),alpha=.12,color=color)
    axes[8].axhline(0,color='#b7c0c7',lw=.7);axes[8].set_title('Reward probability · relative to 0.4',loc='left',fontsize=10);axes[8].set_xlabel('Probability');axes[8].legend(frameon=False,fontsize=8)
    history=coef[:,info['slices']['history']];mean=np.nanmean(history,axis=0);lo=np.nanmin(history,axis=0);hi=np.nanmax(history,axis=0)
    axes[9].errorbar(np.arange(3),mean,yerr=np.maximum(np.array([mean-lo,hi-mean]),0),fmt='o',color=COLOR,capsize=3)
    axes[9].axhline(0,color='#b7c0c7',lw=.7);axes[9].set_xticks(range(3),['Responded','L − R','Reward − omission']);axes[9].tick_params(axis='x',labelsize=7);axes[9].set_title('Previous trial · response and contrasts',loc='left',fontsize=10)
    for ax,side in zip(axes[10:],['left','right']):
        for outcome,color in [('reward',COLOR),('omission','#b06539')]:
            y=vals[side+'_'+outcome];ax.plot(lags,np.nanmean(y,axis=0),color=color,label=outcome.title())
            ax.fill_between(lags,np.nanmin(y,axis=0),np.nanmax(y,axis=0),color=color,alpha=.1)
        ax.axhline(0,color='#b7c0c7',lw=.7);ax.axvline(0,color='#d4dbe0',lw=.8,ls=':')
        ax.set_title(side.title()+' choice / outcome',loc='left',fontsize=10);ax.set_xlabel('Time from registration (s)');ax.legend(frameon=False,fontsize=8)
    for ax in axes[::3]:ax.set_ylabel('Log rate change')
    n_fits=int(np.isfinite(coef).all(axis=1).sum())
    stability=f'{n_fits}/3 folds fitted · shading shows fold range' if n_fits>1 else '1/3 folds fitted · stability unavailable'
    fig.suptitle(f"Unit {unit.id} · {unit.decoder_label.upper()} · {unit.ccf_acronym} · {unit.predicted_cell_type}\nConditional responses · {stability}",fontsize=12)
    fig.savefig(path,dpi=130);plt.close(fig)
    return dict(lags=lags,kernel_names=np.array(list(vals)),kernels=np.stack(list(vals.values()),axis=1),speed=speed,movement=mv,
                probability=probability,probability_effect=pv,history=history)


def aligned(events,counts,prediction,fold,dt):
    # Round event-relative sample positions to the nearest 20-ms bin center.
    offsets=np.arange(-50,100);centers=np.asarray(events)/dt-.5
    index=np.rint(centers[:,None]+offsets).astype(int)
    good=(index.min(axis=1)>=0)&(index.max(axis=1)<len(counts));index=index[good]
    if len(index):
        complete=np.isfinite(prediction[index]).all(axis=1)&(fold[index]==fold[index[:,50],None]).all(axis=1)
        index=index[complete]
    lags=np.arange(-.95,2,.1)
    if not len(index):return lags,np.full(30,np.nan),np.full(30,np.nan),0
    observed=counts[index].reshape(len(index),30,5).mean(axis=(0,2))/dt
    predicted=prediction[index].reshape(len(index),30,5).mean(axis=(0,2))/dt
    return lags,observed,predicted,len(index)


def observed_figure(unit,counts,model,events,meta,path,dt):
    fig,axes=plt.subplots(2,4,figsize=(13,6.3),layout='constrained');axes=axes.ravel();arrays={}
    for ax,name in zip(axes,['cue','left_reward','left_omission','right_reward','right_omission','left_lick','right_lick']):
        # Reward displays align to measured delivery; omissions to expected outcome.
        times=events[name.replace('_reward','_reward_delivery')] if name.endswith('_reward') else events[name]
        lag,y,p,n=aligned(times,counts,model['prediction'],model['fold'],dt)
        arrays[name]=np.stack([lag,y,p]);arrays[name+'_n']=np.asarray(n)
        ax.plot(lag,y,color='#354653',lw=1.5,label='Observed');ax.plot(lag,p,color=COLOR,lw=1.5,label='Predicted')
        ax.axvline(0,color='#cbd2d8',lw=.8,ls=':');ax.set_title(f"{name.replace('_',' ').capitalize()} · n={n}",loc='left',fontsize=10)
        ax.set_xlabel('Time from event (s)');ax.set_ylabel('Firing rate (Hz)')
    pred=model['prediction'];valid=np.isfinite(pred);time=(meta['edges'][:-1]+meta['edges'][1:])/2
    block=(time//60).astype(int);n=np.bincount(block,weights=valid,minlength=block.max()+1)
    obs=np.bincount(block,weights=np.where(valid,counts,0),minlength=len(n));pr=np.bincount(block,weights=np.nan_to_num(pred),minlength=len(n))
    good=n>0;axes[7].plot(np.flatnonzero(good)+.5,obs[good]/n[good]/dt,color='#354653',lw=1.1)
    axes[7].plot(np.flatnonzero(good)+.5,pr[good]/n[good]/dt,color=COLOR,lw=1.1);axes[7].set_title('Session · held-out bins',loc='left',fontsize=10)
    axes[7].set_xlabel('Time (min)');axes[7].set_ylabel('Firing rate (Hz)');axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle(f'Unit {unit.id} · observed and held-out predictions',fontsize=12);fig.savefig(path,dpi=130);plt.close(fig)
    return arrays


def summarize(output):
    style();info=json.loads((output/'design_info.json').read_text());units=pd.read_csv(output/'units.csv');rows=[];optimization=[];tuning=[]
    for unit in units.itertuples():
        path=output/'units'/f'unit_{unit.id:04d}.json'
        if not path.exists() or not path.with_suffix('.npz').exists():continue
        data=json.loads(path.read_text())
        rows.extend([dict(unit_id=unit.id,**r) for r in data['metrics']]);optimization.extend([dict(unit_id=unit.id,**r) for r in data['optimization']]);tuning.extend([dict(unit_id=unit.id,**r) for r in data['tuning']])
    metrics=pd.DataFrame(rows);metrics.to_csv(output/'fold_metrics.csv',index=False)
    pd.DataFrame(optimization).to_csv(output/'optimization.csv',index=False);pd.DataFrame(tuning).to_csv(output/'tuning.csv',index=False)
    fit_status=units.rename(columns={'id':'unit_id'}).copy()
    fitted_folds=metrics.loc[metrics.status.eq('fit')&metrics.group.eq('Full')].groupby('unit_id').size()
    fit_status['fitted_folds']=fit_status.unit_id.map(fitted_folds).fillna(0).astype(int)
    fit_status['fit_status']=np.select([fit_status.fitted_folds.eq(3),fit_status.fitted_folds.gt(0)],['all_folds','partial_folds'],default='too_few_training_spikes')
    fit_status.loc[~fit_status.unit_id.isin(metrics.unit_id.unique()),'fit_status']='pending'
    fit_status.to_csv(output/'unit_fit_status.csv',index=False)
    records=[]
    for (uid,group),part in metrics.loc[metrics.status.eq('fit')].groupby(['unit_id','group'],sort=False):
        records.append(dict(unit_id=uid,group=group,heldout_gain=1-part.full_deviance.sum()/part.reduced_deviance.sum(),
                            positive_folds=int(part.gain.gt(0).sum()),n_folds=len(part),gain_min=part.gain.min(),gain_max=part.gain.max(),
                            all_converged=bool((part.full_converged&part.reduced_converged&part.selected_tuning_converged).all()),
                            full_vs_constant=1-part.full_deviance.sum()/part.null_deviance.sum(),
                            heldout_spikes=int(part.test_spikes.sum()),heldout_bins=int(part.test_bins.sum())))
    reliability=pd.DataFrame(records).merge(units.rename(columns={'id':'unit_id'}),on='unit_id',validate='many_to_one')
    reliability['consistent_gain']=(reliability.n_folds.eq(3)&reliability.positive_folds.eq(3)&reliability.all_converged&reliability.heldout_gain.gt(0))
    reliability.to_csv(output/'reliability.csv',index=False)
    counts=np.load(output/'counts.npy',mmap_mode='r');meta=dict(np.load(output/'design_metadata.npz'));events=dict(np.load(output/'events.npz'))
    plots=output/'plots';plots.mkdir(exist_ok=True);curves=output/'curves';curves.mkdir(exist_ok=True)
    gallery=[]
    for i,unit in enumerate(units.itertuples()):
        path=output/'units'/f'unit_{unit.id:04d}.npz'
        if not path.exists():continue
        model=dict(np.load(path));rel=reliability.loc[reliability.unit_id.eq(unit.id)]
        if not np.isfinite(model['coefficients']).any():continue
        values=kernel_figure(unit,model['coefficients'],info,plots/f'unit_{unit.id:04d}_curves.png')
        aligned_values=observed_figure(unit,counts[i],model,events,meta,plots/f'unit_{unit.id:04d}_observed.png',info['bin_size'])
        np.savez_compressed(curves/f'unit_{unit.id:04d}.npz',**values,**{'aligned_'+k:v for k,v in aligned_values.items()})
        full=rel.loc[rel.group.eq('Full')].iloc[0]
        gallery.append(dict(unit=int(unit.id),sorting=unit.decoder_label,region=unit.ccf_acronym,label=unit.predicted_cell_type,qc=bool(unit.default_qc),
                            gain=float(full.heldout_gain),constant_gain=float(full.full_vs_constant),positive=int(full.positive_folds),n_folds=int(full.n_folds),converged=bool(full.all_converged),
                            groups={r.group:dict(gain=float(r.heldout_gain),positive=int(r.positive_folds),n_folds=int(r.n_folds),converged=bool(r.all_converged)) for r in rel.itertuples()}))
        if (i+1)%30==0:print(f'Plotted {i+1}/{len(units)} units',flush=True)
    groups=['Full']+list(info['groups']);fig,axes=plt.subplots(1,2,figsize=(11,5.8),layout='constrained')
    rng=np.random.default_rng(0)
    for j,kind in enumerate(['sua','mua']):
        ax=axes[j]
        for row,group in enumerate(groups):
            part=reliability.loc[reliability.decoder_label.eq(kind)&reliability.group.eq(group)&reliability.all_converged]
            y=row+rng.uniform(-.18,.18,len(part));ax.scatter(part.heldout_gain*100,y,s=9,color=COLOR,alpha=.38,lw=0)
        ax.axvline(0,color='#9ba7b0',lw=1);ax.set_yticks(range(len(groups)),['All behavior vs drift']+groups[1:]);ax.invert_yaxis()
        ax.set_title(f"{'Single units' if kind=='sua' else 'Multi-units'} · n={sum(units.decoder_label.eq(kind))}",loc='left')
        ax.set_xlabel('Held-out deviance reduction (%)')
    fig.suptitle('Dec 10 · behavioral GLM',fontsize=14);fig.savefig(output/'glm_overview.png',dpi=180);fig.savefig(output/'glm_overview.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.8,5.6),layout='constrained')
    for ax,kind in zip(axes,['sua','mua']):
        values=[];denominators=[]
        for group in groups:
            part=reliability.loc[reliability.decoder_label.eq(kind)&reliability.group.eq(group)]
            values.append(int(part.consistent_gain.sum()))
            denominators.append(int((part.n_folds.eq(3)&part.all_converged).sum()))
        y=np.arange(len(groups));ax.barh(y,denominators,color='#edf1f4',height=.58)
        ax.barh(y,values,color=COLOR,height=.58)
        for row,(value,total) in enumerate(zip(values,denominators)):
            ax.text(max(denominators)*1.03,row,f'{value}/{total}',va='center',fontsize=9,color='#42505c')
        ax.set_yticks(y,['All behavior vs drift']+groups[1:]);ax.invert_yaxis();ax.set_xlim(0,max(denominators)*1.32)
        ax.set_title('Single units' if kind=='sua' else 'Multi-units',loc='left');ax.set_xlabel('Units')
    fig.suptitle('Positive held-out contribution in all three folds',fontsize=13)
    fig.savefig(output/'reliability_overview.png',dpi=180);fig.savefig(output/'reliability_overview.pdf');plt.close(fig)
    summary=dict(fitted_units=len(gallery),expected_units=len(units),n_trials=info['usable_trials'],
                 fit_status_counts=fit_status.fit_status.value_counts().to_dict(),
                 all_fits_converged=bool(pd.DataFrame(optimization).converged.all()),
                 consistent_gain_by_sorting={kind:{group:int(reliability.loc[reliability.decoder_label.eq(kind)&reliability.group.eq(group),'consistent_gain'].sum()) for group in groups} for kind in ['sua','mua']},
                 interpretation='Consistent gain means positive held-out contribution in all three folds with converged fits; not a significance threshold or validated cell class.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    page='''<!doctype html><html><head><meta charset="utf-8"><title>Dec 10 · behavioral GLM</title><style>
body{font:15px system-ui;color:#253b48;background:#f5f7f8;max-width:1200px;margin:32px auto;padding:0 20px}h1{font-size:27px}p{line-height:1.6}select,button{font:inherit;padding:8px;border:1px solid #c8d4dc;border-radius:7px;background:white;margin:4px}img{width:100%;background:white;border-radius:10px;margin:12px 0}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px 12px;border-bottom:1px solid #e8edef;text-align:left}.note{color:#546b78}a{color:#187f82}</style></head><body>
<h1>Dec 10 · behavioral GLM</h1><p>536 trials · striatal single units and multi-units · no embedding predictors</p>
<p class="note">Curves describe conditional firing associations. Shading spans three training-fold fits; it is not a confidence interval. Event-aligned predictions use held-out spikes only for evaluation. Choice and outcome share an event anchor because reward follows registration by less than one 20-ms bin. Lick kernels exclude the registered choice lick.</p>
<label>Sorting <select id="sorting"><option value="all">All</option><option value="sua">Single units</option><option value="mua">Multi-units</option></select></label>
<label>Source QC <select id="qc"><option value="all">All</option><option value="pass">Pass</option><option value="fail">Fail</option></select></label>
<label>Order <select id="order"><option value="id">Unit ID</option><option value="gain">Held-out gain</option></select></label>
<button id="previous">Previous</button><select id="unit"></select><button id="next">Next</button><h2 id="title"></h2><p id="quality"></p>
<table><thead><tr><th>Contribution</th><th>Held-out gain</th><th>Positive folds</th><th>Fits converged</th></tr></thead><tbody id="metrics"></tbody></table>
<p class="note">Gain compares the full model with a refitted model lacking the named predictor group; “Full” compares against smooth drift alone. Overlapping groups are not additive. Positive folds are descriptive reliability, not statistical significance. Previous-trial history includes a valid-response indicator, a left–right contrast, and a reward–omission contrast. Speed curves stay within the observed range; rare speeds have less support.</p>
<img id="curves"><img id="observed"><p><a href="reliability.csv">Reliability table</a> · <a href="fold_metrics.csv">Per-fold metrics</a> · <a href="units.csv">Unit metadata</a> · <a href="design_info.json">Definitions</a></p>
<script>const data=DATA;const el=id=>document.getElementById(id);let shown=[];
function refresh(){const last=el('unit').value;shown=data.filter(d=>(el('sorting').value==='all'||d.sorting===el('sorting').value)&&(el('qc').value==='all'||d.qc===(el('qc').value==='pass')));shown.sort((a,b)=>el('order').value==='id'?a.unit-b.unit:b.gain-a.gain);el('unit').replaceChildren(...shown.map(d=>new Option('Unit '+d.unit,d.unit)));if(shown.some(d=>String(d.unit)===last))el('unit').value=last;render();}
function render(){let d=shown.find(d=>String(d.unit)===el('unit').value);if(!d)return;el('title').textContent='Unit '+d.unit+' · '+d.sorting.toUpperCase()+' · '+d.region+' · '+d.label;el('quality').textContent='Source default QC: '+(d.qc?'pass':'fail')+' · Full-model gain: '+(d.gain*100).toFixed(2)+'% · Positive in '+d.positive+'/'+d.n_folds+' fitted folds (3 planned)';el('metrics').innerHTML=Object.entries(d.groups).map(([k,v])=>'<tr><td>'+k+'</td><td>'+(v.gain*100).toFixed(3)+'%</td><td>'+v.positive+'/'+v.n_folds+'</td><td>'+(v.converged?'Yes':'No — inspect optimization.csv')+'</td></tr>').join('');const id=String(d.unit).padStart(4,'0');el('curves').src='plots/unit_'+id+'_curves.png';el('observed').src='plots/unit_'+id+'_observed.png';}
el('sorting').onchange=refresh;el('qc').onchange=refresh;el('order').onchange=refresh;el('unit').onchange=render;el('previous').onclick=()=>{el('unit').selectedIndex=Math.max(0,el('unit').selectedIndex-1);render()};el('next').onclick=()=>{el('unit').selectedIndex=Math.min(shown.length-1,el('unit').selectedIndex+1);render()};refresh();</script></body></html>'''
    page=page.replace('536 trials',f"{info['usable_trials']} of {info['stored_trials']} trials")
    page=page.replace('<label>Sorting', '<p class="note">Six trials excluded: three with unmatched reward-delivery records and the immediately following trials. Left–right and reward–omission curves are equal-weight contrasts across the other factor; the interaction is the difference between side-specific outcome contrasts. Pre-event terms are retrospective associations.</p><label>Sorting')
    (output/'index.html').write_text(page.replace('DATA',json.dumps(gallery)))
    print(json.dumps(summary,indent=2),flush=True)
