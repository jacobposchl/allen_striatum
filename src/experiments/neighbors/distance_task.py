"""Direct cosine-distance/task screen with held-out temporal blocks.

Selection uses discovery data only. Confirmation compares full and task-group-
omitted predictions on untouched time blocks; SPIKE controls are prespecified.
"""
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.linalg import qr, lstsq
from scipy.stats import chi2, false_discovery_control
from sklearn.preprocessing import SplineTransformer
from omegaconf import OmegaConf

from src.experiments.neighbors.task_associations import GROUPS, CONTROLS, features
from src.data.io import resolve
from src.data.recording import Recording, resolve_recording
from src.embeddings.windows import RecordingExport


def roles(starts):
    # Minutes 3–4 of each four-minute cycle are held out. Thirty-second guards.
    centers = starts + 5
    minute = np.floor(centers / 60).astype(int)
    test = (minute % 4 == 3) & ((minute + 1) * 60 <= starts[-1] + 10)
    gap = np.zeros(len(starts), bool)
    for center in centers[test]:
        gap |= np.abs(centers - center) <= 30
    train = ~gap
    role = np.where(test, 'test', np.where(train, 'train', 'guard'))
    return role, minute // 4


def preparation_identity(cfg):
    exported = RecordingExport(cfg.export)
    with Recording(resolve_recording(cfg.recording, root=cfg.get('recording_root'))) as recording:
        identity = recording.identity
    if identity != exported.source_identity:
        raise ValueError('Recording changed since export; regenerate the embedding windows')
    task = resolve(cfg.task_source)
    task_info = json.loads((task / 'metadata.json').read_text())
    if (task_info.get('recording_source_identity') != identity
            or task_info.get('export_manifest_sha256') != exported.manifest_sha256):
        raise ValueError('Task traces must be regenerated from the same HDF5 recording and embedding export')
    inputs = {name: hashlib.sha256((task / name).read_bytes()).hexdigest()
              for name in ['trials.csv', 'behavior_windows.csv']}
    settings = OmegaConf.to_container(cfg, resolve=True)
    for key in ['stage', 'export', 'recording', 'recording_root', 'output', 'task_source', 'cpu_threads']:
        settings.pop(key, None)
    return dict(format='brainsets_h5_v2', recording_identity=identity,
                export_manifest_sha256=exported.manifest_sha256, task_inputs=inputs, settings=settings)


def require_preparation(cfg, output):
    path = output / 'cohort.json'
    if not path.is_file():
        raise ValueError('No HDF5 preparation exists; start with stage=all in a fresh output directory')
    saved = json.loads(path.read_text()).get('preparation_identity')
    if saved != preparation_identity(cfg):
        raise ValueError('Prepared distances do not match current HDF5 inputs/settings; use a fresh output directory')


def prepare(cfg, output):
    exported = RecordingExport(cfg.export)
    identity, dimension = {}, None
    starts = exported.starts_session
    if (not np.isfinite(starts).all() or not np.isclose(exported.duration, 10)
            or not np.allclose(np.diff(starts), 10, rtol=0, atol=1e-6)):
        raise ValueError('This temporal-block analysis requires contiguous 10-second windows on the session clock')
    offsets = exported.starts_source - starts
    if not len(offsets) or not np.allclose(offsets, offsets[0], rtol=0, atol=1e-6):
        raise ValueError('An established constant source-to-session clock offset is required')
    origin = float(offsets[0])
    frames = [(float(start), 10., np.array([], dtype=str), None) for start in starts]
    for view, b in exported.windows():
        ids = b['global_unit_id'].astype(str)
        vectors = np.asarray(b['pooled'], float)
        if dimension is not None and vectors.shape[1] != dimension:
            raise ValueError('Embedding dimensions differ between windows')
        dimension = vectors.shape[1]
        for k, key in enumerate(ids):
            record = {name: b[name][k].item() for name in ['unit_id','source_unit_row','label','decoder_label','animal','session','recording']}
            record['global_unit_id'] = key
            if key in identity and identity[key] != record:
                raise ValueError('Unit identities change between export windows')
            identity[key] = record
        frames[view] = (float(starts[view]), 10., ids, vectors)
    if not identity:
        raise ValueError('No active units in the scheduled recording windows')
    units = pd.DataFrame(sorted(identity.values(), key=lambda row:int(row['source_unit_row'])))
    lookup = {key:i for i,key in enumerate(units.global_unit_id)}
    embeddings = np.full((len(frames),len(units),dimension),np.nan)
    present = np.zeros((len(frames),len(units)),bool)
    for t,(_,_,ids,vectors) in enumerate(frames):
        if vectors is None:
            continue
        rows = np.array([lookup[key] for key in ids])
        norms = np.linalg.norm(vectors,axis=1)
        valid = np.isfinite(vectors).all(axis=1)&(norms>0)
        embeddings[t,rows[valid]] = vectors[valid]/norms[valid,None]
        present[t,rows] = True
    valid_vectors = np.isfinite(embeddings).all(axis=2)
    eligible = present.mean(axis=0)>float(cfg.min_presence_fraction)
    kept = np.flatnonzero(eligible)
    a,b = np.triu_indices(len(kept),1)
    pair_i,pair_j = kept[a],kept[b]
    distances = np.full((len(pair_i),len(starts)),np.nan,dtype=np.float32)
    population_distance = np.full_like(distances,np.nan)
    global_distance = np.full(len(starts),np.nan)
    for t in range(len(starts)):
        active = np.flatnonzero(valid_vectors[t]); m=len(active)
        e=embeddings[t,active]
        d=np.clip(1-e@e.T,0,2); np.fill_diagonal(d,0)
        row_sums=np.zeros(len(units));row_sums[active]=d.sum(axis=1)
        total=d.sum()/2; global_distance[t]=total/(m*(m-1)/2) if m>1 else np.nan
        live=valid_vectors[t,pair_i]&valid_vectors[t,pair_j]
        values=np.clip(1-np.einsum('ij,ij->i',embeddings[t,pair_i[live]],embeddings[t,pair_j[live]]),0,2)
        distances[live,t]=values
        if m>3:
            population_distance[live,t]=(total-row_sums[pair_i[live]]-row_sums[pair_j[live]]+values)/((m-2)*(m-3)/2)
    print(f'Export: {len(units)} units; eligible {len(kept)}; {len(pair_i)} pairs; {len(starts)} windows',flush=True)
    input_identity=preparation_identity(cfg)
    recording=Recording(resolve_recording(cfg.recording, root=cfg.get('recording_root'))); u=recording.units
    for name in ['ccf_acronym','depth','default_qc','ccf_ap','ccf_dv','ccf_ml']:
        units[name]=u[name].to_numpy()[units.source_unit_row.to_numpy(int)]
    assert np.array_equal(u['id'].to_numpy()[units.source_unit_row.to_numpy(int)].astype(str),units.unit_id.astype(str))
    rates=np.zeros((len(starts),len(units)))
    cv=np.full_like(rates,np.nan);short=np.full_like(rates,np.nan);fano=np.full_like(rates,np.nan)
    edges=np.r_[starts,starts[-1]+10]
    for row,source_row in enumerate(units.source_unit_row.to_numpy(int)):
        ts=recording.spikes(source_row)-origin
        ts=ts[(ts>=edges[0])&(ts<edges[-1])]
        bins=np.floor((ts-edges[0])/10).astype(int);count=np.bincount(bins,minlength=len(starts))
        rates[:,row]=count/10
        isi=np.diff(ts);same=np.diff(bins)==0;which=bins[1:][same];isi=isi[same]
        n=np.bincount(which,minlength=len(starts))
        sums=np.bincount(which,weights=isi,minlength=len(starts)).astype(float)
        squares=np.bincount(which,weights=isi**2,minlength=len(starts)).astype(float)
        reliable=n>=9
        mean=np.divide(sums,n,out=np.zeros_like(sums),where=n>0)
        variance=np.maximum(0,np.divide(squares,n,out=np.zeros_like(sums),where=n>0)-mean**2)
        cv[reliable,row]=np.sqrt(variance[reliable])/mean[reliable]
        small=np.bincount(which,weights=(isi<.01).astype(float),minlength=len(starts))
        short[reliable,row]=small[reliable]/n[reliable]
        fine=np.bincount(np.floor((ts-edges[0])/.02).astype(int),minlength=len(starts)*500).reshape(len(starts),500)
        fine_mean=fine.mean(axis=1)
        fano[count>=10,row]=fine.var(axis=1)[count>=10]/fine_mean[count>=10]
    recording.close()
    units['available_fraction']=present.mean(axis=0);units['eligible']=eligible
    units.to_csv(output/'units.csv',index=False)
    frame=features(resolve(cfg.task_source));assert np.array_equal(frame.start_session_s,starts)
    role,block=roles(starts)
    frame['role']=role;frame['block']=block;frame['available_all']=valid_vectors.sum(axis=1)
    frame['global_mean_distance']=global_distance
    frame.to_csv(output/'design_features.csv',index=False)
    np.savez(output/'prepared.npz',distances=distances,population_distance=population_distance,
             pair_i=pair_i,pair_j=pair_j,rates=rates,isi_cv=cv,short_isi=short,fano=fano,
             starts=starts,origin=origin,valid_vectors=valid_vectors)
    metadata=dict(preparation_identity=input_identity,n_units=len(units),n_eligible=len(kept),n_pairs=len(pair_i),n_windows=len(starts),
                  sorting=units.loc[eligible,'decoder_label'].value_counts().to_dict(),
                  labels=units.loc[eligible,'label'].value_counts().to_dict(),presence_threshold=float(cfg.min_presence_fraction),
                  recording_h5=str(resolve_recording(cfg.recording, root=cfg.get('recording_root'))),export=str(resolve(cfg.export)),origin=float(origin))
    (output/'cohort.json').write_text(json.dumps(metadata,indent=2)+'\n')


def common_design(frame, minutes):
    time=((frame.start_session_s+frame.stop_session_s)/2).to_numpy()
    knots=np.r_[time[0],np.arange(minutes*60,time[-1],minutes*60),time[-1]]
    trend=SplineTransformer(degree=3,knots=knots[:,None],include_bias=False).fit_transform(time[:,None])
    c=np.column_stack([np.ones(len(time)),trend,frame[CONTROLS+['available_all']].to_numpy(float)])
    names=[name for values in GROUPS.values() for name in values]
    z=frame[names].to_numpy(float)
    return c,z,{g:[names.index(name) for name in group] for g,group in GROUPS.items()},names


def pair_controls(c, data, pair, population=True, spike=False, coactivity=None):
    i,j=int(data['pair_i'][pair]),int(data['pair_j'][pair])
    a,b=np.log1p(data['rates'][:,i]),np.log1p(data['rates'][:,j])
    pieces=[c,a[:,None],b[:,None],(a*a)[:,None],(b*b)[:,None],np.abs(a-b)[:,None],(a*b)[:,None]]
    if population:pieces.append(data['population_distance'][pair,:,None])
    if spike:
        pieces.extend([data[name][:,[i,j]] for name in ['isi_cv','short_isi','fano']])
        pieces.append(coactivity[:,None])
    return np.column_stack(pieces)


def standardized(x, train):
    mean=x[train].mean(axis=0);scale=x[train].std(axis=0)
    constant=scale<1e-10
    scale[constant]=1
    mean[constant]=0
    return (x-mean)/scale,mean,scale


def discover(cfg, output):
    require_preparation(cfg, output)
    with np.load(output/'prepared.npz') as archive:
        data={key:archive[key] for key in archive.files}
    frame=pd.read_csv(output/'design_features.csv')
    c,z,groups,names=common_design(frame,int(cfg.trend_minutes))
    known=frame.task_known.to_numpy(bool)
    train=(frame.role=='train').to_numpy()&known;test=(frame.role=='test').to_numpy()&known
    results=[];skipped=[]
    for pair,y in enumerate(data['distances']):
        nuisance=pair_controls(c,data,pair)
        valid=np.isfinite(y)&np.isfinite(nuisance).all(axis=1)&np.isfinite(z).all(axis=1)
        use=train&valid;hold=test&valid
        if use.sum()<cfg.min_train_windows or hold.sum()<cfg.min_test_windows:
            skipped.append(dict(pair=pair,reason='insufficient joint training/test coverage',n_train=int(use.sum()),n_test=int(hold.sum())));continue
        cn,_,_=standardized(nuisance,use);zn,_,_=standardized(z,use)
        q,r,piv=qr(cn[use],mode='economic',pivoting=True)
        rank=int(np.sum(np.abs(np.diag(r))>1e-9*np.abs(r[0,0])))
        q=q[:,:rank]
        rz=zn[use]-q@(q.T@zn[use]);ry=y[use]-q@(q.T@y[use])
        if np.linalg.matrix_rank(rz)<z.shape[1]:
            skipped.append(dict(pair=pair,reason='task design rank deficient',n_train=int(use.sum()),n_test=int(hold.sum())));continue
        inverse=np.linalg.inv(rz.T@rz);beta=inverse@(rz.T@ry)
        residual=ry-rz@beta;sse=float(residual@residual)
        score=np.zeros((len(y),z.shape[1]));score[use]=rz*residual[:,None]
        meat=score.T@score
        for lag in range(1,int(cfg.hac_lags)+1):
            cross=score[lag:].T@score[:-lag]
            meat+=(1-lag/(cfg.hac_lags+1))*(cross+cross.T)
        covariance=inverse@meat@inverse*use.sum()/(use.sum()-rank-z.shape[1])
        for group,cols in groups.items():
            cov=covariance[np.ix_(cols,cols)]
            wald=max(0,float(beta[cols]@np.linalg.pinv(cov)@beta[cols]))
            extra=max(0,float(beta[cols]@np.linalg.solve(inverse[np.ix_(cols,cols)],beta[cols])))
            results.append(dict(pair=pair,group=group,n_train=int(use.sum()),n_test=int(hold.sum()),
                                partial_r2=extra/(sse+extra),p=float(chi2.sf(wald,len(cols))),
                                coefficients=json.dumps({names[k]:float(beta[k]) for k in cols})))
        if pair%1000==0:print(f'Discovery: pair {pair}/{len(data["pair_i"])}',flush=True)
    result=pd.DataFrame(results)
    if result.empty:raise ValueError('No estimable pairs')
    result['q']=false_discovery_control(result.p.to_numpy())
    result.to_csv(output/'discovery.csv',index=False)
    pd.DataFrame(skipped,columns=['pair','reason','n_train','n_test']).to_csv(output/'skipped_pairs.csv',index=False)
    candidates=[]
    for group,part in result.groupby('group',sort=False):
        significant=part.loc[part.q<.05]
        pool=significant if len(significant) else part
        selected=pool.sort_values(['p','pair']).head(int(cfg.top_per_group)).copy()
        selected['selection']='training q < .05' if len(significant) else 'exploratory: no discovery q < .05'
        candidates.append(selected)
    selected=pd.concat(candidates,ignore_index=True)
    selected.to_csv(output/'candidates.csv',index=False)
    print(f'Discovery tests {len(result)}; q<.05 {(result.q<.05).sum()}; candidates {len(selected)}',flush=True)


def predict_models(nuisance,z,y,train,test,cols):
    valid=np.isfinite(y)&np.isfinite(nuisance).all(axis=1)&np.isfinite(z).all(axis=1)
    use=train&valid;hold=test&valid
    cn,_,_=standardized(nuisance,use);zn,_,scale=standardized(z,use)
    # Match discovery's nuisance rank exactly. Absolute rate difference can be
    # redundant with the two rates when one unit is always faster in training.
    q,r,piv=qr(cn[use],mode='economic',pivoting=True)
    rank=int(np.sum(np.abs(np.diag(r))>1e-9*np.abs(r[0,0])))
    q=q[:,:rank];cn=cn[:,piv[:rank]]
    rz=zn[use]-q@(q.T@zn[use]);ry=y[use]-q@(q.T@y[use])
    keep=np.setdiff1d(np.arange(z.shape[1]),cols)
    beta=lstsq(rz,ry,cond=1e-10,lapack_driver='gelsy')[0]
    reduced_beta=lstsq(rz[:,keep],ry,cond=1e-10,lapack_driver='gelsy')[0]
    nuisance_full=lstsq(cn[use],y[use]-zn[use]@beta,cond=1e-10,lapack_driver='gelsy')[0]
    nuisance_reduced=lstsq(cn[use],y[use]-zn[use][:,keep]@reduced_beta,cond=1e-10,lapack_driver='gelsy')[0]
    pf=cn@nuisance_full+zn@beta
    pr=cn@nuisance_reduced+zn[:,keep]@reduced_beta
    return use,hold,pf,pr,beta,scale


def validate(cfg, output):
    require_preparation(cfg, output)
    with np.load(output/'prepared.npz') as archive:
        data={key:archive[key] for key in archive.files}
    frame=pd.read_csv(output/'design_features.csv');units=pd.read_csv(output/'units.csv')
    candidates=pd.read_csv(output/'candidates.csv')
    c,z,groups,names=common_design(frame,int(cfg.trend_minutes))
    known=frame.task_known.to_numpy(bool);train=(frame.role=='train').to_numpy()&known;test=(frame.role=='test').to_numpy()&known
    recording=Recording(resolve_recording(cfg.recording, root=cfg.get('recording_root')))
    relevant=np.unique(np.r_[data['pair_i'][candidates.pair],data['pair_j'][candidates.pair]])
    fine={}
    for row in relevant:
        source_row=int(units.iloc[row].source_unit_row)
        ts=recording.spikes(source_row)-float(data['origin'])
        low=float(frame.start_session_s.iloc[0]);high=float(frame.stop_session_s.iloc[-1])
        ts=ts[(ts>=low)&(ts<high)]
        fine[row]=np.bincount(np.floor((ts-low)/.02).astype(int),minlength=len(frame)*500).reshape(len(frame),500).astype(float)
    recording.close()
    rng=np.random.default_rng(int(cfg.seed));results=[];predictions=[]
    for candidate in candidates.itertuples():
        pair=int(candidate.pair);i,j=int(data['pair_i'][pair]),int(data['pair_j'][pair]);y=data['distances'][pair].astype(float)
        a=fine[i]-fine[i].mean(axis=1,keepdims=True);b=fine[j]-fine[j].mean(axis=1,keepdims=True)
        denominator=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        coactivity=np.divide(np.sum(a*b,axis=1),denominator,out=np.full(len(frame),np.nan),where=denominator>0)
        variants=[('Primary',True,False),('No population control',False,False),('Spike-pattern controls',True,True)]
        for variant,population,spike in variants:
            nuisance=pair_controls(c,data,pair,population,spike,coactivity)
            use,hold,pf,pr,beta,scale=predict_models(nuisance,z,y,train,test,groups[candidate.group])
            if use.sum()<cfg.min_train_windows or hold.sum()<cfg.min_test_windows:
                results.append(dict(pair=pair,group=candidate.group,variant=variant,status='insufficient reliable windows',
                                    unit_a=int(units.iloc[i].unit_id),unit_b=int(units.iloc[j].unit_id),
                                    label_a=units.iloc[i].label,label_b=units.iloc[j].label,
                                    sorting_a=units.iloc[i].decoder_label,sorting_b=units.iloc[j].decoder_label,
                                    n_train=int(use.sum()),n_test=int(hold.sum())));continue
            errors_reduced=(y-pr)**2;errors_full=(y-pf)**2
            improvement=errors_reduced-errors_full
            blocks=np.unique(frame.block.to_numpy()[hold])
            block_sums=np.array([improvement[hold&(frame.block.to_numpy()==block)].sum() for block in blocks])
            block_counts=np.array([np.sum(hold&(frame.block.to_numpy()==block)) for block in blocks])
            draws=rng.integers(len(blocks),size=(2000,len(blocks)))
            boot=block_sums[draws].sum(axis=1)/block_counts[draws].sum(axis=1)
            signs=rng.choice([-1,1],size=(10000,len(blocks)))
            p=(1+np.sum(signs@block_sums>=block_sums.sum()))/(len(signs)+1)
            denominator_error=errors_reduced[hold].sum()
            row=dict(pair=pair,group=candidate.group,variant=variant,status='ok',
                     unit_a=int(units.iloc[i].unit_id),unit_b=int(units.iloc[j].unit_id),
                     label_a=units.iloc[i].label,label_b=units.iloc[j].label,
                     sorting_a=units.iloc[i].decoder_label,sorting_b=units.iloc[j].decoder_label,
                     n_train=int(use.sum()),n_test=int(hold.sum()),n_blocks=len(blocks),
                     discovery_q=float(candidate.q),selection=candidate.selection,
                     heldout_gain=float(improvement[hold].sum()/denominator_error),
                     mean_squared_error_gain=float(improvement[hold].mean()),
                     gain_ci_low=float(np.quantile(boot,.025)),gain_ci_high=float(np.quantile(boot,.975)),p=float(p),
                     coefficients=json.dumps({names[k]:float(beta[k]/scale[k]) for k in groups[candidate.group]}))
            results.append(row)
            predictions.append(pd.DataFrame(dict(pair=pair,group=candidate.group,variant=variant,start_session_s=frame.start_session_s,
                                                  role=frame.role,valid=use|hold,distance=y,full_prediction=pf,reduced_prediction=pr)))
        print(f'Validated {candidate.group}: {units.iloc[i].unit_id} / {units.iloc[j].unit_id}',flush=True)
    result=pd.DataFrame(results);result['q']=np.nan
    # Include all planned validation variants in the reported multiplicity family.
    finite=result.status.eq('ok')
    corrected=false_discovery_control(result.p.fillna(1).to_numpy())
    result.loc[finite,'q']=corrected[finite.to_numpy()]
    result.to_csv(output/'validation.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(output/'predictions.csv',index=False)
    metadata=dict(primary='Direct cosine distance; unique unordered pairs, no kNN scores',
                  controls='Cubic 5-minute time splines; trial choice/cue counts, speed, unit availability; nonlinear terms of both log firing rates; leave-pair-out mean population distance',
                  split='Last minute of every four-minute cycle held out; 30-second guard around validation; all selection and standardization use training data',
                  discovery='Partial group effects and time-grid-aware 60-second Newey-West covariance; BH across every eligible pair x four groups',
                  confirmation='Training-fitted full/group-omitted models compared on held-out squared error; 1-minute validation blocks separated by training periods',
                  confirmation_uncertainty='2000 block bootstrap draws for mean squared-error gain; 10000 block sign flips, assuming symmetric independent block gains; BH over all planned validation variants',
                  spike_controls='Within-window ISI CV, fraction ISI <10ms and 20ms count Fano factor for both units, plus within-window count correlation; ISI/Fano require >=10 spikes per unit',
                  limitations='Single-session offline associations; encoder already pretrained on this session; temporal separation is approximate, not independent biological replication. Additive models can miss interactions. Population control can remove real shared population signals. Reference/sorting artifacts remain possible.',
                  minimum_train=int(cfg.min_train_windows),minimum_test=int(cfg.min_test_windows))
    (output/'method.json').write_text(json.dumps(metadata,indent=2)+'\n')
    from src.experiments.neighbors.plots.distance_task import plot
    plot(result,output)
    print(result[['unit_a','label_a','unit_b','label_b','group','variant','heldout_gain','q','status']].to_string(index=False),flush=True)


def run(cfg):
    output=resolve(cfg.output)
    if cfg.stage=='all':
        if output.exists():raise FileExistsError(output)
        output.mkdir(parents=True);OmegaConf.save(cfg,output/'config.yaml')
        prepare(cfg,output);discover(cfg,output);validate(cfg,output)
    elif cfg.stage=='prepare':
        output.mkdir(parents=True,exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError('Preparation requires an empty output directory')
        OmegaConf.save(cfg,output/'config.yaml')
        prepare(cfg,output);discover(cfg,output);validate(cfg,output)
    elif cfg.stage=='discover':discover(cfg,output);validate(cfg,output)
    elif cfg.stage=='validate':validate(cfg,output)
    else:raise ValueError('stage must be all, prepare, discover or validate')
    return output
