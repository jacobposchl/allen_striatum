"""Resumable recording/unit cache: final coefficients separate from validation."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from src.data.recording import Recording, discover_recordings
from src.data.io import resolve
from src.data.task import recording_path, source_fingerprint
from src.glm import pilot
from src.glm.cache_design import prepare
from src.glm.design import temporal_basis

REQUIRED_EVENTS = ['left_lick', 'right_lick', 'left_reward', 'right_reward']
VERSION = 'rich_active_full_fit_h5_v2'

def write_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def write_csv(path, frame):
    temp = path.with_suffix('.csv.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def inventory(cfg, out):
    rows = []
    for item in discover_recordings(cfg.root):
        if any(cfg.cohort.get(k) and str(item[k[:-1]]) not in list(cfg.cohort[k])
               for k in ['animals', 'sessions', 'recordings']):
            continue
        with Recording(item['path']) as recording:
            missing = []
            if not recording.availability['trials']:
                missing.append('trials')
            for name in REQUIRED_EVENTS:
                if not recording.has_events(name):
                    missing.append(name)
            movement = recording.has_signal('speed')
            units = recording.units
            anatomy_available = units.ccf_acronym.notna().any()
            sorting_available = 'decoder_label' in units
            region = units.ccf_acronym.isin(list(cfg.regions)) if anatomy_available else True
            n_eligible = int((region & units.decoder_label.isin(list(cfg.sorting))).sum()) if sorting_available else -1
            status = ('missing_units' if units.empty else 'missing_behavior' if missing else 'missing_unit_metadata' if not sorting_available or (not anatomy_available and not cfg.allow_missing_anatomy) else 'no_eligible_units' if n_eligible == 0
                      else 'pending_movement' if not movement and not cfg.allow_missing_movement else 'ready')
            rows.append(dict(animal=item['animal'], session=item['session'], recording=item['recording'],
                recording_id=item['id'], recording_h5=str(item['path']), status=status, missing_fields=';'.join(missing),
                movement_available=movement, anatomy_available=bool(anatomy_available), n_eligible_units=n_eligible,
                cache_relative=f"recordings/{item['animal']}/{item['session']}/{item['recording']}"))
    if not rows:
        raise ValueError('No Brainsets recordings match the configured root and cohort')
    frame = pd.DataFrame(rows)
    write_csv(out/'recordings.csv', frame)
    return frame


def basis_file(directory, info):
    lags = np.arange(-50, 101) * .02
    arrays = {'lags_s': lags, 'column_names': np.asarray(info['columns'])}
    for name, a, b, n in info['kernels']:
        arrays[name] = temporal_basis(lags, a, b, n)
    np.savez_compressed(directory/'basis.npz', **arrays)


def final_curves(coef, info):
    lags = np.arange(-50, 101) * .02
    kernels = {name: temporal_basis(lags, a, b, n) @ coef[info['slices'][name]]
               for name, a, b, n in info['kernels']}
    for side, s in [('left', .5), ('right', -.5)]:
        for outcome, r in [('reward', .5), ('omission', -.5)]:
            kernels[side+'_'+outcome] = (kernels['choice_common'] + s*kernels['choice_side']
                + r*kernels['outcome'] + s*r*kernels['side_outcome'])
    prob = np.asarray([.1, .4, .7])
    result = dict(lags_s=lags, kernel_names=np.asarray(list(kernels)), kernels=np.stack(list(kernels.values())),
        probability=prob, probability_effect=coef[info['slices']['probability'], None]*(prob-.4),
        history=coef[info['slices']['history']])
    if 'movement' in info['slices']:
        from sklearn.preprocessing import SplineTransformer
        speed = np.linspace(0, info['observed_speed_quantiles_cm_s'][-1], 161)
        spline = SplineTransformer(degree=2, knots=np.asarray(info['movement_knots_cm_s'])[:,None],
                                   include_bias=False, extrapolation='linear').fit(np.array([[0.], [40.]]))
        basis = spline.transform(speed[:, None])
        result.update(speed_cm_s=speed, movement=(basis-basis[0])@coef[info['slices']['movement']])
    return result


def finish_unit(index, unit, counts, cv_designs, full, info, cfg, directory):
    uid = int(unit.id)
    final = directory/'final'/f'unit_{uid:04d}.npz'
    if final.exists(): return uid
    validation = directory/'validation'
    pilot.fit_one(index, unit, counts, cv_designs, info, cfg, validation)
    cv = json.loads((validation/'units'/f'unit_{uid:04d}.json').read_text())
    candidates = [t for t in cv['tuning'] if t['model']=='Full' and t['converged'] and np.isfinite(t['deviance'])]
    # Compare penalties on identical available inner-validation folds.
    scores = {}
    folds = sorted({t['fold'] for t in candidates})
    for alpha in cfg.alphas:
        part = [t for t in candidates if t['alpha']==alpha]
        if folds and sorted(t['fold'] for t in part)==folds:
            scores[float(alpha)] = sum(t['deviance'] for t in part)
    alpha = min(scores, key=scores.get) if scores else max(map(float, cfg.alphas))
    selected_by = 'pooled_inner_validation_deviance' if scores else 'fixed_strongest_penalty_no_usable_tuning'
    y = np.asarray(counts[index, full['mask']], float)
    meta = dict(global_unit_id=unit.global_unit_id, unit_id=uid, fit_scope='all_eligible_behavioral_bins',
                schema_version=VERSION, schema_variant=info['schema_variant'], alpha=alpha,
                alpha_selection=selected_by, fit_bins=len(y), fit_spikes=int(y.sum()),
                validation_folds=sum(v.get('group')=='Full' and v.get('status')=='fit' for v in cv['metrics']))
    coef = np.full(len(info['columns']), np.nan); intercept = np.nan
    if y.sum() >= cfg.min_train_spikes:
        theta, opt = pilot.fit_poisson(full['x'], y, full['mean'], alpha, int(cfg.max_iter))
        coef = theta[1:]/full['scale']; intercept = theta[0]-full['mean']@theta[1:]
        meta.update(status='fit' if opt['converged'] else 'not_converged', **opt)
    else:
        meta.update(status='insufficient_spikes', converged=False)
    arrays = dict(coefficients=coef, intercept=np.asarray(intercept), **final_curves(coef, info))
    write_json(final.with_suffix('.json'), meta)
    temp = final.with_suffix('.tmp.npz'); np.savez_compressed(temp, **arrays); temp.replace(final)
    print(f"Final {unit.global_unit_id}: {meta['status']}", flush=True)
    return uid


def summarize(out, records):
    unit_rows, coefficient_rows, reliability_rows = [], [], []
    for row in records.itertuples():
        directory = out/row.cache_relative
        if not (directory/'prepared/units.csv').exists(): continue
        units = pd.read_csv(directory/'prepared/units.csv')
        info = json.loads((directory/'prepared/design_info.json').read_text())
        lookup = {i:(name,j) for name, indexes in info['slices'].items() for j,i in enumerate(indexes)}
        for unit in units.to_dict('records'):
            uid = int(unit['id']); path = directory/'final'/f'unit_{uid:04d}.npz'
            unit.update(animal=row.animal, session=row.session, recording=row.recording,
                        schema_version=VERSION, schema_variant=info.get('schema_variant','with_movement'),
                        final_status='pending', final_file=str(path.relative_to(out)),
                        validation_file=str((directory/'validation/units'/path.name).relative_to(out)))
            if path.exists():
                meta = json.loads(path.with_suffix('.json').read_text())
                unit.update(final_status=meta['status'], final_converged=meta['converged'],
                            alpha=meta['alpha'], validation_folds=meta['validation_folds'])
                with np.load(path) as z:
                    for i, coefficient in enumerate(z['coefficients']):
                        feature, basis = lookup[i]
                        coefficient_rows.append(dict(global_unit_id=unit['global_unit_id'],
                            schema_variant=unit['schema_variant'], fit_role='final', feature=feature,
                            basis_index=basis, coefficient=float(coefficient), fit_status=meta['status']))
                    coefficient_rows.append(dict(global_unit_id=unit['global_unit_id'],
                        schema_variant=unit['schema_variant'], fit_role='final', feature='intercept',
                        basis_index=-1, coefficient=float(z['intercept']), fit_status=meta['status']))
            unit_rows.append(unit)
            cvpath = directory/'validation/units'/f'unit_{uid:04d}.json'
            if not cvpath.exists(): continue
            metrics = pd.DataFrame(json.loads(cvpath.read_text())['metrics'])
            if 'full_deviance' not in metrics: continue
            metrics = metrics.dropna(subset=['full_deviance'])
            for group, g in metrics.groupby('group'):
                denom = g.reduced_deviance.sum() if group!='Full' else g.baseline_deviance.sum()
                reliability_rows.append(dict(global_unit_id=unit['global_unit_id'], group=group,
                    schema_variant=unit['schema_variant'], n_folds=len(g),
                    heldout_gain=1-g.full_deviance.sum()/denom if denom>0 else np.nan,
                    positive_folds=int(g.gain.gt(0).sum()),
                    consistent_gain=bool(len(g)==3 and g.gain.gt(0).all() and g.full_converged.all() and g.reduced_converged.all()),
                    all_converged=bool(g.full_converged.all() and g.reduced_converged.all())))
    write_csv(out/'unit_index.csv', pd.DataFrame(unit_rows))
    write_csv(out/'coefficients.csv', pd.DataFrame(coefficient_rows))
    write_csv(out/'reliability.csv', pd.DataFrame(reliability_rows))


def run(cfg):
    from src.common.parallel import available_cpus
    if int(cfg.workers)>available_cpus(): raise ValueError('Workers exceed existing CPU allocation')
    out = resolve(cfg.output); out.mkdir(parents=True, exist_ok=True)
    # All numerical and design code participates in cache identity.
    source_files = [Path(__file__), Path(__file__).with_name('cache_design.py'),
                    Path(pilot.__file__), Path(__file__).with_name('design.py'),
                    Path(__file__).parents[1] / 'data/task.py',
                    Path(__file__).parents[1] / 'data/recording.py']
    options = OmegaConf.to_container(cfg, resolve=True)
    for k in ['workers', 'cpu_threads', 'stage', 'cohort', 'allow_missing_movement', 'root', 'output', 'recording_root']:
        options.pop(k, None)
    signature = hashlib.sha256((''.join(p.read_text() for p in source_files)+json.dumps(options,sort_keys=True)).encode()).hexdigest()
    provenance = out/'schema.json'
    if provenance.exists() and (out/'recordings').exists() and json.loads(provenance.read_text())['signature']!=signature:
        raise ValueError('Cache model definition changed; choose a new output directory')
    write_json(provenance, dict(schema_version=VERSION, signature=signature,
        final='One fit on all usable bins, including bins omitted only by CV boundary guards; original predictor units',
        validation='Separate three-fold fits, tuning audits, out-of-fold predictions and drop-group reliability; not coefficient confidence intervals',
        alpha='Minimize summed converged inner-validation deviance over the same available folds; if none, explicitly flagged strongest-penalty fallback',
        missing='Missing task data prevents fitting; absent movement is omitted only when configured and has its own schema_variant',
        filtering='ACB/CP SUA and MUA when annotated; if allowed, non-noise units with missing anatomy retained as region unknown; preserve source QC',
        curve_units='conditional log firing-rate effect; rate multiplier is exp(kernel)',
        paths='All unit_index paths relative to this cache root'))
    OmegaConf.save(cfg, out/'config.yaml')
    records = inventory(cfg, out)
    print(records[['animal','session','recording','status','n_eligible_units']].to_string(index=False), flush=True)
    if cfg.stage=='inventory': return out
    if cfg.stage=='summarize': summarize(out, records); return out
    for i, row in records.iterrows():
        if row.status!='ready': continue
        directory = out/row.cache_relative; directory.mkdir(parents=True, exist_ok=True)
        prep = directory/'prepared'; prep.mkdir(exist_ok=True)
        validation = directory/'validation'; (validation/'units').mkdir(parents=True, exist_ok=True)
        (directory/'final').mkdir(exist_ok=True)
        local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        for key in ['animal', 'session']: local[key] = str(row[key])
        local.recording = str(row.recording_id)
        local.recording_root = str(cfg.root)
        records.at[i,'status']='preparing'; write_csv(out/'recordings.csv',records)
        try:
            if not (prep/'design_info.json').exists():
                prepare(local, prep)
            info=json.loads((prep/'design_info.json').read_text())
            if info.get('recording_source') != source_fingerprint(recording_path(local)):
                raise ValueError('Prepared model data do not match the selected Brainsets recording; choose a new output')
            info.setdefault('schema_variant','with_movement')
            basis_file(directory,info)
            x=sparse.load_npz(prep/'design.npz'); meta=dict(np.load(prep/'design_metadata.npz'))
            units=pd.read_csv(prep/'units.csv'); counts=np.load(prep/'counts.npy',mmap_mode='r')
            scaled, mean, scale=pilot.scaled(x,meta['usable'],info['slices']['drift'])
            full=dict(x=scaled[meta['usable']],mean=mean,scale=scale,mask=meta['usable'])
            pending=[j for j,u in units.iterrows() if not (directory/'final'/f'unit_{int(u.id):04d}.npz').exists()]
            needs_cv=any(not (validation/'units'/f'unit_{int(units.iloc[j].id):04d}.npz').exists() for j in pending)
            cv_designs=pilot.designs(x,meta,info) if needs_cv else None
            records.at[i,'status']='fitting'; write_csv(out/'recordings.csv',records)
            for start in range(0,len(pending),int(cfg.workers)*2):
                Parallel(n_jobs=int(cfg.workers),backend='threading',batch_size=1)(
                    delayed(finish_unit)(j,units.iloc[j],counts,cv_designs,full,info,local,directory)
                    for j in pending[start:start+int(cfg.workers)*2])
                summarize(out,records)
            records.at[i,'status']='complete'
            records.at[i,'n_final_fits']=sum(json.loads(p.read_text())['status']=='fit' for p in (directory/'final').glob('unit_*.json'))
            del x, scaled, full, cv_designs, counts
        except (ValueError, KeyError, OSError) as exc:
            records.at[i,'status']='blocked'; records.at[i,'reason']=str(exc)
            print(f"Blocked {row.animal}/{row.session}/{row.recording}: {exc}",flush=True)
        write_csv(out/'recordings.csv',records)
        summarize(out,records)
    return out
