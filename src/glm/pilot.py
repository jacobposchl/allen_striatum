"""Resumable CPU pilot retaining full kernels and out-of-fold predictions."""
import json
import hashlib
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from src.data.io import resolve
from src.data.task import recording_path, source_fingerprint
from src.common.analysis import deviance
from src.glm.pilot_design import prepare


def scaled(x, fit, unscaled=()):
    train=x[fit];mean=np.asarray(train.mean(axis=0)).ravel()
    variance=np.asarray(train.power(2).mean(axis=0)).ravel()-mean**2
    # Rare basis functions must not explode outside their training support.
    # Drift splines already have a fixed bounded scale; leave them unscaled.
    scale=np.maximum(np.sqrt(np.maximum(variance,0)),.05)
    scale[np.asarray(unscaled,int)]=1
    return (x@sparse.diags(1/scale)).tocsr(),mean/scale,scale


def fit_poisson(x, y, mean, alpha, max_iter, start=None):
    """Sparse, implicitly centered ridge Poisson fit; intercept unpenalized."""
    n=len(y)
    if not y.sum(): raise ValueError('No training spikes')
    if start is None: start=np.r_[np.log(y.mean()),np.zeros(x.shape[1])]
    xty=np.asarray(x.T@y).ravel()/n-mean*y.mean()
    def objective(theta):
        eta=np.asarray(x@theta[1:]).ravel()+theta[0]-mean@theta[1:]
        if eta.max()>500: return np.inf,np.zeros_like(theta)
        mu=np.exp(eta)
        value=np.mean(mu-y*eta)+alpha*.5*np.dot(theta[1:],theta[1:])
        residual_mean=mu.mean()-y.mean()
        gradient=np.r_[residual_mean,np.asarray(x.T@mu).ravel()/n-mean*mu.mean()-xty+alpha*theta[1:]]
        return value,gradient
    result=minimize(objective,start,jac=True,method='L-BFGS-B',options={'maxiter':max_iter,'gtol':1e-6,'ftol':1e-11,'maxls':30,'maxcor':15})
    # In addition to optimizer status, save the actual final gradient.
    gradient=float(np.max(abs(result.jac)))
    return result.x,dict(converged=bool(result.success and gradient<1e-4),iterations=int(result.nit),max_gradient=gradient,message=str(result.message))


def predict(x, theta, mean):
    eta=np.asarray(x@theta[1:]).ravel()+theta[0]-mean@theta[1:]
    return np.exp(eta)


def choose_alpha(x, validation, y, yval, mean, cfg):
    best=None;start=None;trials=[]
    for alpha in sorted(map(float,cfg.alphas),reverse=True):
        theta,status=fit_poisson(x,y,mean,alpha,int(cfg.max_iter),start)
        start=theta
        loss=deviance(yval,predict(validation,theta,mean))
        trials.append(dict(alpha=alpha,deviance=loss,**status))
        if np.isfinite(loss) and (best is None or loss<best[0]): best=(loss,alpha,theta,status)
    if best is None: raise ValueError('No finite tuning fit')
    return best,trials


def designs(x, meta, info):
    allcols=np.arange(x.shape[1]);out=[]
    for f in range(3):
        masks={role:meta[f'{role}_{f}'] for role in ['fit','validation','train','test']}
        inner,im,isc=scaled(x,masks['fit'],info['slices']['drift']);outer,om,osc=scaled(x,masks['train'],info['slices']['drift'])
        part=dict(masks=masks,inner_mean=im,outer_mean=om,scale=osc,
                  fit=inner[masks['fit']],validation=inner[masks['validation']],train=outer[masks['train']],test=outer[masks['test']])
        part['reduced']={}
        for name,cols in {'Baseline':info['slices']['drift'],**{g:np.setdiff1d(allcols,c) for g,c in info['groups'].items()}}.items():
            cols=np.asarray(cols,int)
            part['reduced'][name]=dict(columns=cols,train=part['train'][:,cols],test=part['test'][:,cols])
        cols=info['slices']['drift'];part['baseline_fit']=part['fit'][:,cols];part['baseline_validation']=part['validation'][:,cols]
        out.append(part)
    return out


def fit_one(index, unit, counts, prepared, info, cfg, output):
    target=output/'units'/f'unit_{int(unit.id):04d}.npz'
    if target.exists(): return int(unit.id),'cached'
    began=time.monotonic();y=np.asarray(counts[index],float);rows=[];audit=[];tuning=[]
    coef=np.full((3,len(info['columns'])),np.nan);intercept=np.full(3,np.nan)
    prediction=np.full(len(y),np.nan,dtype=np.float32);baseline=np.full(len(y),np.nan,dtype=np.float32)
    folds=np.full(len(y),-1,dtype=np.int8)
    for f,d in enumerate(prepared):
        m=d['masks'];ys={k:y[v] for k,v in m.items()}
        if ys['fit'].sum()<int(cfg.min_train_spikes) or ys['train'].sum()<int(cfg.min_train_spikes):
            rows.append(dict(fold=f,group='Full',status='too_few_training_spikes'));continue
        best,records=choose_alpha(d['fit'],d['validation'],ys['fit'],ys['validation'],d['inner_mean'],cfg)
        tuning.extend([dict(fold=f,model='Full',**r) for r in records]);alpha=best[1]
        theta,status=fit_poisson(d['train'],ys['train'],d['outer_mean'],alpha,int(cfg.max_iter))
        audit.append(dict(fold=f,model='Full',alpha=alpha,**status))
        pred=predict(d['test'],theta,d['outer_mean']);full_dev=deviance(ys['test'],pred)
        prediction[m['test']]=pred;folds[m['test']]=f
        coef[f]=theta[1:]/d['scale'];intercept[f]=theta[0]-d['outer_mean']@theta[1:]
        cols=info['slices']['drift'];b=d['reduced']['Baseline']
        best_base,records=choose_alpha(d['baseline_fit'],d['baseline_validation'],ys['fit'],ys['validation'],d['inner_mean'][cols],cfg)
        tuning.extend([dict(fold=f,model='Baseline',**r) for r in records])
        bt,bs=fit_poisson(b['train'],ys['train'],d['outer_mean'][cols],best_base[1],int(cfg.max_iter))
        audit.append(dict(fold=f,model='Baseline',alpha=best_base[1],**bs))
        bp=predict(b['test'],bt,d['outer_mean'][cols]);base_dev=deviance(ys['test'],bp);baseline[m['test']]=bp
        null_dev=deviance(ys['test'],np.full(len(ys['test']),ys['train'].mean()))
        common=dict(fold=f,status='fit',alpha=alpha,train_spikes=int(ys['train'].sum()),test_spikes=int(ys['test'].sum()),
                    test_bins=len(ys['test']),full_deviance=full_dev,baseline_deviance=base_dev,null_deviance=null_dev,
                    full_converged=status['converged'],selected_tuning_converged=best[3]['converged'])
        rows.append(dict(**common,group='Full',reduced_deviance=base_dev,gain=1-full_dev/base_dev,
                         delta_loglik=.5*(base_dev-full_dev),reduced_converged=bs['converged']))
        for name in info['groups']:
            reduced=d['reduced'][name];cols=reduced['columns']
            initial=np.r_[theta[0],theta[1:][cols]]
            reduced_theta,reduced_status=fit_poisson(reduced['train'],ys['train'],d['outer_mean'][cols],alpha,int(cfg.max_iter),initial)
            rp=predict(reduced['test'],reduced_theta,d['outer_mean'][cols]);rd=deviance(ys['test'],rp)
            audit.append(dict(fold=f,model='Drop '+name,alpha=alpha,**reduced_status))
            rows.append(dict(**common,group=name,reduced_deviance=rd,gain=1-full_dev/rd,
                             delta_loglik=.5*(rd-full_dev),reduced_converged=reduced_status['converged']))
    temp=target.with_suffix('.tmp.npz')
    np.savez_compressed(temp,coefficients=coef,intercepts=intercept,prediction=prediction,baseline_prediction=baseline,fold=folds)
    # Write metadata before committing the NPZ completion marker.
    (target.with_suffix('.json')).write_text(json.dumps(dict(unit_id=int(unit.id),metrics=rows,optimization=audit,tuning=tuning,elapsed_s=time.monotonic()-began),indent=2)+'\n')
    temp.replace(target)
    print(f'Unit {unit.id}: {time.monotonic()-began:.1f}s',flush=True)
    return int(unit.id),'fit'


def run(cfg):
    output=resolve(cfg.output);output.mkdir(parents=True,exist_ok=True)
    from src.common.parallel import available_cpus
    if int(cfg.workers)>available_cpus():
        raise ValueError('Requested GLM workers exceed the CPUs in this job step')
    config_path=output/'config.yaml'
    if config_path.exists():
        previous=OmegaConf.to_container(OmegaConf.load(config_path),resolve=True);current=OmegaConf.to_container(cfg,resolve=True)
        for key in ['stage','unit_limit','workers','cpu_threads','recording','recording_root','output']:
            previous.pop(key,None);current.pop(key,None)
        if previous!=current: raise ValueError('Pilot configuration changed; choose a new output')
    else: config_path.write_text(OmegaConf.to_yaml(cfg))
    if cfg.stage in ['all','prepare'] and not (output/'design_info.json').exists(): prepare(cfg,output)
    if (output/'design_info.json').exists():
        prepared_info = json.loads((output/'design_info.json').read_text())
        if prepared_info.get('recording_source') != source_fingerprint(recording_path(cfg)):
            raise ValueError('Prepared model data do not match the selected Brainsets recording; choose a new output')
    if cfg.stage=='prepare': return output
    if cfg.stage in ['all','fit']:
        source_files = [Path(__file__), Path(__file__).with_name('pilot_design.py'),
                        Path(__file__).with_name('design.py'), Path(__file__).parents[1] / 'data/task.py',
                        Path(__file__).parents[1] / 'data/recording.py']
        signature=hashlib.sha256(''.join(path.read_text() for path in source_files).encode()).hexdigest()
        source_path=output/'fit_source.json'
        if source_path.exists() and json.loads(source_path.read_text())['sha256']!=signature:
            raise ValueError('GLM fitting source changed; choose a new output to prevent mixing fits')
        source_path.write_text(json.dumps(dict(sha256=signature,scaling='Training standard deviations floored at 0.05; drift basis unscaled; implicit training-only centering'),indent=2)+'\n')
        info=json.loads((output/'design_info.json').read_text());units=pd.read_csv(output/'units.csv');counts=np.load(output/'counts.npy',mmap_mode='r')
        x=sparse.load_npz(output/'design.npz');meta=dict(np.load(output/'design_metadata.npz'))
        (output/'units').mkdir(exist_ok=True)
        selected=list(units.index if cfg.unit_limit is None else units.index[:int(cfg.unit_limit)])
        selected=[i for i in selected if not (output/'units'/f'unit_{int(units.iloc[i].id):04d}.npz').exists()]
        if selected:
            print(f'Preparing shared designs for {len(selected)} remaining units',flush=True)
            prepared=designs(x,meta,info)
            Parallel(n_jobs=int(cfg.workers),backend='threading',batch_size=1)(delayed(fit_one)(i,units.iloc[i],counts,prepared,info,cfg,output) for i in selected)
    if cfg.stage in ['all','plot']:
        from src.glm.pilot_plot import summarize
        summarize(output)
    return output
