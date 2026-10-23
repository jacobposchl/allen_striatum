"""Held-out-unit prediction of cached GLM summaries from shared encoder structure.

Every learned feature transform is fitted inside an outer unit fold. Labels do
not define representations. This is an exploratory readout of trained recordings,
not a test on unseen sessions or independent GLM data.
"""
import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, SplineTransformer

from src.data.io import resolve
from src.common.plotting import STYLE


MODELS = ['historical', 'passive_pretrained', 'finetuned']
PAIRS = list(itertools.combinations(MODELS, 2))
LABELS = {
    'physiology': 'Physiology',
    'historical': 'Old', 'passive_pretrained': 'Passive', 'finetuned': 'Fine-tuned',
    'shared_historical_passive_pretrained': 'Old + passive shared',
    'shared_historical_finetuned': 'Old + fine-tuned shared',
    'shared_passive_pretrained_finetuned': 'Passive + fine-tuned shared',
}


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def nuisance(frame):
    columns = ['analysis_firing_rate_hz', 'analysis_isi_cv', 'analysis_frac_isi_lt_2ms',
               'analysis_frac_isi_lt_10ms', 'analysis_frac_isi_gt_500ms',
               'peak_to_valley_ms', 'half_width_ms', 'depth', 'available_fraction']
    values = frame[columns].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    values[:, :2] = np.log1p(np.maximum(values[:, :2], 0))
    rate_squared = values[:, 0] ** 2
    quality = frame.default_qc.astype(str).str.lower().eq('true').to_numpy(float)
    mua = frame.decoder_label.eq('mua').to_numpy(float)
    return np.column_stack([values, rate_squared, mua, quality]), columns + ['log_rate_squared', 'mua', 'default_qc']


def _standardize(train, test):
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    return scaler.fit_transform(imputer.fit_transform(train)), scaler.transform(imputer.transform(test))


def _nuisance_transform(train, test, splines=False):
    if not splines:
        return _standardize(train, test)
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    train, test = imputer.fit_transform(train), imputer.transform(test)
    varying = np.flatnonzero(np.std(train[:, :9], axis=0) > 1e-12)
    spline = SplineTransformer(n_knots=5, degree=3, knots='uniform', include_bias=False, extrapolation='linear')
    continuous = spline.fit_transform(train[:, varying])
    continuous_test = spline.transform(test[:, varying])
    # Log-rate squared is replaced by flexible rate splines; sorting, QC, and
    # missing indicators are retained without spline expansion.
    design = np.column_stack([continuous, train[:, 10:]])
    design_test = np.column_stack([continuous_test, test[:, 10:]])
    scaler = StandardScaler()
    return scaler.fit_transform(design), scaler.transform(design_test)


def _corr(x, y):
    return float(pearsonr(x, y).statistic) if len(x) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12 else np.nan


def regularized_cca(x, y, x_test, y_test, dimensions, penalty):
    """Ridge CCA with train-only covariance whitening and standardized scores."""
    x, x_test = _standardize(x, x_test)
    y, y_test = _standardize(y, y_test)
    def inverse_root(covariance):
        values, vectors = np.linalg.eigh(covariance + penalty * np.eye(len(covariance)))
        return (vectors * np.maximum(values, 1e-12) ** -.5) @ vectors.T
    scale = max(len(x) - 1, 1)
    wx, wy = inverse_root(x.T @ x / scale), inverse_root(y.T @ y / scale)
    u, singular, vt = np.linalg.svd(wx @ (x.T @ y / scale) @ wy, full_matrices=False)
    count = min(int(dimensions), len(singular))
    sx, sx_test = _standardize(x @ wx @ u[:, :count], x_test @ wx @ u[:, :count])
    sy, sy_test = _standardize(y @ wy @ vt.T[:, :count], y_test @ wy @ vt.T[:, :count])
    return (sx + sy) / 2, (sx_test + sy_test) / 2, sx, sy, sx_test, sy_test


def run(cfg):
    source, output, targets_path = resolve(cfg.source), resolve(cfg.output), resolve(cfg.targets)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
    units_path, embeddings_path = source / 'units.csv', source / 'embeddings.npz'
    frame = pd.read_csv(units_path, dtype={'global_unit_id': str, 'animal': str})
    targets = pd.read_csv(targets_path, dtype={'global_unit_id': str})
    assert frame.global_unit_id.is_unique and targets.global_unit_id.is_unique
    targets = targets.loc[targets.analysis_included.astype(str).str.lower().eq('true')].copy()
    target_columns = [name for name in targets if name.startswith(('gain_', 'kernel_'))]
    assert target_columns, 'No supported GLM targets'
    with np.load(embeddings_path, allow_pickle=False) as saved:
        assert np.array_equal(saved['global_unit_id'].astype(str), frame.global_unit_id.to_numpy(str))
        matrices = {key: saved[key].astype(float) for key in MODELS}
    frame['_row'] = np.arange(len(frame))
    joined = frame.merge(targets[['global_unit_id'] + target_columns], on='global_unit_id', validate='one_to_one')
    joined.to_csv(output / 'joined_units_targets.csv', index=False)
    predictions, correlations, scores, fold_scores, source_checks, model_comparisons = [], [], [], [], [], []
    rng = np.random.default_rng(int(cfg.seed))
    for session, part in joined.groupby('session', sort=True):
        part = part.reset_index(drop=True)
        if len(part) < 40:
            continue
        print(f'Shared representations and GLM readout: {session}, {len(part)} units', flush=True)
        base, nuisance_columns = nuisance(part)
        values = {key: matrices[key][part._row.to_numpy(int)] for key in MODELS}
        raw_targets = part[target_columns].to_numpy(float)
        eligible = np.sum(np.isfinite(raw_targets), axis=0) >= max(30, int(cfg.folds) * 5)
        names = np.asarray(target_columns)[eligible]
        y_all = raw_targets[:, eligible]
        split = KFold(int(cfg.folds), shuffle=True, random_state=int(cfg.seed))
        per_session = []
        for fold, (train, test) in enumerate(split.split(part)):
            assert not set(train) & set(test)
            train_base, test_base = _nuisance_transform(base[train], base[test], bool(cfg.get('nuisance_spline', False)))
            representations = {'physiology': (train_base, test_base)}
            principal = {}
            for key in MODELS:
                x, xt = _standardize(values[key][train], values[key][test])
                representations[key] = (np.column_stack([train_base, x]), np.column_stack([test_base, xt]))
                pca = PCA(n_components=min(int(cfg.pca_dimensions), x.shape[1], len(train)-1), svd_solver='full')
                principal[key] = (pca.fit_transform(x), pca.transform(xt))
            for first, second in PAIRS:
                x, xt = principal[first]
                y, yt = principal[second]
                shared, shared_test, sx, sy, sxt, syt = regularized_cca(
                    x, y, xt, yt, int(cfg.shared_dimensions), float(cfg.cca_regularization))
                key = f'shared_{first}_{second}'
                shared, shared_test = _standardize(shared, shared_test)
                representations[key] = (np.column_stack([train_base, shared]), np.column_stack([test_base, shared_test]))
                for axis in range(sx.shape[1]):
                    correlations.append(dict(session=session, fold=fold, first=first, second=second,
                        axis=axis+1, train_correlation=_corr(sx[:, axis], sy[:, axis]),
                        heldout_correlation=_corr(sxt[:, axis], syt[:, axis]), n_train=len(train), n_test=len(test)))
            for target_index, target in enumerate(names):
                yy = y_all[:, target_index]
                tr = np.isfinite(yy[train])
                te = np.isfinite(yy[test])
                if tr.sum() < 20 or te.sum() < 3:
                    continue
                mean, sd = yy[train][tr].mean(), yy[train][tr].std()
                if sd <= 1e-12:
                    continue
                for key, (x, xt) in representations.items():
                    model = Ridge(alpha=float(cfg.ridge_alpha))
                    model.fit(x[tr], (yy[train][tr]-mean)/sd)
                    prediction = model.predict(xt[te])*sd + mean
                    for index, estimate in zip(test[te], prediction):
                        per_session.append(dict(global_unit_id=part.global_unit_id.iloc[index], session=session,
                            fold=fold, target=target, representation=key, truth=yy[index], prediction=float(estimate),
                            train_target_mean=float(mean), train_target_sd=float(sd),
                            squared_error=float((yy[index]-estimate)**2)))
            source_checks.append(dict(session=session, fold=fold, n_train=len(train), n_test=len(test),
                                      train_test_overlap=0, transform_scope='outer_training_units_only'))
        records = pd.DataFrame(per_session)
        predictions.append(records)
        for target, subset in records.groupby('target', sort=True):
            pivot = subset.pivot(index='global_unit_id', columns='representation', values='prediction')
            target_values = subset.drop_duplicates('global_unit_id').set_index('global_unit_id').loc[pivot.index, 'truth'].to_numpy()
            physiology = pivot.physiology.to_numpy()
            denominator = np.square(target_values-target_values.mean()).sum()
            if denominator <= 1e-15:
                continue
            base_error = np.square(target_values-physiology)
            draws = rng.integers(len(pivot), size=(int(cfg.bootstrap_samples), len(pivot)))
            for key in pivot:
                estimate = pivot[key].to_numpy()
                error = np.square(target_values-estimate)
                gain = float((base_error-error).sum()/denominator)
                sampled_y = target_values[draws]
                sampled_denominator = np.square(sampled_y-sampled_y.mean(1, keepdims=True)).sum(1)
                sampled_gain = (base_error[draws]-error[draws]).sum(1) / np.maximum(sampled_denominator, 1e-15)
                lower, upper = np.quantile(sampled_gain, [.025, .975])
                scores.append(dict(session=session, target=target, representation=key, n_units=len(pivot),
                    oof_r2=float(1-error.sum()/denominator), oof_correlation=_corr(target_values, estimate),
                    delta_r2_over_physiology=gain, delta_r2_ci_low=float(lower), delta_r2_ci_high=float(upper),
                    mse=float(error.mean()), physiology_mse=float(base_error.mean())))
            for first, second in PAIRS:
                first_error = np.square(target_values-pivot[first].to_numpy())
                second_error = np.square(target_values-pivot[second].to_numpy())
                sampled_gain = (first_error[draws]-second_error[draws]).sum(1) / np.maximum(sampled_denominator, 1e-15)
                lower, upper = np.quantile(sampled_gain, [.025, .975])
                model_comparisons.append(dict(session=session,target=target,first=first,second=second,n_units=len(pivot),
                    delta_r2_second_minus_first=float((first_error-second_error).sum()/denominator),
                    delta_r2_ci_low=float(lower),delta_r2_ci_high=float(upper)))
            for (key, fold), group in subset.groupby(['representation', 'fold']):
                denominator_fold = np.square(group.truth-group.truth.mean()).sum()
                fold_scores.append(dict(session=session, target=target, representation=key, fold=fold,
                    n_units=len(group), mse=float(group.squared_error.mean()),
                    r2=float(1-group.squared_error.sum()/denominator_fold) if denominator_fold>1e-15 else np.nan))
    result = pd.DataFrame(scores)
    assert len(predictions) and len(result), 'No usable sessions/targets'
    predictions = pd.concat(predictions, ignore_index=True)
    assert not predictions.duplicated(['global_unit_id', 'target', 'representation']).any()
    predictions.to_csv(output / 'predictions.csv', index=False)
    result.to_csv(output / 'scores.csv', index=False)
    pd.DataFrame(model_comparisons).to_csv(output / 'model_comparisons.csv', index=False)
    pd.DataFrame(fold_scores).to_csv(output / 'fold_scores.csv', index=False)
    canonical = pd.DataFrame(correlations)
    canonical.to_csv(output / 'canonical_correlations.csv', index=False)
    canonical.groupby(['session', 'first', 'second', 'axis'], as_index=False).agg(
        mean_train_correlation=('train_correlation', 'mean'),
        mean_heldout_correlation=('heldout_correlation', 'mean'),
        min_heldout_correlation=('heldout_correlation', 'min')).to_csv(output / 'canonical_summary.csv', index=False)
    pd.DataFrame(source_checks).to_csv(output / 'fold_audit.csv', index=False)
    _json(output / 'method.json', dict(
        target='Cached held-out GLM drop-group gains plus signed final-kernel scalar summaries; no new GLMs',
        nuisance=nuisance_columns, nuisance_transform='log1p rate and ISI CV; log-rate squared; median imputation with missing indicators; train standardization',
        nuisance_spline=bool(cfg.get('nuisance_spline', False)),
        spline_details='When enabled: train-only cubic additive splines, five uniformly spaced knots on each nonconstant continuous covariate; linear extrapolation. Log-rate squared replaced. Sorting/QC/missing indicators unexpanded.',
        folds='Five shuffled held-out-unit folds within each recording; identical splits for all representations',
        individual='Full standardized embedding plus physiology',
        shared='Train-only standardized embedding -> 10 PCs -> ridge CCA, penalty .1, first5 axes; average standardized paired canonical scores; plus physiology',
        shared_test_inputs='Both encoders used for each held-out unit; not cross-modal prediction from one encoder alone',
        prediction='Ridge alpha10 fixed before outcomes; targets standardized using training units; no hyperparameter selection',
        uncertainty='2000 paired held-out-unit bootstrap samples of prediction errors; descriptive95% intervals; correlated training folds and common recording limit inferential interpretation',
        comparison='R2 improvement versus identical physiology-only predictions; negative values mean worse prediction',
        caveats=['Encoders and GLMs saw these recordings: readout holds out units, not underlying recording data.',
                 'GLM features and embeddings both derive from spikes, so associations are descriptive rather than independent biology.',
                 'Historical export population included noise, new exports exclude noise, changing population-attention context.',
                 'Historical model also differs in training dataset and dimensionality; only passive versus fine-tuned isolates adaptation.',
                 'December18 lacks movement covariates; scalar kernel summaries do not represent complete functional response shapes.',
                 'Multiple exploratory targets; bootstrap intervals are not corrected hypothesis tests.'],
        sources={str(path):_hash(path) for path in [units_path, embeddings_path, targets_path]}))
    _json(output / 'checks.json', dict(unique_unit_targets=True, finite_predictions=bool(np.isfinite(predictions.prediction).all()),
        predictions=len(predictions), sessions=sorted(predictions.session.unique()), folds=int(cfg.folds),
        no_train_test_unit_overlap=True, learned_transforms_fit_to_training_only=True))
    plot(result, canonical, output)
    print(f'Shared GLM readouts complete: {len(result)} scores', flush=True)


def plot(scores, canonical, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    models = list(LABELS)[1:]
    sessions = sorted(scores.session.unique())
    with plt.rc_context(STYLE):
        for prefix, title, filename in [('gain_', 'GLM contribution', 'glm_gain_prediction'),
                                        ('kernel_', 'Signed response', 'signed_kernel_prediction')]:
            targets = sorted(t for t in scores.target.unique() if t.startswith(prefix))
            if not targets:
                continue
            fig, axes = plt.subplots(1, len(sessions), figsize=(13, max(4.8, len(targets)*.40+2)), layout='constrained', squeeze=False)
            for ax, session in zip(axes.flat, sessions):
                subset = scores[scores.session.eq(session)]
                matrix = subset.pivot(index='target', columns='representation', values='delta_r2_over_physiology').reindex(index=targets, columns=models)
                image = ax.imshow(matrix.to_numpy(), cmap='RdBu', norm=TwoSlopeNorm(vmin=-.25,vcenter=0,vmax=.25), aspect='auto')
                for (row, column), value in np.ndenumerate(matrix.to_numpy()):
                    if np.isfinite(value):
                        ax.text(column,row,f'{value:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(value)>.15 else '#172033')
                ax.set(xticks=np.arange(len(models)),xticklabels=[LABELS[k] for k in models],
                       yticks=np.arange(len(targets)),yticklabels=[t.removeprefix(prefix).replace('_',' ') for t in targets],
                       title=f'{session[:10]} · {title}')
                ax.tick_params(axis='x',rotation=55)
            fig.colorbar(image,ax=list(axes.flat),label='Held-out ΔR² over physiology',shrink=.75,extend='both')
            fig.savefig(output/f'{filename}.png',dpi=190)
            plt.close(fig)
        fig, axes = plt.subplots(1,len(sessions),figsize=(10,4.2),layout='constrained',squeeze=False)
        for ax,session in zip(axes.flat,sessions):
            for (first,second),color in zip(PAIRS,['#64748B','#D97706','#0D9488']):
                subset=canonical[canonical.session.eq(session)&canonical['first'].eq(first)&canonical.second.eq(second)]
                grouped=subset.groupby('axis').heldout_correlation
                ax.plot(grouped.mean().index,grouped.mean(),marker='o',color=color,label=LABELS[f'shared_{first}_{second}'].removesuffix(' shared'))
                ax.fill_between(grouped.mean().index,grouped.min(),grouped.max(),alpha=.12,color=color)
            ax.set(title=session[:10],xlabel='Shared axis',ylabel='Held-out correlation',ylim=(-.2,1.02),xticks=np.arange(1,6))
            ax.axhline(0,color='#94A3B8',linewidth=.6)
        axes.flat[-1].legend(frameon=False,fontsize=8)
        fig.savefig(output/'shared_axes.png',dpi=190)
        plt.close(fig)
    (output/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Shared encoder information</title><style>body{max-width:1200px;margin:35px auto;font:17px system-ui;color:#172033}img{width:100%}</style><h1>Shared encoder information</h1><p>Five held-out-unit folds within each recording. Prediction values are improvements over physiology alone. Shading on shared-axis curves spans folds. These are exploratory within-recording associations.</p>'+''.join(f'<img src="{name}.png">' for name in ['shared_axes','glm_gain_prediction','signed_kernel_prediction'])+'<p><a href="scores.csv">Prediction scores</a> · <a href="predictions.csv">Unit predictions</a> · <a href="method.json">Method</a></p></html>')
