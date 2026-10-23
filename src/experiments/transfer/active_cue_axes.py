"""One GLM event kernel, source-learned axes, and frozen-session interpretation."""
import html
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, normalize

from src.common.analysis import correlation, sha256, write_json
from src.embeddings.io import read_table
from src.data.io import resolve
from src.common.plotting import COLORS, STYLE

KERNELS = ['cue', 'choice_common', 'choice_side', 'outcome', 'side_outcome', 'left_lick', 'right_lick']
OTHER = ['choice_common', 'right_minus_left', 'outcome', 'left_lick', 'right_lick']
LABELS = ['Choice / outcome', 'Right − left', 'Reward − omission', 'Left lick', 'Right lick']


def load_data(cfg, out):
    root, glm = resolve(cfg.source), resolve(cfg.glm)
    units = read_table(root/'units.csv')
    targets = read_table(cfg.targets)
    assert units.global_unit_id.is_unique and targets.global_unit_id.is_unique
    targets = targets[targets.analysis_included.astype(str).str.lower().eq('true')]
    columns = ['global_unit_id', 'final_file'] + [c for c in targets if c.startswith('gain_')]
    frame = units.drop(columns=['cluster', 'sua_cluster', 'umap_1', 'umap_2'], errors='ignore').merge(
        targets[columns], on='global_unit_id', validate='one_to_one')
    frame = frame[frame.session.isin([cfg.source_session, cfg.target_session])].reset_index(drop=True)
    membership = read_table(root/'clusters.csv')
    membership = membership[membership.encoder.eq(cfg.encoder)].set_index('global_unit_id').cluster
    frame['cluster'] = frame.global_unit_id.map(membership).astype(int)
    frame['cluster_name'] = 'C' + (frame.cluster+1).astype(str)
    reliability = read_table(glm/'reliability.csv')
    reliability = reliability[reliability.group.eq('Cue')]
    reliability = reliability.rename(columns={'positive_folds': 'cue_positive_folds', 'all_converged': 'cue_all_converged'})
    frame = frame.merge(reliability[['global_unit_id', 'cue_positive_folds', 'cue_all_converged']],
                        on='global_unit_id', validate='one_to_one')
    cache, curves, validation = {}, [], []
    for row in frame.itertuples():
        path = glm/row.final_file
        recording = path.parent.parent
        if recording not in cache:
            info = json.loads((recording/'prepared/design_info.json').read_text())
            with np.load(recording/'basis.npz', allow_pickle=False) as z:
                cache[recording] = info, z['cue'].copy()
        info, basis = cache[recording]
        with np.load(path, allow_pickle=False) as z:
            names = list(z['kernel_names'])
            lag = z['lags_s'].copy()
            curves.append(np.stack([z['kernels'][names.index(k)] for k in KERNELS]))
            assert np.allclose(curves[-1][0], basis@z['coefficients'][info['slices']['cue']])
        with np.load(recording/'validation/units'/path.name, allow_pickle=False) as z:
            validation.append(z['coefficients'][:, info['slices']['cue']]@basis.T)
    curves, validation = np.stack(curves), np.stack(validation)
    assert np.isfinite(curves).all() and np.isfinite(validation).all()
    post = (lag >= -1e-8) & (lag <= 1+1e-8)
    lick = (lag >= -.1-1e-8) & (lag <= .1+1e-8)
    for j, name in enumerate(KERNELS):
        window = lick if 'lick' in name else post
        frame[name+'_mean'] = curves[:, j, window].mean(1)
    frame['right_minus_left_mean'] = -frame.choice_side_mean
    frame['cue_early_mean'] = curves[:, 0, (lag >= 0)&(lag <= .3+1e-8)].mean(1)
    frame['cue_late_mean'] = curves[:, 0, (lag >= .5)&(lag <= 1+1e-8)].mean(1)
    frame['cue_supported'] = frame.gain_cue.gt(0)&frame.cue_positive_folds.ge(2)&frame.cue_all_converged.eq(True)
    fold_means = validation[:, :, post].mean(2)
    frame['cue_cv_sign_agreement'] = (np.sign(fold_means) == np.sign(frame.cue_mean.to_numpy()[:, None])).mean(1)
    frame['cue_direction_supported'] = frame.cue_supported&frame.cue_cv_sign_agreement.eq(1)
    frame['cue_cv_shape_correlation'] = [np.nanmedian([correlation(v[a], v[b]) for a,b in [(0,1),(0,2),(1,2)]]) for v in validation]
    with np.load(root/'embeddings.npz', allow_pickle=False) as z:
        ix = pd.Index(z['global_unit_id'].astype(str)).get_indexer(frame.global_unit_id)
        assert (ix >= 0).all()
        x = z[str(cfg.encoder)][ix].astype(float)
        # All exported target neurons may contribute to the label-free centering sensitivity.
        full_ix = pd.Index(z['global_unit_id'].astype(str)).get_indexer(units[units.session.eq(cfg.target_session)].global_unit_id)
        assert (full_ix >= 0).all()
        target_center = normalize(z[str(cfg.encoder)][full_ix]).mean(0)
    np.savez_compressed(out/'observed_kernels.npz', global_unit_id=frame.global_unit_id.to_numpy(str),
                        kernel_names=np.asarray(KERNELS), kernels=curves, cue_validation=validation, lags_s=lag)
    return frame, x, curves, validation, lag, post, target_center


def fit_axes(x, y, components, post):
    scaler = StandardScaler().fit(normalize(x))
    design = scaler.transform(normalize(x))
    model = PLSRegression(n_components=components, scale=False, max_iter=2000, tol=1e-8).fit(design, y)
    x_mean = design.mean(0)
    raw_score = (design-x_mean)@model.x_rotations_
    sd = raw_score.std(0, ddof=1)
    effects = model.y_loadings_*sd
    signs = np.ones(components)
    for j in range(components):
        orient = effects[post, j].mean() if j == 0 else effects[np.argmax(np.abs(effects[:,j])),j]
        signs[j] = 1 if orient >= 0 else -1
    axes = model.x_rotations_*signs/sd
    effects *= signs
    scores = (design-x_mean)@axes
    state = dict(scale_mean=scaler.mean_, scale=scaler.scale_, x_mean=x_mean, axes=axes,
                 effects=effects, y_mean=y.mean(0), x_weights=model.x_weights_, n_iter=np.asarray(model.n_iter_))
    assert np.allclose(y.mean(0)+scores@effects.T, model.predict(design), atol=1e-8)
    return state


def project(x, state, adjustment=None):
    base = normalize(x)
    if adjustment is not None:
        base = base-adjustment
    design = (base-state['scale_mean'])/state['scale']
    return (design-state['x_mean'])@state['axes']


def predict(scores, state, rank):
    return state['y_mean']+scores[:, :rank]@state['effects'][:, :rank].T


def kernel_metrics(y, pred):
    residual = np.square(y-pred).sum()
    total = np.square(y-y.mean(0)).sum()
    a, b = y-y.mean(0), pred-pred.mean(0)
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return dict(kernel_r2=1-residual/total if total > 0 else np.nan,
                kernel_pattern_r=float(np.sum(a*b)/denominator) if denominator > 1e-12 else np.nan,
                rmse=float(np.sqrt(np.mean(np.square(y-pred)))))


def bootstrap(y, pred, count, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(count):
        idx = rng.integers(len(y), size=len(y))
        values.append(kernel_metrics(y[idx], pred[idx])['kernel_r2'])
    return np.quantile(values, [.025,.975])


def neighbors(train, test, k, metric='euclidean', exclude_self=False):
    d = cdist(test, train, metric)
    if exclude_self:
        assert len(train) == len(test)
        np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :min(k, len(train)-int(exclude_self))]


def fit_and_evaluate(cfg, out, frame, x, y, lag, post, target_center):
    source_mask = frame.session.eq(cfg.source_session).to_numpy()
    source, target = frame[source_mask].copy(), frame[~source_mask].copy()
    xs, xt, ys, yt = x[source_mask], x[~source_mask], y[source_mask], y[~source_mask]
    components, k = int(cfg.components), int(cfg.neighbors)
    assert components == 2, 'This analysis prespecifies one and two axes.'
    splits = list(KFold(int(cfg.folds), shuffle=True, random_state=int(cfg.seed)).split(xs))
    oof = np.zeros((components, *ys.shape))
    oof_baseline = np.zeros_like(ys)
    folds = np.full(len(source), -1, int)
    other_s = source[[n+'_mean' for n in OTHER]].to_numpy()
    neighbor_oof = {space: np.zeros_like(other_s) for space in ['cue_axes', 'original_embedding']}
    fold_records = []
    for fold, (tr, te) in enumerate(splits):
        state = fit_axes(xs[tr], ys[tr], components, post)
        test_scores = project(xs[te], state)
        for rank in range(1, components+1):
            oof[rank-1,te] = predict(test_scores, state, rank)
        for space, training, testing, metric in [
            ('cue_axes', project(xs[tr], state), test_scores, 'euclidean'),
            ('original_embedding', normalize(xs[tr]), normalize(xs[te]), 'cosine')]:
            idx = neighbors(training, testing, k, metric)
            neighbor_oof[space][te] = other_s[tr][idx].mean(1)
        folds[te], oof_baseline[te] = fold, ys[tr].mean(0)
        assert not set(tr)&set(te)
        fold_records.append(dict(fold=fold, train_ids=source.iloc[tr].global_unit_id.tolist(),
                                 test_ids=source.iloc[te].global_unit_id.tolist()))
    state = fit_axes(xs, ys, components, post)
    source_scores, target_scores = project(xs, state), project(xt, state)
    # Translation only; no target feature values, covariance rotation, or model refit.
    adjustment = target_center-normalize(xs).mean(0)
    centered_scores = project(xt, state, adjustment)
    np.savez_compressed(out/'model.npz', **state, training_ids=source.global_unit_id.to_numpy(str),
                        target_mean_adjustment=adjustment, lags_s=lag)
    evaluations = {
        'source_heldout': (source, ys, oof, oof_baseline),
        'target_direct': (target, yt, np.stack([predict(target_scores,state,r) for r in [1,2]]), np.tile(ys.mean(0), (len(yt),1))),
        'target_mean_adjusted': (target, yt, np.stack([predict(centered_scores,state,r) for r in [1,2]]), np.tile(ys.mean(0), (len(yt),1))),
    }
    metrics, time_rows = [], []
    for evaluation, (part, truth, predictions, baseline) in evaluations.items():
        for rank, estimate in enumerate(predictions, 1):
            for scope, mask in [('All', np.ones(len(part),bool)), ('SUA', part.decoder_label.eq('sua').to_numpy()),
                                ('Cue supported', part.cue_supported.to_numpy(bool))]:
                if mask.sum() < 8: continue
                stats = kernel_metrics(truth[mask], estimate[mask])
                low, high = bootstrap(truth[mask], estimate[mask], int(cfg.bootstrap_samples), int(cfg.seed)+rank)
                error = np.square(truth[mask]-estimate[mask]).sum()
                reference = np.square(truth[mask]-baseline[mask]).sum()
                metrics.append(dict(evaluation=evaluation, components=rank, scope=scope, n_units=int(mask.sum()), **stats,
                    kernel_r2_ci_low=low, kernel_r2_ci_high=high, source_mean_skill=1-error/reference,
                    post_mean_r=correlation(truth[mask][:,post].mean(1), estimate[mask][:,post].mean(1))))
            for j, time in enumerate(lag):
                stats = kernel_metrics(truth[:,[j]], estimate[:,[j]])
                time_rows.append(dict(evaluation=evaluation, components=rank, lag_s=float(time), **stats))
        np.savez_compressed(out/f'{evaluation}.npz', global_unit_id=part.global_unit_id.to_numpy(str),
                            truth=truth, predictions=predictions, mean_baseline=baseline,
                            fold=folds if evaluation == 'source_heldout' else np.full(len(part), -1))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(time_rows).to_csv(out/'time_metrics.csv', index=False)
    write_json(out/'source_folds.json', fold_records)
    null = []
    rng = np.random.default_rng(int(cfg.seed)+300)
    for permutation in range(int(cfg.permutations)):
        shuffled = ys[rng.permutation(len(ys))]
        pred = np.zeros_like(oof)
        for tr, te in splits:
            fitted = fit_axes(xs[tr], shuffled[tr], components, post)
            scores = project(xs[te], fitted)
            for rank in [1,2]: pred[rank-1,te] = predict(scores,fitted,rank)
        target_shuffle = yt[rng.permutation(len(yt))]
        for rank in [1,2]:
            null.append(dict(permutation=permutation, components=rank,
                source_kernel_r2=kernel_metrics(shuffled,pred[rank-1])['kernel_r2'],
                target_kernel_r2=kernel_metrics(target_shuffle,evaluations['target_direct'][2][rank-1])['kernel_r2']))
        if (permutation+1)%25 == 0:
            print(f'Cue-axis permutation {permutation+1}/{cfg.permutations}', flush=True)
    null = pd.DataFrame(null)
    null.to_csv(out/'permutation_null.csv', index=False)
    permutation_summary = []
    for rank in [1,2]:
        for evaluation, column in [('source_heldout','source_kernel_r2'), ('target_direct','target_kernel_r2')]:
            observed = metrics.query('components == @rank and evaluation == @evaluation and scope == "All"').iloc[0].kernel_r2
            p = (1+(null[null.components.eq(rank)][column] >= observed).sum())/(1+int(cfg.permutations))
            permutation_summary.append(dict(evaluation=evaluation, components=rank, kernel_r2=observed, permutation_p=float(p)))
    pd.DataFrame(permutation_summary).to_csv(out/'permutation_summary.csv', index=False)
    all_scores = np.empty((len(frame),2))
    all_scores[source_mask], all_scores[~source_mask] = source_scores, target_scores
    frame[['cue_axis_1','cue_axis_2']] = all_scores
    frame['cue_prediction_post'] = np.nan
    frame.loc[source_mask,'cue_prediction_post'] = oof[1][:,post].mean(1)
    frame.loc[~source_mask,'cue_prediction_post'] = evaluations['target_direct'][2][1][:,post].mean(1)
    frame['prediction_evaluation'] = np.where(source_mask, 'source_heldout', 'target_direct')
    frame['source_fold'] = -1
    frame.loc[source_mask,'source_fold'] = folds
    lo, hi = np.quantile(source_scores, [.01,.99], axis=0)
    frame['inside_source_axis_range'] = ((all_scores >= lo)&(all_scores <= hi)).all(1)
    frame.to_csv(out/'neurons.csv',index=False)
    return state, evaluations, metrics, all_scores, neighbor_oof, permutation_summary


def interpret(cfg, out, frame, x, scores, neighbor_oof):
    source_mask = frame.session.eq(cfg.source_session).to_numpy()
    ys = frame.loc[source_mask, [n+'_mean' for n in OTHER]].to_numpy()
    yt = frame.loc[~source_mask, [n+'_mean' for n in OTHER]].to_numpy()
    rows, predictions, edges = [], [], []
    for space, xs, xt, metric in [
        ('cue_axes', scores[source_mask], scores[~source_mask], 'euclidean'),
        ('original_embedding', normalize(x[source_mask]), normalize(x[~source_mask]), 'cosine')]:
        cross_idx = neighbors(xs, xt, int(cfg.neighbors), metric)
        local_idx = neighbors(xt, xt, int(cfg.neighbors), metric, exclude_self=True)
        specs = [
            ('source_heldout_neighbors', frame[source_mask], ys, neighbor_oof[space]),
            ('target_from_source_neighbors', frame[~source_mask], yt, ys[cross_idx].mean(1)),
            ('target_local_neighbors', frame[~source_mask], yt, yt[local_idx].mean(1)),
        ]
        for evaluation, part, truth, estimate in specs:
            for j, feature in enumerate(OTHER):
                stat = kernel_metrics(truth[:,[j]], estimate[:,[j]])
                rows.append(dict(space=space, evaluation=evaluation, feature=feature, n_units=len(part),
                                 pearson=correlation(truth[:,j], estimate[:,j]), **stat))
                for gid, value, pred in zip(part.global_unit_id, truth[:,j], estimate[:,j]):
                    predictions.append(dict(space=space, evaluation=evaluation, feature=feature,
                                            global_unit_id=gid, truth=value, prediction=pred))
        for evaluation, idx, reference in [('target_from_source_neighbors',cross_idx,frame[source_mask]),
                                            ('target_local_neighbors',local_idx,frame[~source_mask])]:
            ids = reference.global_unit_id.to_numpy()
            for gid, nn in zip(frame[~source_mask].global_unit_id,idx):
                assert gid not in ids[nn]
                for rank, neighbor in enumerate(ids[nn],1):
                    edges.append(dict(space=space,evaluation=evaluation,global_unit_id=gid,neighbor_id=neighbor,rank=rank))
    pd.DataFrame(rows).to_csv(out/'neighbor_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'neighbor_predictions.csv',index=False)
    pd.DataFrame(edges).to_csv(out/'neighbors.csv',index=False)
    associations, cluster_rows, extreme_rows = [], [], []
    properties = ['cue_mean','cue_early_mean','cue_late_mean','gain_cue']+[n+'_mean' for n in OTHER]+[
        'analysis_firing_rate_hz','analysis_isi_cv','peak_to_valley_ms','half_width_ms','depth','ccf_ap','ccf_dv','ccf_ml']
    for session, part in frame.groupby('session'):
        for axis in ['cue_axis_1','cue_axis_2']:
            for feature in properties:
                valid = part[[axis,feature]].apply(pd.to_numeric,errors='coerce').dropna()
                if len(valid) > 5 and valid[feature].nunique() > 1:
                    associations.append(dict(session=session,axis=axis,feature=feature,n_units=len(valid),
                                              spearman=float(spearmanr(valid[axis],valid[feature]).statistic)))
            for end, selected in [('low',part.nsmallest(6,axis)), ('high',part.nlargest(6,axis))]:
                selected = selected.copy()
                selected['selected_axis'], selected['axis_end'] = axis, end
                extreme_rows.append(selected)
        for cluster, group in part.groupby('cluster_name',sort=True):
            cluster_rows.append(dict(session=session,cluster=cluster,n_units=len(group),n_sua=int(group.decoder_label.eq('sua').sum()),
                n_d1=int(group.label.eq('D1').sum()),n_d2=int(group.label.eq('D2').sum()),n_cue_supported=int(group.cue_supported.sum()),
                axis_1_median=group.cue_axis_1.median(),axis_2_median=group.cue_axis_2.median(),
                cue_mean_median=group.cue_mean.median(),cue_gain_median=group.gain_cue.median(),
                rate_median=group.analysis_firing_rate_hz.median(),isi_cv_median=group.analysis_isi_cv.median(),
                ccf_dv_median=group.ccf_dv.dropna().median() if group.ccf_dv.notna().any() else np.nan))
    pd.DataFrame(associations).to_csv(out/'axis_associations.csv',index=False)
    pd.DataFrame(cluster_rows).to_csv(out/'cluster_descriptions.csv',index=False)
    pd.concat(extreme_rows,ignore_index=True).to_csv(out/'axis_extreme_neurons.csv',index=False)
    return pd.DataFrame(rows)


def scalar_uncertainty(truth, prediction, count, seed):
    rng=np.random.default_rng(seed)
    idx=rng.integers(len(truth),size=(count,len(truth)))
    a,b=truth[idx],prediction[idx]
    ac,bc=a-a.mean(1,keepdims=True),b-b.mean(1,keepdims=True)
    denom=np.linalg.norm(ac,axis=1)*np.linalg.norm(bc,axis=1)
    good=denom>1e-12
    r=np.sum(ac[good]*bc[good],axis=1)/denom[good]
    r2=1-np.square(a-b).sum(1)/np.maximum(np.square(ac).sum(1),1e-15)
    lo,hi=np.quantile(r,[.025,.975]) if len(r) else [np.nan,np.nan]
    r2lo,r2hi=np.quantile(r2,[.025,.975])
    return dict(pearson_ci_low=lo,pearson_ci_high=hi,r2_ci_low=r2lo,r2_ci_high=r2hi)


def summarize_uncertainty(cfg,out,frame,evaluations,metrics,post):
    for idx,row in metrics.iterrows():
        part,truth,pred,_=evaluations[row.evaluation]
        mask=np.ones(len(part),bool) if row.scope=='All' else (
            part.decoder_label.eq('sua').to_numpy() if row.scope=='SUA' else part.cue_supported.to_numpy(bool))
        stats=scalar_uncertainty(truth[mask][:,post].mean(1),pred[int(row.components)-1][mask][:,post].mean(1),
                                 int(cfg.bootstrap_samples),int(cfg.seed)+42)
        metrics.loc[idx,'post_mean_r_ci_low']=stats['pearson_ci_low']
        metrics.loc[idx,'post_mean_r_ci_high']=stats['pearson_ci_high']
    metrics.to_csv(out/'metrics.csv',index=False)
    predictions=read_table(out/'neighbor_predictions.csv').merge(frame[['global_unit_id','decoder_label']],
                        on='global_unit_id',validate='many_to_one')
    rows=[]
    for keys,g in predictions.groupby(['space','evaluation','feature']):
        for scope,part in [('All',g),('SUA',g[g.decoder_label.eq('sua')])]:
            a,b=part.truth.to_numpy(),part.prediction.to_numpy()
            rows.append(dict(zip(['space','evaluation','feature'],keys),scope=scope,n_units=len(part),
                pearson=correlation(a,b),**kernel_metrics(a[:,None],b[:,None]),
                **scalar_uncertainty(a,b,int(cfg.bootstrap_samples),int(cfg.seed)+42)))
    pd.DataFrame(rows).to_csv(out/'neighbor_metrics_by_scope.csv',index=False)
    quartiles=[]
    for session,g in frame.groupby('session'):
        for axis in [1,2]:
            column=f'cue_axis_{axis}'
            for end,part in [('low',g[g[column]<=g[column].quantile(.25)]),('high',g[g[column]>=g[column].quantile(.75)])]:
                quartiles.append(dict(session=session,axis=axis,quartile=end,n_units=len(part),
                    n_sua=int(part.decoder_label.eq('sua').sum()),n_d1=int(part.label.eq('D1').sum()),n_d2=int(part.label.eq('D2').sum()),
                    n_cue_supported=int(part.cue_supported.sum()),cue_mean_median=part.cue_mean.median(),
                    outcome_mean_median=part.outcome_mean.median(),rate_median=part.analysis_firing_rate_hz.median(),
                    isi_cv_median=part.analysis_isi_cv.median(),clusters=part.cluster_name.value_counts().to_json()))
    pd.DataFrame(quartiles).to_csv(out/'axis_quartile_descriptions.csv',index=False)


def figures(cfg, out, frame, curves, cv, lag, post, state, evaluations, metrics, scores, neighbor_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    files = []
    palette = plt.get_cmap('tab10').colors
    low_color, high_color = '#3478B8', '#C56C42'
    source = frame.session.eq(cfg.source_session).to_numpy()
    def save(fig,name):
        fig.savefig(out/name,dpi=185,bbox_inches='tight')
        plt.close(fig)
        files.append(name)
    def zero(ax):
        ax.axvline(0,color='#CBD5E1',lw=.8)
        ax.axhline(0,color='#CBD5E1',lw=.8)
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(1,2,figsize=(8.2,3),layout='constrained',sharey=True)
        for j,ax in enumerate(axes):
            ax.plot(lag,state['effects'][:,j],color=palette[j],lw=2)
            zero(ax)
            ax.set(title=f'Cue axis {j+1}',xlabel='Time from go cue (s)')
        axes[0].set_ylabel('Log-rate change / source SD')
        save(fig,'cue_axis_patterns.png')

        cue_limit = max(np.quantile(np.abs(frame.cue_mean),.95),.001)
        for mask,name,title in [(source,'source','December 10'),(~source,'target','December 18')]:
            part, s = frame[mask], scores[mask]
            fig,axes = plt.subplots(2,3,figsize=(11.2,6.6),layout='constrained')
            im = axes[0,0].scatter(s[:,0],s[:,1],c=part.cue_mean,cmap='RdBu_r',norm=TwoSlopeNorm(0,-cue_limit,cue_limit),s=21,alpha=.85)
            fig.colorbar(im,ax=axes[0,0],shrink=.75,label='Log-rate effect')
            axes[0,0].set_title('Go-cue response · 0–1 s')
            for cluster,g in part.groupby('cluster'):
                axes[0,1].scatter(g.cue_axis_1,g.cue_axis_2,color=palette[int(cluster)%10],s=20,label=f'C{cluster+1}',alpha=.8)
            axes[0,1].legend(fontsize=7,ncol=2,loc='best',handletextpad=.2,columnspacing=.6)
            axes[0,1].set_title('Existing clusters')
            for label,g in sorted(part.groupby('label'),key=lambda pair: pair[0]!='Unlabeled'):
                axes[0,2].scatter(g.cue_axis_1,g.cue_axis_2,color=COLORS.get(label,'#94A3B8'),s=23,label=label,alpha=.85)
            axes[0,2].legend(fontsize=8)
            axes[0,2].set_title('Recorded cell type')
            for ax,column,label,transform in [(axes[1,0],'analysis_firing_rate_hz','Firing rate',lambda v:np.log10(v.clip(.01))),
                                               (axes[1,1],'analysis_isi_cv','ISI CV',lambda v:np.log10(v.clip(.01)))]:
                im=ax.scatter(s[:,0],s[:,1],c=transform(part[column]),cmap='viridis',s=21,alpha=.85)
                fig.colorbar(im,ax=ax,shrink=.75,label='log₁₀ Hz' if 'rate' in column else 'log₁₀ CV')
                ax.set_title(label)
            for label,g in part.groupby('cue_supported'):
                axes[1,2].scatter(g.cue_axis_1,g.cue_axis_2,s=23,color='#149787' if label else '#CBD5E1',label='Supported' if label else 'Uncertain',alpha=.85)
            axes[1,2].legend(fontsize=8)
            axes[1,2].set_title('Cue reliability')
            for ax in axes.flat: ax.set(xlabel='Cue axis 1 · source SD',ylabel='Cue axis 2 · source SD')
            fig.suptitle(title + (' · discovery fit' if name=='source' else ' · frozen source axes'),fontsize=12)
            save(fig,f'cue_axes_{name}.png')

        fig,axes = plt.subplots(2,2,figsize=(9,7),layout='constrained')
        limit=max(np.quantile(np.abs(curves[:,0]),.97),.001)
        for row,(mask,title) in enumerate([(source,'December 10'),(~source,'December 18')]):
            for axis in [0,1]:
                order=np.argsort(scores[mask,axis])
                im=axes[row,axis].imshow(curves[mask,0][order],aspect='auto',origin='lower',
                    extent=[lag[0],lag[-1],0,len(order)],cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
                axes[row,axis].axvline(0,color='#334155',lw=.6)
                axes[row,axis].set(title=f'{title} · axis {axis+1}',xlabel='Time from go cue (s)',ylabel='Neurons · low → high score')
        fig.colorbar(im,ax=list(axes.flat),shrink=.7,label='GLM log-rate effect')
        save(fig,'cue_sorted_kernels.png')

        quartiles=[]
        fig,axes = plt.subplots(2,2,figsize=(9,6.2),layout='constrained',sharex=True,sharey=True)
        for row,(mask,title) in enumerate([(source,'December 10'),(~source,'December 18')]):
            for j in [0,1]:
                score=scores[mask,j]
                for end,selection,color in [('Low',score<=np.quantile(score,.25),low_color),('High',score>=np.quantile(score,.75),high_color)]:
                    values=curves[mask,0][selection]
                    q=np.quantile(values,[.25,.5,.75],axis=0)
                    axes[row,j].plot(lag,q[1],color=color,lw=2,label=f'{end} quartile · n={len(values)}')
                    axes[row,j].fill_between(lag,q[0],q[2],color=color,alpha=.13,lw=0)
                    for t,med,ql,qh in zip(lag,q[1],q[0],q[2]):
                        quartiles.append(dict(session=title,axis=j+1,quartile=end,n_units=len(values),lag_s=t,median=med,q25=ql,q75=qh))
                zero(axes[row,j])
                axes[row,j].set(title=f'{title} · axis {j+1}',xlabel='Time from go cue (s)',ylabel='GLM log-rate effect')
                axes[row,j].legend(fontsize=8)
        save(fig,'cue_axis_quartiles.png')
        pd.DataFrame(quartiles).to_csv(out/'cue_quartile_kernels.csv',index=False)

        fig,axes=plt.subplots(2,2,figsize=(8.7,5.8),layout='constrained',sharex=True,sharey='row')
        for col,(mask,title) in enumerate([(source,'December 10'),(~source,'December 18')]):
            score=scores[mask,0]
            for row,(kernel,label) in enumerate([(0,'Go-cue effect'),(3,'Reward − omission')]):
                for end,selected,color in [('Low axis 1',score<=np.quantile(score,.25),low_color),
                                           ('High axis 1',score>=np.quantile(score,.75),high_color)]:
                    q=np.quantile(curves[mask,kernel][selected],[.25,.5,.75],axis=0)
                    axes[row,col].plot(lag,q[1],color=color,lw=2,label=end)
                    axes[row,col].fill_between(lag,q[0],q[2],color=color,alpha=.12,lw=0)
                zero(axes[row,col])
                axes[row,col].set(ylabel=label+' · log rate' if col==0 else '',
                                  title=title if row==0 else '',xlabel='Time from event (s)' if row==1 else '')
                if row==0 and col==0:axes[row,col].legend(fontsize=8)
        save(fig,'cue_outcome_axis1.png')

        fig,axes = plt.subplots(1,3,figsize=(11.2,3.1),layout='constrained')
        for j,(ev,title) in enumerate([('source_heldout','Held-out Dec 10'),('target_direct','Frozen → Dec 18')]):
            part,truth,pred,_=evaluations[ev]
            a,b=truth[:,post].mean(1),pred[1][:,post].mean(1)
            axes[j].scatter(a,b,c=[palette[int(c)%10] for c in part.cluster],s=20,alpha=.7)
            bounds=[min(a.min(),b.min()),max(a.max(),b.max())]
            axes[j].plot(bounds,bounds,color='#CBD5E1',lw=1)
            axes[j].set(title=title,xlabel='Observed cue · 0–1 s',ylabel='Predicted cue · 0–1 s')
            axes[j].text(.04,.95,f'r = {correlation(a,b):.2f}',transform=axes[j].transAxes,va='top')
        evs=['source_heldout','target_direct','target_mean_adjusted']
        for rank,shift,color in [(1,-.12,palette[0]),(2,.12,palette[1])]:
            g=metrics[(metrics.scope=='All')&(metrics.components==rank)].set_index('evaluation').loc[evs]
            vals=g.kernel_r2.to_numpy()
            # Bootstrap intervals need not contain the point estimate.
            axes[2].vlines(np.arange(3)+shift,g.kernel_r2_ci_low,g.kernel_r2_ci_high,color=color,lw=1.5)
            axes[2].scatter(np.arange(3)+shift,vals,s=27,color=color,label=f'{rank} axis'+('' if rank==1 else 'es'))
        axes[2].axhline(0,color='#94A3B8',lw=.8)
        axes[2].set(xticks=range(3),xticklabels=['Dec 10\nheld out','Dec 18\ndirect','Dec 18\ncentered'],ylabel='Full-kernel R²',title='Prediction across neurons')
        axes[2].legend(fontsize=8)
        save(fig,'cue_transfer_validation.png')

        fig,axes=plt.subplots(2,5,figsize=(14,5.3),layout='constrained',sharex=True)
        other_curves=[curves[:,1],-curves[:,2],curves[:,3],curves[:,5],curves[:,6]]
        for row,(mask,title) in enumerate([(source,'Dec 10'),(~source,'Dec 18')]):
            score=scores[mask,0]
            for col,(feature,label,values) in enumerate(zip(OTHER,LABELS,other_curves)):
                for select,color in [(score<=np.quantile(score,.25),low_color),(score>=np.quantile(score,.75),high_color)]:
                    q=np.quantile(values[mask][select],[.25,.5,.75],axis=0)
                    axes[row,col].plot(lag,q[1],color=color,lw=1.6)
                    axes[row,col].fill_between(lag,q[0],q[2],color=color,alpha=.12,lw=0)
                zero(axes[row,col])
                axes[row,col].set(title=label if row==0 else '',xlabel='Event time (s)' if row==1 else '')
                if col==0:axes[row,col].set_ylabel(title+' · log-rate effect')
        fig.suptitle('Other responses along cue axis 1 · low / high quartiles',fontsize=12)
        save(fig,'other_features_cue_axis_1.png')

        fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained',sharey=True)
        for ax,(ev,title) in zip(axes,[('source_heldout_neighbors','Dec 10 · held out'),('target_from_source_neighbors','Dec 18 · source neighbors'),('target_local_neighbors','Dec 18 · local neighbors')]):
            for space,color,shift,label in [('cue_axes',palette[0],-.09,'Cue axes'),('original_embedding',palette[1],.09,'Original embeddings')]:
                g=neighbor_metrics[(neighbor_metrics.space==space)&(neighbor_metrics.evaluation==ev)].set_index('feature').loc[OTHER]
                ax.scatter(g.pearson,np.arange(len(OTHER))+shift,color=color,label=label,s=28)
            ax.axvline(0,color='#CBD5E1',lw=.8)
            ax.set(title=title,xlabel='Neighbor-mean correlation',yticks=range(len(OTHER)),yticklabels=LABELS,xlim=(-.65,.85))
            ax.invert_yaxis()
        axes[0].legend(fontsize=8,loc='lower right')
        save(fig,'other_feature_neighbors.png')

        part=frame[source]
        fig,axes=plt.subplots(2,3,figsize=(10,5.5),layout='constrained')
        for row in [0,1]:
            for col,coordinate in enumerate(['ccf_ap','ccf_dv','ccf_ml']):
                for cluster,g in part.groupby('cluster'):
                    axes[row,col].scatter(g[coordinate],g[f'cue_axis_{row+1}'],color=palette[int(cluster)%10],s=20,alpha=.7)
                axes[row,col].set(xlabel=coordinate.replace('ccf_','CCF ').upper()+' (µm)',ylabel=f'Cue axis {row+1}')
        save(fig,'source_axis_anatomy.png')

        # Source-defined extremes selected on embeddings; show full and validation kernels for transparency.
        chosen=[]
        for mask in [source,~source]:
            positions=np.flatnonzero(mask)
            for axis in [0,1]:
                order=positions[np.argsort(scores[mask,axis])]
                chosen.extend([(int(order[0]),axis,'Low'),(int(order[-1]),axis,'High')])
        fig,axes=plt.subplots(2,4,figsize=(12,5.5),layout='constrained',sharex=True)
        for ax,(idx,axis,end) in zip(axes.flat,chosen):
            row=frame.iloc[idx]
            for curve in cv[idx]:ax.plot(lag,curve,color='#CBD5E1',lw=.9)
            ax.plot(lag,curves[idx,0],color=palette[int(row.cluster)%10],lw=2)
            zero(ax)
            ax.set(title=f'{row.session[5:10]} · unit {row.unit_id} · {row.label}\nAxis {axis+1} {end.lower()} · {row.cluster_name}',xlabel='Go-cue time (s)',ylabel='Log-rate effect')
        save(fig,'example_neuron_kernels.png')
    return files


def report(cfg,out,frame,state,metrics,permutations,files):
    coverage=[]
    for session,g in frame.groupby('session'):
        coverage.append(dict(session=session,n_units=len(g),n_sua=int(g.decoder_label.eq('sua').sum()),
            n_cue_supported=int(g.cue_supported.sum()),n_direction_supported=int(g.cue_direction_supported.sum()),
            within_source_axis_range=int(g.inside_source_axis_range.sum())))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    method=dict(feature='Go-cue conditional log-rate kernel, -1 to +2 s, 151 samples reconstructed from seven saved basis coefficients.',
        model='Two-component PLS2, unit-normalized embeddings and source-fitted per-coordinate standardization. Y is centered but not variance-scaled across lags. One-axis prefix is a prespecified comparison; two axes are primary. No target-dependent model selection.',
        axes='Linear combinations of source-standardized embedding dimensions. Scores expressed in source training SD; temporal loadings show decoder change per source SD. These are supervised functional directions, not newly unsupervised clusters.',
        source_validation='Five-fold neuron splits; scaling and both cue axes refitted without held-out neuron GLM features. Full-source coordinates are discovery visualizations, not held-out evidence.',
        transfer='Frozen source scaler, axes and kernel decoder applied directly to December 18. Separate sensitivity translates the target normalized-embedding mean to the source mean; all exported target units supply the unlabeled target mean. No fitted rotation or target GLM alignment.',
        metrics='Kernel R2 = 1 - sum squared prediction errors / sum squared deviations from evaluation-session mean kernel at each lag. This measures across-neuron kernel variation, not explained spike-count variance. Source-mean skill uses fold-training/source mean kernel as a deployable baseline. Pattern correlation centers each lag across evaluation neurons.',
        reliability='Positive average held-out Cue gain, positive gains in at least two of three folds, all converged. Direction supported additionally needs all validation 0-1s means to agree with final sign. Flags are descriptive, not confidence probabilities.',
        uncertainty='1000 neuron-bootstrap R2 intervals conditional on fitted predictions and recordings; shared neural fluctuations can make them optimistic. 199 whole-kernel row permutations refit source CV axes; target permutations keep the frozen mapping fixed. Unadjusted exploratory one/two-axis p values; session-level replication is not established by unit-level permutations.',
        neighbors='k=10, Euclidean distance in two source-SD cue scores; cosine distance in original unit-normalized embeddings. Source held-out neighbors use axes learned on training neurons and training-neuron feature means. Target-from-source uses source features only. Target-local uses other target neurons and excludes self: within-target coherence, not held-out cross-session feature prediction.',
        other_features='Post-freeze interpretation only. Choice/common, signed right-left and outcome means use 0..1s; lick means use -0.1..0.1s. Choice/common includes common choice/outcome registration response. Outcome is reward-minus-omission averaged across sides, not RPE. All full other kernels retained.',
        limitations=['Same animal and both sessions used for representation training.','Source GLM includes movement; target lacks this covariate.','Other features from the same GLM are not independent biological validation.','Units include SUA and MUA; recorded labels are not treated as ground truth for all untagged neurons.','Full-fit GLM coverage and embedding coverage may differ.','Cue kernel is an event-associated conditional response, not proof of a sensory or causal mechanism.','No physiological residualization or baseline-superiority requirement.'],
        source_hashes={str(p):sha256(p) for p in [resolve(cfg.source)/'embeddings.npz',resolve(cfg.source)/'clusters.csv',resolve(cfg.targets),Path(__file__)]})
    write_json(out/'method.json',method)
    write_json(out/'summary.json',dict(coverage=coverage,primary_metrics=metrics[(metrics.scope=='All')&(metrics.components==2)].to_dict('records'),permutations=permutations))
    captions={
        'cue_axis_patterns.png':'Temporal decoder patterns per source-standard-deviation step. Axis 2 describes additional covariance after axis 1.',
        'cue_axes_source.png':'Discovery fit on all December 10 neurons. Clusters, cell types and physiology did not determine the axes. Color limits clip the outer 5% of cue means for readability.',
        'cue_axes_target.png':'December 18 projected into frozen December 10 axes. Plot limits adapt to the session; score units remain source SD.',
        'cue_sorted_kernels.png':'Individual final-fit cue kernels ordered by each source-defined axis; colors clip the outer 3% of absolute effects.',
        'cue_axis_quartiles.png':'Observed median kernels with interquartile ranges, using each session’s low/high score quartiles. Source panels are discovery descriptions; target panels test frozen-axis organization.',
        'cue_outcome_axis1.png':'Low/high quartiles of the first frozen cue axis. Median and IQR of cue and outcome kernels. Outcome did not determine the axes. Cue and outcome curves come from the same GLM; their association is supporting interpretation, not an independent measurement.',
        'cue_transfer_validation.png':'Source predictions exclude each neuron from fitting. Target predictions use the frozen source decoder. Full-kernel R² concerns variation between neurons; bootstrap intervals are conditional on these recordings.',
        'other_features_cue_axis_1.png':'Blue: low cue-axis-1 quartile; orange: high quartile. Median and IQR of other saved GLM kernels, inspected only after the cue axes were fixed.',
        'other_feature_neighbors.png':'Left: source held-out neuron coherence. Middle: actual source-to-target prediction. Right: within-target neighborhood coherence, using other target neurons’ values. These are separate questions; no superiority criterion is imposed.',
        'source_axis_anatomy.png':'December 10 CCF coordinates, colored by existing cluster. Anatomy is interpreted rather than regressed out; target CCF annotations are unavailable.',
        'example_neuron_kernels.png':'Embedding-axis extremes. Colored: full-recording kernel; gray: three validation-fit kernels. Selected examples are descriptive; see neuron table for reliability and SUA/MUA flags.',
    }
    table=metrics[metrics.scope.eq('All')][['evaluation','components','n_units','kernel_r2','kernel_r2_ci_low','kernel_r2_ci_high','post_mean_r']]
    source_result=metrics.query('evaluation == "source_heldout" and components == 2 and scope == "All"').iloc[0]
    target_result=metrics.query('evaluation == "target_direct" and components == 2 and scope == "All"').iloc[0]
    neighbors_summary=read_table(out/'neighbor_metrics_by_scope.csv')
    outcome=neighbors_summary.query('space == "cue_axes" and evaluation == "target_from_source_neighbors" and feature == "outcome" and scope == "All"').iloc[0]
    findings=(f'<p><b>Findings:</b> the two cue axes explain {source_result.kernel_r2:.1%} of between-neuron full-kernel variation in held-out December 10 neurons. '
        f'Frozen December 18 predictions rank the average 0–1 s cue response positively (r={target_result.post_mean_r:.2f}), '
        f'but the full-kernel prediction remains poorly calibrated (R²={target_result.kernel_r2:.3f}). '
        f'Using December 10 neighbors in the cue projection to predict December 18 outcome responses gives r={outcome.pearson:.2f} and R²={outcome.kernel_r2:.3f}. '
        'The corresponding association also exists in the original embeddings; the cue projection is an interpretation tool, not evidence that it uniquely contains this information. '
        'Full-kernel R² is below zero in December 18: the frozen curve predictions are worse than that recording’s own mean kernel, despite their positive association with response strength.</p>'
        '<p>High cue-axis-1 neurons tend to have a stronger peri-cue response and a more negative early reward-minus-omission response in both sessions. '
        'Responses begin before the cue, so this is not specifically a sensory cue response. Recorded D1 and D2 cells span the source axes. '
        'Source C1 lies toward high axis 1; C6 lies toward low axis 1, with other clusters spanning the continuum. '
        'Rate covaries with the axes and is retained as biological context. The secondary axis also varies with source CCF position. '
        'These observations support exploring functional organization; they do not establish a cell-type-specific mechanism or reward prediction error.</p>')
    body=['<!doctype html><meta charset="utf-8"><title>Go-cue embedding axes</title>',
          '<style>body{font:16px system-ui;color:#243247;max-width:1120px;margin:40px auto;padding:0 24px;line-height:1.55}img{width:100%;margin-top:18px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #dde3ea}a{color:#226ba5}p{max-width:1000px}</style>',
          '<h1>Go-cue response axes</h1><p>December 10 → December 18 · animal 745502 · fine-tuned encoder. One event kernel, two source-learned directions, frozen target projection.</p>',
          '<p>The same 151-sample cue kernel is the only feature used to learn these axes. One-axis and two-axis readouts are reported without selecting on target performance. Existing clusters remain unchanged.</p>',
          findings,pd.DataFrame(coverage).to_html(index=False,border=0),table.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}'),
          '<p><b>Interpretation:</b> source plots use discovery-fit coordinates. Held-out source predictions and frozen target predictions determine whether the apparent organization generalizes. Other GLM responses are supporting interpretation, not independent validation. This is within-animal transfer between two recordings already seen by the encoder.</p>']
    for name in files:
        body.append(f'<img src="{name}" alt="{html.escape(name)}"><p>{html.escape(captions[name])}</p>')
    body.append('<h2>Reusable results</h2><ul>')
    for name in ['neurons.csv','axis_extreme_neurons.csv','cluster_descriptions.csv','axis_associations.csv','metrics.csv',
                 'neighbor_metrics.csv','neighbor_metrics_by_scope.csv','axis_quartile_descriptions.csv','neighbor_predictions.csv','neighbors.csv','permutation_summary.csv','method.json',
                 'observed_kernels.npz','model.npz','source_heldout.npz','target_direct.npz','target_mean_adjusted.npz','figures_and_tables.zip']:
        body.append(f'<li><a href="{name}">{name}</a></li>')
    body.append('</ul>')
    (out/'index.html').write_text('\n'.join(body)+'\n')
    write_json(out/'integrity_checks.json',dict(unique_unit_ids=bool(frame.global_unit_id.is_unique),n_units=len(frame),
        finite_scores=bool(np.isfinite(frame[['cue_axis_1','cue_axis_2']]).all().all()),
        source_folds_disjoint=True,source_prediction_complete=True,serialized_model_reconstruction=True,
        target_neighbor_self_exclusion=True,feature='cue only',components=2))
    with zipfile.ZipFile(out/'figures_and_tables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.suffix in ['.png','.csv','.json','.html','.yaml','.npz']:
                archive.write(path,path.name)


def run(cfg):
    out=resolve(cfg.output)
    out.mkdir(parents=True,exist_ok=True)
    if cfg.get('stage','all')=='report':
        frame=read_table(out/'neurons.csv')
        with np.load(out/'observed_kernels.npz',allow_pickle=False) as z:
            assert list(z['global_unit_id'])==frame.global_unit_id.tolist()
            curves,cv,lag=z['kernels'],z['cue_validation'],z['lags_s']
        with np.load(out/'model.npz',allow_pickle=False) as z:
            state={key:z[key] for key in z.files}
        post=(lag>=-1e-8)&(lag<=1+1e-8)
        evaluations={}
        for ev in ['source_heldout','target_direct','target_mean_adjusted']:
            with np.load(out/f'{ev}.npz',allow_pickle=False) as z:
                part=frame.set_index('global_unit_id',drop=False).loc[z['global_unit_id']].reset_index(drop=True)
                evaluations[ev]=(part,z['truth'],z['predictions'],z['mean_baseline'])
        metrics=read_table(out/'metrics.csv')
        permutations=read_table(out/'permutation_summary.csv').to_dict('records')
        neighbor_metrics=read_table(out/'neighbor_metrics.csv')
        scores=frame[['cue_axis_1','cue_axis_2']].to_numpy()
        summarize_uncertainty(cfg,out,frame,evaluations,metrics,post)
        files=figures(cfg,out,frame,curves,cv,lag,post,state,evaluations,metrics,scores,neighbor_metrics)
        report(cfg,out,frame,state,metrics,permutations,files)
        print('Refreshed saved results:',out,flush=True)
        return
    OmegaConf.save(cfg,out/'config.yaml')
    frame,x,curves,cv,lag,post,target_center=load_data(cfg,out)
    print(f'Loaded {len(frame)} neurons; learning go-cue axes from December 10 only.',flush=True)
    state,evaluations,metrics,scores,neighbor_oof,permutations=fit_and_evaluate(cfg,out,frame,x,curves[:,0],lag,post,target_center)
    neighbor_metrics=interpret(cfg,out,frame,x,scores,neighbor_oof)
    summarize_uncertainty(cfg,out,frame,evaluations,metrics,post)
    files=figures(cfg,out,frame,curves,cv,lag,post,state,evaluations,metrics,scores,neighbor_metrics)
    # Verify saved model, saved target predictions, and unit identities after serialization.
    with np.load(out/'model.npz',allow_pickle=False) as z:
        recovered={key:z[key] for key in z.files}
    mask=frame.session.eq(cfg.target_session).to_numpy()
    with np.load(out/'target_direct.npz',allow_pickle=False) as z:
        assert list(z['global_unit_id'])==frame.loc[mask,'global_unit_id'].tolist()
        assert np.allclose(predict(project(x[mask],recovered),recovered,2),z['predictions'][1])
    assert all(np.isfinite(values[2]).all() for values in evaluations.values())
    report(cfg,out,frame,state,metrics,permutations,files)
    print(metrics[metrics.scope.eq('All')].to_string(index=False),flush=True)
    print('Completed:',out,flush=True)
