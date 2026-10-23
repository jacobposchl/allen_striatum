"""Descriptive cue-aligned Poisson curves, with entire chronological trial folds."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from src.common.analysis import sha256, write_json
from src.data.io import resolve
from src.data.recording import Recording
from src.data.task import recording_path, source_fingerprint


def raised_cosines(times, cfg):
    start, stop = map(float, cfg.response_s)
    offset = float(cfg.basis_log_offset_s)
    centers = np.linspace(np.log(offset), np.log(stop - start + offset), int(cfg.basis_count))
    basis = np.zeros((len(times), len(centers)))
    inside = (times >= start) & (times < stop)
    phase = (np.log(times[inside, None] - start + offset) - centers) / np.diff(centers)[0]
    basis[inside] = .5 * (1 + np.cos(np.pi * np.clip(phase, -1, 1)))
    assert np.allclose(basis[inside].sum(axis=1), 1)
    return basis, np.exp(centers) - offset + start


def fit_curve(total_counts, n_trials, design, dt, cfg):
    """Aggregating identical trial designs preserves the exact Poisson likelihood."""
    exposure = n_trials * dt
    theta = np.zeros(design.shape[1])
    theta[0] = np.log(max(total_counts.sum(), .5) / (exposure * len(total_counts)))
    if total_counts.sum() == 0:
        return theta, dict(status='no_training_spikes', converged=False, iterations=0, bound_hit=False)
    precision = np.r_[0, np.full(design.shape[1] - 1, float(cfg.coefficient_prior_sd) ** -2)]
    # Divide the entire penalized objective by spike count for numerical conditioning.
    scale = max(float(total_counts.sum()), 1)
    def objective(parameters):
        eta = design @ parameters
        expected = exposure * np.exp(eta)
        value = (expected.sum() - total_counts @ eta + .5 * np.sum(precision * parameters**2)) / scale
        gradient = (design.T @ (expected - total_counts) + precision * parameters) / scale
        return value, gradient
    bounds = [(-25, 15)] + [(-15, 15)] * (design.shape[1] - 1)
    result = minimize(objective, theta, jac=True, method='L-BFGS-B', bounds=bounds,
        options=dict(maxiter=int(cfg.max_iter), ftol=1e-13, gtol=1e-8))
    assert np.isfinite(result.x).all()
    bound_hit = any(abs(value - lo) < 1e-5 or abs(value - hi) < 1e-5
                    for value, (lo, hi) in zip(result.x, bounds))
    return result.x, dict(status='fit' if result.success else str(result.message),
        converged=bool(result.success), iterations=int(result.nit), bound_hit=bound_hit)


def curve_scores(observed, predicted):
    variation = np.sum((observed - observed.mean()) ** 2)
    r2 = 1 - np.sum((observed - predicted) ** 2) / variation if variation > 1e-12 else np.nan
    correlation = (np.corrcoef(observed, predicted)[0, 1]
        if np.std(observed) > 1e-10 and np.std(predicted) > 1e-10 else np.nan)
    return float(r2), float(correlation)


def run(cfg):
    assert int(cfg.folds) == 3 and cfg.fold_policy == 'chronological_thirds'
    source, out = resolve(cfg.source), resolve(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(source / 'unit_peak_timing.csv')
    assert units.region.isin(list(cfg.regions)).all()
    trials = pd.read_csv(source / 'trials.csv')
    assert trials.goCue_start_time.is_monotonic_increasing and trials.id.is_unique
    with np.load(source / 'population_psth.npz') as saved:
        assert np.array_equal(saved['global_unit_id'], units.global_unit_id.to_numpy(str))
        assert np.array_equal(saved['trial_id'], trials.id)
        old_centers, old_rates = saved['bin_centers_s'], saved['firing_rate_hz']
    recording = Recording(recording_path(cfg))
    info = dict(recording.info, recording_id=str(recording.info['id']), recording_h5=str(recording.path))
    source_method = json.loads((source / 'method.json').read_text())
    if source_method.get('recording_source') != source_fingerprint(recording.path):
        raise ValueError('Source PSTHs do not match the selected Brainsets recording')
    dt = float(cfg.bin_size_s)
    edges = np.arange(round((float(cfg.response_s[1]) - float(cfg.baseline_s[0])) / dt) + 1) * dt + float(cfg.baseline_s[0])
    times = (edges[:-1] + edges[1:]) / 2
    baseline = (times >= float(cfg.baseline_s[0])) & (times < float(cfg.baseline_s[1]))
    post = (times >= float(cfg.response_s[0])) & (times < float(cfg.response_s[1]))
    fit_bins = baseline | post
    basis, basis_centers = raised_cosines(times, cfg)
    design = np.column_stack([np.ones(len(times)), basis])
    assert np.linalg.matrix_rank(design[fit_bins]) == int(cfg.basis_count) + 1
    assert np.all(basis[baseline] == 0)
    fold_ids = np.empty(len(trials), dtype=int)
    for fold, indices in enumerate(np.array_split(np.arange(len(trials)), 3)):
        fold_ids[indices] = fold
    trials['cue_validation_fold'] = fold_ids
    trials['cue_validation_period'] = np.array(['early', 'middle', 'late'])[fold_ids]
    trials.to_csv(out / 'trials.csv', index=False)

    counts_path = out / 'trial_counts.npy'
    if counts_path.exists() and (out / 'count_source.json').exists():
        cached = json.loads((out / 'count_source.json').read_text())
        assert cached['recording_source'] == source_fingerprint(recording.path)
        assert cached['units_sha256'] == sha256(source / 'unit_peak_timing.csv')
        assert cached['trials_sha256'] == sha256(source / 'trials.csv')
        assert cached['edges'] == edges.tolist()
        counts = np.load(counts_path, mmap_mode='r')
    else:
        source_rows = units.source_unit_row.to_numpy(int)
        assert np.array_equal(recording.units.iloc[source_rows].id, units.id)
        absolute_edges = trials.goCue_start_time.to_numpy()[:, None] + float(info['clock']['origin_s']) + edges
        counts = np.lib.format.open_memmap(counts_path, mode='w+', dtype=np.uint16,
                                         shape=(len(units), len(trials), len(times)))
        for i, row in enumerate(source_rows):
            unit_spikes = recording.spikes(int(row))
            assert np.all(np.diff(unit_spikes) >= 0)
            binned = np.diff(np.searchsorted(unit_spikes, absolute_edges, side='left'), axis=1)
            assert binned.max() <= np.iinfo(np.uint16).max
            counts[i] = binned
        counts.flush()
        write_json(out / 'count_source.json', dict(recording_id=info['recording_id'], recording_source=source_fingerprint(recording.path), units_sha256=sha256(source / 'unit_peak_timing.csv'),
            trials_sha256=sha256(source / 'trials.csv'), edges=edges.tolist()))
    recording.close()
    assert counts.shape == (len(units), len(trials), len(times))
    old_index = np.array([np.argmin(abs(old_centers - time)) for time in times])
    assert np.allclose(old_centers[old_index], times)
    observed = counts.mean(axis=1) / dt
    assert np.allclose(observed, old_rates[:, old_index], atol=1e-9, rtol=1e-9)
    print('Counts verified against existing 20-ms PSTHs', flush=True)

    full_theta = np.zeros((len(units), design.shape[1]))
    fold_theta = np.zeros((len(units), 3, design.shape[1]))
    heldout_rates = np.zeros((len(units), 3, len(times)))
    full_rates = np.zeros_like(observed)
    fold_rates = np.zeros_like(heldout_rates)
    records, fold_records = [], []
    for i, unit in units.iterrows():
        total = np.asarray(counts[i].sum(axis=0), float)
        theta, status = fit_curve(total[fit_bins], len(trials), design[fit_bins], dt, cfg)
        full_theta[i] = theta
        full_rates[i] = np.exp(design @ theta)
        full_r2, full_r = curve_scores(observed[i, post], full_rates[i, post])
        record = dict(global_unit_id=unit.global_unit_id, **status, full_post_spikes=int(total[post].sum()),
            full_fit_spikes=int(total[fit_bins].sum()), full_psth_r2=full_r2, full_psth_correlation=full_r,
            fitted_baseline_hz=float(np.exp(theta[0])), intercept_log_hz=float(theta[0]))
        record.update({f'basis_{b + 1}_coefficient': float(value) for b, value in enumerate(theta[1:])})
        local = []
        for fold in range(3):
            test = fold_ids == fold
            train = ~test
            assert not np.any(test & train) and np.all(test | train)
            train_total = np.asarray(counts[i, train].sum(axis=0), float)
            test_total = np.asarray(counts[i, test].sum(axis=0), float)
            theta, fold_status = fit_curve(train_total[fit_bins], int(train.sum()), design[fit_bins], dt, cfg)
            fold_theta[i, fold] = theta
            pred = np.exp(design @ theta)
            fold_rates[i, fold] = pred
            heldout_rates[i, fold] = test_total / (test.sum() * dt)
            r2, correlation = curve_scores(heldout_rates[i, fold, post], pred[post])
            null_rate = max(train_total[post].sum(), .5) / (train.sum() * post.sum() * dt)
            delta_ll = float(test_total[post] @ np.log(pred[post] / null_rate)
                - test.sum() * dt * np.sum(pred[post] - null_rate))
            n_spikes = int(test_total[post].sum())
            sse = float(np.sum((heldout_rates[i, fold, post] - pred[post]) ** 2))
            sst = float(np.sum((heldout_rates[i, fold, post] - heldout_rates[i, fold, post].mean()) ** 2))
            result = dict(global_unit_id=unit.global_unit_id, fold=fold, period=['early','middle','late'][fold],
                train_trials=int(train.sum()), test_trials=int(test.sum()), test_post_spikes=n_spikes,
                **fold_status, heldout_psth_r2=r2, heldout_psth_correlation=correlation,
                delta_log_likelihood=delta_ll, bits_per_spike=delta_ll / (n_spikes * np.log(2)) if n_spikes else np.nan,
                post_constant_null_hz=float(null_rate), sse=sse, sst=sst)
            local.append(result)
            fold_records.append(result)
        local = pd.DataFrame(local)
        record['positive_likelihood_folds'] = int((local.delta_log_likelihood.gt(0) & local.converged & local.test_post_spikes.gt(0)).sum())
        record['folds_converged'] = int(local.converged.sum())
        record['heldout_psth_r2_median'] = local.heldout_psth_r2.median()
        record['heldout_psth_correlation_median'] = local.heldout_psth_correlation.median()
        record['heldout_psth_r2_pooled'] = 1 - local.sse.sum() / local.sst.sum() if local.sst.sum() > 1e-12 else np.nan
        record['heldout_bits_per_spike'] = local.delta_log_likelihood.sum() / (local.test_post_spikes.sum() * np.log(2)) if local.test_post_spikes.sum() else np.nan
        pair_corr = [curve_scores(heldout_rates[i, a, post], heldout_rates[i, b, post])[1]
                     for a, b in [(0, 1), (0, 2), (1, 2)]]
        record['observed_period_correlation_median'] = pd.Series(pair_corr).median()
        record['coefficient_fold_sd_mean'] = float(fold_theta[i, :, 1:].std(axis=0).mean())
        records.append(record)
        if i % 75 == 0:
            print(f'Fitted full recording and three folds: {i+1}/{len(units)} units', flush=True)
    metrics = pd.DataFrame(records)
    units = units.merge(metrics, on='global_unit_id', validate='one_to_one', sort=False)
    units['fit_row'] = np.arange(len(units))
    fold_metrics = pd.DataFrame(fold_records)
    fold_metrics.to_csv(out / 'validation_folds.csv', index=False)

    # PCA is a visualization of full-data response shapes, not a validation model.
    effects = full_rates[:, post] - np.exp(full_theta[:, :1])
    norms = np.linalg.norm(effects, axis=1)
    usable = units.converged.to_numpy() & units.full_fit_spikes.gt(0).to_numpy() & (norms > 1e-8)
    shapes = effects[usable] / norms[usable, None]
    pca = PCA(n_components=5, svd_solver='full').fit(shapes)
    coordinates = pca.transform(shapes)
    units['response_pc1'], units['response_pc2'] = np.nan, np.nan
    units.loc[usable, 'response_pc1'] = coordinates[:, 0]
    units.loc[usable, 'response_pc2'] = coordinates[:, 1]
    units['shape_amplitude_l2_hz'] = norms
    units.to_csv(out / 'units.csv', index=False)
    coefficient_names = ['intercept_log_hz'] + [f'basis_{i+1}_coefficient' for i in range(int(cfg.basis_count))]
    units[['global_unit_id'] + coefficient_names + ['status']].to_csv(out / 'full_coefficients.csv', index=False)
    pd.DataFrame(basis, columns=[f'basis_{b+1}' for b in range(int(cfg.basis_count))]).assign(time_s=times).to_csv(out / 'basis.csv', index=False)
    np.savez_compressed(out / 'fits.npz', global_unit_id=units.global_unit_id.to_numpy(str), trial_id=trials.id.to_numpy(),
        time_s=times, bin_edges_s=edges, fit_bins=fit_bins, post_bins=post, basis=basis,
        basis_centers_s=basis_centers, full_coefficients=full_theta, validation_coefficients=fold_theta,
        observed_rate_hz=observed, full_fitted_rate_hz=full_rates, heldout_observed_rate_hz=heldout_rates,
        heldout_predicted_rate_hz=fold_rates, trial_fold=fold_ids,
        pca_components=pca.components_, pca_mean=pca.mean_, pca_explained_variance=pca.explained_variance_ratio_)
    summaries = []
    for scope, group in [('All', units), ('CP', units.loc[units.region.eq('CP')]),
                         ('ACB', units.loc[units.region.eq('ACB')]), ('SUA', units.loc[units.decoder_label.eq('sua')])]:
        eligible = group.loc[group.full_post_spikes.gt(0) & group.converged]
        summaries.append(dict(scope=scope, n_units=len(group), n_eligible=len(eligible),
            median_full_psth_r2=eligible.full_psth_r2.median(),
            median_heldout_psth_r2=eligible.heldout_psth_r2_pooled.median(),
            median_heldout_correlation=eligible.heldout_psth_correlation_median.median(),
            median_observed_period_correlation=eligible.observed_period_correlation_median.median(),
            median_heldout_bits_per_spike=eligible.heldout_bits_per_spike.median(),
            positive_all_three=int(eligible.positive_likelihood_folds.eq(3).sum()),
            positive_pooled=int(eligible.heldout_bits_per_spike.gt(0).sum()),
            heldout_r2_positive=int(eligible.heldout_psth_r2_pooled.gt(0).sum()),
            full_r2_above_half=int(eligible.full_psth_r2.ge(.5).sum()),
            positive_fold_counts=eligible.positive_likelihood_folds.value_counts().sort_index().to_dict()))
    summary = dict(animal=str(cfg.animal), session=str(cfg.session), n_units=len(units), n_trials=len(trials),
        trial_counts_by_fold=np.bincount(fold_ids).tolist(), basis_centers_s=basis_centers.tolist(),
        full_fit_status=units.status.value_counts().to_dict(), validation_fit_status=fold_metrics.status.value_counts().to_dict(),
        full_bound_hits=int(units.bound_hit.sum()), validation_bound_hits=int(fold_metrics.bound_hit.sum()),
        pca_explained_variance=pca.explained_variance_ratio_.tolist(), scopes=summaries)
    write_json(out / 'summary.json', summary)
    OmegaConf.save(cfg, out / 'config.yaml')
    method = dict(
        recording_id=info['recording_id'], recording_source=source_fingerprint(recording_path(cfg)),
        target='Descriptive trial-averaged response around the go cue, not a cue-specific causal effect. Choices, rewards, omissions and licking are pooled. This is a separate compact model from the existing rich behavioral GLM cache.',
        observations='All 318 CP/ACB SUA/MUA units and the same 526 trials from the original population PSTH. Rebin original spikes in exact 20-ms bins; check mean counts against the existing 20-ms PSTH cache. Fit -1 to -0.2 s baseline and 0 to 1 s response; exclude the intervening -0.2 to 0 interval from fitting.',
        model='Poisson counts with log expected rate = unit intercept + five raised-cosine basis coefficients. Bases are evenly spaced in log(time+0.02), giving more resolution near the cue. Exposure equals trials times bin duration. Fitting summed counts over trials is exactly equivalent to fitting each trial because the design and exposure per trial are identical. No smoothing of observed counts. A fixed Gaussian penalty SD=2 on each log-rate coefficient stabilizes sparse units; intercept unpenalized. No hyperparameters selected from held-out trials.',
        zero_training_spikes='If a training fold has no spikes, use a constant half-spike/exposure fallback prediction, mark no_training_spikes and exclude from positive-fold counts. It is not an estimated response shape. No-spike full fits are excluded from PCA.',
        validation='Three contiguous chronological trial groups: early, middle, late. Fit on two and predict the untouched third, then rotate. Full-data coefficients are saved separately for later analysis. Differences across periods can reflect changing task-condition mixtures or firing responses, not necessarily intrinsic tuning drift.',
        metrics='Held-out post-cue Poisson log-likelihood gain compared with a constant post-cue rate estimated exclusively from training trials, divided by held-out spike count and ln(2) to give bits/spike. This tests temporal shape beyond a constant post-cue rate. Also report held-out PSTH correlation and R2, full-data fit R2, and pairwise observed-PSTH correlations between independent periods. Pooled R2 sums each fold SSE and within-fold SST. Positive gain in all three folds is an exploratory repeatability flag, not a significance test or a confidence interval. Do not confuse average-PSTH R2 with prediction of trial-to-trial spike noise.',
        coefficient_reliability='The three training fits overlap in trials. Their coefficient spread is descriptive, not an independent-sample confidence interval. Prefer held-out prediction metrics for response reliability.',
        visualization='Observed/fitted heatmaps retain original depth order within CP/ACB. Normalize all panels by the original full-window unit mean with the same observed baseline. PCA uses full-data baseline-subtracted fitted 0–1 s response curves normalized to unit L2 norm, so coordinates emphasize shape. All converged nonzero shapes are projected; weak/unreliable units remain marked. No clustering or NuCLR analysis in this step.',
        source_hashes={str(p):sha256(p) for p in [Path(__file__), source/'unit_peak_timing.csv', source/'trials.csv', source/'population_psth.npz']})
    write_json(out / 'method.json', method)
    from src.experiments.responses.plots import plot
    plot(cfg, out, units, trials, summary)
    with zipfile.ZipFile(out / 'figures_and_tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.suffix in ['.png','.pdf','.csv','.json','.yaml','.html']:
                archive.write(path, path.name)
    print(json.dumps(summary, indent=2), flush=True)
    print('Completed:', out, flush=True)
