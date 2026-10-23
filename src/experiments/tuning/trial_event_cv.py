"""Three held-out fits using A/B/C sub-blocks within consecutive trial blocks."""
import hashlib
import itertools
import json
import os
import socket
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf

from src.data.io import resolve
from src.data.task import recording_path, source_fingerprint
from src.experiments.tuning.trial_event_glm import (
    CONDITIONS, TERMS, fit_poisson, kernel_curves, prepare, summarize_curve,
)
from src.experiments.tuning.trial_filter import TRIAL_FILTER_POLICY


PROTOCOL = 'threefold_subblocks_v1'
MODELS = ['full', 'common', 'constant']
COMPARISONS = [('full', 'constant'), ('full', 'common'), ('common', 'constant')]


def assign_subblocks(trials, n_blocks, n_folds):
    """Partition stored trials before exclusions, without inspecting spikes."""
    if n_folds != 3 or n_blocks < 1 or len(trials) < n_blocks * n_folds:
        raise ValueError('Require three nonempty sub-blocks per parent block')
    result = trials.copy()
    labels = np.empty(len(result), int)
    for block in range(n_blocks):
        lo, hi = block * len(result) // n_blocks, (block + 1) * len(result) // n_blocks
        for fold in range(n_folds):
            a = lo + fold * (hi - lo) // n_folds
            b = lo + (fold + 1) * (hi - lo) // n_folds
            labels[a:b] = block * n_folds + fold
    result['cv_subblock'] = labels
    result['cv_block'] = labels // n_folds
    result['cv_fold'] = labels % n_folds
    return result


def partition_starts(trials, chosen, task_start, task_stop, cfg):
    """Move time boundaries into gaps so full choice windows stay together.

    One-second training guards are placed at test boundaries. Test time and
    final all-data fits retain all valid exposure. Assignment uses only labels
    and times; neural activity never determines the partition.
    """
    guard = float(cfg.validation.boundary_guard_s)
    if guard < 0:
        raise ValueError('Boundary guard must be nonnegative')
    starts = [task_start]
    for group in range(1, int(cfg.validation.n_blocks) * 3):
        nominal = float(trials.loc[trials.cv_subblock.eq(group), 'start_time'].iloc[0])
        before = chosen.loc[chosen.cv_subblock.lt(group), 'event_s']
        after = chosen.loc[chosen.cv_subblock.ge(group), 'event_s']
        lo = float(before.iloc[-1]) + float(cfg.kernel_window_s[1]) + guard if len(before) else task_start
        hi = float(after.iloc[0]) + float(cfg.kernel_window_s[0]) - guard if len(after) else task_stop
        if lo > hi:
            raise ValueError('Insufficient gap to preserve choice windows and the configured guard')
        starts.append(float(np.clip(nominal, lo, hi)))
    if np.any(np.diff(starts) <= 0):
        raise ValueError('Non-increasing validation boundaries')
    return np.asarray(starts)


def partition_audit(trials, chosen, starts, task_stop, exposure, train_exposure, out):
    rows = []
    ends = np.r_[starts[1:], task_stop]
    for group, (start, stop) in enumerate(zip(starts, ends)):
        stored = trials.loc[trials.cv_subblock.eq(group)]
        choices = chosen.loc[chosen.cv_subblock.eq(group) & chosen.eligible_event_s.gt(0)]
        row = dict(subblock=group, block=group // 3, fold=group % 3,
                   start_s=start, stop_s=stop, first_trial_id=int(stored.id.iloc[0]),
                   last_trial_id=int(stored.id.iloc[-1]), stored_trials=len(stored),
                   choices=len(choices), valid_s=float(exposure[group].sum()),
                   choice_window_s=float(exposure[group, 1:].sum()))
        row.update({name: int(choices.condition.eq(k).sum()) for k, name in enumerate(CONDITIONS)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'subblock_audit.csv', index=False)
    folds = []
    for fold in range(3):
        test = chosen.loc[chosen.cv_fold.eq(fold) & chosen.eligible_event_s.gt(0)]
        train = chosen.loc[chosen.cv_fold.ne(fold) & chosen.eligible_event_s.gt(0)]
        row = dict(fold=fold, train_choices=len(train), test_choices=len(test),
                   train_valid_s=float(train_exposure[fold].sum()),
                   test_valid_s=float(exposure[np.arange(len(exposure)) % 3 == fold].sum()),
                   guard_removed_train_s=float(exposure[np.arange(len(exposure)) % 3 != fold].sum()
                                               - train_exposure[fold].sum()))
        for label, subset in [('train', train), ('test', test)]:
            row.update({f'{label}_{name}': int(subset.condition.eq(k).sum()) for k, name in enumerate(CONDITIONS)})
        folds.append(row)
    pd.DataFrame(folds).to_csv(out / 'fold_audit.csv', index=False)


def fit_unit(index, out, cfg):
    began = time.monotonic()
    target = out / 'unit_fits' / f'row_{index:04d}.npz'
    audit_path = target.with_suffix('.json')
    if target.exists() and audit_path.exists():
        return index, 'cached', 0.
    with np.load(out / 'design.npz') as d:
        x = d['matrix']
        full_exposure = d['block_exposure_s'].sum(axis=0)
        train_exposure = d['cv_train_exposure_s']
    full_counts = np.load(out / 'block_counts.npy', mmap_mode='r')[index].sum(axis=0)
    train_counts = np.load(out / 'cv_train_counts.npy', mmap_mode='r')[index]
    options = dict(prior_sd=float(cfg.coefficient_prior_sd), max_iter=int(cfg.max_iter))
    final, status = fit_poisson(full_counts, full_exposure, x, **options)
    audits = [dict(scope='all_data', model='full', fold=-1, **status)]
    coefs = {name: np.full((3, size), np.nan) for name, size in [('full', 21), ('common', 6), ('constant', 1)]}
    for fold in range(3):
        for model in MODELS:
            design = x[:, :coefs[model].shape[1]]
            # Each optimizer starts from its own training counts, never the
            # full-session estimate or any other model fitted to test data.
            beta, diagnostic = fit_poisson(train_counts[fold], train_exposure[fold], design, **options)
            audits.append(dict(scope='cross_validation', model=model, fold=fold, **diagnostic))
            if diagnostic['converged']:
                coefs[model][fold] = beta
    if not status['converged']:
        final[:] = np.nan
    # Atomic files make resuming an interrupted worker unambiguous.
    temporary = target.with_suffix('.partial.npz')
    np.savez_compressed(temporary, all_data=final, **coefs)
    temporary.replace(target)
    audit_path.write_text(json.dumps(audits, indent=2) + '\n')
    return index, status['status'], time.monotonic() - began


def score_counts(counts, exposure, rates):
    """Exact fine-bin Poisson likelihood differences via sufficient statistics.

    Log exposure and log factorial terms cancel between models. These are
    likelihood DIFFERENCES, not absolute fine-bin likelihoods. Zero-spike test
    groups retain finite differences/bits per second; bits per spike is missing.
    """
    y, e = np.asarray(counts), np.asarray(exposure)
    spikes, seconds = float(y.sum()), float(e.sum())
    result = dict(spikes=spikes, exposure_s=seconds, observed_hz=spikes / seconds if seconds else np.nan)
    for name, rate in rates.items():
        expected = float(e @ rate) if np.isfinite(rate).all() else np.nan
        result[f'{name}_expected_spikes'] = expected
        result[f'{name}_predicted_hz'] = expected / seconds if seconds else np.nan
        result[f'{name}_observed_over_predicted'] = spikes / expected if expected > 0 else np.nan
    for model, reference in COMPARISONS:
        a, b = rates[model], rates[reference]
        gain = np.nan
        if np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all():
            gain = float(y @ (np.log(a) - np.log(b)) - e @ (a - b))
        prefix = f'{model}_vs_{reference}'
        result[f'{prefix}_gain_nats'] = gain
        result[f'{prefix}_bits_per_spike'] = gain / (np.log(2) * spikes) if spikes else np.nan
        result[f'{prefix}_bits_per_s'] = gain / (np.log(2) * seconds) if seconds else np.nan
    return result


def curve_stability(curves):
    """Descriptive pairwise differences for a common grid of condition curves."""
    if not np.isfinite(curves).all():
        return dict(pair_rms_hz=np.nan, relative_pair_rms=np.nan, mean_shape_r=np.nan)
    pairs = list(itertools.combinations(curves, 2))
    rms = float(np.mean([np.sqrt(np.mean((a - b) ** 2)) for a, b in pairs]))
    scale = float(np.sqrt(np.mean(curves.mean(axis=0) ** 2)))
    correlations = [float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) for a, b in pairs
                    if np.std(a) > 1e-10 and np.std(b) > 1e-10]
    return dict(pair_rms_hz=rms, relative_pair_rms=rms / scale if scale > 0 else np.nan,
                mean_shape_r=float(np.mean(correlations)) if correlations else np.nan)


def aggregate(cfg, out):
    units = pd.read_csv(out / 'units.csv')
    prep = json.loads((out / 'preparation.json').read_text())
    subblocks = pd.read_csv(out / 'subblock_audit.csv')
    counts = np.load(out / 'block_counts.npy', mmap_mode='r')
    with np.load(out / 'design.npz') as z:
        d = {key: z[key] for key in ['matrix', 'basis', 'lags_s', 'block_exposure_s',
                                    'cv_train_exposure_s', 'cv_fold_for_group', 'condition_lag_trial_support']}
    x, basis, lags, exposure = d['matrix'], d['basis'], d['lags_s'], d['block_exposure_s']
    n, nl = len(units), len(lags)
    shape = (n, 4, nl)
    observed, fitted, heldout = [np.full(shape, np.nan) for _ in range(3)]
    fold_rates = np.full((n, 3, 4, nl), np.nan)
    fold_observed = np.full_like(fold_rates, np.nan)
    full_coefs, fold_coefs = np.full((n, 21), np.nan), np.full((n, 3, 21), np.nan)
    full_curves = np.full((n, 8, nl), np.nan)
    fold_curves = np.full((n, 3, 8, nl), np.nan)
    fold_rows, subblock_rows, pooled_rows, stability_rows, features, audits, summaries = [], [], [], [], [], [], []
    scopes = dict(choice=np.arange(1, len(x)), all_time=np.arange(len(x)))
    scopes.update({name: np.arange(1 + k * nl, 1 + (k + 1) * nl) for k, name in enumerate(CONDITIONS)})
    total_exposure = exposure.sum(axis=0)
    for i, unit in units.iterrows():
        identity = dict(global_unit_id=unit.global_unit_id, id=int(unit.id))
        with np.load(out / 'unit_fits' / f'row_{i:04d}.npz') as z:
            final = z['all_data']
            coefs = {model: z[model] for model in MODELS}
        audit = json.loads((out / 'unit_fits' / f'row_{i:04d}.json').read_text())
        audits.extend([dict(**identity, **row) for row in audit])
        full_coefs[i], fold_coefs[i] = final, coefs['full']
        full_curves[i] = kernel_curves(final, basis)
        fold_curves[i] = kernel_curves(coefs['full'], basis)
        total_counts = counts[i].sum(axis=0)
        observed[i] = np.divide(total_counts[1:], total_exposure[1:],
                               out=np.full(4 * nl, np.nan), where=total_exposure[1:] > 0).reshape(4, nl)
        fitted[i] = np.exp(x[1:] @ final).reshape(4, nl)
        oof_expected = np.zeros(4 * nl)
        local_folds = []
        for fold in range(3):
            rates = {model: np.exp(x[:, :coefs[model].shape[1]] @ coefs[model][fold]) for model in MODELS}
            groups = np.flatnonzero(d['cv_fold_for_group'] == fold)
            y, e = counts[i, groups].sum(axis=0), exposure[groups].sum(axis=0)
            fold_rates[i, fold] = rates['full'][1:].reshape(4, nl)
            fold_observed[i, fold] = np.divide(y[1:], e[1:], out=np.full(4 * nl, np.nan), where=e[1:] > 0).reshape(4, nl)
            oof_expected += e[1:] * rates['full'][1:]
            for scope, pick in scopes.items():
                row = dict(**identity, fold=fold, scope=scope,
                           **score_counts(y[pick], e[pick], {key: value[pick] for key, value in rates.items()}))
                fold_rows.append(row)
                local_folds.append(row)
            for group in groups:
                for scope, pick in scopes.items():
                    location = subblocks.iloc[group]
                    subblock_rows.append(dict(**identity, subblock=int(group), block=int(group // 3), fold=fold,
                        start_s=float(location.start_s), stop_s=float(location.stop_s), scope=scope,
                        **score_counts(counts[i, group, pick], exposure[group, pick],
                                       {key: value[pick] for key, value in rates.items()})))
        heldout[i] = np.divide(oof_expected, total_exposure[1:], out=np.full(4 * nl, np.nan),
                              where=total_exposure[1:] > 0).reshape(4, nl)
        unit_summary = dict(**identity, full_fit_status=audit[0]['status'],
                            full_converged=bool(audit[0]['converged']),
                            full_cv_folds=int(np.isfinite(coefs['full']).all(axis=1).sum()),
                            cv_complete=bool(all(np.isfinite(b).all() for b in coefs.values())))
        for scope in scopes:
            rows = [row for row in local_folds if row['scope'] == scope]
            pooled = dict(**identity, scope=scope, spikes=sum(r['spikes'] for r in rows),
                          exposure_s=sum(r['exposure_s'] for r in rows))
            for model in MODELS:
                pooled[f'{model}_expected_spikes'] = sum(r[f'{model}_expected_spikes'] for r in rows)
                expected = pooled[f'{model}_expected_spikes']
                pooled[f'{model}_observed_over_predicted'] = pooled['spikes'] / expected if expected > 0 else np.nan
            for model, reference in COMPARISONS:
                prefix = f'{model}_vs_{reference}'
                gain = sum(r[f'{prefix}_gain_nats'] for r in rows)
                pooled[f'{prefix}_gain_nats'] = gain
                pooled[f'{prefix}_bits_per_spike'] = gain / (np.log(2) * pooled['spikes']) if pooled['spikes'] else np.nan
                pooled[f'{prefix}_bits_per_s'] = gain / (np.log(2) * pooled['exposure_s']) if pooled['exposure_s'] else np.nan
                values = np.array([r[f'{prefix}_bits_per_spike'] for r in rows])
                pooled[f'{prefix}_positive_folds'] = int(sum(r[f'{prefix}_gain_nats'] > 0 for r in rows))
                pooled[f'{prefix}_scored_folds'] = int(np.isfinite(values).sum())
                pooled[f'{prefix}_fold_min'] = float(values.min())
                pooled[f'{prefix}_fold_max'] = float(values.max())
            pooled_rows.append(pooled)
            if scope == 'choice':
                unit_summary.update({key: value for key, value in pooled.items() if key not in identity and key != 'scope'})
        stability = curve_stability(fold_rates[i])
        unit_summary.update({f'curve_{key}': value for key, value in stability.items()})
        for k, name in enumerate(CONDITIONS):
            stability_rows.append(dict(**identity, condition=name, **curve_stability(fold_rates[i, :, k])))
        for k, name in enumerate(TERMS + CONDITIONS):
            features.append(dict(**identity, feature=name, **summarize_curve(full_curves[i, k], lags, cfg.summary_window_s)))
        summaries.append(unit_summary)
    result = units.merge(pd.DataFrame(summaries).drop(columns='id'), on='global_unit_id', validate='one_to_one')
    result.to_csv(out / 'unit_summary.csv', index=False)
    for name, rows in [('fold_scores', fold_rows), ('subblock_scores', subblock_rows), ('pooled_scores', pooled_rows),
                       ('curve_stability', stability_rows), ('features', features), ('fit_audit', audits)]:
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
    np.savez_compressed(out / 'profiles.npz', global_unit_id=units.global_unit_id.to_numpy(str),
        condition_names=np.asarray(CONDITIONS), term_names=np.asarray(TERMS), lags_s=lags, basis=basis,
        coefficients=full_coefs, cv_coefficients=fold_coefs, curves_log_rate=full_curves,
        cv_curves_log_rate=fold_curves, observed_hz=observed, full_fitted_hz=fitted,
        heldout_predicted_hz=heldout, cv_fitted_hz=fold_rates, cv_observed_hz=fold_observed,
        condition_exposure_s=total_exposure[1:].reshape(4, nl),
        condition_trial_support=d['condition_lag_trial_support'])
    complete = result.loc[result.cv_complete]
    quality = complete.loc[complete.decoder_label.eq('sua') & complete.default_qc.eq(True)]
    def population(frame):
        return dict(n_units=len(frame), median_bits_per_spike_vs_constant=float(frame.full_vs_constant_bits_per_spike.median()),
                    median_bits_per_spike_vs_common=float(frame.full_vs_common_bits_per_spike.median()),
                    positive_pooled_vs_constant=int(frame.full_vs_constant_gain_nats.gt(0).sum()),
                    positive_pooled_vs_common=int(frame.full_vs_common_gain_nats.gt(0).sum()),
                    positive_all_folds_vs_common=int(frame.full_vs_common_positive_folds.eq(3).sum()),
                    median_relative_curve_rms=float(frame.curve_relative_pair_rms.median()))
    summary = dict(animal=str(cfg.animal), session=str(cfg.session), validation_protocol=PROTOCOL,
        n_units=n, n_choice_trials=prep['n_choice_trials'], condition_names=CONDITIONS,
        condition_trial_counts=prep['condition_trial_counts'], restored_trial_ids=prep['restored_trial_ids'],
        excluded_trial_ids=prep['excluded_trial_ids'], full_fit_status=result.full_fit_status.value_counts().to_dict(),
        cv_complete=int(result.cv_complete.sum()), all_complete_units=population(complete),
        qc_passing_single_units=population(quality),
        fit_status_counts=pd.DataFrame(audits).groupby(['scope', 'model', 'status']).size().reset_index(name='count').to_dict('records'))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    method = dict(
        model='One session-wide Poisson log-link GLM per unit. Intercept plus common, side, outcome and interaction, five identical raised cosines each; 20-ms counts, -1 to +2 s from registered choice. No session-varying tuning, drift, movement, licking, probability or history predictors.',
        partition='Nine consecutive near-equal blocks of stored trials, each split into three consecutive near-equal A/B/C sub-blocks. Fold 0 tests A in all blocks; fold 1 B; fold 2 C. Every valid time bin is tested once. Trial and lag-specific support are saved.',
        guard=f'Time boundaries are placed between complete choice windows, using task timing only. Remove {float(cfg.validation.boundary_guard_s):g} s from the training side of each train/test boundary; retain all test time. No choice window is split or clipped by a guard. Full-data fits have no validation guards. This controls short-range boundary proximity, not slow temporal dependence.',
        references='Training-only shared-event model: intercept plus the five common-event coefficients. Training-only constant model: one intercept estimated from all training time. Full and common models use the same fixed event-coefficient prior; intercepts are unpenalized.',
        penalty=f'Fixed coefficient prior SD={float(cfg.coefficient_prior_sd):g}; no regularization search. Any later selection must use only outer-fold training data.',
        scores='Exact Poisson log-likelihood differences, excluding penalties, evaluated on held-out counts and true exposure. Primary scope: choice windows. Also save each condition and all task time. Pooled bits/spike divides summed log gains by summed test spikes times log(2); it is not an unweighted average of fold scores. Bits/second and raw gains remain available for zero-spike test groups.',
        robustness='Mean pairwise RMS difference in Hz between three condition-curve predictions on the same lag grid; normalized by RMS of their mean curve. Shape correlation is secondary and undefined for flat curves. These are descriptive stability measures, not confidence intervals or significance tests.',
        chronology='One held-out score per original sub-block, ordered by time. Local prediction failure can motivate later investigation; it does not establish changing tuning.',
        exclusions='Current-trial filter; 84/110/213 restored, 83/109/212 and their padding excluded. Source decisions retained in trial_audit.csv.',
        final_fit='Refit the fixed model to every eligible time bin after evaluation for the final descriptive curves. All-data fits never supply initialization or coefficients to cross-validation.',
        interpretation='Positive gain is predictive evidence relative to the specified reference, not a significance threshold or causal encoding label. Fold estimates share training data. Missing or zero-spike fits remain flagged. No bootstrap, block deletion, local tuning fits, or ridge sweep.',
        recording_id=prep['source_info']['recording_id'],
        recording_source=source_fingerprint(recording_path(cfg)))
    (out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def run(cfg):
    from src.cli import require_compute
    require_compute('trial_event_glm', cfg)
    if cfg.validation.get('protocol') != PROTOCOL:
        raise ValueError(f'Only {PROTOCOL} validation is supported')
    if list(cfg.prior_sd_sensitivity):
        raise ValueError('Regularization selection is deferred; keep sensitivity empty')
    if cfg.stage not in ['all', 'prepare', 'fit', 'summarize', 'plot']:
        raise ValueError('Unknown stage')
    out = resolve(cfg.output)
    if cfg.stage in ['all', 'prepare']:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Output exists; resume explicitly with fit/summarize/plot')
        out.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out / 'config.yaml')
        prep = prepare(cfg, out)
        print(f'Prepared {prep["n_units"]} units and {prep["n_choice_trials"]} choices', flush=True)
        if cfg.stage == 'prepare':
            return out
    else:
        saved, current = OmegaConf.to_container(OmegaConf.load(out / 'config.yaml')), OmegaConf.to_container(cfg)
        for key in ['stage', 'workers', 'cpu_threads', 'recording', 'recording_root', 'output']:
            saved.pop(key, None)
            current.pop(key, None)
        if saved != current:
            raise ValueError('Resumed configuration differs from saved analysis')
        prep = json.loads((out / 'preparation.json').read_text())
        if prep['source_info'].get('recording_source') != source_fingerprint(recording_path(cfg)):
            raise ValueError('Selected Brainsets recording changed; prepare a new analysis output')
        if prep.get('validation_protocol') != PROTOCOL or prep.get('trial_filter_policy') != TRIAL_FILTER_POLICY:
            raise ValueError('Prepared data use a different filtering or validation protocol')
    if cfg.stage in ['all', 'fit']:
        allocation = int(os.environ.get('SLURM_CPUS_PER_TASK', cfg.workers))
        if int(cfg.workers) * int(cfg.cpu_threads) > allocation:
            raise ValueError('Requested workers exceed the existing CPU allocation')
        (out / 'unit_fits').mkdir(exist_ok=True)
        (out / 'execution.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),
            hostname=socket.gethostname(), workers=int(cfg.workers), cpu_threads=int(cfg.cpu_threads)), indent=2) + '\n')
        jobs = Parallel(n_jobs=int(cfg.workers), return_as='generator_unordered')(
            delayed(fit_unit)(i, out, cfg) for i in range(prep['n_units']))
        for done, (row, status, elapsed) in enumerate(jobs, 1):
            if done == 1 or done % 40 == 0 or done == prep['n_units']:
                print(f'Fitted {done}/{prep["n_units"]} units; row {row}: {status} ({elapsed:.1f}s)', flush=True)
    if cfg.stage in ['all', 'fit', 'summarize']:
        aggregate(cfg, out)
    from src.experiments.tuning.trial_event_cv_plot import plot
    plot(out)
    print(f'Completed {out}', flush=True)
    return out
