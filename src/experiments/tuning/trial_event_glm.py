"""Axis C: factorial choice-aligned Poisson responses on disjoint task time.

For this session the event windows do not overlap. Counts and exposure may
therefore be summed by condition/lag without changing the Poisson likelihood.
Original time blocks remain available for validation without rearranging time.
"""
import json

import numpy as np
import pandas as pd

from src.data.io import resolve
from src.data.recording import Recording
from src.data.task import load_task, recording_path
from src.glm.design import temporal_basis
from src.experiments.tuning.trial_filter import TRIAL_FILTER_POLICY, apply_current_trial_filter


TERMS = ['common', 'side', 'outcome', 'interaction']
CONDITIONS = ['left_reward', 'left_omission', 'right_reward', 'right_omission']
CODING = np.array([[1, .5, .5, .25], [1, .5, -.5, -.25],
                   [1, -.5, .5, -.25], [1, -.5, -.5, .25]])


def fit_poisson(y, exposure, design, prior_sd=2., max_iter=100, start=None):
    """Newton fit of the exact summed likelihood; intercept is unpenalized."""
    y, exposure = np.asarray(y, float), np.asarray(exposure, float)
    status = dict(converged=False, iterations=0, max_scaled_gradient=np.nan,
                  hessian_condition=np.nan, status='no_spikes')
    if y.sum() <= 0 or exposure.sum() <= 0:
        return np.full(design.shape[1], np.nan), status
    precision = np.r_[0., np.full(design.shape[1] - 1, prior_sd ** -2)]
    theta = np.zeros(design.shape[1])
    theta[0] = np.log(y.sum() / exposure.sum())
    if start is not None and np.isfinite(start).all():
        theta = np.array(start, copy=True)
    scale = max(y.sum(), 1.)

    def evaluate(beta):
        eta = design @ beta
        if eta.max() > 500 or not np.isfinite(eta).all():
            return np.inf, None
        expected = exposure * np.exp(eta)
        return expected.sum() - y @ eta + .5 * np.dot(precision, beta * beta), expected

    status['status'] = 'max_iterations'
    for iteration in range(max_iter):
        value, expected = evaluate(theta)
        gradient = design.T @ (expected - y) + precision * theta
        hessian = design.T @ (expected[:, None] * design) + np.diag(precision)
        scaled = float(np.max(abs(gradient)) / scale)
        status.update(iterations=iteration, max_scaled_gradient=scaled)
        if scaled < 1e-8:
            status.update(converged=True, status='fit')
            break
        direction = np.linalg.solve(hessian, gradient)
        step = 1.
        for _ in range(40):
            candidate = theta - step * direction
            proposed, _ = evaluate(candidate)
            if proposed <= value - 1e-4 * step * np.dot(gradient, direction) + 1e-10:
                theta = candidate
                break
            step *= .5
        else:
            status['status'] = 'line_search_failed'
            break
    value, expected = evaluate(theta)
    gradient = design.T @ (expected - y) + precision * theta
    status['max_scaled_gradient'] = float(np.max(abs(gradient)) / scale)
    status['converged'] = status['max_scaled_gradient'] < 1e-7
    if status['converged']:
        status['status'] = 'fit'
    status['hessian_condition'] = float(np.linalg.cond(
        design.T @ (expected[:, None] * design) + np.diag(precision)))
    return theta, status


def kernel_curves(theta, basis):
    coefficients = np.asarray(theta)[..., 1:].reshape(*np.shape(theta)[:-1], 4, basis.shape[1])
    terms = coefficients @ basis.T
    conditions = np.einsum('ct,...tl->...cl', CODING, terms)
    return np.concatenate([terms, conditions], axis=-2)


def summarize_curve(curve, lags, window):
    """Signed effects and descriptive half-peak lobe, without tuning labels."""
    names = ['mean_log_rate', 'peak_log_rate', 'peak_time_s', 'half_peak_width_s',
             'peak_at_boundary', 'width_truncated', 'same_sign_lobes', 'flat']
    result = dict.fromkeys(names, np.nan)
    if not np.isfinite(curve).all():
        return result
    use = (lags >= window[0]) & (lags < window[1])
    result['mean_log_rate'] = float(curve[use].mean())
    peak = int(np.argmax(abs(curve)))
    magnitude = float(curve[peak])
    flat = abs(magnitude) < 1e-4
    result.update(peak_log_rate=magnitude, flat=int(flat))
    if flat:
        return result
    high = curve * np.sign(magnitude) >= abs(magnitude) / 2
    lo, hi = peak, peak + 1
    while lo > 0 and high[lo - 1]:
        lo -= 1
    while hi < len(curve) and high[hi]:
        hi += 1
    lobes = int(np.sum(np.diff(np.r_[False, high, False].astype(int)) == 1))
    result.update(peak_time_s=float(lags[peak]), half_peak_width_s=float((hi - lo) * (lags[1] - lags[0])),
                  peak_at_boundary=int(peak in [0, len(curve) - 1]),
                  width_truncated=int(lo == 0 or hi == len(curve)), same_sign_lobes=lobes)
    return result


def prepare(cfg, out):
    info, units, trials = load_task(cfg)
    if cfg.trial_filter_policy != TRIAL_FILTER_POLICY:
        raise ValueError('Unsupported trial filter policy')
    trials = pd.DataFrame(apply_current_trial_filter(trials.to_dict(orient='records'), info))
    from src.experiments.tuning.trial_event_cv import PROTOCOL
    if cfg.validation.get('protocol') != PROTOCOL:
        raise ValueError(f'Only {PROTOCOL} validation is supported')
    from src.experiments.tuning.trial_event_cv import assign_subblocks, partition_starts, partition_audit
    trials = assign_subblocks(trials, int(cfg.validation.n_blocks), int(cfg.validation.n_folds))
    units = units.loc[units.ccf_acronym.isin(list(cfg.regions)) & units.decoder_label.isin(list(cfg.sorting))].copy()
    units['region_order'] = units.ccf_acronym.map({name: i for i, name in enumerate(cfg.regions)})
    units = units.sort_values(['region_order', 'depth', 'id'], ascending=[True, False, True]).reset_index(drop=True)
    if not units.global_unit_id.is_unique or not units.global_unit_id.str.startswith(f'{cfg.animal}/{cfg.session}/').all():
        raise ValueError('Unit identity/session mismatch')
    dt = float(cfg.bin_size_s)
    start, stop = map(float, cfg.kernel_window_s)
    nlag = int(round((stop - start) / dt))
    if not np.isclose(nlag * dt, stop - start) or cfg.basis_count != 5:
        raise ValueError('Require five bases and an integer number of lag bins')
    lag_edges = start + np.arange(nlag + 1) * dt
    lags = (lag_edges[:-1] + lag_edges[1:]) / 2
    basis = temporal_basis(lags, start, stop, int(cfg.basis_count))
    design = np.zeros((1 + 4 * nlag, 21))
    design[:, 0] = 1
    for k in range(4):
        design[1 + k * nlag:1 + (k + 1) * nlag, 1:] = np.hstack([basis * weight for weight in CODING[k]])
    chosen = trials.loc[trials.usable & trials.animal_response.isin([0, 1])].copy().reset_index(drop=True)
    chosen['event_s'] = chosen.reward_outcome_time
    chosen['condition'] = (2 * chosen.animal_response.astype(int) + (1 - chosen.rewarded)).astype(int)
    events = chosen.event_s.to_numpy()
    # This exact aggregation is valid only for disjoint event windows.
    if np.any(np.diff(events) < stop - start):
        raise ValueError('Overlapping event windows require a continuous summed-design solver')
    task_start, task_stop = float(trials.loc[trials.usable, 'start_time'].min()), float(info['duration_s'])
    pads = info['excluded_trial_padding_s']
    bad = [(r.start_time - pads['before'], r.stop_time + pads['after'])
           for r in trials.loc[~trials.usable].itertuples()]
    epoch_edges = np.linspace(0, task_stop, 4)
    block_starts = partition_starts(trials, chosen, task_start, task_stop, cfg)
    guard = float(cfg.validation.boundary_guard_s)
    guard_edges = np.r_[block_starts[1:] - guard, block_starts[1:] + guard]
    # Use event-aligned 20-ms bins inside each event and 20-ms bins elsewhere.
    # Split at validity/block/epoch boundaries, preserving a disjoint time axis.
    regular = np.arange(int(np.ceil(task_stop / dt)) + 1) * dt
    owner = np.searchsorted(events + start, regular, side='right') - 1
    active = (owner >= 0) & (regular < events[owner.clip(0)] + stop)
    edges = np.sort(np.unique(np.r_[regular[~active], (events[:, None] + lag_edges).ravel(),
                                        task_start, task_stop, np.asarray(bad).ravel(), block_starts, guard_edges, epoch_edges]))
    edges = edges[(edges >= task_start) & (edges <= task_stop)]
    edges = edges[np.r_[True, np.diff(edges) > 1e-8]]
    centers, lengths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    valid = np.ones(len(centers), bool)
    for lo, hi in bad:
        valid &= (centers < lo) | (centers >= hi)
    owner = np.searchsorted(events + start, centers, side='right') - 1
    active = (owner >= 0) & (centers < events[owner.clip(0)] + stop)
    lag_index = np.floor((centers - events[owner.clip(0)] - start) / dt).astype(int).clip(0, nlag - 1)
    category = np.where(active, 1 + chosen.condition.to_numpy()[owner.clip(0)] * nlag + lag_index, 0)
    group = np.searchsorted(block_starts, centers, side='right') - 1
    epoch = np.searchsorted(epoch_edges, centers, side='right') - 1
    nblocks, ncats = len(block_starts), len(design)
    cv_metadata = {}
    fold_for_group = np.arange(nblocks) % 3
    fold_for_bin = fold_for_group[group]
    train_valid = np.array([valid & (fold_for_bin != f) for f in range(3)])
    for f in range(3):
        for j, boundary in enumerate(block_starts[1:], 1):
            if f in [fold_for_group[j - 1], fold_for_group[j]]:
                train_valid[f] &= abs(centers - boundary) >= guard
        if np.any(valid & (fold_for_bin != f) & active & ~train_valid[f]):
            raise ValueError('A training guard clips a choice window; revise partition boundaries')
    cv_exposure = np.array([np.bincount(category[m], weights=lengths[m], minlength=ncats)
                            for m in train_valid])
    cv_metadata = dict(cv_train_valid=train_valid, cv_train_exposure_s=cv_exposure,
                       cv_fold_for_group=fold_for_group, cv_fold_for_bin=fold_for_bin,
                       cv_subblock_starts_s=block_starts)
    exposure = np.bincount((group * ncats + category)[valid], weights=lengths[valid], minlength=nblocks * ncats).reshape(nblocks, ncats)
    epoch_exposure = np.bincount((epoch * ncats + category)[valid], weights=lengths[valid], minlength=3 * ncats).reshape(3, ncats)
    event_exposure = np.bincount((owner * nlag + lag_index)[valid & active], weights=lengths[valid & active],
                                 minlength=len(chosen) * nlag).reshape(len(chosen), nlag)
    support = np.array([(event_exposure[chosen.condition.eq(k)] > 0).sum(axis=0) for k in range(4)])
    chosen['eligible_event_s'] = event_exposure.sum(axis=1)
    chosen['epoch'] = np.searchsorted(epoch_edges, chosen.event_s, side='right') - 1
    chosen['time_block'] = np.searchsorted(block_starts, chosen.event_s, side='right') - 1
    if not np.array_equal(chosen.time_block, chosen.cv_subblock):
        raise ValueError('Time partition disagrees with trial partition')
    epoch_counts = np.array([[int(((chosen.epoch == e) & (chosen.condition == k) & chosen.eligible_event_s.gt(0)).sum())
                              for k in range(4)] for e in range(3)])
    recording = Recording(recording_path(cfg))
    source_rows = units.source_unit_row.to_numpy(int)
    if not np.array_equal(recording.units.iloc[source_rows].id, units.id):
        raise ValueError('Source unit order mismatch')
    absolute_edges = edges + float(info['clock']['origin_s'])
    fine = np.lib.format.open_memmap(out / 'fine_counts.npy', mode='w+', dtype=np.uint16, shape=(len(units), len(centers)))
    counts = np.zeros((len(units), nblocks, ncats), np.float64)
    epoch_y = np.zeros((len(units), 3, ncats), np.float64)
    cv_counts = np.zeros((len(units), 3, ncats), np.float64)
    for j, row in enumerate(source_rows):
        unit_spikes = recording.spikes(int(row))
        observed = np.diff(np.searchsorted(unit_spikes, absolute_edges, side='left'))
        if observed.max() > np.iinfo(np.uint16).max:
            raise ValueError('Count overflow')
        fine[j] = observed
        counts[j] = np.bincount((group * ncats + category)[valid], weights=observed[valid], minlength=nblocks * ncats).reshape(nblocks, ncats)
        epoch_y[j] = np.bincount((epoch * ncats + category)[valid], weights=observed[valid], minlength=3 * ncats).reshape(3, ncats)
        for f, mask in enumerate(train_valid):
            cv_counts[j, f] = np.bincount(category[mask], weights=observed[mask], minlength=ncats)
    recording.close()
    fine.flush()
    units['eligible_spikes'] = counts.sum(axis=(1, 2)).astype(int)
    units['low_spike_support'] = units.eligible_spikes < int(cfg.validation.minimum_spikes)
    units.to_csv(out / 'units.csv', index=False)
    trials.to_csv(out / 'trial_audit.csv', index=False)
    chosen.to_csv(out / 'choice_trials.csv', index=False)
    np.save(out / 'block_counts.npy', counts)
    np.save(out / 'epoch_counts.npy', epoch_y)
    np.save(out / 'cv_train_counts.npy', cv_counts)
    partition_audit(trials, chosen, block_starts, task_stop, exposure, cv_exposure, out)
    np.savez_compressed(out / 'design.npz', matrix=design, basis=basis, lags_s=lags, lag_edges_s=lag_edges,
                        block_exposure_s=exposure, epoch_exposure_s=epoch_exposure, condition_lag_trial_support=support,
                        epoch_condition_trials=epoch_counts, event_exposure_s=event_exposure,
                        edges_s=edges, centers_s=centers, valid=valid, category=category, group=group, epoch=epoch,
                        condition_names=np.asarray(CONDITIONS), term_names=np.asarray(TERMS), coding=CODING,
                        **cv_metadata)
    condition_counts = [int(((chosen.condition == k) & chosen.eligible_event_s.gt(0)).sum()) for k in range(4)]
    weighted = np.sqrt(exposure.sum(axis=0))[:, None] * design
    preparation = dict(n_units=len(units), n_choice_trials=len(chosen), n_usable_trials=int(trials.usable.sum()),
        trial_filter_policy=TRIAL_FILTER_POLICY,
        validation_protocol=PROTOCOL,
        restored_trial_ids=trials.loc[trials.restored_after_history_exclusion, 'id'].astype(int).tolist(),
        excluded_trial_ids=trials.loc[~trials.usable, 'id'].astype(int).tolist(),
        excluded_intervals_s=bad,
        condition_names=CONDITIONS, condition_trial_counts=condition_counts, epoch_condition_trials=epoch_counts.tolist(),
        n_time_blocks=nblocks, n_fine_bins=len(centers), task_duration_s=task_stop,
        eligible_s=float(exposure.sum()), baseline_s=float(exposure[:, 0].sum()),
        minimum_choice_spacing_s=float(np.diff(events).min()), design_rank=int(np.linalg.matrix_rank(weighted)),
        design_condition=float(np.linalg.cond(weighted)), source_info=info)
    (out / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
    return preparation


def run(cfg):
    from src.experiments.tuning.trial_event_cv import PROTOCOL, run as run_cv
    if cfg.validation.get('protocol') != PROTOCOL:
        raise ValueError(f'Only {PROTOCOL} validation is supported')
    return run_cv(cfg)
