"""Task observations and eligibility derived directly from Brainsets recordings.

Times in this module are recording-relative. Scientific filters are explicit:
``audit_trials`` retains the history-aware task audit; individual analyses may
apply their own current-trial filter without constructing or fitting a GLM.
"""
import numpy as np
import pandas as pd

from src.data.recording import Recording, resolve_recording

CHOICE_MATCH_TOLERANCE_S = .01


def read_trials(recording, origin):
    required = ("id", "start_time", "stop_time", "goCue_start_time",
                "reward_outcome_time", "animal_response", "rewarded_historyL",
                "rewarded_historyR", "reward_delay", "auto_waterL", "auto_waterR",
                "reward_probabilityL", "reward_probabilityR")
    group = recording.trials
    missing = [name for name in required if name not in group]
    if missing:
        raise ValueError(f"Missing behavioral fields required by task observations: {missing}")
    trials = group.copy()
    trials["stored_trial_index"] = np.arange(len(trials))
    for name in ("start_time", "stop_time", "goCue_start_time", "reward_outcome_time"):
        trials[name] = pd.to_numeric(trials[name], errors="coerce") - origin
    for name in required[5:]:
        trials[name] = pd.to_numeric(trials[name], errors="coerce")
    reward_known = trials.rewarded_historyL.isin([0, 1]) & trials.rewarded_historyR.isin([0, 1])
    trials["rewarded"] = np.where(reward_known,
                                  (trials.rewarded_historyL.eq(1) | trials.rewarded_historyR.eq(1)).astype(int), -1)
    trials["outcome_time"] = np.where(trials.animal_response.isin([0, 1]),
                                      trials.reward_outcome_time + trials.reward_delay, np.nan)
    return trials

def match_choice_licks(trials, side_licks):
    """Match a registered response to exactly one event on its recorded side."""
    trials = trials.copy()
    removed = [np.zeros(len(times), bool) for times in side_licks]
    trials["choice_time"] = np.nan
    trials["choice_lick_time_error_s"] = np.nan
    trials["choice_lick_status"] = "no_response"
    for i, trial in trials.loc[trials.animal_response.isin([0, 1])].iterrows():
        side = int(trial.animal_response)
        licks = side_licks[side]
        registration = trial.reward_outcome_time
        trials.loc[i, "choice_lick_status"] = "unmatched"
        if not np.isfinite([registration, trial.goCue_start_time, trial.stop_time]).all():
            continue
        lo = np.searchsorted(licks, max(trial.goCue_start_time, registration - CHOICE_MATCH_TOLERANCE_S))
        hi = np.searchsorted(licks, min(trial.stop_time, registration + CHOICE_MATCH_TOLERANCE_S), side="right")
        candidates = np.arange(lo, hi)
        candidates = candidates[licks[candidates] < trial.stop_time]
        if not len(candidates):
            continue
        error = np.abs(licks[candidates] - registration)
        nearest = candidates[np.isclose(error, error.min(), rtol=0, atol=1e-10)]
        if len(nearest) != 1 or removed[side][nearest[0]]:
            trials.loc[i, "choice_lick_status"] = "ambiguous"
            continue
        selected = int(nearest[0])
        removed[side][selected] = True
        trials.loc[i, "choice_time"] = licks[selected]
        trials.loc[i, "choice_lick_time_error_s"] = licks[selected] - registration
        trials.loc[i, "choice_lick_status"] = "matched"
    licks = np.sort(np.concatenate(side_licks))
    additional = np.sort(np.concatenate([times[~mask] for times, mask in zip(side_licks, removed)]))
    for name, times in (("first_lick_time", licks), ("first_additional_lick_time", additional)):
        trials[name] = np.nan
        for i, trial in trials.iterrows():
            if not np.isfinite([trial.goCue_start_time, trial.stop_time]).all():
                continue
            index = np.searchsorted(times, trial.goCue_start_time)
            if index < len(times) and times[index] < trial.stop_time:
                trials.loc[i, name] = times[index]
    return trials, additional, sum(int(mask.sum()) for mask in removed)


def audit_trials(recording, block_trials=20, strict_reward_timing=False, allow_no_opto=False):
    """Recreate the history-aware behavioral audit from canonical observations."""
    clock = dict(recording.info['clock'])
    origin = float(clock['origin_s'])
    trials = read_trials(recording, origin)
    sides = [recording.events(f'{side}_lick') - origin for side in ['left', 'right']]
    trials, _, _ = match_choice_licks(trials, sides)
    choice = trials.animal_response.isin([0, 1])
    valid = (trials.start_time.ge(0) & trials.stop_time.gt(trials.start_time)
             & trials.goCue_start_time.ge(trials.start_time) & trials.goCue_start_time.lt(trials.stop_time)
             & trials.animal_response.isin([0, 1, 2]) & trials.reward_probabilityL.between(0, 1)
             & trials.reward_probabilityR.between(0, 1) & trials.auto_waterL.eq(0) & trials.auto_waterR.eq(0))
    valid &= (~choice | (trials.choice_lick_status.eq('matched') & trials.rewarded.isin([0, 1])
                        & trials.reward_delay.eq(0) & trials.outcome_time.lt(trials.stop_time)))
    opto_source = recording.opto_start
    if opto_source is not None:
        opto = float(opto_source) - origin
        opto_boundary_source = 'observed_or_session_metadata'
    elif recording.has_events('opto_start') or allow_no_opto:
        # An explicitly recorded empty stream differs from an unavailable stream.
        opto = float(clock['duration_s'])
        opto_boundary_source = 'recorded_empty' if recording.has_events('opto_start') else 'explicitly_assumed_absent'
    else:
        raise ValueError('Optotagging boundary unavailable; set allow_no_opto=true only for a known untagged recording')
    valid &= trials.stop_time.lt(opto - 3)
    extra_deliveries, ambiguous, timing_mismatch = [], set(), set()
    for side, name in enumerate(['left', 'right']):
        actual = recording.events(f'{name}_reward') - origin
        expected_trials = trials.loc[valid & trials.animal_response.eq(side) & trials.rewarded.eq(1)]
        expected = expected_trials.outcome_time.to_numpy()
        matched = (np.array([actual[np.argmin(abs(actual - v))] for v in expected])
                   if len(actual) else np.full(len(expected), np.nan))
        mismatch = ~np.isfinite(matched) | (abs(matched - expected) > .02)
        if strict_reward_timing and mismatch.any():
            raise ValueError('A rewarded trial lacks a matching delivery within one bin')
        timing_mismatch.update(expected_trials.index[mismatch].tolist())
        for value in actual:
            if np.any(abs(matched - value) < 1e-6):
                continue
            containing = trials.index[(trials.start_time <= value) & (trials.stop_time > value)].tolist()
            extra_deliveries.append(dict(side=name, time_s=float(value), trial_rows=containing))
            ambiguous.update(containing)
    following = {i + 1 for i in ambiguous | timing_mismatch if i + 1 < len(trials)}
    trials['exclusion_reason'] = np.where(valid, '', 'invalid_timing_or_task_record')
    trials.loc[list(ambiguous), 'exclusion_reason'] = 'unmatched_reward_delivery'
    trials.loc[list(timing_mismatch), 'exclusion_reason'] = 'reward_delivery_timing_mismatch'
    trials.loc[list(following - ambiguous - timing_mismatch), 'exclusion_reason'] = 'history_after_ambiguous_reward'
    trials['usable'] = valid & ~trials.index.isin(ambiguous | timing_mismatch | following)
    trials['block'] = trials.stored_trial_index // int(block_trials)
    trials['fold'] = trials.block % 3
    trials['previous_left'] = trials.animal_response.shift(1).eq(0).astype(float)
    trials['previous_right'] = trials.animal_response.shift(1).eq(1).astype(float)
    trials['previous_reward'] = trials.rewarded.shift(1).eq(1).astype(float)
    trials['previous_omission'] = (trials.rewarded.shift(1).eq(0) & trials.animal_response.shift(1).isin([0, 1])).astype(float)
    trials['previous_response'] = trials.previous_left + trials.previous_right
    trials['previous_side_contrast'] = .5 * (trials.previous_left - trials.previous_right)
    trials['previous_outcome_contrast'] = .5 * (trials.previous_reward - trials.previous_omission)
    if not trials.usable.any():
        raise ValueError('No eligible behavioral trials')
    info = dict(recording.info, clock=clock, recording_id=str(recording.info['id']),
                recording_h5=str(recording.path),
                duration_s=float(trials.loc[trials.usable, 'stop_time'].max()),
                opto_start_s=opto, opto_boundary_source=opto_boundary_source,
                extra_reward_deliveries=extra_deliveries,
                excluded_trial_padding_s=dict(before=1, after=2))
    return trials, info


def selected_units(recording, regions=None, sorting=('sua', 'mua'), allow_missing_anatomy=False):
    """Retain source identities and apply explicit anatomy/sorting selection."""
    units = recording.units.copy()
    anatomy = units.ccf_acronym.notna() & ~units.ccf_acronym.isin(['', 'unknown', 'Unassigned'])
    units['anatomy_available'] = anatomy
    region = units.ccf_acronym.isin(list(regions)) if regions is not None else np.ones(len(units), bool)
    if allow_missing_anatomy:
        region |= ~anatomy
    units['included'] = region & units.decoder_label.isin(list(sorting))
    return units, units.loc[units.included].reset_index(drop=True)


def recording_path(cfg):
    """Resolve an explicit HDF5 selection, never a legacy cache directory."""
    return resolve_recording(cfg.recording, root=cfg.get('recording_root'))


def load_task(cfg):
    """Read task observations and selected units without any fitted-model inputs."""
    path = recording_path(cfg)
    with Recording(path) as recording:
        trials, info = audit_trials(recording, allow_no_opto=bool(cfg.get('allow_no_opto', False)))
        _, units = selected_units(recording, cfg.get('regions'), cfg.get('sorting', ['sua', 'mua']))
    info['n_units'] = len(units)
    info['recording_source'] = source_fingerprint(path)
    return info, units, trials


def binned_speed(recording, edges, interpolate=0):
    """Average physical absolute speed; preserve unsupported bins as missing."""
    timestamps, values = recording.signal('speed', domain='behavior')
    timestamps = timestamps - float(recording.info['clock']['origin_s'])
    values = np.abs(np.asarray(values, float).reshape(-1))
    dt = float(edges[1] - edges[0])
    bins = np.floor((timestamps - edges[0]) / dt).astype(int)
    good = (bins >= 0) & (bins < len(edges) - 1) & np.isfinite(values)
    samples = np.bincount(bins[good], minlength=len(edges) - 1)
    sums = np.bincount(bins[good], weights=values[good], minlength=len(edges) - 1)
    speed = np.divide(sums, samples, out=np.full(len(samples), np.nan), where=samples > 0)
    if interpolate:
        speed = pd.Series(speed).interpolate(limit=interpolate, limit_area='inside').to_numpy()
    return speed


def task_grid(recording, trials, info, bin_size=.02):
    """History-aware task mask on an explicit grid, including movement coverage."""
    edges = np.arange(int(np.ceil(info['duration_s'] / bin_size)) + 1) * bin_size
    centers = (edges[:-1] + edges[1:]) / 2
    ti = np.searchsorted(trials.start_time.to_numpy(), edges[:-1], side='right') - 1
    safe = ti.clip(0)
    usable = (ti >= 0) & trials.usable.to_numpy()[safe] & (edges[1:] <= trials.stop_time.to_numpy()[safe])
    for row in trials.loc[~trials.usable].itertuples():
        usable &= (centers < row.start_time - 1) | (centers > row.stop_time + 2)
    speed = binned_speed(recording, edges, interpolate=4)
    usable &= np.isfinite(speed)
    return dict(edges=edges, trial_index=ti, speed=speed, usable=usable)


def spike_counts(recording, units, edges):
    """Exact half-open bin counts, read one source unit at a time."""
    absolute = np.asarray(edges) + float(recording.info['clock']['origin_s'])
    counts = np.empty((len(units), len(edges) - 1), dtype=np.uint16)
    for j, row in enumerate(units.source_unit_row):
        values = np.diff(np.searchsorted(recording.spikes(int(row)), absolute, side='left'))
        if values.max(initial=0) > np.iinfo(np.uint16).max:
            raise ValueError('Spike count overflow')
        counts[j] = values
    return counts


def source_fingerprint(path):
    """Portable source identity; active HDF5 exports embed their source digest."""
    with Recording(path) as recording:
        return dict(identity=recording.identity, size=recording.path.stat().st_size)
