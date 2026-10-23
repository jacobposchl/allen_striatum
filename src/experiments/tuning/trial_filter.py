"""Current-trial eligibility for the trial-event GLM, independent of history."""
import math


TRIAL_FILTER_POLICY = 'current_trial_only_v2'
HISTORY_ONLY_REASON = 'history_after_ambiguous_reward'


def _number(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _current_trial_valid(row, info):
    """Recheck the source's own-trial requirements before restoring a row.

    The source audit could overwrite an invalid-record reason with a history
    reason, so the reason alone is insufficient evidence for restoration.
    """
    start, stop, cue = [_number(row, name) for name in
                        ['start_time', 'stop_time', 'goCue_start_time']]
    response = _number(row, 'animal_response')
    if not (all(math.isfinite(t) for t in [start, stop, cue])
            and 0 <= start <= cue < stop <= float(info['duration_s'])
            and stop < float(info['opto_start_s']) - 3
            and response in [0, 1, 2]
            and all(0 <= _number(row, key) <= 1 for key in
                    ['reward_probabilityL', 'reward_probabilityR'])
            and all(_number(row, key) == 0 for key in ['auto_waterL', 'auto_waterR'])):
        return False
    if response == 2:
        return True
    registration = _number(row, 'reward_outcome_time')
    outcome = _number(row, 'outcome_time')
    return (row.get('choice_lick_status') == 'matched'
            and _number(row, 'rewarded') in [0, 1]
            and _number(row, 'reward_delay') == 0
            and cue <= registration < stop
            and registration <= outcome < stop)


def apply_current_trial_filter(records, info):
    """Copy the source audit, restoring valid history-only exclusions.

    Direct reward ambiguities and all other source exclusions remain excluded.
    Source decisions are preserved for provenance; downstream event selection
    and time masks must both use the returned ``usable`` column.
    """
    ambiguous_rows = {int(index) for delivery in info['extra_reward_deliveries']
                      for index in delivery['trial_rows']}
    result = []
    for index, record in enumerate(records):
        row = dict(record)
        flag = str(row['usable']).lower()
        if flag not in ['true', 'false']:
            raise ValueError(f'Invalid source eligibility flag at trial row {index}')
        usable = flag == 'true'
        reason = row.get('exclusion_reason', '')
        reason = reason if isinstance(reason, str) else ''
        row.update(source_usable=usable, source_exclusion_reason=reason,
                   usable=usable, exclusion_reason=reason,
                   restored_after_history_exclusion=False,
                   trial_filter_policy=TRIAL_FILTER_POLICY)
        if not usable and reason == HISTORY_ONLY_REASON:
            if index in ambiguous_rows:
                row['exclusion_reason'] = 'unmatched_reward_delivery'
            elif _current_trial_valid(row, info):
                row.update(usable=True, exclusion_reason='',
                           restored_after_history_exclusion=True)
            else:
                row['exclusion_reason'] = 'invalid_timing_or_task_record'
        result.append(row)
    return result
