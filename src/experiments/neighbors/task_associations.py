"""Exploratory conditional task associations of 10-second neighbor scores.

OLS effects with Bartlett/Newey-West covariance on the original time grid.
Absent target windows contribute zero scores to covariance, retaining true lags.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2, false_discovery_control
from sklearn.preprocessing import SplineTransformer
from omegaconf import OmegaConf

from src.experiments.neighbors.neighbor_pairs import correlation
from src.data.io import resolve


GROUPS = {'Probability': ['probability_left', 'probability_right'],
          'Choice': ['left_minus_right_choices'],
          'Outcome': ['rewarded_choices'],
          'Licking': ['log_left_lick_rate', 'log_right_lick_rate']}
CONTROLS = ['choices', 'cues', 'log_speed', 'available_D1', 'available_D2']


def features(source):
    trials = pd.read_csv(source / 'trials.csv')
    behavior = pd.read_csv(source / 'behavior_windows.csv')
    starts, stops = behavior.start_session_s.to_numpy(), behavior.stop_session_s.to_numpy()
    edges = np.r_[starts, stops[-1]]
    result = behavior[['start_session_s', 'stop_session_s', 'available_D1', 'available_D2']].copy()
    # Exact time-weighted probabilities, with changes anchored to trial starts.
    times = trials.start_time.to_numpy()
    breaks = np.r_[times, max(stops[-1], times[-1]) + 10]
    for side in ['left', 'right']:
        p = trials[f'reward_probability{side[0].upper()}'].to_numpy()
        integral = np.r_[0., np.cumsum(np.diff(breaks) * p)]
        result[f'probability_{side}'] = np.diff(np.interp(edges, breaks, integral)) / (stops - starts)
    # Do not assign probability to windows before task timing is known.
    result['task_known'] = starts >= times[0]
    left = trials.loc[trials.animal_response.eq(0), 'reward_outcome_time']
    right = trials.loc[trials.animal_response.eq(1), 'reward_outcome_time']
    nl, nr = np.histogram(left, edges)[0], np.histogram(right, edges)[0]
    result['choices'] = nl + nr
    result['left_minus_right_choices'] = nl - nr
    result['cues'] = np.histogram(trials.goCue_start_time, edges)[0]
    rewarded = trials.animal_response.isin([0, 1]) & trials.rewarded.eq(True)
    result['rewarded_choices'] = np.histogram(trials.loc[rewarded, 'reward_outcome_time'], edges)[0]
    for side in ['left', 'right']:
        result[f'log_{side}_lick_rate'] = np.log1p(behavior[f'{side}_licks_per_s'])
    result['log_speed'] = np.log1p(behavior.speed_cm_s)
    assert np.all(result.rewarded_choices <= result.choices)
    return result


def design(frame, trend_minutes):
    time = ((frame.start_session_s + frame.stop_session_s) / 2).to_numpy()
    knots = np.r_[time[0], np.arange(trend_minutes * 60, time[-1], trend_minutes * 60), time[-1]]
    trend = SplineTransformer(degree=3, knots=knots[:, None], include_bias=False).fit_transform(time[:, None])
    names = ['intercept'] + [f'time_{i}' for i in range(trend.shape[1])]
    variables = CONTROLS + [v for group in GROUPS.values() for v in group]
    z = frame[variables].to_numpy(float)
    center, scale = np.nanmean(z, axis=0), np.nanstd(z, axis=0)
    if np.any(scale <= 0):
        raise ValueError('Constant model covariate')
    z = (z - center) / scale
    matrix = np.column_stack([np.ones(len(time)), trend, z])
    names += variables
    group_indices = {key: [names.index(v) for v in group] for key, group in GROUPS.items()}
    group_indices['All task'] = [i for indices in group_indices.values() for i in indices]
    return matrix, names, group_indices, dict(zip(variables, scale.tolist()))


def fit_ols(x, y, valid, lags):
    xv, yv = x[valid], y[valid]
    beta, _, rank, _ = np.linalg.lstsq(xv, yv, rcond=None)
    n, p = xv.shape
    if rank != p or n <= p + 20:
        raise ValueError(f'Insufficient rank or observations: {rank}/{p}, n={n}')
    residual = yv - xv @ beta
    # Missing windows retain their original positions in the score time series.
    score = np.zeros_like(x)
    score[valid] = xv * residual[:, None]
    meat = score.T @ score
    for lag in range(1, lags + 1):
        cross = score[lag:].T @ score[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    inverse = np.linalg.inv(xv.T @ xv)
    covariance = inverse @ meat @ inverse * n / (n - p)
    covariance = (covariance + covariance.T) / 2
    return beta, covariance, residual


def run(cfg):
    source, output = resolve(cfg.source), resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    frame = features(source)
    traces = pd.read_csv(source / 'score_traces.csv')
    origin = json.loads((source / 'metadata.json').read_text())
    targets = []
    for pair in origin['pairs']:
        selected = traces.loc[traces.pair.eq(' / '.join(pair['names']))].sort_values('start_session_s')
        assert np.array_equal(selected.start_session_s, frame.start_session_s)
        for side, name in zip(['a', 'b'], pair['names']):
            for mode, column in [('Raw', f'raw_{side}'), ('Adjusted', f'adjusted_{side}')]:
                targets.append(dict(neuron=name, mode=mode, y=selected[column].to_numpy()))
    assert len({(t['neuron'], t['mode']) for t in targets}) == len(targets)
    results, coefficients, residuals = [], [], {}
    for minutes in cfg.trend_minutes:
        x, names, groups, scales = design(frame, int(minutes))
        for lags in cfg.hac_lags:
            primary = minutes == cfg.primary_trend_minutes and lags == cfg.primary_hac_lags
            for target in targets:
                y = target['y']
                valid = frame.task_known.to_numpy() & np.isfinite(x).all(axis=1) & np.isfinite(y)
                beta, covariance, residual = fit_ols(x, y, valid, int(lags))
                full_sse = float(residual @ residual)
                tss = float(np.sum((y[valid] - y[valid].mean())**2))
                base_info = dict(neuron=target['neuron'], mode=target['mode'], trend_minutes=int(minutes),
                                 hac_lags=int(lags), n=int(valid.sum()), n_parameters=x.shape[1], primary=primary)
                for group, indices in groups.items():
                    reduced = np.delete(x[valid], indices, axis=1)
                    reduced_beta = np.linalg.lstsq(reduced, y[valid], rcond=None)[0]
                    reduced_residual = y[valid] - reduced @ reduced_beta
                    reduced_sse = float(reduced_residual @ reduced_residual)
                    c = covariance[np.ix_(indices, indices)]
                    rank = np.linalg.matrix_rank(c)
                    assert rank == len(indices)
                    statistic = max(0., float(beta[indices] @ np.linalg.solve(c, beta[indices])))
                    results.append(dict(**base_info, group=group,
                                        partial_r2=max(0., (reduced_sse - full_sse) / reduced_sse),
                                        delta_r2=max(0., (reduced_sse - full_sse) / tss),
                                        wald_statistic=statistic, p=float(chi2.sf(statistic, rank))))
                    if primary and group == 'All task':
                        baseline = np.full(len(y), np.nan)
                        full = baseline.copy()
                        baseline[valid], full[valid] = reduced_residual, residual
                        residuals[(target['neuron'], target['mode'])] = (baseline, full)
                if primary:
                    for group, indices in groups.items():
                        if group == 'All task':
                            continue
                        for i in indices:
                            se = float(np.sqrt(max(0., covariance[i, i])))
                            coefficients.append(dict(neuron=target['neuron'], mode=target['mode'], group=group,
                                                     covariate=names[i], score_change_per_sd=float(beta[i]),
                                                     ci_low=float(beta[i] - 1.96 * se), ci_high=float(beta[i] + 1.96 * se),
                                                     predictor_sd=scales[names[i]]))
    results = pd.DataFrame(results)
    results['q'] = np.nan
    for _, group in results.groupby(['trend_minutes', 'hac_lags']):
        results.loc[group.index, 'q'] = false_discovery_control(group.p.to_numpy(), method='bh')
    primary = results.loc[results.primary].copy()
    summary = results.groupby(['neuron', 'mode', 'group']).agg(settings_q_lt_05=('q', lambda s: int((s < .05).sum())),
                                                            min_partial_r2=('partial_r2', 'min'), max_partial_r2=('partial_r2', 'max'),
                                                            max_q=('q', 'max')).reset_index()
    primary = primary.merge(summary, on=['neuron', 'mode', 'group'])
    pair_results = []
    for pair in origin['pairs']:
        a, b = pair['names']
        for mode in ['Raw', 'Adjusted']:
            base_a, full_a = residuals[(a, mode)]
            base_b, full_b = residuals[(b, mode)]
            valid = np.isfinite(base_a) & np.isfinite(base_b)
            pair_results.append(dict(neuron_a=a, neuron_b=b, mode=mode, n=int(valid.sum()),
                                     after_controls_r=correlation(base_a[valid], base_b[valid]),
                                     after_controls_and_task_r=correlation(full_a[valid], full_b[valid])))
    output.mkdir(parents=True)
    frame.to_csv(output / 'design_features.csv', index=False)
    results.to_csv(output / 'all_settings.csv', index=False)
    primary.to_csv(output / 'primary_results.csv', index=False)
    pd.DataFrame(coefficients).to_csv(output / 'coefficients.csv', index=False)
    pd.DataFrame(pair_results).to_csv(output / 'pair_residual_correlations.csv', index=False)
    residual_table = pd.DataFrame({'start_session_s': frame.start_session_s})
    for (neuron, mode), (baseline, full) in residuals.items():
        residual_table[f'{neuron}_{mode}_controls'] = baseline
        residual_table[f'{neuron}_{mode}_full'] = full
    residual_table.to_csv(output / 'residuals.csv', index=False)
    metadata = dict(source=str(source), model='OLS, additive simultaneous covariates, contemporaneous 10-second windows',
                    slow_trend='Cubic B-splines with knots every 5 minutes; sensitivity at 2 and 10 minutes',
                    controls='Time splines, total choices, cue count, log1p mean absolute speed, available D1 and D2 counts',
                    task_variables=GROUPS, primary_hac_lags=int(cfg.primary_hac_lags),
                    uncertainty='Bartlett/Newey-West covariance with finite-sample n/(n-p) correction; asymptotic chi-square Wald group tests',
                    missing='Complete target/covariate cases per neuron; HAC lags retain original time-grid positions',
                    multiplicity='Benjamini-Hochberg across all 80 tests (8 neurons x 2 score versions x 5 groups), separately within each trend/HAC setting',
                    effect='Partial R-squared: fraction of residual variance explained by a group beyond all other predictors; descriptive in-sample quantity',
                    selection='Pairs selected for strong correlation in the same recording; q-values are exploratory, not selection-adjusted confirmatory evidence',
                    exclusions='First two windows lack a full window of known task probabilities; missing target scores excluded',
                    limitations='No causal interpretation, no behavioral learning model, no sub-window timing, no cross-validated predictive claims; shared neighbors and firing-rate confounding not removed',
                    primary_tests=len(primary), settings=len(cfg.trend_minutes)*len(cfg.hac_lags))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    OmegaConf.save(cfg, output / 'config.yaml')
    from src.experiments.neighbors.plots.task_associations import plot
    plot(primary, [t['neuron'] for t in targets if t['mode']=='Raw'], output)
    print(primary[['neuron','mode','group','partial_r2','q','settings_q_lt_05']].to_string(index=False), flush=True)
    print(pd.DataFrame(pair_results).to_string(index=False), flush=True)
    return output
