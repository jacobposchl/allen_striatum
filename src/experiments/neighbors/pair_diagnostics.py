"""Descriptive pair diagnostics using saved embeddings and source unit data."""
import json
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.preprocessing import SplineTransformer

from src.experiments.neighbors.neighbor_pairs import correlation
from src.experiments.neighbors.task_associations import design, features
from src.data.features import spike_features
from src.data.io import resolve
from src.data.recording import Recording, resolve_recording
from src.embeddings.windows import RecordingExport


def corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return correlation(np.asarray(x)[mask], np.asarray(y)[mask])


def residual(y, x, valid):
    result = np.full_like(y, np.nan)
    beta = np.linalg.lstsq(x[valid], y[valid], rcond=None)[0]
    result[valid] = y[valid] - x[valid] @ beta
    return result


def nearest_time_distances(a, b):
    indices = np.searchsorted(b, a)
    left = np.abs(a - b[np.maximum(indices - 1, 0)])
    right = np.abs(a - b[np.minimum(indices, len(b) - 1)])
    return np.minimum(left, right)


def run(cfg):
    source, output, export = resolve(cfg.source), resolve(cfg.output), resolve(cfg.export)
    if output.exists():
        raise FileExistsError(output)
    data = np.load(source / 'neighbors.npz', allow_pickle=False)
    units = pd.read_csv(source / 'units.csv')
    assert np.array_equal(units.global_unit_id.to_numpy(str), data['global_unit_id'])
    indices = [int(np.flatnonzero(units.unit_id.eq(int(unit)))[0]) for unit in cfg.units]
    i, j = indices
    labels = units.label.to_numpy(str)
    if labels[i] != labels[j]:
        raise ValueError('This shared same-class contribution diagnostic requires a same-class pair')
    starts, duration = data['start_session_s'], float(data['duration_s'])
    if not np.allclose(np.diff(starts), duration, rtol=0, atol=1e-6):
        raise ValueError('Pair spike diagnostics require a contiguous window selection')
    exported = RecordingExport(cfg.export)
    saved = json.loads((source / 'metadata.json').read_text())
    if saved.get('export_manifest_sha256') != exported.manifest_sha256:
        raise ValueError('Neighbor scores do not match the selected HDF5 export')
    edges = np.r_[starts, starts[-1] + duration]
    score = data['same_class_fraction'][indices]
    neighbors = data['neighbor_row_indices'][indices]
    assert np.isfinite(score).all(), 'This diagnostic expects complete pair coverage'
    shared = np.array([np.intersect1d(a, b) for a, b in zip(*neighbors)], dtype=object)
    shared_n = np.array([len(s) for s in shared])
    shared_same = np.array([np.sum(labels[np.asarray(s, dtype=int)] == labels[i]) for s in shared])
    unique_score = np.full_like(score, np.nan)
    for t, common in enumerate(shared):
        for side in range(2):
            unique = np.setdiff1d(neighbors[side, t], np.asarray(common, dtype=int))
            if len(unique) >= 3:
                unique_score[side, t] = np.mean(labels[unique] == labels[i])

    # Benchmark overlap against all other >75%-available D2 pairs.
    retained = np.flatnonzero((labels == labels[i]) & (data['present'].mean(axis=1) > .75))
    benchmarks = []
    for a, first in enumerate(retained):
        for second in retained[a + 1:]:
            valid = np.isfinite(data['same_class_fraction'][first]) & np.isfinite(data['same_class_fraction'][second])
            na, nb = data['neighbor_row_indices'][first, valid], data['neighbor_row_indices'][second, valid]
            overlap = (na[:, :, None] == nb[:, None, :]).any(axis=2).sum(axis=1)
            benchmarks.append(dict(unit_a=int(units.iloc[first].unit_id), unit_b=int(units.iloc[second].unit_id),
                                   mean_shared=float(overlap.mean()), n_windows=int(valid.sum()),
                                   score_r=corr(data['same_class_fraction'][first], data['same_class_fraction'][second])))
    benchmark = pd.DataFrame(benchmarks)

    # Fixed, class-balanced reference partitions; both query units always excluded.
    rng = np.random.default_rng(int(cfg.seed))
    partitions = np.full((int(cfg.reference_splits), len(units)), -1, dtype=int)
    for seed in range(len(partitions)):
        for label in np.unique(labels):
            eligible = np.flatnonzero((labels == label) & ~np.isin(np.arange(len(units)), indices))
            shuffled = rng.permutation(eligible)
            partitions[seed, shuffled] = np.arange(len(shuffled)) % 2
    split_scores = np.full((len(partitions), 2, 2, len(starts)), np.nan)
    no_partner = np.full_like(score, np.nan)
    cosines = np.full(len(starts), np.nan)
    origin = None
    for t, view in enumerate(data['view_index']):
        batch = exported.window(int(view))
        if batch is None:
            raise ValueError('Requested fully covered pair is absent from an exported window')
        offsets = batch['view_start_source_s'] - batch['view_start_session_s']
        if origin is None:
            origin = float(offsets[0])
        assert np.allclose(offsets, origin, rtol=0, atol=1e-6)
        assert np.allclose(batch['view_start_session_s'], starts[t], rtol=0, atol=1e-6)
        ids = batch['global_unit_id'].astype(str)
        positions = {key: a for a, key in enumerate(ids)}
        reference = np.array([a for a, key in enumerate(units.global_unit_id) if key in positions])
        vectors = np.asarray(batch['pooled'][[positions[units.global_unit_id.iloc[a]] for a in reference]], float)
        norms = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(vectors).all(axis=1) & (norms > 0)
        reference, vectors = reference[valid], vectors[valid] / norms[valid, None]
        qa, qb = [vectors[np.flatnonzero(reference == row)[0]] for row in indices]
        cosines[t] = qa @ qb
        for side, query in enumerate([qa, qb]):
            eligible = ~np.isin(reference, indices)
            refs = reference[eligible]
            distances = 1 - vectors[eligible] @ query
            order = np.lexsort((units.global_unit_id.to_numpy()[refs], distances))
            ranked = refs[order]
            no_partner[side, t] = np.mean(labels[ranked[:10]] == labels[indices[side]])
            for seed, partition in enumerate(partitions):
                for half in [0, 1]:
                    nearest = ranked[partition[ranked] == half][:10]
                    if len(nearest) == 10:
                        split_scores[seed, side, half, t] = np.mean(labels[nearest] == labels[indices[side]])

    recording = Recording(resolve_recording(cfg.recording, root=cfg.get('recording_root')))
    if recording.identity != exported.source_identity:
        recording.close()
        raise ValueError("Recording changed since export; regenerate embeddings")
    u = recording.units
    unit_rows = [int(units.iloc[row].source_unit_row) for row in indices]
    assert [int(u.iloc[row]['id']) for row in unit_rows] == list(cfg.units)
    scalar_names = ['id', 'ks_unit_id', 'unit_name', 'predicted_cell_type', 'decoder_label', 'default_qc',
                    'depth', 'ccf_acronym', 'ccf_ap', 'ccf_dv', 'ccf_ml', 'peak_channel', 'device_name',
                    'peak_to_valley', 'half_width', 'peak_trough_ratio', 'amplitude', 'snr',
                    'isi_violations_ratio', 'rp_contamination', 'presence_ratio', 'firing_rate']
    unit_features = u.iloc[unit_rows].reindex(columns=scalar_names).reset_index(drop=True)
    spikes, waves = [], []
    for row, peak in zip(unit_rows, unit_features.peak_channel):
        times = recording.spikes(row) - origin
        times = np.asarray(times[(times >= edges[0]) & (times < edges[-1])], float)
        assert np.all(np.diff(times) >= 0)
        spikes.append(times)
        waveform = recording.waveform(row)
        channels = recording.waveform_channels(row)
        channel = np.flatnonzero(channels.channel.to_numpy() == int(peak))
        if len(channel) != 1 or waveform.shape[-1] != len(channels):
            raise ValueError('Stored waveform channel mapping does not identify the peak channel')
        waves.append(np.asarray(waveform[:, int(channel[0])], float))
    recording.close()
    rates = np.array([np.histogram(times, edges)[0] / duration for times in spikes])
    assert all(int(np.sum(rate * duration)) == len(times) for rate, times in zip(rates, spikes))
    recomputed = pd.DataFrame([spike_features(times, np.array([[edges[0], edges[-1]]])) for times in spikes])
    for name in recomputed:
        unit_features[f'analysis_{name}'] = recomputed[name]
    unit_features['peak_to_valley_ms'] = unit_features.peak_to_valley * 1000
    unit_features['half_width_ms'] = unit_features.half_width * 1000
    normalized_waves = np.array([w / np.max(np.abs(w)) for w in waves])
    isi_edges = np.geomspace(.0001, 3, 70)
    isi_counts = np.array([np.histogram(np.diff(times), isi_edges)[0] for times in spikes])
    coincidence = {}
    for side in [0, 1]:
        distance = nearest_time_distances(spikes[side], spikes[1-side])
        coincidence[str(cfg.units[side])] = {str(ms): float(np.mean(distance <= ms / 1000)) for ms in [.1, .5, 1]}

    trace_source = source / 'pairwise_gt75/task_traces'
    frame = features(trace_source)
    x, names, groups, _ = design(frame, 5)
    base = np.delete(x, groups['All task'], axis=1)
    valid = frame.task_known.to_numpy() & np.isfinite(x).all(axis=1)
    rate_linear = np.log1p(rates).T
    rate_linear = (rate_linear - rate_linear.mean(axis=0)) / rate_linear.std(axis=0)
    rate_nonlinear = np.column_stack([SplineTransformer(n_knots=4, degree=3, knots='quantile', include_bias=False)
                                      .fit_transform(rate_linear[:, [side]]) for side in [0,1]])
    trace = pd.read_csv(trace_source / 'score_traces.csv')
    part = trace.loc[trace.pair.eq('D2 156 / D2 290')]
    adjusted = part[['adjusted_a', 'adjusted_b']].to_numpy().T
    controls = {'None': np.ones((len(starts), 1)),
                'Both rates (linear log-rate)': np.column_stack([np.ones(len(starts)), rate_linear]),
                'Both rates (nonlinear)': np.column_stack([np.ones(len(starts)), rate_nonlinear]),
                'Time + activity controls': base,
                'Time + controls + task': x,
                'Time + controls + task + both rates': np.column_stack([x, rate_nonlinear])}
    rows, saved_residuals = [], {}
    for mode, values in [('Raw', score), ('Class-centered', adjusted)]:
        for name, covariates in controls.items():
            res = np.array([residual(y, covariates, valid) for y in values])
            rows.append(dict(mode=mode, controls=name, r=corr(*res), n=int(valid.sum())))
            saved_residuals[(mode, name)] = res
    rate_control_results = pd.DataFrame(rows)
    split_records = []
    for seed in range(len(partitions)):
        for half in [0,1]:
            a = split_scores[seed, 0, half]
            for kind, other in [('Shared reference half', half), ('Disjoint reference halves', 1-half)]:
                b = split_scores[seed, 1, other]
                mask = valid & np.isfinite(a) & np.isfinite(b)
                residual_a = residual(a, controls['Time + controls + task + both rates'], mask)
                residual_b = residual(b, controls['Time + controls + task + both rates'], mask)
                split_records.append(dict(seed=seed, orientation=half, reference=kind,
                                          r=corr(a,b), controlled_r=corr(residual_a,residual_b), n=int(mask.sum())))
    split_table = pd.DataFrame(split_records)
    ccf = unit_features[['ccf_ap','ccf_dv','ccf_ml']].to_numpy(float)
    summary = dict(units=list(cfg.units), label='D2', n_windows=len(starts), coverage_s=[float(edges[0]),float(edges[-1])],
                   source=str(source), recording_h5=str(resolve_recording(cfg.recording, root=cfg.get('recording_root'))), source_minus_session_s=origin,
                   sorting=unit_features.decoder_label.tolist(), qc=unit_features.default_qc.tolist(),
                   brain_regions=unit_features.ccf_acronym.tolist(), ccf_distance_um=float(np.linalg.norm(ccf[0]-ccf[1])),
                   depth_difference=float(abs(unit_features.depth.iloc[0]-unit_features.depth.iloc[1])),
                   raw_score_r=corr(*score), raw_rate_r=corr(*rates),
                   rate_r_after_time_task_controls=corr(*[residual(rate, x, valid) for rate in rates]),
                   own_score_rate_r=[corr(score[s],rates[s]) for s in [0,1]],
                   shared_neighbors_mean=float(shared_n.mean()), shared_neighbors_median=float(np.median(shared_n)),
                   shared_neighbors_quantiles=np.quantile(shared_n,[.05,.95]).tolist(),
                   shared_neighbor_fraction=float(shared_n.mean()/10),
                   mean_jaccard=float(np.mean(shared_n/(20-shared_n))),
                   overlap_percentile_among_retained_D2_pairs=float(np.mean(benchmark.mean_shared <= shared_n.mean())*100),
                   benchmark_D2_pairs=len(benchmark), benchmark_mean_overlap=float(benchmark.mean_shared.median()),
                   shared_D2_count_variance=float(np.var(shared_same/10)),
                   mutual_neighbors=[float(np.mean((neighbors[0]==j).any(axis=1))),float(np.mean((neighbors[1]==i).any(axis=1)))],
                   score_r_excluding_partner=corr(*no_partner), unique_neighbors_minimum=3,
                   unique_only_score_r=corr(*unique_score), unique_only_n=int(np.isfinite(unique_score).all(axis=0).sum()),
                   embedding_cosine_mean=float(np.mean(cosines)), embedding_cosine_quantiles=np.quantile(cosines,[.05,.95]).tolist(),
                   peak_waveform_correlation=corr(*normalized_waves), waveform_x_axis='Stored sample index; sampling frequency not established',
                   coincidence_fraction=coincidence,
                   reference_split_summary={kind:dict(n=len(group), r_quantiles=np.quantile(group.r,[.05,.5,.95]).tolist(),
                                                      controlled_r_quantiles=np.quantile(group.controlled_r,[.05,.5,.95]).tolist())
                                            for kind,group in split_table.groupby('reference')},
                   interpretations='Reference splits are sensitivity distributions, not confidence intervals. Changing the candidate pool changes neighborhood scale. Rate controls are descriptive additive fits, not causal mediation. Static anatomy and waveform/ISI describe units, not time-varying causes.',
                   validation='Brainsets source unit IDs, every export clock, window spike-count totals verified; disjoint partitions fixed across time and exclude both queries')
    output.mkdir(parents=True)
    unit_features.to_csv(output / 'unit_features.csv', index=False)
    rate_control_results.to_csv(output / 'rate_controls.csv', index=False)
    benchmark.to_csv(output / 'D2_pair_benchmark.csv', index=False)
    split_table.to_csv(output / 'reference_splits.csv', index=False)
    window = pd.DataFrame(dict(start_session_s=starts, score_156=score[0], score_290=score[1],
                               rate_156=rates[0],rate_290=rates[1], shared_neighbors=shared_n,shared_D2=shared_same,
                               embedding_cosine=cosines,unique_score_156=unique_score[0],unique_score_290=unique_score[1]))
    window.to_csv(output / 'windows.csv', index=False)
    np.savez_compressed(output / 'arrays.npz', waveforms=np.array(waves), normalized_waveforms=normalized_waves,
                        isi_edges=isi_edges,isi_counts=isi_counts,split_scores=split_scores,partitions=partitions,
                        no_partner_scores=no_partner)
    (output / 'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    OmegaConf.save(cfg, output / 'config.yaml')
    from src.experiments.neighbors.plots.pair_diagnostics import plot
    plot(window,unit_features,rate_control_results,split_table,normalized_waves,isi_edges,isi_counts,summary,output)
    print(json.dumps(summary,indent=2),flush=True)
    print(rate_control_results.to_string(index=False),flush=True)
    print(unit_features.to_string(index=False),flush=True)
