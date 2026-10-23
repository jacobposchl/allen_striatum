"""Pairwise-complete correlations of saved same-class neighbor scores."""
import json
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.io import resolve


def correlation(x, y):
    """Pearson r; undefined for fewer than three observations or constant input."""
    if len(x) < 3:
        return np.nan
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator < 1e-12:
        return np.nan
    return float(np.clip(np.dot(x, y) / denominator, -1, 1))


def pairwise(scores, labels, retained):
    """Reference averages include all available labeled units except the pair.

    Pair-specific class means are subtracted, not fitted by regression. Raw and
    adjusted correlations use exactly the same jointly valid time windows.
    """
    valid = np.isfinite(scores)
    values = np.where(valid, scores, 0)
    sums = {label: values[labels == label].sum(axis=0) for label in np.unique(labels)}
    counts = {label: valid[labels == label].sum(axis=0) for label in np.unique(labels)}
    n = len(retained)
    raw, adjusted = np.full((n, n), np.nan), np.full((n, n), np.nan)
    overlap = np.zeros((n, n), dtype=int)
    records = []
    for a, i in enumerate(retained):
        overlap[a, a] = valid[i].sum()
        for b in range(a + 1, n):
            j = retained[b]
            means = []
            for label in [labels[i], labels[j]]:
                total, count = sums[label].copy(), counts[label].copy()
                for excluded in [i, j]:
                    if labels[excluded] == label:
                        total -= values[excluded]
                        count -= valid[excluded]
                means.append(np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0))
            both = valid[i] & valid[j] & np.isfinite(means[0]) & np.isfinite(means[1])
            x, y = scores[i, both], scores[j, both]
            r = correlation(x, y)
            centered = correlation(x - means[0][both], y - means[1][both])
            raw[a, b] = raw[b, a] = r
            adjusted[a, b] = adjusted[b, a] = centered
            overlap[a, b] = overlap[b, a] = both.sum()
            records.append(dict(row_a=a, row_b=b, source_row_a=int(i), source_row_b=int(j),
                                class_pair=f'{labels[i]}–{labels[j]}', n_joint_windows=int(both.sum()),
                                raw_r=r, class_centered_r=centered))
    return raw, adjusted, overlap, pd.DataFrame(records)


def run(cfg):
    source, output = resolve(cfg.source), resolve(cfg.output)
    if output.exists():
        raise FileExistsError(output)
    threshold = float(cfg.min_presence_fraction)
    if not 0 <= threshold < 1:
        raise ValueError('Presence fraction must be in [0, 1)')
    data = np.load(source / 'neighbors.npz', allow_pickle=False)
    units = pd.read_csv(source / 'units.csv')
    assert np.array_equal(units.global_unit_id.to_numpy(str), data['global_unit_id'])
    scores = data['same_class_fraction']
    counts = data['present'].sum(axis=1)
    retained = np.flatnonzero(counts / scores.shape[1] > threshold)
    if len(retained) < 2:
        raise ValueError('Fewer than two eligible neurons')
    labels = units.label.to_numpy(str)
    raw, adjusted, overlap, pairs = pairwise(scores, labels, retained)
    selected = units.iloc[retained].reset_index(drop=True)
    for side in ['a', 'b']:
        rows = pairs[f'row_{side}'].to_numpy()
        pairs[f'global_unit_id_{side}'] = selected.global_unit_id.to_numpy()[rows]
        pairs[f'unit_id_{side}'] = selected.unit_id.to_numpy()[rows]
        pairs[f'label_{side}'] = selected.label.to_numpy()[rows]
    for matrix in [raw, adjusted, overlap]:
        assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert np.all(np.isnan(np.diag(raw))) and np.all(np.isnan(np.diag(adjusted)))
    assert np.all(overlap[np.triu_indices(len(selected), 1)] > 0)
    metadata = dict(source=str(source), n_windows=scores.shape[1],
                    duration_s=float(data['duration_s']),
                    coverage_s=[float(data['start_session_s'][0]), float(data['start_session_s'][-1] + data['duration_s'])],
                    presence_threshold=threshold, threshold_operator='>',
                    minimum_windows=int(np.floor(threshold * scores.shape[1]) + 1),
                    retained_units=len(selected), class_counts=selected.label.value_counts().to_dict(),
                    n_pairs=len(pairs), minimum_joint_windows=int(pairs.n_joint_windows.min()),
                    maximum_joint_windows=int(pairs.n_joint_windows.max()),
                    method='Pearson correlation on pairwise-complete windows; identical supports for raw and adjusted',
                    adjustment='Subtract the available same-class mean at each window; exclude BOTH compared neurons from each reference mean',
                    reference_pool='All original labeled neurons with valid scores in that window, including neurons below the display presence threshold',
                    neighbor_pool='Original k=10 neighbors unchanged; self excluded',
                    missing='Missing scores are excluded, never zero-filled; undefined correlations and diagonal are blank',
                    row_order='Original class then source-unit order, identical in both panels',
                    inference='Descriptive correlations; no significance tests or multiple-comparison correction',
                    class_pair_summary={name: dict(n_pairs=len(group), raw_median=float(group.raw_r.median()),
                                                  adjusted_median=float(group.class_centered_r.median()),
                                                  adjusted_min=float(group.class_centered_r.min()),
                                                  adjusted_max=float(group.class_centered_r.max()))
                                        for name, group in pairs.groupby('class_pair')})
    output.mkdir(parents=True)
    audit = units.copy()
    audit['available_windows'] = counts
    audit['available_fraction'] = counts / scores.shape[1]
    audit['retained'] = np.isin(np.arange(len(units)), retained)
    audit.to_csv(output / 'availability.csv', index=False)
    selected.to_csv(output / 'units.csv', index=False)
    pairs.to_csv(output / 'pairs.csv', index=False)
    np.savez_compressed(output / 'correlations.npz', raw=raw, class_centered=adjusted,
                        n_joint_windows=overlap, global_unit_id=selected.global_unit_id.to_numpy(str),
                        labels=selected.label.to_numpy(str), source_rows=retained)
    for name, matrix in [('raw', raw), ('class_centered', adjusted), ('n_joint_windows', overlap)]:
        pd.DataFrame(matrix, index=selected.global_unit_id, columns=selected.global_unit_id).to_csv(output / f'{name}.csv')
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    OmegaConf.save(cfg, output / 'config.yaml')
    from src.experiments.neighbors.plots.neighbor_pairs import plot
    plot(raw, adjusted, selected, metadata, output)
    print(json.dumps(metadata, indent=2), flush=True)
    cross = pairs.loc[pairs.label_a != pairs.label_b]
    print('Largest cross-class adjusted associations:', flush=True)
    print(cross.loc[cross.class_centered_r.abs().nlargest(5).index,
                    ['unit_id_a', 'label_a', 'unit_id_b', 'label_b', 'raw_r', 'class_centered_r', 'n_joint_windows']].to_string(index=False), flush=True)
    return output
