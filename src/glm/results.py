"""Read the versioned GLM cache without loading spikes or refitting models."""
import json

import numpy as np
import pandas as pd

from src.data.io import resolve


def select_units(cache, *, animal=None, session=None, recording=None, label=None,
                 sorting=None, require_qc=False, fitted_only=True):
    """Return an index subset; IDs retain animal/session/recording identity."""
    frame = pd.read_csv(resolve(cache)/'unit_index.csv', dtype={'animal': str, 'session': str})
    for column, value in [('animal',animal),('session',session),('recording',recording),
                          ('predicted_cell_type',label),('decoder_label',sorting)]:
        if value is not None: frame = frame.loc[frame[column].eq(str(value))]
    if require_qc: frame = frame.loc[frame.default_qc.eq(True)]
    if fitted_only: frame = frame.loc[frame.final_status.eq('fit')]
    return frame.copy()


def load_unit(cache, global_unit_id, *, validation=False, require_converged=True):
    """Final model is the default; CV coefficients require explicit opt-in."""
    root = resolve(cache)
    rows = select_units(root, fitted_only=False)
    rows = rows.loc[rows.global_unit_id.eq(global_unit_id)]
    if len(rows)!=1: raise KeyError(f'Expected one cached unit: {global_unit_id}')
    row = rows.iloc[0]
    if require_converged and row.final_status!='fit':
        raise ValueError(f'Final fit is {row.final_status}: {global_unit_id}')
    path = root/row.final_file
    directory = path.parent.parent
    with np.load(path) as z: final = {name:z[name].copy() for name in z.files}
    reliability = pd.read_csv(root/'reliability.csv')
    result = dict(unit=row.to_dict(), final=final,
                  fit_info=json.loads(path.with_suffix('.json').read_text()),
                  design_info=json.loads((directory/'prepared/design_info.json').read_text()),
                  reliability=reliability.loc[reliability.global_unit_id.eq(global_unit_id)].copy())
    with np.load(directory/'basis.npz') as z:
        result['basis'] = {name:z[name].copy() for name in z.files}
    if validation:
        with np.load(root/row.validation_file) as z:
            result['validation'] = {name:z[name].copy() for name in z.files}
    return result
