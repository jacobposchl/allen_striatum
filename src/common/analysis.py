"""Shared numerical summaries and result provenance for analysis families."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from scipy.stats import pearsonr


def write_json(path, data):
    def clean(value):
        if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return [clean(v) for v in value]
        if isinstance(value, np.ndarray): return clean(value.tolist())
        if isinstance(value, np.generic): return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value): return None
        return value
    path.write_text(json.dumps(clean(data),indent=2,allow_nan=False)+'\n')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def correlation(a,b):
    return float(pearsonr(a,b).statistic) if len(a)>2 and np.std(a)>1e-12 and np.std(b)>1e-12 else np.nan


def deviance(y, prediction):
    prediction = np.maximum(prediction, np.finfo(float).tiny)
    return float(2 * np.sum(xlogy(y, y) - xlogy(y, prediction) - y + prediction))
