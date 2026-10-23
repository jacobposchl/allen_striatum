"""Shared, explicit animal/session/recording selection for every workflow stage."""


def normalized_cohort(cohort=None):
    cohort = cohort or {}
    result = {}
    for field in ("animals", "sessions", "recordings"):
        values = cohort.get(field)
        if values is not None:
            if isinstance(values, (str, bytes)):
                raise ValueError(f"cohort.{field} must be a list or null")
            values = sorted(set(map(str, values)))
            if not values:
                raise ValueError(f"cohort.{field} cannot be an empty list; use null for all")
            result[field] = values
    return result


def select_manifest(manifest, cohort=None, require_nonempty=True, require_all=True):
    """Select recording or neuron rows by stable identities.

    Session names are exact. Recording selectors may be a bare recording name,
    session/recording, or animal/session/recording. Excluded recording rows are
    retained in the selection so callers can report why a recording is absent.
    """
    import pandas as pd

    spec = normalized_cohort(cohort)
    keep = pd.Series(True, index=manifest.index)
    for field, column in (("animals", "animal"), ("sessions", "session")):
        if field not in spec:
            continue
        values = manifest[column].astype(str)
        missing = set(spec[field]) - set(values)
        if missing and require_all:
            raise ValueError(f"Unknown cohort.{field}: {sorted(missing)}")
        keep &= values.isin(spec[field])
    if "recordings" in spec:
        bare = manifest.recording.astype(str)
        relative = manifest.session.astype(str) + "/" + bare
        full = manifest.animal.astype(str) + "/" + relative
        missing = set(spec["recordings"]) - (set(bare) | set(relative) | set(full))
        if missing and require_all:
            raise ValueError(f"Unknown cohort.recordings: {sorted(missing)}")
        keep &= bare.isin(spec["recordings"]) | relative.isin(spec["recordings"]) | full.isin(spec["recordings"])
    selected = manifest.loc[keep].copy()
    if selected.empty and require_nonempty:
        raise ValueError("Cohort selection contains no recordings")
    return selected


def select_keys(dataset, cohort=None):
    selected = select_manifest(dataset.manifest, cohort)
    if "status" in selected:
        selected = selected[selected.status.eq("ready")]
    available = set(dataset.keys())
    keys = [(str(row.animal), f"{row.session}/{row.recording}")
            for row in selected.itertuples(index=False)]
    return [key for key in keys if key in available]


# Unit tables and export metadata carry the same identity columns as manifests.
filter_frame = select_manifest
