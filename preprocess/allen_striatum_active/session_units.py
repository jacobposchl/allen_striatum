"""Optional derived annotations. No units or spikes are removed."""
import json
import numpy as np

from preprocess.allen_striatum_active.session_intervals import qc_intervals


def annotate(source, target, item):
    count = len(source["units/id"]) if "units/id" in source else 0
    windows = qc_intervals(source, item)
    duration = float(np.diff(windows, axis=1).sum())
    ratio = np.full(count, np.nan)
    spike_count = np.full(count, -1, dtype=np.int64)
    qc = np.full(count, -1, dtype=np.int8)
    group = target.create_group("quality")
    group.attrs.update(object="ArrayDict", _unicode_keys=np.array([], dtype="S"),
                       qc_pre_opto_definition="presence_ratio > 0.8 AND amplitude_cutoff < 0.1 AND pre_opto_isi_violations_ratio < 0.5",
                       qc_pre_opto_codes="-1 unavailable, 0 fail, 1 pass",
                       qc_intervals_json=json.dumps(windows.tolist()),
                       isi_threshold_s=0.0015, min_isi_s=0.0,
                       metric_interval="union of baseline and behavior epochs, clipped to available recording bounds and before earliest optotagging epoch/pulse; gaps excluded")
    if count and duration > 0:
        units = source["units"]
        ends = units["spike_times_index"][:]
        for row, (start, end) in enumerate(zip(np.r_[0, ends[:-1]], ends)):
            times = np.asarray(units["spike_times"][int(start):int(end)], float)
            if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                continue
            n, violations = 0, 0
            for low, high in windows:
                a, b = np.searchsorted(times, [low, high], side="left")
                n += b-a
                violations += int((np.diff(times[a:b]) < .0015).sum())
            spike_count[row] = n
            if n:
                ratio[row] = violations * duration / (.003 * n*n)
        if all(k in units for k in ("presence_ratio", "amplitude_cutoff")):
            presence, amplitude = np.asarray(units["presence_ratio"][:], float), np.asarray(units["amplitude_cutoff"][:], float)
            valid = np.isfinite(ratio) & np.isfinite(presence) & np.isfinite(amplitude)
            qc[valid] = ((presence > .8) & (amplitude < .1) & (ratio < .5))[valid]
    group.create_dataset("pre_opto_isi_violations_ratio", data=ratio)
    group.create_dataset("pre_opto_spike_count", data=spike_count)
    group.create_dataset("qc_pre_opto", data=qc)
    # Bounds are first-class canonical intervals as well as audit metadata.
    from temporaldata import Interval
    Interval(start=windows[:, 0], end=windows[:, 1]).to_hdf5(
        target["intervals"].create_group("qc_pre_opto"))
    # Reuse the existing, checked AP site mapping without applying its outcome
    # filter. Preserve all source rows, including unmapped/noise/QC-failed units.
    ranks = np.full(count, np.nan)
    peak = np.full(count, np.nan)
    try:
        from types import SimpleNamespace
        from preprocess.allen_striatum_active.source_units import extract_recording
        frame, audit, geometry = extract_recording(source, SimpleNamespace(**item),
                                                  tip_sites=150, require_qc=False,
                                                  waveform_batch_size=4, exclude_noise=False)
        if len(frame):
            rows = frame.source_unit_row.to_numpy(int)
            ranks[rows] = frame.tip_site_rank
            if "peak_channel" in frame:
                peak[rows] = frame.peak_channel
        group.attrs["site_mapping_json"] = json.dumps(dict(audit=audit, geometry=geometry), default=str)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        group.attrs["site_mapping_error"] = str(error)
    group.create_dataset("tip_site_rank", data=ranks)
    group.create_dataset("resolved_peak_channel", data=peak)
    return dict(qc_pre_opto_pass=int((qc == 1).sum()), qc_pre_opto_fail=int((qc == 0).sum()),
                qc_pre_opto_unavailable=int((qc == -1).sum()))
