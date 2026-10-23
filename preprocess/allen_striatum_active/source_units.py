import numpy as np
import pandas as pd

from src.data.features import waveform_columns


def device_name(nwb, group):
    links = nwb[f"general/extracellular_ephys/{group}"].attrs.get("zarr_link", [])
    devices = [link["path"].rsplit("/", 1)[-1] for link in links if link["name"] == "device"]
    if len(devices) != 1:
        raise ValueError(f"Cannot resolve physical probe for {group}")
    return devices[0]


def extract_recording(nwb, row, tip_sites=150, require_qc=False, waveform_batch_size=4,
                      exclude_noise=False):
    FILTER_STRIATUM = tip_sites is not None
    N_TIP_SITES = tip_sites
    REQUIRE_DEFAULT_QC = require_qc
    WAVEFORM_BATCH_SIZE = waveform_batch_size
    LABEL_COLUMN, UNLABELED_LABEL = "predicted_cell_type", "Unlabeled"
    units = nwb.get("units")
    n_units = 0 if units is None else len(units["id"])
    audit = dict(animal=row.animal, session=row.session, recording=row.recording,
                 total_units=0 if units is None else n_units, retained_units=0,
                 labeled_units=0, unknown_site_units=0, status='ok', detail='')
    if units is None or n_units == 0:
        audit['status'] = 'no units'
        return pd.DataFrame(), audit, []
    has_labels = LABEL_COLUMN in units
    audit['has_label_column'] = has_labels
    frame = pd.DataFrame({'unit_id': units["id"][:], 'label': units[LABEL_COLUMN][:] if has_labels else [None] * n_units})
    frame['label'] = frame.label.map(lambda x: x.decode() if isinstance(x, bytes) else x).astype('string').str.strip()
    frame['label'] = frame.label.mask(frame.label.str.lower().isin(['', 'nan', 'none', 'unknown']), pd.NA)
    frame['label_source'] = 'recorded label'
    frame.loc[frame.label.isna(), 'label_source'] = 'missing value' if has_labels else 'missing label column'
    frame.loc[frame.label.str.lower().eq('untagged').fillna(False), 'label_source'] = 'recorded untagged'
    frame['original_label'] = frame.label.copy()
    frame['label'] = frame.label.mask(frame.label.isna() | frame.label.str.lower().isin(['untagged', 'unlabeled', 'unlabelled']), UNLABELED_LABEL)
    frame['animal'], frame['session'], frame['recording'] = row.animal, row.session, row.recording
    frame['source_unit_row'] = np.arange(n_units)
    frame['tip_site_rank'] = np.nan
    if 'device_name' in units:
        frame['device_name'] = units['device_name'][:]
    if frame.unit_id.duplicated().any():
        raise ValueError('NWB unit IDs must be unique within a recording')
    frame['global_unit_id'] = [f'{row.animal}/{row.session}/{row.recording}/{i}' for i in frame.unit_id]
    for key in ('default_qc', 'decoder_label', 'depth', 'ks_unit_id', 'unit_name',
                'peak_to_valley', 'peak_trough_ratio', 'new_spread_interpolated',
                'spread_interpolated', 'waveform_duration_ms', 'waveform_spread_um',
                'isi_violations_ratio', 'prop_long_isi', 'brain_region', 'probe_id'):
        if key in units and units[key].ndim == 1:
            frame[key] = units[key][:]
    geometry = []
    if FILTER_STRIATUM:
        required = {'device_name'}
        if not required.issubset(units) or "general/extracellular_ephys/electrodes" not in nwb:
            audit.update(status='missing site mapping', detail='Need device_name and electrodes')
            return pd.DataFrame(), audit, []
        electrodes = nwb["general/extracellular_ephys/electrodes"]
        if not {'channel_name', 'group_name', 'rel_y'}.issubset(electrodes):
            audit.update(status='missing site mapping', detail='Need channel_name, group_name, rel_y')
            return pd.DataFrame(), audit, []
        sites = pd.DataFrame({key: electrodes[key][:] for key in ['channel_name', 'group_name', 'rel_y']})
        sites['channel'] = pd.to_numeric(sites.channel_name.astype(str).str.extract(r'^(?:AP|CH)(\d+)$', expand=False), errors='coerce')
        sites = sites[sites.channel.notna()].copy()
        sites['rel_y'] = pd.to_numeric(sites.rel_y, errors='coerce')
        if sites.empty or not np.isfinite(sites.rel_y).all() or sites.duplicated(['group_name', 'channel']).any():
            raise ValueError('AP site geometry missing, nonfinite, or ambiguous')
        sites['device_name'] = sites.group_name.map(lambda g: device_name(nwb, g))
        # Preserve electrode-table AP order for the waveform's channel axis.
        waveform_sites = sites.copy()
        sites = sites.sort_values(['device_name', 'rel_y', 'channel'])
        sites['tip_site_rank'] = sites.groupby('device_name').cumcount() + 1
        for probe, group in sites.groupby('device_name'):
            chosen = group[group.tip_site_rank <= N_TIP_SITES]
            geometry.append(dict(animal=row.animal, session=row.session, recording=row.recording,
                                 probe=probe, available_AP_sites=len(group), selected_sites=len(chosen),
                                 min_rel_y=chosen.rel_y.min(), max_rel_y=chosen.rel_y.max()))
        frame['device_name'] = units['device_name'][:]
        frame['peak_channel'] = pd.to_numeric(units['peak_channel'][:], errors='coerce') if 'peak_channel' in units else np.nan
        frame['site_source'] = 'stored peak_channel'
        # Some newer files store a zero-based index in the probe's AP channel order.
        if 'extremum_channel_index' in units:
            extremum = pd.to_numeric(units['extremum_channel_index'][:], errors='coerce')
            for row_index in range(n_units):
                if pd.notna(frame.loc[row_index, 'peak_channel']):
                    continue
                channel_index = extremum[row_index]
                probe_sites = waveform_sites[waveform_sites.device_name.eq(frame.loc[row_index, 'device_name'])]
                if np.isfinite(channel_index) and channel_index == int(channel_index) and 0 <= channel_index < len(probe_sites):
                    frame.loc[row_index, 'peak_channel'] = probe_sites.channel.iloc[int(channel_index)]
                    frame.loc[row_index, 'site_source'] = 'stored extremum_channel_index'
        missing_peak = frame.peak_channel.isna()
        if missing_peak.any():
            if 'waveform_mean' not in units:
                frame.loc[missing_peak, 'site_source'] = 'unavailable'
            else:
                waveforms = units['waveform_mean']
                if len(waveforms.shape) != 3 or waveforms.shape[0] != n_units:
                    raise ValueError('Expected waveform dimensions: unit, time, AP channel')
                for start in range(0, n_units, WAVEFORM_BATCH_SIZE):
                    stop = min(start + WAVEFORM_BATCH_SIZE, n_units)
                    if not missing_peak.iloc[start:stop].any():
                        continue
                    block = np.asarray(waveforms[start:stop])
                    amplitude = np.max(np.where(np.isfinite(block), np.abs(block), -np.inf), axis=1)
                    for offset, row_index in enumerate(range(start, stop)):
                        if not missing_peak.iloc[row_index]:
                            continue
                        probe_sites = waveform_sites[waveform_sites.device_name.eq(frame.loc[row_index, 'device_name'])]
                        if 'electrodes' in units:
                            # Ragged electrode references preserve each unit's waveform channel order.
                            references = units['electrodes_index']
                            lo = int(references[row_index - 1]) if row_index else 0
                            hi = int(references[row_index])
                            electrode_rows = np.asarray(units['electrodes'][lo:hi], dtype=int)
                            ap_rows = [r for r in electrode_rows if r in waveform_sites.index]
                            probe_sites = waveform_sites.loc[ap_rows]
                        if len(probe_sites) != block.shape[2] or not probe_sites.device_name.eq(frame.loc[row_index, 'device_name']).all():
                            raise ValueError('Waveform channel axis cannot be mapped unambiguously to AP sites')
                        if not np.isfinite(amplitude[offset]).any() or amplitude[offset].max() <= 0:
                            frame.loc[row_index, 'site_source'] = 'unavailable'
                            continue
                        peak = int(np.argmax(amplitude[offset]))
                        frame.loc[row_index, 'peak_channel'] = probe_sites.channel.iloc[peak]
                        frame.loc[row_index, 'site_source'] = 'mean waveform peak'
        audit['site_source'] = ', '.join(sorted(frame.site_source.unique()))
        # Resolve device names through the NWB electrode groups, not global row indices.
        mapping = sites[['group_name', 'channel', 'tip_site_rank']].copy()
        mapping['device_name'] = mapping.group_name.map(lambda g: device_name(nwb, g))
        if mapping.duplicated(['device_name', 'channel']).any():
            raise ValueError('Multiple electrode groups share a device/channel; mapping is ambiguous')
        frame['device_name'] = units['device_name'][:]
        frame = frame.drop(columns='tip_site_rank').merge(
            mapping[['device_name', 'channel', 'tip_site_rank']],
            left_on=['device_name', 'peak_channel'], right_on=['device_name', 'channel'],
            how='left', validate='many_to_one', sort=False)
        audit['unknown_site_units'] = int(frame.tip_site_rank.isna().sum())
    frame['passes_qc'] = True
    if REQUIRE_DEFAULT_QC:
        if 'default_qc' not in frame:
            audit.update(status='missing QC', detail='default_qc unavailable')
            return pd.DataFrame(), audit, geometry
        qc = frame.default_qc.map(lambda x: x.decode() if isinstance(x, bytes) else x)
        qc = qc.astype('string').str.strip().str.lower()
        frame['passes_qc'] = qc.isin(['true', '1', '1.0'])
    frame['passes_noise_filter'] = True
    if exclude_noise:
        if 'decoder_label' not in frame:
            audit.update(status='missing sorting label', detail='decoder_label unavailable')
            return pd.DataFrame(), audit, geometry
        sorting = frame.decoder_label.map(lambda x: x.decode() if isinstance(x, bytes) else x)
        sorting = sorting.astype('string').str.strip().str.lower()
        frame['passes_noise_filter'] = sorting.ne('noise').fillna(False)
    # D1/D2 labels never exclude units. Priority: QC -> noise -> site mapping/cutoff.
    frame['is_labeled'] = frame.label.ne(UNLABELED_LABEL)
    frame['outcome'] = 'retained'
    if FILTER_STRIATUM:
        frame.loc[frame.tip_site_rank.gt(N_TIP_SITES), 'outcome'] = 'outside tip sites'
        frame.loc[frame.tip_site_rank.isna(), 'outcome'] = 'unknown site'
    frame.loc[~frame.passes_noise_filter, 'outcome'] = 'noise / missing sorting label'
    frame.loc[~frame.passes_qc, 'outcome'] = 'QC excluded'
    selected = frame[frame.outcome.eq('retained')]
    audit['retained_units'] = len(selected)
    audit['labeled_units'] = int(selected.is_labeled.sum())
    audit['missing_label_units'] = int(frame.original_label.isna().sum())
    if audit['unknown_site_units']:
        audit.update(status='partial site mapping', detail='Unmapped units excluded from site filter')
    return waveform_columns(frame), audit, geometry
