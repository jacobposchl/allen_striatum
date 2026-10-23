"""Filter broad trial-averaged firing peaks by baseline-relative half-height width."""
import json
import zipfile

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d

from src.common.analysis import sha256, write_json
from src.data.io import resolve
from src.common.plotting import STYLE


def measure_width(rate, centers, peak_index, threshold):
    """Interpolate crossings bounding the contiguous above-threshold peak."""
    left = right = int(peak_index)
    while left > 0 and rate[left - 1] > threshold:
        left -= 1
    while right < len(rate) - 1 and rate[right + 1] > threshold:
        right += 1
    left_censored, right_censored = left == 0, right == len(rate) - 1
    left_s = centers[left] if left_censored else np.interp(
        threshold, rate[[left - 1, left]], centers[[left - 1, left]])
    right_s = centers[right] if right_censored else np.interp(
        threshold, rate[[right + 1, right]], centers[[right + 1, right]])
    assert left_s <= centers[peak_index] <= right_s
    return float(left_s), float(right_s), left_censored, right_censored


def run(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    source, out = resolve(cfg.source), resolve(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    source_method = json.loads((source / 'method.json').read_text())
    units = pd.read_csv(source / 'unit_peak_timing.csv')
    with np.load(source / 'population_psth.npz') as saved:
        assert np.array_equal(saved['global_unit_id'], units.global_unit_id.to_numpy(str))
        centers, edges = saved['bin_centers_s'], saved['bin_edges_s']
        rates, trials = saved['firing_rate_hz'], saved['trial_id']
    dt, maximum = float(cfg.bin_size_s), float(cfg.max_peak_width_s)
    assert np.allclose(np.diff(edges), dt)
    low, high = map(float, cfg.peak_window_s)
    window = np.flatnonzero((centers >= low) & (centers < high))
    sigma = float(cfg.smoothing_sigma_s)
    smooth = gaussian_filter1d(rates, sigma / dt, axis=1, mode='nearest')
    peak_indices = window[smooth[:, window].argmax(axis=1)]
    units['raw_peak_time_s'] = units.peak_time_s
    units['curve_peak_time_s'] = centers[peak_indices]
    units['curve_peak_rate_hz'] = smooth[np.arange(len(units)), peak_indices]
    units['curve_peak_increase_hz'] = units.curve_peak_rate_hz - units.baseline_rate_hz
    units['half_height_rate_hz'] = units.baseline_rate_hz + float(cfg.height_fraction) * units.curve_peak_increase_hz
    records = []
    for i, unit in units.iterrows():
        record = dict(half_height_left_s=np.nan, half_height_right_s=np.nan,
                      peak_width_s=np.nan, peak_width_lower_bound_s=np.nan,
                      left_censored=False, right_censored=False)
        if unit.flat_in_search_window:
            reason = 'no_identifiable_peak'
        elif unit.curve_peak_increase_hz <= 0:
            reason = 'peak_not_above_baseline'
        else:
            left, right, lc, rc = measure_width(smooth[i], centers, peak_indices[i], unit.half_height_rate_hz)
            width = right - left
            record.update(half_height_left_s=left, half_height_right_s=right,
                          peak_width_s=width if not (lc or rc) else np.nan,
                          peak_width_lower_bound_s=width, left_censored=lc, right_censored=rc)
            if width > maximum + 1e-12:
                reason = 'peak_width_above_300ms'
            elif lc or rc:
                reason = 'width_not_fully_observed'
            else:
                reason = 'included'
        if unit.region not in cfg.regions:
            reason = 'region_not_selected'
        record.update(selection_reason=reason, selection_included=reason == 'included')
        records.append(record)
    units = pd.concat([units, pd.DataFrame(records)], axis=1)
    units['source_peak_row'] = np.arange(len(units))
    chosen = units.loc[units.selection_included].copy().reset_index(drop=True)
    chosen['selected_row'] = np.arange(len(chosen))
    assert chosen.peak_width_s.le(maximum + 1e-12).all()
    assert chosen.curve_peak_increase_hz.gt(0).all()
    assert not chosen[['left_censored', 'right_censored']].any().any()
    units.to_csv(out / 'unit_selection.csv', index=False)
    chosen.to_csv(out / 'selected_units.csv', index=False)
    np.savez_compressed(out / 'selected_psth.npz', global_unit_id=chosen.global_unit_id.to_numpy(str),
        trial_id=trials, bin_edges_s=edges, bin_centers_s=centers,
        firing_rate_hz=rates[chosen.source_peak_row], smoothed_firing_rate_hz=smooth[chosen.source_peak_row])

    colors = {'CP': '#DC3C44', 'ACB': '#2563EB'}
    hist_edges = low + np.arange(round((high - low) / dt) + 1) * dt
    bins = pd.DataFrame({'bin_start_s': hist_edges[:-1], 'bin_stop_s': hist_edges[1:]})
    summaries, handles = [], []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
        for region in cfg.regions:
            group = chosen.loc[chosen.region.eq(region)]
            original = units.loc[units.region.eq(region)]
            counts, _ = np.histogram(group.curve_peak_time_s, hist_edges)
            assert counts.sum() == len(group)
            percent = 100 * counts / len(group) if len(group) else np.zeros_like(counts, dtype=float)
            if len(group):
                assert np.isclose(percent.sum(), 100)
            bins[region + '_units'], bins[region + '_percent'] = counts, percent
            ax.stairs(percent, hist_edges, fill=True, color=colors[region], alpha=.15, linewidth=0)
            ax.stairs(percent, hist_edges, color=colors[region], lw=2.2)
            handles.append(Line2D([], [], color=colors[region], lw=2.2, label=f'{region} · {len(group)} units'))
            summaries.append(dict(region=str(region), input_units=len(original), selected_units=len(group),
                median_peak_time_s=group.curve_peak_time_s.median(), median_peak_width_s=group.peak_width_s.median(),
                selected_peaking_after_300ms=int(group.curve_peak_time_s.gt(.3).sum()),
                selected_peaking_after_display=int(group.curve_peak_time_s.ge(float(cfg.display_window_s[1])).sum()),
                excluded=original.loc[~original.selection_included, 'selection_reason'].value_counts().to_dict()))
        ax.set(xlim=list(cfg.display_window_s), ylim=(0, None), xlabel='Peak time after go cue (s)',
               ylabel='Selected units per 20 ms bin (%)', xticks=np.arange(0, .61, .1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(handles=handles, loc='upper right')
        fig.suptitle('Peak width ≤ 300 ms', fontsize=14)
        ax.set_title('Width measured at half-height above baseline', fontsize=10, color='#64748B', pad=12)
        for extension in ['png', 'pdf']:
            fig.savefig(out / f'short_peak_percentages.{extension}', dpi=240, bbox_inches='tight')
        plt.close(fig)

        # Show actual selected and excluded response curves to make the width criterion inspectable.
        fig, axes = plt.subplots(2, 2, figsize=(9, 5.7), sharex=True, layout='constrained')
        examples = []
        for row, region in enumerate(cfg.regions):
            for col, included in enumerate([True, False]):
                ax = axes[row, col]
                candidates = units.loc[units.region.eq(region) & units.peak_width_s.notna()
                    & (units.selection_included if included else units.selection_reason.eq('peak_width_above_300ms'))]
                # Illustrative curves: median width among the stronger half of each group, not extremes.
                candidates = candidates.loc[candidates.curve_peak_increase_hz.ge(candidates.curve_peak_increase_hz.median())]
                if candidates.empty:
                    ax.set_visible(False)
                    continue
                index = (candidates.peak_width_s - candidates.peak_width_s.median()).abs().idxmin()
                unit = units.loc[index]
                baseline, amplitude = unit.baseline_rate_hz, unit.curve_peak_increase_hz
                ax.plot(centers, rates[index] - baseline, color=colors[region], alpha=.22, lw=.8)
                ax.plot(centers, smooth[index] - baseline, color=colors[region], lw=2)
                ax.axhline(0, color='#94A3B8', lw=.8, ls='--')
                ax.axvline(0, color='#94A3B8', lw=.8, ls=':')
                height = float(cfg.height_fraction) * amplitude
                ax.plot([unit.half_height_left_s, unit.half_height_right_s], [height, height],
                        color='#172033', lw=2, marker='|', ms=10)
                ax.set_title(f'{region} · unit {int(unit.id)} · {unit.peak_width_s * 1000:.0f} ms wide', fontsize=10)
                visible_rates = rates[index, (centers >= -.2) & (centers <= 1.6)] - baseline
                y_low, y_high = min(0, visible_rates.min()), max(0, visible_rates.max())
                padding = max(.1, .08 * (y_high - y_low))
                ax.set(xlim=(-.2, 1.6), ylim=(y_low - padding, y_high + padding))
                if row == 1:
                    ax.set_xlabel('Time from go cue (s)')
                if col == 0:
                    ax.set_ylabel('Increase above baseline (Hz)')
                examples.append(dict(global_unit_id=unit.global_unit_id, included=bool(included),
                                     peak_width_s=unit.peak_width_s, peak_time_s=unit.curve_peak_time_s))
        axes[0, 0].text(.5, 1.25, 'Kept: short peaks', transform=axes[0, 0].transAxes, ha='center', fontsize=12)
        axes[0, 1].text(.5, 1.25, 'Excluded: broad peaks', transform=axes[0, 1].transAxes, ha='center', fontsize=12)
        for extension in ['png', 'pdf']:
            fig.savefig(out / f'peak_width_examples.{extension}', dpi=240, bbox_inches='tight')
        plt.close(fig)

    bins.to_csv(out / 'peak_bin_percentages.csv', index=False)
    summary = dict(animal=str(cfg.animal), session=str(cfg.session), n_trials=len(trials),
        bin_size_s=dt, smoothing_sigma_s=sigma, max_peak_width_s=maximum,
        input_units=len(units), selected_units=len(chosen), regions=summaries, examples=examples)
    write_json(out / 'summary.json', summary)
    method = dict(
        recording_id=source_method['recording_id'], recording_source=source_method['recording_source'],
        correction='Supersedes the incorrect early_peaks_300ms latency selection for this user request. No <=300-ms constraint is applied to peak time.',
        curve='Use the cached 20-ms trial-averaged PSTHs over -2 to +3 s, on the same 526 trials. Smooth with a Gaussian of sigma 20 ms (one bin) to reduce bin-to-bin fragmentation. Baseline is the original unsmoothed -1 to -0.2 s mean.',
        peak='Find the maximum of the smoothed curve within 0–1 s. Both peak time and width refer to that curve. Original unsmoothed peak times remain in raw_peak_time_s. Equal maxima choose the earliest bin.',
        width='The width of the contiguous portion surrounding the peak above baseline + 0.5*(peak - baseline). Interpolate threshold crossings between 20-ms bin centers. Search both sides across the full cached -2 to +3 s, without clipping at cue onset, 300 ms, 600 ms, or 1 s. Units whose widths are not fully observed cannot be selected.',
        selection='Keep positive, non-flat peaks with fully measured half-height width <=300 ms. Exclude widths >300 ms; this concerns peak breadth, not latency. No significance or extra unit-quality threshold. SUA and MUA retained.',
        limitation='This measures the broad peak above half-height, not the complete duration of every firing increase. Long lower-amplitude tails or separate later peaks can remain. Weak/noisy responses can have narrow apparent peaks; no reliability claim is made.',
        plot='Each selected unit contributes one smoothed-curve peak time. Heights are exact percent of selected units per 20-ms bin within each region. Full counts over 0–1 s sum to 100%; only 0–0.6 s is displayed. Example curves show raw rates faintly, smoothed rates in color, and measured width as a dark horizontal segment.',
        source_hashes={str(p): sha256(p) for p in [resolve(__file__), source / 'population_psth.npz', source / 'unit_peak_timing.csv']})
    write_json(out / 'method.json', method)
    OmegaConf.save(cfg, out / 'config.yaml')
    (out / 'index.html').write_text('\n'.join([
        '<!doctype html><meta charset="utf-8"><title>Short response peaks</title>',
        '<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;color:#243247;padding:0 24px}img{max-width:100%}</style>',
        '<h1>Peak width ≤ 300 ms</h1>',
        '<p>Filter by the duration of the peak at half-height above baseline, not by when the peak occurs. PSTHs use 20-ms bins and Gaussian smoothing with sigma 20 ms. Peaks are identified over 0–1 s; their widths can extend outside that interval.</p>',
        '<img src="peak_width_examples.png"><img src="short_peak_percentages.png">',
        '<p>The distribution displays 0–0.6 s; percentages use all selected units, including later peaks. This descriptive filter does not remove lower-amplitude sustained tails or establish response reliability.</p>',
        '<p><a href="selected_units.csv">Selected units</a> · <a href="unit_selection.csv">All widths and decisions</a> · <a href="method.json">Methods</a> · <a href="figures_and_tables.zip">Download</a></p>']) + '\n')
    with zipfile.ZipFile(out / 'figures_and_tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.suffix in ['.png', '.pdf', '.csv', '.npz', '.json', '.yaml', '.html']:
                archive.write(path, path.name)
    print(json.dumps(summary, indent=2), flush=True)
    print('Completed:', out, flush=True)
