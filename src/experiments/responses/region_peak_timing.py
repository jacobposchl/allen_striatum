"""Descriptive CP/ACB peak timing from observed spikes, with trial-split support."""
import html
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import norm, ttest_1samp

from src.common.analysis import sha256, write_json
from src.data.io import resolve
from src.data.recording import Recording, resolve_recording
from src.data.task import recording_path, source_fingerprint
from src.common.plotting import STYLE


def fdr_bh(p):
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def density(peaks, grid, bandwidth, low, high):
    # Identical fixed bandwidth for both groups; reflect at both search boundaries.
    images = np.concatenate([peaks, 2*low-peaks, 2*high-peaks])
    values = norm.pdf((grid[:, None]-images[None, :])/bandwidth).sum(1) / (len(peaks)*bandwidth)
    values /= np.trapezoid(values, grid)
    assert np.isclose(np.trapezoid(values, grid), 1)
    return values


def plot_density(cfg, out, units, mask, name, subtitle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(cfg.get('region_colors', {'CP': '#7C3FA1', 'ACB': '#B75B3D'}))
    low, high = map(float, cfg.window_s)
    display_low, display_high = map(float, cfg.get('display_window_s', cfg.window_s))
    grid = np.linspace(low, high, 501)
    curves = pd.DataFrame({'time_s': grid})
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
        for region in cfg.regions:
            peaks = units.loc[mask & units.region.eq(region), 'peak_time_s'].to_numpy()
            if not len(peaks):
                continue
            y = density(peaks, grid, float(cfg.density_bandwidth_s), low, high)
            curves[region] = y
            ax.fill_between(grid, y, color=colors[region], alpha=.18)
            ax.plot(grid, y, color=colors[region], lw=2.3, label=f'{region} · {len(peaks)} units')
        ax.set(xlim=(display_low, display_high), ylim=(0, None), xlabel='Peak time after go cue (s)',
               ylabel=str(cfg.get('density_ylabel', 'Density (s⁻¹)')),
               xticks=list(cfg.get('display_ticks_s', np.linspace(display_low, display_high, 6))))
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right')
        fig.suptitle('CP and ACB response peak timing', fontsize=14)
        ax.set_title(subtitle, fontsize=10, color='#64748B', pad=12)
        for extension in ['png', 'pdf']:
            fig.savefig(out/f'{name}.{extension}', dpi=240, bbox_inches='tight')
        plt.close(fig)
    curves.to_csv(out/f'{name}.csv', index=False)


def plot_scatter(cfg, out, units):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    colors = {'CP': '#DC3C44', 'ACB': '#2563EB'}
    points = units.loc[units.peak_time_s.notna()].copy()
    rng = np.random.default_rng(int(cfg.scatter_seed))
    jitter = float(cfg.scatter_jitter_s)
    points['display_peak_time_s'] = points.peak_time_s + rng.uniform(-jitter, jitter, len(points))
    assert np.all(np.abs(points.display_peak_time_s-points.peak_time_s) <= jitter+1e-12)
    assert points.global_unit_id.is_unique
    points.to_csv(out/'peak_time_strength_points.csv', index=False)
    # Randomize drawing order so one region is not systematically painted over the other.
    points = points.iloc[rng.permutation(len(points))]
    low, high = map(float, cfg.window_s)
    y_min = min(-1.2, float(points.peak_increase_hz.min())*1.2)
    y_max = max(1.0, float(points.peak_increase_hz.max())*1.08)
    names = []
    with plt.rc_context(STYLE):
        for compressed in [False, True]:
            fig, ax = plt.subplots(figsize=(9, 5.8), layout='constrained')
            for supported in [False, True]:
                selected = points.loc[points.replicated_increase.eq(supported)]
                marker_colors = selected.region.map(colors).to_numpy()
                ax.scatter(selected.display_peak_time_s, selected.peak_increase_hz,
                    s=37 if supported else 27,
                    facecolors=marker_colors if supported else 'none',
                    edgecolors='white' if supported else marker_colors,
                    linewidths=.45 if supported else .8,
                    alpha=.80 if supported else .42, zorder=3 if supported else 2)
            ax.axhline(0, color='#94A3B8', lw=.9, ls='--', zorder=1)
            ax.set(xlim=(low, high), ylim=(y_min, y_max),
                xlabel='Peak time after go cue (s)', ylabel='Peak increase above baseline (Hz)',
                xticks=np.linspace(low, high, 6))
            if compressed:
                ax.set_yscale('symlog', linthresh=1, linscale=1)
                ax.set_yticks([-1, 0, 1, 3, 10, 30, 100])
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.set_ylim(y_min, max(y_max, 100))
                ax.set_ylabel('Peak increase above baseline (Hz; expanded near zero)')
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
            handles = [Line2D([], [], marker='o', ls='', markersize=7,
                color=colors[region], label=f'{region} · {int(points.region.eq(region).sum())} units')
                for region in cfg.regions]
            ax.legend(handles=handles, loc='upper right')
            fig.suptitle('Response timing and strength', fontsize=14)
            ax.set_title('Filled: trial-supported increase · Open: other units',
                         fontsize=10, color='#64748B', pad=12)
            note = f'50 ms bins · Horizontal jitter ±{jitter*1000:g} ms'
            if compressed:
                note += ' · Y-axis linear within ±1 Hz, logarithmic beyond'
            fig.supxlabel(note, fontsize=9, color='#64748B')
            name = 'peak_time_strength_scatter' + ('_expanded' if compressed else '')
            for extension in ['png', 'pdf']:
                fig.savefig(out/f'{name}.{extension}', dpi=240, bbox_inches='tight')
            plt.close(fig)
            names.append(name+'.png')
    return names


def run(cfg):
    if cfg.get('mode') not in {None, 'waveform_width_association', 'filter_peak_width', 'rebin_peaks'}:
        raise ValueError(f"Unsupported peak analysis mode: {cfg.mode}")
    if cfg.get('mode') == 'waveform_width_association':
        from src.experiments.responses.response_width_waveforms import run as run_waveforms
        return run_waveforms(cfg)
    if cfg.get('mode') == 'filter_peak_width':
        from src.experiments.responses.region_peak_width import run as run_width
        return run_width(cfg)
    if cfg.get('mode') == 'rebin_peaks':
        return run_rebinned_peaks(cfg)
    source, out = resolve(cfg.source), resolve(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    all_units = pd.read_csv(source/'units.csv')
    trials = pd.read_csv(source/'trials.csv')
    assert trials.goCue_start_time.is_monotonic_increasing
    region_mask = all_units.region.isin(list(cfg.regions)).to_numpy()
    units = all_units.loc[region_mask].copy().reset_index(drop=True)
    with np.load(source/'population_psth.npz') as saved:
        assert np.array_equal(all_units.global_unit_id.to_numpy(str), saved['global_unit_id'])
        assert np.array_equal(trials.id.to_numpy(), saved['trial_id'])
        centers, edges = saved['bin_centers_s'], saved['bin_edges_s']
        psth = saved['firing_rate_hz'][region_mask]
    dt = float(np.diff(edges)[0])
    assert np.allclose(np.diff(edges), dt)
    window = (centers >= float(cfg.window_s[0])) & (centers < float(cfg.window_s[1]))
    baseline = (centers >= float(cfg.baseline_s[0])) & (centers < float(cfg.baseline_s[1]))
    assert window.any() and baseline.any()
    post_centers = centers[window]
    counts = np.load(source/'trial_counts.npy', mmap_mode='r')[region_mask]
    assert counts.shape == (len(units), len(trials), len(centers))
    assert np.allclose(counts.mean(1)/dt, psth)

    post = psth[:, window]
    peak_index = post.argmax(1)
    maximum = post.max(1)
    units['flat_in_search_window'] = np.ptp(post, axis=1) < 1e-12
    units['peak_time_s'] = post_centers[peak_index]
    units.loc[units.flat_in_search_window, 'peak_time_s'] = np.nan
    units['peak_rate_hz'] = maximum
    units['peak_increase_hz'] = maximum - psth[:, baseline].mean(1)
    units['tied_peak_bins'] = np.isclose(post, maximum[:, None], rtol=1e-12, atol=1e-12).sum(1)
    units['peak_at_last_bin'] = (peak_index == len(post_centers)-1) & ~units.flat_in_search_window

    # Alternate chronological 10-trial blocks between folds. Select a peak on one
    # fold and assess its paired baseline difference on the other fold's blocks.
    block_ids = np.arange(len(trials)) // int(cfg.validation_block_trials)
    blocks = np.unique(block_ids)
    block_rates = np.stack([counts[:, block_ids == block].mean(1)/dt for block in blocks])
    weights = np.bincount(block_ids)
    block_baselines = block_rates[:, :, baseline].mean(2)
    p_values = []
    for fold in [0, 1]:
        train = blocks % 2 == fold
        test = ~train
        train_psth = np.average(block_rates[train], axis=0, weights=weights[train])[:, window]
        chosen = train_psth.argmax(1)
        heldout = block_rates[test][:, :, window][:, np.arange(len(units)), chosen] - block_baselines[test]
        p = ttest_1samp(heldout, 0, axis=0, alternative='greater').pvalue
        p = np.nan_to_num(p, nan=1.0)
        p_values.append(p)
        units[f'fold_{fold}_peak_time_s'] = post_centers[chosen]
        units[f'fold_{fold}_heldout_increase_hz'] = heldout.mean(0)
        units[f'fold_{fold}_heldout_p'] = p
    # Both independently selected peaks must predict an increase in the other fold.
    units['replicated_increase_p'] = np.maximum(*p_values)
    units['replicated_increase_q'] = fdr_bh(units.replicated_increase_p.to_numpy())
    units['replicated_increase'] = (units.replicated_increase_q.le(float(cfg.response_fdr))
        & units.peak_increase_hz.gt(0) & ~units.flat_in_search_window)
    units['split_peak_difference_s'] = np.abs(units.fold_0_peak_time_s-units.fold_1_peak_time_s)
    units['split_peaks_within_100ms'] = units.split_peak_difference_s.le(float(cfg.peak_agreement_s)+1e-10)
    units.to_csv(out/'unit_peak_timing.csv', index=False)
    trials.assign(validation_block=block_ids, validation_fold=block_ids % 2).to_csv(out/'trials.csv', index=False)
    scatter_figures = plot_scatter(cfg, out, units)

    views = {
        'all_units': (units.peak_time_s.notna(), 'All non-flat units'),
        'replicated_increases': (units.replicated_increase, 'Reproducible firing increases'),
        'replicated_increases_sua': (units.replicated_increase & units.decoder_label.eq('sua'),
                                    'Reproducible firing increases · single units'),
    }
    summaries = []
    for name, (mask, subtitle) in views.items():
        plot_density(cfg, out, units, mask, f'peak_time_density_{name}', subtitle)
        for region in cfg.regions:
            selected = units.loc[mask & units.region.eq(region)]
            summaries.append(dict(view=name, region=str(region), n_units=len(selected),
                n_sua=int(selected.decoder_label.eq('sua').sum()), n_mua=int(selected.decoder_label.eq('mua').sum()),
                median_peak_s=selected.peak_time_s.median(), q25_peak_s=selected.peak_time_s.quantile(.25),
                q75_peak_s=selected.peak_time_s.quantile(.75),
                fraction_peaking_before_200ms=selected.peak_time_s.lt(.2).mean(),
                fraction_split_peaks_within_100ms=selected.split_peaks_within_100ms.mean(),
                median_split_peak_difference_s=selected.split_peak_difference_s.median(),
                n_tied_peak_bins=int(selected.tied_peak_bins.gt(1).sum()),
                n_peaks_at_last_bin=int(selected.peak_at_last_bin.sum()),
                n_nonpositive_peak_increases=int(selected.peak_increase_hz.le(0).sum())))
    table = pd.DataFrame(summaries)
    table.to_csv(out/'region_summary.csv', index=False)
    summary = dict(animal=str(cfg.animal), session=str(cfg.session), n_trials=len(trials),
        bin_size_s=dt, input_units=len(units),
        flat_units_by_region=units.groupby('region').flat_in_search_window.sum().to_dict(),
        region_summaries=summaries)
    write_json(out/'summary.json', summary)
    method = dict(
        target='Time of the maximum observed mean firing rate in [0,1) seconds after the go cue, using the existing unsmoothed 50-ms PSTH. One value per unit; equal unit weights. This is peak time, not onset latency, response duration, or a cue-specific causal effect. Subtracting a constant pre-cue baseline does not change the peak time.',
        flat_units='Units constant across the search window have no identifiable peak and are excluded from timing densities. If multiple non-flat bins tie for the maximum, choose the earliest and record the tie count.',
        density='Both groups use the same fixed 50-ms Gaussian bandwidth, reflected at 0 and 1 s and normalized to integrate to one within that interval. Smoothing is for displaying the distribution only; individual peak estimates remain at 50-ms bin centers. Heights compare relative prevalence, not unit counts.',
        replication='Partition chronological trials into blocks of 10, alternate blocks between two folds. In each fold select the maximum 50-ms post-cue bin. In the other fold, compare rate at that fixed bin against the same block\'s -1 to -0.2 s baseline using a one-sided t test over trial-block means. Use the larger of the two p-values, then BH adjustment across all CP/ACB units, q <= 0.05. This is an exploratory screening criterion; it validates an increase at each selected peak, not identical peak latency or stationarity. Longer-range dependence can remain.',
        timing_stability='Separately report the absolute difference between the peak times independently selected in the two trial folds. This is not used to select the plotted responsive subset.',
        scatter='One dot per unit with an identifiable peak. X is its 0–1 s peak time and Y is its full-trial peak rate minus its -1 to -0.2 s mean baseline rate, in Hz. CP is red and ACB blue. Filled markers denote replicated increases; open markers retain the other units. Deterministic horizontal jitter of ±10 ms only separates overlapping 50-ms peak bins; no vertical jitter or new clustering is used. Exact and displayed coordinates are saved separately. The primary Y axis is linear; the expanded companion is linear within ±1 Hz and logarithmic beyond. The maximum-based amplitude is descriptive and can be upward-biased by peak selection.',
        scope='Same one recording and one animal; SUA and MUA are retained in the main views, with a single-unit sensitivity figure. Trials pool choices and outcomes. No between-region population p-value is used. Peaks at the last bin may reflect responses still rising beyond 1 s. Post-cue peaks can include choice, outcome, movement and licking.',
        source_hashes={str(p): sha256(p) for p in [Path(__file__), source/'units.csv', source/'population_psth.npz', source/'trials.csv', source/'trial_counts.npy']})
    write_json(out/'method.json', method)
    OmegaConf.save(cfg, out/'config.yaml')
    report = ['<!doctype html><meta charset="utf-8"><title>CP and ACB peak timing</title>',
        '<style>body{font:16px system-ui;line-height:1.5;max-width:1100px;margin:32px auto;color:#243247;padding:0 24px}img{max-width:100%}td,th{padding:6px}</style>',
        f'<h1>{cfg.animal} · {str(cfg.session)[:10]} · CP and ACB peak timing</h1>',
        '<p>Each unit contributes one peak time between 0 and 1 second after the go cue. Both distributions have unit area; larger regions do not receive extra weight. The all-unit view can include noisy peaks and recovery from suppression. The replicated-increase view checks that firing increases are supported by separate trial folds.</p>',
        table.to_html(index=False, float_format=lambda x: f'{x:.3f}')]
    for name in views:
        report.append(f'<p><a href="peak_time_density_{name}.png">{name}</a></p><img src="peak_time_density_{name}.png">')
    for name in scatter_figures:
        report.append(f'<p><a href="{name}">{name}</a></p><img src="{name}">')
    report.extend(f'<p><b>{html.escape(k)}</b>: {html.escape(v)}</p>' for k,v in method.items() if isinstance(v, str))
    report.append('<p><a href="unit_peak_timing.csv">Per-unit results</a> · <a href="region_summary.csv">Regional summaries</a> · <a href="method.json">Methods</a> · <a href="figures_and_tables.zip">Download</a></p>')
    (out/'index.html').write_text('\n'.join(report)+'\n')
    with zipfile.ZipFile(out/'figures_and_tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.suffix in ['.png', '.pdf', '.csv', '.json', '.yaml', '.html']:
                archive.write(p, p.name)
    print(table.to_string(index=False), flush=True)
    print('Flat units:', summary['flat_units_by_region'], flush=True)
    print('Completed:', out, flush=True)


def run_rebinned_peaks(cfg):
    """One peak-time observation per unit, rebinned from original spike timestamps."""
    source, out = resolve(cfg.source), resolve(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    source_cfg = OmegaConf.load(source/'config.yaml')
    source_method = json.loads((source / 'method.json').read_text())
    recording = Recording(resolve_recording(source_method['recording_id'],
        root=cfg.get('recording_root') or source_cfg.get('recording_root')))
    if source_method.get('recording_source') != source_fingerprint(recording.path):
        raise ValueError('Population PSTH does not match the selected Brainsets recording')
    info = dict(recording.info, recording_id=str(recording.info['id']), recording_h5=str(recording.path))
    source_units = pd.read_csv(source/'units.csv')
    selected = source_units.region.isin(list(cfg.regions)).to_numpy()
    units = source_units.loc[selected].copy().reset_index(drop=True)
    trials = pd.read_csv(source/'trials.csv')
    with np.load(source/'population_psth.npz') as saved:
        assert np.array_equal(saved['global_unit_id'][selected], units.global_unit_id.to_numpy(str))
        assert np.array_equal(saved['trial_id'], trials.id.to_numpy())
        source_edges = saved['bin_edges_s'].copy()
        source_rates = saved['firing_rate_hz'][selected].copy()
    dt = float(cfg.bin_size_s)
    n_bins = round((source_edges[-1]-source_edges[0])/dt)
    edges = source_edges[0] + np.arange(n_bins+1)*dt
    assert np.isclose(edges[-1], source_edges[-1])
    centers = .5*(edges[:-1]+edges[1:])
    absolute_edges = trials.goCue_start_time.to_numpy()[:, None] + float(info['clock']['origin_s']) + edges
    assert np.array_equal(recording.units.iloc[units.source_unit_row].id, units.id.to_numpy())
    psth = np.empty((len(units), n_bins))
    for i, source_row in enumerate(units.source_unit_row):
        unit_spikes = recording.spikes(int(source_row))
        assert np.isfinite(unit_spikes).all() and np.all(np.diff(unit_spikes) >= 0)
        counts = np.diff(np.searchsorted(unit_spikes, absolute_edges, side='left'), axis=1)
        psth[i] = counts.mean(0)/dt
        if i % 100 == 0:
            print(f'Binned {i+1}/{len(units)} units at {dt*1000:g} ms', flush=True)
    recording.close()
    # Independent conservation check: 20-ms and 50-ms counts must agree when
    # aggregated into their common 100-ms intervals across the original window.
    old_dt = float(np.diff(source_edges)[0])
    common_dt = .1
    new_group, old_group = round(common_dt/dt), round(common_dt/old_dt)
    assert np.isclose(new_group*dt, common_dt) and np.isclose(old_group*old_dt, common_dt)
    new_counts = psth.reshape(len(units), -1, new_group).sum(2)*dt
    old_counts = source_rates.reshape(len(units), -1, old_group).sum(2)*old_dt
    assert np.allclose(new_counts, old_counts, rtol=1e-9, atol=1e-9), 'Spike counts differ in common 100-ms bins'
    window = (centers >= float(cfg.window_s[0])) & (centers < float(cfg.window_s[1]))
    post = psth[:, window]
    flat = np.ptp(post, axis=1) < 1e-12
    peak_bin = post.argmax(1)
    units['flat_in_search_window'] = flat
    units['peak_time_s'] = np.where(flat, np.nan, centers[window][peak_bin])
    units['peak_rate_hz'] = post.max(1)
    units['tied_peak_bins'] = np.isclose(post, post.max(1)[:, None], rtol=1e-12, atol=1e-12).sum(1)
    units['peak_at_last_bin'] = (peak_bin == post.shape[1]-1) & ~flat
    baseline = (centers >= float(source_cfg.baseline_s[0])) & (centers < float(source_cfg.baseline_s[1]))
    units['baseline_rate_hz'] = psth[:, baseline].mean(1)
    units['peak_increase_hz'] = units.peak_rate_hz-units.baseline_rate_hz
    units.to_csv(out/'unit_peak_timing.csv', index=False)
    trials.to_csv(out/'trials.csv', index=False)
    np.savez_compressed(out/'population_psth.npz', global_unit_id=units.global_unit_id.to_numpy(str),
        trial_id=trials.id.to_numpy(), bin_edges_s=edges, bin_centers_s=centers, firing_rate_hz=psth)
    name = 'peak_time_density_20ms'
    plot_density(cfg, out, units, ~units.flat_in_search_window, name,
        f'{cfg.animal} · {str(cfg.session)[:10]} · {dt*1000:g} millisecond bins')
    summaries = []
    histogram = pd.DataFrame({'peak_bin_center_s':centers[window]})
    for region in cfg.regions:
        group = units.loc[units.region.eq(region)]
        usable = group.loc[~group.flat_in_search_window]
        summaries.append(dict(region=str(region), total_units=len(group), included_units=len(usable),
            flat_units=int(group.flat_in_search_window.sum()), median_peak_time_s=usable.peak_time_s.median(),
            q25_peak_time_s=usable.peak_time_s.quantile(.25), q75_peak_time_s=usable.peak_time_s.quantile(.75),
            fraction_peaking_before_200ms=usable.peak_time_s.lt(.2).mean(),
            tied_peak_units=int(usable.tied_peak_bins.gt(1).sum()),
            last_bin_peaks=int(usable.peak_at_last_bin.sum())))
        histogram[region+'_units'] = np.bincount(peak_bin[(units.region.eq(region) & ~flat).to_numpy()], minlength=post.shape[1])
    histogram.to_csv(out/'peak_bin_counts.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'region_summary.csv', index=False)
    summary = dict(animal=str(cfg.animal), session=str(cfg.session), n_trials=len(trials),
        bin_size_s=dt, peak_window_s=list(cfg.window_s), density_bandwidth_s=float(cfg.density_bandwidth_s),
        display_window_s=list(cfg.get('display_window_s', cfg.window_s)),
        regions=summaries, counts_match_original_in_100ms_bins=True)
    write_json(out/'summary.json', summary)
    method = dict(
        recording_id=info['recording_id'], recording_h5=info['recording_h5'], origin_s=float(info['clock']['origin_s']),
        observations='Reuse exactly the CP/ACB non-noise units and 526 trials in the original population overview. Recount raw spikes in cue-relative half-open 20-ms bins across -2 to +3 s, then average counts across trials and divide by 0.02 s. No interpolation of the previous 50-ms PSTHs and no smoothing of individual PSTHs.',
        peak='For each unit, select the time of its maximum trial-averaged firing rate within 0–1 s. Report the 20-ms bin center. This is one observation per unit, independent of peak height. Exclude only units whose PSTH is flat throughout that interval; equal non-flat maxima use the earliest bin and tie counts are saved. No significance or response-strength selection.',
        density='Overlay the two regional distributions, each normalized to unit area with equal weight per unit. Use the same fixed 20-ms Gaussian density bandwidth for both, reflected at 0 and 1 s. Smoothing applies only to the distribution of peak times. The individual peak estimates retain 20-ms bin resolution.',
        display=f'The displayed x-axis is {list(cfg.get("display_window_s", cfg.window_s))} seconds. Peak estimation and density normalization retain the full {list(cfg.window_s)} second search window. Cropping never changes peak times, density heights, or which units contribute.',
        validation='Rebinned mean counts exactly match the original 50-ms PSTHs when both are aggregated to common 100-ms intervals. Unit IDs and trial IDs are checked against the source cache.',
        scope='One recording; includes SUA and MUA. Peak time is not response onset or a cue-specific causal effect. The method assigns descriptive maxima even to weak/noisy or suppressed non-flat units.',
        source_hashes={str(p):sha256(p) for p in [Path(__file__), source/'units.csv', source/'trials.csv', source/'population_psth.npz']}, recording_source=source_fingerprint(recording.path))
    write_json(out/'method.json', method)
    OmegaConf.save(cfg, out/'config.yaml')
    report = ['<!doctype html><meta charset="utf-8"><title>CP and ACB peak timing · 20 ms</title>',
        '<style>body{font:16px system-ui;line-height:1.5;max-width:1000px;margin:32px auto;color:#243247;padding:0 24px}img{max-width:100%}td,th{padding:6px}</style>',
        '<h1>CP and ACB peak timing · 20 millisecond bins</h1>',
        '<p>Each unit contributes the time of its maximum trial-averaged firing rate between 0 and 1 second after the go cue. PSTHs were rebinned from original spike timestamps on the same 526 trials. Both density curves integrate to one. Only flat PSTHs lack a usable peak.</p>',
        f'<img src="{name}.png">', pd.DataFrame(summaries).to_html(index=False),
        f'<p><a href="{name}.png">PNG</a> · <a href="{name}.pdf">PDF</a> · <a href="unit_peak_timing.csv">Per-unit peaks</a> · <a href="peak_bin_counts.csv">Exact counts by peak bin</a> · <a href="method.json">Methods</a> · <a href="figures_and_tables.zip">Download</a></p>']
    (out/'index.html').write_text('\n'.join(report)+'\n')
    with zipfile.ZipFile(out/'figures_and_tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.suffix in ['.png','.pdf','.csv','.npz','.json','.yaml','.html']:
                archive.write(p,p.name)
    print(json.dumps(summary, indent=2), flush=True)
    print('Completed:', out, flush=True)
