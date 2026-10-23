"""Regional distributions of normalized PSTH values at each post-cue time."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import norm

from src.common.analysis import sha256, write_json
from src.data.io import resolve
from src.common.plotting import STYLE


def run(cfg):
    source, out = resolve(cfg.source), resolve(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(source/'units.csv')
    units = units.loc[units.region.isin(list(cfg.regions))].copy().reset_index(drop=True)
    with np.load(source/'population_psth.npz') as data:
        rows = units.plot_row.to_numpy(int)
        assert np.array_equal(units.global_unit_id.to_numpy(str), data['global_unit_id'][rows])
        window = (data['bin_centers_s'] >= float(cfg.window_s[0])) & (data['bin_centers_s'] < float(cfg.window_s[1]))
        native_time = data['bin_centers_s'][window]
        values = data['relative_change'][rows][:, window]
        means = data['normalization_rate_hz'][rows]
        rates = data['firing_rate_hz'][rows][:, window]
        baseline = data['baseline_rate_hz'][rows]
        valid = means > 0
        assert np.allclose(values[valid], (rates[valid]-baseline[valid, None])/means[valid, None])
        n_trials = len(data['trial_id'])
    assert np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all()
    units['density_included'] = valid
    units['density_exclusion'] = np.where(valid, '', 'zero_mean_firing_undefined_ratio')
    units.to_csv(out/'units.csv', index=False)
    step, bandwidth = float(cfg.response_grid_step), float(cfg.response_bandwidth)
    low = min(float(cfg.display_y_limits[0]), np.nanmin(values)-6*bandwidth)
    high = max(float(cfg.display_y_limits[1]), np.nanmax(values)+6*bandwidth)
    y = np.arange(np.floor(low/step)*step, np.ceil(high/step)*step+step/2, step)
    x = np.linspace(*map(float, cfg.window_s), int(cfg.time_grid_points))
    time_weights = norm.pdf((x[None, :]-native_time[:, None])/float(cfg.time_bandwidth_s))
    time_weights /= time_weights.sum(0)
    density, native_density, summaries, traces = [], [], [], []
    for region in cfg.regions:
        select = units.region.eq(region).to_numpy() & valid
        response = values[select]
        assert len(response)
        d = np.stack([norm.pdf((y[:, None]-response[None, :, j])/bandwidth).mean(1)/bandwidth
                      for j in range(len(native_time))], axis=1)
        smoothed = d @ time_weights
        assert np.allclose(np.trapezoid(smoothed, y, axis=0), 1, atol=1e-6)
        density.append(smoothed)
        native_density.append(d)
        for j, time in enumerate(native_time):
            traces.append(dict(region=str(region), time_s=float(time), n_units=len(response),
                median=float(np.median(response[:, j])), q25=float(np.quantile(response[:, j], .25)),
                q75=float(np.quantile(response[:, j], .75))))
        summaries.append(dict(region=str(region), included_units=len(response),
            zero_mean_units=int((units.region.eq(region).to_numpy() & ~valid).sum()),
            fraction_values_below_display=float((response < float(cfg.display_y_limits[0])).mean()),
            fraction_values_above_display=float((response > float(cfg.display_y_limits[1])).mean())))
    density = np.stack(density)
    scale = float(density.max())
    opacity = float(cfg.max_opacity)*(density/scale)**float(cfg.opacity_power)
    assert np.isfinite(opacity).all() and opacity.min() >= 0 and opacity.max() <= 1
    # Equal multiplicative transmittance for both colors: overlap does not depend
    # on which region is drawn last, and turns purple where both are prevalent.
    colors = np.array([[220, 60, 68], [37, 99, 235]], float)/255
    assert list(cfg.regions) == ['CP', 'ACB']
    image = np.ones((*density.shape[1:], 3), dtype=np.float32)
    for alpha, color in zip(opacity, colors):
        image *= 1-alpha[..., None]*(1-color)
    figures = plot_panels(cfg, out, x, y, density, summaries)
    np.savez_compressed(out/'response_density.npz', regions=np.asarray(list(cfg.regions)),
        global_unit_id=units.global_unit_id.to_numpy(str), density_included=valid,
        native_time_s=native_time, normalized_response=values,
        time_s=x, response_axis=y, conditional_density=density.astype(np.float32),
        native_conditional_density=np.stack(native_density).astype(np.float32),
        common_density_scale=scale, opacity=opacity.astype(np.float32))
    pd.DataFrame(traces).to_csv(out/'regional_response_quantiles.csv', index=False)
    summary = dict(animal=str(cfg.animal), session=str(cfg.session), n_trials=n_trials,
        input_units=len(units), included_units=int(valid.sum()), regions=summaries,
        display_y_limits=list(cfg.display_y_limits), full_y_limits=[float(y[0]), float(y[-1])],
        density_scale=scale, bin_size_s=.05)
    write_json(out/'summary.json', summary)
    method = dict(
        source=str(source),
        observations='At each 50-ms time bin between 0 and 1 s, every eligible CP/ACB unit contributes its trial-averaged normalized response. Uses full response curves rather than one peak per unit. These are distributions across units, not across trials.',
        normalization='Exactly reuse population_psth relative_change: (firing rate at time t minus the unit\'s mean -1 to -0.2 s baseline) divided by its mean rate across the full -2 to +3 s displayed PSTH, pooled over the same 526 trials. This is neither percentage of baseline nor row-maximum normalization. Zero-mean units have an undefined ratio and are excluded; no additional unit filters.',
        density='For each region separately, estimate the normalized-response distribution at each native time with a Gaussian KDE of fixed bandwidth 0.25. Each unit receives equal weight and each distribution has unit integral. For display, mix neighboring native time densities using Gaussian weights with SD 25 ms, normalized at every displayed time. Both regions use identical bandwidths. Original temporal resolution remains 50 ms; smoothing does not add precision.',
        opacity='Use one common maximum density across both regions, all times, and the full response grid: opacity = 0.9 * sqrt(density / common_max). This monotone display transform makes low-density structure visible. No per-region or per-time maximum rescaling. Red CP and blue ACB combine with symmetric multiplicative transmittance on white; overlap is purple and drawing order has no effect. Darkness represents unit density, not uncertainty or statistical confidence.',
        panels='ACB and CP are displayed separately using the same time limits, normalized-response limits, Blues colormap, and shared density normalization. A common square-root color mapping (gamma 0.5) reveals lower-density structure; the single color bar reports density values. No panel-specific color scaling or alpha blending. Density estimates are unchanged from the overlay.',
        range='The main view shows normalized responses from -4 to 8. Density is estimated and normalized over the full data range before cropping the view; out-of-range observations are never clipped into boundary bins or discarded from estimation. Summary records the fraction outside each bound by region. A full-range companion includes the rare large ratios produced by low mean firing rates.',
        interpretation='Descriptive single-recording distributions using SUA and MUA. This display does not identify clusters, track individual units, or isolate cue effects from choice, outcome, and licking.',
        source_hashes={str(p):sha256(p) for p in [Path(__file__), source/'units.csv', source/'population_psth.npz', source/'config.yaml']})
    write_json(out/'method.json', method)
    OmegaConf.save(cfg, out/'config.yaml')
    report = ['<!doctype html><meta charset="utf-8"><title>CP and ACB response density</title>',
        '<style>body{font:16px system-ui;line-height:1.5;max-width:1100px;margin:32px auto;color:#243247;padding:0 24px}img{max-width:100%}</style>',
        '<h1>CP and ACB response density across time</h1>',
        '<p>X is time after the go cue; Y is baseline-subtracted firing divided by unit mean firing. ACB and CP appear in separate panels with identical axes and a shared density color scale. Darker blue indicates higher unit density. Each region has a unit-integral distribution to account for unequal unit counts.</p>',
        '<p>The denominator is each unit’s mean across the original -2 to +3 s PSTH. The main view shows -4 to 8; the full-range version includes rare large ratios. No peak-time selection or response-strength filter is applied. Three zero-mean ACB units have undefined ratios.</p>',
        pd.DataFrame(summaries).to_html(index=False)]
    for name in figures:
        report.append(f'<p><a href="{name}">{name}</a></p><img src="{name}">')
    report.append('<p><a href="units.csv">Unit inclusion</a> · <a href="response_density.npz">Density and response arrays</a> · <a href="method.json">Methods</a> · <a href="figures_and_tables.zip">Download</a></p>')
    (out/'index.html').write_text('\n'.join(report)+'\n')
    with zipfile.ZipFile(out/'figures_and_tables.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.iterdir()):
            if p.suffix in ['.png', '.pdf', '.csv', '.npz', '.json', '.yaml', '.html']:
                archive.write(p, p.name)
    print(json.dumps(summary, indent=2), flush=True)
    print('Completed:', out, flush=True)


def plot_panels(cfg, out, x, y, density, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    region_index = {str(region): i for i, region in enumerate(cfg.regions)}
    summary_by_region = {s['region']: s for s in summaries}
    norm = PowerNorm(gamma=float(cfg.density_color_gamma), vmin=0, vmax=float(density.max()))
    figures = []
    with plt.rc_context(STYLE):
        for full in [False, True]:
            limits = [y[0], y[-1]] if full else list(cfg.display_y_limits)
            select = (y >= limits[0]-1e-10) & (y <= limits[1]+1e-10)
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharex=True, sharey=True,
                                     layout='constrained')
            for ax, region in zip(axes, cfg.panel_order):
                im = ax.imshow(density[region_index[region], select], origin='lower',
                    aspect='auto', interpolation='bilinear', cmap='Blues', norm=norm,
                    extent=(x[0], x[-1], y[select][0], y[select][-1]))
                ax.axhline(0, color='#64748B', lw=.7, ls='--', alpha=.65)
                ax.set(xlim=tuple(cfg.window_s), ylim=limits,
                       xticks=np.linspace(*cfg.window_s, 6))
                ax.set_title(f'{region} · {summary_by_region[region]["included_units"]} units', fontsize=12, pad=10)
            axes[0].set_ylabel('Δ firing rate / unit mean')
            fig.supxlabel('Time from go cue (s)', fontsize=11)
            fig.suptitle(f'{cfg.animal} · {str(cfg.session)[:10]} · 50 millisecond bins', fontsize=13)
            colorbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=.035,
                                   pad=.025, extend='neither', label='Unit density')
            colorbar.set_ticks(np.linspace(0, norm.vmax, 5))
            colorbar.ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(0, norm.vmax, 5)])
            assert axes[0].images[0].norm is axes[1].images[0].norm
            assert axes[0].get_xlim() == axes[1].get_xlim()
            assert axes[0].get_ylim() == axes[1].get_ylim()
            name = 'response_density_panels_acb_cp' + ('_full_range' if full else '')
            for extension in ['png', 'pdf']:
                fig.savefig(out/f'{name}.{extension}', dpi=240, bbox_inches='tight')
            plt.close(fig)
            figures.append(name+'.png')
    return figures


def plot(cfg, out, x, y, rgb, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    figures = []
    with plt.rc_context(STYLE):
        for full in [False, True]:
            limits = [y[0], y[-1]] if full else list(cfg.display_y_limits)
            select = (y >= limits[0]-1e-10) & (y <= limits[1]+1e-10)
            fig, ax = plt.subplots(figsize=(9, 5.6), layout='constrained')
            ax.imshow(rgb[select], origin='lower', aspect='auto', interpolation='bilinear',
                      extent=(x[0], x[-1], y[select][0], y[select][-1]))
            ax.axhline(0, color='#64748B', lw=.8, ls='--', alpha=.7)
            ax.set(xlim=tuple(cfg.window_s), ylim=limits, xlabel='Time from go cue (s)',
                   ylabel='Δ firing rate / unit mean', xticks=np.linspace(*cfg.window_s, 6))
            handles = [Patch(color=color, label=f'{s["region"]} · {s["included_units"]} units')
                for color, s in zip(['#DC3C44', '#2563EB'], summaries)]
            ax.legend(handles=handles, loc='upper right')
            fig.suptitle('CP and ACB response distributions', fontsize=14)
            ax.set_title(f'{cfg.animal} · {str(cfg.session)[:10]} · 50 millisecond bins',
                         fontsize=10, color='#64748B', pad=10)
            fig.supxlabel('Darker = higher unit density · Purple = overlap', fontsize=9, color='#64748B')
            name = 'response_density_cp_acb' + ('_full_range' if full else '')
            for extension in ['png', 'pdf']:
                fig.savefig(out/f'{name}.{extension}', dpi=240, bbox_inches='tight')
            plt.close(fig)
            figures.append(name+'.png')
    return figures
