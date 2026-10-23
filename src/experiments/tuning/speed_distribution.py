"""Task speed distributions from nonoverlapping 500-ms observations."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.common.plotting import STYLE
from src.data.io import resolve
from src.data.recording import Recording


def bin_speed(recording, bin_s, minimum_coverage):
    times, velocity = recording.signal('speed', domain='behavior')
    velocity = np.asarray(velocity).reshape(-1)
    if len(times) != len(velocity) or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Speed requires a strictly increasing one-dimensional time series')
    native_dt = float(np.median(np.diff(times)))
    if native_dt > bin_s:
        raise ValueError('Speed sampling is slower than the requested time bins')
    required = int(np.ceil(minimum_coverage * bin_s / native_dt - 1e-6))
    rows = []
    for start, stop in recording.intervals('behavior'):
        n = int(np.floor((stop - start) / bin_s))
        if n == 0:
            continue
        keep = (times >= start) & (times < start + n * bin_s) & np.isfinite(velocity)
        index = np.floor((times[keep] - start) / bin_s).astype(int)
        count = np.bincount(index, minlength=n)
        total = np.bincount(index, weights=np.abs(velocity[keep]), minlength=n)
        means = np.divide(total, count, out=np.full(n, np.nan), where=count >= required)
        rows.append(pd.DataFrame(dict(start_s=start + np.arange(n) * bin_s,
                                      mean_speed_cm_s=means, native_samples=count,
                                      valid=count >= required)))
    if not rows:
        raise ValueError('No complete task time bins')
    return pd.concat(rows, ignore_index=True), native_dt


def plot(out, summaries, histograms, speed_width):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullLocator

    n = len(summaries)
    columns = min(2, n)
    rows = int(np.ceil(n / columns))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.4 * rows),
                                 sharex=True, sharey=True, squeeze=False, layout='constrained')
        maximum = max(h['count'].max() for h in histograms)
        top = 10 ** np.ceil(np.log10(maximum * 1.2))
        for ax, summary, histogram in zip(axes.ravel(), summaries, histograms):
            edges = np.r_[histogram.speed_left_cm_s, histogram.speed_right_cm_s.iloc[-1]]
            ax.stairs(histogram['count'], edges, color='#247d88', linewidth=1.7)
            ax.set(yscale='log', ylim=(.8, top), xlim=(0, edges[-1]),
                   title=f'{summary["animal"]} · {summary["session"][:10]}')
            ax.text(.97, .94, f'{summary["valid_bins"]:,} bins', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10, color='#64748b')
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_locator(MultipleLocator(2 if edges[-1] <= 16 else 5))
            ax.grid(axis='y', alpha=.2, linewidth=.6)
            ax.tick_params(length=0, pad=6)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        fig.supxlabel('Mean running speed in 500 ms (cm/s)', fontsize=12)
        fig.supylabel('Number of 500-ms bins · log scale', fontsize=12)
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'speed_distributions.{suffix}', dpi=200, bbox_inches='tight')
        # A linear-count version preserves the literal visual prominence of quiet time.
        for ax in axes.ravel()[:n]:
            ax.set_yscale('linear')
            ax.set_ylim(0, maximum * 1.12)
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        fig.supylabel('Number of 500-ms bins', fontsize=12)
        fig.savefig(out / 'speed_distributions_linear.png', dpi=200, bbox_inches='tight')
        plt.close(fig)


def plot_task_traces(out, log_column=False):
    """Stack the saved 500-ms task traces, preserving missing observations."""
    from src.data.io import require_compute
    require_compute()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, NullLocator

    out = Path(out)
    frame = pd.read_csv(out / 'speed_500ms.csv')
    method = json.loads((out / 'method.json').read_text())
    dt = float(method['temporal_bin_s'])
    groups = list(frame.groupby(['animal', 'session'], sort=False))
    xmax = float((frame.start_s.max() + dt) / 60)
    ymax = float(np.ceil(frame.mean_speed_cm_s.max()))
    columns = 2 if log_column else 1
    name = 'running_speed_linear_log' if log_column else 'running_speed_stacked'
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(groups), columns,
                                 figsize=(16 if log_column else 13.5, 3.1 * len(groups)),
                                 sharex=True, sharey='col', squeeze=False, layout='constrained')
        for row, ((animal, session), values) in enumerate(groups):
            speed = values.mean_speed_cm_s.to_numpy().copy()
            speed[~values.valid.to_numpy(bool)] = np.nan
            times = (values.start_s.to_numpy() + dt / 2) / 60
            # Insert explicit breaks if separate behavioral intervals are present.
            breaks = np.flatnonzero(np.diff(times) > dt / 60 * 1.5) + 1
            speed = np.insert(speed, breaks, np.nan)
            times = np.insert(times, breaks, times[breaks])
            end = float((values.start_s.iloc[-1] + dt) / 60)
            for column, ax in enumerate(axes[row]):
                ax.plot(times, speed, color='#247d88', lw=.7 if log_column else .85)
                if end < xmax - 1:
                    ax.axvline(end, color='#94a3b8', lw=.8, ls=(0, (3, 3)))
                    ax.text(end + .7, .85, 'Task end', transform=ax.get_xaxis_transform(),
                            color='#64748b', fontsize=10)
                label = f'{animal} · {session[:10]}'
                if log_column:
                    label += ' · Linear' if column == 0 else ' · Log scale'
                ax.set(title=label, xlim=(0, xmax), ylim=(0, ymax * 1.04))
                ax.title.set_horizontalalignment('left')
                ax.title.set_position((0, 1))
                ax.xaxis.set_major_locator(MultipleLocator(10))
                if column == 1:
                    ax.set_yscale('symlog', linthresh=.1, linscale=.5, base=10)
                    ax.set_ylim(0, 10 ** np.ceil(np.log10(ymax)))
                    ax.set_yticks([0, .1, 1, 10], ['0', '0.1', '1', '10'])
                    ax.yaxis.set_minor_locator(NullLocator())
                else:
                    ax.yaxis.set_major_locator(MultipleLocator(2))
                ax.grid(alpha=.16, lw=.6)
                ax.tick_params(length=0, pad=5)
        fig.supxlabel('Recording time (min)', fontsize=12)
        fig.supylabel('Running speed (cm/s)', fontsize=12)
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'{name}.{suffix}', dpi=200, bbox_inches='tight')
        plt.close(fig)
    (out / f'{name}_method.json').write_text(json.dumps(dict(
        source='speed_500ms.csv', temporal_bin_s=dt,
        signal='Mean absolute wheel velocity; no additional smoothing',
        time='Bin centers, minutes from recording clock origin; shared axes',
        scope='Entire saved task period for each session; baseline and optotagging excluded',
        missing='Missing samples remain gaps; shorter task ends at the marked line',
        speed_scale=('Left linear; right base-10 symlog: linear from 0 to 0.1 cm/s, logarithmic above 0.1 cm/s; zeros retained'
                     if log_column else 'Linear'),
    ), indent=2) + '\n')


def plot_combined(out):
    """Session columns with distribution, linear trace, and log trace rows."""
    from src.data.io import require_compute
    require_compute()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullLocator

    out = Path(out)
    frame = pd.read_csv(out / 'speed_500ms.csv')
    histograms = pd.read_csv(out / 'histograms.csv')
    method = json.loads((out / 'method.json').read_text())
    dt = float(method['temporal_bin_s'])
    groups = list(frame.groupby(['animal', 'session'], sort=False))
    xmax = float((frame.start_s.max() + dt) / 60)
    ymax = float(np.ceil(frame.mean_speed_cm_s.max()))
    hist_top = 10 ** np.ceil(np.log10(histograms['count'].max() * 1.2))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, len(groups), figsize=(7 * len(groups), 10.3),
                                 sharey='row', squeeze=False, layout='constrained')
        for column, ((animal, session), values) in enumerate(groups):
            h = histograms.loc[histograms.animal.eq(animal) & histograms.session.eq(session)]
            edges = np.r_[h.speed_left_cm_s, h.speed_right_cm_s.iloc[-1]]
            ax = axes[0, column]
            ax.stairs(h['count'], edges, color='#247d88', linewidth=1.7)
            ax.set(title=f'{animal} · {session[:10]}', yscale='log', ylim=(.8, hist_top),
                   xlim=(0, edges[-1]), xlabel='Mean speed in 500 ms (cm/s)')
            ax.text(.97, .94, f'{int(h["count"].sum()):,} bins', transform=ax.transAxes,
                    ha='right', va='top', fontsize=10, color='#64748b')
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.grid(axis='y', alpha=.2, linewidth=.6)

            speed = values.mean_speed_cm_s.to_numpy().copy()
            speed[~values.valid.to_numpy(bool)] = np.nan
            times = (values.start_s.to_numpy() + dt / 2) / 60
            breaks = np.flatnonzero(np.diff(times) > dt / 60 * 1.5) + 1
            speed = np.insert(speed, breaks, np.nan)
            times = np.insert(times, breaks, times[breaks])
            end = float((values.start_s.iloc[-1] + dt) / 60)
            for row in [1, 2]:
                ax = axes[row, column]
                ax.plot(times, speed, color='#247d88', linewidth=.7)
                ax.set(xlim=(0, xmax), ylim=(0, ymax * 1.04))
                ax.xaxis.set_major_locator(MultipleLocator(10))
                if row == 1:
                    ax.yaxis.set_major_locator(MultipleLocator(2))
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_yscale('symlog', linthresh=.1, linscale=.5, base=10)
                    ax.set_ylim(0, 10 ** np.ceil(np.log10(ymax)))
                    ax.set_yticks([0, .1, 1, 10], ['0', '0.1', '1', '10'])
                    ax.yaxis.set_minor_locator(NullLocator())
                    ax.set_xlabel('Recording time (min)')
                if end < xmax - 1:
                    ax.axvline(end, color='#94a3b8', lw=.8, ls=(0, (3, 3)))
                    ax.text(end + .7, .86, 'Task end', transform=ax.get_xaxis_transform(),
                            color='#64748b', fontsize=10)
                ax.grid(alpha=.16, linewidth=.6)
        axes[0, 0].set_ylabel('500-ms bins\nLog count')
        axes[1, 0].set_ylabel('Speed (cm/s)\nLinear')
        axes[2, 0].set_ylabel('Speed (cm/s)\nLog scale')
        for ax in axes.ravel():
            ax.tick_params(length=0, pad=5)
        for suffix in ['png', 'pdf', 'svg']:
            fig.savefig(out / f'running_speed_combined.{suffix}', dpi=200, bbox_inches='tight')
        plt.close(fig)
    (out / 'running_speed_combined_method.json').write_text(json.dumps(dict(
        sources=['speed_500ms.csv', 'histograms.csv'], temporal_bin_s=dt,
        layout='Columns: sessions; rows: distribution, linear speed, logarithmic speed',
        signal='Saved mean absolute wheel velocity; no additional smoothing or feature computation',
        distribution='Counts per 0.25 cm/s range on a base-10 logarithmic count axis',
        log_speed='Base-10 symlog: linear from 0 to 0.1 cm/s, logarithmic above; zeros retained',
        time='Minutes from recording clock origin, shared trace limits; full task periods only',
        missing='Missing speed remains missing; shorter task endpoint marked',
    ), indent=2) + '\n')


def run(cfg):
    from src.cli import require_compute
    require_compute('speed_distribution', cfg)
    out = resolve(cfg.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty')
    available, inventory = [], []
    for path in sorted(resolve(cfg.recording_root).glob('*.h5')):
        with Recording(path) as recording:
            quality = recording.signal_quality('speed', domain='behavior')
            row = dict(animal=recording.info['animal'], session=recording.info['session'],
                       recording=recording.info['recording'], source=str(path.resolve()),
                       available=recording.has_signal('speed', domain='behavior'),
                       status=quality['status'])
            inventory.append(row)
            if row['available']:
                available.append(row)
    # One task recording per distinct session; prefer the earliest recording ID.
    chosen, seen = [], set()
    for row in available:
        key = (row['animal'], row['session'])
        if key not in seen and len(chosen) < int(cfg.maximum_sessions):
            chosen.append(row)
            seen.add(key)
    if not chosen:
        raise ValueError('No available task speed signals')
    observations, summaries = [], []
    for row in chosen:
        with Recording(row['source']) as recording:
            frame, native_dt = bin_speed(recording, float(cfg.bin_s), float(cfg.minimum_coverage))
            frame.insert(0, 'session', row['session'])
            frame.insert(0, 'animal', row['animal'])
            frame['start_s'] -= float(recording.info['origin_s'])
            speed = frame.loc[frame.valid, 'mean_speed_cm_s'].to_numpy()
            if not len(speed):
                raise ValueError(f'No valid time bins: {row["source"]}')
            summaries.append(dict(**row, native_dt_s=native_dt, total_bins=len(frame),
                                  valid_bins=len(speed), missing_bins=int((~frame.valid).sum()),
                                  valid_minutes=len(speed) * float(cfg.bin_s) / 60,
                                  zero_speed_bins=int((speed == 0).sum()),
                                  below_half_cm_s_percent=float((speed < .5).mean() * 100),
                                  maximum_cm_s=float(speed.max()),
                                  source_identity=recording.identity))
            observations.append(frame)
    width = float(cfg.speed_bin_cm_s)
    upper = max(width, np.ceil(max(s['maximum_cm_s'] for s in summaries) / width) * width)
    edges = np.arange(int(round(upper / width)) + 1) * width
    histograms = []
    for frame, summary in zip(observations, summaries):
        counts, _ = np.histogram(frame.loc[frame.valid, 'mean_speed_cm_s'], edges)
        assert int(counts.sum()) == summary['valid_bins']
        histograms.append(pd.DataFrame(dict(animal=summary['animal'], session=summary['session'],
            speed_left_cm_s=edges[:-1], speed_right_cm_s=edges[1:], count=counts,
            percent=counts / counts.sum() * 100)))
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / 'config.yaml')
    pd.DataFrame(inventory).to_csv(out / 'speed_availability.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'sessions.csv', index=False)
    pd.concat(observations, ignore_index=True).to_csv(out / 'speed_500ms.csv', index=False)
    pd.concat(histograms, ignore_index=True).to_csv(out / 'histograms.csv', index=False)
    method = dict(requested_sessions=int(cfg.maximum_sessions), available_sessions=len(summaries),
        temporal_bin_s=float(cfg.bin_s), histogram_speed_bin_cm_s=width,
        speed='Mean absolute source velocity in nonoverlapping 500-ms bins, converted to cm/s; no extra smoothing',
        scope='Behavior domain only, including quiet and intertrial time; excludes baseline and optotagging; no trial-outcome or neural-unit filtering',
        missing='Require at least 80% of expected native samples; incomplete terminal bins and unobserved intervals excluded, never replaced with zero',
        plotting='Identical speed bins and axis ranges; counts on a log scale show the sparse running tail; separate linear-count version saved; no bout threshold applied',
        selection='First available task recording from each distinct session in animal/date order; up to four sessions',
        sessions=summaries)
    (out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    plot(out, summaries, histograms, width)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f'Saved {len(summaries)} session panels: {out}', flush=True)
    return out
