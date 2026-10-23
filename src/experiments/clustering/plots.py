"""Compact descriptive plots for the per-recording clustering screen."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


PALETTE = ['#3976B9', '#DB8141', '#299D8F', '#A770B4', '#CB5B70', '#9A9144', '#577488', '#A26E51']
FEATURES = [('analysis_firing_rate_hz', 'Firing rate (Hz)'),
            ('peak_to_valley_ms', 'Peak–valley (ms)'), ('analysis_isi_cv', 'ISI CV')]


def bundle(output, summary):
    """A local browser and portable PNG/table bundle for reviewing all recordings."""
    import html
    import zipfile
    rows = []
    for r in summary.itertuples():
        link = f'recordings/{r.key}/clusters_and_features.png'
        rows.append(f'<tr><td><a href="{html.escape(link)}">{r.animal} · {r.session[:10]}</a></td>'
                    f'<td>{html.escape(r.recording)}</td><td>{r.n_included}</td><td>{r.n_clusters}</td>'
                    f'<td>{r.silhouette:.2f}</td><td>{r.stability_median_ari:.2f}</td>'
                    f'<td>{"Yes" if r.separated_stable else "No"}</td></tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Sep 12 recording clusters</title>
<style>body{font:16px system-ui;color:#253247;max-width:1150px;margin:48px auto;padding:0 24px;line-height:1.5}
a{color:#2869a5}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #e0e6eb}
th{background:#f2f5f8}p{max-width:950px}img{width:100%;height:auto}</style>
<h1>Sep 12 · clusters within each recording</h1>
<p>Nearest-neighbor communities in 48D mean embeddings; SUA and MUA included, noise excluded. Features and labels do not define clusters. Colors identify groups only within a recording.</p>
<p><a href="all_recordings.png">All recordings</a> · <a href="feature_associations.png">Physiology comparisons</a> ·
<a href="cluster_profiles.csv">Cluster feature table</a> · <a href="units.csv">Unit assignments</a> · <a href="method.json">Methods and provenance</a></p>
<p>The descriptive screen requires silhouette ≥0.20, median adjusted Rand index (ARI) ≥0.70 across neighborhood settings and seeds, and at least eight units per group. Passing is not evidence of a specific cell type. Rate and ISI summaries derive from the same spike trains used by the encoder. Waveform differences do not establish cholinergic identity.</p>
<table><thead><tr><th>Animal · date</th><th>Recording</th><th>Units</th><th>Groups</th><th>Silhouette</th><th>Stability</th><th>Screen</th></tr></thead><tbody>'''
    (output / 'index.html').write_text(page + ''.join(rows) + '</tbody></table></html>')
    paths = [output / 'index.html', *output.glob('*.png'), *output.glob('*.csv'),
             *output.glob('*.json'), output / 'config.yaml',
             *output.glob('recordings/*/clusters_and_features.png')]
    with zipfile.ZipFile(output / 'figures_and_tables.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(output))


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#344152', 'text.color': '#253247',
                         'axes.edgecolor': '#B5BDC7', 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'pdf.fonttype': 42})


def strip(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def scatter_clusters(ax, f):
    for c, g in f.groupby('cluster'):
        ax.scatter(g.umap_1, g.umap_2, s=16, color=PALETTE[int(c) % len(PALETTE)],
                   alpha=.85, linewidths=0, label=f'C{int(c) + 1}')
    strip(ax)


def plot(units, summary, tests, output):
    style()
    n = len(summary)
    fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(13, 2.65 * np.ceil(n / 4)))
    for ax, (_, row) in zip(axes.flat, summary.iterrows()):
        f = units.loc[units.key.eq(row.key) & units.included]
        scatter_clusters(ax, f)
        experiment = row.recording.replace('experiment', 'e').replace('_recording', 'r')
        ax.set_title(f'{row.animal} · {row.session[:10]} · {experiment}', fontsize=9)
        ax.text(.02, .01, f'{len(f)} units · S {row.silhouette:.2f} · ARI {row.stability_median_ari:.2f}',
                transform=ax.transAxes, fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle('Sep 12 · clusters within each recording', fontsize=15, y=.998)
    fig.text(.5, .002, 'Colors are recording-specific · saved UMAP display · S: 48D silhouette · ARI: stability across settings',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .015, 1, .985))
    fig.savefig(output / 'all_recordings.png', dpi=180)
    fig.savefig(output / 'all_recordings.pdf')
    plt.close(fig)

    for _, row in summary.iterrows():
        f = units.loc[units.key.eq(row.key) & units.included]
        clusters = sorted(f.cluster.unique())
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.6), gridspec_kw={'height_ratios': [1.2, 1]})
        scatter_clusters(axes[0, 0], f)
        axes[0, 0].set_title('Embedding clusters')
        axes[0, 0].legend(frameon=False, fontsize=8, ncol=2, loc='best', handletextpad=.2)
        for j, (feature, label) in enumerate(FEATURES, 1):
            values = f[feature].to_numpy(float)
            valid = np.isfinite(values)
            color = np.log10(np.maximum(values, .001)) if feature == 'analysis_firing_rate_hz' else values
            if valid.any():
                lo, hi = np.nanquantile(color[valid], [.02, .98])
                sc = axes[0, j].scatter(f.umap_1, f.umap_2, c=color, cmap='viridis',
                                         norm=Normalize(lo, max(hi, lo + 1e-9)), s=16, linewidths=0)
                bar = fig.colorbar(sc, ax=axes[0, j], fraction=.04, pad=.02)
                bar.ax.tick_params(labelsize=8)
            else:
                axes[0, j].text(.5, .5, 'Unavailable', ha='center', transform=axes[0, j].transAxes)
            axes[0, j].set_title('log₁₀ rate (Hz)' if feature == 'analysis_firing_rate_hz' else label)
            strip(axes[0, j])
            ax = axes[1, j - 1]
            arrays = [f.loc[f.cluster.eq(c), feature].dropna().to_numpy(float) for c in clusters]
            box = ax.boxplot(arrays, showfliers=False, patch_artist=True, widths=.55,
                             medianprops={'color': '#253247', 'linewidth': 1.3})
            ax.set_xticks(np.arange(1, len(clusters) + 1), [f'C{c + 1}' for c in clusters])
            for c, patch in zip(clusters, box['boxes']):
                patch.set_facecolor(PALETTE[c % len(PALETTE)])
                patch.set_alpha(.6)
            rng = np.random.default_rng(0)
            for c, arr in zip(clusters, arrays):
                ax.scatter(c + 1 + rng.uniform(-.14, .14, len(arr)), arr, s=7,
                           color=PALETTE[c % len(PALETTE)], alpha=.45, linewidths=0)
            ax.set_ylabel(label)
            if feature == 'analysis_firing_rate_hz':
                ax.set_yscale('log')
            ax.grid(axis='y', alpha=.12)
        ax = axes[1, 3]
        sua = [int((f.cluster.eq(c) & f.decoder_label.eq('sua')).sum()) for c in clusters]
        mua = [int((f.cluster.eq(c) & f.decoder_label.eq('mua')).sum()) for c in clusters]
        ax.bar(np.arange(len(clusters)), sua, color='#4D6785', label='SUA')
        ax.bar(np.arange(len(clusters)), mua, bottom=sua, color='#CDD6DF', label='MUA')
        ax.set_xticks(np.arange(len(clusters)), [f'C{c + 1}' for c in clusters])
        ax.set_ylabel('Units')
        ax.legend(frameon=False)
        ax.set_title('Sorting composition')
        fig.suptitle(f'{row.animal} · {row.session[:10]} · {row.recording}', fontsize=14)
        fig.text(.5, .012, f'{len(f)} units · 48D silhouette {row.silhouette:.2f} · stability ARI {row.stability_median_ari:.2f} · colors apply within this recording',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .03, 1, .95))
        directory = output / 'recordings' / row.key
        fig.savefig(directory / 'clusters_and_features.png', dpi=180)
        fig.savefig(directory / 'clusters_and_features.pdf')
        plt.close(fig)

    feature_order = ['analysis_firing_rate_hz', 'peak_to_valley_ms', 'half_width_ms',
                     'analysis_isi_cv', 'analysis_frac_isi_lt_10ms', 'depth', 'available_fraction']
    selected = tests.loc[tests.scope.eq('SUA + MUA')]
    effect = selected.pivot(index='key', columns='feature', values='epsilon_squared').reindex(index=summary.key, columns=feature_order)
    q = selected.pivot(index='key', columns='feature', values='q').reindex(index=summary.key, columns=feature_order)
    fig, ax = plt.subplots(figsize=(10, 10))
    m = ax.imshow(effect.to_numpy(), aspect='auto', cmap='Blues', vmin=0, vmax=1)
    for i in range(len(effect)):
        for j in range(len(feature_order)):
            if np.isfinite(effect.iloc[i, j]):
                ax.text(j, i, f'{effect.iloc[i, j]:.2f}' + ('*' if q.iloc[i, j] < .05 else ''),
                        ha='center', va='center', fontsize=8,
                        color='white' if effect.iloc[i, j] > .65 else '#253247')
    ax.set_yticks(np.arange(n), [f'{r.animal} · {r.session[:10]} · {r.recording.replace("experiment", "e").replace("_recording", "r")}' for r in summary.itertuples()])
    ax.set_xticks(np.arange(len(feature_order)), ['Rate', 'Peak–valley', 'Half-width', 'ISI CV', 'ISI <10 ms', 'Depth', 'Coverage'], rotation=35, ha='right')
    ax.set_title('Cluster differences · effect size', pad=15)
    fig.colorbar(m, ax=ax, label='Rank effect size (ε²)', shrink=.6)
    fig.text(.53, .015, '* Exploratory FDR q < 0.05 · SUA + MUA · blank: unavailable', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(output / 'feature_associations.png', dpi=180)
    fig.savefig(output / 'feature_associations.pdf')
    plt.close(fig)
