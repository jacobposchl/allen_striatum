"""Matched-scale correlation matrices in fixed neuron order."""
import numpy as np
from src.common.plotting import STYLE


def plot(raw, adjusted, units, metadata, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    labels = units.label.to_numpy()
    names = [str(row.unit_id) for row in units.itertuples()]
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + .5
    cmap = LinearSegmentedColormap.from_list('correlation', ['#2166AC', '#F7F7F5', '#B54A30'])
    cmap.set_bad('#DCE1E7')
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(17, 9.7))
        fig.subplots_adjust(left=.055, right=.91, bottom=.15, top=.82, wspace=.16)
        for ax, matrix, title in zip(axes, [raw, adjusted], ['Raw scores', 'Class-wide fluctuations removed']):
            image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
            ax.set_title(title, fontsize=14, pad=37)
            ax.set_xticks(np.arange(len(units)), names, rotation=90, fontsize=6)
            ax.set_yticks(np.arange(len(units)), names, fontsize=6)
            ax.tick_params(length=0, pad=3)
            ax.set_xlabel('Neuron ID', labelpad=10)
            for boundary in boundaries:
                ax.axhline(boundary, color='#475569', linewidth=.8)
                ax.axvline(boundary, color='#475569', linewidth=.8)
            for label, color in [('D1', '#7161A5'), ('D2', '#218B87')]:
                rows = np.flatnonzero(labels == label)
                if not len(rows):
                    continue
                lo, hi = (rows[0] / len(units), (rows[-1] + 1) / len(units))
                ax.plot([lo, hi], [1.014, 1.014], transform=ax.transAxes, clip_on=False, color=color, lw=4, solid_capstyle='butt')
                ax.text((lo + hi) / 2, 1.035, label, color=color, transform=ax.transAxes, ha='center', fontsize=11)
            for spine in ax.spines.values():
                spine.set_visible(False)
        cax = fig.add_axes([.935, .34, .012, .32])
        bar = fig.colorbar(image, cax=cax, ticks=[-1, -.5, 0, .5, 1])
        bar.set_label('Pearson r', labelpad=9)
        bar.outline.set_visible(False)
        fig.suptitle('745502 · Dec 10 · Sep 12 model', x=.055, y=.97, ha='left', fontsize=17)
        fig.text(.055, .935, f"{len(units)} neurons · {metadata['n_windows']} × 10-second windows · availability >{metadata['presence_threshold']:.0%}", fontsize=11, color='#64748B')
        fig.text(.055, .055, 'Shared available windows per pair · Same neuron order and color scale · Diagonal omitted', fontsize=10, color='#64748B')
        fig.text(.055, .025, 'Class mean excludes both compared neurons; original labeled reference pool retained.', fontsize=10, color='#64748B')
        for suffix in ['png', 'pdf']:
            fig.savefig(output / f'pairwise_correlations.{suffix}', dpi=240, bbox_inches='tight', pad_inches=.18)
        plt.close(fig)
