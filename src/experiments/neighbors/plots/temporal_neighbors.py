"""A fixed row per labeled unit, colored by same-class neighbor fraction."""
import numpy as np

from src.common.plotting import STYLE


def plot(agreement, units, starts, duration, information, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    from matplotlib.patches import Patch

    k = information["k"]
    colors = LinearSegmentedColormap.from_list("label_agreement", ["#8B1026", "#F7F5EF", "#00552E"], N=k + 1)
    colors.set_bad("#BCC3CB")
    norm = BoundaryNorm(np.linspace(-.5 / k, 1 + .5 / k, k + 2), k + 1)
    height = max(7, len(units) * .095 + 1.4)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(15.5, height))
        fig.subplots_adjust(left=.105, right=.86, bottom=.055, top=.945)
        # Draw each contiguous segment separately so excluded periods remain gaps.
        splits = np.flatnonzero(~np.isclose(np.diff(starts), duration, rtol=0, atol=1e-6)) + 1
        ax.set_facecolor("#BCC3CB")
        for columns in np.split(np.arange(len(starts)), splits):
            if not len(columns):
                continue
            image = ax.imshow(np.ma.masked_invalid(agreement[:, columns]), aspect="auto", interpolation="nearest",
                              cmap=colors, norm=norm, origin="upper",
                              extent=(starts[columns[0]] / 60, (starts[columns[-1]] + duration) / 60,
                                      len(units) - .5, -.5))
        ax.set_xlim(starts[0] / 60, (starts[-1] + duration) / 60)
        ax.set_yticks(np.arange(len(units)), [f"{row.label} · {row.unit_id}" for row in units.itertuples()], fontsize=6.5)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.set_xlabel("Time in session (min)", labelpad=9)
        ax.set_ylabel("Labeled unit", labelpad=10)
        ax.set_title(f"{information['animal']} · {information['session'][:10]}\n{k}-neighbor label agreement", loc="left", fontsize=14, pad=13)
        boundaries = np.flatnonzero(units.label.to_numpy()[1:] != units.label.to_numpy()[:-1]) + .5
        for boundary in boundaries:
            ax.axhline(boundary, color="#334155", linewidth=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cax = fig.add_axes([.89, .55, .016, .25])
        bar = fig.colorbar(image, cax=cax, ticks=[0, .5, 1])
        bar.ax.set_yticklabels(["0 · opposite", f"{k//2} · mixed", f"{k} · same"])
        bar.set_label(f"Same-class neighbors (of {k})", labelpad=10)
        bar.outline.set_visible(False)
        fig.legend(handles=[Patch(facecolor="#BCC3CB", label="Unavailable")], loc="upper left",
                   bbox_to_anchor=(.88, .52), fontsize=9)
        fig.text(.89, .47, "Cosine distance\nSame window\nSelf excluded", fontsize=9, linespacing=1.5, va="top")
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.15)
        plt.close(fig)
