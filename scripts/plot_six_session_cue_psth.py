"""Six cue-aligned population panels with bounded, serial data reduction."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import require_compute
from src.data.recording import Recording, resolve_recording
from src.data.task import audit_trials
OUT = ROOT / "results/experiments/active_function/six_session_cue_psth"
SESSIONS = [("745502", "2024-12-10_12-06-14"), ("745502", "2024-12-18_11-34-52"),
            ("754313", "2025-03-31_14-54-44"), ("754313", "2025-04-02_17-05-37"),
            ("754313", "2025-04-08_15-52-46"), ("754313", "2025-04-09_15-18-24")]
EDGES = -2 + np.arange(101) * .05
TAG_COLORS = {"D1": "#2563EB", "D2": "#D97706", "Unlabeled": "#D4D9E1"}
REGION_COLORS = {"CP": "#9467BD", "ACB": "#8C564B", "Unknown": "#9AA7B8",
                 "GPe": "#477A8F", "MA": "#749D80", "SI": "#C09A63",
                 "VL": "#A884A1", "PAL": "#7D8A60", "STR": "#7F9CAC"}


def trial_metadata(recording):
    trials, info = audit_trials(recording)
    origin = float(info["clock"]["origin_s"])
    cue = trials.goCue_start_time
    keep = (trials.usable.eq(True) & np.isfinite(cue) & (cue - 2 >= trials.start_time)
            & (cue + 3 <= trials.stop_time) & (cue - 2 >= 0)
            & (cue + 3 <= info["opto_start_s"]))
    selected = trials.loc[keep].sort_values("goCue_start_time").reset_index(drop=True)
    if not len(selected) or not selected.id.is_unique:
        raise ValueError("No valid unique cue trials.")
    return selected, {"raw_trials": len(trials), "eligible_trials": len(selected),
                      "trial_metadata_source": str(recording.path), "recording_origin_s": origin}


def select_units(recording):
    g = recording.units
    fields = ["id", "decoder_label", "predicted_cell_type", "depth", "device_name", "ccf_acronym", "label"]
    u = g[[key for key in fields if key in g] + ["source_unit_row"]].copy()
    u = u.loc[u.decoder_label.isin(["sua", "mua"])].copy()
    if not u.label.isin(TAG_COLORS).all():
        raise ValueError("Unexpected cell-type label.")
    anatomy_available = "ccf_acronym" in u and u.ccf_acronym.notna().any()
    u["region"] = u.ccf_acronym.fillna("Unknown").astype(str) if anatomy_available else "Unknown"
    striatal = u.region.isin(["CP", "ACB"])
    if striatal.any():
        u = u.loc[striatal].copy()
        region_order = [r for r in ["CP", "ACB"] if r in u.region.unique()]
        scope = "CP and ACB"
    else:
        region_order = u.groupby("region").depth.median().sort_values(ascending=False, kind="stable").index.tolist()
        scope = "All non-noise units; region unavailable" if not anatomy_available else "All non-noise units in other recorded regions; no non-noise CP/ACB units"
    u["region_order"] = u.region.map({r: i for i, r in enumerate(region_order)})
    u = u.sort_values(["region_order", "device_name", "depth", "id"],
                      ascending=[True, True, False, True], kind="stable").reset_index(drop=True)
    u["plot_row"] = np.arange(len(u))
    return u, scope


def prepare_panel(animal, session):
    destination = OUT / f"{animal}_{session[:10]}"
    path = resolve_recording(f"{animal}/{session}/experiment1_recording1")
    with Recording(path) as recording:
        units, scope = select_units(recording)
        trials, trial_audit = trial_metadata(recording)
        print(f"Reducing {animal} {session[:10]}: {len(units)} units, {len(trials)} trials", flush=True)
        cues = trials.goCue_start_time.to_numpy()
        display_queries = cues[:, None] + EDGES[30:61]
        interval_queries = cues[:, None] + np.array([-1., -.2, -2., 3.])
        rate = np.empty((len(units), 30))
        baseline, mean = np.empty(len(units)), np.empty(len(units))
        for iteration, row in enumerate(units.sort_values("source_unit_row").itertuples()):
            spikes = recording.spikes(row.source_unit_row) - trial_audit["recording_origin_s"]
            if not np.isfinite(spikes).all() or np.any(np.diff(spikes) < 0):
                raise ValueError("Invalid spike timestamps.")
            positions = np.searchsorted(spikes, display_queries, side="left")
            bounds = np.searchsorted(spikes, interval_queries, side="left")
            rate[row.plot_row] = np.diff(positions, axis=1).mean(axis=0) / .05
            baseline[row.plot_row] = (bounds[:, 1] - bounds[:, 0]).mean() / .8
            mean[row.plot_row] = (bounds[:, 3] - bounds[:, 2]).mean() / 5
            if iteration % 100 == 99:
                print(f"  {iteration + 1}/{len(units)} units", flush=True)
    method = "Exact half-open 50-ms bins from canonical Brainsets spikes; one unit at a time"
    relative = np.divide(rate - baseline[:, None], mean[:, None],
                         out=np.full_like(rate, np.nan), where=mean[:, None] > 0)
    units["baseline_rate_hz"] = baseline
    units["normalization_rate_hz"] = mean
    meta = {"animal": animal, "session": session, "source": str(path), "unit_scope": scope,
            "units": len(units), "region_counts": units.groupby("region", sort=False).size().to_dict(),
            "tag_counts": units.label.value_counts().to_dict(), "silent_units": int((mean == 0).sum()),
            "method": method, **trial_audit}
    destination.mkdir(parents=True, exist_ok=True)
    units.to_csv(destination / "units.csv", index=False)
    trials.to_csv(destination / "trials.csv", index=False)
    np.savez_compressed(destination / "cue_psth.npz", firing_rate_hz=rate, relative_change=relative,
                        baseline_rate_hz=baseline, normalization_rate_hz=mean, bin_edges_s=EDGES[30:61])
    (destination / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Ready {animal} {session[:10]}: {len(units)} units, {len(trials)} trials", flush=True)
    return units, relative, meta


def load_panel(destination):
    units = pd.read_csv(destination / "units.csv")
    meta = json.loads((destination / "metadata.json").read_text())
    with np.load(destination / "cue_psth.npz", allow_pickle=False) as saved:
        values = saved["relative_change"]
    return units, values, meta


def draw(panels):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "text.color": "#243247", "axes.labelcolor": "#334155",
                         "xtick.color": "#46556A"})
    # Shared display limits; underlying response values remain unclipped.
    limit = 3.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#D4D9E1")
    if len(panels) != 5:
        raise ValueError("This layout requires five sessions plus one legend panel.")
    fig = plt.figure(figsize=(16.8, 10.6), facecolor="white")
    outer = fig.add_gridspec(2, 3, left=.025, right=.985, top=.94, bottom=.09,
                            wspace=.13, hspace=.23)
    for index, (units, values, meta) in enumerate(panels):
        n = len(units)
        row, column = divmod(index, 3)
        grid = outer[row, column].subgridspec(1, 3, width_ratios=[.97, .14, 5.05], wspace=.06)
        date = datetime.strptime(meta["session"][:10], "%Y-%m-%d").strftime("%b %d, %Y")
        region_ax = fig.add_subplot(grid[0, 0])
        region_ax.set(xlim=(0, 1), ylim=(n, 0))
        region_ax.axis("off")
        groups = [(name, int(g.plot_row.min()), int(g.plot_row.max()) + 1) for name, g in units.groupby("region", sort=False)]
        label_y = np.array([(a + b) / 2 for _, a, b in groups])
        minimum_gap = max(n * .105, 10)
        label_y[0] = max(label_y[0], n * .055)
        for j in range(1, len(label_y)):
            label_y[j] = max(label_y[j], label_y[j - 1] + minimum_gap)
        if label_y[-1] > n - n * .055:
            label_y[-1] = n - n * .055
            for j in range(len(label_y) - 2, -1, -1):
                label_y[j] = min(label_y[j], label_y[j + 1] - minimum_gap)
        for (name, start, stop), y in zip(groups, label_y):
            region_ax.plot([.96, .96], [start, stop], color=REGION_COLORS.get(name, "#8D99AE"), lw=4, solid_capstyle="butt")
            region_ax.plot([.78, .90], [y, (start + stop) / 2], color="#A5AFBD", lw=.9)
            label = "Region unknown" if name == "Unknown" else f"{name} · {stop - start}"
            if name == "Unknown":
                label = "Unknown"
            region_ax.text(.72, y, label, ha="right", va="center", fontsize=12,
                           weight="normal", linespacing=1.4)
        tag_ax = fig.add_subplot(grid[0, 1])
        rgba = np.array([to_rgba(TAG_COLORS[name]) for name in units.label])[:, None, :]
        tag_ax.imshow(rgba, extent=(0, 1, n, 0), aspect="auto", interpolation="nearest")
        tag_ax.axis("off")
        ax = fig.add_subplot(grid[0, 2])
        ax.set_title(f"{meta['animal']} · {date}", fontsize=14.5, pad=13, weight="normal")
        im = ax.imshow(values, extent=(-.5, 1, n, 0), aspect="auto", interpolation="nearest",
                       cmap=cmap, vmin=-limit, vmax=limit, rasterized=True)
        ax.axvline(0, color="#45556C", ls=(0, (3, 3)), lw=1.05)
        for _, _, stop in groups[:-1]:
            ax.axhline(stop, color="white", lw=1.1)
        ax.set(xticks=[-.5, 0, .5, 1], yticks=[], xlim=(-.5, 1), ylim=(n, 0))
        ax.set_xticklabels(["−0.5", "Q", "0.5", "1"])
        ax.tick_params(axis="x", length=3, width=.8, pad=7, labelsize=13)
        ax.get_xticklabels()[1].set_weight("bold")
        ax.get_xticklabels()[1].set_color("#243247")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.supxlabel("Time from go cue (seconds)", x=.53, y=.025, fontsize=15)

    # The sixth grid position is the shared key, with no title or footnotes.
    legend_ax = fig.add_subplot(outer[1, 2])
    legend_ax.axis("off")
    box = legend_ax.get_position()
    color_ax = fig.add_axes([box.x0 + .10 * box.width, box.y0 + .64 * box.height,
                             .84 * box.width, .045 * box.height])
    bar = fig.colorbar(im, cax=color_ax, orientation="horizontal", ticks=np.arange(-3, 4))
    bar.ax.set_xticklabels(["−3", "−2", "−1", "0", "+1", "+2", "+3"])
    bar.ax.tick_params(length=3, width=.8, labelsize=12.5, pad=5)
    bar.outline.set_edgecolor("#CCD5DF")
    bar.outline.set_linewidth(.7)
    legend_ax.text(.52, .75, "Δ firing / mean firing", transform=legend_ax.transAxes,
                   ha="center", va="center", fontsize=14.5)
    legend_ax.legend(handles=[Patch(color=TAG_COLORS[k], label=k) for k in ["D1", "D2", "Unlabeled"]],
                     loc="center left", bbox_to_anchor=(.08, .34), ncol=1, frameon=False,
                     fontsize=13.5, handlelength=1.2, handletextpad=.7, labelspacing=.85)
    legend_ax.text(.94, .34, "50 millisecond bins", transform=legend_ax.transAxes,
                   ha="right", va="center", fontsize=12.5, color="#526070")
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"five_session_go_cue.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    provenance = {
        "panels": [p[2] for p in panels], "display_window_s": [-.5, 1], "bin_size_s": .05,
        "normalization": "Per unit, subtract its trial-averaged -1 to -0.2 s baseline and divide by its trial-averaged mean over -2 to +3 s. Same definition as the supplied December 10 figure; not z-scoring or peak normalization. Zero-mean units stay gray.",
        "trials": "All eligible outcomes pooled, including no-response trials. Use the original behavioral validity criteria and require the full -2 to +3 s normalization window to fall within the trial and before optotagging. No reward/omission or choice-side balancing.",
        "units": "Same non-noise SUA/MUA criterion as the reference, without default-QC or rate cutoffs. Restrict to CP/ACB when these annotations and non-noise units exist. December 18 includes all non-noise units because anatomy is unavailable. April 8 is omitted from this figure. Recorded D1/D2 labels are preserved.",
        "ordering": "CP then ACB where available; other regions by median depth. Within regions, probe then descending stored depth then unit ID. Each row is one unit; rows across different sessions are not matched neurons.",
        "color_limits": [-limit, limit], "smoothing": "None",
        "layout": "Three columns by two rows: five session heatmaps and a shared legend in the bottom-right position. Q labels the go cue at time zero. Color scale saturates at -3 and +3; exact response arrays remain unchanged.",
        "resource_policy": "Serial chunk reads, one unit at a time; paced to 10% of one CPU during reduction. No workers, model fits, compute-node allocation, or full-session spike matrices.",
    }
    (OUT / "five_session_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    require_compute()
    OUT.mkdir(parents=True, exist_ok=True)
    panels = [load_panel(OUT / f"{a}_{s[:10]}") if args.render_only else prepare_panel(a, s)
              for a, s in SESSIONS if s[:10] != "2025-04-08"]
    draw(panels)
    print(json.dumps({"output": str(OUT), "sessions": len(panels)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
