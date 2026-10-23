"""Split cached CP/ACB cue responses by trial outcome; no raw spike processing."""

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
SOURCE = ROOT / "results/experiments/active_function/population_psth_dec10"
OUT = ROOT / "results/experiments/active_function/reward_omission_psth_dec10"
WINDOW = (-.5, 1.)
TAG_COLORS = {"D1": "#2563EB", "D2": "#D97706", "Unlabeled": "#D4D9E1"}
REGION_COLORS = {"CP": "#9467BD", "ACB": "#8C564B"}
INK = "#334155"


def load_responses():
    units = pd.read_csv(SOURCE / "units_cp_acb.csv")
    trials = pd.read_csv(SOURCE / "trials.csv")
    summary = json.loads((SOURCE / "summary.json").read_text())
    rows = units.source_plot_row.to_numpy(int)
    if not np.array_equal(units.plot_row, np.arange(len(units))):
        raise ValueError("Preserve the reference figure's exact unit order.")
    if units.region.drop_duplicates().tolist() != ["CP", "ACB"]:
        raise ValueError("Expected CP then ACB in the reference figure.")
    if not units.label.isin(TAG_COLORS).all():
        raise ValueError("Unrecognized cell-type annotation.")

    with np.load(SOURCE / "population_psth.npz", allow_pickle=False) as cached:
        if not np.array_equal(cached["global_unit_id"][rows], units.global_unit_id.to_numpy(str)):
            raise ValueError("Unit identities do not match cached response rows.")
        if not np.array_equal(cached["trial_id"], trials.id.to_numpy()):
            raise ValueError("Trial identities do not match cached count columns.")
        full_edges = cached["bin_edges_s"]
        lo = np.flatnonzero(np.isclose(full_edges, WINDOW[0])).item()
        hi = np.flatnonzero(np.isclose(full_edges, WINDOW[1])).item()
        edges = full_edges[lo:hi + 1]
        dt = float(np.diff(edges).mean())
        if not np.allclose(np.diff(edges), .05):
            raise ValueError("Expected the reference's exact 50-ms bins.")
        baseline = cached["baseline_rate_hz"][rows]
        normalization = cached["normalization_rate_hz"][rows]
        pooled_reference = cached["firing_rate_hz"][rows, lo:hi]
        relative_reference = cached["relative_change"][rows, lo:hi]

    # Memory map the existing count cache and read only the 318 requested units
    # and 30 displayed time bins (about 10 MB). There is no spike rebinning.
    cached_counts = np.load(SOURCE / "trial_counts.npy", mmap_mode="r", allow_pickle=False)
    if cached_counts.shape[1:] != (len(trials), len(full_edges) - 1):
        raise ValueError("Unexpected cached trial-count dimensions.")
    counts = np.asarray(cached_counts[rows, :, lo:hi])
    pooled = counts.mean(axis=1, dtype=np.float64) / dt
    if not np.allclose(pooled, pooled_reference, rtol=1e-10, atol=1e-10):
        raise ValueError("Cached counts do not reproduce the reference PSTH.")
    pooled_relative = np.divide(pooled - baseline[:, None], normalization[:, None],
                                out=np.full_like(pooled, np.nan), where=normalization[:, None] > 0)
    if not np.allclose(pooled_relative, relative_reference, rtol=1e-10, atol=1e-10, equal_nan=True):
        raise ValueError("Normalization does not reproduce the reference response.")

    responded = trials.animal_response.isin([0, 1]).to_numpy()
    if not trials.rewarded.isin([0, 1]).all():
        raise ValueError("Missing or nonbinary trial reward outcome.")
    masks = {"reward": responded & trials.rewarded.eq(1).to_numpy(),
             "omission": responded & trials.rewarded.eq(0).to_numpy()}
    if np.any(masks["reward"] & masks["omission"]) or not np.array_equal(masks["reward"] | masks["omission"], responded):
        raise ValueError("Outcome groups must partition the responding trials.")
    rates, relative = {}, {}
    for condition, mask in masks.items():
        if not mask.any():
            raise ValueError(f"No trials for {condition}.")
        rates[condition] = counts[:, mask, :].mean(axis=1, dtype=np.float64) / dt
        relative[condition] = np.divide(rates[condition] - baseline[:, None], normalization[:, None],
                                        out=np.full_like(rates[condition], np.nan),
                                        where=normalization[:, None] > 0)
    trials["outcome_panel"] = np.select([masks["reward"], masks["omission"]],
                                       ["reward", "omission"], default="excluded_no_response")
    limit = float(summary["color_limits"]["relative_change"])
    audit = {
        "animal": "745502", "session": "2024-12-10_12-06-14",
        "recording": "experiment1_recording1", "source_cache": str(SOURCE),
        "n_units": len(units), "region_counts": units.region.value_counts().to_dict(),
        "tag_counts": units.label.value_counts().to_dict(),
        "sorting_counts": units.decoder_label.value_counts().to_dict(),
        "source_trials": len(trials), "condition_trials": {k: int(v.sum()) for k, v in masks.items()},
        "excluded_no_response_trials": int((~responded).sum()),
        "window_s": list(WINDOW), "bin_size_s": dt, "bin_count": len(edges) - 1,
        "color_limits": [-limit, limit], "silent_units_retained_as_gray": int((normalization == 0).sum()),
        "trial_selection": "Split the eligible pre-optotagging trials stored by the population-PSTH analysis. Reward = a recorded reward on a responding trial; omission = a responding trial without reward. Exclude no-response trials from both panels. Preserve original trial eligibility even though the display window is shorter. No side matching or trial balancing.",
        "averaging": "Average cached per-trial spike counts within each outcome group, then divide by the 50-ms bin duration. Exact existing bins, no smoothing or new spike processing.",
        "normalization": "For both conditions, subtract the same original per-unit pooled baseline from -1 to -0.2 s and divide by the same original per-unit pooled mean over -2 to +3 s. Both reference quantities pool all 526 original trials, including no-response trials. Cropping and outcome splitting do not recompute these quantities. This is baseline-subtracted firing divided by mean firing, not a z-score.",
        "color_scale": "Both panels retain the reference figure's symmetric color limits. Values beyond the limits saturate; exact unclipped arrays are saved.",
        "row_order": "Exactly units_cp_acb.csv: CP followed by ACB, preserving probe/depth/unit-ID order within each region. The same row identifies the same unit in both panels. Recorded D1/D2/Unlabeled annotations are unchanged.",
        "interpretation": "Descriptive cue-aligned responses, not a causal reward effect: outcome follows the cue and the two trial groups can differ in choice side and other behavioral variables. Both single-unit and multi-unit clusters are retained as in the source.",
        "checks": {"unit_and_trial_id_alignment": True, "cached_counts_reproduce_reference_psth": True,
                   "normalization_reproduces_reference_response": True, "outcomes_disjoint": True},
    }
    return units, trials, edges, baseline, normalization, rates, relative, audit


def draw(units, edges, relative, audit):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    n = len(units)
    fig = plt.figure(figsize=(12.6, 8.6), facecolor="white")
    grid = fig.add_gridspec(1, 5, left=.025, right=.98, bottom=.21, top=.89,
                           width_ratios=[.95, .11, 4.8, .12, 4.8], wspace=.07)
    region_ax = fig.add_subplot(grid[0, 0])
    region_ax.set(xlim=(0, 1), ylim=(n, 0))
    region_ax.axis("off")
    boundaries = []
    for region, group in units.groupby("region", sort=False):
        start, stop = int(group.plot_row.min()), int(group.plot_row.max()) + 1
        center = (start + stop) / 2
        region_ax.plot([.96, .96], [start, stop], color=REGION_COLORS[region], lw=4,
                       solid_capstyle="butt")
        region_ax.plot([.76, .90], [center, center], color="#9AA7B8", lw=.7)
        region_ax.text(.70, center, f"{region} · {len(group)}", ha="right", va="center", fontsize=10)
        if stop < n:
            boundaries.append(stop)
    tag_ax = fig.add_subplot(grid[0, 1])
    strips = np.array([to_rgba(TAG_COLORS[label]) for label in units.label])[:, None, :]
    tag_ax.imshow(strips, aspect="auto", interpolation="nearest", extent=(0, 1, n, 0))
    tag_ax.axis("off")
    tag_ax.set_title("Tag", fontsize=9, pad=9)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(TAG_COLORS["Unlabeled"])
    axes = []
    for condition, column in [("reward", 2), ("omission", 4)]:
        ax = fig.add_subplot(grid[0, column])
        im = ax.imshow(relative[condition], aspect="auto", interpolation="nearest",
                       extent=(edges[0], edges[-1], n, 0), cmap=cmap,
                       vmin=audit["color_limits"][0], vmax=audit["color_limits"][1], rasterized=True)
        ax.axvline(0, color="#8D9BAE", lw=.8, ls="--")
        for boundary in boundaries:
            ax.axhline(boundary, color="white", lw=.8)
        ax.set(xlim=WINDOW, ylim=(n, 0), yticks=[], xticks=[-.5, 0, .5, 1],
               xlabel="Time from go cue (s)")
        ax.set_xticklabels(["−0.5", "0", "0.5", "1"])
        ax.tick_params(axis="x", length=3, width=.7, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        title = "Rewarded" if condition == "reward" else "Omission"
        ax.set_title(f"{title} · {audit['condition_trials'][condition]} trials", fontsize=12, pad=11)
        axes.append(ax)
    fig.suptitle(f"745502 · 2024-12-10 · {n} units · 50 ms bins", x=.5, y=.96, fontsize=14)
    left, right = axes[0].get_position().x0, axes[1].get_position().x1
    color_ax = fig.add_axes([left + .19 * (right - left), .105, .62 * (right - left), .024])
    bar = fig.colorbar(im, cax=color_ax, orientation="horizontal", extend="neither", ticks=[-3, -2, -1, 0, 1, 2, 3])
    bar.set_label("Δ firing / unit mean", fontsize=10, labelpad=5)
    bar.ax.tick_params(length=3, width=.7, labelsize=9)
    bar.outline.set_edgecolor("#D7DEE6")
    bar.outline.set_linewidth(.7)
    fig.legend(handles=[Patch(color=TAG_COLORS[label], label=label) for label in ["D1", "D2", "Unlabeled"]],
               loc="lower center", bbox_to_anchor=(.5, .015), frameon=False, ncol=3,
               fontsize=9.5, handlelength=1.6, columnspacing=2)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"reward_omission_within_neuron.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    require_compute()
    units, trials, edges, baseline, normalization, rates, relative, audit = load_responses()
    OUT.mkdir(parents=True, exist_ok=True)
    units.to_csv(OUT / "units.csv", index=False)
    trials.to_csv(OUT / "trial_groups.csv", index=False)
    np.savez_compressed(OUT / "responses.npz", global_unit_id=units.global_unit_id.to_numpy(str),
                        bin_edges_s=edges, bin_centers_s=(edges[:-1] + edges[1:]) / 2,
                        baseline_rate_hz=baseline, normalization_rate_hz=normalization,
                        reward_firing_rate_hz=rates["reward"], omission_firing_rate_hz=rates["omission"],
                        reward_relative_change=relative["reward"], omission_relative_change=relative["omission"],
                        reward_trial_id=trials.loc[trials.outcome_panel.eq("reward"), "id"].to_numpy(),
                        omission_trial_id=trials.loc[trials.outcome_panel.eq("omission"), "id"].to_numpy())
    (OUT / "provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
    draw(units, edges, relative, audit)
    print(json.dumps({"output": str(OUT), **{k: audit[k] for k in
                      ["n_units", "region_counts", "condition_trials", "excluded_no_response_trials",
                       "window_s", "bin_count", "silent_units_retained_as_gray", "checks"]}}, indent=2))


if __name__ == "__main__":
    main()
