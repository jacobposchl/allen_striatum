"""Render a behavioral overview using Brainsets behavioral signals and recording intervals."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import require_compute
from src.data.recording import Recording, resolve_recording
SOURCE = "745502/2024-12-10_12-06-14/experiment1_recording1"
OUT_BASE = ROOT / "results/experiments/dataset_inventory"
LEFT = "#3078B5"
RIGHT = "#D27A22"
REWARD_FRACTION = "#2B8A70"
REWARD_WINDOW = 10
INK = "#283748"
MUTED = "#657080"
GRID = "#E9EDF1"


def load_behavior(window):
    recording = Recording(resolve_recording(SOURCE))
    info = recording.info
    origin = float(info["origin_s"])
    intervals = recording.intervals("behavior")
    if len(intervals) != 1:
        raise ValueError("Expected one behavior interval for this recording.")
    behavior_start, behavior_end = map(float, intervals[0] - origin)
    if recording.opto_start is None:
        raise ValueError("The selected recording has no established optotagging onset.")
    opto_start = float(recording.opto_start - origin)
    start_s, end_s = ((behavior_start, min(behavior_end, opto_start))
                      if window == "behavior" else (0., 1800.))
    if not 0 <= start_s < end_s <= opto_start:
        raise ValueError("The plotted interval must precede optotagging.")
    group = recording.trials
    fields = ["id", "start_time", "stop_time", "reward_probabilityL", "reward_probabilityR",
              "animal_response", "rewarded_historyL", "rewarded_historyR"]
    trials = {name: group[name].to_numpy() for name in fields}
    for name in ("start_time", "stop_time"):
        trials[name] = trials[name] - origin
    if not np.all(np.diff(trials["start_time"]) > 0):
        raise ValueError("Trial starts must be strictly increasing.")
    selected = (trials["start_time"] < end_s) & (trials["stop_time"] > start_s)

    # A causal reward fraction uses trial outcomes, not counts of water events.
    # Include no-response trials as unrewarded and require ten completed trials.
    if not np.all(np.diff(trials["stop_time"]) > 0):
        raise ValueError("Trial completion times must be strictly increasing.")
    for side in ("L", "R"):
        if not np.isin(trials[f"rewarded_history{side}"], [0, 1]).all():
            raise ValueError("Reward outcomes must be binary and nonmissing.")
    trials["rewarded"] = (trials["rewarded_historyL"] == 1) | (trials["rewarded_historyR"] == 1)
    responded = np.isin(trials["animal_response"], [0, 1])
    if np.any(trials["rewarded"] & ~responded):
        raise ValueError("A no-response trial unexpectedly contains a reward.")
    cumulative = np.r_[0, np.cumsum(trials["rewarded"], dtype=int)]
    trials["reward_count_last10"] = np.full(len(selected), np.nan)
    trials["reward_count_last10"][REWARD_WINDOW - 1:] = cumulative[REWARD_WINDOW:] - cumulative[:-REWARD_WINDOW]
    trials["reward_fraction_last10"] = trials["reward_count_last10"] / REWARD_WINDOW
    completed = selected & (trials["stop_time"] <= end_s)
    fraction_in_window = trials["reward_fraction_last10"][completed]
    fraction_in_window = fraction_in_window[np.isfinite(fraction_in_window)]

    events = {}
    for side in ("left", "right"):
        for kind in ("lick", "reward_delivery"):
            key = f"{side}_{kind}"
            times = recording.events(f"{side}_{'lick' if kind == 'lick' else 'reward'}") - origin
            if not np.isfinite(times).all() or (np.diff(times) < 0).any():
                raise ValueError(f"Invalid event timestamps: {key}")
            events[key] = times[(times >= start_s) & (times < end_s)]

    speed_time, speed = recording.signal("speed")
    attrs = recording.signal_info("speed")
    speed = np.asarray(speed).reshape(-1)
    if len(speed) != len(speed_time):
        raise ValueError("Expected one running-speed value per timestamp.")
    speed_time = speed_time - origin
    in_window = (speed_time >= start_s) & (speed_time < end_s)
    speed_time, speed = speed_time[in_window], np.abs(speed[in_window])
    recording.close()
    if not len(speed_time):
        raise ValueError("No running-speed samples in the selected interval.")
    if not np.isfinite(speed_time).all() or (np.diff(speed_time) <= 0).any():
        raise ValueError("Invalid running-speed timestamps.")
    valid = np.isfinite(speed)
    # Align bins to recording-clock seconds, clipping only the two edge bins.
    bin_edges = np.r_[start_s, np.arange(np.floor(start_s) + 1, end_s), end_s]
    counts, _ = np.histogram(speed_time[valid], bins=bin_edges)
    totals, _ = np.histogram(speed_time[valid], bins=bin_edges, weights=speed[valid])
    mean_speed = np.divide(totals, counts, out=np.full(len(counts), np.nan), where=counts > 0)

    audit = {
        "animal": "745502", "session": "2024-12-10_12-06-14",
        "recording": "experiment1_recording1", "source": str(recording.path),
        "clock_metadata": str(recording.path), "recording_origin_source_s": origin,
        "clock_method": info.get("clock_method", "canonical_source_clock"), "window": window,
        "window_recording_s": [start_s, end_s],
        "behavior_epoch_recording_s": [behavior_start, behavior_end],
        "first_optotagging_pulse_recording_s": opto_start,
        "window_note": "Behavior mode uses the Brainsets behavior interval clipped at the first optotagging pulse. First30min mode uses recording seconds [0, 1800).",
        "clock_note": "All timestamps use the HDF5 recording origin. No neural arrays were read.",
        "trial_count_overlapping_window": int(selected.sum()),
        "trial_ids_overlapping_window": trials["id"][selected].astype(int).tolist(),
        "probabilities_full_recording": {}, "probabilities_in_window": {},
        "probability_note": "Programmed trial-table reward probabilities, held from each trial start until the next trial start, ending at the earlier of the window end or last trial end. These are baiting probabilities; accumulated available rewards depend on past choices and outcomes. No probability is drawn before the first recorded trial.",
        "event_counts_in_window": {k: len(v) for k, v in events.items()},
        "event_note": "Every stored lick and actual water-delivery timestamp in the half-open plotting window is drawn once. Reward events are not inferred from choice times.",
        "reward_fraction": {
            "window_trials": REWARD_WINDOW,
            "method": "Trailing count of rewarded trials divided by 10, including the just-completed trial and the preceding nine trials. A trial is rewarded if either stored rewarded_history flag is true. No-response trials count as unrewarded. The first nine trials are missing, not partial windows. Values update at trial completion and are held until the next completed trial; no future outcomes or additional smoothing are used.",
            "completed_trials_in_window": int(completed.sum()),
            "rewarded_trials_in_window": int((completed & trials["rewarded"]).sum()),
            "unrewarded_choice_trials_in_window": int((completed & responded & ~trials["rewarded"]).sum()),
            "no_response_trials_in_window": int((completed & ~responded).sum()),
            "n_full_windows": len(fraction_in_window),
            "min": float(fraction_in_window.min()),
            "max": float(fraction_in_window.max()),
        },
        "running_speed": {
            "source": "signals/speed",
            "source_unit": attrs["unit"], "calibration": "Applied by the canonical recording reader",
            "method": "Mean absolute calibrated velocity in nonoverlapping one-second recording-time bins, with edge bins restricted to the window; bins with no finite samples stay missing. No speed clipping or further smoothing.",
            "n_samples_in_window": len(speed), "n_finite_samples": int(valid.sum()),
            "sample_period_s": float(np.median(np.diff(speed_time))),
            "first_sample_recording_s": float(speed_time[0]),
            "last_sample_recording_s": float(speed_time[-1]),
            "one_second_mean_min_cm_s": float(np.nanmin(mean_speed)),
            "one_second_mean_max_cm_s": float(np.nanmax(mean_speed)),
        },
    }
    for side in ("L", "R"):
        p = trials[f"reward_probability{side}"]
        if not np.array_equal(np.unique(p), np.array([.1, .4, .7])):
            raise ValueError("Update probability ticks to match the stored values.")
        audit["probabilities_full_recording"][side] = np.unique(p).tolist()
        audit["probabilities_in_window"][side] = np.unique(p[selected]).tolist()
    return trials, selected, events, mean_speed, counts, bin_edges, audit


def draw(trials, selected, events, mean_speed, bin_edges, audit, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(13.2, 6.6),
                             gridspec_kw={"height_ratios": [1.2, 1.9, 1.25, 1.25]})
    fig.subplots_adjust(left=.135, right=.975, bottom=.105, top=.88, hspace=.33)
    title = "Behavior before optotagging" if audit["window"] == "behavior" else "First 30 minutes"
    end_s = audit["window_recording_s"][1]
    fig.suptitle(f"745502  ·  2024-12-10  ·  {title}", x=.5, y=.965,
                 fontsize=15, weight="bold", color=INK)

    for ax in axes:
        ax.set_xticks(np.arange(0, end_s / 60 + .001, 5 if end_s <= 1800 else 10))
        # Keep the established recording clock. The brief baseline is blank.
        ax.set_xlim(0, end_s / 60)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, lw=.7)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=7, labelsize=9.5)

    reward_ax = axes[0]
    completed = selected & (trials["stop_time"] <= end_s)
    visible_fraction = completed & np.isfinite(trials["reward_fraction_last10"])
    reward_time = trials["stop_time"][visible_fraction] / 60
    fraction = trials["reward_fraction_last10"][visible_fraction]
    reward_ax.step(np.r_[reward_time, end_s / 60], np.r_[fraction, fraction[-1]],
                   where="post", color=REWARD_FRACTION, lw=1.5, solid_capstyle="butt")
    reward_ax.set_ylim(-.04, 1.04)
    reward_ax.set_yticks([0, .5, 1], ["0", "0.5", "1"])
    reward_ax.grid(axis="y", color=GRID, lw=.7)
    reward_ax.set_ylabel("Reward fraction\n(last 10 trials)", rotation=0, ha="right", va="center",
                         labelpad=22, color=INK, linespacing=1.5)

    probability = axes[1]
    probability.legend(handles=[Line2D([], [], color=LEFT, lw=2.1, label="Left"),
                                Line2D([], [], color=RIGHT, lw=2.0, ls=(0, (4, 2)), label="Right")],
                       loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2,
                       frameon=False, fontsize=9.5, handlelength=2.6, columnspacing=2,
                       labelcolor=INK, borderaxespad=0)
    t = trials["start_time"][selected] / 60
    # Append the right edge to show the last trial's probability to its end.
    # A dashed right trace keeps equal probabilities visible in both colors.
    for side, color, style in [("L", LEFT, "-"), ("R", RIGHT, (0, (4, 2)))]:
        p = trials[f"reward_probability{side}"][selected]
        right_edge = min(end_s, float(trials["stop_time"][selected][-1])) / 60
        probability.step(np.r_[t, right_edge], np.r_[p, p[-1]], where="post",
                         color=color, lw=2.0, ls=style, solid_capstyle="butt")
    probability.set_ylim(.04, .76)
    probability.set_yticks([.1, .4, .7], ["0.1", "0.4", "0.7"])
    probability.grid(axis="y", color=GRID, lw=.7)
    probability.set_ylabel("Reward\nprobability", rotation=0, ha="right", va="center",
                           labelpad=22, color=INK, linespacing=1.5)

    speed_ax = axes[2]
    centers_min = (bin_edges[:-1] + bin_edges[1:]) / 120
    speed_ax.fill_between(centers_min, 0, mean_speed, color="#CBD5DF", alpha=.65, lw=0)
    speed_ax.plot(centers_min, mean_speed, color="#52687B", lw=.9)
    speed_upper = max(7.5, float(np.nanmax(mean_speed)) * 1.1)
    speed_ax.set_ylim(0, speed_upper)
    tick_step = 3 * max(1, int(np.ceil(speed_upper / 9)))
    speed_ax.set_yticks(np.arange(0, speed_upper, tick_step))
    speed_ax.grid(axis="y", color=GRID, lw=.7)
    speed_ax.set_ylabel("Speed\n(cm/s)", rotation=0, ha="right", va="center",
                        labelpad=24, color=INK, linespacing=1.5)

    # One compact raster keeps each side's licks directly beside its rewards.
    # Licks occupy the outer lanes; reward deliveries occupy the inner lanes.
    raster = axes[3]
    rows = [("right_lick", "Right licks", "#202020", 3.5),
            ("right_reward_delivery", "Right rewards", "#E43D3D", 2.5),
            ("left_reward_delivery", "Left rewards", "#E43D3D", 1.5),
            ("left_lick", "Left licks", "#202020", .5)]
    raster.set_ylim(0, 4)
    raster.set_yticks([row[3] for row in rows], [row[1] for row in rows])
    raster.tick_params(axis="y", pad=10, labelsize=10)
    raster.axhline(2, color="#E2E7EC", lw=.7, zorder=1)
    for tick, (key, label, color, center) in zip(raster.get_yticklabels(), rows):
        raster.vlines(events[key] / 60, center - .47, center + .47, color=color,
                      lw=.75 if key.endswith("lick") else 1.05, zorder=2)
        tick.set_color(color)
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].spines["bottom"].set_color("#CCD4DD")
    axes[-1].spines["bottom"].set_linewidth(.7)
    axes[-1].tick_params(axis="x", length=3, width=.7, pad=9)
    axes[-1].set_xlabel("Recording time (min)", labelpad=10, fontsize=10.5)

    for extension in ("png", "svg", "pdf"):
        fig.savefig(out / f"session_behavior_{audit['window']}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def save_csv(out, name, fields, rows):
    with (out / name).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=["behavior", "first30min"], default="behavior")
    window = parser.parse_args().window
    trials, selected, events, mean_speed, counts, bin_edges, audit = load_behavior(window)
    out = OUT_BASE / f"behavior_745502_2024-12-10_{window}"
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out, "trial_probabilities.csv",
             ["trial_id", "trial_start_recording_s", "trial_stop_recording_s", "left_probability", "right_probability"],
             zip(*(trials[key][selected] for key in ["id", "start_time", "stop_time", "reward_probabilityL", "reward_probabilityR"])))
    save_csv(out, "events.csv", ["event", "recording_time_s"],
             ((key, float(t)) for key, times in events.items() for t in times))
    completed = selected & (trials["stop_time"] <= audit["window_recording_s"][1])
    save_csv(out, "reward_fraction_10trials.csv",
             ["trial_id", "trial_end_recording_s", "animal_response", "rewarded",
              "reward_count_last10", "reward_fraction_last10"],
             ((int(trials["id"][i]), float(trials["stop_time"][i]), int(trials["animal_response"][i]),
               int(trials["rewarded"][i]),
               int(trials["reward_count_last10"][i]) if np.isfinite(trials["reward_count_last10"][i]) else "",
               float(trials["reward_fraction_last10"][i]) if np.isfinite(trials["reward_fraction_last10"][i]) else "")
              for i in np.flatnonzero(completed)))
    save_csv(out, "running_speed_1s.csv", ["bin_start_recording_s", "bin_end_recording_s", "mean_speed_cm_s", "n_finite_samples"],
             ((float(bin_edges[i]), float(bin_edges[i + 1]), float(v) if np.isfinite(v) else "", int(counts[i]))
              for i, v in enumerate(mean_speed)))
    (out / "provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
    draw(trials, selected, events, mean_speed, bin_edges, audit, out)
    print(json.dumps({"output": str(out), "n_trials": audit["trial_count_overlapping_window"],
                      "window_recording_s": audit["window_recording_s"],
                      "first_optotagging_pulse_recording_s": audit["first_optotagging_pulse_recording_s"],
                      "probabilities": audit["probabilities_in_window"],
                      "event_counts": audit["event_counts_in_window"],
                      "reward_fraction": audit["reward_fraction"],
                      "maximum_mean_speed_cm_s": audit["running_speed"]["one_second_mean_max_cm_s"]}, indent=2))


if __name__ == "__main__":
    main()
