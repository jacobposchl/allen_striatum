"""Plot one recorded December 10 trial using small behavioral arrays only."""
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import require_compute
from src.data.recording import Recording, resolve_recording
SOURCE = "745502/2024-12-10_12-06-14/experiment1_recording1"
OUT = ROOT / "results/experiments/dataset_inventory/trial_timing_745502_2024-12-10"


def load_timing():
    recording = Recording(resolve_recording(SOURCE))
    group = recording.trials
    fields = ["id", "start_time", "stop_time", "delay_start_time", "goCue_start_time",
              "reward_outcome_time", "animal_response", "rewarded_historyL", "rewarded_historyR",
              "auto_waterL", "auto_waterR", "reward_delay", "reward_size_left", "reward_size_right"]
    trials = {key: group[key].to_numpy() for key in fields}
    responded = np.isin(trials["animal_response"], [0, 1])
    rewarded = responded & ((trials["rewarded_historyL"] == 1) | (trials["rewarded_historyR"] == 1))
    delivery = np.full(len(responded), np.nan)
    for side, code in (("left", 0), ("right", 1)):
        events = recording.events(f"{side}_reward")
        if not len(events):
            continue
        indices = np.flatnonzero(rewarded & (trials["animal_response"] == code))
        choice = trials["reward_outcome_time"][indices]
        expected = choice + trials["reward_delay"][indices]
        positions = np.searchsorted(events, choice)
        actual = events[np.minimum(positions, len(events) - 1)]
        matched = ((positions < len(events)) & (np.abs(actual - expected) <= .02)
                   & (actual < trials["stop_time"][indices]))
        delivery[indices[matched]] = actual[matched]

    # The cue duration is programmed, not a duration inferred from an offset event.
    metadata = recording.sidecars.get("session.json", {})
    notes = " ".join(epoch.get("notes", "") or "" for epoch in metadata.get("stimulus_epochs", []))
    if "duration of go cue is 100ms" not in notes:
        raise ValueError("Confirm the session's programmed go-cue duration before plotting.")
    cue_duration = .100
    metrics = {
        "total_trial_s": trials["stop_time"] - trials["start_time"],
        "intertrial_interval_s": trials["delay_start_time"] - trials["start_time"],
        "precue_delay_s": trials["goCue_start_time"] - trials["delay_start_time"],
        "go_cue_programmed_s": np.full(len(responded), cue_duration),
        "choice_latency_s": np.where(responded, trials["reward_outcome_time"] - trials["goCue_start_time"], np.nan),
        "choice_to_reward_s": delivery - trials["reward_outcome_time"],
        "consumption_pause_s": trials["stop_time"] - np.where(rewarded, delivery, trials["reward_outcome_time"]),
    }
    summary = {}
    for name, values in metrics.items():
        finite = values[np.isfinite(values)]
        if np.any(finite < 0):
            raise ValueError(f"Negative duration in {name}")
        summary[name] = dict(zip(["min", "q25", "median", "q75", "max"],
                                map(float, np.quantile(finite, [0, .25, .5, .75, 1]))))
        summary[name]["n"] = len(finite)

    # Choose an actual rewarded trial near the session medians, not a synthetic
    # concatenation of medians. A 50-ms floor stabilizes the nearly fixed delay.
    reference = np.column_stack([metrics[k] for k in
                                 ["intertrial_interval_s", "precue_delay_s", "choice_latency_s"]])
    q25, q75 = np.nanquantile(reference, [.25, .75], axis=0)
    distance = np.nansum(np.abs(reference - np.nanmedian(reference, axis=0)) / np.maximum(q75 - q25, .05), axis=1)
    eligible = (rewarded & np.isfinite(delivery) & np.isfinite(reference).all(axis=1)
                & (trials["auto_waterL"] == 0) & (trials["auto_waterR"] == 0))
    if not eligible.any():
        raise ValueError("No rewarded trial with complete verified event timing.")
    index = np.flatnonzero(eligible)[np.argmin(distance[eligible])]
    origin = trials["start_time"][index]
    events = {"trial_start": 0., "delay_start": float(trials["delay_start_time"][index] - origin),
              "cue_start": float(trials["goCue_start_time"][index] - origin),
              "choice": float(trials["reward_outcome_time"][index] - origin),
              "reward": float(delivery[index] - origin), "trial_end": float(trials["stop_time"][index] - origin)}
    side = "left" if trials["animal_response"][index] == 0 else "right"
    example = {"trial_id": int(trials["id"][index]), "choice_side": side,
               "reward_uL": float(trials[f"reward_size_{side}"][index]), "events_s": events,
               "durations_s": {name: float(values[index]) for name, values in metrics.items()}}
    params = {key: np.unique(group[key][:]).tolist() for key in
              ["ITI_min", "ITI_max", "ITI_beta", "delay_min", "delay_max", "delay_beta",
               "response_duration", "reward_delay", "reward_consumption_duration"]}
    audit = {"source": str(SOURCE), "animal": "745502", "session": "2024-12-10_12-06-14",
             "n_trials": len(responded), "n_responded": int(responded.sum()),
             "n_rewarded": int(rewarded.sum()), "n_verified_deliveries": int(np.isfinite(delivery).sum()),
             "parameters": params, "duration_summary_s": summary, "example": example,
             "extended_precue_delays_over_1_1_s": int((metrics["precue_delay_s"] > 1.1).sum()),
             "method": "All stored trials from this recording. Actual timestamp differences; reward delivery matched to a same-side event within 20 ms of expected delivery and before trial end. Response statistics exclude no-response trials. Consumption/pause begins at verified delivery when rewarded, otherwise registered choice/timeout.",
             "cue": "100 ms programmed duration verified in the embedded session metadata; overlaps the choice period. Shown as a stripe inside the same timeline, not an added sequential phase.",
             "example_selection": "Rewarded trial nearest the session median ITI, pre-cue delay, and choice latency, using IQR scaling with a 50-ms minimum scale; automatic-water trials excluded."}
    recording.close()
    return trials, metrics, audit


def plot(example):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    e, d = example["events_s"], example["durations_s"]
    fig, ax = plt.subplots(figsize=(12.8, 3.25))
    fig.subplots_adjust(left=.055, right=.975, bottom=.22, top=.79)
    fig.suptitle(f"745502  ·  2024-12-10  ·  Rewarded trial {example['trial_id']}",
                 x=.5, y=.94, fontsize=15, weight="bold", color="#283748")
    colors = {"iti": "#CBD4DE", "delay": "#8573A7", "choice": "#498DC2",
              "reward": "#268B70", "pause": "#E5BAAC", "cue": "#DDB64B"}
    stages = [(0., e["delay_start"], "iti"), (e["delay_start"], e["cue_start"], "delay"),
              (e["cue_start"], e["choice"], "choice"), (e["choice"], e["reward"], "reward"),
              (e["reward"], e["trial_end"], "pause")]
    for start, end, key in stages:
        ax.add_patch(Rectangle((start, -.25), end - start, .5, facecolor=colors[key], edgecolor="none", zorder=2))
    # Cue and response begin together. The upper stripe encodes that overlap.
    ax.add_patch(Rectangle((e["cue_start"], .11), d["go_cue_programmed_s"], .14,
                           facecolor=colors["cue"], edgecolor="none", zorder=3))
    ax.vlines([e["delay_start"], e["cue_start"]], -.25, .25, colors="white", lw=.7, zorder=4)
    ax.vlines(e["reward"], -.25, .25, colors=colors["reward"], lw=1.2, zorder=4)
    for x, label, duration, color in [
        (e["delay_start"] / 2, "Inter-trial interval", f"{d['intertrial_interval_s']:.2f} s", "#2B3B4F"),
        ((e["delay_start"] + e["cue_start"]) / 2, "Pre-cue delay", f"{d['precue_delay_s'] * 1000:.0f} ms", "white"),
        ((e["reward"] + e["trial_end"]) / 2, "Consumption / pause", f"{d['consumption_pause_s']:.2f} s", "#483D3A")]:
        ax.text(x, .065, label, ha="center", va="center", color=color, fontsize=11.2, weight="bold", zorder=5)
        ax.text(x, -.105, duration, ha="center", va="center", color=color, fontsize=10.3, zorder=5)

    connector = dict(arrowstyle="-", lw=.8, shrinkA=4, shrinkB=2,
                     connectionstyle="angle,angleA=0,angleB=90,rad=3")
    ax.annotate("Go cue · 100 ms", xy=(e["cue_start"] + .05, .25),
                xytext=(e["cue_start"] - .45, .72), ha="right", va="center", fontsize=10.4,
                color="#92701C", arrowprops={**connector, "color": "#B99A42"})
    ax.annotate(f"Reward · {example['reward_uL']:g} µL\n{d['choice_to_reward_s'] * 1000:.1f} ms after choice",
                xy=(e["reward"], .25), xytext=(e["reward"] + .50, .73),
                ha="left", va="center", fontsize=10.4, linespacing=1.5,
                color=colors["reward"], arrowprops={**connector, "color": colors["reward"]})
    ax.annotate(f"Choice · {d['choice_latency_s'] * 1000:.0f} ms",
                xy=((e["cue_start"] + e["choice"]) / 2, -.25),
                xytext=(e["choice"] - .45, -.59), ha="right", va="center", fontsize=10.4,
                color="#3475A7", arrowprops={**connector, "color": "#498DC2"})
    ax.set_ylim(-.88, 1.00)
    ax.set_xlim(0, np.ceil(e["trial_end"]))
    ax.set_xticks(np.arange(0, np.ceil(e["trial_end"]) + 1))
    ax.set_yticks([])
    ax.set_xlabel("Time from trial start (s)", fontsize=10.5, labelpad=10, color="#536273")
    ax.tick_params(axis="x", colors="#657080", labelsize=10, length=3, width=.7, pad=5)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#CCD4DD")
    ax.spines["bottom"].set_linewidth(.7)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"trial_timing.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    require_compute()
    trials, metrics, audit = load_timing()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "trial_durations.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial_id", *metrics])
        writer.writeheader()
        for i, trial_id in enumerate(trials["id"]):
            writer.writerow({"trial_id": int(trial_id), **{k: float(v[i]) if np.isfinite(v[i]) else "" for k, v in metrics.items()}})
    (OUT / "timing_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    plot(audit["example"])
    print(json.dumps({"output": str(OUT), "trial_id": audit["example"]["trial_id"],
                      "trials": audit["n_trials"], "extended_precue_delays": audit["extended_precue_delays_over_1_1_s"]}))


if __name__ == "__main__":
    main()
