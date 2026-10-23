"""Recording timelines and signal availability from Brainsets HDF5 files."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import require_compute
from src.data.recording import Recording, discover_recordings
DEFAULT_ROOT = ROOT / "data/processed/active"
COLORS = {"745502": "#365C99", "745505": "#97522C", "753618": "#765491", "754313": "#2F7374"}
DOMAIN_COLORS = {"baseline": "#6093CA", "behavior": "#35A88F", "optotagging": "#CB692B", "unknown": "#C8CCD1"}
SIGNALS = {
    "Trials": ("table", "trials"),
    "Left licks": ("event", "left_lick"),
    "Right licks": ("event", "right_lick"),
    "Left rewards": ("event", "left_reward"),
    "Right rewards": ("event", "right_reward"),
    "Speed": ("signal", "speed"),
    "Pupil diameter": ("signal", "pupil_diameter"),
    "Pupil position": ("signal", "pupil_position"),
    "Body keypoints": ("signal", "body_keypoints"),
    "Face keypoints": ("signal", "face_keypoints"),
    "Tongue keypoints": ("signal", "tongue_keypoints"),
}


def nonempty_series(group):
    """Count available leaves in a canonical single- or multi-series signal."""
    if not bool(group.attrs.get("available", True)):
        return False
    status = group.attrs.get("status", "present")
    if isinstance(status, bytes):
        status = status.decode()
    if status in {"missing", "invalid", "missing_timestamps"}:
        return False
    if "timestamps" in group:
        return len(group["timestamps"]) > 0
    return any(nonempty_series(child) for child in group.values() if hasattr(child, "keys"))


def read_inventory(root):
    """Read canonical metadata and presence, without loading spike arrays."""
    rows, coverage = [], {}
    for item in discover_recordings(root):
        with Recording(item["source"]) as recording:
            info = recording.info
            row = {key: str(info[key]) for key in ("animal", "session", "recording")}
            clock_available = all(info.get(name) is not None for name in
                                  ("origin_s", "session_offset_s", "duration_s"))
            row.update(source=str(recording.path), clock_available=clock_available,
                       start_min=None, end_min=None)
            if clock_available:
                origin = float(info["origin_s"])
                offset = float(info["session_offset_s"])
                row.update(start_min=offset / 60,
                           end_min=(offset + float(info["duration_s"])) / 60)
            for domain in ("baseline", "behavior", "optotagging"):
                row[f"{domain}_available"] = f"intervals/{domain}" in recording.file
                row[f"{domain}_intervals_min"] = []
                if not clock_available or not row[f"{domain}_available"]:
                    continue
                spans = recording.intervals(domain)
                row[f"{domain}_intervals_min"] = [
                    [float((low - origin + offset) / 60), float((high - origin + offset) / 60)]
                    for low, high in spans]
            key = tuple(row[k] for k in ("animal", "session", "recording"))
            coverage[key] = {}
            for label, (kind, name) in SIGNALS.items():
                if kind == "table":
                    present = "trials/start" in recording.file and len(recording.file["trials/start"]) > 0
                elif kind == "event":
                    path = f"events/{name}"
                    present = path in recording.file and nonempty_series(recording.file[path])
                else:
                    path = f"signals/{name}"
                    present = path in recording.file and nonempty_series(recording.file[path])
                coverage[key][label] = bool(present)
            rows.append(row)
    if not rows:
        raise ValueError(f"No Brainsets recordings found in {root}")
    return sorted(rows, key=lambda row: (row["animal"], row["session"], row["recording"])), coverage


def intervals(row):
    """Draw stored domain spans exactly; unclassified time stays unavailable."""
    if not row["clock_available"]:
        return []
    lo, hi = row["start_min"], row["end_min"]
    named = [(domain, max(lo, start), min(hi, end))
             for domain in ("baseline", "behavior")
             for start, end in row[f"{domain}_intervals_min"]
             if min(hi, end) > max(lo, start)]
    unknown, cursor = [], lo
    for start, end in sorted((start, end) for _, start, end in named):
        if start > cursor:
            unknown.append(("unknown", cursor, start))
        cursor = max(cursor, end)
    if cursor < hi:
        unknown.append(("unknown", cursor, hi))
    opto = [("optotagging", max(lo, start), min(hi, end))
            for start, end in row["optotagging_intervals_min"]
            if min(hi, end) > max(lo, start)]
    return unknown + named + opto


def plot(rows, coverage, output):
    if not rows:
        raise ValueError("No recording rows are available to plot")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "hatch.linewidth": .7})
    repeat = Counter((r["animal"], r["session"]) for r in rows)
    seen = Counter()
    ys, labels, suffixes = [], [], []
    y = 0
    previous = None
    separators = []
    for row in rows:
        if previous is not None and row["animal"] != previous:
            separators.append(y - .3)
            y += .55
        ys.append(y)
        key = row["animal"], row["session"]
        seen[key] += 1
        suffixes.append(f"({seen[key]})" if repeat[key] > 1 else "")
        labels.append(f"{row['animal']}   {row['session'][:10]}")
        previous = row["animal"]
        y += 1

    fig = plt.figure(figsize=(14.6, max(7.4, len(rows) * .24 + 2.0)))
    ax = fig.add_axes([.195, .105, .435, .71])
    signals_ax = fig.add_axes([.675, .105, .305, .71], sharey=ax)
    fig.text(.5, .954, "Session data coverage", ha="center", fontsize=17, weight="bold", color="#233040")
    ax.set_title("Recording time", loc="left", fontsize=11, weight="bold", color="#354254", pad=44)
    signals_ax.set_title("Stored signals", loc="left", fontsize=11, weight="bold", color="#354254", pad=44)

    bar_height = .59
    for y, row in zip(ys, rows):
        if not row["clock_available"]:
            ax.text(0, y, "Timing unavailable", va="center", fontsize=8, color="#657080")
        for domain, start, end in intervals(row):
            is_opto = domain == "optotagging"
            ax.broken_barh([(start, end - start)], (y - bar_height / 2, bar_height),
                          facecolor="none" if is_opto else DOMAIN_COLORS[domain],
                          edgecolor=DOMAIN_COLORS[domain] if is_opto else "none",
                          linewidth=.85 if is_opto else 0,
                          hatch="////" if is_opto else None, zorder=4 if is_opto else 3)
    for divider in separators:
        ax.axhline(divider, color="#E5E9ED", lw=.65, xmin=-.25, clip_on=False, zorder=1)
        signals_ax.axhline(divider, color="#E5E9ED", lw=.65, zorder=1)
    ax.set_yticks(ys, labels, fontsize=9.4)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(COLORS.get(row["animal"], "#526070"))
        tick.set_fontweight("normal")
    ax.tick_params(axis="y", length=0, pad=25)
    for y, suffix, row in zip(ys, suffixes, rows):
        if suffix:
            ax.text(-.006, y, suffix, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    color=COLORS.get(row["animal"], "#526070"), fontsize=8.2, clip_on=False)
    ax.set_ylim(ys[-1] + .7, -.7)
    latest = max((r["end_min"] for r in rows if r["clock_available"]), default=0)
    ax.set_xlim(0, max(100, 20 * ((latest + 19.999) // 20)))
    ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 20))
    ax.tick_params(axis="x", length=0, pad=9, labelsize=9.5, colors="#657080")
    ax.set_xlabel("Time from session start (min)", fontsize=9.5, color="#526070", labelpad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#DCE2E8")
    ax.spines["bottom"].set_linewidth(.8)
    ax.xaxis.grid(True, color="#E9EDF1", linewidth=.7)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=DOMAIN_COLORS["baseline"], label="Baseline"),
               Patch(facecolor=DOMAIN_COLORS["behavior"], label="Behavior"),
               Patch(facecolor="none", edgecolor=DOMAIN_COLORS["optotagging"], hatch="////", label="Optotagging")]
    if any(d == "unknown" for r in rows for d, _, _ in intervals(r)):
        handles.append(Patch(facecolor=DOMAIN_COLORS["unknown"], label="Epoch unavailable"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.008), ncol=len(handles),
              frameon=False, fontsize=9.2, handlelength=1.6, handleheight=.8,
              columnspacing=1.4, borderaxespad=0, labelcolor="#4A5564")

    # The two panels share the same row positions, including recording repeats
    # and gaps between animals. Each row uses that recording's stored signals.
    signals_ax.set_xlim(-.5, len(SIGNALS) - .5)
    signals_ax.tick_params(axis="y", left=False, labelleft=False)
    signals_ax.set_xticks([])
    signals_ax.spines[:].set_visible(False)
    for y, row in zip(ys, rows):
        flags = coverage[(row["animal"], row["session"], row["recording"])]
        for j, name in enumerate(SIGNALS):
            present = flags[name]
            signals_ax.add_patch(Rectangle((j - .43, y - .34), .86, .68,
                                            facecolor="#2878A8" if present else "#F1F3F5",
                                            edgecolor="none", zorder=2))
    groups = [(0, 0, "Trials", []), (1, 2, "Licks", ["L", "R"]),
              (3, 4, "Rewards", ["L", "R"]), (5, 5, "Speed", []),
              (6, 7, "Pupil", ["Size", "Pos."]), (8, 10, "Keypoints", ["Body", "Face", "Tongue"])]
    for first, last, title, sublabels in groups:
        signals_ax.text((first + last) / 2, 1.058 if sublabels else 1.030, title,
                        transform=signals_ax.get_xaxis_transform(), ha="center", va="center",
                        fontsize=9, color="#4A5564", clip_on=False)
        if sublabels:
            signals_ax.plot([first - .33, last + .33], [1.036, 1.036],
                            transform=signals_ax.get_xaxis_transform(), color="#DDE4EA",
                            lw=.7, clip_on=False)
            for j, label in zip(range(first, last + 1), sublabels):
                signals_ax.text(j, 1.012, label, transform=signals_ax.get_xaxis_transform(),
                                ha="center", va="center", fontsize=8.1, color="#6B7787", clip_on=False)
        if last < len(SIGNALS) - 1:
            signals_ax.axvline(last + .5, color="#EFF2F5", lw=.6, zorder=0)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / f"session_recording_coverage.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/experiments/dataset_inventory/active_coverage")
    args = parser.parse_args()
    rows, coverage = read_inventory(args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "recording_intervals.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value
                          for key, value in row.items()} for row in rows)
    shown_sessions = {(row["animal"], row["session"]) for row in rows}
    shown_recordings = {(row["animal"], row["session"], row["recording"]) for row in rows}
    session_coverage = {}
    with (args.output / "recording_signal_coverage.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["animal", "session", "recording", "shown_in_figure", *SIGNALS])
        writer.writeheader()
        for key, flags in sorted(coverage.items()):
            animal, session, recording = key
            writer.writerow({"animal": animal, "session": session, "recording": recording,
                             "shown_in_figure": key in shown_recordings, **flags})
            aggregate = session_coverage.setdefault((animal, session), {name: False for name in SIGNALS})
            for name in SIGNALS:
                aggregate[name] |= flags[name]
    with (args.output / "session_signal_coverage.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["animal", "session", "shown_in_figure", *SIGNALS])
        writer.writeheader()
        for (animal, session), flags in sorted(session_coverage.items()):
            writer.writerow({"animal": animal, "session": session,
                             "shown_in_figure": (animal, session) in shown_sessions, **flags})
    provenance = {
        "data_root": str(args.data_root.resolve()),
        "scope": "All canonical recordings in the selected HDF5 directory.",
        "animals": len({r['animal'] for r in rows}),
        "sessions": len({(r['animal'], r['session']) for r in rows}),
        "recordings": len(rows),
        "timing": "Canonical source intervals mapped to session time using each recording's stored origin and offset.",
        "epochs": "Intervals clipped to recording bounds; optotagging overlays behavior.",
        "baseline": "Only explicitly stored baseline intervals. Unclassified time and gaps remain unavailable.",
        "signal_coverage_meaning": "Blue = a nonempty canonical signal is available; gray = unavailable. Presence does not imply full-duration coverage or valid values throughout.",
        "signal_fields": SIGNALS,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot(rows, coverage, args.output)
    print(json.dumps({"output": str(args.output), "animals": provenance["animals"],
                      "sessions": provenance["sessions"], "recordings": len(rows)}))


if __name__ == "__main__":
    main()
