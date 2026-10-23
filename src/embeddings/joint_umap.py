"""Joint encoder-space UMAP with explicit dataset markers and ChAT examples."""
import html
import json
import re
import zipfile

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.embeddings.io import load_means, export_manifest, provenance, read_table
from src.data.io import resolve
from src.common.plotting import COLORS, STYLE
from src.embeddings.suite import _fit, _labels, _palette, _plot, SESSION_COLORS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def cached_means(source, output, dataset):
    manifest_path, manifest = export_manifest(source)
    identity = provenance(manifest_path, manifest)
    cache = output / f"{dataset}_means.npz"
    metadata = output / f"{dataset}_units.csv"
    audit = output / f"{dataset}_source.json"
    if cache.exists() and metadata.exists() and audit.exists():
        saved = json.loads(audit.read_text())
        if any(identity.get(key) != saved.get(key) for key in
               ("export_manifest_sha256", "encoder_signature")):
            raise ValueError("Cached means do not match the requested export")
        frame = read_table(metadata)
        with np.load(cache, allow_pickle=False) as data:
            values = data["embeddings"]
            if not np.array_equal(data["global_unit_id"], frame.global_unit_id.astype(str)):
                raise ValueError("Cached mean identities do not match metadata")
        return values, frame, saved
    print(f"Averaging {dataset} exported windows...", flush=True)
    values, frame, identity = load_means(source)
    np.savez_compressed(cache, embeddings=values,
                        global_unit_id=frame.global_unit_id.to_numpy(dtype=str))
    frame.to_csv(metadata, index=False)
    write_json(audit, identity)
    print(f"Loaded {dataset}: {len(frame)} units, {values.shape[1]} dimensions", flush=True)
    return values, frame, identity


def combined_plot(frame, category, palette, output, legend_labels):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sessions = category == "session_key"
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(19, 10) if sessions else (13, 8))
        grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1] if sessions else [3, 1], wspace=.02)
        ax, legend_ax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
        for dataset, marker, size, alpha, edge in (
            ("passive", "o", 11, .65, "none"),
            ("active", "D", 29, .95, "#111827"),
        ):
            subset = frame[frame.dataset.eq(dataset)].sample(frac=1, random_state=0)
            ax.scatter(subset.umap_1, subset.umap_2,
                       c=[palette[value] for value in subset[category]],
                       s=size, marker=marker, alpha=alpha, edgecolors=edge,
                       linewidths=.45 if dataset == "active" else 0, rasterized=True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.set_title("Colored by session" if sessions else "Colored by animal",
                     loc="left", fontsize=16, pad=15)
        handles = [Line2D([], [], linestyle="", marker="o", markersize=6,
                          markerfacecolor="#64748B", markeredgecolor="none", label="Passive"),
                   Line2D([], [], linestyle="", marker="D", markersize=6,
                          markerfacecolor="#64748B", markeredgecolor="#111827", label="Active")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.01), ncol=2)
        keys = list(palette)
        category_handles = []
        for key in keys:
            is_active = frame.loc[frame[category].eq(key), "dataset"].eq("active").any()
            category_handles.append(Line2D([], [], linestyle="", marker="D" if is_active else "o",
                markersize=5, markerfacecolor=palette[key],
                markeredgecolor="#111827" if is_active else "none",
                label=legend_labels[key]))
        legend_ax.legend(handles=category_handles, loc="center left", ncol=3 if sessions else 1,
                         fontsize=7.8 if sessions else 9, labelspacing=.8,
                         handletextpad=.35, columnspacing=.9)
        legend_ax.set_axis_off()
        fig.subplots_adjust(left=.025, right=.99, top=.92, bottom=.045)
        fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=.15)
        plt.close(fig)


def run(cfg):
    import matplotlib
    matplotlib.use("Agg")
    output = resolve(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    saved_config = output / "config.yaml"
    if saved_config.exists() and OmegaConf.to_container(OmegaConf.load(saved_config), resolve=True) != OmegaConf.to_container(cfg, resolve=True):
        raise ValueError("Output contains a different configuration; choose a fresh output")
    OmegaConf.save(cfg, saved_config, resolve=True)
    matrices, frames, sources = [], [], {}
    for dataset in ("passive", "active"):
        values, frame, info = cached_means(cfg[dataset], output, dataset)
        if frame.global_unit_id.duplicated().any() or not np.isfinite(values).all():
            raise ValueError("Invalid unit means")
        frame["dataset"] = dataset
        frame["animal"] = frame.animal.astype(str)
        frame["session"] = frame.session.astype(str)
        frame["label"] = _labels(frame)
        frame["session_key"] = dataset + ":" + frame.animal + ":" + frame.session
        matrices.append(values)
        frames.append(frame)
        sources[dataset] = info
    signature = sources["passive"].get("encoder_signature")
    if not signature or signature != sources["active"].get("encoder_signature"):
        raise ValueError("Joint UMAP requires the exact same encoder checkpoint")
    values, frame = np.concatenate(matrices), pd.concat(frames, ignore_index=True)
    if frame[["dataset", "global_unit_id"]].duplicated().any():
        raise ValueError("Duplicate dataset/unit identities")
    print(f"Fitting combined UMAP: {len(frame)} units", flush=True)
    coordinate_path = output / "combined_coordinates.csv"
    if coordinate_path.exists():
        old = read_table(coordinate_path)
        if not old[["dataset", "global_unit_id"]].equals(frame[["dataset", "global_unit_id"]]):
            raise ValueError("Existing coordinates have different unit identities")
        coordinates = old[["umap_1", "umap_2"]].to_numpy()
    else:
        coordinates = _fit(values, cfg)
    if not np.isfinite(coordinates).all():
        raise ValueError("Nonfinite coordinates")
    frame["umap_1"], frame["umap_2"] = coordinates.T
    frame.to_csv(coordinate_path, index=False)
    np.savez_compressed(output / "mean_embeddings.npz", embeddings=values,
                        global_unit_id=frame.global_unit_id.to_numpy(dtype=str),
                        dataset=frame.dataset.to_numpy(dtype=str))
    sessions = frame.groupby(["dataset", "animal", "session", "session_key"], sort=True).agg(
        n_units=("global_unit_id", "size"), n_chat=("label", lambda x: int(x.eq("ChAT").sum()))).reset_index()
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    sessions["display_label"] = [f"{animal} · {date_pattern.search(session).group()}"
                                 for animal, session in zip(sessions.animal, sessions.session)]
    repeated_dates = sessions.display_label.duplicated(keep=False)
    for index in sessions.index[repeated_dates]:
        clock = re.search(r"_(\d{2})-(\d{2})-(\d{2})(?:$|_)", sessions.at[index, "session"])
        if clock is None:
            raise ValueError("Cannot disambiguate repeated session dates")
        sessions.at[index, "display_label"] += f" {clock.group(1)}:{clock.group(2)}:{clock.group(3)}"
    if sessions.display_label.duplicated().any():
        raise ValueError("Session legend labels must be unique")
    sessions.to_csv(output / "sessions.csv", index=False)
    animals = sorted(frame.animal.unique())
    animal_palette = _palette(animals, SESSION_COLORS)
    session_palette = _palette(sessions.session_key.tolist(), SESSION_COLORS)
    write_json(output / "palettes.json", {"animal": animal_palette, "session": session_palette})
    combined_plot(frame, "animal", animal_palette, output / "combined_by_animal.png",
                  {a: a for a in animals})
    combined_plot(frame, "session_key", session_palette, output / "combined_by_session.png",
                  dict(zip(sessions.session_key, sessions.display_label)))
    # Choose by label count alone, before inspecting UMAP geometry; prefer distinct animals.
    candidates = sessions[sessions.dataset.eq("passive") & sessions.n_chat.gt(0)].sort_values(
        ["n_chat", "animal", "session"], ascending=[False, True, True])
    preferred = pd.concat([candidates.drop_duplicates("animal"), candidates]).drop_duplicates("session_key")
    selected = preferred.head(int(cfg.passive_examples))
    if len(selected) != int(cfg.passive_examples):
        raise ValueError("Not enough passive sessions with ChAT labels")
    selected.to_csv(output / "selected_passive_sessions.csv", index=False)
    figures = ["combined_by_animal.png", "combined_by_session.png"]
    for row in selected.itertuples(index=False):
        keep = frame.session_key.eq(row.session_key).to_numpy()
        local = frame.loc[keep].copy()
        print(f"Fitting passive example {row.display_label}: {row.n_units} units, {row.n_chat} ChAT", flush=True)
        date = re.search(r"\d{4}-\d{2}-\d{2}", row.session).group()
        stem = f"passive_{row.animal}_{date}_cell_types"
        local_path = output / f"{stem}_coordinates.csv"
        if local_path.exists():
            previous = read_table(local_path)
            if not np.array_equal(previous.global_unit_id, local.global_unit_id):
                raise ValueError("Per-session coordinate identities differ")
            local_coordinates = previous[["umap_1", "umap_2"]].to_numpy()
        else:
            local_coordinates = _fit(values[keep], cfg)
        if not np.isfinite(local_coordinates).all():
            raise ValueError("Invalid per-session coordinates")
        local[["umap_1", "umap_2"]] = local_coordinates
        local.to_csv(local_path, index=False)
        counts = local.label.value_counts()
        palette = _palette(sorted(counts.index), SESSION_COLORS)
        palette.update({label: COLORS[label] for label in counts.index if label in COLORS})
        _plot(local, category="cell_type", title=row.display_label,
              output=output / f"{stem}.png", palette=palette,
              legend_labels={label: f"{label} ({count})" for label, count in counts.items()})
        figures.append(f"{stem}.png")
    metadata = {"sources": sources, "encoder": "fine-tuned epoch 10", "n_units": len(frame),
                "counts": frame.groupby("dataset").size().to_dict(), "n_dimensions": values.shape[1],
                "pooling": "Mean of each unit's available 10-second window embeddings",
                "umap": {"metric": "cosine", "neighbors": int(cfg.neighbors), "min_dist": float(cfg.min_dist), "seed": int(cfg.seed)},
                "selection": "Most ChAT-labeled units, preferring distinct animals; no UMAP-based selection",
                "inclusion": "All exported units, including single units and multiunits; no additional QC filter",
                "geometry": "One shared fit for combined views; independently fitted per-session examples",
                "interpretation": "Descriptive visualization, not held-out cell-type validation",
                "figures": figures}
    write_json(output / "method.json", metadata)
    write_json(output / "checks.json", {"same_encoder": True, "finite_means_and_coordinates": True,
                                        "unique_dataset_unit_identities": True,
                                        "shared_combined_coordinates": True})
    with zipfile.ZipFile(output / "umap_figures.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in figures:
            archive.write(output / name, name)
    content = "".join(f'<figure><a href="{html.escape(name)}"><img src="{html.escape(name)}" style="max-width:100%"></a></figure>' for name in figures)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Passive and active UMAPs</title><body style="font:16px system-ui;max-width:1500px;margin:35px auto;padding:20px"><h1>Passive and active embeddings</h1><p>Fine-tuned encoder · one point per recorded unit. Active units: outlined diamonds. Per-session maps are independently fitted.</p><p><a href="umap_figures.zip">Download PNGs</a></p>' + content + '</body>')
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata
