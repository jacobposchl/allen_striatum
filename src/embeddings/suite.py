"""Export a reproducible collection of combined and per-session UMAP figures.

Every point is one unit's mean across its available exported active windows.
The combined views reuse one fit; each session receives its own independent fit.
"""
import json
import html
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.embeddings.io import load_means
from src.data.io import resolve
from src.common.plotting import COLORS, STYLE


# Distinct categorical colors, including more than tab20's twenty entries.
SESSION_COLORS = (
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#393B79", "#637939",
    "#8C6D31", "#843C39", "#7B4173", "#3182BD", "#31A354", "#756BB1",
    "#E6550D", "#969696", "#E7298A", "#66A61E", "#A6761D", "#1B9E77",
)
ANIMAL_COLORS = ("#2563EB", "#E87928", "#169A80", "#A35AB6", "#C94B63", "#7C862E")
UNLABELED = {"", "nan", "none", "na", "unlabeled", "untagged", "unknown"}


def _palette(keys, initial):
    from matplotlib.colors import hsv_to_rgb, to_hex
    colors = list(initial)
    for index in range(len(colors), len(keys)):
        colors.append(to_hex(hsv_to_rgb([(index * 0.61803398875) % 1, .65, .8])))
    return dict(zip(keys, colors))


def _labels(frame):
    labels = frame["label"].fillna("Unlabeled").astype(str).str.strip()
    return labels.mask(labels.str.lower().isin(UNLABELED), "Unlabeled")


def _slug(animal, session):
    name = f"{animal}_{session}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"Animal/session identity is not a safe filename: {name!r}")
    return name


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fit(values, cfg):
    import umap
    reducer = umap.UMAP(
        n_neighbors=min(int(cfg.neighbors), len(values) - 1),
        min_dist=float(cfg.min_dist), metric="cosine", random_state=int(cfg.seed),
        n_jobs=1,
    )
    coordinates = reducer.fit_transform(values)
    if coordinates.shape != (len(values), 2) or not np.isfinite(coordinates).all():
        raise ValueError("UMAP returned invalid coordinates")
    return coordinates


def _plot(frame, *, category, title, output, palette, legend_labels=None, seed=0):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cell_types = category == "cell_type"
    values = _labels(frame) if cell_types else frame[category].astype(str)
    keys = [key for key in palette if values.eq(key).any()]
    long_legend = category == "session_key"
    figsize = (10.4, 7.1) if long_legend else (8.4, 6.6)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        if cell_types:
            unlabeled = values.eq("Unlabeled")
            local = len(frame) < 500
            ax.scatter(frame.loc[unlabeled, "umap_1"], frame.loc[unlabeled, "umap_2"],
                       s=20 if local else 10, c=palette["Unlabeled"], alpha=.55,
                       linewidths=0, rasterized=True)
            for label in keys:
                if label == "Unlabeled":
                    continue
                selected = values.eq(label)
                ax.scatter(frame.loc[selected, "umap_1"], frame.loc[selected, "umap_2"],
                           s=43 if local else 27, c=palette[label], alpha=.95,
                           linewidths=.35, edgecolors="white", rasterized=True)
            keys = [key for key in keys if key != "Unlabeled"] + (["Unlabeled"] if unlabeled.any() else [])
        else:
            # Mix drawing order so the last animal or session does not obscure others.
            order = np.random.default_rng(seed).permutation(len(frame))
            ax.scatter(frame.umap_1.iloc[order], frame.umap_2.iloc[order],
                       c=[palette[value] for value in values.iloc[order]],
                       s=12, alpha=.8, linewidths=0, rasterized=True)
        handles = [Line2D([], [], linestyle="", marker="o", markersize=6,
                          markerfacecolor=palette[key], markeredgewidth=0,
                          label=(legend_labels or {}).get(key, key)) for key in keys]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, .5),
                  fontsize=8.5 if long_legend else 10, labelspacing=.65,
                  handletextpad=.45, borderaxespad=.8)
        ax.set_title(title, loc="left", fontsize=13, fontweight="medium", pad=15)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.margins(.08)
        fig.savefig(output, dpi=220, bbox_inches="tight", pad_inches=.18)
        plt.close(fig)


def run(cfg):
    """Create the complete figure collection, refusing to overwrite any run."""
    if not cfg.get("embeddings") or cfg.get("coordinates"):
        raise ValueError("A UMAP suite requires embeddings and no coordinates input")
    if cfg.get("predictions") or cfg.get("features"):
        raise ValueError("A UMAP suite colors original cell-type labels; omit predictions and features")
    if int(cfg.neighbors) < 2 or not 0 <= float(cfg.min_dist) <= 1:
        raise ValueError("UMAP neighbors must be >= 2 and min_dist in [0,1]")
    per_session_colorings = list(cfg.get("per_session_colorings", ["cell_type"]))
    if (not per_session_colorings or len(set(per_session_colorings)) != len(per_session_colorings)
            or set(per_session_colorings) - {"cell_type", "animal", "session"}):
        raise ValueError("per_session_colorings must contain unique choices from cell_type, animal, session")
    output = resolve(cfg.output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output is not empty: {output}; choose a new output directory")

    print("Reading exports and averaging windows once per unit...", flush=True)
    values, frame, information = load_means(cfg.embeddings, cfg.get("cohort"))
    for column in ("global_unit_id", "animal", "session", "label"):
        if column not in frame:
            raise ValueError(f"Missing required embedding metadata: {column}")
    if frame[["global_unit_id", "animal", "session"]].isna().any().any():
        raise ValueError("Missing unit, animal, or session identity")
    if frame.global_unit_id.duplicated().any():
        raise ValueError("Mean embedding identities must be unique")
    frame["animal"], frame["session"] = frame.animal.astype(str), frame.session.astype(str)
    frame["session_key"] = [_slug(a, s) for a, s in zip(frame.animal, frame.session)]
    frame["label"] = _labels(frame)
    grouped = list(frame.groupby(["animal", "session"], sort=True))
    too_small = [(animal, session, len(group)) for (animal, session), group in grouped if len(group) < 4]
    if too_small:
        raise ValueError(f"Independent session UMAPs require at least four units: {too_small}")

    import matplotlib
    matplotlib.use("Agg")
    import umap

    output.mkdir(parents=True, exist_ok=True)
    session_output = output / "per_session"
    session_output.mkdir()
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    np.savez_compressed(output / "mean_embeddings.npz", embeddings=values,
                        global_unit_id=frame.global_unit_id.to_numpy(dtype=str))
    frame.to_csv(output / "unit_metadata.csv", index=False)

    animals = sorted(frame.animal.unique())
    sessions = [(_slug(animal, session), animal, session) for (animal, session), _ in grouped]
    animal_palette = _palette(animals, ANIMAL_COLORS)
    session_palette = _palette([key for key, _, _ in sessions], SESSION_COLORS)
    labeled_types = sorted(set(frame.label) - {"Unlabeled"})
    label_palette = _palette(labeled_types, SESSION_COLORS)
    label_palette.update({key: COLORS[key] for key in labeled_types if key in COLORS})
    label_palette["Unlabeled"] = "#CBD1D9"
    concise = [f"{animal} · {session[:10]}" for _, animal, session in sessions]
    repeats = Counter(concise)
    session_names = {key: (name if repeats[name] == 1 else f"{animal} · {session}")
                     for (key, animal, session), name in zip(sessions, concise)}
    manifest = json.loads(resolve(information["export_manifest"]).read_text())
    information.update(
        created_utc=datetime.now(timezone.utc).isoformat(),
        geometry="UMAP of one mean exported embedding per unit",
        scope="All exported units in the selected cohort; no single-unit filter",
        unit_types_included=sorted(frame.decoder_label.dropna().astype(str).unique()) if "decoder_label" in frame else None,
        dimensions=int(values.shape[1]), n_units=len(frame), n_animals=len(animals), n_sessions=len(grouped),
        cell_type_labels="Original exported annotations; unlabeled units remain unlabeled",
        encoder_provenance="Recorded in export" if manifest.get("encoder_signature") else
            "Legacy export manifest lacks encoder identity; reported identity may be inferred from nearby run metadata",
        combined_geometry="The three all-session figures share identical coordinates",
        session_geometry="Each animal/session is fitted independently; positions and distances are not comparable across panels",
        umap=dict(metric="cosine", neighbors=int(cfg.neighbors), min_dist=float(cfg.min_dist),
                  seed=int(cfg.seed), n_jobs=1, version=umap.__version__,
                  neighbors_for_small_sessions="min(requested neighbors, number of units minus one)"),
        colors=dict(cell_type=label_palette, animal=animal_palette, session=session_palette),
        session_legend=session_names,
    )
    _json(output / "mean_embeddings.meta.json", information)

    print(f"Fitting combined UMAP: {len(frame)} units, {len(grouped)} sessions...", flush=True)
    coordinates = _fit(values, cfg)
    combined = frame.assign(umap_1=coordinates[:, 0], umap_2=coordinates[:, 1])
    combined.to_csv(output / "all_sessions_coordinates.csv", index=False)
    _json(output / "all_sessions_coordinates.meta.json", dict(information, fit_scope="all sessions"))
    figures = []
    for category, suffix, title, palette, legends in (
        ("cell_type", "cell_type", "All sessions · cell type", label_palette, None),
        ("animal", "animal", "All sessions · animal", animal_palette, None),
        ("session_key", "session", "All sessions · session", session_palette, session_names),
    ):
        figure = output / f"all_sessions_{suffix}.png"
        _plot(combined, category=category, title=title, output=figure, palette=palette,
              legend_labels=legends, seed=int(cfg.seed))
        figures.append(figure)

    inventory = []
    for index, ((animal, session), group) in enumerate(grouped, start=1):
        name = _slug(animal, session)
        print(f"Fitting session {index}/{len(grouped)}: {animal} {session} ({len(group)} units)...", flush=True)
        coordinates = _fit(values[group.index.to_numpy()], cfg)
        local = group.assign(umap_1=coordinates[:, 0], umap_2=coordinates[:, 1])
        local.to_csv(session_output / f"{name}_coordinates.csv", index=False)
        _json(session_output / f"{name}_coordinates.meta.json",
              dict(information, fit_scope="independent session", animal=animal, session=session,
                   n_units=len(group), n_animals=1, n_sessions=1,
                   neighbors_used=min(int(cfg.neighbors), len(group) - 1)))
        session_figures = {}
        for coloring in per_session_colorings:
            category, palette, legends = {
                "cell_type": ("cell_type", label_palette, None),
                "animal": ("animal", animal_palette, None),
                "session": ("session_key", session_palette, session_names),
            }[coloring]
            suffix = "" if coloring == "cell_type" else f"_{coloring}"
            figure = session_output / f"{name}{suffix}.png"
            _plot(local, category=category, title=f"{session_names[name]} · {coloring.replace('_', ' ')}",
                  output=figure, palette=palette, legend_labels=legends, seed=int(cfg.seed))
            figures.append(figure)
            session_figures[coloring] = str(figure.relative_to(output))
        record = dict(animal=animal, session=session, n_units=len(group),
                      n_labeled=int(group.label.ne("Unlabeled").sum()),
                      n_recordings=group.recording.nunique() if "recording" in group else None,
                      figure=session_figures.get("cell_type", next(iter(session_figures.values()))))
        record.update({f"figure_{key}": value for key, value in session_figures.items()})
        record.update({f"n_{label}": int(group.label.eq(label).sum()) for label in label_palette})
        if "decoder_label" in group:
            record.update({f"n_{kind}": int(group.decoder_label.eq(kind).sum())
                           for kind in information["unit_types_included"]})
        inventory.append(record)
    pd.DataFrame(inventory).to_csv(output / "session_inventory.csv", index=False)
    archive = output / "umap_figures.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for figure in figures:
            bundle.write(figure, figure.relative_to(output))
    information.update(n_figures=len(figures), figures=[str(path.relative_to(output)) for path in figures],
                       archive=archive.name, status="complete")
    _json(output / "suite.meta.json", information)
    gallery = "".join(
        f'<figure><figcaption>{html.escape(str(path.relative_to(output)))}</figcaption>'
        f'<a href="{html.escape(str(path.relative_to(output)))}"><img style="max-width:100%" '
        f'src="{html.escape(str(path.relative_to(output)))}"></a></figure>' for path in figures)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Session UMAPs</title>'
        '<body style="font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px">'
        '<h1>Session UMAPs</h1><p>One point per recorded unit; original cell-type labels. '
        'Each session has its own fit. Colorings within a fit share coordinates.</p>'
        '<p><a href="umap_figures.zip">Download all PNGs</a></p>' + gallery + '</body>')
    print(f"Saved {len(figures)} PNGs and {archive}", flush=True)
    return information
