"""Plot cached coordinates or compute one UMAP from mean neuron embeddings."""
import json

import numpy as np
import pandas as pd

from src.embeddings.io import attach_table, filter_cohort, load_means, read_table
from src.data.io import resolve
from src.common.plotting import COLORS, STYLE


def _join(frame, path, information):
    path = resolve(path)
    meta_path = path.with_suffix(".meta.json")
    status = "identity_only"
    if meta_path.exists():
        other = json.loads(meta_path.read_text())
        for key in ("data_fingerprint", "encoder_signature"):
            if information.get(key) and other.get(key) and information[key] != other[key]:
                raise ValueError(f"Plot input mismatch: {key}")
        status = "identity_and_available_provenance"
    information.setdefault("joins", []).append(dict(path=str(path), validation=status))
    return attach_table(frame, read_table(path))


def run(cfg):
    if cfg.get("suite", False):
        from src.embeddings.suite import run as run_suite
        return run_suite(cfg)
    if bool(cfg.get("coordinates")) == bool(cfg.get("embeddings")):
        raise ValueError("Specify exactly one of coordinates or embeddings")
    output = resolve(cfg.output)
    color = str(cfg.color)
    if "/" in color or "\\" in color:
        raise ValueError("color must be a column name, not a path")
    if cfg.get("coordinates"):
        path = resolve(cfg.coordinates)
        frame = filter_cohort(read_table(path), cfg.get("cohort"))
        information = dict(coordinates=str(path), geometry="reused saved coordinates")
        metadata = path.with_suffix(".meta.json")
        if metadata.exists():
            information.update(json.loads(metadata.read_text()))
        if not {"umap_1", "umap_2"}.issubset(frame.columns):
            raise ValueError("Coordinate CSV must contain umap_1 and umap_2")
    else:
        if int(cfg.neighbors) < 2 or not 0 <= float(cfg.min_dist) <= 1:
            raise ValueError("UMAP neighbors must be >= 2 and min_dist in [0,1]")
        values, frame, information = load_means(cfg.embeddings, cfg.get("cohort"))
        if len(frame) < 4:
            raise ValueError("UMAP requires at least four selected neurons")
        import umap
        reducer = umap.UMAP(n_neighbors=min(int(cfg.neighbors), len(frame)-1), min_dist=float(cfg.min_dist),
                            metric="cosine", random_state=int(cfg.seed), n_jobs=1)
        coordinates = reducer.fit_transform(values)
        frame["umap_1"], frame["umap_2"] = coordinates[:, 0], coordinates[:, 1]
        information.update(geometry="UMAP of mean neuron embeddings", seed=int(cfg.seed),
                           neighbors=int(cfg.neighbors), min_dist=float(cfg.min_dist))
    if not len(frame):
        raise ValueError("No neurons match the requested cohort")
    if cfg.get("predictions"):
        frame = _join(frame, cfg.predictions, information)
    if cfg.get("features"):
        frame = _join(frame, cfg.features, information)
    if color not in frame:
        raise ValueError(f"Unknown color column {color!r}; available columns: {', '.join(frame.columns)}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        x, y = frame.umap_1, frame.umap_2
        categorical = color in {"label", "predicted_label", "argmax_label", "animal", "session", "recording", "decoder_label"}
        numeric = pd.to_numeric(frame[color], errors="coerce")
        if not categorical and numeric.notna().any():
            valid = np.isfinite(numeric)
            ax.scatter(x[~valid], y[~valid], c="#DCE1E7", s=10, linewidths=0)
            signed = numeric[valid].min() < 0 < numeric[valid].max()
            kwargs = {"vmin": 0, "vmax": 1} if color.startswith("p_") else {}
            if signed:
                bound = np.max(np.abs(numeric[valid]))
                kwargs.update(vmin=-bound, vmax=bound)
            points = ax.scatter(x[valid], y[valid], c=numeric[valid], cmap="RdBu_r" if signed else "viridis",
                                s=12, linewidths=0, rasterized=True, **kwargs)
            fig.colorbar(points, ax=ax, shrink=.7, label=color.replace("_", " "))
        else:
            labels = frame[color].fillna("Unlabeled").astype(str)
            palette = plt.get_cmap("tab20")
            for index, label in enumerate(sorted(labels.unique())):
                selected = labels.eq(label)
                ax.scatter(x[selected], y[selected], c=[COLORS.get(label, palette(index % 20))],
                           s=12, linewidths=0, label=label, rasterized=True)
            ax.legend(loc="center left", bbox_to_anchor=(1, .5), markerscale=1.5)
        ax.set(xticks=[], yticks=[], title=color.replace("_", " "))
        ax.set_aspect("equal", adjustable="datalim")
        for spine in ax.spines.values():
            spine.set_visible(False)
        output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output / f"embedding_{color}.png", dpi=180)
        plt.close(fig)
    # Always write derived files separately from the input coordinate CSV.
    table_path = output / "coordinates.csv"
    if cfg.get("coordinates") and table_path.resolve() == resolve(cfg.coordinates).resolve():
        table_path = output / "coordinates_view.csv"
    frame.to_csv(table_path, index=False)
    information.update(n_neurons=len(frame), color=color)
    table_path.with_suffix(".meta.json").write_text(json.dumps(information, indent=2) + "\n")
    print(output / f"embedding_{color}.png")
    return information
