"""Compare linear identity decoding on exactly matched, neuron-level embeddings.

The neuron holdout asks whether embeddings separate the identities represented
in training. The additional animal probe holds out entire sessions, asking
whether animal information transfers to an unseen session of a known animal.
The NuCLR encoders themselves are frozen and saw these recordings in pretraining.
"""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import socket
import time
import warnings

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.embeddings.io import IDENTITY, read_table
from src.data.io import resolve


def _json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(name):
    result = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if not result:
        raise ValueError("Every source needs a nonempty model name")
    return result


def _load_source(source, expected_dimensions):
    directory = resolve(source.directory)
    paths = {"embeddings": directory / "mean_embeddings.npz",
             "metadata": directory / "unit_metadata.csv",
             "provenance": directory / "mean_embeddings.meta.json"}
    with np.load(paths["embeddings"], allow_pickle=False) as arrays:
        values = arrays["embeddings"].astype(np.float64)
        ids = arrays["global_unit_id"].astype(str)
    if values.ndim != 2 or values.shape != (len(ids), expected_dimensions):
        raise ValueError(f"Invalid embedding dimensions in {directory}: {values.shape}")
    if not np.isfinite(values).all() or len(set(ids)) != len(ids):
        raise ValueError(f"Nonfinite vectors or duplicate unit IDs in {directory}")
    frame = read_table(paths["metadata"])
    if not set(IDENTITY).issubset(frame.columns):
        raise ValueError(f"Missing identity columns in {paths['metadata']}")
    if frame[list(IDENTITY)].isna().any().any() or frame.global_unit_id.duplicated().any():
        raise ValueError(f"Missing or duplicate identities in {paths['metadata']}")
    if set(frame.global_unit_id) != set(ids):
        raise ValueError(f"Embedding and metadata IDs differ in {directory}")
    frame = frame.set_index("global_unit_id").loc[ids].reset_index()
    # Construct sessions from their identities; do not trust an auxiliary column.
    frame["session_key"] = frame.animal + "/" + frame.session
    info = json.loads(paths["provenance"].read_text())
    if info.get("n_units", len(ids)) != len(ids):
        raise ValueError(f"Cached provenance has the wrong neuron count in {directory}")
    if info.get("dimensions", expected_dimensions) != expected_dimensions:
        raise ValueError(f"Cached provenance has the wrong embedding dimension in {directory}")
    return values, frame, {"name": str(source.name), "directory": str(directory),
                           "files": {key: {"path": str(path), "sha256": _hash(path)}
                                     for key, path in paths.items()},
                           "cached_provenance": info, "n_available_units": len(ids)}


def _matched_sources(sources, expected_dimensions, expected_units):
    loaded = [_load_source(source, expected_dimensions) for source in sources]
    common = sorted(set.intersection(*(set(frame.global_unit_id) for _, frame, _ in loaded)))
    if not common or (expected_units is not None and len(common) != expected_units):
        raise ValueError(f"Expected {expected_units} matched units; found {len(common)}")
    matrices, metadata, information = [], None, []
    for values, frame, info in loaded:
        indices = pd.Series(np.arange(len(frame)), index=frame.global_unit_id).loc[common].to_numpy()
        matched = frame.iloc[indices].reset_index(drop=True)
        if metadata is not None:
            for key in IDENTITY:
                if not matched[key].equals(metadata[key]):
                    raise ValueError(f"Matched-unit {key} differs across models")
            for key in ("decoder_label", "source_unit_row", "unit_id"):
                if key in matched and key in metadata:
                    if not matched[key].astype(str).equals(metadata[key].astype(str)):
                        raise ValueError(f"Matched-unit {key} differs across models")
        else:
            metadata = matched
        matrices.append(values[indices])
        info["n_common_units"] = len(common)
        info["n_excluded_noncommon_units"] = len(frame) - len(common)
        information.append(info)
    return matrices, metadata, information


def _scores(truth, prediction, classes):
    counts = confusion_matrix(truth, prediction, labels=classes)
    support = counts.sum(axis=1)
    recall = np.divide(counts.diagonal(), support, out=np.zeros(len(classes), dtype=float), where=support > 0)
    # For leave-session-out folds, only one animal is present in the test set.
    # Its fold-level balanced accuracy is its recall. Pooled scores use all animals.
    balanced_accuracy = float(recall[support > 0].mean())
    return {"accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": balanced_accuracy}, counts, support, recall


def _evaluate(values, metadata, target, splits, *, model, task, cfg, directory):
    truth = metadata[target].to_numpy(dtype=str)
    classes = np.unique(truth)
    if len(classes) < 2:
        raise ValueError(f"Target {target} has fewer than two classes")
    predictions = np.empty(len(truth), dtype=object)
    probabilities = np.full((len(truth), len(classes)), np.nan)
    fold_ids = np.full(len(truth), -1, dtype=int)
    rows = []
    started = time.monotonic()
    print(f"{model}: {task}, {len(splits)} folds, {len(truth)} units", flush=True)
    for fold, (train, test) in enumerate(splits):
        if set(truth[train]) != set(classes):
            raise ValueError(f"{model}/{task} fold {fold} lacks a training class")
        if (fold_ids[test] >= 0).any() or np.intersect1d(train, test).size:
            raise ValueError("Invalid overlapping cross-validation folds")
        scaler = StandardScaler()
        train_values = scaler.fit_transform(values[train])
        test_values = scaler.transform(values[test])
        # L2 is the default across supported sklearn versions; the explicit
        # penalty argument is deprecated in sklearn 1.8.
        classifier = LogisticRegression(C=float(cfg.C), class_weight="balanced",
                                        solver="lbfgs", max_iter=int(cfg.max_iter),
                                        random_state=int(cfg.seed))
        fit_started = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                classifier.fit(train_values, truth[train])
            except ConvergenceWarning as error:
                raise RuntimeError(f"Linear probe failed to converge: {model}/{task}, fold {fold}") from error
        if not np.array_equal(classifier.classes_, classes):
            raise ValueError("Unexpected classifier class ordering")
        probability = classifier.predict_proba(test_values)
        predicted = classifier.classes_[probability.argmax(axis=1)]
        if not np.isfinite(probability).all():
            raise ValueError(f"Nonfinite probe predictions in {model}/{task}, fold {fold}")
        predictions[test], probabilities[test], fold_ids[test] = predicted, probability, fold
        scores, _, support, _ = _scores(truth[test], predicted, classes)
        rows.append({"model": model, "task": task, "fold": fold, "n_train": len(train),
                     "n_test": len(test), "n_train_classes": len(classes),
                     "n_test_classes": int((support > 0).sum()),
                     "heldout_sessions": "|".join(sorted(metadata.iloc[test].session_key.unique())),
                     "iterations": int(classifier.n_iter_.max()), "converged": True,
                     "fit_seconds": time.monotonic() - fit_started,
                     "log_loss": float(log_loss(truth[test], probability, labels=classes)), **scores})
    if (fold_ids < 0).any() or not np.isfinite(probabilities).all():
        raise ValueError("Cross-validation did not predict every unit exactly once")
    scores, counts, support, recalls = _scores(truth, predictions, classes)
    summary = {"model": model, "task": task, "target": target, "n_units": len(truth),
               "n_classes": len(classes), "n_sessions": int(metadata.session_key.nunique()),
               "n_animals": int(metadata.animal.nunique()), "dimensions": values.shape[1],
               "n_folds": len(splits), "n_converged_folds": len(rows),
               "uniform_chance": 1 / len(classes), "C": float(cfg.C),
               "seconds": time.monotonic() - started,
               "log_loss": float(log_loss(truth, probabilities, labels=classes)), **scores}
    directory.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(directory / "fold_metrics.csv", index=False)
    result = metadata[list(IDENTITY)].copy()
    result["fold"], result["true_label"], result["predicted_label"] = fold_ids, truth, predictions
    result["correct"] = truth == predictions
    for column, label in enumerate(classes):
        result[f"p_{label}"] = probabilities[:, column]
    result.to_csv(directory / "heldout_predictions.csv", index=False)
    pd.DataFrame({"class": classes, "n_units": support, "recall": recalls}).to_csv(
        directory / "per_class_recall.csv", index=False)
    pd.DataFrame(counts, index=pd.Index(classes, name="true_label"), columns=classes).to_csv(
        directory / "confusion_counts.csv")
    pd.DataFrame(counts / support[:, None], index=pd.Index(classes, name="true_label"), columns=classes).to_csv(
        directory / "confusion_row_normalized.csv")
    print(f"{model}: {task}: balanced accuracy {scores['balanced_accuracy']:.4f}, accuracy {scores['accuracy']:.4f}", flush=True)
    return summary, rows


def _plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from src.common.plotting import STYLE
    tasks = [("session", "Session\nHeld-out units"), ("animal", "Animal\nHeld-out units"),
             ("animal_leave_session_out", "Animal\nHeld-out sessions")]
    colors = ["#2563EB", "#D97706", "#059669", "#A855F7"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.8), sharey=True)
        for axis, (task, title) in zip(axes, tasks):
            rows = summary.loc[summary.task == task]
            axis.bar(np.arange(len(rows)), rows.balanced_accuracy, color=colors[:len(rows)], width=.58)
            axis.axhline(float(rows.uniform_chance.iloc[0]), color="#94A3B8", linewidth=1, linestyle="--")
            for x, score in enumerate(rows.balanced_accuracy):
                axis.text(x, score + .02, f"{score:.1%}", ha="center", va="bottom", fontsize=10,
                          zorder=3, bbox=dict(facecolor="white", edgecolor="none", pad=1))
            axis.set_xticks(np.arange(len(rows)), rows.model)
            axis.set_title(title, fontsize=11, pad=12)
            axis.set_ylim(0, 1.08)
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=.12)
        axes[0].set_ylabel("Balanced accuracy")
        fig.text(.985, .015, "Dashed: chance", ha="right", fontsize=9, color="#64748B")
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def run(cfg):
    """Run fixed linear probes on a matched cohort; intended for a CPU allocation."""
    output = resolve(cfg.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite identity-probe results: {output}")
    sources = cfg.sources
    names = [str(source.name) for source in sources]
    if len(names) < 2 or len({_slug(name) for name in names}) != len(names):
        raise ValueError("Provide at least two uniquely named embedding sources")
    if int(cfg.n_splits) < 2 or int(cfg.cpu_threads) < 1 or float(cfg.C) <= 0 or int(cfg.max_iter) < 1:
        raise ValueError("Invalid linear-probe configuration")
    expected = None if cfg.expected_common_units is None else int(cfg.expected_common_units)
    matrices, metadata, information = _matched_sources(sources, int(cfg.expected_dimensions), expected)
    group_counts = metadata.session_key.value_counts()
    if group_counts.min() < int(cfg.n_splits):
        raise ValueError("Every session needs at least n_splits neurons")
    splitter = StratifiedKFold(n_splits=int(cfg.n_splits), shuffle=True, random_state=int(cfg.seed))
    neuron_splits = list(splitter.split(np.zeros(len(metadata)), metadata.session_key))
    session_splits = []
    for session in sorted(metadata.session_key.unique()):
        selected = metadata.session_key.to_numpy() == session
        session_splits.append((np.flatnonzero(~selected), np.flatnonzero(selected)))
    # Validate that every animal remains represented before fitting either model.
    animals = set(metadata.animal)
    if any(set(metadata.iloc[train].animal) != animals for train, _ in session_splits):
        raise ValueError("Animal leave-session-out needs at least two sessions per animal")
    output.mkdir(parents=True)
    OmegaConf.save(cfg, output / "config.yaml")
    cohort = metadata.copy()
    cohort["neuron_fold"] = -1
    cohort["heldout_session_fold"] = -1
    for fold, (_, test) in enumerate(neuron_splits):
        cohort.loc[test, "neuron_fold"] = fold
    for fold, (_, test) in enumerate(session_splits):
        cohort.loc[test, "heldout_session_fold"] = fold
    cohort.to_csv(output / "cohort.csv", index=False)
    cohort.groupby(["animal", "session", "recording"], sort=True).size().rename("n_units").to_csv(
        output / "cohort_counts.csv")
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "python": platform.python_version(), "cpu_threads": int(cfg.cpu_threads),
            "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")},
            "sources": information, "n_common_units": len(metadata),
            "method": {"input": "Original mean 48D embeddings, one vector per recorded unit; no UMAP coordinates",
                       "cohort": "Exact intersection of source unit IDs; no single-unit filter",
                       "probe": "L2 logistic regression, balanced class weights, fixed C, no tuning",
                       "scaling": "StandardScaler fitted separately on each training fold only",
                       "neuron_split": "Identical session-stratified folds for every model and both identity targets",
                       "animal_session_split": "One entire animal/session held out per fold; each animal remains in training",
                       "metrics": "Pooled out-of-fold predictions: accuracy and mean per-class recall (balanced accuracy)",
                       "chance": "Uniform random-label baseline, descriptive only",
                       "scope": "Held out from the probe only; all recordings were available to NuCLR pretraining",
                       "limitation": "Recorded units may not represent distinct biological neurons across sessions"}}
    _json(output / "provenance.json", info)
    summaries, folds = [], []
    with threadpool_limits(limits=int(cfg.cpu_threads)):
        for model, values in zip(names, matrices):
            tasks = [("session", "session_key", neuron_splits), ("animal", "animal", neuron_splits),
                     ("animal_leave_session_out", "animal", session_splits)]
            for task, target, splits in tasks:
                summary, rows = _evaluate(values, metadata, target, splits, model=model, task=task, cfg=cfg,
                                          directory=output / _slug(model) / task)
                summaries.append(summary)
                folds.extend(rows)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    _json(output / "summary.json", {"results": summaries, "n_common_units": len(metadata),
                                    "metrics_use_pooled_predictions": True})
    _plot(summary_frame, output / "linear_probe_comparison.png")
    print(f"Identity-probe results: {output}", flush=True)
    return output
