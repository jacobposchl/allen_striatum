"""Session-held-out readouts and probabilistic, putative cell-type assignments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.embeddings.io import load_means
from src.data.io import resolve


def _matrix(values, units, kind, names):
    if kind == "embedding":
        return np.asarray(values, dtype=float)
    if kind != "baseline":
        raise ValueError("features must be embedding or baseline")
    columns = []
    for name in names:
        column = pd.to_numeric(units[name], errors="coerce") if name in units else pd.Series(np.nan, index=units.index)
        values = column.to_numpy(dtype=float)
        if name == "firing_rate_hz":
            values = np.where(values >= 0, np.log1p(np.maximum(values, 0)), np.nan)
        columns.append(values)
    if not columns:
        raise ValueError("At least one baseline feature is required")
    return np.column_stack(columns)


def _fit(x, y, cfg):
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(x, float)
    keep = np.isfinite(x).any(axis=0)
    if not keep.any():
        raise ValueError("No finite input features available to fit the classifier")
    usable = np.isfinite(x[:, keep]).any(axis=1)
    y = np.asarray(y)
    if set(np.unique(y[usable])) != set(np.unique(y)):
        raise ValueError("A training class has no neurons with usable features")
    y = y[usable]
    finite = np.where(np.isfinite(x[usable][:, keep]), x[usable][:, keep], np.nan)
    median = np.nanmedian(finite, axis=0)
    x = np.where(np.isfinite(finite), finite, median)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.
    classifier = LogisticRegression(C=float(cfg.C), class_weight="balanced", max_iter=int(cfg.max_iter), random_state=int(cfg.seed))
    classifier.fit((x - mean) / scale, y)
    if np.max(classifier.n_iter_) >= int(cfg.max_iter):
        raise ValueError("Cell-type classifier did not converge; increase max_iter or adjust regularization")
    return dict(keep=keep, median=median, mean=mean, scale=scale,
                coef=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_.astype(str))


def _predict(model, x):
    x = np.asarray(x, float)
    if x.shape[1] != len(model["keep"]):
        raise ValueError("Classifier feature dimension does not match the input")
    x = x[:, model["keep"]]
    usable = np.isfinite(x).any(axis=1)
    x = np.where(np.isfinite(x), x, model["median"])
    logits = ((x - model["mean"]) / model["scale"]) @ model["coef"].T + model["intercept"]
    if len(model["classes"]) == 2 and logits.shape[1] == 1:
        logits = np.column_stack([np.zeros(len(x)), logits[:, 0]])
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[~usable] = np.nan
    return probabilities


def _feature_availability(x, names, retained=None):
    finite = np.isfinite(np.asarray(x, float))
    available = finite.any(axis=0)
    retained = available if retained is None else np.asarray(retained, dtype=bool)
    return dict(
        configured_features=list(names),
        available_features=[str(name) for name, present in zip(names, available) if present],
        unavailable_features=[str(name) for name, present in zip(names, available) if not present],
        retained_features=[str(name) for name, present in zip(names, retained) if present],
        missing_fraction={str(name): float(1 - finite[:, index].mean()) if len(finite) else None
                          for index, name in enumerate(names)},
        n_units_without_retained_features=int((~finite[:, retained].any(axis=1)).sum()),
    )


def _scored(frame, classes):
    if not len(frame):
        return frame.copy()
    usable = np.isfinite(frame[[f"p_{name}" for name in classes]].to_numpy(float)).all(axis=1)
    return frame.loc[usable].copy()


def _metrics(frame, classes):
    from sklearn.metrics import balanced_accuracy_score, log_loss

    frame = _scored(frame, classes)
    result = dict(n_scored=len(frame), accuracy=None, balanced_accuracy=None,
                  log_loss=None, recall_by_class={})
    if len(frame):
        result.update(accuracy=float(np.mean(frame.target == frame.prediction)),
                      balanced_accuracy=float(balanced_accuracy_score(frame.target, frame.prediction)),
                      log_loss=float(log_loss(frame.target, frame[[f"p_{name}" for name in classes]], labels=classes)),
                      recall_by_class={str(label): float(np.mean(frame.loc[frame.target.eq(label), "prediction"] == label))
                                       for label in classes if frame.target.eq(label).any()})
    return result


def _shared_comparison(first, second, classes):
    """Compare readouts only where both actually produced held-out predictions."""
    first, second = _scored(first, classes), _scored(second, classes)
    shared = set(first.global_unit_id) & set(second.global_unit_id) if len(first) and len(second) else set()
    selected = [frame.loc[frame.global_unit_id.isin(shared)].copy() if len(frame) else frame.copy()
                for frame in (first, second)]
    if shared:
        targets = [frame.set_index("global_unit_id").target.sort_index() for frame in selected]
        if not targets[0].equals(targets[1]):
            raise ValueError("Held-out readouts disagree on target labels for shared neuron identities")
    summary = dict(
        cohort="Identical neurons successfully scored by both held-out readouts",
        n_shared_units=len(shared),
        n_shared_sessions=int(selected[0][["animal", "session"]].drop_duplicates().shape[0]) if shared else 0,
        embedding=_metrics(selected[0], classes), baseline=_metrics(selected[1], classes),
    )
    return summary, pd.concat(selected, ignore_index=True)


def _evaluate(x, y, units, cfg, feature_kind):
    if cfg.holdout not in {"session", "animal"}:
        raise ValueError("holdout must be session or animal")
    groups = units.animal.astype(str)
    if cfg.holdout == "session":
        groups = groups + "/" + units.session.astype(str)
    classes = np.unique(y)
    predictions, folds = [], []
    for group in sorted(groups.unique()):
        test = groups.eq(group).to_numpy()
        train = ~test
        summary = dict(features=feature_kind, held_out=group, n_train=int(train.sum()), n_test=int(test.sum()))
        if feature_kind == "baseline":
            availability = _feature_availability(x[train], list(cfg.baseline_features))
            summary.update(retained_features=json.dumps(availability["retained_features"]),
                           feature_missing_fraction_train=json.dumps(availability["missing_fraction"]))
        if set(np.unique(y[train])) != set(classes):
            folds.append(dict(**summary, status="skipped", reason="training_partition_missing_class"))
            continue
        try:
            model = _fit(x[train], y[train], cfg)
        except ValueError as error:
            folds.append(dict(**summary, status="skipped", reason=str(error)))
            continue
        probability = _predict(model, x[test])
        usable = np.isfinite(probability).all(axis=1)
        predicted = np.full(int(test.sum()), "insufficient_features", dtype=object)
        predicted[usable] = model["classes"][probability[usable].argmax(axis=1)]
        table = units.loc[test, ["global_unit_id", "animal", "session", "recording"]].copy()
        table["target"], table["prediction"], table["features"] = y[test], predicted, feature_kind
        table["prediction_status"] = np.where(usable, "scored", "insufficient_features")
        for index, name in enumerate(model["classes"]):
            table[f"p_{name}"] = probability[:, index]
        predictions.append(table)
        folds.append(dict(**summary, status="scored" if usable.any() else "skipped",
                          reason="" if usable.any() else "no_test_neurons_with_usable_features",
                          n_train_usable=int(np.isfinite(x[train][:, model["keep"]]).any(axis=1).sum()),
                          n_test_scored=int(usable.sum()), n_test_insufficient_features=int((~usable).sum()),
                          accuracy=float(np.mean(predicted[usable] == y[test][usable])) if usable.any() else None))
    frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    summary = dict(features=feature_kind, holdout=str(cfg.holdout), n_labeled=len(y), **_metrics(frame, classes),
                   interpretation="Readout holdout only; the encoder may have seen these sessions during pretraining")
    if feature_kind == "baseline":
        summary["feature_availability"] = _feature_availability(x, list(cfg.baseline_features))
    return frame, pd.DataFrame(folds), summary


def run(cfg):
    if not cfg.get("embeddings"):
        raise ValueError("Set embeddings to an exported pooled directory")
    if cfg.action not in {"fit", "predict"}:
        raise ValueError("celltypes action must be fit or predict")
    if not 0 <= float(cfg.min_probability) <= 1 or float(cfg.C) <= 0:
        raise ValueError("Require min_probability in [0,1] and C > 0")
    output = resolve(cfg.output)
    expected_file = output / ("classifier.npz" if cfg.action == "fit" else "predictions.csv")
    if expected_file.exists():
        raise FileExistsError(f"Output already exists: {expected_file}; choose a new output directory")
    values, units, source = load_means(cfg.embeddings, cfg.get("cohort"))
    if cfg.single_units_only:
        if "decoder_label" not in units:
            raise ValueError("single_units_only requires sorting labels in the embedding metadata")
        selected = units.decoder_label.astype(str).str.lower().eq("sua").to_numpy()
        values, units = values[selected], units.loc[selected].reset_index(drop=True)
    if not len(units):
        raise ValueError("No units remain after cohort/single-unit filtering")
    names = list(cfg.baseline_features)
    if cfg.action == "predict":
        if not cfg.get("classifier"):
            raise ValueError("Set classifier to the fitted classifier.npz path")
        classifier_path = resolve(cfg.classifier)
        model_info = json.loads(classifier_path.with_suffix(".json").read_text())
        if model_info["features"] == "embedding":
            if not source.get("encoder_signature") or source["encoder_signature"] != model_info.get("encoder_signature"):
                raise ValueError("Classifier and target embeddings must use the same encoder checkpoint. Re-encode labeled passive examples and refit the readout after fine-tuning.")
        with np.load(classifier_path, allow_pickle=False) as saved:
            model = {key: saved[key] for key in saved.files}
        x = _matrix(values, units, model_info["features"], model_info["baseline_features"])
        probability = _predict(model, x)
        usable = np.isfinite(probability).all(axis=1)
        table = units.copy()
        for index, label in enumerate(model["classes"]):
            table[f"p_{label}"] = probability[:, index]
        table["max_probability"] = probability.max(axis=1)
        argmax = np.full(len(table), "insufficient_features", dtype=object)
        argmax[usable] = model["classes"][probability[usable].argmax(axis=1)]
        table["argmax_label"] = argmax
        table["predicted_label"] = table.argmax_label.where(table.max_probability >= cfg.min_probability, "uncertain")
        table["prediction_status"] = np.where(usable,
                                               np.where(table.max_probability >= cfg.min_probability, "predicted", "uncertain"),
                                               "insufficient_features")
        output.mkdir(parents=True, exist_ok=True)
        table.to_csv(output / "predictions.csv", index=False)
        information = dict(**source, classifier=str(classifier_path), n_units=len(table),
                           n_insufficient_features=int((~usable).sum()),
                           probability_interpretation=model_info["probability_interpretation"], min_probability=float(cfg.min_probability))
        if model_info["features"] == "baseline":
            information["feature_availability"] = _feature_availability(x, model_info["baseline_features"], model["keep"])
        (output / "predictions.meta.json").write_text(json.dumps(information, indent=2) + "\n")
        from omegaconf import OmegaConf
        OmegaConf.save(cfg, output / "predict_config.yaml")
        print(output / "predictions.csv")
        return information

    if "label" not in units:
        raise ValueError("Labeled passive embeddings must include a label column")
    mapping = {str(k).lower(): str(v) for k, v in cfg.label_map.items()}
    targets = units.label.astype(str).str.strip().str.lower().map(mapping)
    selected = targets.isin(list(cfg.classes)).to_numpy()
    values, labeled, y = values[selected], units.loc[selected].reset_index(drop=True), targets.loc[selected].to_numpy(str)
    if set(np.unique(y)) != set(map(str, cfg.classes)) or len(cfg.classes) < 2:
        raise ValueError("All configured classes need labeled examples; inspect label_map, cohort and unit selection")
    if cfg.features == "embedding" and not source.get("encoder_signature"):
        raise ValueError("Cannot identify the encoder checkpoint for this export. Export it with the new workflow first.")
    x = _matrix(values, labeled, str(cfg.features), names)
    model = _fit(x, y, cfg)
    predictions, folds, summary = _evaluate(x, y, labeled, cfg, str(cfg.features))
    summaries = [summary]
    paired_comparison = None
    shared_predictions = None
    if cfg.evaluate_baseline and cfg.features != "baseline":
        baseline = _matrix(values, labeled, "baseline", names)
        baseline_predictions, baseline_folds, baseline_summary = _evaluate(baseline, y, labeled, cfg, "baseline")
        paired_comparison, shared_predictions = _shared_comparison(predictions, baseline_predictions, np.unique(y))
        predictions = pd.concat([predictions, baseline_predictions], ignore_index=True)
        folds = pd.concat([folds, baseline_folds], ignore_index=True)
        summaries.append(baseline_summary)
    information = dict(**source, features=str(cfg.features), baseline_features=names,
                       classes=list(map(str, model["classes"])), n_labeled=len(y),
                       label_map=dict(cfg.label_map), single_units_only=bool(cfg.single_units_only),
                       probability_interpretation="Uncalibrated class-balanced logistic probabilities; putative assignments, not verified cell identities",
                       evaluation=summaries)
    if paired_comparison is not None:
        information["paired_comparison"] = paired_comparison
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "classifier.npz", **model)
    (output / "classifier.json").write_text(json.dumps(information, indent=2, allow_nan=False) + "\n")
    folds.to_csv(output / "folds.csv", index=False)
    predictions.to_csv(output / "heldout_predictions.csv", index=False)
    if shared_predictions is not None:
        shared_predictions.to_csv(output / "shared_heldout_predictions.csv", index=False)
    from omegaconf import OmegaConf
    OmegaConf.save(cfg, output / "config.yaml")
    print(output / "classifier.npz")
    return information
