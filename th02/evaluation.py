"""Probability validation, calibration, official scoring, and error selection."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unicodedata
import warnings

import numpy as np


LABELS = ("winner_model_a", "winner_model_b", "winner_tie")
TEMPERATURES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def validate_probabilities(probabilities) -> np.ndarray:
    """Validate the official three-class probability contract without rescaling."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("probabilities must be between zero and one")
    if np.any(np.abs(values.sum(axis=1) - 1.0) > 1e-6):
        raise ValueError("probability rows must sum to one within 1e-6")
    return values


def _targets(y, size: int) -> np.ndarray:
    values = np.asarray(y)
    if values.shape != (size,):
        raise ValueError("targets must be one dimensional and match probabilities")
    if size == 0:
        raise ValueError("evaluation requires at least one sample")
    if not np.all(np.isin(values, (0, 1, 2))):
        raise ValueError("targets must be class indices 0, 1, or 2")
    return values.astype(np.int64)


def _log_loss(y: np.ndarray, probabilities: np.ndarray) -> float:
    correct = probabilities[np.arange(len(y)), y]
    return float(-np.log(np.clip(correct, 1e-15, 1.0 - 1e-15)).mean())


def symmetrize(original, swapped) -> np.ndarray:
    """Map swapped A/B predictions to original labels and average both views."""
    original = validate_probabilities(original)
    swapped = validate_probabilities(swapped)
    if original.shape != swapped.shape:
        raise ValueError("original and swapped predictions must have equal shape")
    return validate_probabilities((original + swapped[:, [1, 0, 2]]) * 0.5)


def temperature_scale(probabilities, temperature: float) -> np.ndarray:
    probabilities = validate_probabilities(probabilities)
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(probabilities, 1e-15, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    scaled /= scaled.sum(axis=1, keepdims=True)
    return validate_probabilities(scaled)


def _temperature_search(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    candidates = []
    for temperature in TEMPERATURES:
        loss = _log_loss(y, temperature_scale(probabilities, temperature))
        candidates.append((loss, abs(temperature - 1.0), temperature))
    loss, _, temperature = min(candidates)
    return float(temperature), float(loss)


def select_calibration(y, lex_raw, lex_sym, sem_sym) -> dict:
    """Choose all calibration settings using only the internal calibration set."""
    lex_raw = validate_probabilities(lex_raw)
    lex_sym = validate_probabilities(lex_sym)
    sem_sym = validate_probabilities(sem_sym)
    if lex_raw.shape != lex_sym.shape or lex_raw.shape != sem_sym.shape:
        raise ValueError("all calibration predictions must have equal shape")
    y = _targets(y, len(lex_raw))
    baseline_temperature, baseline_loss = _temperature_search(y, lex_raw)
    symmetric_temperature, symmetric_loss = _temperature_search(y, lex_sym)
    grid = []
    for alpha in ALPHAS:
        fused = (1.0 - alpha) * lex_sym + alpha * sem_sym
        for temperature in TEMPERATURES:
            loss = _log_loss(y, temperature_scale(fused, temperature))
            grid.append({"alpha": alpha, "temperature": temperature, "log_loss": loss})
    selected = min(
        grid,
        key=lambda row: (row["log_loss"], abs(row["temperature"] - 1.0), row["alpha"]),
    )
    return {
        "baseline_temperature": baseline_temperature,
        "symmetric_baseline_temperature": symmetric_temperature,
        "alpha": selected["alpha"],
        "temperature": selected["temperature"],
        "baseline_log_loss": baseline_loss,
        "symmetric_baseline_log_loss": symmetric_loss,
        "fused_log_loss": selected["log_loss"],
        "grid": grid,
    }


def metrics(y, probabilities) -> dict:
    probabilities = validate_probabilities(probabilities)
    y = _targets(y, len(probabilities))
    predicted = probabilities.argmax(axis=1)
    per_class = {}
    for index, name in enumerate(LABELS):
        truth_mask = y == index
        predicted_mask = predicted == index
        true_positive = int(np.count_nonzero(truth_mask & predicted_mask))
        support = int(np.count_nonzero(truth_mask))
        predicted_count = int(np.count_nonzero(predicted_mask))
        per_class[name] = {
            "precision": true_positive / predicted_count if predicted_count else 0.0,
            "recall": true_positive / support if support else 0.0,
            "support": support,
        }
    one_hot = np.eye(3, dtype=np.float64)[y]
    return {
        "log_loss": _log_loss(y, probabilities),
        "accuracy": float(np.mean(predicted == y)),
        "brier_sum": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "per_class": per_class,
        "count": len(y),
    }


def write_predictions(samples, probabilities, path) -> None:
    probabilities = validate_probabilities(probabilities)
    identifiers = [str(sample.id) for sample in samples]
    if len(identifiers) != len(probabilities):
        raise ValueError("prediction count does not match sample count")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("prediction sample IDs must be unique")
    if any(not identifier for identifier in identifiers):
        raise ValueError("prediction sample IDs must be nonempty")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=destination.parent,
            prefix=destination.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(("id", *LABELS))
            for identifier, row in zip(identifiers, probabilities):
                writer.writerow((identifier, *(format(float(value), ".17g") for value in row)))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_csv(path) -> list[dict]:
    csv.field_size_limit(2**31 - 1)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", *LABELS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain id and all three label columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no samples")
    identifiers = [row["id"] for row in rows]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(rows):
        raise ValueError(f"{path} contains empty or duplicate sample IDs")
    return rows


def score_official(truth_path, pred_path) -> float:
    """Validate exact ID coverage before invoking the supplied unmodified scorer."""
    truth = _read_csv(truth_path)
    predicted = _read_csv(pred_path)
    truth_ids = {row["id"] for row in truth}
    prediction_ids = {row["id"] for row in predicted}
    if truth_ids != prediction_ids:
        missing = sorted(truth_ids - prediction_ids)[:5]
        extra = sorted(prediction_ids - truth_ids)[:5]
        raise ValueError(f"prediction ID coverage mismatch, missing={missing}, extra={extra}")
    validate_probabilities([[float(row[label]) for label in LABELS] for row in predicted])
    for row in truth:
        values = [row[label] for label in LABELS]
        if any(value not in ("0", "1") for value in values) or values.count("1") != 1:
            raise ValueError(f"invalid one-hot truth labels for sample {row['id']}")
    metric_path = Path(__file__).resolve().parents[1] / "evaluation" / "metric.py"
    spec = importlib.util.spec_from_file_location("th02_supplied_metric", metric_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the supplied evaluation/metric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return float(module.score(str(truth_path), str(pred_path)))


def _content_key(sample) -> str:
    fields = []
    for values in (sample.prompt, sample.response_a, sample.response_b):
        fields.append([" ".join(unicodedata.normalize("NFKC", value).split()) for value in values])
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


def select_error_indices(samples, y, baseline, improved, count: int = 12) -> list[int]:
    """Select content-distinct error/uncertainty cases with varied observable traits."""
    baseline = validate_probabilities(baseline)
    improved = validate_probabilities(improved)
    if baseline.shape != improved.shape or len(samples) != len(baseline):
        raise ValueError("sample and prediction counts must agree")
    y = _targets(y, len(improved))
    if count < 0:
        raise ValueError("error sample count must be nonnegative")
    wrong_baseline = baseline.argmax(axis=1) != y
    wrong_improved = improved.argmax(axis=1) != y
    confidence = improved.max(axis=1)
    uncertain = confidence <= 0.5
    eligible = wrong_baseline | wrong_improved | uncertain
    correct_probability = improved[np.arange(len(y)), y]
    loss = -np.log(np.clip(correct_probability, 1e-15, 1.0))
    entropy = -np.sum(improved * np.log(np.clip(improved, 1e-15, 1.0)), axis=1)
    selected = []
    used_content = set()

    def choose(indices, key, limit: int = 1):
        added = 0
        for index in sorted(indices, key=key, reverse=True):
            if len(selected) >= count or added >= limit:
                break
            index = int(index)
            content = _content_key(samples[index])
            if eligible[index] and content not in used_content:
                selected.append(index)
                used_content.add(content)
                added += 1

    regressions = np.flatnonzero(~wrong_baseline & wrong_improved)
    choose(regressions, lambda index: loss[index], 2)
    choose(np.flatnonzero(wrong_improved & (confidence >= 0.5)), lambda index: confidence[index], 2)
    choose(np.flatnonzero(uncertain), lambda index: entropy[index], 2)
    traits = (
        lambda sample: any(sample.null_counts)
        or not any(value.strip() for value in sample.response_a)
        or not any(value.strip() for value in sample.response_b),
        lambda sample: len(sample.prompt) > 1,
        lambda sample: sum(len(text) for text in sample.texts) > 8000,
        lambda sample: sample.response_a == sample.response_b,
    )
    for trait in traits:
        indices = [index for index, sample in enumerate(samples) if eligible[index] and trait(sample)]
        choose(indices, lambda index: loss[index])
    choose(np.flatnonzero(eligible), lambda index: (loss[index], entropy[index]), count)
    if len(selected) < count:
        warnings.warn(
            f"requested {count} error cases but found {len(selected)} content-distinct eligible samples",
            RuntimeWarning,
            stacklevel=2,
        )
    return selected
