#!/usr/bin/env python3
"""Independently verify a completed full run using the supplied official scorer.

Run from the repository root:
    python scripts/verify_training.py --output-dir artifacts --data-dir data

Only the Python standard library is required. Log Loss remains the primary
metric; confidence and fixed ten-bin ECE are descriptive diagnostics only.
This script never selects model or calibration settings.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("winner_model_a", "winner_model_b", "winner_tie")
EXPERIMENTS = (
    "prior", "baseline", "baseline_calibrated", "symmetric_baseline",
    "symmetric_baseline_calibrated", "semantic", "fusion", "fusion_calibrated",
)
TOLERANCE = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_truth(path: Path) -> dict[str, int]:
    truth = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        require({"id", *LABELS}.issubset(reader.fieldnames or []),
                f"{path}: missing ID or label columns")
        for row in reader:
            identifier = row.get("id")
            require(bool(identifier), f"{path}: empty ID at CSV line {reader.line_num}")
            require(identifier not in truth, f"{path}: duplicate ID {identifier}")
            values = [row.get(label) for label in LABELS]
            require(all(value in ("0", "1") for value in values) and values.count("1") == 1,
                    f"{path}: invalid one-hot label for ID {identifier}")
            truth[identifier] = values.index("1")
    require(bool(truth), f"{path}: no samples")
    return truth


def prediction_diagnostics(path: Path, truth: dict[str, int]) -> dict:
    seen = set()
    bins = [{"lower": index / 10, "upper": (index + 1) / 10,
             "count": 0, "correct": 0, "confidence_sum": 0.0}
            for index in range(10)]
    confidence_sum = error_confidence_sum = error_loss_sum = loss_sum = 0.0
    errors = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        require(reader.fieldnames == ["id", *LABELS], f"{path}: unexpected columns/order")
        for row in reader:
            identifier = row.get("id")
            require(identifier in truth, f"{path}: unknown or empty ID {identifier!r}")
            require(identifier not in seen, f"{path}: duplicate ID {identifier}")
            seen.add(identifier)
            try:
                probabilities = [float(row[label]) for label in LABELS]
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid probability for ID {identifier}") from error
            require(all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities),
                    f"{path}: probability outside finite [0, 1] for ID {identifier}")
            require(abs(sum(probabilities) - 1.0) <= 1e-6,
                    f"{path}: probability row does not sum to one for ID {identifier}")
            confidence = max(probabilities)
            predicted = probabilities.index(confidence)
            correct = predicted == truth[identifier]
            loss = -math.log(max(1e-15, min(1 - 1e-15, probabilities[truth[identifier]])))
            loss_sum += loss
            confidence_sum += confidence
            bucket = bins[min(9, int(confidence * 10))]
            bucket["count"] += 1
            bucket["correct"] += int(correct)
            bucket["confidence_sum"] += confidence
            if not correct:
                errors += 1
                error_confidence_sum += confidence
                error_loss_sum += loss
    missing = set(truth) - seen
    require(not missing, f"{path}: missing {len(missing)} IDs; examples {sorted(missing)[:5]}")
    count = len(truth)
    ece = 0.0
    for bucket in bins:
        size = bucket["count"]
        bucket["accuracy"] = bucket["correct"] / size if size else None
        bucket["mean_confidence"] = bucket.pop("confidence_sum") / size if size else None
        if size:
            ece += size / count * abs(bucket["accuracy"] - bucket["mean_confidence"])
    return {
        "count": count, "independent_log_loss": loss_sum / count,
        "accuracy": 1 - errors / count, "mean_confidence": confidence_sum / count,
        "error_count": errors,
        "error_mean_confidence": error_confidence_sum / errors if errors else None,
        "error_mean_log_loss": error_loss_sum / errors if errors else None,
        "confidence_ece_10_bins": ece, "confidence_bins": bins,
    }


def verify(output: Path, data: Path, report: dict) -> None:
    state = read_json(output / "run_state.json")
    metrics = read_json(output / "metrics.json")
    require(state.get("status") == "complete", "run_state.status must be complete")
    require(metrics.get("run_status") == "complete", "metrics.run_status must be complete")
    require(metrics.get("smoke") is False, "metrics must describe a full non-smoke run")
    configuration = state.get("configuration", {})
    require(configuration.get("smoke") is False, "run_state must describe a full non-smoke run")
    require(configuration.get("seed") == metrics.get("seed"), "random-seed records disagree")

    required_artifacts = (
        "models/baseline.joblib", "models/semantic.pt", "models/calibration.json",
        "models/selection_locked.json",
    )
    for name in required_artifacts:
        path = output / name
        require(path.is_file() and path.stat().st_size > 0, f"missing/empty artifact: {path}")
    report["model_artifacts"] = {
        name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
        for name in required_artifacts
    }
    locked = read_json(output / "models/selection_locked.json")
    calibration = read_json(output / "models/calibration.json")
    require(locked.get("seed") == metrics.get("seed"), "selection lock seed disagrees")
    require(locked.get("calibration") == calibration == metrics.get("calibration"),
            "saved calibration differs from the selection lock or metrics")

    metric_path = ROOT / "evaluation/metric.py"
    report["official_metric_sha256"] = digest(metric_path)
    report["data_sha256"] = {name: digest(data / name) for name in ("train.csv", "dev.csv")}
    manifest = {}
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, name = line.split(maxsplit=1)
            manifest[name] = expected
    require(report["official_metric_sha256"] == manifest.get("evaluation/metric.py"),
            "official evaluation/metric.py differs from MANIFEST.sha256")
    for name, sha256 in report["data_sha256"].items():
        require(sha256 == manifest.get("data/" + name), f"{name} differs from MANIFEST.sha256")
        require(sha256 == configuration.get("data_sha256", {}).get(name),
                f"{name} differs from the dataset recorded by this run")

    train = read_truth(data / "train.csv")
    dev = read_truth(data / "dev.csv")
    require(not set(train).intersection(dev), "train and dev share sample IDs")
    report["counts"] = {"train": len(train), "dev": len(dev)}
    split_manifest = read_json(data / "split_manifest.json")
    require(split_manifest.get("counts", {}).get("train") == len(train),
            "train row count differs from data/split_manifest.json")
    require(split_manifest.get("counts", {}).get("dev") == len(dev),
            "dev row count differs from data/split_manifest.json")
    splits = metrics.get("splits", {})
    for name in ("fit", "tune", "calibration", "refit"):
        require(type(splits.get(name)) is int and splits[name] > 0, f"invalid {name} split count")
    require(splits["fit"] + splits["tune"] + splits["calibration"] == len(train),
            "metrics split counts do not cover the full train dataset")
    require(splits["fit"] + splits["tune"] == splits["refit"], "refit count disagrees")
    profile = read_json(output / "data_profile.json")
    for name, count in report["counts"].items():
        require(profile.get(name, {}).get("rows") == count, f"data_profile {name} count disagrees")

    spec = importlib.util.spec_from_file_location("th02_official_verification_metric", metric_path)
    require(spec is not None and spec.loader is not None, "cannot import official scorer")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    results = metrics.get("results", {})
    report["experiments"] = {}
    for name in EXPERIMENTS:
        path = output / "predictions" / (name + ".csv")
        recorded = results.get(name, {})
        require(recorded.get("count") == len(dev), f"{name}: recorded dev count disagrees")
        values = prediction_diagnostics(path, dev)
        score = float(official.score(str(data / "dev.csv"), str(path)))
        require(math.isfinite(score), f"{name}: official scorer returned non-finite loss")
        values["official_log_loss"] = score
        values["prediction_sha256"] = digest(path)
        for key in ("log_loss", "official_log_loss"):
            value = recorded.get(key)
            require(isinstance(value, (int, float)) and math.isfinite(value),
                    f"{name}: missing/invalid recorded {key}")
            require(abs(score - value) <= TOLERANCE,
                    f"{name}: {key} differs from official score by {abs(score - value):.3g}")
        require(abs(score - values["independent_log_loss"]) <= TOLERANCE,
                f"{name}: independently computed Log Loss differs from official score")
        require(abs(values["accuracy"] - recorded.get("accuracy", float("nan"))) <= TOLERANCE,
                f"{name}: accuracy record disagrees with predictions")
        report["experiments"][name] = values


def save_report(path: Path, report: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    csv.field_size_limit(sys.maxsize)
    output, data = args.output_dir.resolve(), args.data_dir.resolve()
    report = {
        "status": "failed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output), "data_dir": str(data),
        "primary_metric": "three-class Log Loss, probabilities clipped to [1e-15, 1-1e-15]",
        "score_tolerance": TOLERANCE,
        "diagnostic_note": "ECE uses 10 fixed confidence bins [0,.1), ... [.9,1]; "
                           "auxiliary diagnostics do not select hyperparameters or replace Log Loss.",
    }
    try:
        require(output.is_dir(), f"output directory does not exist: {output}")
        verify(output, data, report)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    try:
        if output.is_dir():
            save_report(output / "verification.json", report)
    except Exception as error:
        print(f"FAIL: cannot write verification.json: {error}", file=sys.stderr)
        return 1
    if report["status"] != "passed":
        print("FAIL: " + report["error"], file=sys.stderr)
        return 1
    print(f"PASS: full train={report['counts']['train']}, dev={report['counts']['dev']}; "
          f"all {len(EXPERIMENTS)} predictions match official Log Loss within {TOLERANCE:g}.")
    for name, values in report["experiments"].items():
        print(f"  {name}: Log Loss={values['official_log_loss']:.9f}, "
              f"ECE10={values['confidence_ece_10_bins']:.6f}")
    print(f"Saved {output / 'verification.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
