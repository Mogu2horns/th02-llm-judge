"""Controlled frozen-encoder head experiments on the original fit/tune split.

Run ``python -m th02.head_experiments --phase all``.  Official dev is never
loaded here.  The production feature builder is reused; only the classifier
and explicitly named optimization/ablation settings change.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260916, 20260917, 20260918)
HEADS = {"H0_linear": (), "H1_128": (128,), "H2_512": (512,), "H3_128_128": (128, 128)}
LABELS = ("winner_model_a", "winner_model_b", "winner_tie")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configure_process(gpu: str) -> None:
    """Apply the measured server workaround before importing NumPy/PyTorch."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "8"
    temporary = ROOT / ".cache/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if libc.prctl(42, 0, 0, 0, 0) != 1:
        raise RuntimeError("Process-local THP disabling could not be verified")


def build_network(hidden: tuple[int, ...], input_dim: int = 2706, dropout: float = 0.2):
    """Three raw logits, ReLU hidden layers, one final-hidden dropout."""
    from torch import nn

    modules = []
    previous = input_dim
    for index, width in enumerate(hidden):
        if width < 1:
            raise ValueError("Hidden widths must be positive")
        modules.extend((nn.Linear(previous, width), nn.ReLU()))
        if index == len(hidden) - 1:
            modules.append(nn.Dropout(dropout))
        previous = width
    modules.append(nn.Linear(previous, 3))
    return nn.Sequential(*modules)


def extended_metrics(labels, probabilities) -> dict:
    from th02.evaluation import metrics

    result = metrics(labels, probabilities)
    for row in result["per_class"].values():
        precision, recall = row["precision"], row["recall"]
        row["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result["macro_f1"] = statistics.mean(row["f1"] for row in result["per_class"].values())
    return result


class HeadModel:
    """Serializable general head using the original semantic feature contract."""

    def __init__(self, network, scaler, hidden, device="cuda", numeric_features=True):
        self.network = network.to(device)
        self.scaler = scaler
        self.hidden = tuple(hidden)
        self.device = device
        self.numeric_features = bool(numeric_features)

    def predict_proba(self, samples, embeddings, swapped=False, batch_size=256):
        import numpy as np
        import torch
        from th02.data import structural_features
        from th02.semantic import _batch_features

        if len(samples) != len(embeddings):
            raise ValueError("Samples and embeddings must have the same length")
        structural = structural_features(samples, swapped=swapped)
        output = np.empty((len(samples), 3), dtype=np.float64)
        self.network.eval()
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                stop = min(start + batch_size, len(samples))
                vectors = np.array(embeddings[start:stop], dtype=np.float32, copy=True)
                if swapped:
                    vectors[:, [1, 2]] = vectors[:, [2, 1]]
                features = _batch_features(vectors, structural[start:stop], self.scaler, self.device)
                if not self.numeric_features:
                    features = features[:, :2688]
                output[start:stop] = self.network(features).softmax(dim=1).cpu().numpy()
        output /= output.sum(axis=1, keepdims=True)
        return output

    def save(self, path: Path, metadata: dict) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "th02_controlled_head_v1",
            "state_dict": {key: value.detach().cpu() for key, value in self.network.state_dict().items()},
            "config": {"hidden": list(self.hidden), "dropout": 0.2,
                       "numeric_features": self.numeric_features,
                       "input_dim": 2706 if self.numeric_features else 2688},
            "scaler": {"mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
                       "var": self.scaler.var_.tolist(),
                       "n_samples_seen": int(self.scaler.n_samples_seen_)},
            "label_order": list(LABELS), "metadata": metadata,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)


def load_head(path: Path, device="cuda") -> HeadModel:
    import numpy as np
    import torch
    from sklearn.preprocessing import StandardScaler

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "th02_controlled_head_v1" or payload["label_order"] != list(LABELS):
        raise ValueError("Unrecognized head checkpoint format or label order")
    config = payload["config"]
    expected_dim = 2706 if config["numeric_features"] else 2688
    if config["input_dim"] != expected_dim:
        raise ValueError("Head checkpoint input dimension does not match its feature mode")
    scaler = StandardScaler()
    for name in ("mean", "scale", "var"):
        setattr(scaler, name + "_", np.asarray(payload["scaler"][name], dtype=np.float64))
    scaler.n_samples_seen_ = payload["scaler"]["n_samples_seen"]
    scaler.n_features_in_ = 18
    network = build_network(tuple(config["hidden"]), expected_dim, config["dropout"])
    network.load_state_dict(payload["state_dict"])
    return HeadModel(network, scaler, config["hidden"], device, config["numeric_features"])


def fit_fixed_head(samples, embeddings, indices, hidden, epochs, seed=20260916,
                   learning_rate=1e-3, device="cuda", numeric_features=True,
                   swap_augmentation=True):
    """Refit a locked configuration for fixed epochs, without validation selection.

    Returns ``(HeadModel, metadata)``. Call ``model.save(path, metadata)`` and
    ``symmetrize(model.predict_proba(...), model.predict_proba(..., swapped=True))``
    for independent calibration or final evaluation sets.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    from th02.data import structural_features, targets
    from th02.semantic import _batch_features, _fit_scaler, _seed_everything

    if epochs < 1:
        raise ValueError("Fixed refit epochs must be positive")
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    structural = structural_features(samples)
    structural_swapped = structural_features(samples, swapped=True)
    scaler = _fit_scaler(embeddings, structural, structural_swapped, indices)
    model = HeadModel(build_network(tuple(hidden), 2706 if numeric_features else 2688),
                      scaler, hidden, device, numeric_features)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=learning_rate, weight_decay=1e-4)
    labels = targets(samples)
    history = []
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.network.train()
        total_loss = 0.0
        order = rng.permutation(indices)
        for start in range(0, len(order), 256):
            selected = order[start:start + 256]
            vectors = np.array(embeddings[selected], dtype=np.float32, copy=True)
            numeric = structural[selected].copy()
            batch_labels = labels[selected].copy()
            mask = rng.random(len(selected)) < 0.5
            if swap_augmentation:
                vectors[mask, 1], vectors[mask, 2] = vectors[mask, 2].copy(), vectors[mask, 1].copy()
                numeric[mask] = structural_swapped[selected[mask]]
                batch_labels[mask] = np.asarray([1, 0, 2])[batch_labels[mask]]
            features = _batch_features(vectors, numeric, scaler, device)
            if not numeric_features:
                features = features[:, :2688]
            expected = torch.as_tensor(batch_labels, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.network(features), expected)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite fixed-head refit loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(selected)
        row = {"epoch": epoch, "train_online_loss": total_loss / len(indices)}
        history.append(row)
        print(f"[heads-refit] {json.dumps(row)}", flush=True)
    metadata = {"phase": "locked_fixed_epoch_refit", "hidden": list(hidden), "epochs": epochs,
                "seed": seed, "learning_rate": learning_rate, "weight_decay": 1e-4,
                "batch_size": 256, "fit_count": len(indices), "numeric_features": numeric_features,
                "swap_augmentation": swap_augmentation, "history": history,
                "elapsed_seconds": time.monotonic() - started}
    return model, metadata


@dataclass(frozen=True)
class Trial:
    head: str
    seed: int
    learning_rate: float = 1e-3
    numeric_features: bool = True
    swap_augmentation: bool = True
    scheduler: bool = False
    max_epochs: int = 50
    patience: int = 8

    @property
    def name(self) -> str:
        suffix = "" if self.numeric_features else "_no_numeric"
        suffix += "" if self.swap_augmentation else "_no_train_swap"
        suffix += "_plateau" if self.scheduler else ""
        return f"{self.head}_lr{self.learning_rate:g}{suffix}_seed{self.seed}"

    def configuration(self) -> dict:
        return {"head": self.head, "hidden": list(HEADS[self.head]), "seed": self.seed,
                "learning_rate": self.learning_rate, "numeric_features": self.numeric_features,
                "swap_augmentation": self.swap_augmentation, "scheduler": self.scheduler,
                "max_epochs": self.max_epochs, "patience": self.patience,
                "min_delta": 1e-8, "optimizer": "AdamW", "weight_decay": 1e-4,
                "batch_size": 256, "dropout": 0.2, "symmetric_inference": True}


class ExperimentData:
    """Fit-only standardization and cached two-view features on one GPU."""

    def __init__(self, device="cuda"):
        import numpy as np
        import torch
        from th02.data import load_samples, structural_features, targets
        from th02 import semantic

        self.device = device
        production = ROOT / "artifacts"
        train_path = ROOT / "data/train.csv"
        split_path = production / "splits.json"
        self.sources = {"train_sha256": sha256(train_path), "split_sha256": sha256(split_path),
                        "head_source_sha256": sha256(Path(__file__)),
                        "semantic_source_sha256": sha256(ROOT / "th02/semantic.py")}
        original = json.loads((production / "run_state.json").read_text())
        if self.sources["train_sha256"] != original["configuration"]["data_sha256"]["train.csv"]:
            raise ValueError("Training CSV differs from the frozen original experiment")
        self.samples_all = load_samples(train_path)
        id_to_index = {sample.id: index for index, sample in enumerate(self.samples_all)}
        split_record = json.loads(split_path.read_text())
        self.splits_global = {
            name: np.asarray([id_to_index[identifier] for identifier in split_record[name]["ids"]],
                             dtype=np.int64)
            for name in ("fit", "tune", "calibration", "refit")}
        for first, second in (("fit", "tune"), ("fit", "calibration"), ("tune", "calibration")):
            left, right = self.splits_global[first], self.splits_global[second]
            if set(left).intersection(right):
                raise ValueError(f"Sample ID overlap: {first}/{second}")
            if {self.samples_all[int(i)].group_hash for i in left}.intersection(
                    self.samples_all[int(i)].group_hash for i in right):
                raise ValueError(f"Normalized dialogue overlap: {first}/{second}")
        self.embeddings_all = np.load(production / "cache/train/embeddings.npy", mmap_mode="r")
        metadata = json.loads((production / "cache/train/metadata.json").read_text())
        if self.embeddings_all.shape != (len(self.samples_all), 3, semantic.EMBEDDING_DIM):
            raise ValueError("Invalid frozen embedding cache shape")
        if semantic._fingerprint(self.samples_all, metadata["model"], device) != metadata["fingerprint"]:
            raise ValueError("Frozen embedding cache fingerprint mismatch")
        self.sources["embedding_fingerprint"] = metadata["fingerprint"]
        self.structural = structural_features(self.samples_all)
        self.structural_swapped = structural_features(self.samples_all, swapped=True)
        self.scaler = semantic._fit_scaler(self.embeddings_all, self.structural,
                                           self.structural_swapped, self.splits_global["fit"])
        # Only fit+tune features/labels are accessible to the optimizer and selector.
        selected = np.sort(np.concatenate((self.splits_global["fit"], self.splits_global["tune"])))
        mapping = np.full(len(self.samples_all), -1, dtype=np.int64)
        mapping[selected] = np.arange(len(selected))
        self.fit_indices = mapping[self.splits_global["fit"]]
        self.tune_indices = mapping[self.splits_global["tune"]]
        self.samples = [self.samples_all[int(i)] for i in selected]
        self.labels = targets(self.samples)
        self.direct = torch.empty((len(selected), 2706), dtype=torch.float32, device=device)
        self.swapped = torch.empty_like(self.direct)
        for start in range(0, len(selected), 256):
            stop = min(start + 256, len(selected))
            indices = selected[start:stop]
            vectors = np.array(self.embeddings_all[indices], dtype=np.float32, copy=True)
            self.direct[start:stop] = semantic._batch_features(
                vectors, self.structural[indices], self.scaler, device)
            vectors[:, [1, 2]] = vectors[:, [2, 1]]
            self.swapped[start:stop] = semantic._batch_features(
                vectors, self.structural_swapped[indices], self.scaler, device)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        self.split_counts = {name: len(indices) for name, indices in self.splits_global.items()}
        print(f"[heads] prepared original split {self.split_counts}; cached two-view features", flush=True)

    def predict(self, model: HeadModel, indices):
        import numpy as np
        import torch

        original = np.empty((len(indices), 3), dtype=np.float64)
        swapped = np.empty_like(original)
        model.network.eval()
        with torch.inference_mode():
            for start in range(0, len(indices), 256):
                stop = min(start + 256, len(indices))
                selected = torch.as_tensor(indices[start:stop], dtype=torch.long, device=self.device)
                for features, output in ((self.direct, original), (self.swapped, swapped)):
                    batch = features.index_select(0, selected)
                    if not model.numeric_features:
                        batch = batch[:, :2688]
                    output[start:stop] = model.network(batch).softmax(dim=1).cpu().numpy()
        original /= original.sum(axis=1, keepdims=True)
        swapped /= swapped.sum(axis=1, keepdims=True)
        return (original + swapped[:, [1, 0, 2]]) * 0.5


def run_trial(data: ExperimentData, trial: Trial, output: Path) -> dict:
    import numpy as np
    import torch
    from torch.nn import functional as F
    from th02.evaluation import write_predictions
    from th02.semantic import _seed_everything

    directory = output / "trials" / trial.name
    result_path = directory / "metrics.json"
    configuration = trial.configuration()
    identity = {"configuration": configuration, "sources": data.sources}
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("identity") != identity:
            raise ValueError(f"Completed trial identity changed; preserve evidence and use a new output: {trial.name}")
        for name, expected in previous["artifact_sha256"].items():
            if sha256(directory / name) != expected:
                raise ValueError(f"Completed trial artifact changed: {trial.name}/{name}")
        print(f"[heads] resume: reuse completed {trial.name}", flush=True)
        return previous
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "configuration.json", identity)
    _seed_everything(trial.seed)
    rng = np.random.default_rng(trial.seed)
    model = HeadModel(build_network(HEADS[trial.head], 2706 if trial.numeric_features else 2688),
                      data.scaler, HEADS[trial.head], data.device, trial.numeric_features)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=trial.learning_rate, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=1e-8,
        threshold_mode="abs", min_lr=1e-6) if trial.scheduler else None)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_accuracy, best_accuracy_loss, best_accuracy_epoch = -1.0, float("inf"), 0
    history = []
    started = time.monotonic()
    print(f"[heads] start {trial.name} {configuration}", flush=True)
    for epoch in range(1, trial.max_epochs + 1):
        model.network.train()
        order = rng.permutation(data.fit_indices)
        total_loss = 0.0
        for start in range(0, len(order), 256):
            selected_array = order[start:start + 256]
            selected = torch.as_tensor(selected_array, dtype=torch.long, device=data.device)
            features = data.direct.index_select(0, selected)
            batch_labels = data.labels[selected_array].copy()
            # Generate the mask even without augmentation to hold permutations constant.
            mask = rng.random(len(selected_array)) < 0.5
            if trial.swap_augmentation:
                gpu_mask = torch.as_tensor(mask, dtype=torch.bool, device=data.device)
                features[gpu_mask] = data.swapped.index_select(0, selected[gpu_mask])
                batch_labels[mask] = np.asarray([1, 0, 2])[batch_labels[mask]]
            if not trial.numeric_features:
                features = features[:, :2688]
            expected = torch.as_tensor(batch_labels, dtype=torch.long, device=data.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.network(features), expected)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss in {trial.name}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(selected_array)
        probabilities = data.predict(model, data.tune_indices)
        tune = extended_metrics(data.labels[data.tune_indices], probabilities)
        row = {"epoch": epoch, "train_online_loss": total_loss / len(data.fit_indices),
               "learning_rate": optimizer.param_groups[0]["lr"], "tune": tune}
        history.append(row)
        if tune["log_loss"] < best_loss - 1e-8:
            best_loss, best_epoch, stale = tune["log_loss"], epoch, 0
            model.save(directory / "best.pt", {"phase": "fit_only_head_selection", "epoch": epoch,
                                               "identity": identity})
            np.save(directory / "best_tune_probabilities.npy", probabilities)
            write_predictions([data.samples[int(i)] for i in data.tune_indices], probabilities,
                              directory / "best_tune_predictions.csv")
        else:
            stale += 1
        if (tune["accuracy"] > best_accuracy or
                (tune["accuracy"] == best_accuracy and tune["log_loss"] < best_accuracy_loss)):
            best_accuracy, best_accuracy_loss, best_accuracy_epoch = tune["accuracy"], tune["log_loss"], epoch
            model.save(directory / "best_accuracy.pt", {
                "phase": "fit_only_accuracy_secondary_selection", "epoch": epoch,
                "tie_break": "lower tune Log Loss", "identity": identity})
            np.save(directory / "best_accuracy_tune_probabilities.npy", probabilities)
            write_predictions([data.samples[int(i)] for i in data.tune_indices], probabilities,
                              directory / "best_accuracy_tune_predictions.csv")
        if scheduler is not None:
            scheduler.step(tune["log_loss"])
        write_json(directory / "progress.json", {"status": "running", "identity": identity,
                   "best_epoch": best_epoch, "epochs": history})
        print(f"[heads] {trial.name} epoch={epoch}/{trial.max_epochs} "
              f"train={row['train_online_loss']:.9f} tune={tune['log_loss']:.9f} "
              f"acc={tune['accuracy']:.6f} macro_f1={tune['macro_f1']:.6f} "
              f"best={best_epoch} stale={stale}", flush=True)
        if stale >= trial.patience:
            break
    del model
    best = load_head(directory / "best.pt", data.device)
    fit_probabilities = data.predict(best, data.fit_indices)
    tune_probabilities = data.predict(best, data.tune_indices)
    saved = np.load(directory / "best_tune_probabilities.npy")
    max_difference = float(np.max(np.abs(saved - tune_probabilities)))
    if max_difference != 0.0:
        raise RuntimeError("Checkpoint reload did not reproduce saved best probabilities")
    accuracy_best = load_head(directory / "best_accuracy.pt", data.device)
    accuracy_fit_probabilities = data.predict(accuracy_best, data.fit_indices)
    accuracy_tune_probabilities = data.predict(accuracy_best, data.tune_indices)
    accuracy_difference = float(np.max(np.abs(
        np.load(directory / "best_accuracy_tune_probabilities.npy") - accuracy_tune_probabilities)))
    if accuracy_difference != 0.0:
        raise RuntimeError("Accuracy checkpoint reload did not reproduce saved probabilities")
    result = {"status": "complete", "identity": identity, "name": trial.name,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "selected_epoch": best_epoch, "epochs_run": len(history), "epochs": history,
              "elapsed_seconds": time.monotonic() - started,
              "parameter_count": sum(value.numel() for value in best.network.parameters()),
              "best_checkpoint_metrics": {
                  "fit_eval": extended_metrics(data.labels[data.fit_indices], fit_probabilities),
                  "tune_eval": extended_metrics(data.labels[data.tune_indices], tune_probabilities)},
              "checkpoint_reload_max_abs_difference": max_difference,
              "selected_accuracy_epoch": best_accuracy_epoch,
              "best_accuracy_checkpoint_metrics": {
                  "fit_eval": extended_metrics(data.labels[data.fit_indices], accuracy_fit_probabilities),
                  "tune_eval": extended_metrics(data.labels[data.tune_indices], accuracy_tune_probabilities)},
              "accuracy_checkpoint_reload_max_abs_difference": accuracy_difference,
              "official_dev_loaded": False, "calibration_used_for_selection": False,
              "split_counts": data.split_counts,
              "artifact_sha256": {name: sha256(directory / name) for name in
                  ("best.pt", "best_tune_probabilities.npy", "best_tune_predictions.csv",
                   "best_accuracy.pt", "best_accuracy_tune_probabilities.npy", "best_accuracy_tune_predictions.csv")}}
    write_json(result_path, result)
    write_json(directory / "progress.json", {"status": "complete", "best_epoch": best_epoch,
                                              "epochs_run": len(history)})
    print(f"[heads] complete {trial.name}: epoch={best_epoch} "
          f"tune={best_loss:.9f} elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    return result


def aggregate(results: list[dict]) -> list[dict]:
    groups = {}
    for result in results:
        config = dict(result["identity"]["configuration"])
        del config["seed"]
        groups.setdefault(json.dumps(config, sort_keys=True), []).append(result)
    rows = []
    for key, trials in groups.items():
        row = {"configuration": json.loads(key), "seeds": sorted(
            trial["identity"]["configuration"]["seed"] for trial in trials),
            "trials": [trial["name"] for trial in trials],
            "selected_epochs": [trial["selected_epoch"] for trial in trials],
            "selected_accuracy_epochs": [trial["selected_accuracy_epoch"] for trial in trials],
            "elapsed_seconds": sum(trial["elapsed_seconds"] for trial in trials),
            "parameter_count": trials[0]["parameter_count"]}
        for name in ("log_loss", "accuracy", "macro_f1", "brier_sum"):
            values = [trial["best_checkpoint_metrics"]["tune_eval"][name] for trial in trials]
            row[name + "_mean"] = statistics.mean(values)
            row[name + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            accuracy_values = [trial["best_accuracy_checkpoint_metrics"]["tune_eval"][name] for trial in trials]
            row["accuracy_checkpoint_" + name + "_mean"] = statistics.mean(accuracy_values)
            row["accuracy_checkpoint_" + name + "_std"] = statistics.stdev(accuracy_values) if len(accuracy_values) > 1 else 0.0
        row["recall"] = {}
        for label in LABELS:
            values = [trial["best_checkpoint_metrics"]["tune_eval"]["per_class"][label]["recall"]
                      for trial in trials]
            row["recall"][label] = {"mean": statistics.mean(values),
                                    "std": statistics.stdev(values) if len(values) > 1 else 0.0}
        rows.append(row)
    return sorted(rows, key=lambda row: (row["log_loss_mean"], row["configuration"]["head"]))


def summarize(output: Path, sources: dict) -> dict:
    import numpy as np
    from th02.evaluation import write_predictions

    results = [json.loads(path.read_text()) for path in sorted((output / "trials").glob("*/metrics.json"))]
    if any(result["identity"]["sources"] != sources for result in results):
        raise ValueError("Cannot aggregate trials from differing source/data identities")
    groups = aggregate(results)
    main = [row for row in groups if row["seeds"] == list(SEEDS)
            and row["configuration"]["numeric_features"]
            and row["configuration"]["swap_augmentation"]]
    phase1 = [row for row in main if row["configuration"]["learning_rate"] == 1e-3
              and not row["configuration"]["scheduler"]]
    result = {"status": "complete" if len(phase1) == 4 and len(main) >= 6 else "partial",
              "official_dev_loaded": False, "calibration_used_for_selection": False,
              "selection_metric": "three-seed mean best held-out tune Log Loss",
              "groups": groups, "completed_trials": len(results), "sources": sources,
              "phase1_selected": min(phase1, key=lambda row: row["log_loss_mean"]) if phase1 else None,
              "selected": min(main, key=lambda row: row["log_loss_mean"]) if main else None,
              "highest_accuracy_group": max(main, key=lambda row: (
                  row["accuracy_checkpoint_accuracy_mean"], -row["accuracy_checkpoint_log_loss_mean"])) if main else None,
              "training_seed_for_final_refit": SEEDS[0],
              "final_refit_epoch_rule": "median best epoch across the three preregistered training seeds",
              "label_order": list(LABELS)}
    if result["selected"]:
        selected = result["selected"]
        result["final_refit_epochs"] = int(statistics.median(selected["selected_epochs"]))
        probabilities = np.mean([np.load(output / "trials" / name / "best_tune_probabilities.npy")
                                  for name in selected["trials"]], axis=0)
        np.save(output / "selected_tune_ensemble_probabilities.npy", probabilities)
        # Preserve IDs by using the stored prediction file as the source order.
        import csv
        from types import SimpleNamespace
        with (output / "trials" / selected["trials"][0] / "best_tune_predictions.csv").open() as stream:
            samples = [SimpleNamespace(id=row["id"]) for row in csv.DictReader(stream)]
        write_predictions(samples, probabilities, output / "selected_tune_ensemble_predictions.csv")
        result["selected_tune_ensemble_path"] = "selected_tune_ensemble_predictions.csv"
        accuracy = result["highest_accuracy_group"]
        result["final_accuracy_refit_epochs"] = int(statistics.median(accuracy["selected_accuracy_epochs"]))
        accuracy_probabilities = np.mean([np.load(output / "trials" / name /
                                       "best_accuracy_tune_probabilities.npy")
                                       for name in accuracy["trials"]], axis=0)
        np.save(output / "accuracy_tune_ensemble_probabilities.npy", accuracy_probabilities)
        write_predictions(samples, accuracy_probabilities, output / "accuracy_tune_ensemble_predictions.csv")
    write_json(output / "summary.json", result)
    lines = ["# 冻结 Encoder 分类头受控实验", "",
             "复用原始 MiniLM embedding、固定 fit/tune/calibration ID。分类头只使用 fit 训练，"
             "tune 选配置及轮数；未读取官方 dev，未使用 calibration 标签调参。", "",
             "每项均为三个预先固定 seed 的最佳 tune checkpoint 均值 ± 样本标准差。"
             "主指标为三分类 Log Loss；Accuracy 并非选型指标。", "",
             "| Head | LR | Numeric | Train swap | Schedule | Params | Best NLL epochs | Tune Log Loss | NLL ckpt Accuracy | Macro-F1 | Brier | Best Accuracy ckpt |",
             "|---|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in groups:
        cfg = row["configuration"]
        lines.append(f"| {cfg['head']} | {cfg['learning_rate']:g} | {cfg['numeric_features']} | "
                     f"{cfg['swap_augmentation']} | {cfg['scheduler']} | {row['parameter_count']:,} | "
                     f"{row['selected_epochs']} | {row['log_loss_mean']:.6f} ± {row['log_loss_std']:.6f} | "
                     f"{row['accuracy_mean']:.4%} ± {row['accuracy_std']:.4%} | "
                     f"{row['macro_f1_mean']:.6f} ± {row['macro_f1_std']:.6f} | "
                     f"{row['brier_sum_mean']:.6f} ± {row['brier_sum_std']:.6f} | "
                     f"{row['accuracy_checkpoint_accuracy_mean']:.4%} ± {row['accuracy_checkpoint_accuracy_std']:.4%} |")
    if result["selected"]:
        selected = result["selected"]
        accuracy = result["highest_accuracy_group"]
        lines += ["", f"当前按 Log Loss 优选：`{selected['configuration']}`。",
                  f"另报 Accuracy 最高配置：`{accuracy['configuration']}`，"
                  "不以此更改主指标选型。",
                  f"Accuracy 次目标按每个 seed 的最高 tune Accuracy 选轮（同 Accuracy 选较低 Log Loss），"
                  f"再以三 seed 均值比较；其重训轮数中位数为 `{result['final_accuracy_refit_epochs']}`。",
                  f"最终固定种子 `{SEEDS[0]}`，fit+tune 重训轮数按预定中位数规则为 "
                  f"`{result['final_refit_epochs']}`。", "",
                  "各轮 tune Accuracy、Macro-F1、各类 Precision/Recall/F1、Brier、在线训练 Loss "
                  "见各 trial 的 `metrics.json`。`best.pt` 保存最佳轮而非最后轮；"
                  "每个 checkpoint 的重载预测差值必须为 0。", "",
                  "数值特征消融移除全部 18 个数值项（15 个文本统计、3 个 cosine）；"
                  "交换增强消融只移除训练时的交换，推理仍对称。两种消融均不参与主配置筛选。",
                  "在线训练 Loss 包含 dropout 与移动权重，不能与固定模型的 tune Loss 直接相减。",
                  "官方 dev 已在历史实验中查看，后续 dev 结果应如实标记为重复使用给定开发集。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("capacity", "learning-rate", "ablations", "scheduler", "all"),
                        default="all")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts-improved-heads")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    production = (ROOT / "artifacts").resolve()
    if output == production or production in output.parents or output in production.parents:
        raise ValueError("Experiment output must be separate from original artifacts")
    configure_process(args.gpu)
    import torch
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the controlled server experiment")
    output.mkdir(parents=True, exist_ok=True)
    # An advisory lock prevents overlapping processes from duplicating trial writes.
    import fcntl
    with (output / ".study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data = ExperimentData("cuda")
        write_json(output / "study.json", {"sources": data.sources, "seeds": list(SEEDS),
                   "split_counts": data.split_counts, "official_dev_loaded": False,
                   "gpu": args.gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(), "process_thp_disabled": True,
                   "epochs": 50, "patience": 8, "feature_cache": "two-view FP32 GPU cache",
                   "selection_metric": "mean three-seed tune Log Loss", "label_order": list(LABELS)})
        if args.phase in ("capacity", "all"):
            for head in HEADS:
                for seed in SEEDS:
                    run_trial(data, Trial(head, seed), output)
                    summarize(output, data.sources)
        summary = summarize(output, data.sources)
        if args.phase in ("learning-rate", "all"):
            if summary["phase1_selected"] is None or len([row for row in summary["groups"]
                    if row["configuration"]["learning_rate"] == 1e-3]) < 4:
                raise ValueError("Complete the four capacity configurations before learning-rate selection")
            head = summary["phase1_selected"]["configuration"]["head"]
            for rate in (3e-4, 1e-4):
                for seed in SEEDS:
                    run_trial(data, Trial(head, seed, learning_rate=rate), output)
                    summarize(output, data.sources)
        summary = summarize(output, data.sources)
        if args.phase in ("ablations", "scheduler"):
            if summary["status"] != "complete":
                raise ValueError("Complete capacity and learning-rate studies before ablations/scheduler")
            selected = summary["selected"]["configuration"]
            for seed in SEEDS:
                if args.phase == "scheduler":
                    trials = [Trial(selected["head"], seed, selected["learning_rate"], scheduler=True)]
                else:
                    trials = [Trial(selected["head"], seed, selected["learning_rate"], numeric_features=False),
                              Trial(selected["head"], seed, selected["learning_rate"], swap_augmentation=False)]
                for trial in trials:
                    run_trial(data, trial, output)
                    summarize(output, data.sources)
        final = summarize(output, data.sources)
        print(json.dumps({"status": final["status"], "completed_trials": final["completed_trials"],
                          "selected": final["selected"], "final_refit_epochs": final.get("final_refit_epochs")},
                         ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
