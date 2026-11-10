#!/usr/bin/env python3
"""Replay the original MLP selection stage to recover held-out tune metrics.

This audit does not load official dev data or alter the production artifacts.
It observes the existing training loop; it does not search new hyperparameters.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure_process(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "8"
    temporary = ROOT / ".cache/tmp"
    os.environ["TMPDIR"] = str(temporary)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if libc.prctl(42, 0, 0, 0, 0) != 1:  # PR_GET_THP_DISABLE
        raise RuntimeError("Process-local THP setting was not applied")


def report_markdown(result: dict) -> str:
    lines = ["# 原 MLP 选轮过程：事后审计重放", "",
             "本文件补算原日志未保存的 held-out tune 指标，不是新架构搜索。"
             "使用原始 train 的 fit/tune ID、同一 embedding 缓存、原种子与原训练函数；"
             "未读取或评估官方 dev，未覆盖正式模型。", "",
             f"重放状态：`{result['status']}`；逐轮原始 Log Loss 核对容差为 "
             f"`{TOLERANCE:g}`，最大差值为 `{result['max_abs_log_loss_difference']:.12g}`。", "",
             "训练在线 Loss 包含 dropout、随机 A/B 交换及每批更新过程，"
             "与 eval 模式的指标不是同条件绝对值。后面的 fit_eval 和 tune_eval "
             "使用相同的 eval＋A/B 对称预测，便于比较泛化差距。", "",
             "| 轮次 | 在线训练 Loss | tune Log Loss | tune Accuracy | tune Macro-F1 | tune Brier |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in result["epochs"]:
        values = row["tune"]
        lines.append(f"| {row['epoch']} | {row['train_online_loss']:.9f} | "
                     f"{values['log_loss']:.9f} | {values['accuracy']:.6f} | "
                     f"{values['macro_f1']:.6f} | {values['brier_sum']:.9f} |")
    lines += ["", f"按 tune Log Loss 选中第 **{result['selected_epoch']}** 轮。"
              "不是按最高 Accuracy 事后改选。", "",
              "## 最佳选轮模型的同条件评估", "",
              "| 集合 | 样本数 | Log Loss | Accuracy | Macro-F1 | Brier |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, values in result["best_checkpoint_metrics"].items():
        lines.append(f"| {name} | {values['count']} | {values['log_loss']:.9f} | "
                     f"{values['accuracy']:.6f} | {values['macro_f1']:.6f} | "
                     f"{values['brier_sum']:.9f} |")
    lines += ["", "## 最佳选轮模型的 tune 分类别指标", "",
              "| 类别 | Precision | Recall | F1 | Support |",
              "|---|---:|---:|---:|---:|"]
    for name, values in result["best_checkpoint_metrics"]["tune_eval"]["per_class"].items():
        lines.append(f"| {name} | {values['precision']:.6f} | {values['recall']:.6f} | "
                     f"{values['f1']:.6f} | {values['support']} |")
    lines += ["", "逐轮预测在 `predictions/tune_epoch_*.csv`，最佳选轮模型为 "
              "`best_selection.pt`；它仅拟合 fit 子集，不是已并入 tune 重训的正式模型。"
              "每轮分类别指标、来源哈希及一致性检查见 `metrics.json`。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts-tuning-audit")
    parser.add_argument("--gpu", default="3")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    production = ROOT / "artifacts"
    if output == production or production in output.parents or output in production.parents:
        raise ValueError("Audit output must be separate from production artifacts")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty audit output directory to preserve prior evidence")
    output.mkdir(parents=True, exist_ok=True)
    configure_process(args.gpu)
    sys.path.insert(0, str(ROOT))

    import numpy as np
    import torch
    from th02.data import LABELS, load_samples, structural_features, targets
    from th02.evaluation import metrics, symmetrize, write_predictions
    from th02 import semantic

    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the original GPU selection replay")
    reference_path = production / "metrics.json"
    split_path = production / "splits.json"
    metadata_path = production / "cache/train/metadata.json"
    reference = json.loads(reference_path.read_text())
    split_record = json.loads(split_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    state = json.loads((production / "run_state.json").read_text())
    source_record = json.loads((production / "execution.json").read_text())
    protected = [reference_path, split_path, metadata_path, production / "models/semantic.pt",
                 production / "models/calibration.json", production / "run_state.json"]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    train_path = ROOT / "data/train.csv"
    if sha256(train_path) != state["configuration"]["data_sha256"]["train.csv"]:
        raise ValueError("Training data differs from the original experiment")
    for name in ("th02/semantic.py", "th02/data.py", "th02/evaluation.py"):
        if sha256(ROOT / name) != source_record["source_sha256"][name]:
            raise ValueError(f"Training source differs from the original experiment: {name}")
    samples = load_samples(train_path)
    id_to_index = {sample.id: index for index, sample in enumerate(samples)}
    splits = {name: np.asarray([id_to_index[identifier] for identifier in split_record[name]["ids"]],
                              dtype=np.int64)
              for name in ("fit", "tune")}
    if set(splits["fit"]).intersection(splits["tune"]):
        raise ValueError("Fit and tune sample IDs overlap")
    if {samples[int(i)].group_hash for i in splits["fit"]}.intersection(
            samples[int(i)].group_hash for i in splits["tune"]):
        raise ValueError("Fit and tune normalized groups overlap")
    embeddings = np.load(production / "cache/train/embeddings.npy", mmap_mode="r")
    if embeddings.shape != (len(samples), 3, semantic.EMBEDDING_DIM):
        raise ValueError("Embedding cache shape differs")
    if semantic._fingerprint(samples, metadata["model"], "cuda") != metadata["fingerprint"]:
        raise ValueError("Embedding cache fingerprint differs")
    structural = structural_features(samples)
    structural_swapped = structural_features(samples, swapped=True)
    tune_samples = [samples[int(index)] for index in splits["tune"]]
    original_history = reference["semantic_training"]["selection_history"]
    seed = int(reference["seed"])
    observations = []
    holder = {}
    best_loss = float("inf")
    original_predict = semantic.SemanticModel.predict_proba
    original_log_loss = semantic._log_loss

    def extended_metrics(labels, probabilities):
        values = metrics(labels, probabilities)
        for entry in values["per_class"].values():
            precision, recall = entry["precision"], entry["recall"]
            entry["f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values["macro_f1"] = float(np.mean([entry["f1"] for entry in values["per_class"].values()]))
        return values

    def observed_predict(self, *arguments, **keywords):
        holder["model"] = self
        return original_predict(self, *arguments, **keywords)

    def observed_loss(labels, probabilities):
        nonlocal best_loss
        value = original_log_loss(labels, probabilities)
        epoch = len(observations) + 1
        expected = original_history[epoch - 1]["tune_log_loss"] if epoch <= len(original_history) else None
        row = {"epoch": epoch, "tune": extended_metrics(labels, probabilities),
               "original_tune_log_loss": expected,
               "abs_log_loss_difference": abs(value - expected) if expected is not None else None}
        observations.append(row)
        write_predictions(tune_samples, probabilities,
                          output / "predictions" / f"tune_epoch_{epoch:02d}.csv")
        if value < best_loss - 1e-8:
            best_loss = value
            holder["model"].save(output / "best_selection.pt", {
                "phase": "fit_only_selection_replay", "epoch": epoch,
                "seed": seed, "fit_count": len(splits["fit"]), "tune_count": len(splits["tune"]),
            })
        print(f"[audit] epoch={epoch} tune={json.dumps(row['tune'])}", flush=True)
        return value

    with patch.object(semantic.SemanticModel, "predict_proba", observed_predict), \
            patch.object(semantic, "_log_loss", observed_loss):
        _, history, selected_epoch = semantic._train_head(
            samples, embeddings, structural, structural_swapped, splits["fit"], seed,
            "cuda", 20, tune_indices=splits["tune"],
        )
    for observation, row in zip(observations, history):
        observation["train_online_loss"] = row["train_log_loss"]
    differences = [row["abs_log_loss_difference"] for row in observations]
    history_matches = (len(history) == len(original_history)
                       and all(value is not None and value <= TOLERANCE for value in differences)
                       and selected_epoch == reference["semantic_training"]["selected_epochs"])
    selected = semantic.load_semantic(output / "best_selection.pt", device="cuda")
    best_metrics = {}
    for name in ("fit", "tune"):
        indices = splits[name]
        batch_samples = [samples[int(index)] for index in indices]
        batch_embeddings = embeddings[indices]
        probabilities = symmetrize(
            selected.predict_proba(batch_samples, batch_embeddings),
            selected.predict_proba(batch_samples, batch_embeddings, swapped=True),
        )
        best_metrics[name + "_eval"] = extended_metrics(targets(batch_samples), probabilities)
    unchanged = all(sha256(ROOT / name) == value for name, value in before.items())
    if not unchanged:
        raise RuntimeError("A protected production artifact changed during the audit")
    result = {
        "status": "reproduced" if history_matches else "replay_mismatch",
        "audit_type": "post_hoc_replay_of_original_selection_not_new_hyperparameter_search",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_dev_loaded": False, "production_artifacts_unchanged": unchanged,
        "seed": seed, "gpu": args.gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
        "architecture": "2706 -> 128 ReLU dropout 0.2 -> 3",
        "selection_history_matches_original": history_matches,
        "log_loss_tolerance": TOLERANCE,
        "max_abs_log_loss_difference": max(value for value in differences if value is not None),
        "selected_epoch": selected_epoch, "epochs": observations,
        "best_checkpoint_metrics": best_metrics, "source_sha256": before,
        "train_sha256": sha256(train_path), "split_counts": {k: len(v) for k, v in splits.items()},
        "label_order": list(LABELS),
    }
    write_json(output / "metrics.json", result)
    (output / "REPORT.md").write_text(report_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected_epoch": selected_epoch,
                      "max_abs_log_loss_difference": result["max_abs_log_loss_difference"],
                      "best_checkpoint_metrics": best_metrics}, ensure_ascii=False), flush=True)
    return 0 if history_matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
