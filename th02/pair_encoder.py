"""Joint question/answer MiniLM fine tuning with an A/B symmetric three-class head.

This candidate uses the existing immutable fit/tune IDs. It does not select any
hyperparameter on the official dev set. Checkpoints include the fitted feature
scaler and complete encoder weights, so inference does not need training data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from .data import Sample, structural_features, targets
from .evaluation import metrics, validate_probabilities, write_predictions


@dataclass(frozen=True)
class PairConfig:
    max_length: int = 384
    prompt_budget: int = 96
    hidden_dim: int = 256
    dropout: float = 0.2
    encoder_lr: float = 2e-5
    head_lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    accumulation: int = 4
    epochs: int = 5
    patience: int = 2
    warmup_fraction: float = 0.1
    seed: int = 20260916


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def head_tail(ids: list[int], budget: int) -> list[int]:
    if budget < 0:
        raise ValueError("A token budget cannot be negative")
    if len(ids) <= budget:
        return ids
    head = (budget + 1) // 2
    tail = budget - head
    return ids[:head] + (ids[-tail:] if tail else [])


def pair_token_ids(tokenizer, prompt_ids: list[int], answer_ids: list[int],
                   config: PairConfig) -> tuple[dict, dict]:
    """Explicit budgets; short prompts return unused capacity to the answer."""
    content = config.max_length - tokenizer.num_special_tokens_to_add(pair=True)
    if not 1 <= config.prompt_budget < content:
        raise ValueError("prompt_budget must be positive and leave room for an answer")
    q_budget = min(config.prompt_budget, len(prompt_ids))
    a_budget = min(len(answer_ids), content - q_budget)
    q_budget = min(len(prompt_ids), content - a_budget)
    encoded = tokenizer.prepare_for_model(
        head_tail(prompt_ids, q_budget), pair_ids=head_tail(answer_ids, a_budget),
        add_special_tokens=True, truncation=False, padding="max_length",
        max_length=config.max_length, return_attention_mask=True,
        return_token_type_ids=True, verbose=False,
    )
    return encoded, {"prompt_truncated": len(prompt_ids) > q_budget,
                     "answer_truncated": len(answer_ids) > a_budget}


class TokenCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.arrays = {key: np.load(self.directory / f"{key}.npy", mmap_mode="r")
                       for key in ("input_ids", "attention_mask", "token_type_ids")}

    def batch(self, indices, device: str) -> dict[str, torch.Tensor]:
        return {key: torch.as_tensor(np.asarray(array[indices]), dtype=torch.long,
                                     device=device).reshape(-1, array.shape[-1])
                for key, array in self.arrays.items()}


def prepare_token_cache(samples: Sequence[Sample], model_path: str | Path,
                        directory: str | Path, config: PairConfig) -> TokenCache:
    """CPU-only cache with a full text/tokenizer/config fingerprint."""
    from transformers import AutoTokenizer

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    fingerprint = hashlib.sha256()
    fingerprint.update(json.dumps({"version": 1, "max_length": config.max_length,
                                   "prompt_budget": config.prompt_budget},
                                  sort_keys=True).encode())
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        path = model_path / name
        if path.exists():
            fingerprint.update(name.encode())
            fingerprint.update(path.read_bytes())
    for sample in samples:
        fingerprint.update(json.dumps([sample.id, *sample.texts],
                                      ensure_ascii=False).encode())
    expected = fingerprint.hexdigest()
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        cache = TokenCache(directory)
        if cache.metadata.get("fingerprint") == expected:
            if all(array.shape == (len(samples), 2, config.max_length)
                   for array in cache.arrays.values()):
                print(f"[pair] reusing token cache {directory}", flush=True)
                return cache
        raise ValueError("Token cache differs from the requested data/config; use a new directory")
    shape = (len(samples), 2, config.max_length)
    arrays = {key: np.lib.format.open_memmap(directory / f"{key}.npy", mode="w+",
                                            dtype=np.int32 if key == "input_ids" else np.uint8,
                                            shape=shape)
              for key in ("input_ids", "attention_mask", "token_type_ids")}
    counts = {"prompt_truncated": 0, "answer_a_truncated": 0, "answer_b_truncated": 0}
    start = time.monotonic()
    for offset in range(0, len(samples), 128):
        batch = samples[offset:offset + 128]
        texts = [text for sample in batch for text in sample.texts]
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False,
                            return_attention_mask=False, return_token_type_ids=False,
                            verbose=False)["input_ids"]
        for local, sample in enumerate(batch):
            prompt, a, b = encoded[3 * local:3 * local + 3]
            for side, answer in enumerate((a, b)):
                values, truncation = pair_token_ids(tokenizer, prompt, answer, config)
                for key, array in arrays.items():
                    array[offset + local, side] = values[key]
                if side == 0:
                    counts["prompt_truncated"] += int(truncation["prompt_truncated"])
                counts[f"answer_{'a' if side == 0 else 'b'}_truncated"] += int(truncation["answer_truncated"])
        if offset % 4096 == 0 or offset + 128 >= len(samples):
            print(f"[pair] tokenized {min(offset + 128, len(samples))}/{len(samples)}", flush=True)
    for array in arrays.values():
        array.flush()
    write_json(metadata_path, {"fingerprint": expected, "count": len(samples),
                               "shape": list(shape), "max_length": config.max_length,
                               "prompt_budget": config.prompt_budget,
                               "truncation_counts": counts,
                               "elapsed_seconds": time.monotonic() - start})
    return TokenCache(directory)


class PairNetwork(nn.Module):
    def __init__(self, model_path: str | Path, config: PairConfig, pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        model_path = str(model_path)
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        else:
            encoder_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            self.encoder = AutoModel.from_config(encoder_config)
        self.side_dim = 2 * self.encoder.config.hidden_size
        self.side_norm = nn.LayerNorm(self.side_dim)
        self.head = nn.Sequential(nn.Linear(5 * self.side_dim + 15, config.hidden_dim),
                                  nn.ReLU(), nn.Dropout(config.dropout),
                                  nn.Linear(config.hidden_dim, 3))

    def encode(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        side = self.side_norm(torch.cat([hidden[:, 0], mean], dim=-1))
        return side.reshape(-1, 2, self.side_dim)

    def classify(self, side: torch.Tensor, numeric: torch.Tensor,
                 swapped: bool | torch.Tensor = False) -> torch.Tensor:
        a, b = side[:, 0], side[:, 1]
        if isinstance(swapped, torch.Tensor):
            selected = swapped.unsqueeze(-1)
            a, b = torch.where(selected, b, a), torch.where(selected, a, b)
        elif swapped:
            a, b = b, a
        return self.head(torch.cat([a, b, a - b, (a - b).abs(), a * b, numeric], dim=-1))


def extended_metrics(y, probabilities) -> dict:
    value = metrics(y, probabilities)
    f1s = []
    for entry in value["per_class"].values():
        p, r = entry["precision"], entry["recall"]
        entry["f1"] = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(entry["f1"])
    value["macro_f1"] = float(np.mean(f1s))
    return value


class PairEncoderModel:
    def __init__(self, network: PairNetwork, scaler: StandardScaler,
                 config: PairConfig, model_path: str | Path, device: str):
        self.network, self.scaler, self.config = network, scaler, config
        self.model_path, self.device = str(model_path), device
        self.network.to(device)

    def predict_cached(self, cache: TokenCache, indices, structural: np.ndarray,
                       swapped_structural: np.ndarray, batch_size: int = 32) -> np.ndarray:
        self.network.eval()
        result = []
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                selected = np.asarray(indices[start:start + batch_size])
                original = torch.as_tensor(self.scaler.transform(structural[selected]),
                                           dtype=torch.float32, device=self.device)
                swapped = torch.as_tensor(self.scaler.transform(swapped_structural[selected]),
                                          dtype=torch.float32, device=self.device)
                with torch.autocast(device_type=torch.device(self.device).type,
                                    dtype=torch.float16, enabled=self.device.startswith("cuda")):
                    side = self.network.encode(cache.batch(selected, self.device))
                    first = self.network.classify(side, original).float().softmax(-1)
                    second = self.network.classify(side, swapped, True).float().softmax(-1)
                    probabilities = (first + second[:, [1, 0, 2]]) * 0.5
                result.append(probabilities.cpu().numpy())
        return validate_probabilities(np.concatenate(result))

    def predict_proba(self, samples: Sequence[Sample], cache_dir: str | Path,
                      batch_size: int = 32) -> np.ndarray:
        cache = prepare_token_cache(samples, self.model_path, cache_dir, self.config)
        return self.predict_cached(cache, np.arange(len(samples)), structural_features(samples),
                                   structural_features(samples, swapped=True), batch_size)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": "th02-pair-encoder-v1", "config": asdict(self.config),
                   "model_path": self.model_path, "state_dict": self.network.state_dict(),
                   "scaler_mean": self.scaler.mean_, "scaler_scale": self.scaler.scale_,
                   "scaler_var": self.scaler.var_, "scaler_samples": self.scaler.n_samples_seen_,
                   "metadata": metadata or {}}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)


def load_pair_encoder(path: str | Path, device: str = "cpu",
                      model_path: str | Path | None = None) -> PairEncoderModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "th02-pair-encoder-v1":
        raise ValueError("Unsupported pair encoder checkpoint")
    config = PairConfig(**payload["config"])
    model_path = model_path or payload["model_path"]
    network = PairNetwork(model_path, config, pretrained=False)
    network.load_state_dict(payload["state_dict"])
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = (payload["scaler_mean"], payload["scaler_scale"],
                                                payload["scaler_var"])
    scaler.n_samples_seen_, scaler.n_features_in_ = payload["scaler_samples"], 15
    return PairEncoderModel(network, scaler, config, model_path, device)


def train_pair_encoder(samples: Sequence[Sample], fit_indices, tune_indices,
                       model_path: str | Path, output_dir: str | Path,
                       config: PairConfig = PairConfig(), device: str = "cuda",
                       resume: bool = False) -> dict:
    """Fit only fit_indices; tune_indices may be empty for fixed-epoch final refit.

    Resume restores model, AdamW, scheduler, AMP scaler, and all RNG states at
    the last completed epoch. A partially completed epoch is deterministically
    replayed. Best selection weights are separate from the resume checkpoint.
    """
    if min(config.batch_size, config.accumulation, config.epochs, config.patience) < 1:
        raise ValueError("Batch, accumulation, epochs and patience must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fit_indices, tune_indices = np.asarray(fit_indices), np.asarray(tune_indices)
    if set(fit_indices).intersection(tune_indices):
        raise ValueError("Fit and tune rows overlap")
    if not len(fit_indices):
        raise ValueError("Fit data cannot be empty")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
    cache = prepare_token_cache(samples, model_path, output / "token_cache", config)
    structural = structural_features(samples)
    swapped_structural = structural_features(samples, swapped=True)
    scaler = StandardScaler().fit(np.concatenate([structural[fit_indices],
                                                  swapped_structural[fit_indices]]))
    network = PairNetwork(model_path, config)
    model = PairEncoderModel(network, scaler, config, model_path, device)
    parameters = [{"params": [], "lr": config.encoder_lr, "weight_decay": config.weight_decay},
                  {"params": [], "lr": config.encoder_lr, "weight_decay": 0.0},
                  {"params": [], "lr": config.head_lr, "weight_decay": config.weight_decay},
                  {"params": [], "lr": config.head_lr, "weight_decay": 0.0}]
    for name, param in network.named_parameters():
        group = 0 if name.startswith("encoder.") else 2
        if param.ndim < 2:
            group += 1
        parameters[group]["params"].append(param)
    optimizer = torch.optim.AdamW(parameters)
    steps_epoch = math.ceil(math.ceil(len(fit_indices) / config.batch_size) / config.accumulation)
    total_steps = steps_epoch * config.epochs
    warmup = max(1, int(total_steps * config.warmup_fraction))
    def multiplier(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    labels = targets(samples)
    numeric = scaler.transform(structural).astype(np.float32)
    numeric_swapped = scaler.transform(swapped_structural).astype(np.float32)
    history, best_loss, best_epoch, start_epoch = [], float("inf"), 0, 1
    resume_path = output / "resume.pt"
    fit_probe = np.sort(np.random.RandomState(20260916).choice(
        fit_indices, size=min(2048, len(fit_indices)), replace=False))
    provenance = {"config": asdict(config), "token_fingerprint": cache.metadata["fingerprint"],
                  "fit_ids": [samples[int(i)].id for i in fit_indices],
                  "tune_ids": [samples[int(i)].id for i in tune_indices],
                  "device": device, "parameter_count": sum(p.numel() for p in network.parameters()),
                  "fit_eval_probe": "2048 fit IDs sampled without replacement, fixed seed20260916; final best uses all fit",
                  "fit_probe_ids": [samples[int(i)].id for i in fit_probe],
                  "selection_metric": "tune three-class Log Loss", "official_dev_loaded": False}
    if resume and resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["provenance"] != provenance:
            raise ValueError("Resume data, config or device differs")
        network.load_state_dict(saved["network"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        amp_scaler.load_state_dict(saved["amp_scaler"])
        history, best_loss, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        start_epoch = saved["epoch"] + 1
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    elif any((output / name).exists() for name in ("best.pt", "metrics.json", "resume.pt")):
        raise ValueError("Output already contains training; pass resume=True or use a new directory")
    write_json(output / "provenance.json", provenance)
    overall_start = time.monotonic()
    active_microbatch = config.batch_size
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.monotonic()
        network.train()
        order = np.random.permutation(fit_indices)
        effective_batch = config.batch_size * config.accumulation
        groups = [order[i:i + effective_batch] for i in range(0, len(order), effective_batch)]
        running_loss, running_count = 0.0, 0
        skipped_amp_updates = 0
        for group_number, group in enumerate(groups):
            # OOM retries replay only the not-yet-updated accumulation group.
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if device.startswith("cuda") else []
            while True:
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                try:
                    for offset in range(0, len(group), active_microbatch):
                        selected = group[offset:offset + active_microbatch]
                        swap = torch.rand(len(selected), device=device) < 0.5
                        original_numeric = torch.as_tensor(numeric[selected], device=device)
                        other_numeric = torch.as_tensor(numeric_swapped[selected], device=device)
                        batch_numeric = torch.where(swap.unsqueeze(-1), other_numeric, original_numeric)
                        batch_labels = torch.as_tensor(labels[selected], dtype=torch.long, device=device)
                        batch_labels = torch.where(swap & (batch_labels != 2), 1 - batch_labels, batch_labels)
                        with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                            enabled=device.startswith("cuda")):
                            side = network.encode(cache.batch(selected, device))
                            logits = network.classify(side, batch_numeric, swap)
                            raw_loss = F.cross_entropy(logits.float(), batch_labels)
                        amp_scaler.scale(raw_loss * len(selected) / len(group)).backward()
                        group_loss += float(raw_loss.detach()) * len(selected)
                    break
                except torch.cuda.OutOfMemoryError:
                    if active_microbatch <= 1:
                        raise
                    active_microbatch = max(1, active_microbatch // 2)
                    optimizer.zero_grad(set_to_none=True)
                    # Drop references to the failed activation graph before empty_cache.
                    side = logits = raw_loss = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state_all(cuda_rng)
                    print(f"[pair] OOM: retry effective batch {len(group)} with microbatch={active_microbatch}", flush=True)
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            previous_scale = amp_scaler.get_scale()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            if amp_scaler.get_scale() >= previous_scale:
                scheduler.step()
            else:
                skipped_amp_updates += 1
            optimizer.zero_grad(set_to_none=True)
            running_loss += group_loss
            running_count += len(group)
            if (group_number + 1) % 100 == 0:
                print(f"[pair] epoch={epoch} update={group_number + 1}/{len(groups)} online_loss={running_loss / running_count:.6f}", flush=True)
        row = {"epoch": epoch, "train_online_loss": running_loss / running_count,
               "active_microbatch": active_microbatch, "effective_batch": effective_batch,
               "skipped_amp_updates": skipped_amp_updates}
        for name, indices in (("fit_probe", fit_probe), ("tune", tune_indices)):
            if not len(indices):
                continue
            probabilities = model.predict_cached(cache, indices, structural, swapped_structural)
            row[name] = extended_metrics(labels[indices], probabilities)
            if name == "tune":
                write_predictions([samples[int(i)] for i in indices], probabilities,
                                  output / "predictions" / f"tune_epoch_{epoch:02d}.csv")
        row["elapsed_seconds"] = time.monotonic() - epoch_start
        history.append(row)
        selection_loss = row.get("tune", row["fit_probe"])["log_loss"]
        if selection_loss < best_loss - 1e-8 or not len(tune_indices):
            best_loss, best_epoch = selection_loss, epoch
            model.save(output / "best.pt", {"epoch": epoch, "metrics": row,
                                            "phase": "selection" if len(tune_indices) else "fixed_epoch_refit"})
        checkpoint = {"provenance": provenance, "epoch": epoch, "history": history,
                      "best_loss": best_loss, "best_epoch": best_epoch,
                      "network": network.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "amp_scaler": amp_scaler.state_dict(),
                      "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []}
        temporary = output / "resume.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(resume_path)
        write_json(output / "metrics.json", {"status": "running", "config": asdict(config),
                                             "epochs": history, "best_epoch": best_epoch,
                                             "best_tune_log_loss": best_loss,
                                             "elapsed_seconds_this_call": time.monotonic() - overall_start})
        print(f"[pair] completed {json.dumps(row)} best_epoch={best_epoch}", flush=True)
        if len(tune_indices) and epoch - best_epoch >= config.patience:
            break
    result = {"status": "complete", "config": asdict(config), "epochs": history,
              "best_epoch": best_epoch, "best_tune_log_loss": best_loss,
              "elapsed_seconds_this_call": time.monotonic() - overall_start}
    best = load_pair_encoder(output / "best.pt", device=device)
    best_fit = best.predict_cached(cache, fit_indices, structural, swapped_structural)
    result["best_fit_full_eval"] = extended_metrics(labels[fit_indices], best_fit)
    if len(tune_indices):
        best_tune = best.predict_cached(cache, tune_indices, structural, swapped_structural)
        result["best_tune_eval"] = extended_metrics(labels[tune_indices], best_tune)
        write_predictions([samples[int(i)] for i in tune_indices], best_tune,
                          output / "predictions" / "tune_best.csv")
        if abs(result["best_tune_eval"]["log_loss"] - best_loss) > 1e-10:
            raise RuntimeError("Reloaded best checkpoint predictions differ")
    result["elapsed_seconds_this_call"] = time.monotonic() - overall_start
    write_json(output / "metrics.json", result)
    return result
