"""Frozen MiniLM embeddings and a small, independently trained semantic branch."""

from __future__ import annotations

import gc
import hashlib
import json
import os
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


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
EMBEDDING_DIM = 384
MAX_TOKENS = 256
INPUT_DIM = 7 * EMBEDDING_DIM + 18
FIELD_NAMES = ("prompt", "response_a", "response_b")
LABEL_ORDER = ("winner_model_a", "winner_model_b", "winner_tie")
SWAP_LABELS = np.asarray([1, 0, 2], dtype=np.int64)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _resolve_model_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    revision = MODEL_REVISION if str(model_path) == MODEL_ID else None
    return Path(snapshot_download(repo_id=str(model_path), revision=revision)).resolve()


def _model_identity(path: Path) -> dict:
    """Hash actual files so different local weights cannot share an embedding cache."""
    names = (
        "config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
    )
    digest = hashlib.sha256()
    files = []
    for name in names:
        file_path = path / name
        if not file_path.is_file():
            continue
        digest.update(name.encode("utf-8"))
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(name)
    if "config.json" not in files:
        raise ValueError(f"Model configuration is missing in {path}")
    return {"path": str(path), "revision": path.name, "sha256": digest.hexdigest(), "files": files}


def _fingerprint(samples: Sequence[Sample], model: dict, device: str) -> str:
    digest = hashlib.sha256()
    config = {
        "algorithm": "head-tail-256-masked-mean-fp32-l2-v1",
        "model_sha256": model["sha256"],
        "precision": "fp16" if torch.device(device).type == "cuda" else "fp32",
        "shape": [len(samples), 3, EMBEDDING_DIM],
    }
    digest.update(json.dumps(config, sort_keys=True).encode("utf-8"))
    for sample in samples:
        digest.update(json.dumps([sample.id, *sample.texts], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_encoder(model_path: Path, device: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    if encoder.config.hidden_size != EMBEDDING_DIM:
        raise ValueError(f"Expected {EMBEDDING_DIM} embedding dimensions")
    encoder.requires_grad_(False)
    encoder.eval().to(device)
    if torch.device(device).type == "cuda":
        encoder.half()
    return tokenizer, encoder


def _tokenize_batch(tokenizer, texts: Sequence[str]):
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_budget = MAX_TOKENS - special_count
    if content_budget < 1:
        raise ValueError("Tokenizer has too many special tokens")
    encoded = []
    truncated = []
    for text in texts:
        token_ids = tokenizer(
            text, add_special_tokens=False, truncation=False,
            return_attention_mask=False, return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        is_truncated = len(token_ids) > content_budget
        if is_truncated:
            head = (content_budget + 1) // 2
            tail = content_budget - head
            token_ids = token_ids[:head] + token_ids[-tail:] if tail else token_ids[:head]
        encoded.append(tokenizer.prepare_for_model(
            token_ids, add_special_tokens=True, return_attention_mask=True, truncation=False,
        ))
        truncated.append(is_truncated)
    return tokenizer.pad(encoded, padding=True, return_tensors="pt"), truncated


def _encode_batch(encoder, token_batch: dict, device: str) -> np.ndarray:
    token_batch = {name: value.to(device) for name, value in token_batch.items()}
    with torch.inference_mode():
        hidden = encoder(**token_batch).last_hidden_state.float()
        mask = token_batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = F.normalize(pooled, p=2, dim=1)
    result = pooled.cpu().numpy().astype(np.float32, copy=False)
    if result.shape[1] != EMBEDDING_DIM or not np.isfinite(result).all():
        raise ValueError("Encoder returned invalid embeddings")
    return result


def encode_samples(
    samples: Sequence[Sample], cache_dir: str | Path, model_path: str | Path,
    device: str = "cuda", batch_size: int = 32,
):
    """Encode fields separately, resuming only durably flushed cache batches."""
    if not samples:
        raise ValueError("Cannot encode an empty dataset")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the selected semantic encoding device")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_directory = _resolve_model_path(model_path)
    identity = _model_identity(model_directory)
    fingerprint = _fingerprint(samples, identity, device)
    shape = (len(samples), 3, EMBEDDING_DIM)
    array_path = cache_dir / "embeddings.npy"
    metadata_path = cache_dir / "metadata.json"
    progress_path = cache_dir / "progress.json"

    def compatible_state(path):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            array = np.load(array_path, mmap_mode="r")
            valid = state.get("fingerprint") == fingerprint and array.shape == shape
            valid = valid and array.dtype == np.float32
            del array
            return state if valid else None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    complete = compatible_state(metadata_path)
    if complete and complete.get("completed_fields") == len(samples) * 3:
        print(f"[semantic] reuse complete embedding cache {cache_dir}", flush=True)
        return np.load(array_path, mmap_mode="r"), complete
    state = compatible_state(progress_path)
    if state:
        completed = int(state.get("completed_fields", 0))
        if not 0 <= completed <= len(samples) * 3:
            state = None
    if state:
        array = np.load(array_path, mmap_mode="r+")
        truncated_ids = state["truncated_ids"]
        completed = int(state["completed_fields"])
    else:
        array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32, shape=shape)
        truncated_ids = {name: [] for name in FIELD_NAMES}
        completed = 0
        if metadata_path.exists():
            metadata_path.unlink()
    resumed_from = completed
    flattened = array.reshape(-1, EMBEDDING_DIM)
    tokenizer, encoder = _load_encoder(model_directory, device)
    start_time = time.monotonic()
    last_log = start_time
    active_batch_size = batch_size
    try:
        while completed < len(samples) * 3:
            stop = min(completed + active_batch_size, len(samples) * 3)
            positions = list(range(completed, stop))
            texts = [samples[position // 3].texts[position % 3] for position in positions]
            token_batch, truncated = _tokenize_batch(tokenizer, texts)
            try:
                encoded = _encode_batch(encoder, token_batch, device)
            except torch.cuda.OutOfMemoryError:
                if torch.device(device).type != "cuda" or active_batch_size <= 8:
                    raise
                active_batch_size = 16 if active_batch_size > 16 else 8
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[semantic] CUDA OOM, retry batch_size={active_batch_size}", flush=True)
                continue
            flattened[completed:stop] = encoded
            array.flush()
            for position, was_truncated in zip(positions, truncated):
                if was_truncated:
                    truncated_ids[FIELD_NAMES[position % 3]].append(samples[position // 3].id)
            completed = stop
            state = {
                "fingerprint": fingerprint, "shape": list(shape), "model": identity,
                "completed_fields": completed, "total_fields": len(samples) * 3,
                "truncated_ids": truncated_ids, "batch_size": active_batch_size,
                "max_tokens": MAX_TOKENS, "pooling": "masked mean FP32, L2 normalized",
                "precision": "fp16" if torch.device(device).type == "cuda" else "fp32",
            }
            _write_json(progress_path, state)
            now = time.monotonic()
            if now - last_log >= 10 or completed == len(samples) * 3:
                print(f"[semantic] encoded {completed}/{len(samples) * 3} fields", flush=True)
                last_log = now
        state["resumed_from_fields"] = resumed_from
        state["elapsed_seconds_this_call"] = time.monotonic() - start_time
        state["truncation_rates"] = {
            name: len(ids) / len(samples) for name, ids in truncated_ids.items()
        }
        _write_json(metadata_path, state)
    finally:
        del encoder
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return np.load(array_path, mmap_mode="r"), state


def _numeric_features(embeddings: np.ndarray, structural: np.ndarray) -> np.ndarray:
    prompt, answer_a, answer_b = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
    cosines = np.stack([
        np.einsum("ij,ij->i", prompt, answer_a),
        np.einsum("ij,ij->i", prompt, answer_b),
        np.einsum("ij,ij->i", answer_a, answer_b),
    ], axis=1)
    return np.concatenate([structural, cosines], axis=1).astype(np.float32, copy=False)


def _batch_features(
    embeddings: np.ndarray, structural: np.ndarray, scaler: StandardScaler, device: str,
) -> torch.Tensor:
    numeric = scaler.transform(_numeric_features(embeddings, structural)).astype(np.float32)
    vectors = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    prompt, answer_a, answer_b = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return torch.cat([
        prompt, answer_a, answer_b, answer_a - answer_b, (answer_a - answer_b).abs(),
        prompt * answer_a, prompt * answer_b,
        torch.as_tensor(numeric, dtype=torch.float32, device=device),
    ], dim=1)


def _fit_scaler(embeddings, structural, structural_swapped, indices) -> StandardScaler:
    scaler = StandardScaler()
    for start in range(0, len(indices), 256):
        selected = indices[start:start + 256]
        vectors = np.array(embeddings[selected], dtype=np.float32, copy=True)
        scaler.partial_fit(_numeric_features(vectors, structural[selected]))
        vectors[:, [1, 2]] = vectors[:, [2, 1]]
        scaler.partial_fit(_numeric_features(vectors, structural_swapped[selected]))
    return scaler


def _new_network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3),
    )


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class SemanticModel:
    def __init__(self, network: nn.Module, scaler: StandardScaler, device: str = "cuda"):
        self.network = network.to(device)
        self.scaler = scaler
        self.device = device

    def predict_proba(
        self, samples: Sequence[Sample], embeddings: np.ndarray,
        swapped: bool = False, batch_size: int = 256,
    ) -> np.ndarray:
        if len(samples) != len(embeddings):
            raise ValueError("Samples and embedding lengths differ")
        self.network.eval()
        structural = structural_features(samples, swapped=swapped)
        output = np.empty((len(samples), 3), dtype=np.float64)
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                stop = min(start + batch_size, len(samples))
                vectors = np.array(embeddings[start:stop], dtype=np.float32, copy=True)
                if swapped:
                    vectors[:, [1, 2]] = vectors[:, [2, 1]]
                features = _batch_features(vectors, structural[start:stop], self.scaler, self.device)
                output[start:stop] = self.network(features).softmax(dim=1).cpu().numpy()
        output /= output.sum(axis=1, keepdims=True)
        return output

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": {key: value.detach().cpu() for key, value in self.network.state_dict().items()},
            "scaler": {
                "mean": self.scaler.mean_.tolist(), "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(), "n_samples_seen": int(self.scaler.n_samples_seen_),
            },
            "config": {"input_dim": INPUT_DIM, "hidden_dim": 128, "dropout": 0.2},
            "label_order": list(LABEL_ORDER), "metadata": metadata or {},
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)


def load_semantic(path: str | Path, device: str = "cuda") -> SemanticModel:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["config"]["input_dim"] != INPUT_DIM or payload["label_order"] != list(LABEL_ORDER):
        raise ValueError("Semantic checkpoint configuration differs from this implementation")
    scaler = StandardScaler()
    for name in ("mean", "scale", "var"):
        setattr(scaler, name + "_", np.asarray(payload["scaler"][name], dtype=np.float64))
    scaler.n_features_in_ = 18
    scaler.n_samples_seen_ = payload["scaler"]["n_samples_seen"]
    network = _new_network()
    network.load_state_dict(payload["state_dict"])
    return SemanticModel(network, scaler, device)


def _log_loss(labels, probabilities):
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1)).mean())


def _train_head(
    samples, embeddings, structural, structural_swapped, fit_indices, seed, device,
    epochs, tune_indices=None,
):
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    scaler = _fit_scaler(embeddings, structural, structural_swapped, fit_indices)
    model = SemanticModel(_new_network(), scaler, device)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)
    labels = targets(samples)
    history = []
    best_loss = float("inf")
    best_epoch = 1
    patience = 0
    for epoch in range(1, epochs + 1):
        model.network.train()
        total_loss = 0.0
        order = rng.permutation(fit_indices)
        for start in range(0, len(order), 256):
            selected = order[start:start + 256]
            vectors = np.array(embeddings[selected], dtype=np.float32, copy=True)
            numeric = structural[selected].copy()
            batch_labels = labels[selected].copy()
            swap_mask = rng.random(len(selected)) < 0.5
            vectors[swap_mask, 1], vectors[swap_mask, 2] = (
                vectors[swap_mask, 2].copy(), vectors[swap_mask, 1].copy(),
            )
            numeric[swap_mask] = structural_swapped[selected[swap_mask]]
            batch_labels[swap_mask] = SWAP_LABELS[batch_labels[swap_mask]]
            features = _batch_features(vectors, numeric, scaler, device)
            expected = torch.as_tensor(batch_labels, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.network(features), expected)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite semantic training loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(selected)
        record = {"epoch": epoch, "train_log_loss": total_loss / len(fit_indices)}
        if tune_indices is not None:
            tune_samples = [samples[int(index)] for index in tune_indices]
            tune_vectors = embeddings[tune_indices]
            direct = model.predict_proba(tune_samples, tune_vectors)
            swapped = model.predict_proba(tune_samples, tune_vectors, swapped=True)[:, SWAP_LABELS]
            validation_loss = _log_loss(labels[tune_indices], (direct + swapped) / 2)
            record["tune_log_loss"] = validation_loss
            if validation_loss < best_loss - 1e-8:
                best_loss, best_epoch, patience = validation_loss, epoch, 0
            else:
                patience += 1
        history.append(record)
        print(f"[semantic] epoch {epoch}/{epochs} {json.dumps(record)}", flush=True)
        if tune_indices is not None and patience >= 3:
            break
    return model, history, best_epoch


def train_semantic(
    samples: Sequence[Sample], embeddings: np.ndarray, splits: dict,
    output_dir: str | Path, seed: int, device: str = "cuda", max_epochs: int = 20,
):
    if max_epochs < 1 or max_epochs > 20:
        raise ValueError("max_epochs must be between 1 and 20")
    if len(samples) != len(embeddings) or embeddings.shape[1:] != (3, EMBEDDING_DIM):
        raise ValueError("Expected embeddings shaped (number_of_samples, 3, 384)")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the selected semantic training device")
    if (targets(samples) < 0).any():
        raise ValueError("Semantic training requires labels for every sample")
    structural = structural_features(samples)
    structural_swapped = structural_features(samples, swapped=True)
    start_time = time.monotonic()
    selection, selection_history, best_epoch = _train_head(
        samples, embeddings, structural, structural_swapped, splits["fit"], seed,
        device, max_epochs, tune_indices=splits["tune"],
    )
    del selection
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    final_model, refit_history, _ = _train_head(
        samples, embeddings, structural, structural_swapped, splits["refit"], seed,
        device, best_epoch,
    )
    metadata = {
        "selected_epochs": best_epoch, "selection_history": selection_history,
        "refit_history": refit_history, "seed": seed, "device": device,
        "fit_count": len(splits["fit"]), "tune_count": len(splits["tune"]),
        "refit_count": len(splits["refit"]), "elapsed_seconds": time.monotonic() - start_time,
        "architecture": "2706 -> 128 ReLU dropout 0.2 -> 3",
        "optimizer": {"name": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 256},
        "augmentation": "independent 50% A/B exchange per training sample and epoch",
        "scaler_fit": "refit indices, original and exchanged order, 18 numeric features only",
    }
    final_model.save(Path(output_dir) / "models" / "semantic.pt", metadata)
    return final_model, metadata
