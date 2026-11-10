"""Run the fixed-split TH02 experiment and produce reproducible artifacts."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260916


@contextmanager
def output_lock(directory: Path):
    """Let the OS release the writer lock even after a killed training process."""
    handle = (directory / ".run.lock").open("a+b")
    acquired = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> list[dict]:
    checks = []
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        actual = file_digest(ROOT / relative)
        checks.append({"file": relative, "sha256": actual, "matches": actual == expected})
    if not all(check["matches"] for check in checks):
        raise ValueError("The supplied files do not match MANIFEST.sha256")
    return checks


def inspect_resources() -> dict:
    import psutil

    memory = psutil.virtual_memory()
    cpu_name = platform.processor()
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    result = {
        "cpu": cpu_name,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(),
        "ram_total_gib": memory.total / 2**30,
        "ram_available_gib": memory.available / 2**30,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15,
        )
        name, total, free, driver = next(csv.reader(process.stdout.splitlines()))
        result["gpu"] = {"name": name.strip(), "memory_total_mib": float(total),
                         "memory_free_mib": float(free), "driver": driver.strip()}
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as error:
        result["gpu_error"] = str(error)
    return result


def resolve_encoder(explicit: str | None) -> str:
    if explicit:
        return explicit
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    cached = hub / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshots = cached / "snapshots"
    preferred = snapshots / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    if (preferred / "model.safetensors").is_file():
        return str(preferred)
    if snapshots.is_dir():
        for snapshot in sorted(snapshots.iterdir()):
            if (snapshot / "model.safetensors").is_file():
                return str(snapshot)
    return "sentence-transformers/all-MiniLM-L6-v2"


def configure_randomness(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def write_truth(samples, path: Path) -> None:
    from .data import LABELS

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", *LABELS))
        for sample in samples:
            writer.writerow((sample.id, *(int(sample.label == index) for index in range(3))))


def run_experiment(args, output: Path, resources: dict, manifest: list[dict]) -> None:
    import numpy as np
    import torch

    from .data import load_samples, make_splits, profile_data, targets
    from .evaluation import (metrics, score_official, select_calibration, symmetrize,
                             temperature_scale, write_predictions)
    from .environment import installed_versions
    from .lexical import train_baseline
    from .reporting import write_reports
    from .semantic import encode_samples, train_semantic

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run setup.ps1 to install the locked CUDA environment.")
    resources["torch"] = torch.__version__
    resources["torch_cuda"] = torch.version.cuda
    configure_randomness(args.seed)
    started = time.monotonic()
    timings = {}
    print("Loading and validating data", flush=True)
    limit = 512 if args.smoke else None
    train = load_samples(args.data_dir / "train.csv", limit=limit)
    dev = load_samples(args.data_dir / "dev.csv", limit=128 if args.smoke else None)
    profile = profile_data(train, dev)
    write_json(output / "data_profile.json", profile)
    splits = make_splits(train, args.seed)
    write_json(output / "splits.json", {
        key: {"count": len(indices), "ids": [train[index].id for index in indices]}
        for key, indices in splits.items()
    })
    print("Fitting TF-IDF baseline", flush=True)
    stage = time.monotonic()
    lexical, baseline_training = train_baseline(train, splits, output, args.seed)
    timings["baseline_seconds"] = time.monotonic() - stage
    stage = time.monotonic()
    encoder_path = resolve_encoder(args.model_path)
    print(f"Encoding train with {encoder_path}", flush=True)
    train_embeddings, train_encoding = encode_samples(
        train, output / "cache/train", encoder_path, device=args.device,
        batch_size=8 if args.smoke else 32,
    )
    print("Fitting semantic classification head", flush=True)
    semantic, semantic_training = train_semantic(
        train, train_embeddings, splits, output, args.seed, device=args.device,
        max_epochs=2 if args.smoke else 20,
    )
    timings["semantic_train_seconds"] = time.monotonic() - stage
    calibration_indices = splits["calibration"]
    calibration_samples = [train[index] for index in calibration_indices]
    calibration_embeddings = train_embeddings[calibration_indices]
    lex_raw = lexical.predict_proba(calibration_samples)
    lex_sym = symmetrize(lex_raw, lexical.predict_proba(calibration_samples, swapped=True))
    sem_raw = semantic.predict_proba(calibration_samples, calibration_embeddings)
    sem_sym = symmetrize(
        sem_raw, semantic.predict_proba(calibration_samples, calibration_embeddings, swapped=True),
    )
    calibration = select_calibration(targets(calibration_samples), lex_raw, lex_sym, sem_sym)
    write_json(output / "models/calibration.json", calibration)
    write_json(output / "models/selection_locked.json", {
        "seed": args.seed, "baseline": baseline_training, "semantic": semantic_training,
        "calibration": calibration, "encoder": encoder_path,
        "note": "Selected using official train only, before dev prediction",
    })
    del calibration_embeddings
    print("Configuration locked. Encoding and evaluating dev", flush=True)
    stage = time.monotonic()
    dev_embeddings, dev_encoding = encode_samples(
        dev, output / "cache/dev", encoder_path, device=args.device,
        batch_size=8 if args.smoke else 32,
    )
    raw = lexical.predict_proba(dev)
    symmetric = symmetrize(raw, lexical.predict_proba(dev, swapped=True))
    semantic_raw = semantic.predict_proba(dev, dev_embeddings)
    semantic_symmetric = symmetrize(
        semantic_raw, semantic.predict_proba(dev, dev_embeddings, swapped=True),
    )
    alpha = calibration["alpha"]
    fusion = (1.0 - alpha) * symmetric + alpha * semantic_symmetric
    prior = np.bincount(targets(train), minlength=3).astype(np.float64)
    prior /= prior.sum()
    predictions = {
        "prior": np.tile(prior, (len(dev), 1)),
        "baseline": raw,
        "baseline_calibrated": temperature_scale(raw, calibration["baseline_temperature"]),
        "symmetric_baseline": symmetric,
        "symmetric_baseline_calibrated": temperature_scale(
            symmetric, calibration["symmetric_baseline_temperature"]),
        "semantic": semantic_symmetric,
        "fusion": fusion,
        "fusion_calibrated": temperature_scale(fusion, calibration["temperature"]),
    }
    truth_path = args.data_dir / "dev.csv"
    if args.smoke:
        truth_path = output / "smoke_truth.csv"
        write_truth(dev, truth_path)
    results = {}
    labels = targets(dev)
    for name, probabilities in predictions.items():
        path = output / "predictions" / f"{name}.csv"
        write_predictions(dev, probabilities, path)
        result = metrics(labels, probabilities)
        result["official_log_loss"] = score_official(truth_path, path)
        if abs(result["log_loss"] - result["official_log_loss"]) > 1e-12:
            raise AssertionError(f"Official score mismatch for {name}")
        results[name] = result
        print(f"{name} Log Loss {result['log_loss']:.9f} accuracy {result['accuracy']:.6f}", flush=True)
    # Both views of the complete ensemble must obey the same A/B mapping.
    swapped_lex = symmetrize(lexical.predict_proba(dev, swapped=True), raw)
    swapped_sem = symmetrize(semantic.predict_proba(dev, dev_embeddings, swapped=True), semantic_raw)
    swapped_fusion = temperature_scale(
        (1 - alpha) * swapped_lex + alpha * swapped_sem, calibration["temperature"],
    )
    swap_error = float(np.max(np.abs(swapped_fusion[:, [1, 0, 2]] - predictions["fusion_calibrated"])))
    if swap_error > 1e-6:
        raise AssertionError(f"A/B symmetry failed, max error {swap_error}")
    timings["dev_seconds"] = time.monotonic() - stage
    timings["total_seconds"] = time.monotonic() - started
    experiment = {
        "run_status": "complete", "smoke": args.smoke, "seed": args.seed,
        "splits": {key: len(value) for key, value in splits.items()},
        "resources": resources, "manifest": manifest, "timings": timings,
        "baseline_training": baseline_training, "semantic_training": semantic_training,
        "calibration": calibration, "results": results, "swap_max_abs_error": swap_error,
        "dependencies": installed_versions(),
    }
    write_json(output / "metrics.json", experiment)
    write_reports(output, train, dev, profile, experiment, predictions,
                  {"train": train_encoding, "dev": dev_encoding})
    print(f"Artifacts saved to {output}", flush=True)


def execute(args, output: Path, identity: dict) -> int:
    state_path = output / "run_state.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("configuration") != identity:
            raise ValueError("Output directory belongs to a different run. Choose another --output-dir.")
    state = {"configuration": identity, "status": "preflight"}
    try:
        resources = inspect_resources()
        manifest = verify_manifest()
        write_json(output / "preflight.json", {"resources": resources, "manifest": manifest})
        problems = []
        if not args.smoke and resources["ram_available_gib"] < 6:
            problems.append(f"Available RAM {resources['ram_available_gib']:.2f} GiB, required 6 GiB")
        if args.device == "cuda" and resources.get("gpu", {}).get("memory_free_mib", 0) < 2048:
            problems.append("Available GPU memory is below 2048 MiB")
        if problems:
            state.update(status="blocked_resources", problems=problems)
            write_json(state_path, state)
            print("Resource preflight blocked full training. " + ". ".join(problems), flush=True)
            return 2
        if args.preflight_only:
            state["status"] = "preflight_passed"
        else:
            state["status"] = "running"
            write_json(state_path, state)
            run_experiment(args, output, resources, manifest)
            state["status"] = "complete"
        write_json(state_path, state)
        return 0
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(state_path, state)
        traceback.print_exc()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-path")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="512 train / 128 dev, at most two MLP epochs")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    output = (args.output_dir or Path("artifacts-smoke" if args.smoke else "artifacts")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        args.model_path = resolve_encoder(args.model_path)
        hub_cache = ROOT / ".cache/huggingface/hub"
        hub_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hub_cache.parent)
        os.environ["HF_HUB_CACHE"] = str(hub_cache)
        identity = {"seed": args.seed, "smoke": args.smoke, "device": args.device,
                    "data_dir": str(args.data_dir), "model_path": resolve_encoder(args.model_path),
                    "data_sha256": {name: file_digest(args.data_dir / name)
                                    for name in ("train.csv", "dev.csv")}}
        with output_lock(output):
            return execute(args, output, identity)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
