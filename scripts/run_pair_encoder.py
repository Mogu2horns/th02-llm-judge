#!/usr/bin/env python3
"""Run one predeclared joint MiniLM candidate on the immutable fit/tune IDs."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure_process(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "8"
    os.environ["TMPDIR"] = str(ROOT / ".cache/tmp")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0 or libc.prctl(42, 0, 0, 0, 0) != 1:
        raise RuntimeError("Could not disable process-local THP")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts-improved-encoder")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/all-MiniLM-L6-v2")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--prompt-budget", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Separate 48-fit/12-tune smoke output required")
    args = parser.parse_args()
    configure_process(args.gpu)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from th02.data import load_samples
    from th02.pair_encoder import PairConfig, prepare_token_cache, train_pair_encoder, write_json

    torch.set_num_threads(8)
    all_samples = load_samples(ROOT / "data/train.csv")
    split_record = json.loads((ROOT / "artifacts/splits.json").read_text())
    by_id = {sample.id: sample for sample in all_samples}
    fit_ids, tune_ids = split_record["fit"]["ids"], split_record["tune"]["ids"]
    if set(fit_ids).intersection(tune_ids):
        raise ValueError("Fixed fit/tune IDs overlap")
    if args.smoke:
        if args.output_dir.resolve() == (ROOT / "artifacts-improved-encoder").resolve():
            raise ValueError("Smoke must use a separate output directory")
        fit_ids, tune_ids = fit_ids[:48], tune_ids[:12]
    samples = [by_id[identifier] for identifier in fit_ids + tune_ids]
    del all_samples, by_id
    fit, tune = np.arange(len(fit_ids)), np.arange(len(fit_ids), len(samples))
    if {samples[int(i)].group_hash for i in fit}.intersection(
            samples[int(i)].group_hash for i in tune):
        raise ValueError("Normalized content groups overlap")
    config = PairConfig(max_length=args.max_length, prompt_budget=args.prompt_budget,
                        epochs=args.epochs, batch_size=args.batch_size,
                        accumulation=args.accumulation, encoder_lr=args.encoder_lr,
                        head_lr=args.head_lr, seed=args.seed)
    if args.prepare_only:
        prepare_token_cache(samples, args.model_path, args.output_dir / "token_cache", config)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("This training entry requires CUDA; --prepare-only uses CPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "execution.json", {"gpu": args.gpu,
                "gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "fit_count": len(fit), "tune_count": len(tune), "official_dev_loaded": False,
                "calibration_used": False, "smoke": args.smoke})
    train_pair_encoder(samples, fit, tune, args.model_path, args.output_dir,
                       config=config, device="cuda", resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
