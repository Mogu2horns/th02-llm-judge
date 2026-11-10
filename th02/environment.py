"""Validate CUDA and write active dependency versions without build-machine URLs."""

import argparse
import importlib.metadata
import os
from pathlib import Path


def installed_versions() -> dict[str, str]:
    names = {distribution.metadata["Name"]
             for distribution in importlib.metadata.distributions()}
    # Resolve by import precedence when a project package shadows its private base.
    return {name: importlib.metadata.version(name) for name in sorted(names, key=str.casefold)}


def write_dependency_lock(path: Path) -> None:
    lines = ["--extra-index-url https://download.pytorch.org/whl/cu121"]
    excluded = {"pip", "setuptools", "wheel"}
    lines.extend(f"{name}=={version}" for name, version in installed_versions().items()
                 if name.lower() not in excluded)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-path", type=Path, default=Path("requirements-lock.txt"))
    args = parser.parse_args()
    import torch

    if torch.version.cuda != "12.1" or not torch.cuda.is_available():
        raise RuntimeError("CUDA 12.1 runtime check failed")
    write_dependency_lock(args.lock_path)
    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
