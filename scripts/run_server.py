"""Run the unchanged experiment with process-local Linux memory settings."""

import ctypes
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    # On a fragmented, shared server, synchronous THP compaction can dominate
    # NumPy allocations. Affect this process only; leave host settings alone.
    os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if libc.prctl(42, 0, 0, 0, 0) != 1:  # PR_GET_THP_DISABLE
        raise RuntimeError("Failed to verify process-local THP setting")
    print("[server] Transparent huge pages disabled for this process", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("th02.run", run_name="__main__")


if __name__ == "__main__":
    main()
