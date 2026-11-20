"""Compile the packed 1-bit CPU kernel for the pinned torch and copy it to --out.

Run inside the build venv with the MSVC environment loaded (vcvars64.bat). The app ships the result,
so end users never need a compiler.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="folder that receives the .pyd")
    p.add_argument("--work", required=True, help="scratch build folder (emptied first)")
    a = p.parse_args()

    work, out = Path(a.work), Path(a.out)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    os.environ["LITTLERUN_BUILD_DIR"] = str(work)
    os.environ.pop("LITTLERUN_KERNEL_DIR", None)  # force a fresh compile

    import torch

    from littlerun import kernels

    mod = kernels.packed_kernel()
    if mod is None:
        print(f"kernel build failed: {kernels.packed_kernel_error()}", file=sys.stderr)
        return 1
    built = Path(mod.__file__)
    if "serial" in built.name:
        print("warning: OpenMP build failed; shipping the single-threaded kernel", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("littlerun_signed_gemm*"):
        old.unlink()
    shutil.copy2(built, out / built.name)
    simd = "avx512" if mod.avx512_available() else ("avx2" if mod.avx2_available() else "scalar")
    print(f"kernel {built.name} for torch {torch.__version__} -> {out} (this CPU: {simd})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
