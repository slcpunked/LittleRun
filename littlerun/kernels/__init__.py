"""Optional compiled CPU kernel for packed 1-bit factors.

A prebuilt module is used when LITTLERUN_KERNEL_DIR points at a folder holding
`littlerun_signed_gemm` (.pyd/.so) built for the installed torch version; the Windows
app ships one so end users need no compiler. Otherwise it is built on first use with
torch.utils.cpp_extension (needs a C++ compiler).
If neither works, `packed_kernel()` returns None and callers fall back.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("littlerun.kernels")
_lock = threading.Lock()
_mod = None
_failed: str | None = None
_source: str | None = None  # "prebuilt" | "jit"

_MODULE = "littlerun_signed_gemm"


def _load_prebuilt(folder: Path):
    """Import a prebuilt extension from `folder`, or raise."""
    import torch  # noqa: F401 - registers torch's DLL directory before the extension loads

    hits = sorted(folder.glob(f"{_MODULE}*.pyd" if os.name == "nt" else f"{_MODULE}*.so"))
    if not hits:
        raise FileNotFoundError(f"no {_MODULE} module in {folder}")
    if os.name == "nt":
        os.add_dll_directory(str(folder))  # OpenMP runtime (vcomp140.dll) shipped next to the module
    # The serial fallback build is named littlerun_signed_gemm_serial; the init symbol follows the file name.
    spec = importlib.util.spec_from_file_location(hits[0].name.split(".")[0], hits[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def packed_kernel():
    """Return the compiled module or None."""
    global _mod, _failed, _source
    if _mod is not None or _failed is not None:
        return _mod
    with _lock:
        if _mod is not None or _failed is not None:
            return _mod
        prebuilt = os.environ.get("LITTLERUN_KERNEL_DIR")
        if prebuilt:
            try:
                _mod, _source = _load_prebuilt(Path(prebuilt)), "prebuilt"
                return _mod
            except Exception as e:  # noqa: BLE001 - fall through to a JIT build
                log.warning("prebuilt kernel in %s unusable (%s); trying a local build", prebuilt, e)
        try:
            from torch.utils.cpp_extension import load

            src = Path(__file__).with_name("signed_gemm.cpp")
            build = Path(os.environ.get("LITTLERUN_BUILD_DIR", Path.home() / ".cache" / "littlerun" / "kernels"))
            build.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":  # MSVC (untested)
                cflags, ldflags = ["/O2", "/openmp"], []
            else:
                cflags, ldflags = ["-O3", "-fopenmp"], ["-fopenmp"]
            try:
                _mod = load(name="littlerun_signed_gemm", sources=[str(src)], build_directory=str(build),
                            extra_cflags=cflags, extra_ldflags=ldflags, verbose=False)
            except Exception as first:  # e.g. Apple clang without OpenMP: build single-threaded instead
                log.warning("kernel build with OpenMP failed (%s); retrying without", first)
                _mod = load(name="littlerun_signed_gemm_serial", sources=[str(src)], build_directory=str(build),
                            extra_cflags=[f for f in cflags if "openmp" not in f], extra_ldflags=[],
                            verbose=False)
            _source = "jit"
        except Exception as e:  # noqa: BLE001 - optional accelerator
            _failed = f"{type(e).__name__}: {e}"
            log.warning("packed kernel unavailable: %s", _failed)
    return _mod


def packed_kernel_error() -> str | None:
    return _failed


def packed_kernel_source() -> str | None:
    """How the loaded kernel was obtained: "prebuilt", "jit", or None if not loaded."""
    return _source
