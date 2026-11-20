"""Optional compiled CPU kernel for packed 1-bit factors.

Built on first use with torch.utils.cpp_extension (needs a C++ compiler).
If the build fails, `packed_kernel()` returns None and callers fall back.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("littlerun.kernels")
_lock = threading.Lock()
_mod = None
_failed: str | None = None


def packed_kernel():
    """Return the compiled module or None."""
    global _mod, _failed
    if _mod is not None or _failed is not None:
        return _mod
    with _lock:
        if _mod is not None or _failed is not None:
            return _mod
        try:
            from torch.utils.cpp_extension import load

            src = Path(__file__).with_name("signed_gemm.cpp")
            build = Path(os.environ.get("LITTLERUN_BUILD_DIR", Path.home() / ".cache" / "littlerun" / "kernels"))
            build.mkdir(parents=True, exist_ok=True)
            _mod = load(name="littlerun_signed_gemm", sources=[str(src)], build_directory=str(build),
                        extra_cflags=["-O3", "-fopenmp"], extra_ldflags=["-fopenmp"], verbose=False)
        except Exception as e:  # noqa: BLE001 - optional accelerator
            _failed = f"{type(e).__name__}: {e}"
            log.warning("packed kernel unavailable: %s", _failed)
    return _mod


def packed_kernel_error() -> str | None:
    return _failed
