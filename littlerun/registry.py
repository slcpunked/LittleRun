"""Lightweight typed registries (not a plugin marketplace).

Compatibility is resolved here before any model memory is allocated.
"""

from __future__ import annotations

from .errors import CompatibilityError


class _Registry:
    kind = "item"

    def __init__(self):
        self._items: dict = {}

    def register(self, key, factory):
        if key in self._items:
            raise ValueError(f"duplicate {self.kind} registration: {key}")
        self._items[key] = factory

    def get(self, key):
        try:
            return self._items[key]
        except KeyError:
            raise CompatibilityError(f"no {self.kind} registered for {key!r}; known: {sorted(self._items)}") from None

    def keys(self):
        return sorted(self._items)


class ArchitectureRegistry(_Registry):
    """Keyed by (architecture_id, adapter_revision)."""
    kind = "architecture adapter"


class CodecRegistry(_Registry):
    """Keyed by (compression_family, format_version)."""
    kind = "codec"


class BackendRegistry(_Registry):
    """Backends are probed in priority order with supports(operation_spec)."""
    kind = "backend"

    def resolve(self, spec):
        for key in self.keys():
            backend = self._items[key]()
            if backend.supports(spec):
                return backend
        raise CompatibilityError(f"no backend supports {spec}")


ARCHITECTURES = ArchitectureRegistry()
CODECS = CodecRegistry()
BACKENDS = BackendRegistry()


def _register_builtins():
    from .arch.qwen3 import Qwen3Adapter
    from .backends.torch_ref import TorchReferenceBackend
    from .codecs.littlebit import CODEC_VERSION, FAMILY, LittleBitCodec

    ARCHITECTURES.register(("qwen3", "1"), Qwen3Adapter)
    CODECS.register((FAMILY, CODEC_VERSION), LittleBitCodec)
    BACKENDS.register("90-torch-ref", TorchReferenceBackend)


_register_builtins()
