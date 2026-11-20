"""Typed errors. Every load-time failure is fail-closed and names the offending item."""


class LittleRunError(Exception):
    """Base class for runtime errors."""


class ManifestError(LittleRunError):
    """The manifest is malformed, incompatible, or disagrees with the checkpoint."""


class CheckpointError(LittleRunError):
    """Tensor inventory, shape, dtype, packing, or hash validation failed."""


class CompatibilityError(LittleRunError):
    """No registered adapter, codec, or backend supports the requested combination."""


class RequestError(LittleRunError):
    """A client request is invalid (maps to HTTP 400)."""

    def __init__(self, message: str, param: str | None = None):
        super().__init__(message)
        self.param = param


class QueueFullError(LittleRunError):
    """The bounded request queue is full (maps to HTTP 429)."""


class NotReadyError(LittleRunError):
    """The model is loading or failed to load (maps to HTTP 503)."""
