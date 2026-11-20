"""Inference coordinator: model residency, bounded queue, one decode stream, cancellation.

Lifecycle: RECEIVED -> VALIDATED -> QUEUED -> RUNNING -> COMPLETED | CANCELLED | FAILED.
Only the worker thread touches the model. HTTP/CLI clients interact through handles.
"""

from __future__ import annotations

import enum
import itertools
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

from .errors import NotReadyError, QueueFullError, RequestError
from .sampling import SamplingParams, apply_penalties, make_generator, sample

log = logging.getLogger("littlerun.engine")


class State(str, enum.Enum):
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


TERMINAL = {State.COMPLETED, State.CANCELLED, State.FAILED}


@dataclass
class GenRequest:
    prompt_ids: list[int]
    max_tokens: int
    sampling: SamplingParams
    stop: list[str] = field(default_factory=list)


@dataclass
class Event:
    kind: str  # "delta" | "done" | "error"
    text: str = ""
    finish_reason: str | None = None
    usage: dict | None = None
    error: str | None = None


class RequestHandle:
    _ids = itertools.count()

    def __init__(self, req: GenRequest):
        self.id = f"req-{uuid.uuid4().hex[:16]}"
        self.req = req
        self.state = State.RECEIVED
        self.events: queue.Queue[Event] = queue.Queue()
        self._cancel = threading.Event()
        self.metrics: dict = {"created": time.time()}

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def result(self, timeout: float | None = None) -> tuple[str, str, dict]:
        """Block until terminal. Returns (text, finish_reason, usage)."""
        parts = []
        deadline = None if timeout is None else time.time() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.time())
            ev = self.events.get(timeout=remaining)
            if ev.kind == "delta":
                parts.append(ev.text)
            elif ev.kind == "done":
                return "".join(parts), ev.finish_reason, ev.usage
            else:
                raise RuntimeError(ev.error)


class Runner:
    """Interface the engine needs from a loaded model (real or fake)."""

    model_id: str
    max_context: int
    stop_ids: set[int]

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def chat_prompt(self, messages: list[dict], enable_thinking: bool) -> str: ...
    def new_session(self): ...  # returns obj with prefill(ids)->logits, decode(tok)->logits
    def info(self) -> dict: ...


class ModelRunner(Runner):
    def __init__(self, loaded):
        self.lm = loaded
        self.model_id = loaded.model_id
        self.max_context = loaded.options.max_context
        self.stop_ids = loaded.adapter.stop_token_ids(loaded.tokenizer, loaded.generation_config)
        self._tok_lock = threading.Lock()
        self._cache = loaded.adapter.create_kv_cache(loaded.model, self.max_context, loaded.device)
        self._cached_ids: list[int] = []  # tokens whose K/V are valid in the cache, in order
        self.prefix_reuse = True

    def encode(self, text):
        with self._tok_lock:
            return self.lm.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        with self._tok_lock:
            return self.lm.tokenizer.decode(ids, skip_special_tokens=True)

    def chat_prompt(self, messages, enable_thinking):
        with self._tok_lock:
            return self.lm.adapter.apply_chat_template(self.lm.tokenizer, messages, enable_thinking)

    def new_session(self):
        """Single-stream session. Reuses the cached K/V of the longest common prompt prefix."""
        runner, lm, cache = self, self.lm, self._cache

        class _S:
            reused = 0

            def prefill(self, ids):
                keep = 0
                if runner.prefix_reuse:
                    prev = runner._cached_ids
                    lim = min(len(prev), len(ids) - 1)  # always run >= 1 token to get logits
                    while keep < lim and prev[keep] == ids[keep]:
                        keep += 1
                runner._cached_ids = []  # invalid until this prefill succeeds
                cache.length = keep
                logits = lm.adapter.prefill(lm.model, ids[keep:], cache)
                runner._cached_ids = list(ids)
                self.reused = keep
                return logits

            def decode(self, tok):
                cached = runner._cached_ids
                runner._cached_ids = []
                logits = lm.adapter.decode(lm.model, tok, cache)
                cached.append(tok)
                runner._cached_ids = cached
                return logits

        return _S()

    def info(self):
        m = self.lm.manifest
        return {
            "backend": self.lm.backend.name,
            "device": str(self.lm.device),
            "dtype": str(self.lm.dtype).replace("torch.", ""),
            "factor_storage": self.lm.options.factor_storage,
            "max_context": self.max_context,
            "architecture": m["architecture"],
            "compression": {"family": m["compression"]["family"], "version": m["compression"]["version"],
                            "bits_per_weight": m.get("factors", {}).get("measured_bits_per_weight")},
            "capabilities": m.get("capabilities", {}),
            "source": {"repository": m["source"]["repository"], "revision": m["source"]["revision"]},
            "license": m["provenance"].get("license"),
        }


class Engine:
    def __init__(self, *, max_queue: int = 4, max_tokens_limit: int = 2048):
        self.max_queue = max_queue
        self.max_tokens_limit = max_tokens_limit
        self.state = "loading"
        self.error: str | None = None
        self.runner: Runner | None = None
        self._q: queue.Queue[RequestHandle] = queue.Queue(maxsize=max_queue)
        self._running: RequestHandle | None = None
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="littlerun-worker", daemon=True)
        self._worker.start()

    # ----------------------------------------------------------------- residency
    def load_async(self, factory):
        """factory() -> Runner. Runs in a background thread; readiness flips when done."""

        def _run():
            try:
                self.runner = factory()
                self.state = "ready"
                log.info("model ready: %s", self.runner.model_id)
            except Exception as e:  # noqa: BLE001 - surfaced via /health
                log.exception("model load failed")
                self.error = f"{type(e).__name__}: {e}"
                self.state = "failed"

        t = threading.Thread(target=_run, name="littlerun-loader", daemon=True)
        t.start()
        return t

    def set_runner(self, runner: Runner):
        self.runner = runner
        self.state = "ready"

    @property
    def ready(self) -> bool:
        return self.state == "ready" and self.runner is not None

    def shutdown(self):
        self._stop.set()

    def queue_depth(self) -> int:
        return self._q.qsize() + (1 if self._running else 0)

    # ----------------------------------------------------------------- submission
    def submit(self, req: GenRequest) -> RequestHandle:
        if not self.ready:
            raise NotReadyError(f"model not ready (state={self.state})" + (f": {self.error}" if self.error else ""))
        h = RequestHandle(req)
        if not req.prompt_ids:
            raise RequestError("prompt is empty after tokenization", "prompt")
        if req.max_tokens < 1 or req.max_tokens > self.max_tokens_limit:
            raise RequestError(f"max_tokens must be in [1, {self.max_tokens_limit}]", "max_tokens")
        if len(req.prompt_ids) + req.max_tokens > self.runner.max_context:
            raise RequestError(
                f"prompt ({len(req.prompt_ids)} tokens) + max_tokens ({req.max_tokens}) exceeds context "
                f"{self.runner.max_context}", "max_tokens")
        h.state = State.VALIDATED
        try:
            h.state = State.QUEUED
            self._q.put_nowait(h)
        except queue.Full:
            h.state = State.FAILED
            raise QueueFullError(f"queue full ({self.max_queue} pending)") from None
        return h

    # ----------------------------------------------------------------- worker
    def _loop(self):
        while not self._stop.is_set():
            try:
                h = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._running = h
            try:
                if h.cancelled:
                    h.state = State.CANCELLED
                    h.events.put(Event("done", finish_reason="cancelled", usage=self._usage(h, 0)))
                    continue
                self._generate(h)
            except Exception as e:  # noqa: BLE001 - request fails, engine survives
                log.exception("request %s failed", h.id)
                h.state = State.FAILED
                h.events.put(Event("error", error=f"{type(e).__name__}: {e}"))
                _release_device_memory()
            finally:
                self._running = None

    @staticmethod
    def _usage(h, n):
        p = len(h.req.prompt_ids)
        u = {"prompt_tokens": p, "completion_tokens": n, "total_tokens": p + n}
        if h.metrics.get("cached_prompt_tokens"):
            u["prompt_tokens_details"] = {"cached_tokens": h.metrics["cached_prompt_tokens"]}
        return u

    def _generate(self, h: RequestHandle):
        r, req = self.runner, h.req
        h.state = State.RUNNING
        gen = make_generator(req.sampling)
        sess = r.new_session()
        t0 = time.perf_counter()
        logits = sess.prefill(req.prompt_ids)
        h.metrics["prefill_s"] = time.perf_counter() - t0
        h.metrics["cached_prompt_tokens"] = getattr(sess, "reused", 0)
        out: list[int] = []
        emitted = 0  # chars of decoded text already sent
        hold = max((len(s) for s in req.stop), default=1) - 1
        finish = None
        t_first = None
        while True:
            if h.cancelled:
                finish = "cancelled"
                break
            if req.sampling.penalized:
                logits = apply_penalties(logits, req.sampling, out, req.prompt_ids)
            tok = sample(logits, req.sampling, gen)
            out.append(tok)
            if t_first is None:
                t_first = time.perf_counter()
                h.metrics["ttft_s"] = t_first - t0
            if tok in r.stop_ids:
                finish = "stop"
                text = r.decode(out[:-1])
                break
            text = r.decode(out)
            cut = _find_stop(text, req.stop)
            if cut is not None:
                text = text[:cut]
                finish = "stop"
                break
            if len(out) >= req.max_tokens:
                finish = "length"
                break
            safe = len(text) - hold
            while safe > emitted and text[safe - 1] == "�":
                safe -= 1  # do not emit a partial UTF-8 sequence
            if safe > emitted:
                h.events.put(Event("delta", text=text[emitted:safe]))
                emitted = safe
            logits = sess.decode(tok)
        if finish != "cancelled":
            text = text.rstrip("�") if finish == "length" else text
            if len(text) > emitted:
                h.events.put(Event("delta", text=text[emitted:]))
        n = len(out)
        dt = time.perf_counter() - (t_first or t0)
        h.metrics.update({"completion_tokens": n, "decode_tok_s": (n - 1) / dt if n > 1 and dt > 0 else None,
                          "total_s": time.perf_counter() - t0})
        h.state = State.CANCELLED if finish == "cancelled" else State.COMPLETED
        h.events.put(Event("done", finish_reason=finish, usage=self._usage(h, n)))
        log.info("%s %s tokens=%d ttft=%.3fs decode=%s tok/s", h.id, finish, n, h.metrics.get("ttft_s") or -1,
                 f"{h.metrics['decode_tok_s']:.2f}" if h.metrics.get("decode_tok_s") else "n/a")


def _find_stop(text: str, stops: list[str]) -> int | None:
    best = None
    for s in stops:
        i = text.find(s)
        if i >= 0 and (best is None or i < best):
            best = i
    return best


def _release_device_memory():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
