"""OpenAI-style HTTP API (localhost by default; no telemetry, no remote code)."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
import uuid
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .engine import TERMINAL, Engine, Event, GenRequest
from .errors import NotReadyError, QueueFullError, RequestError
from .sampling import SamplingParams

log = logging.getLogger("littlerun.server")


# --------------------------------------------------------------------------- schemas


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict | None = None
    seed: int | None = None
    n: int = 1
    user: str | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    @field_validator("stop")
    @classmethod
    def _stop(cls, v):
        if v is None:
            return []
        v = [v] if isinstance(v, str) else v
        if len(v) > 4 or any(not s for s in v):
            raise ValueError("stop must be up to 4 non-empty strings")
        return v

    def check_unsupported(self):
        if self.n != 1:
            raise RequestError("only n=1 is supported", "n")
        if self.presence_penalty or self.frequency_penalty:
            raise RequestError("presence/frequency penalties are not supported", "presence_penalty")
        if self.stream_options and set(self.stream_options) - {"include_usage"}:
            raise RequestError("only stream_options.include_usage is supported", "stream_options")


class CompletionRequest(_Base):
    prompt: str | list[str]
    echo: bool = False
    logprobs: int | None = None
    suffix: str | None = None

    def check_unsupported(self):
        super().check_unsupported()
        if self.echo or self.logprobs is not None or self.suffix is not None:
            raise RequestError("echo, logprobs and suffix are not supported", "echo")
        if isinstance(self.prompt, list) and len(self.prompt) != 1:
            raise RequestError("exactly one prompt is supported", "prompt")


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str | list[TextPart]
    name: str | None = None

    def text(self) -> str:
        return self.content if isinstance(self.content, str) else "".join(p.text for p in self.content)


class ChatRequest(_Base):
    messages: list[Message] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    tools: list | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None
    chat_template_kwargs: dict | None = None

    def check_unsupported(self):
        super().check_unsupported()
        if self.tools or (self.tool_choice not in (None, "none")):
            raise RequestError("tool calling is not a validated capability of this model/runtime", "tools")
        if self.response_format and self.response_format.get("type") != "text":
            raise RequestError("only response_format type 'text' is supported", "response_format")
        if self.chat_template_kwargs and set(self.chat_template_kwargs) - {"enable_thinking"}:
            raise RequestError("only chat_template_kwargs.enable_thinking is supported", "chat_template_kwargs")
        if self.messages[-1].role == "assistant":
            raise RequestError("last message must not be from the assistant", "messages")


# --------------------------------------------------------------------------- helpers


def _err(status: int, message: str, etype: str, param: str | None = None, code: str | None = None):
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": etype, "param": param, "code": code}})


def _next_event(h, timeout: float) -> Event | None:
    try:
        return h.events.get(timeout=timeout)
    except queue.Empty:
        return None


def create_app(engine: Engine, model_id: str, *, default_max_tokens: int = 512) -> FastAPI:
    app = FastAPI(title="LittleRun", version="0.1.0", docs_url=None, redoc_url=None)

    @app.exception_handler(RequestValidationError)
    async def _validation(_req, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return _err(400, f"{loc}: {first.get('msg', 'invalid request')}", "invalid_request_error", loc or None)

    @app.exception_handler(RequestError)
    async def _bad(_req, exc: RequestError):
        return _err(400, str(exc), "invalid_request_error", exc.param)

    @app.exception_handler(NotReadyError)
    async def _not_ready(_req, exc):
        return _err(503, str(exc), "service_unavailable", code="model_not_ready")

    @app.exception_handler(QueueFullError)
    async def _full(_req, exc):
        return _err(429, str(exc), "rate_limit_error", code="queue_full")

    @app.exception_handler(Exception)
    async def _internal(_req, exc):
        log.exception("internal error")
        return _err(500, "internal server error", "server_error")

    # ------------------------------------------------------------------ health
    @app.get("/health")
    async def health():
        body = {"live": True, "ready": engine.ready, "state": engine.state, "queue_depth": engine.queue_depth()}
        if engine.error:
            body["error"] = engine.error
        return body

    @app.get("/health/live")
    async def live():
        return {"live": True}

    @app.get("/health/ready")
    async def ready():
        if not engine.ready:
            return JSONResponse(status_code=503, content={"ready": False, "state": engine.state, "error": engine.error})
        return {"ready": True}

    @app.get("/v1/models")
    async def models():
        entry = {"id": model_id, "object": "model", "created": 0, "owned_by": "local", "ready": engine.ready}
        if engine.ready:
            entry["littlerun"] = engine.runner.info()
        return {"object": "list", "data": [entry]}

    # ------------------------------------------------------------------ generation
    def _check_model(name: str):
        if name != model_id:
            return _err(404, f"model {name!r} not found; available: {model_id!r}", "invalid_request_error",
                        "model", "model_not_found")
        if not engine.ready:
            raise NotReadyError(f"model not ready (state={engine.state})")
        return None

    def _sampling(body: _Base) -> SamplingParams:
        return SamplingParams(temperature=body.temperature, top_p=body.top_p, top_k=body.top_k, seed=body.seed)

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        body.check_unsupported()
        if (resp := _check_model(body.model)) is not None:
            return resp
        prompt = body.prompt if isinstance(body.prompt, str) else body.prompt[0]
        ids = engine.runner.encode(prompt)
        h = engine.submit(GenRequest(ids, body.max_tokens or default_max_tokens, _sampling(body), body.stop or []))
        return await _respond(h, body, request, chat=False)

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request):
        body.check_unsupported()
        if (resp := _check_model(body.model)) is not None:
            return resp
        msgs = [{"role": m.role, "content": m.text()} for m in body.messages]
        thinking = (body.chat_template_kwargs or {}).get("enable_thinking", True)
        ids = engine.runner.encode(engine.runner.chat_prompt(msgs, bool(thinking)))
        max_tokens = body.max_completion_tokens or body.max_tokens or default_max_tokens
        h = engine.submit(GenRequest(ids, max_tokens, _sampling(body), body.stop or []))
        return await _respond(h, body, request, chat=True)

    async def _respond(h, body: _Base, request: Request, *, chat: bool):
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"

        if not body.stream:
            parts, done = [], None
            try:
                while done is None:
                    ev = await asyncio.to_thread(_next_event, h, 0.25)
                    if ev is None:
                        if await request.is_disconnected():
                            h.cancel()
                            return JSONResponse(status_code=499, content={})
                        continue
                    if ev.kind == "delta":
                        parts.append(ev.text)
                    elif ev.kind == "error":
                        return _err(500, ev.error or "generation failed", "server_error")
                    else:
                        done = ev
            finally:
                if h.state not in TERMINAL:
                    h.cancel()
            text = "".join(parts)
            choice = ({"index": 0, "message": {"role": "assistant", "content": text},
                       "finish_reason": done.finish_reason} if chat else
                      {"index": 0, "text": text, "logprobs": None, "finish_reason": done.finish_reason})
            return {"id": rid, "object": obj, "created": created, "model": model_id, "choices": [choice],
                    "usage": done.usage}

        include_usage = bool((body.stream_options or {}).get("include_usage"))
        chunk_obj = "chat.completion.chunk" if chat else "text_completion"

        def chunk(delta_text: str | None, finish: str | None, role: bool = False, usage=None, choices=True):
            c = {"id": rid, "object": chunk_obj, "created": created, "model": model_id}
            if choices:
                if chat:
                    delta = {}
                    if role:
                        delta["role"] = "assistant"
                    if delta_text is not None:
                        delta["content"] = delta_text
                    c["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
                else:
                    c["choices"] = [{"index": 0, "text": delta_text or "", "logprobs": None, "finish_reason": finish}]
            else:
                c["choices"] = []
            if usage is not None:
                c["usage"] = usage
            return f"data: {json.dumps(c, ensure_ascii=False)}\n\n"

        async def gen():
            try:
                if chat:
                    yield chunk("", None, role=True)
                while True:
                    ev = await asyncio.to_thread(_next_event, h, 0.25)
                    if ev is None:
                        if await request.is_disconnected():
                            h.cancel()
                            return
                        continue
                    if ev.kind == "delta":
                        yield chunk(ev.text, None)
                    elif ev.kind == "error":
                        err = {"error": {"message": ev.error, "type": "server_error"}}
                        yield f"data: {json.dumps(err)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    else:
                        yield chunk(None if chat else "", ev.finish_reason)
                        if include_usage:
                            yield chunk(None, None, usage=ev.usage, choices=False)
                        yield "data: [DONE]\n\n"
                        return
            finally:
                if h.state not in TERMINAL:
                    h.cancel()  # client disconnect releases the decode stream

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
