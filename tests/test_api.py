"""API-layer tests: SSE order, [DONE], finish reasons, cancellation, disconnect, overload, errors, recovery."""

import json
import socket
import threading
import time

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from littlerun.engine import Engine, GenRequest, Runner, State
from littlerun.sampling import SamplingParams
from littlerun.server import create_app

V = 64
EOS = 0


class FakeRunner(Runner):
    """Char-level tokenizer over a-z; model emits the next letter, EOS after 'z' or on '!'."""

    model_id = "fake"
    max_context = 64
    stop_ids = {EOS}

    def __init__(self, delay=0.0, fail_on="#"):
        self.delay, self.fail_on = delay, fail_on

    def encode(self, text):
        return [ord(c) - 60 if "a" <= c <= "z" else (1 if c == "!" else 2 if c == "#" else 3) for c in text]

    def decode(self, ids):
        return "".join(chr(i + 60) if 37 <= i <= 62 else "" for i in ids)

    def chat_prompt(self, messages, enable_thinking):
        return messages[-1]["content"]

    def new_session(self):
        runner = self

        def logits_for(tok):
            if tok == 2:
                raise RuntimeError("injected failure")
            nxt = EOS if tok in (1, 62) else (tok + 1 if 37 <= tok < 62 else 37)
            out = torch.full((V,), -1e9)
            out[nxt] = 0.0
            return out

        class S:
            def prefill(self, ids):
                return logits_for(ids[-1])

            def decode(self, tok):
                time.sleep(runner.delay)
                return logits_for(tok)

        return S()

    def info(self):
        return {"backend": "fake", "max_context": self.max_context}


def make(delay=0.0, max_queue=4):
    eng = Engine(max_queue=max_queue, max_tokens_limit=48)
    eng.set_runner(FakeRunner(delay))
    return eng, TestClient(create_app(eng, "fake"))


def sse(resp):
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(line[6:])
    return events


def test_health_and_models():
    eng = Engine()
    c = TestClient(create_app(eng, "fake"))
    assert c.get("/health").json()["ready"] is False
    assert c.get("/health/ready").status_code == 503
    assert c.post("/v1/completions", json={"model": "fake", "prompt": "a"}).status_code == 503
    eng.set_runner(FakeRunner())
    assert c.get("/health/ready").status_code == 200
    data = c.get("/v1/models").json()["data"]
    assert data[0]["id"] == "fake" and data[0]["littlerun"]["max_context"] == 64


def test_completion_nonstream_and_finish_reasons():
    _, c = make()
    r = c.post("/v1/completions", json={"model": "fake", "prompt": "w", "max_tokens": 10, "temperature": 0})
    body = r.json()
    assert r.status_code == 200
    assert body["choices"][0]["text"] == "xyz" and body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5}
    r = c.post("/v1/completions", json={"model": "fake", "prompt": "a", "max_tokens": 3, "temperature": 0})
    assert r.json()["choices"][0] | {} and r.json()["choices"][0]["text"] == "bcd"
    assert r.json()["choices"][0]["finish_reason"] == "length"
    r = c.post("/v1/completions", json={"model": "fake", "prompt": "a", "max_tokens": 20, "stop": ["fg"]})
    assert r.json()["choices"][0]["text"] == "bcde" and r.json()["choices"][0]["finish_reason"] == "stop"


def test_chat_stream_order_done_and_usage():
    _, c = make()
    req = {"model": "fake", "messages": [{"role": "user", "content": "p"}], "max_tokens": 20, "temperature": 0,
           "stream": True, "stream_options": {"include_usage": True}}
    with c.stream("POST", "/v1/chat/completions", json=req) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        ev = sse(r)
    assert ev[-1] == "[DONE]"
    chunks = [json.loads(e) for e in ev[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks if ch["choices"])
    assert text == "qrstuvwxyz"
    finals = [ch for ch in chunks if ch["choices"] and ch["choices"][0]["finish_reason"]]
    assert len(finals) == 1 and finals[0]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 11
    assert len({ch["id"] for ch in chunks}) == 1
    nonstream = c.post("/v1/chat/completions", json={**req, "stream": False, "stream_options": None}).json()
    assert nonstream["choices"][0]["message"]["content"] == text


@pytest.mark.parametrize("payload,status,param", [
    ({"model": "fake", "prompt": "a", "temperature": 3}, 400, "temperature"),
    ({"model": "fake", "prompt": "a", "bogus": 1}, 400, "bogus"),
    ({"model": "fake", "prompt": "a", "n": 2}, 400, "n"),
    ({"model": "fake", "prompt": "a", "presence_penalty": 3}, 400, "presence_penalty"),
    ({"model": "fake", "prompt": "a", "repetition_penalty": 0}, 400, "repetition_penalty"),
    ({"model": "fake", "prompt": "a", "logprobs": 2}, 400, "echo"),
    ({"model": "fake", "prompt": "a", "max_tokens": 0}, 400, "max_tokens"),
    ({"model": "fake", "prompt": "a", "max_tokens": 64}, 400, "max_tokens"),
    ({"model": "other", "prompt": "a"}, 404, "model"),
    ({"model": "fake", "prompt": "", "max_tokens": 4}, 400, "prompt"),
])
def test_invalid_completion_params(payload, status, param):
    _, c = make()
    r = c.post("/v1/completions", json=payload)
    assert r.status_code == status, r.text
    assert r.json()["error"]["param"].startswith(param)


@pytest.mark.parametrize("extra,param", [
    ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "tools"),
    ({"response_format": {"type": "json_object"}}, "response_format"),
    ({"messages": [{"role": "tool", "content": "x"}]}, "messages"),
    ({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}, "messages"),
])
def test_invalid_chat_params(extra, param):
    _, c = make()
    r = c.post("/v1/chat/completions", json={"model": "fake", "messages": [{"role": "user", "content": "a"}], **extra})
    assert r.status_code == 400 and r.json()["error"]["param"].startswith(param), r.text


def test_cancellation_and_recovery():
    eng, c = make(delay=0.05)
    h = eng.submit(GenRequest([37], 40, SamplingParams(0.0)))
    time.sleep(0.15)
    h.cancel()
    text, finish, usage = h.result(5)
    assert finish == "cancelled" and h.state == State.CANCELLED and usage["completion_tokens"] < 25
    r = c.post("/v1/completions", json={"model": "fake", "prompt": "x", "max_tokens": 5, "temperature": 0})
    assert r.json()["choices"][0]["text"] == "yz"


def test_queued_request_cancelled_before_running():
    eng, _ = make(delay=0.05)
    running = eng.submit(GenRequest([37], 10, SamplingParams(0.0)))
    queued = eng.submit(GenRequest([37], 10, SamplingParams(0.0)))
    queued.cancel()
    assert running.result(5)[1] == "length"
    assert queued.result(5)[1] == "cancelled" and queued.state == State.CANCELLED


def test_overload_returns_429_then_recovers():
    eng, c = make(delay=0.05, max_queue=1)
    a = eng.submit(GenRequest([37], 20, SamplingParams(0.0)))
    time.sleep(0.1)  # a is running
    b = eng.submit(GenRequest([37], 20, SamplingParams(0.0)))  # fills the queue
    r = c.post("/v1/completions", json={"model": "fake", "prompt": "a", "max_tokens": 2})
    assert r.status_code == 429 and r.json()["error"]["code"] == "queue_full"
    a.cancel(), b.cancel()
    a.result(5), b.result(5)
    assert c.post("/v1/completions", json={"model": "fake", "prompt": "a", "max_tokens": 2}).status_code == 200


def test_generation_failure_returns_500_and_engine_survives():
    _, c = make()
    r = c.post("/v1/completions", json={"model": "fake", "prompt": "#", "max_tokens": 3})
    assert r.status_code == 500 and "injected" in r.json()["error"]["message"]
    with c.stream("POST", "/v1/completions", json={"model": "fake", "prompt": "#", "max_tokens": 3, "stream": True}) as s:
        assert s.status_code == 200
        ev = sse(s)
    assert "error" in json.loads(ev[0]) and ev[-1] == "[DONE]"
    assert c.post("/v1/completions", json={"model": "fake", "prompt": "a", "max_tokens": 2}).status_code == 200


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_client_disconnect_cancels_request():
    import uvicorn

    eng = Engine(max_queue=2, max_tokens_limit=48)
    eng.set_runner(FakeRunner(delay=0.05))
    seen = []
    orig = eng.submit
    eng.submit = lambda req: seen.append(orig(req)) or seen[-1]
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(eng, "fake"), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    with httpx.Client(timeout=10) as client:
        with client.stream("POST", f"http://127.0.0.1:{port}/v1/completions",
                           json={"model": "fake", "prompt": "a", "max_tokens": 40, "stream": True}) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    break  # read one chunk then hang up
    h = seen[0]
    for _ in range(100):
        if h.state in (State.CANCELLED, State.COMPLETED):
            break
        time.sleep(0.05)
    server.should_exit = True
    t.join(5)
    assert h.state == State.CANCELLED, h.state

