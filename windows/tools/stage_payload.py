"""Write build/payload/payload.json from the pinned checkpoints and the build's version pins.

Model file sizes come from the Hugging Face API at the pinned revision (no model download).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path


def hf_sizes(repo: str, revision: str) -> dict[str, int]:
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "LittleRun-build"})
    with urllib.request.urlopen(req, timeout=60) as r:
        info = json.load(r)
    return {s["rfilename"]: int(s.get("size") or 0) for s in info.get("siblings", [])}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--payload", required=True)
    p.add_argument("--repo-root", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--uv", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--torch", required=True)
    p.add_argument("--cpu-index", required=True)
    p.add_argument("--cuda-index", required=True)
    p.add_argument("--cuda-min-driver", type=float, required=True)
    p.add_argument("--wheel", required=True)
    a = p.parse_args()

    root = Path(a.repo_root)
    sys.path.insert(0, str(root))
    from littlerun.cli import PINNED

    payload = Path(a.payload)
    app = json.loads((root / "windows" / "models.json").read_text(encoding="utf-8"))
    (payload / "manifests").mkdir(parents=True, exist_ok=True)
    models = {}
    for name, extra in app["models"].items():
        spec = PINNED[name]
        sizes = hf_sizes(spec["repo"], spec["revision"])
        missing = [f for f in spec["files"] if f not in sizes]
        if missing:
            raise SystemExit(f"{name}: files not found at {spec['revision']}: {missing}")
        src = root / "littlerun" / "manifests" / spec["manifest"]
        shutil.copy2(src, payload / "manifests" / spec["manifest"])
        models[name] = {
            "displayName": extra["displayName"],
            "modelId": extra["modelId"],
            "repo": spec["repo"],
            "revision": spec["revision"],
            "files": spec["files"],
            "sha256": spec["sha256"],
            "license": spec["license"],
            "licenseUrl": extra.get("licenseUrl", ""),
            "manifest": f"manifests/{spec['manifest']}",
            "sizeBytes": sum(sizes[f] for f in spec["files"]),
            "chatTuned": bool(extra.get("chatTuned")),
            "note": extra.get("note", ""),
        }
        print(f"model {name}: {len(spec['files'])} files, {models[name]['sizeBytes'] / 2**30:.2f} GiB")

    kernel = sorted(f"kernel/{k.name}" for k in (payload / "kernel").glob("*") if k.is_file()) \
        if (payload / "kernel").is_dir() else []
    doc = {
        "version": a.version,
        "uv": a.uv,
        "python": a.python,
        "torch": a.torch,
        "torchIndexes": {"cpu": a.cpu_index, "cuda": a.cuda_index},
        "cudaMinDriver": a.cuda_min_driver,
        "wheel": a.wheel,
        "requirements": "requirements.txt",
        "kernel": kernel,
        "defaultModel": app["default"],
        "models": models,
    }
    (payload / "payload.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {payload / 'payload.json'} (kernel files: {len(kernel)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
