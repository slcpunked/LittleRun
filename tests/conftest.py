import os
import sys
from pathlib import Path

import pytest

from littlerun.testing.fixtures import build_fixture_with_manifest


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    ref = build_fixture_with_manifest(d)
    return d, ref


@pytest.fixture(scope="session")
def samsung_ref():
    """Import the SamsungLabs/LittleBit reference implementation if LITTLEBIT_REF_PATH is set."""
    path = os.environ.get("LITTLEBIT_REF_PATH")
    if not path or not Path(path, "quantization").is_dir():
        pytest.skip("LITTLEBIT_REF_PATH not set to a SamsungLabs/LittleBit checkout")
    if path not in sys.path:
        sys.path.insert(0, path)
    import quantization  # noqa: F401

    return path
