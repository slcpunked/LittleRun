"""Packing ABI and codec validation (loader-layer tests)."""

import json
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from littlerun.codecs.littlebit import (LittleBitCodec, pack_signs, padding_is_clean, samsung_rank_for_bits,
                                        unpack_signs)
from littlerun.errors import CheckpointError, ManifestError
from littlerun.manifest import MANIFEST_NAME, build_manifest, load_manifest, verify_against_checkpoint
from littlerun.registry import ARCHITECTURES
from littlerun.safetensors_io import TensorReader, read_inventory
from littlerun.testing.fixtures import write_tiny_checkpoint


@pytest.mark.parametrize("rows,cols", [(1, 1), (3, 31), (4, 32), (5, 33), (7, 64), (2, 100), (9, 824)])
def test_pack_roundtrip(rows, cols):
    g = torch.Generator().manual_seed(rows * 1000 + cols)
    s = torch.where(torch.randn(rows, cols, generator=g) < 0, -1, 1).to(torch.int8)
    p = pack_signs(s)
    assert p.dtype == torch.int32 and p.shape == (rows, (cols + 31) // 32)
    assert padding_is_clean(p, cols)
    assert torch.equal(unpack_signs(p, cols), s)


def test_bit_order_lsb_first_and_sign_convention():
    s = torch.ones(1, 32, dtype=torch.int8)
    s[0, 0] = -1  # bit 0
    s[0, 31] = -1  # bit 31 -> sign bit of the int32 word
    p = pack_signs(s)
    assert p.item() == (1 | (1 << 31)) - 2**32
    zeros = torch.zeros(1, 5)
    assert torch.equal(unpack_signs(pack_signs(zeros), 5), torch.ones(1, 5, dtype=torch.int8))  # sign(0)=+1


def test_dirty_padding_detected():
    p = pack_signs(torch.ones(2, 20, dtype=torch.int8))
    p[1, 0] |= 1 << 25
    assert not padding_is_clean(p, 20)


@pytest.mark.reference
def test_matches_samsung_packer(samsung_ref):
    from quantization.utils.binary_packer import binary_packer, binary_unpacker

    g = torch.Generator().manual_seed(1)
    for cols in (5, 32, 77, 824):
        s = torch.where(torch.randn(6, cols, generator=g) < 0, -1, 1).to(torch.int8)
        assert torch.equal(pack_signs(s), binary_packer(s))
        assert torch.equal(unpack_signs(binary_packer(s), cols), binary_unpacker(binary_packer(s), (6, cols)))


def test_rank_formula_matches_real_checkpoint_layout():
    # Ranks observed in LittleBitLLM/littlebit-qwen3-8b-0.55bpw (eff_bit=0.55, residual=True).
    assert samsung_rank_for_bits(4096, 4096, 0.55, True) == 544
    assert samsung_rank_for_bits(4096, 1024, 0.55, True) == 208
    assert samsung_rank_for_bits(4096, 12288, 0.55, True) == 824
    assert samsung_rank_for_bits(12288, 4096, 0.55, True) == 824


def _expected(d):
    a = ARCHITECTURES.get(("qwen3", "1"))()
    cfg = a.validate_config(json.loads((d / "config.json").read_text()))
    return a, cfg


def test_validate_tiny(tiny_ckpt):
    d, _ = tiny_ckpt
    a, cfg = _expected(d)
    r = TensorReader(d)
    groups, claimed = LittleBitCodec().validate_tensors(r.inventory, r, a.linear_shapes(cfg), residual=True)
    assert len(groups) == 14 and all(g.ranks == [20, 12] for g in groups.values())
    assert set(r.inventory) == claimed | set(a.dense_tensors(cfg))


def _mutate(src, dst, fn):
    shutil.copytree(src, dst)
    sd = load_file(str(dst / "model.safetensors"))
    fn(sd)
    save_file(sd, str(dst / "model.safetensors"))
    return dst


@pytest.mark.parametrize("mutation,match", [
    (lambda sd: sd.pop("model.layers.0.self_attn.q_proj.v2_R"), "incomplete factor path"),
    (lambda sd: sd.pop("model.layers.1.mlp.down_proj.U_packed"), "missing factor groups"),
    (lambda sd: sd.__setitem__("model.layers.0.mlp.up_proj.u1", torch.ones(1, 7)), "u1"),
    (lambda sd: sd.__setitem__("model.layers.0.mlp.up_proj.V_shape", torch.tensor([21, 64])), "stored shapes"),
    (lambda sd: sd.__setitem__("model.layers.0.mlp.up_proj._split_dim_final", torch.tensor(24)), "_split_dim_final"),
    (lambda sd: sd.__setitem__("model.layers.0.mlp.up_proj.U_packed",
                               sd["model.layers.0.mlp.up_proj.U_packed"].to(torch.int64)), "I32"),
    (lambda sd: sd["model.layers.0.mlp.down_proj.V_packed"].__setitem__((0, -1), 1 << 20), "padding"),
])
def test_codec_rejects_corruption(tiny_ckpt, tmp_path, mutation, match):
    src, _ = tiny_ckpt
    d = _mutate(src, tmp_path / "bad", mutation)
    a, cfg = _expected(d)
    r = TensorReader(d)
    with pytest.raises(CheckpointError, match=match):
        LittleBitCodec().validate_tensors(r.inventory, r, a.linear_shapes(cfg), residual=True)


def test_residual_mismatch_rejected(tiny_ckpt):
    d, _ = tiny_ckpt
    a, cfg = _expected(d)
    with pytest.raises(CheckpointError, match="residual"):
        LittleBitCodec().validate_tensors(read_inventory(d), None, a.linear_shapes(cfg), residual=False)


def test_unknown_tensor_fails_closed(tmp_path):
    write_tiny_checkpoint(tmp_path, extra_tensor="model.layers.0.mystery.weight")
    with pytest.raises(ManifestError, match="unclaimed"):
        build_manifest(tmp_path, source_repository="x/y", source_revision="r", license_id="t")


def test_manifest_detects_changes(tiny_ckpt, tmp_path):
    src, ref = tiny_ckpt
    m = load_manifest(src / MANIFEST_NAME)
    assert m["factors"]["groups"] == 14 and m["factors"]["residual"] is True
    verify_against_checkpoint(m, src)
    d = tmp_path / "c"
    shutil.copytree(src, d)
    cfg = json.loads((d / "config.json").read_text())
    cfg["rope_theta"] = 1.0
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ManifestError, match="config.json"):
        verify_against_checkpoint(m, d)


def test_manifest_hash_mismatch(tiny_ckpt, tmp_path):
    src, _ = tiny_ckpt
    d = tmp_path / "h"
    shutil.copytree(src, d)
    f = d / "model.safetensors"
    b = bytearray(f.read_bytes())
    b[-3] ^= 0xFF  # same size, different content
    f.write_bytes(bytes(b))
    m = load_manifest(src / MANIFEST_NAME)
    verify_against_checkpoint(m, d, verify_hashes=False)  # size/inventory alone cannot see it
    with pytest.raises(ManifestError, match="sha256"):
        verify_against_checkpoint(m, d)


def test_manifest_rejects_incompatible_contract(tiny_ckpt, tmp_path):
    src, _ = tiny_ckpt
    m = json.loads((src / MANIFEST_NAME).read_text())
    m["compression"]["bit_order"] = "msb_first"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    with pytest.raises(ManifestError, match="bit_order"):
        load_manifest(p)
