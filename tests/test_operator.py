"""Factorized linear operator: math, orientation, residual, edge inputs."""

import pytest
import torch

from littlerun.backends.torch_ref import FactorizedLinear


def _factors(m, n, ranks, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    paths, W = [], torch.zeros(n, m, dtype=torch.float64)
    for s in ranks:
        U = torch.where(torch.randn(n, s, generator=g) < 0, -1, 1).to(torch.int8)
        V = torch.where(torch.randn(s, m, generator=g) < 0, -1, 1).to(torch.int8)
        sc = {k: (torch.rand(sz, generator=g) + 0.1).to(dtype) for k, sz in (("u1", n), ("u2", s), ("v1", s), ("v2", m))}
        paths.append({"U": U, "V": V, **sc})
        d = {k: v.double() for k, v in sc.items()}
        W += (d["u1"][:, None] * U.double() * d["u2"][None]) @ (d["v1"][:, None] * V.double() * d["v2"][None])
    return paths, W


@pytest.mark.parametrize("m,n,ranks", [(37, 13, [9]), (64, 72, [20, 12]), (1, 1, [1]), (33, 65, [31, 33])])
@pytest.mark.parametrize("storage", ["int8", "compute"])
def test_matches_dense_reconstruction(m, n, ranks, storage):
    paths, W = _factors(m, n, ranks)
    bias = torch.randn(n)
    op = FactorizedLinear(m, n, paths, bias, torch.float32, storage)
    x = torch.randn(3, 5, m)
    y = op(x)
    ref = x.double() @ W.t() + bias.double()
    assert y.shape == (3, 5, n)
    err = (y.double() - ref).abs()
    rel = err.max() / ref.abs().max().clamp_min(1e-12)
    assert rel < 1e-5, rel


def test_orientation_is_not_transposed():
    # A transposed interpretation would change shapes for a non-square layer; check values on a basis vector.
    paths, W = _factors(5, 3, [4], seed=3)
    op = FactorizedLinear(5, 3, paths, None, torch.float32)
    for j in range(5):
        e = torch.zeros(1, 5)
        e[0, j] = 1
        assert torch.allclose(op(e)[0].double(), W[:, j], rtol=1e-5, atol=1e-6)


def test_zero_constant_and_extreme_inputs():
    paths, W = _factors(16, 8, [6, 4], seed=2)
    op = FactorizedLinear(16, 8, paths, None, torch.float32)
    assert torch.equal(op(torch.zeros(2, 16)), torch.zeros(2, 8))
    c = torch.full((1, 16), 3.0)
    assert torch.allclose(op(c).double(), c.double() @ W.t(), rtol=1e-5)
    big = torch.full((1, 16), 1e30)
    op(big)  # fp32 overflow to inf is acceptable; it must not raise
    assert op(torch.full((1, 16), 1e3)).isfinite().all()


def test_residual_path_contributes():
    paths, W = _factors(16, 8, [6, 4], seed=5)
    full = FactorizedLinear(16, 8, paths, None, torch.float32)
    primary = FactorizedLinear(16, 8, paths[:1], None, torch.float32)
    resid = FactorizedLinear(16, 8, paths[1:], None, torch.float32)
    x = torch.randn(4, 16)
    assert torch.allclose(full(x), primary(x) + resid(x), atol=1e-5)


def test_bf16_close_to_fp32():
    paths, W = _factors(256, 128, [64, 64], seed=7)
    x = torch.randn(4, 256)
    y32 = FactorizedLinear(256, 128, paths, None, torch.float32)(x)
    y16 = FactorizedLinear(256, 128, paths, None, torch.bfloat16)(x.bfloat16()).float()
    rel = (y16 - y32).norm() / y32.norm()
    assert rel < 2e-2, rel


@pytest.mark.reference
def test_matches_samsung_littlebitlinear(samsung_ref):
    """Same packed factors through Samsung LittleBitLinear (binarized mode) and ours, bf16 and fp32."""
    from torch import nn
    from quantization.functions import SmoothSign
    from quantization.modules import LittleBitLinear

    torch.manual_seed(0)
    lin = nn.Linear(96, 80, bias=False)
    lin.__class__ = LittleBitLinear
    lin.__quant_convert__(do_train=True, quant_func=SmoothSign, eff_bit=1.0, residual=True, use_itq=True,
                          itq_n_iter=5)
    for dtype in (torch.float32, torch.bfloat16):
        paths = []
        for suf in ("", "_R"):
            p = {k: getattr(lin, k + suf).data.reshape(-1) for k in ("u1", "u2", "v1", "v2")}
            p["U"] = torch.where(getattr(lin, "U" + suf).data < 0, -1, 1).to(torch.int8)
            p["V"] = torch.where(getattr(lin, "V" + suf).data < 0, -1, 1).to(torch.int8)
            paths.append(p)
        ours = FactorizedLinear(96, 80, paths, None, dtype)
        ref = LittleBitLinear.__new__(LittleBitLinear)
        nn.Module.__init__(ref)
        ref.in_features, ref.out_features, ref.residual, ref.bias, ref._binarized = 96, 80, True, None, True
        for suf in ("", "_R"):
            for k in ("U", "V"):
                setattr(ref, k + suf, nn.Parameter(torch.where(getattr(lin, k + suf).data < 0, -1.0, 1.0).to(dtype)))
            for k in ("u1", "u2", "v1", "v2"):
                setattr(ref, k + suf, nn.Parameter(getattr(lin, k + suf).data.to(dtype)))
        x = torch.randn(7, 96).to(dtype)
        a, b = ours(x).float(), ref(x).float()
        assert torch.equal(a, b), (a - b).abs().max()


@pytest.mark.skipif(not __import__("littlerun.backends.torch_ref", fromlist=["x"]).int8mm_available(),
                    reason="aten._weight_int8pack_mm unavailable")
@pytest.mark.parametrize("m,n,ranks", [(37, 13, [9]), (64, 72, [20, 12]), (96, 40, [33, 31]), (4096, 1024, [208])])
@pytest.mark.parametrize("rows", [1, 3, 16, 17, 40])
def test_int8mm_matches_dense_in_bf16(m, n, ranks, rows):
    """Fused int8 kernel (rows <= 16) and its cast fallback (rows > 16), odd K padded to 32."""
    paths, W = _factors(m, n, ranks, seed=m + n)
    op = FactorizedLinear(m, n, paths, None, torch.bfloat16, "int8mm")
    ref_op = FactorizedLinear(m, n, paths, None, torch.bfloat16, "compute")
    x = torch.randn(rows, m)
    y = op(x.bfloat16()).double()
    ref = x.double() @ W.t()
    rel = ((y - ref).norm() / ref.norm()).item()
    rel_ref = ((ref_op(x.bfloat16()).double() - ref).norm() / ref.norm()).item()
    assert rel < 1.5e-2 and rel <= rel_ref * 1.5 + 1e-3, (rel, rel_ref)


def test_int8mm_rejects_fp32():
    paths, _ = _factors(8, 8, [4])
    with pytest.raises(ValueError, match="bf16"):
        FactorizedLinear(8, 8, paths, None, torch.float32, "int8mm")


def _packed_paths(paths):
    from littlerun.codecs.littlebit import pack_signs

    return [{"U_packed": pack_signs(p["U"]), "V_packed": pack_signs(p["V"]), "rank": p["U"].shape[1],
             **{k: p[k] for k in ("u1", "u2", "v1", "v2")}} for p in paths]


def _kernel_or_skip():
    from littlerun.kernels import packed_kernel

    if packed_kernel() is None:
        pytest.skip("packed kernel did not build")


@pytest.mark.parametrize("m,n,ranks", [(37, 13, [9]), (64, 72, [20, 12]), (96, 40, [33, 31]), (512, 300, [64, 40])])
@pytest.mark.parametrize("rows", [1, 2, 16, 17, 40])
def test_packed_matches_dense_fp32(m, n, ranks, rows):
    _kernel_or_skip()
    paths, W = _factors(m, n, ranks, seed=m * n)
    op = FactorizedLinear(m, n, _packed_paths(paths), torch.randn(n), torch.float32, "packed")
    x = torch.randn(rows, m)
    ref = x.double() @ W.t() + op.bias.double()
    rel = ((op(x).double() - ref).norm() / ref.norm()).item()
    assert rel < 1e-5, rel


@pytest.mark.parametrize("rows", [1, 17])
def test_packed_bf16_matches_compute_mode(rows):
    _kernel_or_skip()
    paths, W = _factors(256, 128, [64, 40], seed=11)
    x = torch.randn(rows, 256).bfloat16()
    a = FactorizedLinear(256, 128, _packed_paths(paths), None, torch.bfloat16, "packed")(x).double()
    b = FactorizedLinear(256, 128, paths, None, torch.bfloat16, "compute")(x).double()
    ref = x.double() @ W.t()
    ea, eb = ((a - ref).norm() / ref.norm()).item(), ((b - ref).norm() / ref.norm()).item()
    assert ea < 1.5e-2 and ea <= eb * 1.5 + 1e-3, (ea, eb)


def test_packed_scalar_and_avx_paths_agree():
    _kernel_or_skip()
    from littlerun.codecs.littlebit import pack_signs
    from littlerun.kernels import packed_kernel

    k = packed_kernel()
    s = torch.where(torch.randn(70, 100) < 0, -1, 1).to(torch.int8)
    p = pack_signs(s)
    x = torch.nn.functional.pad(torch.randn(5, 100), (0, 28))
    ref = x[:, :100].double() @ s.double().t()
    for force_scalar in (False, True):
        y = k.signed_matmul(p, x, force_scalar).double()
        assert ((y - ref).abs().max() / ref.abs().max()).item() < 1e-5
    assert torch.equal(k.unpack_signs(p, torch.float32)[:, :100].to(torch.int8), s)
    assert torch.equal(k.unpack_signs(p, torch.bfloat16)[:, :100].to(torch.int8), s)
