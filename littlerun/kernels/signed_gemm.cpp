// Packed-sign matrix product for LittleBit factors (CPU).
//
// out[m, r] = sum_k sign(r, k) * x[m, k]
//   sign(r, k) = +1 if bit k of row r is 0, -1 if it is 1
//   bits: int32 words, row-major, LSB first ("littlebit-int32-lsb-v1")
//
// Computed as rowsum(x[m]) - 2 * sum_{k: bit=1} x[m, k] with fp32 accumulation.
// x must be fp32, contiguous, and zero-padded to words*32 columns.
// AVX-512 path: one masked add per 16 activations; scalar path elsewhere.

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace {

constexpr int64_t kRowBlock = 32;  // weight rows per task (mask block stays in L1)
constexpr int64_t kMTile = 4;      // activation rows per inner tile

void scalar_block(const uint32_t* bits, const float* x, const float* rowsum, float* out,
                  int64_t r0, int64_t r1, int64_t M, int64_t W, int64_t R) {
  const int64_t K = W * 32;
  for (int64_t m = 0; m < M; ++m) {
    const float* xm = x + m * K;
    for (int64_t r = r0; r < r1; ++r) {
      const uint32_t* row = bits + r * W;
      float neg = 0.f;
      for (int64_t w = 0; w < W; ++w) {
        uint32_t b = row[w];
        const float* xw = xm + w * 32;
        while (b) {
          int j = __builtin_ctz(b);
          neg += xw[j];
          b &= b - 1;
        }
      }
      out[m * R + r] = rowsum[m] - 2.f * neg;
    }
  }
}

__attribute__((target("avx512f")))
void avx512_block(const uint32_t* bits, const float* x, const float* rowsum, float* out,
                  int64_t r0, int64_t r1, int64_t M, int64_t W, int64_t R) {
  const int64_t K = W * 32;
  for (int64_t m0 = 0; m0 < M; m0 += kMTile) {
    const int64_t mt = std::min(kMTile, M - m0);
    for (int64_t r = r0; r < r1; ++r) {
      const uint32_t* row = bits + r * W;
      __m512 lo[kMTile], hi[kMTile];
      for (int64_t i = 0; i < kMTile; ++i) { lo[i] = _mm512_setzero_ps(); hi[i] = _mm512_setzero_ps(); }
      for (int64_t w = 0; w < W; ++w) {
        const uint32_t b = row[w];
        if (!b) continue;
        const __mmask16 ml = (__mmask16)(b & 0xFFFFu);
        const __mmask16 mh = (__mmask16)(b >> 16);
        for (int64_t i = 0; i < mt; ++i) {
          const float* xw = x + (m0 + i) * K + w * 32;
          lo[i] = _mm512_mask_add_ps(lo[i], ml, lo[i], _mm512_loadu_ps(xw));
          hi[i] = _mm512_mask_add_ps(hi[i], mh, hi[i], _mm512_loadu_ps(xw + 16));
        }
      }
      for (int64_t i = 0; i < mt; ++i) {
        const float neg = _mm512_reduce_add_ps(_mm512_add_ps(lo[i], hi[i]));
        out[(m0 + i) * R + r] = rowsum[m0 + i] - 2.f * neg;
      }
    }
  }
}

bool has_avx512() {
  static const bool v = __builtin_cpu_supports("avx512f");
  return v;
}

}  // namespace

torch::Tensor signed_matmul(torch::Tensor packed, torch::Tensor x, bool force_scalar) {
  TORCH_CHECK(packed.device().is_cpu() && x.device().is_cpu(), "CPU tensors required");
  TORCH_CHECK(packed.scalar_type() == torch::kInt32 && packed.dim() == 2, "packed must be 2-D int32");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && x.dim() == 2, "x must be 2-D float32");
  TORCH_CHECK(packed.is_contiguous() && x.is_contiguous(), "contiguous inputs required");
  const int64_t R = packed.size(0), W = packed.size(1), M = x.size(0);
  TORCH_CHECK(x.size(1) == W * 32, "x must be zero-padded to words*32 columns");
  auto out = torch::empty({M, R}, x.options());
  auto rowsum = x.sum(1).contiguous();
  const uint32_t* bits = reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>());
  const float* xp = x.data_ptr<float>();
  const float* rs = rowsum.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const bool vec = has_avx512() && !force_scalar;
  at::parallel_for(0, R, kRowBlock, [&](int64_t r0, int64_t r1) {
    if (vec) avx512_block(bits, xp, rs, op, r0, r1, M, W, R);
    else scalar_block(bits, xp, rs, op, r0, r1, M, W, R);
  });
  return out;
}

// Expand packed signs to a dense +-1 matrix [R, W*32] (float32 or bfloat16) for GEMM-based prefill.
// Uses a 256-entry byte table; callers narrow the result to the logical K columns.
template <typename T>
static void expand(const uint32_t* bits, T* o, int64_t R, int64_t W, T pos, T neg) {
  static thread_local T lut[256][8];
  static thread_local bool init = false;
  if (!init) {
    for (int b = 0; b < 256; ++b)
      for (int j = 0; j < 8; ++j) lut[b][j] = ((b >> j) & 1) ? neg : pos;
    init = true;
  }
  const int64_t K = W * 32;
  for (int64_t r = 0; r < R; ++r) {
    const uint32_t* row = bits + r * W;
    T* orow = o + r * K;
    for (int64_t w = 0; w < W; ++w) {
      const uint32_t v = row[w];
      T* dst = orow + w * 32;
      std::memcpy(dst, lut[v & 0xFF], 8 * sizeof(T));
      std::memcpy(dst + 8, lut[(v >> 8) & 0xFF], 8 * sizeof(T));
      std::memcpy(dst + 16, lut[(v >> 16) & 0xFF], 8 * sizeof(T));
      std::memcpy(dst + 24, lut[v >> 24], 8 * sizeof(T));
    }
  }
}

torch::Tensor unpack_signs(torch::Tensor packed, torch::ScalarType dtype) {
  TORCH_CHECK(packed.scalar_type() == torch::kInt32 && packed.dim() == 2 && packed.is_contiguous(),
              "packed must be contiguous 2-D int32");
  TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kBFloat16, "dtype must be float32 or bfloat16");
  const int64_t R = packed.size(0), W = packed.size(1);
  auto out = torch::empty({R, W * 32}, packed.options().dtype(dtype));
  const uint32_t* bits = reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>());
  at::parallel_for(0, R, 64, [&](int64_t r0, int64_t r1) {
    if (dtype == torch::kBFloat16) {
      uint16_t* o = reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()) + r0 * W * 32;
      expand<uint16_t>(bits + r0 * W, o, r1 - r0, W, 0x3F80u, 0xBF80u);
    } else {
      expand<float>(bits + r0 * W, out.data_ptr<float>() + r0 * W * 32, r1 - r0, W, 1.f, -1.f);
    }
  });
  return out;
}

bool avx512_available() { return has_avx512(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("signed_matmul", &signed_matmul, "packed-sign matmul (fp32 accumulate)");
  m.def("unpack_signs", &unpack_signs, "expand packed signs to dense +-1");
  m.def("avx512_available", &avx512_available);
}
