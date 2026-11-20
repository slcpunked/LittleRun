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

// M == 1 (decode): 8 independent accumulators hide the add latency.
__attribute__((target("avx512f")))
void avx512_row1(const uint32_t* bits, const float* x, float rowsum, float* out,
                 int64_t r0, int64_t r1, int64_t W) {
  for (int64_t r = r0; r < r1; ++r) {
    const uint32_t* row = bits + r * W;
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps(), a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
    int64_t w = 0;
    for (; w + 4 <= W; w += 4) {
      const uint32_t b0 = row[w], b1 = row[w + 1], b2 = row[w + 2], b3 = row[w + 3];
      const float* xw = x + w * 32;
      a0 = _mm512_mask_add_ps(a0, (__mmask16)(b0 & 0xFFFFu), a0, _mm512_loadu_ps(xw));
      a1 = _mm512_mask_add_ps(a1, (__mmask16)(b0 >> 16), a1, _mm512_loadu_ps(xw + 16));
      a2 = _mm512_mask_add_ps(a2, (__mmask16)(b1 & 0xFFFFu), a2, _mm512_loadu_ps(xw + 32));
      a3 = _mm512_mask_add_ps(a3, (__mmask16)(b1 >> 16), a3, _mm512_loadu_ps(xw + 48));
      a4 = _mm512_mask_add_ps(a4, (__mmask16)(b2 & 0xFFFFu), a4, _mm512_loadu_ps(xw + 64));
      a5 = _mm512_mask_add_ps(a5, (__mmask16)(b2 >> 16), a5, _mm512_loadu_ps(xw + 80));
      a6 = _mm512_mask_add_ps(a6, (__mmask16)(b3 & 0xFFFFu), a6, _mm512_loadu_ps(xw + 96));
      a7 = _mm512_mask_add_ps(a7, (__mmask16)(b3 >> 16), a7, _mm512_loadu_ps(xw + 112));
    }
    for (; w < W; ++w) {
      const uint32_t b = row[w];
      const float* xw = x + w * 32;
      a0 = _mm512_mask_add_ps(a0, (__mmask16)(b & 0xFFFFu), a0, _mm512_loadu_ps(xw));
      a1 = _mm512_mask_add_ps(a1, (__mmask16)(b >> 16), a1, _mm512_loadu_ps(xw + 16));
    }
    const __m512 sum = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)),
                                     _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7)));
    out[r] = rowsum - 2.f * _mm512_reduce_add_ps(sum);
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

// AVX2 path (CPUs without AVX-512): expand each byte of sign bits to an 8-lane mask.
__attribute__((target("avx2")))
static inline __m256 byte_mask(uint32_t byte) {
  const __m256i sel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  const __m256i m = _mm256_and_si256(_mm256_set1_epi32((int)byte), sel);
  return _mm256_castsi256_ps(_mm256_cmpeq_epi32(m, sel));
}

__attribute__((target("avx2")))
static inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
  return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2")))
void avx2_block(const uint32_t* bits, const float* x, const float* rowsum, float* out,
                int64_t r0, int64_t r1, int64_t M, int64_t W, int64_t R) {
  const int64_t K = W * 32;
  for (int64_t m = 0; m < M; ++m) {
    const float* xm = x + m * K;
    for (int64_t r = r0; r < r1; ++r) {
      const uint32_t* row = bits + r * W;
      __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
      for (int64_t w = 0; w < W; ++w) {
        const uint32_t b = row[w];
        const float* xw = xm + w * 32;
        a0 = _mm256_add_ps(a0, _mm256_and_ps(_mm256_loadu_ps(xw), byte_mask(b & 0xFFu)));
        a1 = _mm256_add_ps(a1, _mm256_and_ps(_mm256_loadu_ps(xw + 8), byte_mask((b >> 8) & 0xFFu)));
        a2 = _mm256_add_ps(a2, _mm256_and_ps(_mm256_loadu_ps(xw + 16), byte_mask((b >> 16) & 0xFFu)));
        a3 = _mm256_add_ps(a3, _mm256_and_ps(_mm256_loadu_ps(xw + 24), byte_mask(b >> 24)));
      }
      const float neg = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
      out[m * R + r] = rowsum[m] - 2.f * neg;
    }
  }
}

bool has_avx2() {
  static const bool v = __builtin_cpu_supports("avx2");
  return v;
}

bool has_avx512() {
  static const bool v = __builtin_cpu_supports("avx512f");
  return v;
}

}  // namespace

// path: 0 = best available, 1 = scalar, 2 = AVX2, 3 = AVX-512
torch::Tensor signed_matmul(torch::Tensor packed, torch::Tensor x, int64_t path) {
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
  int64_t use = path;
  if (use == 0) use = has_avx512() ? 3 : (has_avx2() ? 2 : 1);
  TORCH_CHECK(use != 3 || has_avx512(), "AVX-512 not supported on this CPU");
  TORCH_CHECK(use != 2 || has_avx2(), "AVX2 not supported on this CPU");
  at::parallel_for(0, R, kRowBlock, [&](int64_t r0, int64_t r1) {
    if (use == 3 && M == 1) avx512_row1(bits, xp, rs[0], op, r0, r1, W);
    else if (use == 3) avx512_block(bits, xp, rs, op, r0, r1, M, W, R);
    else if (use == 2) avx2_block(bits, xp, rs, op, r0, r1, M, W, R);
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
bool avx2_available() { return has_avx2(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("signed_matmul", &signed_matmul, "packed-sign matmul (fp32 accumulate)");
  m.def("unpack_signs", &unpack_signs, "expand packed signs to dense +-1");
  m.def("avx512_available", &avx512_available);
  m.def("avx2_available", &avx2_available);
}
