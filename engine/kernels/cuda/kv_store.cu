// kv.store@1 as one kernel: each token's K and V rows copied to their slots of the pool, the job
// of SGLang's store_kvcache (python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh, e838360),
// written here against the PyTorch C++ extension API instead of its tvm-ffi runtime. One block per
// token; its threads copy 16 bytes at a time, K and V in the same pass.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace {

constexpr int64_t kVec = 16;  // bytes a thread moves at once (uint4)

template <typename Loc>
__global__ void store_kv_kernel(const char* __restrict__ k, const char* __restrict__ v, char* __restrict__ k_cache,
                                char* __restrict__ v_cache, const Loc* __restrict__ loc, int64_t slots,
                                int64_t k_stride, int64_t v_stride, int64_t k_cache_stride,
                                int64_t v_cache_stride, int64_t k_vecs, int64_t v_vecs) {
  const int64_t token = blockIdx.x;
  const int64_t slot = static_cast<int64_t>(loc[token]);
  if (slot < 0 || slot >= slots) {
    __trap();  // index_put would fail its bounds check here too
  }
  const uint4* ks = reinterpret_cast<const uint4*>(k + token * k_stride);
  const uint4* vs = reinterpret_cast<const uint4*>(v + token * v_stride);
  uint4* kd = reinterpret_cast<uint4*>(k_cache + slot * k_cache_stride);
  uint4* vd = reinterpret_cast<uint4*>(v_cache + slot * v_cache_stride);
  for (int64_t i = threadIdx.x; i < k_vecs; i += blockDim.x) {
    kd[i] = ks[i];
  }
  for (int64_t i = threadIdx.x; i < v_vecs; i += blockDim.x) {
    vd[i] = vs[i];
  }
}

// [rows, ...] with everything after the first dim dense: the bytes of a row, else -1.
int64_t row_bytes(const torch::Tensor& t) {
  int64_t expected = 1;
  for (int64_t d = t.dim() - 1; d >= 1; --d) {
    if (t.size(d) != 1 && t.stride(d) != expected) {
      return -1;
    }
    expected *= t.size(d);
  }
  return expected * t.element_size();
}

bool aligned(const torch::Tensor& t) {
  return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % kVec == 0 && (t.stride(0) * t.element_size()) % kVec == 0;
}

}  // namespace

// k_cache, v_cache [slots, heads, head_dim]; loc [tokens]; k, v [tokens, heads, head_dim], rows
// possibly strided (views of a projection). Returns false, having done nothing, when the tensors
// are not what the kernel moves (another dtype than the pool's, a row that is not dense or not a
// whole number of 16-byte vectors, misaligned): the caller then stores them another way.
bool store(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor loc, torch::Tensor k, torch::Tensor v) {
  if (!k_cache.is_cuda() || !v_cache.is_cuda() || !loc.is_cuda() || !k.is_cuda() || !v.is_cuda()) {
    return false;
  }
  if (k.scalar_type() != k_cache.scalar_type() || v.scalar_type() != v_cache.scalar_type()) {
    return false;
  }
  if (loc.dim() != 1 || (loc.scalar_type() != torch::kInt64 && loc.scalar_type() != torch::kInt32) ||
      (loc.numel() > 1 && loc.stride(0) != 1)) {
    return false;
  }
  const int64_t tokens = loc.numel();
  if (k.size(0) != tokens || v.size(0) != tokens) {
    return false;
  }
  const int64_t k_row = row_bytes(k), v_row = row_bytes(v);
  if (k_row <= 0 || v_row <= 0 || k_row != row_bytes(k_cache) || v_row != row_bytes(v_cache)) {
    return false;
  }
  if (k_row % kVec != 0 || v_row % kVec != 0) {
    return false;
  }
  if (tokens == 0) {
    return true;
  }
  if (!aligned(k) || !aligned(v) || !aligned(k_cache) || !aligned(v_cache)) {
    return false;
  }
  const c10::cuda::CUDAGuard guard(k_cache.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t k_vecs = k_row / kVec, v_vecs = v_row / kVec;
  const int64_t widest = k_vecs > v_vecs ? k_vecs : v_vecs;
  const int threads = static_cast<int>(widest >= 256 ? 256 : ((widest + 31) / 32) * 32);
  const int64_t slots = k_cache.size(0) < v_cache.size(0) ? k_cache.size(0) : v_cache.size(0);
  const auto* kp = static_cast<const char*>(k.data_ptr());
  const auto* vp = static_cast<const char*>(v.data_ptr());
  auto* kcp = static_cast<char*>(k_cache.data_ptr());
  auto* vcp = static_cast<char*>(v_cache.data_ptr());
  const int64_t ks = k.stride(0) * k.element_size(), vs = v.stride(0) * v.element_size();
  const int64_t kcs = k_cache.stride(0) * k_cache.element_size(), vcs = v_cache.stride(0) * v_cache.element_size();
  if (loc.scalar_type() == torch::kInt64) {
    store_kv_kernel<int64_t><<<static_cast<unsigned>(tokens), threads, 0, stream>>>(
        kp, vp, kcp, vcp, loc.data_ptr<int64_t>(), slots, ks, vs, kcs, vcs, k_vecs, v_vecs);
  } else {
    store_kv_kernel<int32_t><<<static_cast<unsigned>(tokens), threads, 0, stream>>>(
        kp, vp, kcp, vcp, loc.data_ptr<int32_t>(), slots, ks, vs, kcs, vcs, k_vecs, v_vecs);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("store", &store, "kv.store@1: K and V rows to their slots, one kernel");
}
