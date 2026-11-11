# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang e838360 (kernels/jit/csrc/elementwise/kvcache.cuh; LICENSE.sglang): its
# store_kvcache kernel (the split of a row over warps, 16-byte loads and stores, where it waits for
# and releases its neighbours under programmatic dependent launch), written against PyTorch's
# extension headers in place of SGLang's own JIT headers.
"""CUDA C++ source of ``kv.store@1``, kept in a Python module so ``core.deps``'s import graph sees
every edit to the C++/CUDA kernel and reloads ``kernels.cuda`` + ``core.plan`` in place."""

from __future__ import annotations

CUDA_SOURCE = r"""
// kv.store@1 as one kernel: each token's K and V rows copied to their slots of the pool, matching
// SGLang's store_kvcache (python/sglang/kernels/jit/csrc/elementwise/kvcache.cuh, e838360) with
// multi-warp 16-byte vector loads/stores and SM90+/SM100 Programmatic Dependent Launch (PDL).
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int64_t kVec = 16;  // bytes per uint4

template <typename Loc, bool kUsePDL>
__global__ void store_kv_kernel(const char* __restrict__ k, const char* __restrict__ v,
                                char* __restrict__ k_cache, char* __restrict__ v_cache,
                                const Loc* __restrict__ loc, int64_t tokens, int64_t slots,
                                int64_t warps_per_token, int64_t k_stride, int64_t v_stride,
                                int64_t k_cache_stride, int64_t v_cache_stride,
                                int64_t k_vecs, int64_t v_vecs) {
  const int64_t warp_id = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  const int64_t token = warp_id / warps_per_token;
  if (token >= tokens) {
    return;
  }
  const int64_t split = warp_id % warps_per_token;
  const int64_t tid = split * 32 + threadIdx.x;
  const int64_t width = warps_per_token * 32;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
#endif

  const int64_t slot = static_cast<int64_t>(loc[token]);
  if (slot < 0 || slot >= slots) {
    __trap();
  }
  const uint4* __restrict__ ks = reinterpret_cast<const uint4*>(k + token * k_stride);
  const uint4* __restrict__ vs = reinterpret_cast<const uint4*>(v + token * v_stride);
  uint4* __restrict__ kd = reinterpret_cast<uint4*>(k_cache + slot * k_cache_stride);
  uint4* __restrict__ vd = reinterpret_cast<uint4*>(v_cache + slot * v_cache_stride);

  if (k_vecs <= width && v_vecs <= width) {
    uint4 kv{0, 0, 0, 0};
    uint4 vv{0, 0, 0, 0};
    if (tid < k_vecs) {
      kv = ks[tid];
    }
    if (tid < v_vecs) {
      vv = vs[tid];
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr (kUsePDL) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
    if (tid < k_vecs) {
      kd[tid] = kv;
    }
    if (tid < v_vecs) {
      vd[tid] = vv;
    }
  } else {
    for (int64_t i = tid; i < k_vecs; i += width) {
      kd[i] = ks[i];
    }
    for (int64_t i = tid; i < v_vecs; i += width) {
      vd[i] = vs[i];
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr (kUsePDL) {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
  }
}

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

int sm_major(int dev) {
  static int cache[64] = {0};
  if (dev >= 0 && dev < 64 && cache[dev] > 0) {
    return cache[dev];
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (dev >= 0 && dev < 64 && major > 0) {
    cache[dev] = major;
  }
  return major;
}

template <typename Kernel, typename... Args>
void launch_store(dim3 grid, dim3 block, cudaStream_t stream, bool use_pdl, Kernel kernel, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr{};
  if (use_pdl) {
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
  } else {
    cfg.attrs = nullptr;
    cfg.numAttrs = 0;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}

}  // namespace

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
  const int dev = k_cache.get_device();
  const c10::cuda::CUDAGuard guard(dev);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev).stream();
  const int64_t k_vecs = k_row / kVec, v_vecs = v_row / kVec;
  const int64_t widest = k_vecs > v_vecs ? k_vecs : v_vecs;
  int64_t warps_per_token = (widest + 31) / 32;
  if (warps_per_token > 8) {
    warps_per_token = 8;
  }
  if (warps_per_token < 1) {
    warps_per_token = 1;
  }
  const int64_t total_warps = tokens * warps_per_token;
  const unsigned warps_per_block = total_warps <= 256 ? 1u : (total_warps <= 1024 ? 2u : 4u);
  const dim3 block(32u, warps_per_block, 1u);
  const dim3 grid(static_cast<unsigned>((total_warps + warps_per_block - 1) / warps_per_block), 1u, 1u);
  const int64_t slots = k_cache.size(0) < v_cache.size(0) ? k_cache.size(0) : v_cache.size(0);
  const auto* kp = static_cast<const char*>(k.data_ptr());
  const auto* vp = static_cast<const char*>(v.data_ptr());
  auto* kcp = static_cast<char*>(k_cache.data_ptr());
  auto* vcp = static_cast<char*>(v_cache.data_ptr());
  const int64_t ks = k.stride(0) * k.element_size(), vs = v.stride(0) * v.element_size();
  const int64_t kcs = k_cache.stride(0) * k_cache.element_size(), vcs = v_cache.stride(0) * v_cache.element_size();
  const bool use_pdl = sm_major(dev) >= 9;
  if (loc.scalar_type() == torch::kInt64) {
    if (use_pdl) {
      launch_store(grid, block, stream, true, store_kv_kernel<int64_t, true>,
                   kp, vp, kcp, vcp, loc.data_ptr<int64_t>(), tokens, slots, warps_per_token, ks, vs, kcs, vcs, k_vecs,
                   v_vecs);
    } else {
      launch_store(grid, block, stream, false, store_kv_kernel<int64_t, false>,
                   kp, vp, kcp, vcp, loc.data_ptr<int64_t>(), tokens, slots, warps_per_token, ks, vs, kcs, vcs, k_vecs,
                   v_vecs);
    }
  } else {
    if (use_pdl) {
      launch_store(grid, block, stream, true, store_kv_kernel<int32_t, true>,
                   kp, vp, kcp, vcp, loc.data_ptr<int32_t>(), tokens, slots, warps_per_token, ks, vs, kcs, vcs, k_vecs,
                   v_vecs);
    } else {
      launch_store(grid, block, stream, false, store_kv_kernel<int32_t, false>,
                   kp, vp, kcp, vcp, loc.data_ptr<int32_t>(), tokens, slots, warps_per_token, ks, vs, kcs, vcs, k_vecs,
                   v_vecs);
    }
  }
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("store", &store, "kv.store@1: K and V rows to their slots, one kernel with SM90+ PDL");
}
"""
