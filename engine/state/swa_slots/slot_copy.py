# SPDX-License-Identifier: Apache-2.0
"""Adapted from DeepInfer a679bb86, csrc/engine/scheduler.hpp and csrc/cuda_utils.cpp.
Our ledger owns the pins. Disjoint slot moves use a single device grid across the per-layer tensors.
License in engine/policies/eplb/LICENSE.deepinfer.
"""

from functools import cache
from typing import Any

from kernels.cuda_build import build_cuda_extension

SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace {
constexpr int kFields = 64, kMoves = 64, kTile = 8192;
struct Field { uint8_t* data; int64_t bytes; };
struct Move { int64_t src, dst; };
struct Fields { Field f[kFields]; };
struct Moves { Move m[kMoves]; };

// Descriptors and slot IDs travel in the launch parameters. Reading them requires no
// staging tensor or host/device synchronization. A block owns one tile of one destination.
__global__ void slot_tiles(const Fields fields, const Moves moves) {
  const Field field = fields.f[blockIdx.y];
  const Move move = moves.m[blockIdx.z];
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * kTile;
  if (offset >= field.bytes) return;
  const int n = static_cast<int>(field.bytes - offset < kTile ? field.bytes - offset : kTile);
  auto* dst = field.data + move.dst * field.bytes + offset;
  const auto* src = move.src < 0 ? nullptr : field.data + move.src * field.bytes + offset;
  if (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src) | n) & 15) == 0) {
    auto* d = reinterpret_cast<uint4*>(dst);
    const auto* s = reinterpret_cast<const uint4*>(src);
    for (int i = threadIdx.x; i < n / 16; i += blockDim.x)
      d[i] = s ? s[i] : make_uint4(0, 0, 0, 0);
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src ? src[i] : 0;
  }
}

void launch(const std::vector<Field>& fields, const std::vector<Move>& moves, cudaStream_t stream) {
  for (size_t fi = 0; fi < fields.size(); fi += kFields) {
    Fields group{};
    const auto nfields = std::min<size_t>(kFields, fields.size() - fi);
    int64_t largest = 0;
    for (size_t j = 0; j < nfields; ++j) {
      group.f[j] = fields[fi + j];
      largest = std::max(largest, group.f[j].bytes);
    }
    if (largest == 0) continue;
    for (size_t mi = 0; mi < moves.size(); mi += kMoves) {
      Moves batch{};
      const auto n = std::min<size_t>(kMoves, moves.size() - mi);
      std::copy_n(moves.begin() + mi, n, batch.m);
      slot_tiles<<<dim3((largest + kTile - 1) / kTile, nfields, n), 128, 0, stream>>>(group, batch);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
}
} // namespace

void clone_and_zero(const std::vector<torch::Tensor>& buffers,
                    const std::vector<std::pair<int64_t, int64_t>>& moves,
                    const std::vector<int64_t>& zeros) {
  if (buffers.empty() || (moves.empty() && zeros.empty())) return;
  const auto device = buffers.front().device();
  const c10::cuda::CUDAGuard guard(device);
  const auto stream = at::cuda::getCurrentCUDAStream();
  std::vector<Field> fields;
  fields.reserve(buffers.size());
  int64_t capacity = INT64_MAX;
  for (const auto& buf : buffers) {
    TORCH_CHECK(buf.device() == device && buf.dim() >= 1 && buf.is_contiguous() && buf.size(0) > 0,
                "expected contiguous slot tensors on one CUDA device");
    capacity = std::min(capacity, buf.size(0));
    fields.push_back({static_cast<uint8_t*>(buf.data_ptr()), buf.numel() / buf.size(0) * buf.element_size()});
  }
  std::vector<Move> copies, clears;
  std::unordered_set<int64_t> dsts;
  bool ordered = false;
  for (const auto& [s, d] : moves) {
    TORCH_CHECK(s >= 0 && s < capacity && d >= 0 && d < capacity, "invalid slot copy");
    if (s == d) continue;
    if (!dsts.insert(d).second) ordered = true;
    copies.push_back({s, d});
  }
  for (const auto& move : copies) if (dsts.count(move.src)) ordered = true;
  dsts.clear();
  for (int64_t d : zeros) {
    TORCH_CHECK(d >= 0 && d < capacity, "invalid slot clear");
    if (dsts.insert(d).second) clears.push_back({-1, d});
  }
  if (ordered) {
    // Preserve ordered read-after-write and last-write semantics for the generic primitive.
    for (const auto& f : fields) for (const auto& m : copies)
      C10_CUDA_CHECK(cudaMemcpyAsync(f.data + m.dst * f.bytes, f.data + m.src * f.bytes,
                                     f.bytes, cudaMemcpyDeviceToDevice, stream));
  } else if (!copies.empty()) {
    launch(fields, copies, stream);
  }
  if (!clears.empty()) {
    launch(fields, clears, stream);
  }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("clone_and_zero", &clone_and_zero);
}
"""


@cache
def operator() -> Any:
    return build_cuda_extension("rsinfer_swa_slots", "slot_copy.cu", SOURCE)
