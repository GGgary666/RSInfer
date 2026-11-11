# Copyright 2023-2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""What the kernels of this unit ask about the GPU. From SGLang e838360 (kernels/arch/_cuda.py,
kernels/jit/utils/arch.py, srt/utils/common.py; LICENSE.sglang), for NVIDIA GPUs only."""

from __future__ import annotations

from functools import cache

import torch


@cache
def get_device_sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@cache
def is_arch_support_pdl() -> bool:
    """``CudaArch.supports_pdl``: programmatic dependent launch, from SM90 on."""
    return get_device_sm() // 10 >= 9


@cache
def is_sm90_supported() -> bool:
    """``_check_cuda_device_version(device_capability_majors=[9], cuda_version=(12, 3))``."""
    cuda = torch.version.cuda
    if cuda is None or get_device_sm() // 10 != 9:
        return False
    major, minor = (int(part) for part in cuda.split(".")[:2])
    return (major, minor) >= (12, 3)
