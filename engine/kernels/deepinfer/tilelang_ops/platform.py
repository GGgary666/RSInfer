# Copyright DeepInfer authors
# SPDX-License-Identifier: Apache-2.0
"""What the operators here read of DeepInfer's ``PLATFORM``: they run on CUDA only."""

from __future__ import annotations

from typing import Final

import torch


class _Cuda:
    is_gpu: Final = True
    is_npu: Final = False
    fp8_sf_type: Final = torch.int32
    """Four ue8m0 exponents packed into one word."""


PLATFORM: Final = _Cuda()
