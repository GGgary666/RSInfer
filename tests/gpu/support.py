"""Shared by the GPU tests: a Qwen3.8-27B checkpoint cut to a few layers, and HF's model on it.

Importing it turns TF32 off, here and in the engines the tests start (``NVIDIA_TF32_OVERRIDE``):
cuDNN's convolutions default to TF32 on these GPUs, and the f32 comparisons with HF are meant to
be f32 (on a SM100 the first logits of the partial model differed by 8e-3 with it on)."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

REPO = Path(__file__).resolve().parents[2]
TEXT = {  # Qwen3.8-27B's text_config
    "vocab_size": 248320, "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
    "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "linear_num_value_heads": 48,
    "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "full_attention_interval": 4, "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6, "tie_word_embeddings": False, "attn_output_gate": True, "hidden_act": "silu",
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                        "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    "model_type": "qwen3_5_text"}


def make_checkpoint(tmp: pytest.TempPathFactory, layers: int, seed: int = 1) -> Path:
    """Random weights for the first ``layers`` layers, written by ``deploy/agxinfer.py checkpoint``."""
    src = tmp.mktemp("config")
    (src / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"],
                                                 "model_type": "qwen3_5", "text_config": TEXT}))
    out = tmp.mktemp("ckpt")
    subprocess.run([sys.executable, str(REPO / "deploy" / "agxinfer.py"), "checkpoint", "--config", str(src),
                    "--layers", str(layers), "--out", str(out), "--dtype", "bf16", "--seed", str(seed)],
                   env={**os.environ, "PYTHONPATH": str(REPO / "engine")}, check=True, timeout=1800)
    return out


def hf_model_from(checkpoint: Path, dtype: torch.dtype) -> Qwen3_5ForCausalLM:
    """HF's text model on the GPU with a checkpoint's weights (a real one, or ``make_checkpoint``'s)."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    with torch.device("cuda"):
        hf = Qwen3_5ForCausalLM._from_config(  # pyright: ignore[reportPrivateUsage]
            Qwen3_5TextConfig(**{k: v for k, v in text.items() if k != "model_type"}), dtype=dtype).eval()
    weights: dict[str, torch.Tensor] = {}
    for file in sorted(checkpoint.glob("*.safetensors")):
        for name, t in load_file(str(file)).items():  # on the CPU: the GPU holds the model only
            if "visual" not in name and not name.startswith("mtp"):
                weights[name.replace("model.language_model.", "model.")] = t.to(dtype)
    missing, unexpected = hf.load_state_dict(weights, strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing[:5], unexpected[:5])
    return hf


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()
