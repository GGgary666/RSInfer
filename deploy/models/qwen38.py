"""Qwen3.8 (``Qwen3_5ForConditionalGeneration``, ``Qwen3_5ForCausalLM``): what the engine's units
need to serve it (``models.qwen38``, ``state.gdn_slots``), and random-weight checkpoints in its
layout for tests.

A recipe is what ``agxinfer`` knows about one model family: ``matches`` a ``config.json``,
``add_arguments`` for the flags only it has, ``config`` for the config keys only its units read,
and ``checkpoint``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from deploy.models.hf import TOKENIZER_FILES, cut, read_config, text_config

NAME = "qwen38"
SERVED_NAME = "Qwen3.8-27B"
ARCHITECTURES = frozenset({"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"})
MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})


def matches(cfg: dict[str, Any]) -> bool:
    return bool(ARCHITECTURES & set(cfg.get("architectures") or [])) or cfg.get("model_type") in MODEL_TYPES


def add_arguments(group: argparse._ArgumentGroup) -> None:
    group.add_argument("--ssm-dtype", default="auto", help="state.gdn.ssm_dtype (SGLang's --mamba-ssm-dtype)")
    group.add_argument("--conv-dtype", default="bf16", help="state.gdn.conv_dtype (SGLANG_MAMBA_CONV_DTYPE); f32 to "
                       "match an f32 HF reference")


def config(args: argparse.Namespace, cfg: dict[str, Any], model: Path) -> dict[str, Any]:
    """The keys of this family's units. ``cfg`` is the config served (cut to ``--layers``)."""
    import msgspec

    from models.qwen38.config import Qwen3_5TextConfig
    from models.qwen38.unit import states

    decls = states(Qwen3_5TextConfig.from_hf(cfg), args.dtype, args.page_size)
    return {"alloc.states": msgspec.json.decode(msgspec.json.encode(decls)), "model.dtype": args.dtype,
            "state.gdn.ssm_dtype": args.ssm_dtype, "state.gdn.conv_dtype": args.conv_dtype,
            **({"model.hf_config": cfg} if args.layers else {"model.path": str(model)})}


def checkpoint(args: argparse.Namespace) -> int:
    """Random weights in a Qwen3.8 checkpoint's layout (``model.language_model.*``, plus ``visual``
    and ``mtp`` tensors the loader skips), with HF's initialization and a seed, the config cut to
    ``--layers``: for tests that compare against HF."""
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    source = args.config.resolve()
    cfg = read_config(source)
    if args.layers:
        cfg = cut(cfg, args.layers)
    t = text_config(cfg)
    dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    torch.manual_seed(args.seed)
    hf_cfg = Qwen3_5TextConfig(**{k: v for k, v in t.items() if k not in ("model_type",)})
    with torch.device(device):
        model = Qwen3_5ForCausalLM._from_config(hf_cfg, dtype=dtype)  # pyright: ignore[reportPrivateUsage]
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() == 1:  # as tests/units/support/worker.py: a misplaced norm weight shows
                    p.add_(0.1 * torch.randn_like(p))
    multimodal = "text_config" in cfg
    tensors: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if multimodal and name.startswith("model."):
            name = "model.language_model." + name[len("model."):]
        tensors[name] = value.detach().to("cpu").contiguous()
    tensors["mtp.fc.weight"] = torch.zeros(8, 8, dtype=dtype)  # skipped by the loader, as in a real checkpoint
    tensors["model.visual.patch_embed.proj.weight"] = torch.zeros(8, 8, dtype=dtype)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    half = len(names) // 2
    index: dict[str, str] = {}
    for i, part in enumerate((names[:half], names[half:])):
        file = f"model-{i + 1:05d}-of-00002.safetensors"
        save_file({n: tensors[n] for n in part}, str(out / file), metadata={"format": "pt"})
        index |= {n: file for n in part}
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=1))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    tokenizer = (args.tokenizer or source).resolve()
    for name in TOKENIZER_FILES:
        if (tokenizer / name).is_file() and tokenizer != out:
            shutil.copy(tokenizer / name, out / name)
    size = sum(v.numel() * v.element_size() for v in tensors.values())
    print(f"{out}: {t['num_hidden_layers']} layers, {size / 2**30:.2f} GiB of {args.dtype} weights", file=sys.stderr)
    return 0
