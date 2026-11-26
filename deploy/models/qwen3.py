"""Deployment recipe for Qwen3ForCausalLM on Ascend and CPU."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

NAME = "qwen3"
SERVED_NAME = "Qwen3-0.6B"
PAGE_SIZE = 128
DEFAULT_DEVICE = "npu"
WORKER_ONLY = frozenset({"models.qwen3", "kernels.ascend", "layers.attn", "layers.linear", "layers.norm",
                         "layers.rope", "layers.activation"})


def matches(cfg: dict[str, Any]) -> bool:
    return cfg.get("model_type") == "qwen3" or "Qwen3ForCausalLM" in (cfg.get("architectures") or [])


def add_arguments(group: argparse._ArgumentGroup) -> None:
    pass


def config(args: argparse.Namespace, cfg: dict[str, Any], model: Path) -> dict[str, Any]:
    import msgspec

    from models.qwen3.config import Config

    parsed = Config.from_hf(cfg)
    if args.speculative_algorithm:
        raise SystemExit("Qwen3 does not provide a speculative draft model")
    if not 0 < args.max_context <= parsed.max_positions:
        raise SystemExit(f"--max-context must be in 1..{parsed.max_positions}")
    if args.device == "npu" and args.cuda_graph != "off":
        raise SystemExit("Qwen3 on Ascend supports eager execution. Set --cuda-graph off")
    states = parsed.states(args.dtype, args.page_size, args.kv_dtype)
    return {"alloc.states": msgspec.json.decode(msgspec.json.encode(states)), "model.dtype": args.dtype,
            **({"model.hf_config": cfg} if args.layers else {"model.path": str(model)})}


def checkpoint(args: argparse.Namespace) -> int:
    raise SystemExit("Qwen3 uses the Hugging Face checkpoint or serve --dummy")
