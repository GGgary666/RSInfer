"""RSInfer worker recipe for the GLM-5.3-Flash W8A8 text model on Ascend."""

from __future__ import annotations

import argparse
import os
import tempfile
import uuid
from importlib.util import find_spec
from pathlib import Path
from typing import Any

NAME = "glm53-native"
SERVED_NAME = "GLM-5.3-Flash"
PAGE_SIZE = 128
DEFAULT_DEVICE = "npu"
WORKER_ONLY = frozenset({"models.glm53", "layers.glm53", "state.gdn_slots", "state.glm53_cache"})


def matches(cfg: dict[str, Any]) -> bool:
    text = cfg.get("text_config", cfg)
    return text.get("model_type") == "glm5_next_text" and bool(text.get("mla_use_nope") and text.get("mhc"))


def add_arguments(group: argparse._ArgumentGroup) -> None:
    group.add_argument("--memcache-client-config", type=Path, help="native GLM MemCache client configuration")
    group.add_argument("--prefix-manifest", type=Path, help="prefix index saved by cache-save")
    group.add_argument("--glm-moe-backend", choices=("grouped", "dfc"), default="grouped",
                       help="routed expert operator for native GLM")
    group.add_argument("--glm-dp-size", type=int, default=1,
                       help="independent scheduling domains with tensor and expert parallelism within each domain")
    group.add_argument("--glm-prefix-peer", action="store_true",
                       help="replicate a shared prefix to an idle HCCL scheduling domain under load")


def mesh(args: argparse.Namespace) -> dict[str, int]:
    ranks, domains = len(args.rank_gpus), args.glm_dp_size
    if domains < 1 or ranks % domains:
        raise SystemExit("--glm-dp-size must divide the number of accelerator ranks")
    return {"dp": domains, "tp": ranks // domains} if domains > 1 else {"tp": ranks}


def worker_env(moe: str = "grouped") -> dict[str, str]:
    roots: list[Path] = []
    for package, relative in (("vllm_ascend", "_cann_ops_custom/vendors/custom_transformer"),
                              ("fla_npu", "opp/vendors/fla_npu_transformer")):
        spec = find_spec(package)
        if spec is None or spec.origin is None:
            raise SystemExit(f"GLM-5.3-Flash requires {package} in its Ascend environment")
        roots.append(Path(spec.origin).parent / relative)
    # CANN resolves custom tiling libraries when it initializes the device, before the unit loads.
    paths = [str(p) for p in roots if p.is_dir()]
    paths.extend(p for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if p)
    env = {"ASCEND_CUSTOM_OPP_PATH": os.pathsep.join(dict.fromkeys(paths)), "RSINFER_GLM_MOE": moe}
    mode = os.environ.get("HCCL_OP_EXPANSION_MODE", "AIV" if moe == "dfc" else "")
    if mode:
        env["HCCL_OP_EXPANSION_MODE"] = mode
    if mode.upper() == "AIV":
        # Each TP rank owns the same sampled token history and must agree on the reduction results.
        env["HCCL_DETERMINISTIC"] = os.environ.get("HCCL_DETERMINISTIC", "true")
    if moe == "dfc":
        env["HCCL_BUFFSIZE"] = os.environ.get("HCCL_BUFFSIZE", "256")
    return env


def config(args: argparse.Namespace, cfg: dict[str, Any], model: Path) -> dict[str, Any]:
    import msgspec

    from models.glm53_config import KDA, KV, Config

    if args.device != "npu" or args.dtype != "bf16" or args.kv_dtype not in ("auto", "bf16"):
        raise SystemExit("GLM-5.3-Flash requires --device npu with BF16 activations and KV")
    if args.layers and args.speculative_algorithm:
        raise SystemExit("GLM-5.3-Flash MTP requires the complete checkpoint")
    if args.dummy:
        raise SystemExit("GLM-5.3-Flash native requires the W8A8 checkpoint")
    if len(set(args.rank_gpus)) != len(args.rank_gpus):
        raise SystemExit("GLM-5.3-Flash HCCL requires one rank per die")
    parsed = Config.from_hf(cfg)
    if not 0 < args.max_context <= parsed.max_context:
        raise SystemExit(f"--max-context must be in 1..{parsed.max_context}")
    ranks, tp = len(args.rank_gpus), mesh(args)["tp"]
    if tp == 1 and args.glm_moe_backend == "dfc":
        raise SystemExit("DFC uses HCCL and needs more than one rank")
    verify = args.speculative_num_draft_tokens if args.speculative_algorithm else 0
    states = parsed.states(tp, args.page_size, verify)
    radix = not args.disable_radix_cache and {d.kind for d in states} == {KDA, KV}
    if args.glm_prefix_peer and (args.glm_dp_size < 2 or not radix):
        raise SystemExit("--glm-prefix-peer requires multiple domains and the hybrid radix cache")
    if args.prefix_manifest is not None and not (args.memcache_client_config and radix):
        raise SystemExit("--prefix-manifest needs MemCache client configuration and the hybrid radix cache")
    for source in (args.memcache_client_config, args.prefix_manifest):
        if source is not None and not source.is_file():
            raise SystemExit(f"{source} is unavailable")
    meet = args.run.resolve() / "comm"
    meet.mkdir(parents=True, exist_ok=True)
    return {
        "alloc.states": msgspec.json.decode(msgspec.json.encode(states)),
        "alloc.radix_cache": radix,
        "alloc.prefix_checkpoint": str(args.prefix_manifest.resolve()) if args.prefix_manifest else "",
        "glm53.memcache_config": str(args.memcache_client_config.resolve()) if args.memcache_client_config else "",
        "frontend.reasoning_end": "</think>",
        "model.dtype": args.dtype, "comm.size": ranks, "comm.device_backend": "hccl",
        **({"comm.domains": args.glm_dp_size} if args.glm_dp_size > 1 else {}),
        **({"comm.prefix_peer": True, "sched.prefix_peer": True} if args.glm_prefix_peer else {}),
        "comm.rendezvous": f"file://{meet / uuid.uuid4().hex}", "sched.domains": args.glm_dp_size,
        **({"model.hf_config": cfg} if args.layers else {"model.path": str(model)}),
    }


def checkpoint(args: argparse.Namespace) -> int:
    raise SystemExit("GLM-5.3-Flash native uses the original W8A8 checkpoint")


async def save_prefix(control: str, world: int, output: Path, timeout: float) -> int:
    import json

    import zmq.asyncio

    from core.change.types import ChangeMeta, StateRequest
    from core.common.ids import UnitId
    from core.control.client import ControlClient, summary
    from core.control.protocol import ApiChange

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rsinfer-prefix-", dir=output.parent) as directory:
        staged = Path(directory) / "checkpoint.json"
        args = {"path": str(staged), "nonce": uuid.uuid4().hex, "world": world, "timeout": timeout}
        change = ApiChange(
            meta=ChangeMeta(intent="Native GLM MemCache prefix checkpoint", when_busy="wait", wait_s=timeout),
            state=[StateRequest(UnitId(unit), "save_prefix", args) for unit in ("core.state.alloc", "state.glm53_cache")])
        zctx = zmq.asyncio.Context()
        client = ControlClient(zctx, control, timeout_s=timeout * 2)
        try:
            result = (await client.apply(change)).unwrap().result
        finally:
            client.close()
            zctx.destroy(linger=0)
        print(json.dumps(summary(result), indent=2))
        if result.status != "committed":
            return 1
        staged.replace(output)
        return 0
