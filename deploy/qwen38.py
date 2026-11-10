"""Qwen3.8 on this engine, from the command line: one host, a frontend, the scheduler host and one
worker (TP1), started through ``python -m core.control serve --spawn-agents``.

    uv run --extra gpu --extra frontend python deploy/qwen38.py serve --model <HF dir> [--layers 8] [--dummy]
    uv run --extra gpu --extra frontend python deploy/qwen38.py chat "Hello"
    uv run --extra gpu python deploy/qwen38.py checkpoint --config <HF dir> --layers 4 --out <dir>

- ``plan`` writes ``topology.json`` and ``config.json`` into the run directory and prints the serve
  command; ``serve`` does that and runs it until Ctrl-C.
- ``--layers N`` keeps the first N layers of the model's config (every fourth one full attention,
  as in the full model): a part of the model to run where the whole one does not fit.
- Without ``--weights`` the weights are the model directory's ``*.safetensors``; with ``--dummy``,
  or when there are none, dummy weights (``initialize_dummy_weights``).
- ``checkpoint`` writes random weights for a config (HF's initialization, seeded) in the layout of
  a Qwen3.8 checkpoint (``model.language_model.*``, plus ``visual`` and ``mtp`` tensors the loader
  skips), with the config cut to ``--layers``, for tests that compare against HF.
- Start-up waits up to ``--timeout`` (3600 s) for the processes, whose units load the weights
  before they check in.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "engine"))

HOST = "host-a"
FRONTEND = {"host": HOST, "role": "frontend", "index": 0}
SCHED = {"host": HOST, "role": "sched", "index": 0}
WORKER = {"host": HOST, "role": "worker", "index": 0}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "generation_config.json")


def read_config(model: Path) -> dict[str, Any]:
    return json.loads((model / "config.json").read_text())


def text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("text_config", cfg)


def cut(cfg: dict[str, Any], layers: int) -> dict[str, Any]:
    """The config with only its first ``layers`` decoder layers."""
    out = json.loads(json.dumps(cfg))
    t = text_config(out)
    if not 0 < layers <= t["num_hidden_layers"]:
        raise SystemExit(f"--layers must be in 1..{t['num_hidden_layers']}")
    t["num_hidden_layers"] = layers
    if "layer_types" in t:
        t["layer_types"] = t["layer_types"][:layers]
    return out


# ------------------------------------------------------------------ plan


def plan(args: argparse.Namespace) -> tuple[Path, Path, dict[str, float]]:
    import msgspec

    from core.common.ids import HostId
    from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, Topology
    from models.qwen38.config import Qwen3_5TextConfig
    from models.qwen38.unit import states

    model = args.model.resolve()
    cfg = read_config(model)
    if args.layers:
        cfg = cut(cfg, args.layers)
    weights = "" if args.dummy else str((args.weights or model).resolve())
    if weights and not list(Path(weights).glob("*.safetensors")):
        print(f"no *.safetensors in {weights}: dummy weights", file=sys.stderr)
        weights = ""
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    cuda = args.device == "cuda"
    host = HostId(HOST)
    topology = Topology(
        hosts=[HostSpec(host, args.bind, str(run / HOST / "snapshots"))], mesh=Mesh({"tp": 1}),
        placement=[RankPlacement(0, host, worker_gpu(args.gpu) if cuda else 0)],
        roles=[RoleSpec("frontend", [], hosts=[host]), RoleSpec("sched", [], hosts=[host], leads_epochs=True),
               RoleSpec("worker", [], per_rank=True,
                        platform="cuda_graph" if cuda and args.cuda_graph != "off" else "torch_eager",
                        platform_options={"cuda": 1} if cuda else {})])
    decls = states(Qwen3_5TextConfig.from_hf(cfg), args.dtype, args.page_size)
    config: dict[str, Any] = {
        # frontend
        "frontend.model": args.name, "frontend.tokenizer_path": str((args.tokenizer or model).resolve()),
        "frontend.model_path": str(model.resolve()),
        "frontend.http_bind": args.bind, "frontend.http_port": args.port, "frontend.scheduler": SCHED,
        "frontend.response_timeout_s": float(args.timeout),
        # scheduler host
        "sched.workers": [WORKER], "sched.frontend_host": HOST, "sched.max_batch": args.max_rows,
        "sched.token_budget": args.token_budget, "sched.overlap": args.overlap,
        "alloc.states": msgspec.json.decode(msgspec.json.encode(decls)),
        # worker
        "worker.scheduler": SCHED, "model.dtype": args.dtype, "weights.path": weights,
        "state.kv.page_size": args.page_size, "state.kv.dtype": args.kv_dtype, "state.gdn.ssm_dtype": args.ssm_dtype,
        "state.gdn.conv_dtype": args.conv_dtype, "batch.max_rows": args.max_rows,
        "batch.max_context_len": args.max_context, "holder.mem_fraction": args.mem_fraction,
        "holder.max_slots": args.max_slots, "holder.cpu_bytes": args.cpu_bytes,
        "plan.cuda_graph": args.cuda_graph, "plan.graph_buckets": args.graph_buckets,
        "plan.graph_memory_bytes": args.graph_memory_bytes,
        **({"model.hf_config": cfg} if args.layers else {"model.path": str(model)}),
    }
    topo_file, config_file = run / "topology.json", run / "config.json"
    topo_file.write_bytes(msgspec.json.encode(topology))
    config_file.write_text(json.dumps(config, indent=1))
    tunables = {"supervisor.checkin_timeout_s": float(args.timeout), "change.prepare_timeout_s": float(args.timeout)}
    return topo_file, config_file, tunables


def worker_gpu(requested: int | None) -> int:
    """The physical GPU the topology places the worker on. The node agent sets each engine
    process's ``CUDA_VISIBLE_DEVICES`` from the topology alone, so without ``--gpu`` the caller's
    own ``CUDA_VISIBLE_DEVICES`` decides here, or every engine of a machine lands on GPU 0."""
    if requested is not None:
        return requested
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if not first:
        return 0
    if not first.isdigit():
        raise SystemExit(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}: give the worker's GPU "
                         "by index with --gpu")
    return int(first)


def serve_command(args: argparse.Namespace, topo: Path, config: Path, tunables: dict[str, float]) -> list[str]:
    return [sys.executable, "-m", "core.control", "serve", "--source", str(args.source.resolve()), "--topology", str(topo),
            "--state-dir", str(args.run.resolve() / "control"), "--host", HOST, "--bind", args.bind,
            "--config", str(config), "--spawn-agents", "--platform",
            json.dumps({"name": "cuda_graph" if args.device == "cuda" and args.cuda_graph != "off" else "torch_eager"}),
            "--tunables", json.dumps(tunables)]


def _stop_with_parent() -> None:
    """The engine stops (SIGTERM, an orderly shutdown) when this command goes, however it goes: an
    engine nobody holds keeps its GPU memory."""
    if sys.platform == "linux":
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(REPO / "engine")}


def cmd_plan(args: argparse.Namespace) -> int:
    topo, config, tunables = plan(args)
    print(" ".join(f"'{a}'" if " " in a or "{" in a else a for a in serve_command(args, topo, config, tunables)))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    topo, config, tunables = plan(args)
    control = args.run.resolve() / "control" / "control.json"
    control.unlink(missing_ok=True)
    proc = subprocess.Popen(serve_command(args, topo, config, tunables), env=env(), start_new_session=True,
                            preexec_fn=_stop_with_parent)
    signal.signal(signal.SIGINT, lambda *_: proc.send_signal(signal.SIGTERM))
    signal.signal(signal.SIGTERM, lambda *_: proc.send_signal(signal.SIGTERM))
    while proc.poll() is None and not control.exists():
        time.sleep(0.5)
    if control.exists():
        ready = json.loads(control.read_text())
        ok = isinstance(ready["result"], dict) and ready["result"].get("status") == "committed"
        print(f"engine {'up' if ok else 'failed'}: http://{args.bind}:{args.port} (control {ready['control']})",
              file=sys.stderr, flush=True)
    return proc.wait()


# ------------------------------------------------------------------ chat


def cmd_chat(args: argparse.Namespace) -> int:
    """One request, its answer streamed to stdout (server-sent events)."""
    if args.completion:
        path, body = "/v1/completions", {"prompt": args.prompt}
    else:
        path, body = "/v1/chat/completions", {"messages": [{"role": "user", "content": args.prompt}]}
    body |= {"model": args.name, "max_tokens": args.max_tokens, "temperature": args.temperature, "stream": True}
    request = urllib.request.Request(args.url.rstrip("/") + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data: "):])
            if "error" in event:
                print(f"\nerror: {event['error']}", file=sys.stderr)
                return 1
            choice = event["choices"][0]
            text = choice.get("text") if args.completion else (choice.get("delta") or {}).get("content")
            print(text or "", end="", flush=True)
            if choice.get("finish_reason"):
                print(f"\n[{choice['finish_reason']}]", file=sys.stderr)
    return 0


# ------------------------------------------------------------------ checkpoint


def cmd_checkpoint(args: argparse.Namespace) -> int:
    """Random weights in a Qwen3.8 checkpoint's layout, with HF's initialization and a seed."""
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


# ------------------------------------------------------------------ main


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="deploy/qwen38.py", description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("plan", "serve"):
        s = sub.add_parser(name, help="write topology and config" if name == "plan" else "plan, then run the engine")
        s.add_argument("--model", required=True, type=Path, help="HF model directory (config.json, tokenizer)")
        s.add_argument("--weights", type=Path, help="directory of *.safetensors (default: the model directory)")
        s.add_argument("--dummy", action="store_true", help="dummy weights")
        s.add_argument("--tokenizer", type=Path, help="tokenizer directory (default: the model directory)")
        s.add_argument("--layers", type=int, default=0, help="keep only the first N decoder layers")
        s.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        s.add_argument("--gpu", type=int, help="the worker's CUDA device, by physical index (default: the first "
                       "of CUDA_VISIBLE_DEVICES, else 0)")
        s.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
        s.add_argument("--page-size", type=int, default=16)
        s.add_argument("--kv-dtype", default="auto", help="state.kv.dtype (SGLang's --kv-cache-dtype)")
        s.add_argument("--ssm-dtype", default="auto", help="state.gdn.ssm_dtype (SGLang's --mamba-ssm-dtype)")
        s.add_argument("--conv-dtype", default="bf16", help="state.gdn.conv_dtype (SGLANG_MAMBA_CONV_DTYPE); f32 to match "
                       "an f32 HF reference")
        s.add_argument("--max-rows", type=int, default=64, help="running requests at most")
        s.add_argument("--max-context", type=int, default=4096, help="tokens of one request, prompt and output")
        s.add_argument("--token-budget", type=int, default=2048, help="prompt tokens per step")
        s.add_argument("--mem-fraction", type=float, default=0.85, help="device share of weights and states")
        s.add_argument("--max-slots", type=int, default=0, help="GDN slots (0: by holder.state_ratio)")
        s.add_argument("--cpu-bytes", type=int, default=1 << 30, help="state memory on a CPU device")
        s.add_argument("--overlap", action="store_true", help="one step of overlap (sched.overlap)")
        s.add_argument("--cuda-graph", choices=("auto", "off", "required"), default="auto")
        s.add_argument("--graph-buckets", type=int, nargs="*", default=[], help="decode row capacities")
        s.add_argument("--graph-memory-bytes", type=int, default=1 << 30, help="graph construction memory budget")
        s.add_argument("--name", default="Qwen3.8-27B", help="model name the frontend answers with")
        s.add_argument("--bind", default="127.0.0.1")
        s.add_argument("--port", type=int, default=8000)
        s.add_argument("--run", type=Path, default=Path("run/qwen38"), help="state and snapshot directory")
        s.add_argument("--source", type=Path, default=REPO,
                       help="workspace whose engine/ the engine starts from (default: this repo); change it and "
                            "apply with python -m core.control.client apply --source, without a restart")
        s.add_argument("--timeout", type=int, default=3600, help="seconds for start-up and for a response")
    c = sub.add_parser("chat", help="send one request and stream the answer")
    c.add_argument("prompt")
    c.add_argument("--url", default="http://127.0.0.1:8000")
    c.add_argument("--completion", action="store_true", help="/v1/completions instead of chat")
    c.add_argument("--max-tokens", type=int, default=128)
    c.add_argument("--temperature", type=float, default=0.0)
    c.add_argument("--name", default="Qwen3.8-27B")
    c.add_argument("--timeout", type=int, default=3600)
    k = sub.add_parser("checkpoint", help="random weights in a Qwen3.8 checkpoint's layout")
    k.add_argument("--config", required=True, type=Path, help="directory with the config.json to follow")
    k.add_argument("--layers", type=int, default=0)
    k.add_argument("--out", required=True, type=Path)
    k.add_argument("--tokenizer", type=Path, help="copy the tokenizer files from here (default: --config)")
    k.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    k.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="where to initialize")
    k.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "serve": cmd_serve, "chat": cmd_chat, "checkpoint": cmd_checkpoint}[args.command](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
