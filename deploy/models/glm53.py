"""GLM-5.3-Flash W8A8 on vLLM Ascend with a persistent AscendStore MemCache pool."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import shutil
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from deploy.local_process import Service, write_json
from deploy.memcache import CLIENT, MemCache, Settings, port_open
from deploy.models.hf import read_config

NAME = "GLM-5.3-Flash"
ARCHITECTURE = "Glm5NextForConditionalGeneration"
REPO = Path(__file__).resolve().parents[2]
PROFILE_ENV = {
    "VLLM_USE_V2_MODEL_RUNNER": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_BUFFSIZE": "3328",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "MALLOC_ARENA_MAX": "1",
    "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT": "480",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3000",
    "GLM_OPT_ASYNC_SPAWN": "1",
    "GLM_OPT_MTP_SHARD_FILTER": "1",
    "GLM_OPT_MOE_PLAIN_ADD": "1",
    "GLM_OPT_FAST_DETOK": "1",
    "GLM_OPT_KDA_FUSED_FG_BMM": "1",
    "GLM_OPT_FAST_CHAT_API": "1",
    "GLM_OPT_TRITON_MHC": "0",
    "GLM_OPT_FAST_KPOOL_DECODE": "0",
    "GLM_OPT_FAST_CP_CUMSUM": "0",
    "GLM_OPT_KDA_ZERO_COPY_CONV": "0",
    "GLM_OPT_INDEXER_REUSE": "0",
}


@dataclass(frozen=True)
class Launch:
    model: str
    hooks: str
    host: str
    port: int
    devices: str
    dp: int
    tp: int
    max_batched_tokens: int
    max_seqs: int
    max_model_len: int
    mem_util: float
    timeout: float
    cache: Settings

    @property
    def url(self) -> str:
        addr = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(self.host, self.host)
        return f"http://{f'[{addr}]' if ':' in addr else addr}:{self.port}"

    def validate(self) -> None:
        if ARCHITECTURE not in read_config(Path(self.model)).get("architectures", []):
            raise ValueError(f"{self.model} must be the {ARCHITECTURE} weight directory")
        if (min(self.dp, self.tp, self.max_batched_tokens, self.max_seqs, self.max_model_len, self.timeout) <= 0
                or not 0 < self.mem_util < 1 or not 0 < self.port <= 65535):
            raise ValueError("GLM launch sizes, timeout, memory utilization or port are out of range")
        if self.devices:
            chosen = self.devices.split(",")
            if len(chosen) != self.dp * self.tp or len(set(chosen)) != len(chosen) or not all(
                i.isdecimal() for i in chosen
            ):
                raise ValueError("--npus needs one distinct non-negative index per DP × TP rank")
        if self.hooks and not (Path(self.hooks) / "sitecustomize.py").is_file():
            raise ValueError(f"{self.hooks} does not contain sitecustomize.py")

    def command(self, cache: MemCache) -> list[str]:
        vllm = shutil.which("vllm")
        if vllm is None:
            raise RuntimeError("vllm executable was not found in the Ascend runtime")
        # The GLM optimization reference uses 6 target slots per request (5 MTP drafts).
        graph_sizes = [n for n in (24, 48, 72, 96, 144, 192, 288, 384) if n <= self.max_seqs * 6]
        graph = {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": graph_sizes or [self.max_seqs * 6]}
        spec = {"num_speculative_tokens": 5, "method": "mtp", "enforce_eager": False,
                "use_local_argmax_reduction": True}
        ascend = {"multistream_overlap_shared_expert": True, "enable_prefill_mc2": True,
                  "enable_dsa_cp": True, "enable_shared_expert_dp": True}
        cmd = [vllm, "serve", self.model, "--host", self.host, "--port", str(self.port),
               "--served-model-name", NAME, "--max-model-len", str(self.max_model_len),
               "--data-parallel-size", str(self.dp), "--tensor-parallel-size", str(self.tp),
               "--enable-expert-parallel", "--seed", "1024", "--safetensors-load-strategy", "lazy",
               "--max-num-seqs", str(self.max_seqs), "--max-num-batched-tokens", str(self.max_batched_tokens),
               "--trust-remote-code", "--quantization", "ascend", "--enable-prefix-caching",
               "--enable-prompt-tokens-details", "--tool-call-parser", "glm47", "--reasoning-parser", "glm47",
               "--enable-auto-tool-choice", "--kv-cache-dtype", "int8", "--limit-mm-per-prompt",
               '{"image":8,"video":0}', "--gpu-memory-utilization", str(self.mem_util),
               "--speculative-config", json.dumps(spec), "--compilation-config", json.dumps(graph),
               "--additional-config", json.dumps(ascend), "--async-scheduling", "--api-server-count", "1"]
        if cache.enabled:
            cmd += ["--kv-transfer-config", json.dumps(
                {"kv_connector": "AscendStoreConnector", "kv_role": "kv_both",
                 "kv_connector_extra_config": {"backend": "memcache"}})]
        return cmd

    def environment(self, cache: MemCache) -> dict[str, str]:
        env = {**PROFILE_ENV, **os.environ, "PYTHONUNBUFFERED": "1"}
        if self.devices:
            env["ASCEND_RT_VISIBLE_DEVICES"] = self.devices
        if self.hooks:
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (self.hooks, env.get("PYTHONPATH", ""))))
        if cache.enabled:
            env["MMC_LOCAL_CONFIG_PATH"] = str(cache.client_path)
        return env


def request(url: str, key: str, timeout: float = 2, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    req = urllib.request.Request(url, headers=headers, data=json.dumps(payload).encode() if payload is not None else None)
    with CLIENT.open(req, timeout=timeout) as response:
        return json.load(response)


def api_ready(url: str, key: str) -> bool:
    try:
        return any(entry["id"] == NAME for entry in request(f"{url}/v1/models", key).get("data", []))
    except (OSError, ValueError):
        return False


def api_key(path: Path) -> str:
    if not path.is_file():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(secrets.token_urlsafe(32) + "\n")
    return path.read_text().strip()


def launch(run: Path, opts: Launch, *, restart: bool, fresh_cache: bool, dry_run: bool) -> int:
    opts.validate()
    engine = Service(run, "vllm")
    cache = MemCache(run / "memcache", opts.cache, REPO)
    cmd = opts.command(cache)
    env = opts.environment(cache)
    if dry_run:
        print(json.dumps({"argv": cmd, "launch": asdict(opts)}, indent=2))
        return 0
    if fresh_cache and not cache.managed:
        raise ValueError("--refresh-memcache requires a pool managed by this run directory")
    start = time.monotonic()
    if restart:
        engine.stop()
    elif engine.running:
        raise RuntimeError("GLM is already running in this directory. Use glm53 restart")
    if port_open(opts.url):
        raise RuntimeError(f"GLM endpoint {opts.url} is in use")
    write_json(run / "launch-attempt.json", asdict(opts))
    if fresh_cache:
        cache.stop()
    cache_start = time.monotonic()
    cache_mode = cache.prepare(opts.timeout)
    cache_seconds = time.monotonic() - cache_start
    key = api_key(run / "api.key")
    env["VLLM_API_KEY"] = key
    if engine.log.is_file():
        engine.log.replace(run / "vllm.previous.log")
    started = time.monotonic()
    process = engine.start(cmd, env, cwd=REPO)
    try:
        engine.wait_ready(process, lambda: api_ready(opts.url, key), opts.timeout)
        ready = time.monotonic()
        probe = request(f"{opts.url}/v1/chat/completions", key, opts.timeout,
                        {"model": NAME, "messages": [{"role": "user", "content": "只输出 2+2 的结果。"}],
                         "temperature": 0, "max_tokens": 64, "reasoning_effort": "low"})
        if not probe.get("choices") or not probe.get("usage", {}).get("completion_tokens"):
            raise RuntimeError("GLM first inference returned no tokens")
    except BaseException:
        engine.stop()
        raise
    report = {"url": opts.url, "pid": process.pid, "memcache": cache_mode, "memcache_metrics": cache.metrics_url,
              "memcache_s": round(cache_seconds, 3), "ready_s": round(ready - started, 3),
              "first_inference_s": round(time.monotonic() - ready, 3), "total_s": round(time.monotonic() - start, 3)}
    write_json(run / "launch.json", asdict(opts))
    write_json(run / "ready.json", report)
    write_json(run / "first-inference.json", probe)
    write_json(run / "profile-env.json", {name: env[name] for name in PROFILE_ENV})
    print(json.dumps(report, ensure_ascii=False))
    return 0


def read_launch(run: Path, *, attempt: bool = False) -> Launch:
    path = run / ("launch-attempt.json" if attempt else "launch.json")
    if not path.is_file():
        raise ValueError(f"{run} has no successful GLM launch")
    data = json.loads(path.read_text())
    return Launch(cache=Settings(**data.pop("cache")), **data)


def status(run: Path) -> int:
    opts = read_launch(run, attempt=True)
    cache = MemCache(run / "memcache", opts.cache, REPO)
    key_path = run / "api.key"
    running = Service(run, "vllm").running
    print(json.dumps({"url": opts.url, "running": running,
                      "ready": running and key_path.is_file() and api_ready(opts.url, key_path.read_text().strip()),
                      "memcache_healthy": cache.healthy(), "pool_managed": cache.managed,
                      "memcache_pool_running": cache.pool.running, "metrics_url": cache.metrics_url}))
    return 0


def _absolute(path: Path | None) -> str:
    return str(path.resolve()) if path else ""


def run_command(args: argparse.Namespace) -> int:
    run = args.run.resolve()
    action = args.glm_action
    opts = Launch(str(args.model.resolve()), _absolute(args.hooks_dir), args.host, args.port,
                  args.npus or os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""), args.dp, args.tp,
                  args.max_batched_tokens, args.max_seqs, args.max_model_len, args.mem_util, args.timeout,
                  Settings(_absolute(args.memcache_client_config), _absolute(args.memcache_meta_config),
                           _absolute(args.memcache_pool_config), args.memcache_metrics_url,
                           args.memcache_device, args.memcache_max_starts)) if action == "serve" else None
    try:
        if opts is not None and args.dry_run:
            return launch(run, opts, restart=False, fresh_cache=False, dry_run=True)
        if action == "status":
            return status(run)
        run.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (run / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if opts is not None:
                return launch(run, opts, restart=False, fresh_cache=False, dry_run=False)
            if action == "restart":
                previous = read_launch(run)
                updates = {name: getattr(args, name) for name in ("max_batched_tokens", "max_seqs")
                           if getattr(args, name) is not None}
                return launch(run, replace(previous, **updates), restart=True, fresh_cache=args.refresh_memcache,
                              dry_run=args.dry_run)
            engine = Service(run, "vllm")
            if action == "stop":
                engine.stop()
                if args.with_memcache:
                    Service(run / "memcache", "pool").stop()
                    Service(run / "memcache", "meta").stop()
                    (run / "memcache" / "pool.json").unlink(missing_ok=True)
                return 0
            raise ValueError(f"unknown GLM action {action}")
    except (OSError, RuntimeError, ValueError) as e:
        print(f"glm53 {action} failed: {e}", file=sys.stderr)
        return 1


def add_parser(subparsers: Any) -> None:
    glm = subparsers.add_parser("glm53", help="serve GLM-5.3-Flash on vLLM Ascend with MemCache")
    commands = glm.add_subparsers(dest="glm_action", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("model", type=Path)
    serve.add_argument("--hooks-dir", type=Path, help="GLM optimization hooks directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--npus", default="")
    serve.add_argument("--dp", type=int, default=2)
    serve.add_argument("--tp", type=int, default=8)
    serve.add_argument("--max-batched-tokens", type=int, default=8192)
    serve.add_argument("--max-seqs", type=int, default=32)
    serve.add_argument("--max-model-len", type=int, default=1048576)
    serve.add_argument("--mem-util", type=float, default=0.94)
    serve.add_argument("--timeout", type=float, default=600)
    serve.add_argument("--memcache-client-config", type=Path)
    serve.add_argument("--memcache-meta-config", type=Path)
    serve.add_argument("--memcache-pool-config", type=Path)
    serve.add_argument("--memcache-metrics-url", default="")
    serve.add_argument("--memcache-device", type=int, default=0)
    serve.add_argument("--memcache-max-starts", type=int, default=12)
    serve.add_argument("--dry-run", action="store_true")
    restart = commands.add_parser("restart")
    restart.add_argument("--max-batched-tokens", type=int)
    restart.add_argument("--max-seqs", type=int)
    restart.add_argument("--refresh-memcache", action="store_true")
    restart.add_argument("--dry-run", action="store_true")
    stop = commands.add_parser("stop")
    stop.add_argument("--with-memcache", action="store_true")
    status = commands.add_parser("status")
    for command in (serve, restart, stop, status):
        command.add_argument("--run", type=Path, default=Path("run/glm53"))
