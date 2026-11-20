"""``rsinfer``: run this engine on one machine, and change it while it runs.

    python deploy/rsinfer.py serve <model dir> [--layers 8] [--dummy] [--detach]
    python deploy/rsinfer.py status | stop | sleep | wake           [--run DIR]
    python deploy/rsinfer.py control apply --source <workspace>     [--run DIR]
    python deploy/rsinfer.py chat "Hello" | complete "The capital of France is"
    python deploy/rsinfer.py checkpoint --config <HF dir> --layers 4 --out <dir>

Two kinds of command:

- The engine's life. ``serve`` starts one: a frontend, the scheduler host and one worker (TP1),
  through ``python -m core.control serve --spawn-agents``; in the foreground until Ctrl-C, or with
  ``--detach`` in the background. ``status`` says how it is, ``stop`` ends it, ``sleep`` gives its
  device memory back and ``wake`` takes it again.
- Changes to a running engine. ``control check|apply|status|introspect|logs`` is the control client
  (``python -m core.control.client``), with the engine's control address found for you.

A run directory (``--run``, default ``run/rsinfer``) is one engine: its topology and config, its
control plane's state, its snapshots and logs, and ``engine.json`` (its process and addresses).
Every command finds its engine through it.

What a model needs beyond the engine's own keys comes from its recipe (``deploy/models/``), picked
by the model's ``config.json``. ``--set KEY=VALUE`` sets any other config key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "engine")]

from deploy.models import ds1017, qwen38  # noqa: E402
from deploy.models.hf import cut, read_config  # noqa: E402

if TYPE_CHECKING:
    from core.common.ids import UnitId

RECIPES: tuple[types.ModuleType, ...] = (qwen38, ds1017)
HOST = "host-a"
SCHED = {"host": HOST, "role": "sched", "index": 0}
SPEC_ALGORITHMS = {"EAGLE": "eagle", "NEXTN": "eagle"}
"""SGLang's --speculative-algorithm values the engine takes: the MTP layer drafts (SGLang runs NEXTN as EAGLE)."""
WORKER = {"host": HOST, "role": "worker", "index": 0}
DEFAULT_RUN = Path("run/rsinfer")
ENGINE_FILE = "engine.json"


def recipe_for(cfg: dict[str, Any]) -> types.ModuleType:
    for recipe in RECIPES:
        if recipe.matches(cfg):
            return recipe
    kinds = cfg.get("architectures") or cfg.get("model_type")
    raise SystemExit(f"no recipe in deploy/models/ serves {kinds}: known are {[r.NAME for r in RECIPES]}")


def env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(REPO / "engine")}


def worker_units(source: Path, recipe: types.ModuleType) -> list[UnitId]:
    """The worker's units in the workspace ``source``: the units of its model's family, so those only
    another family runs stay out (two model units would both provide model@1)."""
    from core.sourcetree import read_tree

    others = frozenset().union(*(r.WORKER_ONLY for r in RECIPES if r is not recipe))
    return [u for u in read_tree(source.resolve() / "engine").role_units(None, "worker") if u not in others]


# ------------------------------------------------------------------ the run directory


def engine_record(run: Path) -> dict[str, Any]:
    """What ``serve`` wrote into ``engine.json``, completed from the run directory's own files (an
    engine started some other way has none): the control address from ``control/control.json``,
    the URL and the model name from ``config.json``."""
    path = run / ENGINE_FILE
    record: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
    control = run / "control" / "control.json"
    if "control" not in record and control.is_file():
        record["control"] = json.loads(control.read_text())["control"]
    config = run / "config.json"
    if config.is_file():
        values = json.loads(config.read_text())
        if "url" not in record and "frontend.http_port" in values:
            record["url"] = f"http://{reachable(values.get('frontend.http_bind') or '')}:{values['frontend.http_port']}"
        if "served_model_name" not in record and "frontend.model" in values:
            record["served_model_name"] = values["frontend.model"]
    return record


def control_address(run: Path) -> str:
    path = run / "control" / "control.json"
    if not path.is_file():
        raise SystemExit(f"{run}: no engine has come up here (no control/control.json); start one with serve")
    return str(json.loads(path.read_text())["control"])


def _argv(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")[:-1]
    except OSError:
        return []


def _processes() -> dict[int, list[str]]:
    """Every process's argv: from ``/proc`` (exact arguments) or, elsewhere, from ``ps``."""
    if Path("/proc/self/cmdline").exists():
        out: dict[int, list[str]] = {}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit() and (argv := _argv(int(entry.name))):
                out[int(entry.name)] = argv
        return out
    ps = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True, check=True)
    lines = (line.strip().split(None, 1) for line in ps.stdout.splitlines() if line.strip())
    return {int(p[0]): p[1].split() for p in lines if len(p) == 2}


def engine_processes(run: Path) -> tuple[int | None, list[int]]:
    """The engine of ``run``: its control plane (``core.control serve --state-dir <run>/control``)
    and the processes started for it, node agents and process hosts, whose arguments name a path
    in its host's directory (``<run>/host-a/``: the snapshots, the processes' specs). Only these two
    places count, not any path under ``run``: another engine may run from a directory inside it."""
    run = run.resolve()
    control_dir, host_dir = str(run / "control"), str(run / HOST)
    control: int | None = None
    procs: list[int] = []
    for pid, argv in _processes().items():
        if pid == os.getpid():
            continue
        if "core.control" in argv and "serve" in argv and _value(argv, "--state-dir") == control_dir:
            control = pid
            procs.append(pid)
        elif _is_engine(argv) and any(a == host_dir or a.startswith(host_dir + os.sep) for a in argv):
            procs.append(pid)
    return control, sorted(procs)


def _value(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def _is_engine(argv: list[str]) -> bool:
    """A process the control plane started: a node agent or a process host (not, say, a shell
    whose command names the run directory)."""
    return any(m in argv for m in ("core.nodeagent", "unit.runtime.host", "core.control"))


# ------------------------------------------------------------------ serve


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


def rank_gpus(args: argparse.Namespace) -> list[int]:
    """The physical GPU of each worker (rank): ``--gpus`` names one a rank; else ``--ranks`` workers on
    the one ``--gpu`` gives (or on the CPU: the numbers are then not used)."""
    if args.gpus:
        try:
            gpus = [int(g) for g in args.gpus.split(",")]
        except ValueError:
            raise SystemExit(f"--gpus {args.gpus!r}: physical GPU indices separated by commas") from None
        if args.gpu is not None or args.ranks not in (1, len(gpus)) or args.device != "cuda":
            raise SystemExit("--gpus names one GPU a worker: not with --gpu, another --ranks, or --device cpu")
        return gpus
    if args.ranks < 1:
        raise SystemExit("--ranks must be at least 1")
    return [worker_gpu(args.gpu) if args.device == "cuda" else 0] * args.ranks


def worker(rank: int) -> dict[str, Any]:
    return {**WORKER, "index": rank}


def parse_set(items: list[str]) -> dict[str, Any]:
    """``KEY=VALUE`` pairs; a value that parses as JSON is taken as such, else as a string."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set {item!r}: KEY=VALUE expected")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def plan(args: argparse.Namespace) -> tuple[Path, Path, dict[str, float], types.ModuleType]:
    """Write ``topology.json`` and ``config.json`` into the run directory."""
    import msgspec

    from core.change.config import ChangeTunables
    from core.common.ids import HostId
    from core.supervisor import HostSpec, Mesh, RankPlacement, RoleSpec, Topology

    model = Path(args.model).resolve()
    cfg = read_config(model)
    recipe = recipe_for(cfg)
    args.page_size = args.page_size or recipe.PAGE_SIZE
    workers = worker_units(args.source, recipe)
    if args.layers:
        cfg = cut(cfg, args.layers)
    weights = "" if args.dummy else str((args.weights or model).resolve())
    if weights and not list(Path(weights).glob("*.safetensors")):
        print(f"no *.safetensors in {weights}: dummy weights", file=sys.stderr)
        weights = ""
    args.dummy = not weights
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    cuda = args.device == "cuda"
    host = HostId(HOST)
    graphs = cuda and args.cuda_graph != "off"
    # Several workers are data parallel: each has requests of its own (a scheduling domain). Whether
    # they also share the model between them (expert parallel) is the model family's (its recipe).
    args.rank_gpus = rank_gpus(args)
    ranks = len(args.rank_gpus)
    topology = Topology(
        hosts=[HostSpec(host, args.control_host, str(run / HOST / "snapshots"))],
        mesh=Mesh({"tp": 1} if ranks == 1 else {"dp": ranks}),
        placement=[RankPlacement(rank, host, gpu) for rank, gpu in enumerate(args.rank_gpus)],
        roles=[RoleSpec("frontend", [], hosts=[host]), RoleSpec("sched", [], hosts=[host], leads_epochs=True),
               RoleSpec("worker", workers, per_rank=True, platform="cuda_graph" if graphs else "torch_eager",
                        platform_options={"cuda": 1} if cuda else {},
                        env=dict(getattr(recipe, "WORKER_ENV", {})) if cuda else {})])
    config: dict[str, Any] = {
        # frontend
        "frontend.model": args.served_model_name or recipe.SERVED_NAME,
        "frontend.tokenizer_path": str((args.tokenizer or model).resolve()), "frontend.model_path": str(model),
        "frontend.http_bind": args.host, "frontend.http_port": args.port, "frontend.scheduler": SCHED,
        "frontend.response_timeout_s": float(args.timeout),
        # scheduler host
        "sched.workers": [worker(rank) for rank in range(ranks)], "sched.domains": ranks,
        "sched.frontend_host": HOST, "sched.max_batch": args.max_rows,
        "sched.token_budget": args.token_budget, "sched.overlap": args.overlap,
        "alloc.radix_cache": not args.disable_radix_cache,
        # speculation (doc/specs/mtp.md), SGLang's flags
        "spec.algorithm": SPEC_ALGORITHMS[args.speculative_algorithm.upper()] if args.speculative_algorithm else "",
        "spec.num_steps": args.speculative_num_steps, "spec.eagle_topk": args.speculative_eagle_topk,
        "spec.num_draft_tokens": args.speculative_num_draft_tokens,
        # worker
        "worker.scheduler": SCHED, "weights.path": weights,
        "state.kv.page_size": args.page_size, "state.kv.dtype": args.kv_dtype,
        "batch.max_rows": args.max_rows, "batch.max_context_len": args.max_context,
        # Workers on one card (--ranks) share its part between them.
        "holder.mem_fraction": args.mem_fraction / max(args.rank_gpus.count(gpu) for gpu in args.rank_gpus),
        "holder.max_slots": args.max_slots, "holder.cpu_bytes": args.cpu_bytes,
        "plan.cuda_graph": args.cuda_graph, "plan.graph_buckets": args.graph_buckets,
        "plan.graph_memory_bytes": args.graph_memory_bytes,
        **recipe.config(args, cfg, model),
        **parse_set(args.set),
    }
    topo_file, config_file = run / "topology.json", run / "config.json"
    topo_file.write_bytes(msgspec.json.encode(topology))
    config_file.write_text(json.dumps(config, indent=1))
    # A change's execution may read the weights again (a wake), as start-up does: it gets as long
    # as start-up, and not less than the engine's own limit.
    tunables = {"supervisor.checkin_timeout_s": float(args.timeout), "change.prepare_timeout_s": float(args.timeout),
                "change.execute_timeout_s": max(float(args.timeout), ChangeTunables().execute_timeout_s)}
    return topo_file, config_file, tunables, recipe


def serve_command(args: argparse.Namespace, topo: Path, config: Path, tunables: dict[str, float]) -> list[str]:
    graphs = args.device == "cuda" and args.cuda_graph != "off"
    return [sys.executable, "-m", "core.control", "serve", "--source", str(args.source.resolve()), "--topology", str(topo),
            "--state-dir", str(args.run.resolve() / "control"), "--host", HOST, "--bind", args.control_host,
            "--config", str(config), "--spawn-agents", "--platform",
            json.dumps({"name": "cuda_graph" if graphs else "torch_eager"}), "--tunables", json.dumps(tunables),
            "--proc-tunables", json.dumps(args.proc_tunables)]


def reachable(host: str) -> str:
    """The address a client on this machine reaches a server listening on ``host`` at: the
    loopback one for a wildcard (``0.0.0.0``, ``::``), an IPv6 address in brackets."""
    if host in ("", "0.0.0.0"):
        return "127.0.0.1"
    if host == "::":
        return "[::1]"
    return f"[{host}]" if ":" in host else host


def _stop_with_parent() -> None:
    """In the foreground, the engine stops (SIGTERM, an orderly shutdown) when this command goes,
    however it goes: an engine nobody holds keeps its GPU memory."""
    if sys.platform == "linux":
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def cmd_serve(args: argparse.Namespace) -> int:
    args.model = args.model or args.model_option
    if not args.model:
        raise SystemExit("serve: give the model directory (rsinfer serve <model dir>)")
    run = args.run.resolve()
    running, procs = engine_processes(run)
    if running is not None:
        raise SystemExit(f"an engine already runs in {run} (process {running}): rsinfer stop --run {args.run}")
    if procs:  # its control plane is gone, not what it started: a worker left behind keeps its GPU memory
        raise SystemExit(f"processes {procs} of an earlier engine in {run} are still running: stop them first with "
                         f"rsinfer stop --run {args.run}")
    topo, config, tunables, recipe = plan(args)
    command = serve_command(args, topo, config, tunables)
    if args.dry_run:
        print(" ".join(f"'{a}'" if " " in a or "{" in a else a for a in command))
        return 0
    control = run / "control" / "control.json"
    control.unlink(missing_ok=True)
    log = None
    if args.detach:
        log = open(run / "engine.log", "ab")  # noqa: SIM115 - the engine writes to it after this command left
        proc = subprocess.Popen(command, env=env(), start_new_session=True, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=log)
    else:
        proc = subprocess.Popen(command, env=env(), start_new_session=True, preexec_fn=_stop_with_parent)
        signal.signal(signal.SIGINT, lambda *_: proc.send_signal(signal.SIGTERM))
        signal.signal(signal.SIGTERM, lambda *_: proc.send_signal(signal.SIGTERM))
    url = f"http://{reachable(args.host)}:{args.port}"
    record = {"pid": proc.pid, "url": url, "model": str(Path(args.model).resolve()), "recipe": recipe.NAME,
              "served_model_name": args.served_model_name or recipe.SERVED_NAME, "source": str(args.source.resolve()),
              "detached": bool(args.detach), "started": time.time()}
    (run / ENGINE_FILE).write_text(json.dumps(record, indent=1))
    while proc.poll() is None and not control.exists():
        time.sleep(0.5)
    if control.exists():
        ready = json.loads(control.read_text())
        ok = isinstance(ready["result"], dict) and ready["result"].get("status") == "committed"
        (run / ENGINE_FILE).write_text(json.dumps({**record, "control": ready["control"]}, indent=1))
        print(f"engine {'up' if ok else 'failed'}: {url} (control {ready['control']})", file=sys.stderr, flush=True)
        if args.detach:
            print(f"logs in {run / 'engine.log'}; stop it with: {prog()} stop --run {args.run}", file=sys.stderr)
            return 0 if ok else 1
    elif args.detach:
        print(f"the engine exited with {proc.returncode} before it came up: see {run / 'engine.log'}", file=sys.stderr)
        return 1
    return proc.wait()


def prog() -> str:
    return "python deploy/rsinfer.py"


# ------------------------------------------------------------------ stop, status


def cmd_stop(args: argparse.Namespace) -> int:
    """SIGTERM to the control plane, which stops what it started (without it, to every process
    that is left: nobody stops them for one another); once ``--timeout`` passed, SIGKILL to
    whatever of the engine is left (a process left behind keeps its GPU memory)."""
    run = args.run.resolve()
    control, procs = engine_processes(run)
    if not procs:
        print(f"no engine runs in {run}", file=sys.stderr)
        return 0
    for pid in [control] if control is not None else procs:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        _, procs = engine_processes(run)
        if not procs:
            print(f"stopped the engine in {run}", file=sys.stderr)
            return 0
        time.sleep(0.3)
    for pid in procs:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(f"the engine in {run} did not stop within {args.timeout} s: killed processes {procs}", file=sys.stderr)
    return 1


def _health(url: str) -> str:
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return "ok" if r.status == 200 else f"HTTP {r.status}"
    except (urllib.error.URLError, OSError) as e:
        return f"no answer ({e})"


def cmd_status(args: argparse.Namespace) -> int:
    import msgspec
    import zmq.asyncio

    from core.common.result import Failed
    from core.control.client import ControlClient

    run = args.run.resolve()
    control, procs = engine_processes(run)
    record = engine_record(run)
    out: dict[str, Any] = {"run": str(run), "running": control is not None, "pid": control, "processes": len(procs)}
    if control is None:
        print(json.dumps(out, indent=2))
        return 1
    out |= {k: record[k] for k in ("url", "control", "model", "served_model_name", "source") if k in record}
    if "url" in record:
        out["health"] = _health(record["url"])
    if "control" in record:
        async def ask() -> Any:
            zctx = zmq.asyncio.Context()
            client = ControlClient(zctx, record["control"], timeout_s=10.0)
            try:
                return await client.status()
            finally:
                client.close()
                zctx.destroy(linger=0)

        reply = asyncio.run(ask())
        if isinstance(reply, Failed):
            out["control_status"] = msgspec.to_builtins(reply.diagnostics)
        else:
            s = reply.value
            out |= {"identity": str(s.identity), "busy": s.busy, "queued": s.queued,
                    "recovery_required": s.recovery_required,
                    "procs": {str(p.proc): p.state for p in s.procs}}
            if args.json:
                out["control_status"] = msgspec.to_builtins(s)
    print(json.dumps(out, indent=2, default=str))
    return 0


# ------------------------------------------------------------------ control, sleep, wake


def control_argv(args: argparse.Namespace) -> list[str]:
    return ["--control", args.control or control_address(args.run.resolve())]


def running_topology(run: Path) -> dict[str, Any] | None:
    """The topology the engine in ``run`` runs under now: the ``plan.topology`` of the last committed
    change that carried one (the control plane's event log, including changes applied through
    ``core.control.client`` directly), else ``topology.json``."""
    import sqlite3

    db = run / "control" / "events.sqlite"
    if db.is_file():
        with sqlite3.connect(db) as conn:
            for (raw,) in conn.execute("SELECT plan_json FROM changes WHERE status = 'committed' "
                                       "AND plan_json IS NOT NULL ORDER BY id DESC"):
                topology = json.loads(raw).get("topology")
                if topology is not None:
                    return dict(topology)
    current = run / "topology.json"
    return json.loads(current.read_text()) if current.is_file() else None


def changed_topology(run: Path, rest: list[str]) -> tuple[Path | None, Path | None, bool]:
    """For ``check`` or ``apply``: a temporary topology file to add (when ``--source`` changes the
    worker's units), the topology file the change carries (explicit ``--topology`` or that one), and
    whether the change commits (an ``apply`` that is no dry run). ``serve`` lists the worker's units
    in the topology, and the control plane takes a role's listed units as they are: a unit added to
    the workspace would not load, and one removed would be missed. So the list is worked out again
    from the workspace over ``running_topology(run)``."""
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--source", type=Path)
    options.add_argument("--topology", type=Path)
    options.add_argument("--dry-run", action="store_true")
    given = options.parse_known_args(rest[1:])[0] if rest[:1] in (["check"], ["apply"]) else None
    if given is None:
        return None, None, False
    commits = rest[0] == "apply" and not given.dry_run
    if given.topology is not None:
        return None, given.topology.resolve(), commits
    recipe = next((r for r in RECIPES if r.NAME == engine_record(run).get("recipe")), None)
    topology = running_topology(run) if recipe is not None and given.source is not None else None
    if topology is None or recipe is None or given.source is None or not (given.source / "engine").is_dir():
        return None, None, False
    workers = [r for r in topology["roles"] if r["name"] == "worker"]
    units = worker_units(given.source, recipe)
    if not workers or not workers[0]["units"] or units == workers[0]["units"]:
        return None, None, False
    workers[0]["units"] = units
    changed = run / "topology.changed.json"
    changed.write_text(json.dumps(topology))
    return changed, changed, commits


def cmd_control(args: argparse.Namespace) -> int:
    from core.control.client import main as client

    rest = list(args.rest)
    if rest and rest[0] == "--":
        rest = rest[1:]
    run = args.run.resolve()
    # An engine named by its address alone has no run directory to tell its topology by.
    added, carried, commits = (None, None, False) if args.control else changed_topology(run, rest)
    try:
        code = client([*control_argv(args), *rest, *(["--topology", str(added)] if added else [])])
        if carried is not None and commits and code == 0 and run.is_dir():
            target = run / "topology.json"
            if carried != target:  # committed: keep the run directory's topology.json in sync
                shutil.copyfile(carried, target)
        return code
    finally:
        if added is not None:
            added.unlink(missing_ok=True)


def cmd_sleep(args: argparse.Namespace, asleep: bool) -> int:
    """``holder.sleep``: the worker gives back its weights, state pools and decode graphs (a
    change: the engine must be empty, or ``--wait-empty`` / ``--force``); waking loads the weights
    from the checkpoint again."""
    from core.control.client import main as client

    busy = ["--force"] if getattr(args, "force", False) else \
        ["--wait-empty", str(args.wait_empty)] if getattr(args, "wait_empty", None) is not None else []
    return client([*control_argv(args), "apply", "--config", json.dumps({"holder.sleep": asleep}),
                   "--intent", "sleep" if asleep else "wake", *busy])


# ------------------------------------------------------------------ chat, complete


def _target(args: argparse.Namespace) -> tuple[str, str]:
    record = engine_record(args.run.resolve()) if args.url is None else {}
    url = args.url or record.get("url") or "http://127.0.0.1:8000"
    name = args.model_name or record.get("served_model_name")
    if not name:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as r:
                name = json.loads(r.read())["data"][0]["id"]
        except (urllib.error.URLError, OSError, KeyError, IndexError):
            name = qwen38.SERVED_NAME
    return url.rstrip("/"), str(name)


def cmd_chat(args: argparse.Namespace, completion: bool) -> int:
    """One request, its answer streamed to stdout (server-sent events)."""
    url, name = _target(args)
    if completion:
        path, body = "/v1/completions", {"prompt": args.prompt}
    else:
        path, body = "/v1/chat/completions", {"messages": [{"role": "user", "content": args.prompt}]}
    body |= {"model": name, "max_tokens": args.max_tokens, "temperature": args.temperature, "stream": True}
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        response = urllib.request.urlopen(request, timeout=args.timeout)
    except urllib.error.HTTPError as e:
        print(f"error: HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        return 1
    with response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data: "):])
            if "error" in event:
                print(f"\nerror: {event['error']}", file=sys.stderr)
                return 1
            choice = event["choices"][0]
            text = choice.get("text") if completion else (choice.get("delta") or {}).get("content")
            print(text or "", end="", flush=True)
            if choice.get("finish_reason"):
                print(f"\n[{choice['finish_reason']}]", file=sys.stderr)
    return 0


def cmd_checkpoint(args: argparse.Namespace) -> int:
    return recipe_for(read_config(args.config.resolve())).checkpoint(args)


# ------------------------------------------------------------------ the command line


def _run_option(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", type=Path, default=DEFAULT_RUN, help=f"the engine's run directory (default {DEFAULT_RUN})")


def _serve_options(s: argparse.ArgumentParser) -> None:
    s.add_argument("model", nargs="?", help="Hugging Face model directory (config.json, tokenizer, *.safetensors)")
    s.add_argument("--model", dest="model_option", help=argparse.SUPPRESS)
    m = s.add_argument_group("model")
    m.add_argument("--weights", type=Path, help="directory of *.safetensors (default: the model directory)")
    m.add_argument("--dummy", action="store_true", help="dummy weights")
    m.add_argument("--tokenizer", type=Path, help="tokenizer directory (default: the model directory)")
    m.add_argument("--layers", type=int, default=0, help="keep only the first N decoder layers")
    m.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    m.add_argument("--served-model-name", "--name", dest="served_model_name",
                   help="the model name the frontend answers with (default: the recipe's)")
    for recipe in RECIPES:
        recipe.add_arguments(s.add_argument_group(f"{recipe.NAME} models"))
    d = s.add_argument_group("device and memory")
    d.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    d.add_argument("--gpu", type=int, help="the worker's CUDA device, by physical index (default: the first of "
                   "CUDA_VISIBLE_DEVICES, else 0)")
    d.add_argument("--gpus", help="several workers, one on each of these CUDA devices (physical indices separated by "
                   "commas): each serves requests of its own (data parallel), and a model family that splits its "
                   "weights over them does (DS 1017: its experts and Engram tables)")
    d.add_argument("--ranks", type=int, default=1, help="that many workers on one device (--gpu, or the CPU): for "
                   "trying several ranks where there is one card")
    d.add_argument("--mem-fraction", type=float, default=0.85, help="device share of weights and states "
                   "(holder.mem_fraction; SGLang's --mem-fraction-static)")
    d.add_argument("--page-size", type=int, default=0,
                   help="tokens a KV page (default: the model family's, 16 for Qwen, 128 for DS 1017)")
    d.add_argument("--kv-dtype", default="auto", help="state.kv.dtype (SGLang's --kv-cache-dtype)")
    d.add_argument("--max-slots", type=int, default=0, help="GDN slots (0: by holder.state_ratio)")
    d.add_argument("--cpu-bytes", type=int, default=1 << 30, help="state memory on a CPU device")
    d.add_argument("--cuda-graph", choices=("auto", "off", "required"), default="off",
                   help="decode graphs (plan.cuda_graph); auto and required run the worker on cuda_graph")
    d.add_argument("--graph-buckets", type=int, nargs="*", default=[],
                   help="rows of each decode graph (default: SGLang's sizes up to min(--max-rows, 64); powers of "
                        "two up to 8 while the captured attention is the fixed-extent one)")
    d.add_argument("--graph-memory-bytes", type=int, default=1 << 30, help="device memory the graphs may keep")
    sched = s.add_argument_group("scheduling")
    sched.add_argument("--max-rows", type=int, default=64, help="running requests at most")
    sched.add_argument("--max-context", type=int, default=4096, help="tokens of one request, prompt and output")
    sched.add_argument("--token-budget", type=int, default=2048, help="prompt tokens per step")
    sched.add_argument("--overlap", action="store_true", help="one step of overlap (sched.overlap)")
    sched.add_argument("--disable-radix-cache", action="store_true",
                       help="no prefix cache (alloc.radix_cache = false), as SGLang's flag")
    spec = s.add_argument_group("speculative decoding (doc/specs/mtp.md; SGLang's flags)")
    spec.add_argument("--speculative-algorithm", choices=sorted(SPEC_ALGORITHMS) + sorted(a.lower() for a in SPEC_ALGORITHMS),
                      help="the model's MTP layer drafts (spec.algorithm); off without it")
    spec.add_argument("--speculative-num-steps", type=int, default=3)
    spec.add_argument("--speculative-eagle-topk", type=int, default=1, help="only 1, a chain")
    spec.add_argument("--speculative-num-draft-tokens", type=int, default=4)
    e = s.add_argument_group("engine")
    e.add_argument("--host", "--bind", dest="host", default="127.0.0.1",
                   help="address the frontend (HTTP) listens on, e.g. 0.0.0.0 for every interface")
    e.add_argument("--control-host", default="127.0.0.1",
                   help="address the control plane and the node agent listen on; whoever reaches it can change the "
                        "engine's code (default: this machine only)")
    e.add_argument("--port", type=int, default=8000)
    _run_option(s)
    e.add_argument("--source", type=Path, default=REPO, help="workspace whose engine/ the engine starts from "
                   "(default: this repo); change it and apply it with control apply --source, without a restart")
    e.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="any other config key (a JSON value, or a string); may repeat")
    e.add_argument("--proc-tunables", type=json.loads, default={}, metavar="JSON",
                   help="process host.*, ipc.* and change.* tunables, e.g. {\"ipc.shm_capacity\":16777216}")
    e.add_argument("--timeout", type=int, default=3600, help="seconds for start-up and for a response; "
                   "above 3600, also for a change's execution (a wake reads the weights again)")
    e.add_argument("--detach", action="store_true", help="run in the background: return once the engine is up")
    e.add_argument("--dry-run", action="store_true", help="write topology.json and config.json, print the "
                   "command, start nothing")


def _engine_target(p: argparse.ArgumentParser) -> None:
    _run_option(p)
    p.add_argument("--control", help="the control address (default: the run directory's engine's)")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rsinfer", description="Run this engine on one machine, and change it while "
                                 "it runs. Every command finds its engine through a run directory (--run).")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
    _serve_options(sub.add_parser("serve", help="start an engine for a model"))
    st = sub.add_parser("status", help="whether the run directory's engine runs, and how it is")
    _run_option(st)
    st.add_argument("--json", action="store_true", help="also the control plane's whole status")
    stop = sub.add_parser("stop", help="stop the run directory's engine")
    _run_option(stop)
    stop.add_argument("--timeout", type=float, default=60.0, help="seconds before what is left is killed")
    for name, text in (("sleep", "give the device memory back (holder.sleep); requests are refused"),
                       ("wake", "load the weights again and serve")):
        p = sub.add_parser(name, help=text)
        _engine_target(p)
        if name == "sleep":
            busy = p.add_mutually_exclusive_group()
            busy.add_argument("--wait-empty", type=float, nargs="?", const=3600.0, default=None, metavar="SECONDS")
            busy.add_argument("--force", action="store_true")
    c = sub.add_parser("control", help="the control client: check, apply, status, introspect, logs",
                       description="Arguments after the command go to python -m core.control.client; "
                                   "e.g. control apply --source . --dry-run")
    _engine_target(c)
    c.add_argument("rest", nargs=argparse.REMAINDER, metavar="COMMAND ...")
    for name in ("chat", "complete"):
        p = sub.add_parser(name, help="send one request and stream the answer" +
                           (" (/v1/completions)" if name == "complete" else " (/v1/chat/completions)"))
        p.add_argument("prompt")
        _run_option(p)
        p.add_argument("--url", help="the frontend (default: the run directory's engine's)")
        p.add_argument("--model-name", help="the model the request names (default: what the engine serves)")
        p.add_argument("--max-tokens", type=int, default=128)
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--timeout", type=int, default=3600)
    k = sub.add_parser("checkpoint", help="random weights in a model's checkpoint layout, for tests")
    k.add_argument("--config", required=True, type=Path, help="directory with the config.json to follow")
    k.add_argument("--layers", type=int, default=0)
    k.add_argument("--out", required=True, type=Path)
    k.add_argument("--tokenizer", type=Path, help="copy the tokenizer files from here (default: --config)")
    k.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    k.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="where to initialize")
    k.add_argument("--seed", type=int, default=0)
    return ap


def _hoist(argv: list[str]) -> list[str]:
    """``control apply --source . --run X``: the engine options may come after the client's
    command too; move them before it, where argparse looks for them."""
    if not argv or argv[0] != "control":
        return argv
    head, rest = ["control"], argv[1:]
    out: list[str] = []
    i = 0
    while i < len(rest):
        if rest[i] in ("--run", "--control") and i + 1 < len(rest):
            head += rest[i:i + 2]
            i += 2
        else:
            out.append(rest[i])
            i += 1
    return [*head, *out]


def main(argv: list[str]) -> int:
    args = parser().parse_args(_hoist(argv))
    match args.command:
        case "serve":
            return cmd_serve(args)
        case "stop":
            return cmd_stop(args)
        case "status":
            return cmd_status(args)
        case "control":
            return cmd_control(args)
        case "sleep" | "wake":
            return cmd_sleep(args, args.command == "sleep")
        case "chat" | "complete":
            return cmd_chat(args, args.command == "complete")
        case "checkpoint":
            return cmd_checkpoint(args)
        case _:
            raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
