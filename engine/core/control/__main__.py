"""``python -m core.control serve --source <workspace> --topology <file> --state-dir <dir> ...``

Starts the control-plane process (doc/specs/control-plane.md §1): the startup
change brings the engine from nothing to its first identity, then it serves the
control interface until SIGTERM. When it is ready it writes ``control.json`` in
the state directory (the control endpoint and the startup result) and prints the
same line on stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import msgspec
import zmq.asyncio

from core.change.config import ChangeTunables
from core.common.ids import HostId
from core.common.result import Failed
from core.common.tunables import load_tunables
from core.control.host import ControlPlane, ingest
from core.supervisor.supervisor import SupervisorTunables
from core.supervisor.topology import Topology


async def _spawn_agents(topo: Topology, control: str, platform: str, tunables: str
                        ) -> list[asyncio.subprocess.Process]:
    """One node agent per host of the topology, on this machine (single machine = one-host case)."""
    out: list[asyncio.subprocess.Process] = []
    for h in topo.hosts:
        out.append(await asyncio.create_subprocess_exec(
            sys.executable, "-m", "core.nodeagent", "--host", h.id, "--address", h.address,
            "--supervisor", control, "--root", h.snapshot_root, "--platform", platform, "--tunables", tunables,
            env=dict(os.environ), start_new_session=True,
        ))
    return out


async def serve(args: argparse.Namespace) -> int:
    topo = msgspec.json.decode(args.topology.read_bytes(), type=Topology)
    config = json.loads(args.config.read_text()) if args.config else {}
    tun = json.loads(args.tunables)
    zctx = zmq.asyncio.Context()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    cp = ControlPlane(
        zctx, host=HostId(args.host), bind=args.bind, state_dir=args.state_dir,
        sup_tun=load_tunables(SupervisorTunables, "supervisor", tun),
        change_tun=load_tunables(ChangeTunables, "change", tun),
        proc_tunables=json.loads(args.proc_tunables),
    )
    await cp.start()
    agents = await _spawn_agents(topo, cp.address, args.platform, args.agent_tunables) if args.spawn_agents else []
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    code = 0
    try:
        ready = await cp.sup.wait_for_hosts([h.id for h in topo.hosts], cp.sup.tun.rpc_timeout_s)
        result = ready if isinstance(ready, Failed) else await cp.boot(ingest(cp.blobs, args.source), topo, config)
        line = {"control": cp.address, "result": msgspec.to_builtins(
            result.diagnostics if isinstance(result, Failed) else result.value)}
        (args.state_dir / "control.json").write_text(json.dumps(line))
        print(json.dumps(line), flush=True)
        if isinstance(result, Failed) or result.value.status != "committed":
            code = 1
        else:
            await stop.wait()
    finally:
        await cp.sup.stop_all()
        cp.close()
        for a in agents:
            if a.returncode is None:
                os.killpg(a.pid, signal.SIGKILL)
                await a.wait()
        zctx.destroy(linger=0)
    return code


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="core.control")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="start the engine and serve the control interface")
    s.add_argument("--source", required=True, type=Path, help="workspace whose engine/ tree to start from")
    s.add_argument("--topology", required=True, type=Path, help="topology as a JSON file")
    s.add_argument("--state-dir", required=True, type=Path, help="blob store, event log, control snapshots")
    s.add_argument("--host", required=True, help="the HostId this control plane runs on")
    s.add_argument("--bind", required=True, help="address to bind the control endpoint on (topology data)")
    s.add_argument("--config", type=Path, help="config values as a JSON file")
    s.add_argument("--spawn-agents", action="store_true", help="start a node agent for every host here")
    s.add_argument("--platform", default='{"name": "fake"}', help="platform info for spawned node agents")
    s.add_argument("--tunables", default="{}", help="supervisor.* and change.* tunables as JSON")
    s.add_argument("--proc-tunables", default="{}", help="host.*, ipc.*, change.* tunables for processes")
    s.add_argument("--agent-tunables", default="{}", help="nodeagent.* tunables for spawned node agents")
    args = ap.parse_args(argv)
    return asyncio.run(serve(args))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
