"""``python -m core.nodeagent --host host-a --address <ip> --supervisor tcp://... --root <dir>``"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import msgspec

from core.common.ids import HostId
from core.common.tunables import load_tunables
from core.nodeagent.agent import AgentTunables, NodeAgent
from platforms.base import PlatformInfo


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="core.nodeagent")
    ap.add_argument("--host", required=True)
    ap.add_argument("--address", required=True, help="this host's address, from the topology data")
    ap.add_argument("--supervisor", required=True)
    ap.add_argument("--root", required=True, type=Path, help="local snapshot root")
    ap.add_argument("--run-root", type=Path)
    ap.add_argument("--platform", default='{"name": "fake"}', help="platform info as JSON")
    ap.add_argument("--tunables", default="{}", help='JSON, e.g. {"nodeagent.heartbeat_interval_s": 0.2}')
    args = ap.parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    run_root = args.run_root or args.root / ".run"
    run_root.mkdir(parents=True, exist_ok=True)
    agent = NodeAgent(
        host=HostId(args.host), address=args.address, supervisor=args.supervisor,
        snapshot_root=args.root, run_root=run_root,
        platform=msgspec.convert(json.loads(args.platform), PlatformInfo),
        tun=load_tunables(AgentTunables, "nodeagent", json.loads(args.tunables)),
    )
    return asyncio.run(agent.run())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
