"""Rejected new declarations must preserve existing values and import paths."""
import sys

import pytest

from core.change.protocol import Abort, Prepare
from core.change.types import ProcActions
from core.common.ids import Identity
from core.context.config import ConfigEntry
from core.common.result import Failed
from tests.adversarial.test_transaction_boundaries import Peer, peer as peer
from tests.support import PROC, spec, write_tree


@pytest.mark.parametrize("failure", ["type", "owner"])
def test_failed_staging_preserves_existing_config_and_import_paths(peer: Peer, failure: str) -> None:
    persistent, fresh = f"{peer.pkg}.persistent", f"{peer.pkg}.fresh"
    owner = spec(peer.unit.id, f"{peer.pkg}.main:Unit", config={
        persistent: {"type": "int", "default": 1}})
    peer.host.runtime.config.declare_manifest(owner.manifest, {persistent: 42}).unwrap()
    addon = spec(f"{peer.pkg}.addon", f"{peer.pkg}.addon:Unit", config={
        fresh: {"type": "int", "default": 1},
        persistent: {"type": "int", "default": 2}})
    files = {f"{peer.pkg}/addon.py": "class Unit:\n    def __init__(self, ctx, config): pass\n"}
    candidate = write_tree(peer.root / "v2" / "engine", files)
    before = peer.host.runtime.config.entries()
    paths = list(sys.path)
    package_paths = list(sys.modules[peer.pkg].__path__)
    config = {
        fresh: ConfigEntry(addon.id, addon.manifest.config[fresh], 99),
        persistent: ConfigEntry(owner.id if failure == "type" else addon.id,
                                owner.manifest.config[persistent], "invalid" if failure == "type" else 7)}
    result = peer.host.runtime.stage_new_units([addon], code_root=str(candidate), config=config)
    assert isinstance(result, Failed)
    assert peer.host.runtime.config.entries() == before
    assert sys.path == paths and list(sys.modules[peer.pkg].__path__) == package_paths
    clean = spec(addon.id, f"{peer.pkg}.addon:Unit", config={fresh: {"type": "int", "default": 1}})
    write_tree(peer.root / "v3" / "engine", files)
    peer.send(Prepare(3, Identity("v3"), ProcActions(PROC, stage=[clean], config={fresh: 6}), 60)).unwrap()
    assert peer.host.runtime.config.entries()[persistent].value == 42
    assert peer.host.runtime.config.entries()[fresh].value == 6
    peer.send(Abort(3)).unwrap()
    assert peer.host.runtime.config.entries() == before
