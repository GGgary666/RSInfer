"""A GLM prefix manifest is imported once at worker bootstrap."""

from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest

from models.glm53_config import KDA
from state.glm53_cache.unit import Cache
from state.prefix_checkpoint import PrefixCheckpoint


def test_layer_reload_does_not_reimport_bootstrap_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = PrefixCheckpoint("first", 1, 128, KDA, [[]])
    path = tmp_path / "prefix.json"
    path.write_bytes(msgspec.json.encode(checkpoint))
    config = {"glm53.memcache_config": "", "alloc.prefix_checkpoint": str(path), "holder.sleep": False}
    imports: list[tuple[PrefixCheckpoint, str]] = []

    def transfer(cache: Cache, index: PrefixCheckpoint, *, write: bool) -> None:
        assert not write
        imports.append((index, cache.layer_code))

    monkeypatch.setattr(Cache, "transfer", transfer)
    ctx = SimpleNamespace(use=lambda name: object(), platform=None, instance=SimpleNamespace(generation=1))
    cache = Cache(ctx, config)
    cache.bind([], {}, "model", "original")
    cache.bind([], {}, "model", "after-layer-reload")
    assert cache.layer_code == "after-layer-reload"
    ctx.instance.generation = 2
    reloaded = Cache(ctx, config)
    reloaded.bind([], {}, "model", "after-store-reload")
    assert imports == [(checkpoint, "original")]
