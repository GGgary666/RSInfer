"""AscendStore's MemCache client configuration and optional local pool lifecycle."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import sys
import time
import urllib.request
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from urllib.parse import urlsplit

from deploy.local_process import Service, write_json

CLIENT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
DRAM = "ock.mmc.local_service.dram.size"
META_URL = "ock.mmc.meta_service_url"
META_STORE = "ock.mmc.meta_service.config_store_url"
POOL_STORE = "ock.mmc.local_service.config_store_url"
METRICS = "ock.mmc.meta_service.metrics_url"
POOL_READY = "RSInfer MemCache pool ready"


def fetch(url: str, timeout: float = 2) -> bytes:
    addr = urlsplit(url)
    if addr.hostname is None or addr.scheme not in ("http", "https"):
        raise ValueError(f"invalid MemCache metrics URL {url}")
    # The MetaService HTTP implementation ignores requests with Connection: close.
    connect = http.client.HTTPSConnection if addr.scheme == "https" else http.client.HTTPConnection
    connection = connect(addr.hostname, addr.port, timeout=timeout)
    target = addr.path or "/"
    if addr.query:
        target += "?" + addr.query
    try:
        connection.request("GET", target)
        response = connection.getresponse()
        if response.status != 200:
            raise OSError(f"MemCache metrics endpoint returned HTTP {response.status}")
        return response.read()
    finally:
        connection.close()


def port_open(url: str) -> bool:
    address = urlsplit(url)
    if address.hostname is None or address.port is None:
        raise ValueError(f"MemCache endpoint must name a host and port: {url}")
    try:
        with socket.create_connection((address.hostname, address.port), timeout=0.5):
            return True
    except OSError:
        return False


def config_values(text: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith(("#", ";")):
            pairs[key.strip()] = value.strip()
    return pairs


@dataclass(frozen=True)
class Settings:
    client: str = ""
    meta: str = ""
    pool: str = ""
    metrics: str = ""
    device: int = 0
    max_starts: int = 12


class MemCache:
    def __init__(self, directory: Path, settings: Settings, repo: Path) -> None:
        self.directory, self.settings, self.repo = directory, settings, repo
        self.meta = Service(directory, "meta")
        self.pool = Service(directory, "pool")
        if bool(settings.meta) != bool(settings.pool):
            raise ValueError("supply both --memcache-meta-config and --memcache-pool-config")
        if settings.max_starts < 1 or settings.device < 0:
            raise ValueError("MemCache needs a positive start limit and a non-negative device index")
        self.managed = bool(settings.meta)
        self.meta_text = Path(settings.meta).read_text() if self.managed else ""
        self.pool_text = Path(settings.pool).read_text() if self.managed else ""
        self.client_text = Path(settings.client).read_text() if settings.client else self.pool_text
        self.client_values = config_values(self.client_text)
        meta = config_values(self.meta_text)
        self.metrics_url = settings.metrics or (meta[METRICS].rstrip("/") + "/metrics" if METRICS in meta else "")
        self.client_path = directory / "client.conf"
        self.pool_record = directory / "pool.json"
        self.signature = hashlib.sha256(f"{settings.device}\n{self.meta_text}\0{self.pool_text}".encode()).hexdigest()
        if self.enabled and (not self.metrics_url or not self.client_values.get(META_URL)
                             or not self.client_values.get(POOL_STORE)):
            raise ValueError("MemCache needs a metrics URL and client MetaService/config-store endpoints")
        if self.managed:
            pool = config_values(self.pool_text)
            if (meta.get(META_URL) != pool.get(META_URL) or meta.get(META_STORE) != pool.get(POOL_STORE)
                    or meta.get(META_URL) != self.client_values.get(META_URL)
                    or meta.get(META_STORE) != self.client_values.get(POOL_STORE)):
                raise ValueError("MemCache meta, pool and client endpoints must match")
            if DRAM not in pool:
                raise ValueError("MemCache pool config must set its DRAM size")
        if settings.metrics and not self.enabled:
            raise ValueError("a MemCache metrics URL requires a client or pool config")

    @property
    def enabled(self) -> bool:
        return bool(self.managed or self.settings.client)

    def healthy(self) -> bool:
        if not self.enabled:
            return False
        try:
            return any(float(line.rsplit(maxsplit=1)[-1]) > 0 for line in fetch(self.metrics_url).splitlines()
                       if line.startswith(b"memcache_total_capacity_bytes{"))
        except (OSError, ValueError, http.client.HTTPException):
            return False

    def meta_ready(self) -> bool:
        try:
            return (b"memcache_" in fetch(self.metrics_url) and port_open(self.client_values[META_URL])
                    and port_open(self.client_values[POOL_STORE]))
        except (OSError, http.client.HTTPException):
            return False

    def prepare(self, timeout: float) -> str:
        if not self.enabled:
            return "disabled"
        self.directory.mkdir(parents=True, exist_ok=True)
        # The persistent pool owns DRAM. Workers and schedulers are clients with no local pool.
        lines = [line for line in self.client_text.splitlines() if line.partition("=")[0].strip() != DRAM]
        self.client_path.write_text("\n".join([*lines, f"{DRAM} = 0GB", ""]))
        if not self.managed:
            if not (self.healthy() and self.meta_ready()):
                raise RuntimeError(f"external MemCache is unavailable at {self.metrics_url}")
            return "external"
        record = json.loads(self.pool_record.read_text()) if self.pool_record.is_file() else {}
        if (record.get("signature") == self.signature and record.get("starts", 0) < self.settings.max_starts
                and self.meta.running and self.pool.running and self.healthy()):
            write_json(self.pool_record, {**record, "starts": record["starts"] + 1})
            return "reused"
        self.stop()
        for endpoint in (self.client_values[META_URL], self.client_values[POOL_STORE], self.metrics_url):
            if port_open(endpoint):
                raise RuntimeError(f"MemCache endpoint {endpoint} is in use")
        meta_config, pool_config = self.directory / "meta.conf", self.directory / "pool.conf"
        meta_config.write_text(self.meta_text)
        pool_config.write_text(self.pool_text)
        self.pool.log.unlink(missing_ok=True)
        env = dict(os.environ)
        try:
            meta = self.meta.start([sys.executable, "-u", "-c", "from memcache_hybrid import MetaService; MetaService.main()"],
                                   {**env, "MMC_META_CONFIG_PATH": str(meta_config)}, cwd=self.repo)
            self.meta.wait_ready(meta, self.meta_ready, timeout)
            pool = self.pool.start([sys.executable, "-u", "-m", "deploy.memcache", str(self.settings.device)],
                                   {**env, "MMC_LOCAL_CONFIG_PATH": str(pool_config)}, cwd=self.repo)
            self.pool.wait_ready(pool, lambda: POOL_READY in self.pool.log.read_text(errors="replace")
                                 and self.healthy(), timeout)
            write_json(self.pool_record, {"signature": self.signature, "starts": 1, "pid": pool.pid})
        except BaseException:
            self.stop()
            raise
        return "started"

    def stop(self) -> None:
        if self.managed:
            self.pool.stop()
            self.meta.stop()
            self.pool_record.unlink(missing_ok=True)


def serve_pool(device: int) -> None:
    store = import_module("memcache_hybrid").DistributedObjectStore()
    code = store.init(device)
    if code != 0:
        raise RuntimeError(f"MemCache DistributedObjectStore.init failed ({code})")
    print(POOL_READY, flush=True)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    serve_pool(int(sys.argv[1]))
