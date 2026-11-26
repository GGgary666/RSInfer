"""Measure OpenAI streaming on Qwen3 or replay the GLM 16K/60K AISBench suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "engine")]

from deploy.local_process import write_json
from deploy.memcache import CLIENT, fetch


@dataclass(frozen=True)
class Result:
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    output_sha256: str


def stream(url: str, model: str, prompt: str, output: int, timeout: float) -> Result:
    body = {"model": model, "prompt": prompt, "max_tokens": output, "temperature": 0, "ignore_eos": True,
            "stream": True}
    req = urllib.request.Request(f"{url}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    first = last = 0.0
    text: list[str] = []
    usage: dict[str, Any] = {}
    complete = False
    start = time.perf_counter()
    with CLIENT.open(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                complete = True
                break
            data = json.loads(payload)
            if "error" in data:
                raise RuntimeError(f"inference error: {data['error']}")
            choice = data["choices"][0]
            if piece := choice.get("text"):
                last = time.perf_counter()
                first = first or last
                text.append(piece)
            if data.get("usage"):
                usage = data["usage"]
    if not complete or not first or usage.get("completion_tokens") != output:
        raise RuntimeError(f"expected {output} output tokens and a complete stream, got {usage}")
    return Result(usage["prompt_tokens"], output, (first - start) * 1000,
                  (last - first) * 1000 / (output - 1) if output > 1 else 0,
                  (time.perf_counter() - start) * 1000, hashlib.sha256("".join(text).encode()).hexdigest())


def latency(values: list[float]) -> dict[str, float]:
    quantiles = statistics.quantiles(values, n=100, method="inclusive") if len(values) > 1 else values * 99
    return {"avg": round(statistics.mean(values), 3), "p50": round(quantiles[49], 3),
            "p90": round(quantiles[89], 3), "p99": round(quantiles[98], 3)}


def qwen(args: argparse.Namespace) -> int:
    from tokenizers import Tokenizer

    if (min(args.samples, args.concurrency, args.prompt_tokens, args.output_tokens, args.repeat, args.timeout) < 1
            or args.shared_prefix < 0 or args.shared_prefix >= args.prompt_tokens):
        raise ValueError("benchmark counts must be positive and shared prefix shorter than the prompt")
    tokenizer = Tokenizer.from_file(str(args.tokenizer.resolve()))
    url = args.url.rstrip("/")

    def fit(prefix: str, suffix: str, length: int) -> str:
        n = length - len(tokenizer.encode(prefix + suffix, add_special_tokens=False).ids)
        for _ in range(8):
            if n < 0:
                break
            prompt = prefix + " alpha" * n + suffix
            size = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            if size == length:
                return prompt
            n += length - size
        raise ValueError(f"could not construct a {length}-token benchmark prompt")

    args.out.mkdir(parents=True, exist_ok=False)
    settings = {key: getattr(args, key) for key in
                ("samples", "concurrency", "prompt_tokens", "output_tokens", "shared_prefix", "repeat", "salt")}
    write_json(args.out / "parameters.json", {"url": url, "model": args.model, **settings})
    for repeat in range(args.repeat):
        prefix = (fit(f"Shared benchmark prefix {args.salt}-{repeat}\n", "\n", args.shared_prefix)
                  if args.shared_prefix else "")
        prompts = [fit(f"{prefix}Benchmark {args.salt}-{repeat}-{i:02d}\n",
                       "\nContinue with a short article about science.\n", args.prompt_tokens)
                   for i in range(args.samples)]
        if prefix:
            stream(url, args.model, prefix + " Warm up this prefix.\n", 1, args.timeout)
        else:
            stream(url, args.model, f"Warmup {args.salt}-{repeat}\nThe capital of France is", 8, args.timeout)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            start = time.perf_counter()
            rows = list(pool.map(lambda prompt: stream(url, args.model, prompt, args.output_tokens, args.timeout), prompts))
            elapsed = time.perf_counter() - start
        if any(row.prompt_tokens != args.prompt_tokens for row in rows):
            raise RuntimeError("the server tokenized at least one prompt to a different length")
        output_tps = sum(r.output_tokens for r in rows) / elapsed
        total_tps = sum(r.output_tokens + r.prompt_tokens for r in rows) / elapsed
        info = {"repeat": repeat + 1, "duration_s": round(elapsed, 3), "output_tps": round(output_tps, 3),
                "total_tps": round(total_tps, 3), "ttft_ms": latency([r.ttft_ms for r in rows]),
                "tpot_ms": latency([r.tpot_ms for r in rows]), "latency_ms": latency([r.latency_ms for r in rows])}
        write_json(args.out / f"run-{repeat + 1}.json", {"metrics": info, "requests": [asdict(r) for r in rows]})
        print(json.dumps(info), flush=True)
    return 0


def cache_counters(snapshot: bytes) -> dict[str, float]:
    names = ("memcache_get_requests_total", "memcache_get_successes_total", "memcache_alloc_failures_total",
             "memcache_batch_alloc_failures_total")
    counters: dict[str, float] = {name: 0.0 for name in names}
    for line in snapshot.decode().splitlines():
        if line and not line.startswith("#"):
            metric, value, *_ = line.split()
            name = metric.split("{", 1)[0]
            if name in counters:
                counters[name] += float(value)
    return counters


def native_launch(run: Path, model: str) -> dict[str, Any]:
    import msgspec

    from core.change.blobs import BlobStore
    from core.change.state import decode_manifest

    # A hot apply creates a new identity and can leave the startup JSON untouched.
    state = run / "control"
    uri = (state / "events.sqlite").as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        row = db.execute(
            "SELECT changes.new_identity, identities.manifest_hash FROM changes "
            "JOIN identities ON identities.identity = changes.new_identity "
            "WHERE changes.status = 'committed' ORDER BY changes.id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError(f"{run} has no committed engine identity")
    identity, digest = row
    data = BlobStore(state / "blobs").get(digest)
    if data is None:
        raise RuntimeError(f"{run} has no manifest for identity {identity}")
    manifest = decode_manifest(data)
    return {"model": model, "identity": identity, "config": manifest.config,
            "topology": msgspec.to_builtins(manifest.topology)}


def glm(args: argparse.Namespace) -> int:
    run, out = args.run.resolve(), args.out.resolve()
    native = (run / "engine.json").is_file() and not (run / "ready.json").is_file()
    if native:
        ready = json.loads((run / "engine.json").read_text())
        if ready["recipe"] != "glm53-native":
            raise ValueError("the native GLM benchmark requires the glm53-native recipe")
        options = native_launch(run, ready["model"])
        key = out / "api.key"
    else:
        ready = json.loads((run / "ready.json").read_text())
        options = json.loads((run / "launch.json").read_text())
        key = run / "api.key"
    addr = urlsplit(ready["url"])
    if not addr.hostname or not addr.port:
        raise ValueError("GLM run has no HTTP endpoint")
    suite = args.suite.resolve()
    suite_files = ("prepare.py", "glm_bench_adapter.py", "run_benchmark.py", "summarize.py")
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name in suite_files:
        shutil.copyfile(suite / name, out / name)
    if native:
        # The native endpoint is local. AISBench's HTTP adapter takes a key file in either mode.
        key.write_text("rsinfer-local\n")
        for name in ("run_benchmark.py", "summarize.py"):
            script = out / name
            script.write_text(script.read_text().replace("rf'^vllm:", "rf'^rsinfer:"))

    # The supplied suite generates the datasets, salt and AISBench settings. Point its copies
    # at this run without changing the workload or touching the reference directory.
    prepare = out / "prepare.py"
    text = prepare.read_text()
    text = text.replace('MODEL = Path("/local/models/GLM-5.3-Flash-w8a8")', f"MODEL = Path({options['model']!r})")
    prepare.write_text(text)

    def run_script(name: str, *argv: str) -> None:
        subprocess.run([sys.executable, str(out / name), *argv], cwd=out, check=True)

    run_script("prepare.py", str(key))
    parameters = out / "parameters.json"
    write_json(parameters, {**json.loads(parameters.read_text()), "api_base": ready["url"]})
    for case in ("short", "long"):
        config = out / case / "aisbench_config.py"
        config.write_text(config.read_text().replace(
            'host_ip="127.0.0.1", host_port=8000', f"host_ip={addr.hostname!r}, host_port={addr.port}"))
    metrics = ready.get("memcache_metrics")
    delta = {}
    before = fetch(metrics, timeout=5) if metrics else b""
    if metrics:
        (out / "memcache_before.prom").write_bytes(before)
    run_script("run_benchmark.py", str(key))
    if metrics:
        after = fetch(metrics, timeout=5)
        (out / "memcache_after.prom").write_bytes(after)
        initial = cache_counters(before)
        delta = {name: count - initial[name] for name, count in cache_counters(after).items()}
    run_script("summarize.py")
    accuracy = None
    if args.accuracy_check:
        command = [sys.executable, str(args.accuracy_check.resolve()), "--key-path", str(key),
                   "--out", str(out / "accuracy.json"), "--api-base", ready["url"]]
        if args.accuracy_baseline:
            command += ["--baseline", str(args.accuracy_baseline.resolve())]
        accuracy = subprocess.run(command, cwd=out, check=False).returncode
    write_json(out / "experiment.json", {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                                         "ready": ready, "launch": options, "reference_suite": str(suite),
                                         "memcache_delta": delta, "accuracy_exit_code": accuracy})
    return accuracy or 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="workload", required=True)
    dense = commands.add_parser("qwen")
    dense.add_argument("--url", default="http://127.0.0.1:8000")
    dense.add_argument("--model", default="Qwen3-0.6B")
    dense.add_argument("--tokenizer", type=Path, required=True)
    dense.add_argument("--out", type=Path, required=True)
    dense.add_argument("--samples", type=int, default=16)
    dense.add_argument("--concurrency", type=int, default=4)
    dense.add_argument("--prompt-tokens", type=int, default=512)
    dense.add_argument("--shared-prefix", type=int, default=0)
    dense.add_argument("--output-tokens", type=int, default=64)
    dense.add_argument("--repeat", type=int, default=3)
    dense.add_argument("--salt", default="0")
    dense.add_argument("--timeout", type=float, default=180)
    reference = commands.add_parser("glm53")
    reference.add_argument("--run", type=Path, required=True)
    reference.add_argument("--suite", type=Path, required=True)
    reference.add_argument("--accuracy-check", type=Path)
    reference.add_argument("--accuracy-baseline", type=Path)
    reference.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return qwen(args) if args.workload == "qwen" else glm(args)


if __name__ == "__main__":
    sys.exit(main())
