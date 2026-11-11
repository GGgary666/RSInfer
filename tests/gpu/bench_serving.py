"""Serving throughput over HTTP, the same load against any OpenAI-compatible server: this engine
(``deploy/qwen38.py serve``) or a native SGLang serving the same checkpoint.

Run manually; this is not a pytest timing assertion. Per concurrency level, that many clients
each send requests one after another (``/v1/completions``, streamed, greedy, ``ignore_eos``, so
every request generates ``--max-tokens`` tokens). Every request of a run has its own prompt, drawn
from a seed: no two share a prefix worth caching, also not across levels (a server with a prefix
cache would otherwise answer a later level's prompts from an earlier level's), and two servers
under test get the same prompts.

    uv run python tests/gpu/bench_serving.py --url http://127.0.0.1:8000 --name agxinfer \\
        --output run/bench/agxinfer.json

Reported per level: output tokens per second over the whole level (all requests, wall clock),
and per request the time to the first token, the time per output token after the first (from
the first streamed event to the last, over the tokens the server counted less one), and the
latency. One request per client is sent before each level and not counted (kernels compiled
on first use, graphs of a batch size replayed for the first time).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

WORDS = ("the of and to in is that for it as was with be by on not he this are or his from at which but have an "
         "had they you were their one all we can her has there been if more when will would who so no out up "
         "into do any your what than them some could time these two may then other its only new very over such "
         "our man me even most made after also did many before must through years where much way well down "
         "should because each just those people how too little state good world make still see own work long "
         "get here between both life being under never day same another know while last might us great old "
         "year off come since against go came right used take three").split()


def prompt(seed: int, words: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(words))


async def one(client: httpx.AsyncClient, url: str, model: str, text: str, max_tokens: int) -> dict[str, Any]:
    """One streamed completion: when it started, and when its first and its last token arrived. A
    token's event counts whether or not it carries text: a server holds back the bytes of an
    incomplete character, and the text of a stop token."""
    body = {"model": model, "prompt": text, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    start = time.perf_counter()
    first = last = 0.0
    usage: dict[str, int] = {}
    async with client.stream("POST", url + "/v1/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data: "):])
            if "error" in event:
                raise RuntimeError(f"the server answered {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            if event.get("choices"):
                last = time.perf_counter()
                first = first or last
    end = time.perf_counter()
    tokens = int(usage.get("completion_tokens", max_tokens))
    return {"start": start, "ttft_s": (first or end) - start, "latency_s": end - start, "completion_tokens": tokens,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "tpot_ms": 1000 * (last - first) / (tokens - 1) if tokens > 1 and last > first else 0.0}


async def level(url: str, model: str, clients: int, requests: int, words: int, max_tokens: int, seed: int,
                timeout: float) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=clients, max_keepalive_connections=clients)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def run(index: int, count: int, base: int) -> list[dict[str, Any]]:
            return [await one(client, url, model, prompt(base + 1000 * index + i, words), max_tokens)
                    for i in range(count)]

        await asyncio.gather(*(run(c, 1, seed + 500_000_000) for c in range(clients)))  # not counted
        started = time.perf_counter()
        results = [r for per_client in await asyncio.gather(*(run(c, requests, seed) for c in range(clients)))
                   for r in per_client]
        elapsed = time.perf_counter() - started
    tokens = sum(r["completion_tokens"] for r in results)

    def summary(key: str) -> dict[str, float]:
        values = sorted(r[key] for r in results)
        return {"median": statistics.median(values), "p99": values[min(len(values) - 1, int(0.99 * len(values)))]}

    return {"clients": clients, "requests": len(results), "seconds": elapsed, "completion_tokens": tokens,
            "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in results),
            "output_tokens_per_second": tokens / elapsed,
            "input_tokens_per_second": sum(r["prompt_tokens"] for r in results) / elapsed,
            "ttft_s": summary("ttft_s"), "tpot_ms": summary("tpot_ms"), "latency_s": summary("latency_s")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="the server, e.g. http://127.0.0.1:8000")
    parser.add_argument("--name", required=True, help="what is being measured; kept in the output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3.8-27B", help="the model name the requests carry")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--requests", type=int, default=4, help="requests per client and level")
    parser.add_argument("--prompt-words", type=int, default=32, help="words per prompt (about as many tokens)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if min(args.requests, args.prompt_words, args.max_tokens, *args.concurrency) < 1:
        parser.error("counts must be positive")
    result: dict[str, Any] = {"name": args.name, "config": {k: str(v) if isinstance(v, Path) else v
                                                           for k, v in vars(args).items()}, "levels": []}
    for index, clients in enumerate(args.concurrency):
        measured = asyncio.run(level(args.url, args.model, clients, args.requests, args.prompt_words,
                                     args.max_tokens, args.seed + 1_000_000 * index, args.timeout))
        result["levels"].append(measured)
        print("BENCH_LEVEL " + json.dumps({"name": args.name, **measured}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("AGXINFER_SERVING_BENCH_OK", flush=True)


if __name__ == "__main__":
    main()
