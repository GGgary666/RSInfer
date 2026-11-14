"""Manual end-to-end GPU job for the GitHub/tmux runner, not a pytest test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep", type=float, default=0)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()
    output = Path(os.environ["RESULT_DIR"])
    print(f"job={os.environ['SM100_JOB_ID']} CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
    a = torch.arange(256, dtype=torch.float32, device="cuda").reshape(16, 16)
    expected = a.cpu() @ a.cpu().T
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = a @ a.T
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        result = a @ a.T
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    props = torch.cuda.get_device_properties(0)
    metrics = {"device": props.name, "capability": [props.major, props.minor],
               "torch": torch.__version__, "cuda_graph": True, "matmul_equal_cpu": True}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "summary.md").write_text("# SM100 冒烟结果\n\n真实 CUDA graph 矩阵乘法与 CPU 结果相同。\n")
    print(f"GPU_SMOKE_OK {metrics}", flush=True)
    if args.sleep:
        print(f"sleeping {args.sleep}s (for a cancellation smoke)", flush=True)
        time.sleep(args.sleep)
    if args.fail:
        print("INTENTIONAL_FAILURE: partial results should still be uploaded", flush=True)
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
