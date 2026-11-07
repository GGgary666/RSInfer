# Probe 1: what a real CUDA graph actually depends on (by address vs by value).
import gc, json, torch, triton, triton.language as tl

dev = "cuda"
torch.manual_seed(0)
res = {}


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def close(a, b):
    return bool(torch.allclose(a.float(), b.float(), atol=1e-1, rtol=1e-2))


x = torch.randn(64, 1024, device=dev, dtype=torch.bfloat16)
W = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
holder = {"W": W, "alpha": 0.5}
g, y = capture(lambda: torch.matmul(x, holder["W"]) * holder["alpha"])
g.replay(); torch.cuda.synchronize()
res["baseline_ok"] = close(y, x @ W * 0.5)

W2 = torch.randn_like(W)
W.copy_(W2); g.replay(); torch.cuda.synchronize()
res["refit_in_place_seen_by_replay"] = close(y, x @ W2 * 0.5)

x2 = torch.randn_like(x)
x.copy_(x2); g.replay(); torch.cuda.synchronize()
res["input_content_change_seen"] = close(y, x2 @ W2 * 0.5)

holder["alpha"] = 2.0
g.replay(); torch.cuda.synchronize()
res["python_scalar_baked_at_capture"] = close(y, x2 @ W2 * 0.5)

W3 = torch.randn_like(W)
holder["W"] = W3
g.replay(); torch.cuda.synchronize()
res["rebinding_to_new_address_ignored_silently"] = close(y, x2 @ W2 * 0.5) and not close(y, x2 @ W3 * 0.5)

# Free the captured weight while the graph is alive, then reuse the memory.
expected = (x2 @ W2 * 0.5).clone()
del W, W2
gc.collect(); torch.cuda.empty_cache()
junk = [torch.full((1024, 1024), 7.0, device=dev, dtype=torch.bfloat16) for _ in range(64)]
err = None
try:
    g.replay(); torch.cuda.synchronize()
except Exception as e:  # noqa: BLE001
    err = repr(e)
res["replay_after_holder_freed_raises"] = err
res["replay_after_holder_freed_reads_garbage"] = (err is None) and not close(y, expected)
del junk

# Triton kernel inside a graph; refit input.
@triton.jit
def _add(a, b, o, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(o + i, tl.load(a + i, mask=m) + tl.load(b + i, mask=m), mask=m)

a = torch.randn(1 << 20, device=dev); b = torch.randn_like(a); o = torch.empty_like(a)
gt, _ = capture(lambda: _add[(a.numel() // 1024,)](a, b, o, a.numel(), BLOCK=1024))
a.copy_(torch.ones_like(a)); gt.replay(); torch.cuda.synchronize()
res["triton_in_graph_refit_ok"] = close(o, torch.ones_like(a) + b)

# Graph private pools: two graphs alive at once, then drop one.
def reserved():
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    return torch.cuda.memory_reserved()

r0 = reserved()
bufs = [torch.randn(4096, 4096, device=dev) for _ in range(2)]
g1, _ = capture(lambda: (bufs[0] @ bufs[1]).relu())
r1 = reserved()
g2, _ = capture(lambda: (bufs[0] @ bufs[1]).sigmoid())
r2 = reserved()
del g1; r3 = reserved()
del g2; r4 = reserved()
res["pool_mb"] = {k: round((v - r0) / 2**20) for k, v in
                  dict(after_g1=r1, after_g2=r2, drop_g1=r3, drop_g2=r4).items()}
res["capability"] = list(torch.cuda.get_device_capability(0))
res["sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
res["torch"] = torch.__version__
print("RESULT " + json.dumps(res))
