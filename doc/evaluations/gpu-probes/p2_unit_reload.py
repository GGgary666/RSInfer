# Probe 2: unload / re-import / reinstall a unit that owns a torch.library op,
# a module-level CUDA stream and a Triton kernel, and re-capture a decode graph.
import gc, importlib, json, os, sys, time, torch

UNIT = '''
import torch, triton, triton.language as tl
STREAM = torch.cuda.Stream()
LIB = torch.library.Library("probeunit", "FRAGMENT")
LIB.define("scale(Tensor x, float a) -> Tensor")

@triton.jit
def _scale(x, o, a, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) * a, mask=m)

def _impl(x, a):
    o = torch.empty_like(x)
    _scale[(triton.cdiv(x.numel(), 1024),)](x, o, a, x.numel(), BLOCK=1024)
    return o
LIB.impl("scale", _impl, "CUDA")

class Model:
    """Decode step over state owned by the holder: h = scale(x @ W, 0.5) + kv.sum(0)."""
    def __init__(self, state):
        self.W, self.kv = state["W"], state["kv"]
    def step(self, x):
        return torch.ops.probeunit.scale(x @ self.W, 0.5) + self.kv.sum(0)

def unload():
    LIB._destroy()
'''

MODE = sys.argv[1] if len(sys.argv) > 1 else "destroy"
d = "/tmp/probe_units"; os.makedirs(d, exist_ok=True)
open(f"{d}/probe_unit.py", "w").write(UNIT)
sys.path.insert(0, d)
dev = "cuda"
state = {"W": torch.randn(2048, 2048, device=dev), "kv": torch.randn(16, 64, 2048, device=dev)}
x = torch.randn(64, 2048, device=dev)


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    return g, out


rows, err = [], None
base_res = None
for it in range(30):
    try:
        t0 = time.perf_counter()
        mod = importlib.import_module("probe_unit")
        t1 = time.perf_counter()
        model = mod.Model(state)
        g, y = capture(lambda: model.step(x))
        t2 = time.perf_counter()
        g.replay(); torch.cuda.synchronize()
        ok = bool(torch.allclose(y, (x @ state["W"]) * 0.5 + state["kv"].sum(0), atol=1e-3, rtol=1e-3))
        # Unload in the prescribed order: user (graph, model) first, then the unit module.
        del g, y, model
        if MODE == "destroy":
            mod.unload()
        del mod; sys.modules.pop("probe_unit", None)
        gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
        t3 = time.perf_counter()
        st = torch.cuda.memory_stats()
        rows.append(dict(it=it, ok=ok, import_ms=round((t1 - t0) * 1e3, 1),
                         build_capture_ms=round((t2 - t1) * 1e3, 1), unload_ms=round((t3 - t2) * 1e3, 1),
                         alloc_mb=round(torch.cuda.memory_allocated() / 2**20, 1),
                         reserved_mb=round(torch.cuda.memory_reserved() / 2**20, 1),
                         inactive_split_mb=round(st.get("inactive_split_bytes.all.current", 0) / 2**20, 1),
                         op_still_registered=hasattr(torch.ops.probeunit, "scale")))
    except Exception as e:  # noqa: BLE001
        err = f"iteration {it}: {e!r}"[:300]
        break

summary = dict(mode=MODE, iterations=len(rows), error=err,
               all_ok=all(r["ok"] for r in rows),
               first=rows[0] if rows else None, last=rows[-1] if rows else None,
               reserved_growth_mb=(rows[-1]["reserved_mb"] - rows[0]["reserved_mb"]) if rows else None,
               threads=len(os.listdir("/proc/self/task")))
print("RESULT " + json.dumps(summary))
