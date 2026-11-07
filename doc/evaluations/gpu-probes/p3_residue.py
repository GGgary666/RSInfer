# Probe 3: where does the per-reload memory growth seen in probe 2 come from?
#
# usage: python p3_residue.py          # every mode, each in a fresh subprocess
#        python p3_residue.py <mode>   # one mode
#
# Reload modes repeat "capture a decode-like graph, replay, check, drop everything"
# ITERS times and record the caching allocator after each iteration:
#   p2_like            warm up on a new torch.cuda.Stream() each time, capture on torch's
#                      default capture stream (exactly what probe 2 did)
#   p2_like_clear_ws   the same, plus torch._C._cuda_clearCublasWorkspaces() after each unload
#   p2_like_no_cublas  the same, but the graph has no matmul (no cuBLAS call)
#   shared             warm up and capture on one stream owned by the "platform"
# pools: graph private memory pools -- separate vs shared, what keeps a pool alive, and
#        whether a graph replayed out of capture order clobbers a newer graph's output.
import gc, json, os, subprocess, sys, torch

ITERS = int(os.environ.get("ITERS", "80"))
RELOAD_MODES = ["p2_like", "p2_like_clear_ws", "p2_like_no_cublas", "shared"]


def mb(x):
    return round(x / 2**20, 1)


def snap():
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    st = torch.cuda.memory_stats()
    return dict(alloc=mb(torch.cuda.memory_allocated()), reserved=mb(torch.cuda.memory_reserved()),
                inactive_split=mb(st.get("inactive_split_bytes.all.current", 0)),
                segments=st.get("segment.all.current", 0))


def reload_loop(mode):
    dev = "cuda"
    torch.manual_seed(0)
    W = torch.randn(2048, 2048, device=dev)
    kv = torch.randn(16, 64, 2048, device=dev)
    x = torch.randn(64, 2048, device=dev)
    if mode == "p2_like_no_cublas":
        step, ref = (lambda: (x * 0.5 + kv.sum(0)).relu()), (lambda: (x * 0.5 + kv.sum(0)).relu())
    else:
        step, ref = (lambda: (x @ W) * 0.5 + kv.sum(0)), (lambda: (x @ W) * 0.5 + kv.sum(0))
    shared = torch.cuda.Stream()
    streams, rows = set(), []
    for it in range(ITERS):
        s = shared if mode == "shared" else torch.cuda.Stream()
        streams.add(s.cuda_stream)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=shared if mode == "shared" else None):
            y = step()
        g.replay(); torch.cuda.synchronize()
        ok = bool(torch.allclose(y, ref(), atol=1e-3, rtol=1e-3))
        del g, y
        if mode == "p2_like_clear_ws":
            torch._C._cuda_clearCublasWorkspaces()
        rows.append(dict(it=it, ok=ok, **snap()))
    pick = lambda i: rows[min(i, len(rows) - 1)]
    return dict(mode=mode, iterations=len(rows), all_ok=all(r["ok"] for r in rows),
                distinct_streams=len(streams), at0=pick(0), at31=pick(31), at32=pick(32),
                at63=pick(63), last=rows[-1],
                alloc_growth_0_31=round(pick(31)["alloc"] - pick(0)["alloc"], 1),
                alloc_growth_32_last=round(rows[-1]["alloc"] - pick(32)["alloc"], 1),
                reserved_growth_total=round(rows[-1]["reserved"] - rows[0]["reserved"], 1),
                cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))


def pools():
    dev = "cuda"
    a = torch.randn(4096, 4096, device=dev); b = torch.randn_like(a)   # 64 MiB each
    s = torch.cuda.Stream()

    def cap(fn, pool=None):
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool, stream=s):
            o = fn()
        return g, o

    g, o = cap(lambda: a @ b); del g, o          # warm cuBLAS on s outside the measurement
    base = snap()
    r = {}
    rel = lambda: {k: round(v - base[k], 1) for k, v in snap().items() if k in ("alloc", "reserved")}
    g1, o1 = cap(lambda: (a @ b).relu()); r["g1"] = rel()
    g2, o2 = cap(lambda: (a @ b).sigmoid()); r["g1_g2_separate_pools"] = rel()
    del g1; r["del_g1_keep_o1"] = rel()
    del o1; r["del_o1"] = rel()
    del g2, o2; r["del_g2_o2"] = rel()
    h = torch.cuda.graph_pool_handle()
    g3, o3 = cap(lambda: (a @ b).relu(), pool=h); r["g3_shared_pool"] = rel()
    g4, o4 = cap(lambda: (a @ b).sigmoid(), pool=h); r["g3_g4_shared_pool"] = rel()
    del g3, o3, g4, o4; r["del_shared"] = rel()
    torch._C._cuda_clearCublasWorkspaces(); r["after_clear_cublas_ws"] = rel()

    # Shared pool, captured in order g5 then g6, but g5's output is dropped before g6 is
    # captured (the "old version" no longer needs it). Replaying g5 after g6 -- out of
    # capture order, as an old graph would while a new one is being prepared -- may
    # overwrite g6's output.
    h2 = torch.cuda.graph_pool_handle()
    g5, o5 = cap(lambda: (a @ b) * 1.0, pool=h2)
    span5 = (o5.data_ptr(), o5.data_ptr() + o5.numel() * 4)
    del o5
    g6, o6 = cap(lambda: (a @ b) * 2.0, pool=h2)
    p6 = o6.data_ptr()
    r["shared_pool_new_output_overlaps_freed_old_output"] = span5[0] <= p6 < span5[1]
    g6.replay(); torch.cuda.synchronize()
    want = (a @ b) * 2.0
    r["g6_ok_before_old_replay"] = bool(torch.allclose(o6, want, rtol=1e-3, atol=1e-2))
    g5.replay(); torch.cuda.synchronize()
    r["g6_output_clobbered_by_old_replay"] = not bool(torch.allclose(o6, want, rtol=1e-3, atol=1e-2))
    return dict(mode="pools", base=base, rel_mb=r)


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        print("RESULT " + json.dumps(pools() if mode == "pools" else reload_loop(mode)), flush=True)
        return
    for mode in RELOAD_MODES + ["pools"]:
        p = subprocess.run([sys.executable, __file__, mode], capture_output=True, text=True)
        out = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        print(out[-1] if out else "RESULT " + json.dumps(dict(mode=mode, rc=p.returncode,
                                                                stderr=p.stderr[-2000:])), flush=True)


if __name__ == "__main__":
    main()
