# Probe 10: what a capture costs and what it tolerates.
#   phases   The fla decode reload loop of probe 5, with the capture split into its phases:
#            two warm-up calls, synchronize, empty_cache, host empty_cache, gc.collect, and the
#            recording itself. Looks for where probe 5's 0.2-5 s capture times on stack B went.
#   threads  Another thread keeps running eager CUDA work (allocations, matmuls, checked
#            against a CPU-side expectation) while the main thread captures, once per
#            capture_error_mode. Does the capture fail, does the other thread fail, are results
#            wrong? Each mode runs in a fresh subprocess.
# usage: python p10_capture_internals.py [phases|threads|threads:<mode>]
import gc, importlib, json, os, subprocess, sys, threading, time

if sys.argv[1:] == ["phases"]:
    os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/p10_triton_cache_{os.getpid()}")
import torch  # noqa: E402

DEV = "cuda"


def ms(t):
    return round((time.perf_counter() - t) * 1e3, 1)


def purge(prefix):
    for n in [n for n in sys.modules if n == prefix or n.startswith(prefix + ".")]:
        del sys.modules[n]


def phases_mode():
    B, H, HV, K, V, SLOTS = 32, 16, 48, 128, 128, 64
    torch.manual_seed(0)
    q = torch.randn(B, 1, H, K, device=DEV, dtype=torch.bfloat16); k = torch.randn_like(q)
    v = torch.randn(B, 1, HV, V, device=DEV, dtype=torch.bfloat16)
    g = -torch.rand(B, 1, HV, device=DEV) * 0.5
    beta = torch.rand(B, 1, HV, device=DEV).to(torch.bfloat16)
    pool = torch.randn(SLOTS, HV, K, V, device=DEV) * 0.05
    slots = torch.randperm(SLOTS, device=DEV)[:B]
    s = torch.cuda.Stream()
    rows = []
    for it in range(int(os.environ.get("ITERS", "10"))):
        purge("fla")
        t = time.perf_counter(); gdr = importlib.import_module("fla.ops.gated_delta_rule"); imp = ms(t)

        def step():
            o, s1 = gdr.fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=pool[slots],
                                                         output_final_state=True, use_qk_l2norm_in_kernel=True)
            pool.index_copy_(0, slots, s1)
            return o
        t = time.perf_counter(); step(); torch.cuda.synchronize(); first = ms(t)
        r = dict(it=it, import_ms=imp, first_call_ms=first, gc_objects=len(gc.get_objects()))
        s.wait_stream(torch.cuda.current_stream())
        t = time.perf_counter()
        with torch.cuda.stream(s):
            step()
        torch.cuda.synchronize(); r["warmup1_ms"] = ms(t)
        t = time.perf_counter()
        with torch.cuda.stream(s):
            step()
        torch.cuda.synchronize(); r["warmup2_ms"] = ms(t)
        torch.cuda.current_stream().wait_stream(s)
        t = time.perf_counter(); torch.cuda.synchronize(); r["sync_ms"] = ms(t)
        t = time.perf_counter(); torch.cuda.empty_cache(); r["empty_cache_ms"] = ms(t)
        t = time.perf_counter(); torch._C._host_emptyCache(); r["host_empty_cache_ms"] = ms(t)
        t = time.perf_counter(); gc.collect(); r["gc_collect_ms"] = ms(t)
        graph = torch.cuda.CUDAGraph()
        t = time.perf_counter()
        with torch.cuda.graph(graph, stream=s):
            step()
        r["graph_context_ms"] = ms(t)
        t = time.perf_counter(); graph.replay(); torch.cuda.synchronize(); r["replay_ms"] = ms(t)
        rows.append(r)
        print("ROW " + json.dumps(r), flush=True)
        del graph, step, gdr
    keys = [k for k in rows[0] if k.endswith("_ms")]
    return dict(mode="phases", iterations=len(rows), torch=torch.__version__,
                ranges={k: [min(r[k] for r in rows[1:]), max(r[k] for r in rows[1:])] for k in keys})


def threads_one(mode):
    stop, errors, bad, done = threading.Event(), [], [0], [0]
    w = torch.randn(1024, 1024, device=DEV)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):           # the expected value from the same kernel, before any capture
        exp = (torch.full((1024, 1024), 1.0, device=DEV) @ w)[0].clone()
    side.synchronize()

    def other():
        torch.cuda.set_device(0)
        try:
            with torch.cuda.stream(side):
                while not stop.is_set():
                    x = torch.full((1024, 1024), 1.0, device=DEV)   # allocation on another stream
                    y = x @ w
                    if not torch.equal(y[0], exp):
                        bad[0] += 1
                    done[0] += 1
                    side.synchronize()
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e)[:300])

    a = torch.randn(2048, 2048, device=DEV)
    s = torch.cuda.Stream()

    def body():
        y = a
        for _ in range(2000):
            y = torch.tanh(y @ a * 1e-3)
        return y
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    th = threading.Thread(target=other); th.start()
    time.sleep(0.5)
    before = done[0]
    cap_err = None
    g = torch.cuda.CUDAGraph()
    t = time.perf_counter()
    try:
        with torch.cuda.graph(g, stream=s, capture_error_mode=mode):
            out = body()
    except Exception as e:  # noqa: BLE001
        cap_err = repr(e)[:300]
    cap = ms(t)
    during = done[0] - before
    time.sleep(0.5)
    stop.set(); th.join(timeout=30)
    replay_ok = None
    if cap_err is None:
        g.replay(); torch.cuda.synchronize()
        replay_ok = bool(torch.allclose(out, body(), rtol=1e-3, atol=1e-3))
    return dict(mode=f"threads:{mode}", capture_error=cap_err, capture_ms=cap, other_thread_errors=errors,
                other_iterations_total=done[0], other_iterations_during_capture=during,
                other_wrong_results=bad[0], replay_ok=replay_ok, torch=torch.__version__)


def main():
    arg = sys.argv[1] if sys.argv[1:] else "all"
    if arg == "phases":
        print("RESULT " + json.dumps(phases_mode()), flush=True)
    elif arg.startswith("threads:"):
        print("RESULT " + json.dumps(threads_one(arg.split(":", 1)[1])), flush=True)
    else:
        modes = [f"threads:{x}" for x in ("global", "thread_local", "relaxed")]
        if arg == "all":
            modes = ["phases"] + modes
        for m in modes:
            p = subprocess.run([sys.executable, __file__, m], capture_output=True, text=True, timeout=1200)
            out = [l for l in p.stdout.splitlines() if l.startswith(("RESULT", "ROW"))]
            print("\n".join(out) if out else "RESULT " + json.dumps(dict(mode=m, rc=p.returncode,
                                                                           stderr=p.stderr[-1500:])), flush=True)


if __name__ == "__main__":
    main()
