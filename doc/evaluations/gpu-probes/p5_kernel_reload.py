# Probe 5: unload / re-import the kernel libraries Qwen3.8 needs, in one process, and
# re-capture decode graphs with them:
#   fla              GatedDeltaNet decode (fused_recurrent, Triton) under a graph, with the
#                    recurrent state in a slot pool. Qwen3.8-27B TP1 shapes: 16 key heads,
#                    48 value heads, head dim 128.
#   fla_prefill      one eager prefill call (chunk, Triton autotune), same shapes; bounded
#                    by a timeout because its autotune did not return on this stack.
#   fla_stale_graph  capture with fla, drop the library (sys.modules + every reference),
#                    then replay the old graph: does it still run the old kernels?
#   flashinfer       paged decode attention (JIT), Qwen3.8 full-attention shapes:
#                    24 q heads, 4 kv heads, head dim 256; capture, re-plan, replay.
# Every mode prints one ROW line per iteration, then a RESULT line.
#
# Needs fla-core and flashinfer-python importable (e.g. pip install --no-deps --target DIR
# fla-core flashinfer-python apache-tvm-ffi; PYTHONPATH=DIR).
# usage: python p5_kernel_reload.py            # every mode, each in a fresh subprocess
#        python p5_kernel_reload.py <mode>
import gc, importlib, itertools, json, math, os, subprocess, sys, time

MODES = {"fla": 1800, "fla_prefill": 600, "fla_stale_graph": 600, "flashinfer": 1800}   # timeout, s
if sys.argv[1:] and sys.argv[1].startswith("fla"):
    os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/probe_triton_cache_{os.getpid()}")   # cold
if sys.argv[1:] == ["flashinfer"]:
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", f"/tmp/probe_flashinfer_{os.getpid()}")  # cold

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda"


def ms(t):
    return round((time.perf_counter() - t) * 1e3, 1)


def purge(prefix):
    for n in [n for n in sys.modules if n == prefix or n.startswith(prefix + ".")]:
        del sys.modules[n]


def mem():
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    return dict(alloc_mb=round(torch.cuda.memory_allocated() / 2**20, 1),
                reserved_mb=round(torch.cuda.memory_reserved() / 2**20, 1))


def n_ops():
    return len(torch._C._dispatch_get_all_op_names())


def capture(fn, s):
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    return g, out


def close(a, b, tol):
    a, b = a.float(), b.float()
    return bool(((a - b).abs().max() / b.abs().max().clamp_min(1e-6)) < tol)


# ---------------------------------------------------------------- fla (GatedDeltaNet)
B, H, HV, K, V, SLOTS, TP = 32, 16, 48, 128, 128, 64, 256


def gdn_ref(q, k, v, g, beta, S):
    """Recurrent gated delta rule in fp32 (fla's naive_recurrent, with GVA and qk l2norm)."""
    r = HV // H
    q = F.normalize(q.float(), dim=-1).repeat_interleave(r, dim=2) * K ** -0.5
    k = F.normalize(k.float(), dim=-1).repeat_interleave(r, dim=2)
    v, g, beta, S = v.float(), g.float(), beta.float(), S.float().clone()
    o = torch.empty(v.shape, device=DEV)
    for t in range(q.shape[1]):
        S = S * g[:, t].exp()[..., None, None]
        vn = (v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], S)) * beta[:, t][..., None]
        S = S + k[:, t][..., None] * vn[..., None, :]
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
    return o, S


def gdn_inputs(b, t):
    q = torch.randn(b, t, H, K, device=DEV, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(b, t, HV, V, device=DEV, dtype=torch.bfloat16)
    g = -torch.rand(b, t, HV, device=DEV) * 0.5
    beta = torch.rand(b, t, HV, device=DEV).to(torch.bfloat16)
    return q, k, v, g, beta


def fla_setup():
    torch.manual_seed(0)
    dec = gdn_inputs(B, 1)
    pre = gdn_inputs(1, TP)
    pool = torch.randn(SLOTS, HV, K, V, device=DEV) * 0.05
    slots = torch.randperm(SLOTS, device=DEV)[:B]
    return dec, pre, pool, slots


def fla_decode_step(gdr, dec, pool, slots):
    q, k, v, g, beta = dec

    def step():
        o, S1 = gdr.fused_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=pool[slots], output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        pool.index_copy_(0, slots, S1)
        return o
    return step


def fla_check_replay(graph, out, dec, pool, slots):
    before = pool.clone()
    o_ref, S_ref = gdn_ref(*dec, before[slots])
    graph.replay(); torch.cuda.synchronize()
    return close(out, o_ref, 2e-2) and close(pool[slots], S_ref, 2e-2)


def fla_mode():
    iters = int(os.environ.get("ITERS", "10"))
    dec, pre, pool, slots = fla_setup()
    s = torch.cuda.Stream()
    rows, err, ops0 = [], None, n_ops()
    for it in range(iters):
        try:
            t = time.perf_counter(); gdr = importlib.import_module("fla.ops.gated_delta_rule"); imp = ms(t)
            step = fla_decode_step(gdr, dec, pool, slots)
            t = time.perf_counter(); step(); torch.cuda.synchronize(); first = ms(t)
            t = time.perf_counter(); graph, out = capture(step, s); cap = ms(t)
            dec_ok = fla_check_replay(graph, out, dec, pool, slots)
            del graph, out, step, gdr
            purge("fla")
            rows.append(dict(it=it, decode_ok=dec_ok, import_ms=imp, decode_first_call_ms=first,
                             capture_ms=cap, new_ops=n_ops() - ops0, **mem()))
            print("ROW " + json.dumps(rows[-1]), flush=True)
        except Exception as e:  # noqa: BLE001
            err = f"iteration {it}: {e!r}"[:400]
            break
    return dict(mode="fla", iterations=len(rows), error=err, all_ok=all(r["decode_ok"] for r in rows),
                first=rows[0] if rows else None, last=rows[-1] if rows else None,
                threads=len(os.listdir("/proc/self/task")))


def fla_prefill_mode():
    _, pre, _, _ = fla_setup()
    gdr = importlib.import_module("fla.ops.gated_delta_rule")
    call = lambda: gdr.chunk_gated_delta_rule(
        *pre, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    print("ROW " + json.dumps(dict(started="first chunk_gated_delta_rule call")), flush=True)
    t = time.perf_counter(); o, S = call(); torch.cuda.synchronize(); first = ms(t)
    t = time.perf_counter(); call(); torch.cuda.synchronize(); second = ms(t)
    o_ref, S_ref = gdn_ref(*pre, torch.zeros(1, HV, K, V, device=DEV))
    return dict(mode="fla_prefill", ok=close(o, o_ref, 3e-2) and close(S, S_ref, 3e-2),
                first_call_ms=first, second_call_ms=second)


def fla_stale_graph_mode():
    dec, _, pool, slots = fla_setup()
    s = torch.cuda.Stream()
    gdr = importlib.import_module("fla.ops.gated_delta_rule")
    graph, out = capture(fla_decode_step(gdr, dec, pool, slots), s)
    ok_before = fla_check_replay(graph, out, dec, pool, slots)
    del gdr
    purge("fla")
    gc.collect()
    err, ok_after = None, None
    try:
        ok_after = fla_check_replay(graph, out, dec, pool, slots)
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:300]
    still_loaded = [n for n in sys.modules if n == "fla" or n.startswith("fla.")]
    return dict(mode="fla_stale_graph", ok_before_purge=ok_before, ok_after_purge=ok_after,
                error=err, fla_modules_left=len(still_loaded))


# ---------------------------------------------------------------- flashinfer (paged decode)
NQ, NKV, D, PAGE, FB, NPAGES = 24, 4, 256, 16, 16, 8192


def fi_batch(lens, gen):
    npg = [(L + PAGE - 1) // PAGE for L in lens]
    perm = torch.randperm(NPAGES, generator=gen)[:sum(npg)].to(torch.int32)
    pages, off = [], 0
    for n in npg:
        pages.append(perm[off:off + n]); off += n
    indptr = torch.tensor([0, *itertools.accumulate(npg)], dtype=torch.int32)
    last = torch.tensor([(L - 1) % PAGE + 1 for L in lens], dtype=torch.int32)
    return indptr.to(DEV), perm.to(DEV), last.to(DEV), pages


def fi_ref(q, kc, vc, lens, pages):
    out = torch.empty(q.shape, device=DEV, dtype=torch.float32)
    r = NQ // NKV
    for b, (L, pg) in enumerate(zip(lens, pages)):
        Kb = kc[pg.long().to(DEV)].reshape(-1, NKV, D)[:L].float().repeat_interleave(r, dim=1)
        Vb = vc[pg.long().to(DEV)].reshape(-1, NKV, D)[:L].float().repeat_interleave(r, dim=1)
        p = torch.softmax(torch.einsum("hd,lhd->hl", q[b].float(), Kb) / math.sqrt(D), dim=-1)
        out[b] = torch.einsum("hl,lhd->hd", p, Vb)
    return out


def flashinfer_mode():
    iters = int(os.environ.get("ITERS", "4"))
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    kc = torch.randn(NPAGES, PAGE, NKV, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=DEV)
    bufs = (torch.zeros(FB + 1, dtype=torch.int32, device=DEV),
            torch.zeros(NPAGES, dtype=torch.int32, device=DEV),
            torch.zeros(FB, dtype=torch.int32, device=DEV))
    q = torch.randn(FB, NQ, D, device=DEV, dtype=torch.bfloat16)
    short = [int(x) for x in torch.randint(1, 64, (FB,), generator=gen)]
    long_ = [int(x) for x in torch.randint(1000, 4000, (FB,), generator=gen)]
    bA, bB = fi_batch(short, gen), fi_batch(long_, gen)
    s = torch.cuda.Stream()
    rows, err, ops0 = [], None, n_ops()
    for it in range(iters):
        try:
            t = time.perf_counter(); fi = importlib.import_module("flashinfer"); imp = ms(t)
            w = fi.BatchDecodeWithPagedKVCacheWrapper(
                ws, "NHD", use_cuda_graph=True, use_tensor_cores=True, paged_kv_indptr_buffer=bufs[0],
                paged_kv_indices_buffer=bufs[1], paged_kv_last_page_len_buffer=bufs[2])
            plan = lambda b: w.plan(b[0], b[1], b[2], NQ, NKV, D, PAGE,
                                    q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            t = time.perf_counter(); plan(bA); w.run(q, (kc, vc)); torch.cuda.synchronize(); first = ms(t)
            t = time.perf_counter(); graph, out = capture(lambda: w.run(q, (kc, vc)), s); cap = ms(t)
            graph.replay(); torch.cuda.synchronize()
            ok_a = close(out, fi_ref(q, kc, vc, short, bA[3]), 2e-2)
            # Re-plan for a batch of the same size but very different lengths, then replay
            # the graph captured with the old plan (the sglang "metadata out of graph" path).
            t = time.perf_counter(); plan(bB); replan = ms(t)
            q.copy_(torch.randn_like(q))
            graph.replay(); torch.cuda.synchronize()
            ref_b = fi_ref(q, kc, vc, long_, bB[3])
            ok_b = close(out, ref_b, 2e-2)
            eager_b = w.run(q, (kc, vc)); torch.cuda.synchronize()
            rows.append(dict(it=it, short_ok=ok_a, replan_long_then_replay_ok=ok_b,
                             eager_long_ok=close(eager_b, ref_b, 2e-2),
                             replay_bitwise_equals_eager=bool(torch.equal(out, eager_b)),
                             import_ms=imp, plan_and_first_run_ms=first, capture_ms=cap,
                             replan_ms=replan))
            plan(bA)
            del graph, out, eager_b, w, plan, fi
            purge("flashinfer")
            rows[-1].update(new_ops=n_ops() - ops0, **mem())
            print("ROW " + json.dumps(rows[-1]), flush=True)
        except Exception as e:  # noqa: BLE001
            err = f"iteration {it}: {e!r}"[:600]
            break
    return dict(mode="flashinfer", iterations=len(rows), error=err,
                threads=len(os.listdir("/proc/self/task")))


def main():
    if sys.argv[1:]:
        fn = dict(fla=fla_mode, fla_prefill=fla_prefill_mode, fla_stale_graph=fla_stale_graph_mode,
                  flashinfer=flashinfer_mode)
        print("RESULT " + json.dumps(fn[sys.argv[1]]()), flush=True)
        return
    for mode, timeout in MODES.items():
        try:
            p = subprocess.run([sys.executable, __file__, mode], timeout=timeout)
            if p.returncode:
                print("RESULT " + json.dumps(dict(mode=mode, rc=p.returncode)), flush=True)
        except subprocess.TimeoutExpired:
            print("RESULT " + json.dumps(dict(mode=mode, timed_out_after_s=timeout)), flush=True)


if __name__ == "__main__":
    main()
