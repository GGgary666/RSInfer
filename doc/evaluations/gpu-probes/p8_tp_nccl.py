# Probe 8: tensor parallelism with NCCL under CUDA graphs, and what changing TP costs.
#   inproc   4 rank processes. A TP4 MLP decode step (column-parallel, row-parallel, one
#            all_reduce) captured in a graph; in-place refit; communicator re-creation loop
#            (leak check); TP4 -> TP2 inside the same processes (new group, reshard,
#            recapture), then the TP4 graph and group are dropped.
#   restart  stop the 4 rank processes and start 2 new ones with TP2, which is what the
#            action plan does for a TP change: time from spawn to the first correct step.
# Needs 4 idle GPUs (CUDA_VISIBLE_DEVICES).
# usage: python p8_tp_nccl.py            # both modes
#        python p8_tp_nccl.py inproc|restart
import gc, json, os, socket, sys, time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

D, FF, B, ITERS = 4096, 16384, 32, 8
WFILE = os.environ.get("P8_WEIGHTS", "/tmp/p8_weights.pt")   # inherited by the rank processes


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def weights():
    g = torch.Generator().manual_seed(0)
    return (torch.randn(D, FF, generator=g) * D ** -0.5).to(torch.bfloat16), \
           (torch.randn(FF, D, generator=g) * FF ** -0.5).to(torch.bfloat16)


def shard(W1, W2, r, tp):
    f = FF // tp
    return W1[:, r * f:(r + 1) * f].contiguous().cuda(), W2[r * f:(r + 1) * f].contiguous().cuda()


def step(x, w1, w2, group):
    y = F.gelu(x @ w1) @ w2
    dist.all_reduce(y, group=group)
    return y


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


def rel_err(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def free_mib():
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    return round(torch.cuda.mem_get_info()[0] / 2**20)


def ms(t):
    return round((time.perf_counter() - t) * 1e3, 1)


def init(rank, world, port):
    t = time.perf_counter()
    torch.cuda.set_device(rank)
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    cuda_ms = ms(t)
    f0 = free_mib()
    t = time.perf_counter()
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=120), device_id=torch.device("cuda", rank))
    z = torch.ones(1, device="cuda"); dist.all_reduce(z); torch.cuda.synchronize()
    return dict(cuda_context_ms=cuda_ms, init_and_first_allreduce_ms=ms(t), world_comm_mib=f0 - free_mib())


def inproc_rank(rank, world, port, q):
    try:
        r = dict(rank=rank, **init(rank, world, port))
        W1, W2 = weights()
        torch.manual_seed(1)
        x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        ref = F.gelu(x @ W1.cuda()) @ W2.cuda()
        s = torch.cuda.Stream()

        # TP4 in a graph, on its own group.
        g4 = dist.new_group(list(range(world)))
        w1, w2 = shard(W1, W2, rank, world)
        graph4, out4 = capture(lambda: step(x, w1, w2, g4), s)
        graph4.replay(); torch.cuda.synchronize()
        r["tp4_graph_rel_err"] = rel_err(out4, ref)
        # Refit in place (new W2 values, same addresses): no recapture.
        W2b = (W2.float() * 0.5).to(torch.bfloat16)
        w2.copy_(shard(W1, W2b, rank, world)[1])
        graph4.replay(); torch.cuda.synchronize()
        r["tp4_refit_rel_err"] = rel_err(out4, F.gelu(x @ W1.cuda()) @ W2b.cuda())
        w2.copy_(shard(W1, W2, rank, world)[1])

        # Communicator re-creation loop: new group, capture, replay, drop graph, destroy group.
        rows, base = [], None
        for it in range(ITERS):
            t = time.perf_counter()
            gi = dist.new_group(list(range(world)))
            z = torch.ones(1, device="cuda"); dist.all_reduce(z, group=gi); torch.cuda.synchronize()
            t_new = ms(t)
            t = time.perf_counter(); gr, out = capture(lambda: step(x, w1, w2, gi), s); t_cap = ms(t)
            gr.replay(); torch.cuda.synchronize()
            err = rel_err(out, ref)
            del gr, out
            dist.destroy_process_group(gi)
            f = free_mib()
            base = f if base is None else base
            rows.append(dict(it=it, new_group_ms=t_new, capture_ms=t_cap, rel_err=err,
                             free_drop_mib=base - f, threads=len(os.listdir("/proc/self/task"))))
        r["recreate"] = dict(first=rows[0], last=rows[-1], max_rel_err=max(x["rel_err"] for x in rows))

        # TP4 -> TP2 in the same processes. new_group is called by every rank.
        f_before = free_mib()
        t = time.perf_counter()
        g2 = dist.new_group([0, 1])
        if rank < 2:
            z = torch.ones(1, device="cuda"); dist.all_reduce(z, group=g2); torch.cuda.synchronize()
            r["tp2_new_group_ms"] = ms(t)
            t = time.perf_counter(); w1b, w2b = shard(W1, W2, rank, 2); torch.cuda.synchronize()
            r["tp2_reshard_ms"] = ms(t)
            t = time.perf_counter(); graph2, out2 = capture(lambda: step(x, w1b, w2b, g2), s)
            r["tp2_capture_ms"] = ms(t)
            graph2.replay(); torch.cuda.synchronize()
            r["tp2_graph_rel_err"] = rel_err(out2, ref)
            r["tp2_vs_tp4_bitwise_equal"] = bool(torch.equal(out2, out4))
        # Old version goes: users (graph, outputs, shards) first, then the communicator.
        del graph4, out4, w1, w2
        dist.destroy_process_group(g4)
        r["free_gain_after_dropping_tp4_mib"] = free_mib() - f_before
        if rank < 2:
            graph2.replay(); torch.cuda.synchronize()
            r["tp2_ok_after_dropping_tp4"] = rel_err(out2, ref) < 2e-2
            del graph2, out2
        dist.barrier()
        dist.destroy_process_group()
        r["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        q.put(r)
    except Exception as e:  # noqa: BLE001
        q.put(dict(rank=rank, error=repr(e)[:500]))


def restart_rank(rank, world, port, t_spawn, q):
    try:
        t_start = time.time()
        r = dict(rank=rank, tp=world, spawn_to_main_ms=round((t_start - t_spawn) * 1e3, 1),
                 **init(rank, world, port))
        t = time.perf_counter()
        W = torch.load(WFILE, mmap=True)
        w1, w2 = shard(W["W1"], W["W2"], rank, world); torch.cuda.synchronize()
        r["load_and_shard_ms"] = ms(t)
        torch.manual_seed(1)
        x = torch.randn(B, D, device="cuda", dtype=torch.bfloat16)
        t = time.perf_counter(); graph, out = capture(lambda: step(x, w1, w2, dist.group.WORLD), torch.cuda.Stream())
        r["capture_ms"] = ms(t)
        graph.replay(); torch.cuda.synchronize()
        r["spawn_to_first_step_ms"] = round((time.time() - t_spawn) * 1e3, 1)
        r["rel_err"] = rel_err(out, F.gelu(x @ W["W1"].cuda()) @ W["W2"].cuda())
        del graph, out
        dist.barrier()
        dist.destroy_process_group()
        q.put(r)
    except Exception as e:  # noqa: BLE001
        q.put(dict(rank=rank, tp=world, error=repr(e)[:500]))


def launch(target, world, extra=(), timeout=600):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    ps = [ctx.Process(target=target, args=(r, world, port, *extra, q)) for r in range(world)]
    for p in ps:
        p.start()
    out = []
    t_end = time.time() + timeout
    while len(out) < world and time.time() < t_end:
        try:
            out.append(q.get(timeout=5))
        except Exception:  # noqa: BLE001
            if not any(p.is_alive() for p in ps):
                break
    for p in ps:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    return sorted(out, key=lambda r: r["rank"]), [p.exitcode for p in ps]


def main():
    modes = sys.argv[1:] or ["inproc", "restart"]
    if "inproc" in modes:
        res, codes = launch(inproc_rank, 4)
        print("RESULT " + json.dumps(dict(mode="inproc", exitcodes=codes, ranks=res)), flush=True)
    if "restart" in modes:
        W1, W2 = weights()
        torch.save(dict(W1=W1, W2=W2), WFILE)
        out = {}
        for tp in (4, 2):
            t = time.time()
            res, codes = launch(restart_rank, tp, extra=(t,))
            out[f"tp{tp}"] = dict(exitcodes=codes, wall_ms=round((time.time() - t) * 1e3, 1), ranks=res)
        os.remove(WFILE)
        print("RESULT " + json.dumps(dict(mode="restart", **out)), flush=True)


if __name__ == "__main__":
    main()
