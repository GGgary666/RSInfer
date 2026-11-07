# Probe 11: changing TP with Qwen3.8-27B's real weights (the text model and MTP, bf16, ~51 GiB).
#   restart  2 new rank processes for TP2: CUDA context, NCCL init, then each rank reads its half
#            of every tensor straight from the checkpoint and copies it to its GPU.
#   inproc   4 rank processes read their TP4 quarters into one flat buffer each. Then TP4 -> TP2
#            in the same processes: rank 1 sends its buffer to rank 0 and rank 3 to rank 2 over
#            NVLink (NCCL send/recv), ranks 0 and 2 assemble their halves, ranks 1 and 3 free
#            theirs. A sample of the halves is compared with the checkpoint, and the time to read
#            the missing quarter from the checkpoint instead is measured too.
# Sharding: every tensor with 2+ dims along dim 0 when it divides by 4 (all of Qwen3.8's do),
# others replicated. Real TP splits some tensors along dim 1; the bytes moved are the same.
# Before each mode the share of the checkpoint already in the page cache is reported (mincore).
# usage: python p11_tp_real_weights.py --model DIR [restart] [inproc]
import ctypes, gc, json, mmap, os, socket, sys, time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

PAGE_SZ = os.sysconf("SC_PAGE_SIZE")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def ms(t):
    return round((time.perf_counter() - t) * 1e3, 1)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def plan(model):
    """(file, name, shape, sharded) for every non-vision tensor, in checkpoint order."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    out = []
    for f in sorted(set(idx.values())):
        with safe_open(os.path.join(model, f), "pt") as s:
            for k in s.keys():
                if "visual" in k:
                    continue
                shape = s.get_slice(k).get_shape()
                out.append((f, k, shape, len(shape) >= 2 and shape[0] % 4 == 0))
    return out


def resident_fraction(model, files):
    import numpy as np
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    total = res = 0
    for f in files:
        path = os.path.join(model, f)
        n = os.path.getsize(path)
        pages = (n + PAGE_SZ - 1) // PAGE_SZ
        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), n, access=mmap.ACCESS_COPY)    # private: nothing is written back
            buf = ctypes.c_char.from_buffer(mm)
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(n), vec) == 0:
                res += int((np.frombuffer(vec, dtype=np.uint8) & 1).sum())
            del buf
            mm.close()
        total += pages
    return round(res / total, 3)


def piece(shape, r, tp):
    rows = shape[0] // tp
    return r * rows, (r + 1) * rows


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def load(model, items, r, tp):
    """Read this rank's piece of every sharded tensor into one flat GPU buffer; replicate the rest."""
    from safetensors import safe_open
    sizes = [numel(sh) // tp if sharded else 0 for _, _, sh, sharded in items]
    flat = torch.empty(sum(sizes), dtype=torch.bfloat16, device="cuda")
    rep, off, nbytes, opened = {}, 0, 0, {}
    for (f, k, sh, sharded), n in zip(items, sizes):
        s = opened.get(f) or opened.setdefault(f, safe_open(os.path.join(model, f), "pt"))
        if sharded:
            a, b = piece(sh, r, tp)
            t = s.get_slice(k)[a:b]
            flat[off:off + n].copy_(t.reshape(-1))
            off += n
        else:
            t = s.get_tensor(k)
            rep[k] = t.cuda()
        nbytes += t.numel() * t.element_size()
    torch.cuda.synchronize()
    return flat, rep, sizes, nbytes


def init(rank, world, port):
    t = time.perf_counter()
    torch.cuda.set_device(rank)
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    r = dict(cuda_context_ms=ms(t))
    t = time.perf_counter()
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            timeout=timedelta(seconds=600), device_id=torch.device("cuda", rank))
    z = torch.ones(1, device="cuda"); dist.all_reduce(z); torch.cuda.synchronize()
    r["nccl_init_ms"] = ms(t)
    return r


def restart_rank(rank, world, port, model, items, t_spawn, q):
    try:
        r = dict(rank=rank, spawn_to_main_ms=round((time.time() - t_spawn) * 1e3, 1), **init(rank, world, port))
        t = time.perf_counter()
        flat, rep, _, nbytes = load(model, items, rank, world)
        r["load_s"] = round(ms(t) / 1e3, 2)
        r["load_gib"] = round(nbytes / 2**30, 2)
        r["load_gib_per_s"] = round(nbytes / 2**30 / r["load_s"], 2)
        dist.barrier(); torch.cuda.synchronize()
        r["spawn_to_weights_loaded_s"] = round(time.time() - t_spawn, 2)
        dist.destroy_process_group()
        q.put(r)
    except Exception as e:  # noqa: BLE001
        q.put(dict(rank=rank, error=repr(e)[:500]))


def inproc_rank(rank, world, port, model, items, t_spawn, q):
    try:
        r = dict(rank=rank, **init(rank, world, port))
        t = time.perf_counter()
        flat, rep, sizes, nbytes = load(model, items, rank, 4)
        r["tp4_load_s"] = round(ms(t) / 1e3, 2)
        r["tp4_load_gib"] = round(nbytes / 2**30, 2)
        dist.barrier()
        free0 = torch.cuda.mem_get_info()[0]

        # TP4 -> TP2 on ranks {0, 2}: 1 -> 0 and 3 -> 2. Every rank calls new_group.
        t = time.perf_counter()
        g2 = dist.new_group([0, 2])
        if rank in (0, 2):
            z = torch.ones(1, device="cuda"); dist.all_reduce(z, group=g2); torch.cuda.synchronize()
            r["tp2_new_group_ms"] = ms(t)
        dist.barrier()
        CH = 256 << 20                      # elements per message (512 MiB)
        got = torch.empty_like(flat) if rank in (0, 2) else None
        t = time.perf_counter()
        for a in range(0, flat.numel(), CH):
            if rank in (0, 2):
                dist.recv(got[a:a + CH], src=rank + 1)
            else:
                dist.send(flat[a:a + CH], dst=rank - 1)
        torch.cuda.synchronize()
        r["transfer_s"] = round(ms(t) / 1e3, 3)
        r["transfer_gib"] = round(flat.numel() * 2 / 2**30, 2)
        r["transfer_gib_per_s"] = round(r["transfer_gib"] / r["transfer_s"], 1)
        if rank in (0, 2):
            # Assemble the TP2 halves: per tensor, own quarter then the received one.
            t = time.perf_counter()
            half = torch.empty(2 * flat.numel(), dtype=torch.bfloat16, device="cuda")
            src = dst = 0
            for n in sizes:
                if n:
                    half[dst:dst + n].copy_(flat[src:src + n])
                    half[dst + n:dst + 2 * n].copy_(got[src:src + n])
                    src += n; dst += 2 * n
            torch.cuda.synchronize()
            r["assemble_ms"] = ms(t)
            del got, flat; gc.collect(); torch.cuda.empty_cache()
            # Compare a sample of the halves with the checkpoint (TP2 piece rank // 2).
            from safetensors import safe_open
            sharded = [(i, it) for i, it in enumerate(items) if it[3]]
            offs, o = {}, 0
            for i, it in enumerate(items):
                if it[3]:
                    offs[i] = o; o += 2 * sizes[i]
            ok = True
            for i, (f, k, sh, _) in sharded[:: max(1, len(sharded) // 12)]:
                a, b = piece(sh, rank // 2, 2)
                with safe_open(os.path.join(model, f), "pt") as s:
                    want = s.get_slice(k)[a:b].reshape(-1).cuda()
                ok = ok and bool(torch.equal(half[offs[i]:offs[i] + 2 * sizes[i]], want))
            r["sample_matches_checkpoint"] = ok
            # The other way to get the missing quarter: read it from the checkpoint.
            t = time.perf_counter()
            _, _, _, nb = load(model, items, rank + 1, 4)
            r["reread_missing_quarter_s"] = round(ms(t) / 1e3, 2)
            r["reread_gib"] = round(nb / 2**30, 2)
        else:
            del flat, rep; gc.collect(); torch.cuda.empty_cache()
            r["freed_gib"] = round((torch.cuda.mem_get_info()[0] - free0) / 2**30, 2)
        dist.barrier()
        dist.destroy_process_group()
        q.put(r)
    except Exception as e:  # noqa: BLE001
        q.put(dict(rank=rank, error=repr(e)[:500]))


def launch(target, world, model, items, timeout=1800):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port, t = free_port(), time.time()
    ps = [ctx.Process(target=target, args=(r, world, port, model, items, t, q)) for r in range(world)]
    for p in ps:
        p.start()
    out, t_end = [], time.time() + timeout
    while len(out) < world and time.time() < t_end:
        try:
            out.append(q.get(timeout=5))
        except Exception:  # noqa: BLE001
            if not any(p.is_alive() for p in ps):
                break
    for p in ps:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    return dict(exitcodes=[p.exitcode for p in ps], wall_s=round(time.time() - t, 1),
                ranks=sorted(out, key=lambda r: r["rank"]))


def main():
    model = arg("--model")
    modes = [m for m in ("restart", "inproc") if m in sys.argv] or ["restart", "inproc"]
    items = plan(model)
    files = sorted({f for f, *_ in items})
    total = sum(numel(sh) * 2 for _, _, sh, _ in items)
    print("RESULT " + json.dumps(dict(mode="plan", tensors=len(items), gib=round(total / 2**30, 2),
                                      sharded=sum(1 for *_, s in items if s))), flush=True)
    for m in modes:
        before = resident_fraction(model, files)
        res = launch(restart_rank if m == "restart" else inproc_rank, 2 if m == "restart" else 4, model, items)
        print("RESULT " + json.dumps(dict(mode=m, page_cache_resident_before=before, **res)), flush=True)


if __name__ == "__main__":
    main()
