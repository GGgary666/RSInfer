# Probe 4: grow a KV pool by mapping more physical memory into a reserved virtual range
# (CUDA VMM: cuMemAddressReserve / cuMemCreate / cuMemMap), and check what a graph
# captured before the grow still sees. Needs cuda-python (cuda.bindings).
#
# usage: python p4_vmm_grow.py
import gc, json, time, torch, triton, triton.language as tl
from cuda.bindings import driver as cu


def ok(r):
    if r[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(repr(r[0]))
    return None if len(r) == 1 else (r[1] if len(r) == 2 else r[1:])


torch.zeros(1, device="cuda"); torch.cuda.synchronize()   # primary context is now current
DEV = torch.cuda.current_device()
PROP = cu.CUmemAllocationProp()
PROP.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
PROP.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
PROP.location.id = DEV
GRAN = ok(cu.cuMemGetAllocationGranularity(
    PROP, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))
ACCESS = cu.CUmemAccessDesc()
ACCESS.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
ACCESS.location.id = DEV
ACCESS.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
CLOSED = []


class Range:
    """Reserved virtual addresses; physical chunks are mapped at increasing offsets."""

    def __init__(self, nbytes, tag=""):
        self.reserved, self.tag, self.chunks, self.mapped, self.closed = nbytes, tag, [], 0, False
        self.base = int(ok(cu.cuMemAddressReserve(nbytes, GRAN, 0, 0)))

    def grow(self, nbytes):
        h = ok(cu.cuMemCreate(nbytes, PROP, 0))
        ptr = cu.CUdeviceptr(self.base + self.mapped)
        ok(cu.cuMemMap(ptr, nbytes, 0, h, 0))
        ok(cu.cuMemSetAccess(ptr, nbytes, [ACCESS], 1))
        self.chunks.append((self.mapped, nbytes, h)); self.mapped += nbytes

    def shrink_last(self):
        off, n, h = self.chunks.pop()
        ok(cu.cuMemUnmap(cu.CUdeviceptr(self.base + off), n)); ok(cu.cuMemRelease(h))
        self.mapped -= n

    def close(self):
        while self.chunks:
            self.shrink_last()
        ok(cu.cuMemAddressFree(cu.CUdeviceptr(self.base), self.reserved))
        self.closed = True; CLOSED.append(self.tag)

    def __del__(self):
        if not self.closed:
            self.close()


class _Iface:
    def __init__(self, owner, ptr, n):
        self.owner = owner   # the tensor keeps this object alive, and with it the Range
        self.__cuda_array_interface__ = dict(shape=(n,), typestr="<f4", data=(ptr, False), version=3)


def view(rng, n_floats, offset=0):
    return torch.as_tensor(_Iface(rng, rng.base + offset, n_floats), device="cuda")


@triton.jit
def _gather(pool, table, out, PAGE: tl.constexpr):
    i = tl.program_id(0)
    blk = tl.load(table + i).to(tl.int64)
    offs = tl.arange(0, PAGE)
    tl.store(out + i * PAGE + offs, tl.load(pool + blk * PAGE + offs) * 2.0)


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


def main():
    res = dict(granularity_kib=GRAN // 1024)
    s = torch.cuda.Stream()
    CHUNK, PAGE = 1 << 30, 4096                    # 1 GiB chunks, 16 KiB pages of fp32
    PPC = CHUNK // 4 // PAGE                        # pages per chunk
    rng = Range(64 << 30, "kv")
    free0 = torch.cuda.mem_get_info()[0]; res0 = torch.cuda.memory_reserved()
    t = time.perf_counter(); rng.grow(CHUNK); res["grow_1gib_idle_ms"] = round((time.perf_counter() - t) * 1e3, 2)
    res["free_drop_mib_after_1gib"] = round((free0 - torch.cuda.mem_get_info()[0]) / 2**20)
    res["torch_reserved_changed"] = torch.cuda.memory_reserved() != res0

    kv = view(rng, CHUNK // 4)
    kv.view(PPC, PAGE).copy_(torch.arange(PPC, device="cuda", dtype=torch.float32)[:, None].expand(PPC, PAGE))
    table = (torch.arange(64, device="cuda", dtype=torch.int32) * 7)
    out = torch.empty(64, PAGE, device="cuda")
    g_gather, _ = capture(lambda: _gather[(64,)](kv, table, out, PAGE=PAGE), s)
    g_sum, tot = capture(lambda: kv.sum(), s)
    g_gather.replay(); g_sum.replay(); torch.cuda.synchronize()
    res["before_grow_ok"] = bool(torch.equal(out[:, 0], table.float() * 2))
    sum0 = float(tot)

    t = time.perf_counter(); rng.grow(CHUNK); res["grow_1gib_second_ms"] = round((time.perf_counter() - t) * 1e3, 2)
    kv2 = view(rng, 2 * CHUNK // 4)
    res["base_address_unchanged"] = kv2.data_ptr() == kv.data_ptr()
    kv2.view(2 * PPC, PAGE)[PPC:].copy_(
        torch.arange(PPC, 2 * PPC, device="cuda", dtype=torch.float32)[:, None].expand(PPC, PAGE))
    table.copy_(PPC + torch.arange(64, device="cuda", dtype=torch.int32) * 13)
    g_gather.replay(); g_sum.replay(); torch.cuda.synchronize()
    res["graph_from_before_grow_reads_new_pages"] = bool(torch.equal(out[:, 0], table.float() * 2))
    res["graph_with_baked_view_shape_ignores_grown_region"] = float(tot) == sum0

    # Grow while another stream is busy: does mapping wait for in-flight work?
    big = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    g_busy, _ = capture(lambda: [big @ big for _ in range(20)][-1], s)
    with torch.cuda.stream(s):
        for _ in range(10):
            g_busy.replay()
        ev = torch.cuda.Event(); ev.record(s)
    t = time.perf_counter(); rng.grow(CHUNK); dt = (time.perf_counter() - t) * 1e3
    res["grow_1gib_under_load_ms"] = round(dt, 2)
    res["grow_returned_while_other_stream_busy"] = not ev.query()
    t = time.perf_counter(); ev.synchronize(); res["busy_work_left_after_grow_ms"] = round((time.perf_counter() - t) * 1e3, 1)
    t = time.perf_counter(); rng.grow(8 * CHUNK); res["grow_8gib_idle_ms"] = round((time.perf_counter() - t) * 1e3, 2)

    # Shrink the last chunk: physical memory goes back to the device.
    free_before = torch.cuda.mem_get_info()[0]
    t = time.perf_counter(); rng.shrink_last(); res["shrink_8gib_ms"] = round((time.perf_counter() - t) * 1e3, 2)
    res["shrink_freed_mib"] = round((torch.cuda.mem_get_info()[0] - free_before) / 2**20)
    g_gather.replay(); torch.cuda.synchronize()
    res["graph_ok_after_shrinking_unused_tail"] = bool(torch.equal(out[:, 0], table.float() * 2))

    # Lifetime: a tensor view keeps the Range (owner) alive; dropping the last view frees it.
    small = Range(1 << 30, "small"); small.grow(16 * GRAN)
    tv = view(small, 16 * GRAN // 4); tv.fill_(3.0)
    del small; gc.collect()
    res["owner_kept_alive_by_view"] = "small" not in CLOSED
    if "small" not in CLOSED:
        res["view_valid_after_owner_name_dropped"] = bool((tv == 3.0).all())
    del tv; gc.collect()
    res["owner_closed_after_last_view_dropped"] = "small" in CLOSED

    del g_gather, g_sum, g_busy, tot, kv, kv2; gc.collect()
    rng.close()
    res["capability"] = list(torch.cuda.get_device_capability(0))
    res["sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
