# Probe 6 (contract probe): a truncated Qwen3.8-shaped decoder (3 GatedDeltaNet layers + 1
# full-attention layer, twice) built from sglang-shaped pieces, run as units would be:
#   state units  kv.paged   paged K/V of the attention layers, CUDA VMM, grows in place
#                gdn.slots  one fixed slot per request: conv state + recurrent state
#                weights    every weight tensor
#   logic units  backend    attention backend (flashinfer, sglang's "metadata out of graph"
#                           split) and GDN backend (fla); its module is reloaded mid-run
#                plan       captures the decode graph, checks recorded deps before replay
# The same request schedule runs twice: eagerly, and with graph replay plus mid-run events
# (KV grow without recapture, in-place weight refit, backend code reload with recapture while
# requests are in flight). Tokens must match exactly.
#
# Needs fla-core, flashinfer-python, cuda-python; imports Range from p4_vmm_grow.py.
# usage: python p6_contract.py
import gc, importlib, json, os, random, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p4_vmm_grow import GRAN, Range  # noqa: E402

DEV = "cuda"
H, HV, K, V, KC = 16, 48, 128, 128, 4            # Qwen3.8-27B GatedDeltaNet (TP1)
NQ, NKV, D, PAGE = 24, 4, 256, 16                # Qwen3.8-27B full attention (TP1)
C = 2 * H * K + HV * V                           # conv channels (q, k, v)
DM, VOCAB, B, SLOTS, MAX_PAGES = 1024, 4096, 16, 33, 16384
LAYERS = ["linear", "linear", "linear", "full"] * 2
STEPS, REFIT_AT, RELOAD_AT = 160, 50, 90
SCRATCH_SLOT, SCRATCH_PAGE = SLOTS - 1, 0        # captured warm-up writes go here, not to live state
PAGE_BYTES = PAGE * NKV * D * 2

BACKEND = '''
import torch, torch.nn.functional as F
import flashinfer
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
VERSION = {version}


class AttnDecode:
    """sglang-shaped: per-step metadata is prepared out of the graph into fixed buffers."""

    def __init__(self, max_bs, max_pages, nq, nkv, d, page):
        self.nq, self.nkv, self.d, self.page = nq, nkv, d, page
        self.ws = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        self.indptr = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
        self.indices = torch.zeros(max_pages, dtype=torch.int32, device="cuda")
        self.last = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        self.kv_loc = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        self.w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.ws, "NHD", use_cuda_graph=True, use_tensor_cores=True,
            paged_kv_indptr_buffer=self.indptr, paged_kv_indices_buffer=self.indices,
            paged_kv_last_page_len_buffer=self.last)

    def buffers(self):
        return dict(ws=self.ws, indptr=self.indptr, indices=self.indices, last=self.last,
                    kv_loc=self.kv_loc)

    def out_graph(self, seq_lens, page_lists):
        indptr, idx, last, loc = [0], [], [], []
        for L, pages in zip(seq_lens, page_lists):
            idx += pages[: (L + self.page - 1) // self.page]
            indptr.append(len(idx))
            last.append((L - 1) % self.page + 1)
            loc.append(pages[(L - 1) // self.page] * self.page + (L - 1) % self.page)
        self.kv_loc.copy_(torch.tensor(loc, dtype=torch.int64))
        self.w.plan(torch.tensor(indptr, dtype=torch.int32), torch.tensor(idx, dtype=torch.int32),
                    torch.tensor(last, dtype=torch.int32), self.nq, self.nkv, self.d, self.page,
                    q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)

    def forward(self, q, k, v, kc, vc):
        n = q.shape[0]
        {kv_write}
        return self.w.run(q.view(n, self.nq, self.d), (kc, vc))


class GdnDecode:
    def __init__(self, max_bs):
        self.slot_idx = torch.zeros(max_bs, dtype=torch.int64, device="cuda")

    def buffers(self):
        return dict(slot_idx=self.slot_idx)

    def out_graph(self, slots):
        self.slot_idx.copy_(torch.tensor(slots, dtype=torch.int64))

    def forward(self, qkv, a, b, conv_w, conv_state, ssm, H, HV, K, V):
        n = qkv.shape[0]
        win = torch.cat([conv_state[self.slot_idx], qkv[..., None]], dim=-1)
        conv_state.index_copy_(0, self.slot_idx, win[..., 1:].contiguous())
        y = F.silu((win.float() * conv_w).sum(-1)).to(qkv.dtype)
        q, k, v = y.split([H * K, H * K, HV * V], dim=-1)
        o, s1 = fused_recurrent_gated_delta_rule(
            q.reshape(n, 1, H, K).contiguous(), k.reshape(n, 1, H, K).contiguous(),
            v.reshape(n, 1, HV, V).contiguous(),
            g=-F.softplus(a.float()).view(n, 1, HV), beta=torch.sigmoid(b.float()).view(n, 1, HV),
            initial_state=ssm[self.slot_idx], output_final_state=True, use_qk_l2norm_in_kernel=True)
        ssm.index_copy_(0, self.slot_idx, s1)
        return o.reshape(n, HV * V)
'''
# Version 2 writes K/V with index_put_ instead of index_copy_: different code, same values.
KV_WRITE = {
    1: "kc.view(-1, self.nkv, self.d).index_copy_(0, self.kv_loc, k.view(n, self.nkv, self.d))\n"
       "        vc.view(-1, self.nkv, self.d).index_copy_(0, self.kv_loc, v.view(n, self.nkv, self.d))",
    2: "kc.view(-1, self.nkv, self.d)[self.kv_loc] = k.view(n, self.nkv, self.d)\n"
       "        vc.view(-1, self.nkv, self.d)[self.kv_loc] = v.view(n, self.nkv, self.d)",
}
UNIT_DIR = f"/tmp/p6_units_{os.getpid()}"


def load_backend(version):
    # One directory per version, like the engine's per-identity snapshot directories.
    # Rewriting one file in place can import stale bytecode (see p7_stale_bytecode.py).
    d = f"{UNIT_DIR}/v{version}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/p6_backend.py", "w") as f:
        f.write(BACKEND.format(version=version, kv_write=KV_WRITE[version]))
    sys.path[:] = [p for p in sys.path if not p.startswith(UNIT_DIR)]
    sys.path.insert(0, d)
    sys.modules.pop("p6_backend", None)
    importlib.invalidate_caches()
    return importlib.import_module("p6_backend")


class _Iface16:
    def __init__(self, owner, ptr, n):
        self.owner = owner
        self.__cuda_array_interface__ = dict(shape=(n,), typestr="<i2", data=(ptr, False), version=3)


class KvPool:
    """State unit kv.paged: one VMM range per K/V buffer; views span the whole reservation."""

    def __init__(self, n_layers, init_pages):
        self.ranges, self.k, self.v, self.mapped, self.free, self.grows = [], [], [], 0, [], []
        for _ in range(n_layers):
            pair = (Range(MAX_PAGES * PAGE_BYTES), Range(MAX_PAGES * PAGE_BYTES))
            self.ranges.append(pair)
        self.grow(init_pages)
        for rk, rv in self.ranges:
            mk = lambda r: torch.as_tensor(_Iface16(r, r.base, MAX_PAGES * PAGE_BYTES // 2), device=DEV) \
                .view(torch.bfloat16).view(MAX_PAGES, PAGE, NKV, D)
            self.k.append(mk(rk)); self.v.append(mk(rv))
        self.free.remove(SCRATCH_PAGE)

    def grow(self, n_pages):
        per = GRAN // PAGE_BYTES
        n_pages = -(-n_pages // per) * per
        t = time.perf_counter()
        for pair in self.ranges:
            for r in pair:
                r.grow(n_pages * PAGE_BYTES)
        self.grows.append(round((time.perf_counter() - t) * 1e3, 2))
        self.free += range(self.mapped, self.mapped + n_pages)
        self.mapped += n_pages

    def alloc(self):
        if not self.free:
            self.grow(128)
        return self.free.pop(0)

    def close(self):
        self.k.clear(); self.v.clear(); gc.collect()
        for pair in self.ranges:
            for r in pair:
                r.close()


class GdnSlots:
    """State unit gdn.slots: per GDN layer, conv state and recurrent state, one slot per request."""

    def __init__(self, n_layers):
        self.conv = [torch.zeros(SLOTS, C, KC - 1, device=DEV, dtype=torch.bfloat16) for _ in range(n_layers)]
        self.ssm = [torch.zeros(SLOTS, HV, K, V, device=DEV) for _ in range(n_layers)]
        self.free = [s for s in range(SLOTS) if s != SCRATCH_SLOT]

    def alloc(self):
        s = self.free.pop(0)
        for c, m in zip(self.conv, self.ssm):
            c[s].zero_(); m[s].zero_()
        return s


class Weights:
    def __init__(self):
        g = torch.Generator(device=DEV).manual_seed(1234)
        r = lambda *shape, scale: (torch.randn(*shape, device=DEV, generator=g) * scale).to(torch.bfloat16)
        self.t = {"emb": r(VOCAB, DM, scale=1.0), "lm": r(DM, VOCAB, scale=DM ** -0.5),
                  "lm_alt": r(DM, VOCAB, scale=DM ** -0.5)}
        for i, kind in enumerate(LAYERS):
            if kind == "linear":
                self.t[f"{i}.in"] = r(DM, C + 2 * HV, scale=DM ** -0.5)
                self.t[f"{i}.conv"] = (torch.randn(C, KC, device=DEV, generator=g) * 0.5)
                self.t[f"{i}.out"] = r(HV * V, DM, scale=0.5 * (HV * V) ** -0.5)
            else:
                self.t[f"{i}.q"] = r(DM, NQ * D, scale=DM ** -0.5)
                self.t[f"{i}.k"] = r(DM, NKV * D, scale=DM ** -0.5)
                self.t[f"{i}.v"] = r(DM, NKV * D, scale=DM ** -0.5)
                self.t[f"{i}.o"] = r(NQ * D, DM, scale=0.5 * (NQ * D) ** -0.5)


def rms(x):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


def forward(tok, w, kv, gdn, be):
    attn, lin = be
    x = w.t["emb"][tok]
    gi = ai = 0
    for i, kind in enumerate(LAYERS):
        h = rms(x)
        if kind == "linear":
            qkv, a, b = (h @ w.t[f"{i}.in"]).split([C, HV, HV], dim=-1)
            o = lin.forward(qkv, a, b, w.t[f"{i}.conv"], gdn.conv[gi], gdn.ssm[gi], H, HV, K, V)
            x = x + o @ w.t[f"{i}.out"]; gi += 1
        else:
            o = attn.forward(h @ w.t[f"{i}.q"], h @ w.t[f"{i}.k"], h @ w.t[f"{i}.v"], kv.k[ai], kv.v[ai])
            x = x + o.reshape(x.shape[0], NQ * D) @ w.t[f"{i}.o"]; ai += 1
    logits = (rms(x) @ w.t["lm"]).float()
    return logits, logits.argmax(-1)


class Plan:
    """Plan unit: records by-address and by-value deps at capture, checks them before replay."""

    def __init__(self, deps_addr, deps_value):
        self.deps_addr, self.deps_value = deps_addr, deps_value
        self.addr = {n: self._sig(f()) for n, f in deps_addr.items()}
        self.value = {n: f() for n, f in deps_value.items()}

    @staticmethod
    def _sig(t):
        return t.data_ptr(), tuple(t.shape), t.dtype

    def stale(self):
        bad = [n for n, f in self.deps_addr.items() if self._sig(f()) != self.addr[n]]
        return bad + [n for n, f in self.deps_value.items() if f() != self.value[n]]


def run(mode):
    torch.manual_seed(0)
    w = Weights()
    kv = KvPool(LAYERS.count("full"), init_pages=64)
    gdn = GdnSlots(LAYERS.count("linear"))
    mod = load_backend(1)
    st = dict(be=(mod.AttnDecode(B, MAX_PAGES, NQ, NKV, D, PAGE), mod.GdnDecode(B)), mod=mod)
    rnd = random.Random(7)
    new_req = lambda: dict(target=rnd.randint(8, 200), toks=[rnd.randrange(VOCAB)], L=1, pages=[],
                           slot=gdn.alloc())
    reqs = [new_req() for _ in range(B)]
    tok = torch.zeros(B, dtype=torch.int64, device=DEV)
    s = torch.cuda.Stream()
    out, rec = dict(tokens=[], logits=[]), dict(captures=[], stale_seen=[], guard_us=[], step_ms=[])
    graph = plan = None

    def prepare(live):
        a, l = st["be"]
        if live:
            a.out_graph([r["L"] for r in reqs], [r["pages"] for r in reqs]); l.out_graph([r["slot"] for r in reqs])
        else:
            a.out_graph([1] * B, [[SCRATCH_PAGE]] * B); l.out_graph([SCRATCH_SLOT] * B)

    def capture():
        prepare(live=False)
        t = time.perf_counter()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                forward(tok, w, kv, gdn, st["be"])
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            o = forward(tok, w, kv, gdn, st["be"])
        rec["captures"].append(round((time.perf_counter() - t) * 1e3, 1))
        addr = {f"w.{n}": (lambda n=n: w.t[n]) for n in w.t}
        addr |= {f"kv.k{i}": (lambda i=i: kv.k[i]) for i in range(len(kv.k))}
        addr |= {f"kv.v{i}": (lambda i=i: kv.v[i]) for i in range(len(kv.v))}
        addr |= {f"gdn.conv{i}": (lambda i=i: gdn.conv[i]) for i in range(len(gdn.conv))}
        addr |= {f"gdn.ssm{i}": (lambda i=i: gdn.ssm[i]) for i in range(len(gdn.ssm))}
        addr |= {f"attn.{n}": (lambda n=n: st["be"][0].buffers()[n]) for n in st["be"][0].buffers()}
        addr |= {f"gdn_be.{n}": (lambda n=n: st["be"][1].buffers()[n]) for n in st["be"][1].buffers()}
        addr["tok"] = lambda: tok
        value = {"backend.version": lambda: st["mod"].VERSION, "backend.instance": lambda: id(st["be"][0]),
                 "bs": lambda: B}
        rec["n_deps"] = dict(by_address=len(addr), by_value=len(value))
        return g, o, Plan(addr, value)

    for step in range(STEPS):
        t0 = time.perf_counter()
        for r in reqs:
            if (r["L"] - 1) // PAGE >= len(r["pages"]):
                r["pages"].append(kv.alloc())
        tok.copy_(torch.tensor([r["toks"][-1] for r in reqs]))
        if step == REFIT_AT:
            w.t["lm"].copy_(w.t["lm_alt"])            # refit in place: no recapture
        if mode == "graph":
            if graph is None:
                graph, gout, plan = capture()
            prepare(live=True)
            t = time.perf_counter(); bad = plan.stale(); rec["guard_us"].append((time.perf_counter() - t) * 1e6)
            assert not bad, bad
            graph.replay()
            logits, nxt = gout
        else:
            prepare(live=True)
            logits, nxt = forward(tok, w, kv, gdn, st["be"])
        out["tokens"].append(nxt.tolist()); out["logits"].append(logits.cpu())
        rec["step_ms"].append((time.perf_counter() - t0) * 1e3)
        for r, t in zip(reqs, out["tokens"][-1]):
            r["toks"].append(t); r["L"] += 1
        for i, r in enumerate(reqs):
            if len(r["toks"]) >= r["target"]:
                kv.free += r["pages"]; gdn.free.append(r["slot"])
                reqs[i] = new_req()
        if step == RELOAD_AT:
            # Backend code changes while all B requests are in flight: state units stay,
            # the plan unit recaptures. No drain.
            rec["in_flight_at_reload"] = sum(1 for r in reqs if r["L"] > 1)
            t = time.perf_counter()
            mod = load_backend(2)
            rec["version_after_reload"] = mod.VERSION
            st.update(be=(mod.AttnDecode(B, MAX_PAGES, NQ, NKV, D, PAGE), mod.GdnDecode(B)), mod=mod)
            if mode == "graph":
                rec["stale_seen"] = plan.stale()
                del graph, gout, plan
                graph = None
                graph, gout, plan = capture()
            rec["reload_ms"] = round((time.perf_counter() - t) * 1e3, 1)
    rec["grows_ms"] = kv.grows
    rec["mapped_pages"] = kv.mapped
    if mode == "graph":
        kv.k[0] = torch.empty_like(kv.k[0])           # a "grow" that reallocates instead of mapping
        rec["stale_after_reallocating_kv"] = plan.stale()
        del graph, gout, plan
    del st
    kv.close()
    return out, rec


def main():
    e_out, e_rec = run("eager")
    g_out, g_rec = run("graph")
    first_diff = next((i for i, (a, b) in enumerate(zip(e_out["tokens"], g_out["tokens"])) if a != b), None)
    upto = first_diff if first_diff is not None else STEPS
    max_logit_diff = max((float((a - b).abs().max()) for a, b in zip(e_out["logits"][:upto], g_out["logits"][:upto])),
                         default=None)
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 3)
    res = dict(
        steps=STEPS, tokens_equal=first_diff is None, first_token_divergence_step=first_diff,
        max_abs_logit_diff_before_divergence=max_logit_diff,
        logits_bitwise_equal=all(torch.equal(a, b) for a, b in zip(e_out["logits"], g_out["logits"])),
        kv_grows=len(g_rec["grows_ms"]), kv_grow_ms=g_rec["grows_ms"], mapped_pages=g_rec["mapped_pages"],
        in_flight_at_reload=g_rec["in_flight_at_reload"], version_after_reload=g_rec["version_after_reload"],
        reload_and_recapture_ms=g_rec["reload_ms"],
        capture_ms=g_rec["captures"], stale_deps_of_old_graph_after_reload=g_rec["stale_seen"],
        stale_after_reallocating_kv=g_rec["stale_after_reallocating_kv"],
        guarded_deps=g_rec["n_deps"], guard_us_median=med(g_rec["guard_us"]),
        graph_step_ms_median=med(g_rec["step_ms"][1:]), eager_step_ms_median=med(e_rec["step_ms"][1:]))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
