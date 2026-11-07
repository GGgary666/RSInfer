# Probe 9: graph recapture at engine scale, switching versions while the GPU is busy, and
# refitting a middle layer.
#   recapture  A decoder of Qwen3.8-27B's size (64 layers: 48 GatedDeltaNet + 16 full attention,
#              hidden 5120, MLP 17408, vocab 248320, random weights, about 50 GB), with decode
#              graphs for sglang's default batch-size list. Measures:
#                - capturing the whole list in one go;
#                - serving asynchronously (the host enqueues ahead of the GPU, the next input
#                  token is fed on the device) while a new backend version is recaptured one
#                  batch size per step boundary, and the stall each slice adds;
#                - the switch to the new version.
#              Tokens are compared with an eager run of the same schedule.
#   refit      A small model of the same shape. Middle-layer weights are refit in place while
#              requests are in flight. Compared with a fresh run that processes the same token
#              history with the new weights: continuing on the old KV/GDN state, and dropping
#              that state and recomputing it from the token history.
# Needs fla-core and flashinfer. usage: python p9_recapture.py recapture|refit
import gc, importlib, json, os, sys, time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

DEV = "cuda"
H, HV, K, V, KC = 16, 48, 128, 128, 4
NQ, NKV, D, PAGE = 24, 4, 256, 16
C = 2 * H * K + HV * V
BS_LIST = [1, 2, 4, 8, 12] + list(range(16, 257, 8))      # sglang's default decode batch sizes
SCRATCH_PAGE = 0


@dataclass
class Cfg:
    layers: list
    dm: int
    ffn: int
    vocab: int
    serve_bs: int
    slots: int
    max_pages: int
    bs_list: list = field(default_factory=list)

    @property
    def scratch_slot(self):
        return self.slots - 1


BIG = Cfg(layers=(["linear"] * 3 + ["full"]) * 16, dm=5120, ffn=17408, vocab=248320, serve_bs=64,
          slots=66, max_pages=2048, bs_list=BS_LIST)
SMALL = Cfg(layers=(["linear"] * 3 + ["full"]) * 2, dm=1024, ffn=2048, vocab=4096, serve_bs=16,
            slots=18, max_pages=512, bs_list=[16])

BACKEND = '''
import torch, torch.nn.functional as F
import flashinfer
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
VERSION = {version}


def pin(xs, dtype):
    return torch.tensor(xs, dtype=dtype).pin_memory()


class AttnDecode:
    """sglang-shaped: one cuda-graph wrapper per batch size over shared buffers; per-step
    metadata is written out of the graph, in stream order, without blocking the host."""

    def __init__(self, max_bs, max_pages, nq, nkv, d, page):
        self.nq, self.nkv, self.d, self.page = nq, nkv, d, page
        self.ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
        self.indptr = torch.zeros(max_bs + 1, dtype=torch.int32, device="cuda")
        self.indices = torch.zeros(max_pages, dtype=torch.int32, device="cuda")
        self.indices_dev = torch.zeros(max_pages, dtype=torch.int32, device="cuda")
        self.last = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        self.kv_loc = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        self.w, self.cur = {{}}, None

    def wrapper(self, bs):
        if bs not in self.w:
            self.w[bs] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.ws, "NHD", use_cuda_graph=True, use_tensor_cores=True,
                paged_kv_indptr_buffer=self.indptr[: bs + 1], paged_kv_indices_buffer=self.indices,
                paged_kv_last_page_len_buffer=self.last[:bs])
        return self.w[bs]

    def out_graph(self, seq_lens, page_lists):
        indptr, idx, last, loc = [0], [], [], []
        for L, pages in zip(seq_lens, page_lists):
            idx += pages[: (L + self.page - 1) // self.page]
            indptr.append(len(idx))
            last.append((L - 1) % self.page + 1)
            loc.append(pages[(L - 1) // self.page] * self.page + (L - 1) % self.page)
        bs = len(seq_lens)
        self.kv_loc[:bs].copy_(pin(loc, torch.int64), non_blocking=True)
        dev_idx = self.indices_dev[: len(idx)]
        dev_idx.copy_(pin(idx, torch.int32), non_blocking=True)
        self.cur = self.wrapper(bs)
        self.cur.plan(pin(indptr, torch.int32), dev_idx, pin(last, torch.int32), self.nq, self.nkv,
                      self.d, self.page, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
                      non_blocking=True)

    def forward(self, q, k, v, kc, vc):
        n = q.shape[0]
        loc = self.kv_loc[:n]
        kc.view(-1, self.nkv, self.d).index_copy_(0, loc, k.view(n, self.nkv, self.d))
        vc.view(-1, self.nkv, self.d).index_copy_(0, loc, v.view(n, self.nkv, self.d))
        return self.cur.run(q.view(n, self.nq, self.d), (kc, vc))


class GdnDecode:
    def __init__(self, max_bs):
        self.slot_idx = torch.zeros(max_bs, dtype=torch.int64, device="cuda")

    def out_graph(self, slots):
        self.slot_idx[: len(slots)].copy_(pin(slots, torch.int64), non_blocking=True)

    def forward(self, qkv, a, b, conv_w, conv_state, ssm, H, HV, K, V):
        n = qkv.shape[0]
        idx = self.slot_idx[:n]
        win = torch.cat([conv_state[idx], qkv[..., None]], dim=-1)
        conv_state.index_copy_(0, idx, win[..., 1:].contiguous())
        y = F.silu((win.float() * conv_w).sum(-1)).to(qkv.dtype)
        q, k, v = y.split([H * K, H * K, HV * V], dim=-1)
        o, s1 = fused_recurrent_gated_delta_rule(
            q.reshape(n, 1, H, K).contiguous(), k.reshape(n, 1, H, K).contiguous(),
            v.reshape(n, 1, HV, V).contiguous(),
            g=-F.softplus(a.float()).view(n, 1, HV), beta=torch.sigmoid(b.float()).view(n, 1, HV),
            initial_state=ssm[idx], output_final_state=True, use_qk_l2norm_in_kernel=True)
        ssm.index_copy_(0, idx, s1)
        return o.reshape(n, HV * V)
'''
UNIT_DIR = f"/tmp/p9_units_{os.getpid()}"


def load_backend(version):
    d = f"{UNIT_DIR}/v{version}"            # one directory per version, never rewritten
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/p9_backend.py", "w") as f:
        f.write(BACKEND.format(version=version))
    sys.path[:] = [p for p in sys.path if not p.startswith(UNIT_DIR)]
    sys.path.insert(0, d)
    sys.modules.pop("p9_backend", None)
    importlib.invalidate_caches()
    return importlib.import_module("p9_backend")


def new_backend(mod, cfg):
    return mod.AttnDecode(max(cfg.bs_list), cfg.max_pages, NQ, NKV, D, PAGE), mod.GdnDecode(max(cfg.bs_list))


def ms(t):
    return round((time.perf_counter() - t) * 1e3, 2)


def free_gib():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0] / 2**30


def make_weights(cfg, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def r(*shape, scale):
        t = torch.empty(*shape, device=DEV, dtype=torch.bfloat16)
        return t.normal_(0.0, scale, generator=g)
    w = {"emb": r(cfg.vocab, cfg.dm, scale=1.0), "lm": r(cfg.dm, cfg.vocab, scale=cfg.dm ** -0.5)}
    for i, kind in enumerate(cfg.layers):
        if kind == "linear":
            w[f"{i}.in"] = r(cfg.dm, C + 2 * HV, scale=cfg.dm ** -0.5)
            w[f"{i}.conv"] = torch.empty(C, KC, device=DEV).normal_(0.0, 0.5, generator=g)
            w[f"{i}.out"] = r(HV * V, cfg.dm, scale=0.5 * (HV * V) ** -0.5)
        else:
            w[f"{i}.qkv"] = r(cfg.dm, (NQ + 2 * NKV) * D, scale=cfg.dm ** -0.5)
            w[f"{i}.o"] = r(NQ * D, cfg.dm, scale=0.5 * (NQ * D) ** -0.5)
        w[f"{i}.gu"] = r(cfg.dm, 2 * cfg.ffn, scale=cfg.dm ** -0.5)
        w[f"{i}.down"] = r(cfg.ffn, cfg.dm, scale=0.5 * cfg.ffn ** -0.5)
    return w


class State:
    """kv.paged and gdn.slots; the last slot and page 0 are scratch, for capture warm-up."""

    def __init__(self, cfg):
        n_full, n_lin = cfg.layers.count("full"), cfg.layers.count("linear")
        mk = lambda: torch.zeros(cfg.max_pages, PAGE, NKV, D, device=DEV, dtype=torch.bfloat16)
        self.k = [mk() for _ in range(n_full)]
        self.v = [mk() for _ in range(n_full)]
        self.conv = [torch.zeros(cfg.slots, C, KC - 1, device=DEV, dtype=torch.bfloat16) for _ in range(n_lin)]
        self.ssm = [torch.zeros(cfg.slots, HV, K, V, device=DEV) for _ in range(n_lin)]
        self.free_pages = list(range(1, cfg.max_pages))
        self.free_slots = list(range(cfg.slots - 1))

    def reset_slot(self, s):
        for c, m in zip(self.conv, self.ssm):
            c[s].zero_(); m[s].zero_()


def rms(x):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


def forward(tok, w, st, be, cfg):
    attn, lin = be
    x = w["emb"][tok]
    n = x.shape[0]
    gi = ai = 0
    for i, kind in enumerate(cfg.layers):
        h = rms(x)
        if kind == "linear":
            qkv, a, b = (h @ w[f"{i}.in"]).split([C, HV, HV], dim=-1)
            o = lin.forward(qkv, a, b, w[f"{i}.conv"], st.conv[gi], st.ssm[gi], H, HV, K, V)
            x = x + o @ w[f"{i}.out"]; gi += 1
        else:
            q, k, v = (h @ w[f"{i}.qkv"]).split([NQ * D, NKV * D, NKV * D], dim=-1)
            o = attn.forward(q.contiguous(), k.contiguous(), v.contiguous(), st.k[ai], st.v[ai])
            x = x + o.reshape(n, NQ * D) @ w[f"{i}.o"]; ai += 1
        g, u = (rms(x) @ w[f"{i}.gu"]).split([cfg.ffn, cfg.ffn], dim=-1)
        x = x + (F.silu(g) * u) @ w[f"{i}.down"]
    logits = (rms(x) @ w["lm"]).float()
    return logits, logits.argmax(-1)


PHASES = []                                # per capture: where the time went


def capture(bs, tok, w, st, be, cfg, s, pool):
    """Capture one batch size with the warm-up writes going to scratch state."""
    ph, t = dict(bs=bs), time.perf_counter()
    be[0].out_graph([1] * bs, [[SCRATCH_PAGE]] * bs)
    be[1].out_graph([cfg.scratch_slot] * bs)
    ph["prepare_ms"] = ms(t); t = time.perf_counter()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            forward(tok[:bs], w, st, be, cfg)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    ph["warmup_ms"] = ms(t)
    # What torch.cuda.graph's __enter__ does, timed one by one (its own calls are then cheap).
    t = time.perf_counter(); torch.cuda.empty_cache(); ph["empty_cache_ms"] = ms(t)
    t = time.perf_counter(); torch._C._host_emptyCache(); ph["host_empty_cache_ms"] = ms(t)
    g = torch.cuda.CUDAGraph()
    t = time.perf_counter()
    with torch.cuda.graph(g, pool=pool, stream=s):
        out = forward(tok[:bs], w, st, be, cfg)
    ph["record_ms"] = ms(t)
    PHASES.append(ph)
    return g, out


class Engine:
    """One model, one schedule of serve_bs requests that all stay in flight."""

    def __init__(self, cfg, w, mode, queue_depth):
        self.cfg, self.w, self.mode, self.q = cfg, w, mode, queue_depth
        self.st = State(cfg)
        self.tok = torch.zeros(max(cfg.bs_list), dtype=torch.int64, device=DEV)
        self.s = torch.cuda.Stream()
        self.versions = {}                  # version -> dict(be=..., graphs={bs: (g, out)}, pool=...)
        self.cur = None
        self.reqs = []
        self.pending, self.tokens, self.logits = [], [], []
        self.ends = []                      # completion event of every step
        self.step_no = 0

    def add_version(self, v):
        mod = load_backend(v)
        self.versions[v] = dict(be=new_backend(mod, self.cfg), graphs={},
                                pool=torch.cuda.graph_pool_handle(), mod=mod)
        if self.cur is None:
            self.cur = v

    def capture_bs(self, v, bs):
        ver = self.versions[v]
        t = time.perf_counter()
        ver["graphs"][bs] = capture(bs, self.tok, self.w, self.st, ver["be"], self.cfg, self.s, ver["pool"])
        return ms(t)

    def start(self, first_tokens):
        bs = self.cfg.serve_bs
        self.reqs = [dict(L=1, pages=[], slot=self.st.free_slots.pop(0)) for _ in range(bs)]
        for r in self.reqs:
            self.st.reset_slot(r["slot"])
        self.tok[:bs].copy_(first_tokens)

    def step(self, feed=None, keep_logits=False):
        bs = self.cfg.serve_bs
        for r in self.reqs:
            if (r["L"] - 1) // PAGE >= len(r["pages"]):
                r["pages"].append(self.st.free_pages.pop(0))
        if feed is not None:                # teacher forcing: the input token comes from a history
            self.tok[:bs].copy_(feed, non_blocking=True)
        ver = self.versions[self.cur]
        attn, lin = ver["be"]
        attn.out_graph([r["L"] for r in self.reqs], [r["pages"] for r in self.reqs])
        lin.out_graph([r["slot"] for r in self.reqs])
        if self.mode == "graph":
            g, (logits, nxt) = ver["graphs"][bs]
            g.replay()
        else:
            logits, nxt = forward(self.tok[:bs], self.w, self.st, ver["be"], self.cfg)
        self.tok[:bs].copy_(nxt)            # next input stays on the device
        host = torch.empty(bs, dtype=torch.int64, pin_memory=True)
        host.copy_(nxt, non_blocking=True)
        lg = logits.clone() if keep_logits else None
        ev = torch.cuda.Event(enable_timing=True); ev.record()
        self.ends.append(ev)
        self.pending.append((host, ev, lg))
        while len(self.pending) > self.q:
            self.retire()
        for r in self.reqs:
            r["L"] += 1
        self.step_no += 1

    def retire(self):
        host, ev, lg = self.pending.pop(0)
        ev.synchronize()
        self.tokens.append(host.clone())
        if lg is not None:
            self.logits.append(lg.cpu())

    def drain(self):
        while self.pending:
            self.retire()

    def gaps_ms(self, a, b):
        """Time between consecutive step completions, steps a..b-1 -> a+1..b."""
        return [self.ends[i].elapsed_time(self.ends[i + 1]) for i in range(a, b)]


def recapture_mode():
    cfg = BIG
    t = time.perf_counter(); w = make_weights(cfg, 0); torch.cuda.synchronize(); init_s = ms(t) / 1e3
    gen = torch.Generator().manual_seed(7)
    first = torch.randint(0, cfg.vocab, (cfg.serve_bs,), generator=gen).to(DEV)
    S1, S2 = 30, 20                         # steps on v1 before the slices, steps on v2 after
    n_slices = len(cfg.bs_list)
    total = S1 + n_slices + S2
    res = dict(weights_gib=round(sum(x.numel() * x.element_size() for x in w.values()) / 2**30, 1),
               weights_init_s=round(init_s, 1), bs_list=cfg.bs_list)

    # Eager reference of the same schedule (the backend version changes at the same step).
    eng = Engine(cfg, w, "eager", queue_depth=0)
    eng.add_version(1); eng.add_version(2); eng.start(first)
    t = time.perf_counter()
    for i in range(total):
        if i == S1 + n_slices:
            eng.cur = 2
        eng.step()
    eng.drain()
    res["eager_step_ms_median"] = sorted(eng.gaps_ms(1, total - 1))[(total - 2) // 2]
    ref_tokens = torch.stack(eng.tokens)
    res["distinct_tokens_in_reference"] = int(ref_tokens.unique().numel())
    del eng; gc.collect(); torch.cuda.empty_cache()

    eng = Engine(cfg, w, "graph", queue_depth=2)
    eng.add_version(1)
    f0 = free_gib()
    t = time.perf_counter()
    per_bs = {bs: eng.capture_bs(1, bs) for bs in sorted(cfg.bs_list, reverse=True)}
    res["capture_all_v1_s"] = round(ms(t) / 1e3, 2)
    res["capture_per_bs_ms"] = dict(min=min(per_bs.values()), median=sorted(per_bs.values())[len(per_bs) // 2],
                                    max=max(per_bs.values()), bs1=per_bs[1], bs256=per_bs[256])
    res["graph_pool_v1_gib"] = round(f0 - free_gib(), 2)
    eng.start(first)
    for i in range(S1):
        eng.step()

    # New backend code: recapture one batch size per step boundary while v1 keeps serving.
    t = time.perf_counter(); eng.add_version(2); res["load_v2_ms"] = ms(t)
    slices = []
    for bs in sorted(cfg.bs_list, reverse=True):
        t = time.perf_counter()
        torch.cuda.current_stream().synchronize()      # no other CUDA work while capturing
        wait = ms(t)
        cap = eng.capture_bs(2, bs)
        slices.append(dict(bs=bs, wait_inflight_ms=wait, capture_ms=cap))
        eng.step()
    res["graph_pool_v1_plus_v2_gib"] = round(f0 - free_gib(), 2)
    # Switch: the next step simply uses v2's graphs.
    sw = len(eng.ends)
    eng.cur = 2
    for i in range(S2):
        eng.step()
    eng.drain()
    del eng.versions[1]; gc.collect(); torch.cuda.empty_cache()
    res["graph_pool_after_dropping_v1_gib"] = round(f0 - free_gib(), 2)

    base = eng.gaps_ms(1, S1 - 1)
    sliced = eng.gaps_ms(S1 - 1, S1 - 1 + n_slices)
    after = eng.gaps_ms(sw, sw + S2 - 1)
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 2)
    res["step_gap_ms"] = dict(v1_median=med(base), during_slices_median=med(sliced),
                              during_slices_max=round(max(sliced), 2), switch_gap=round(eng.gaps_ms(sw - 1, sw)[0], 2),
                              v2_median=med(after))
    res["slices"] = dict(n=len(slices), wait_inflight_ms_median=med([x["wait_inflight_ms"] for x in slices]),
                         capture_ms_median=med([x["capture_ms"] for x in slices]),
                         capture_ms_max=max(x["capture_ms"] for x in slices),
                         total_capture_s=round(sum(x["capture_ms"] for x in slices) / 1e3, 2))
    res["slice_detail"] = slices
    n_start = len(cfg.bs_list)
    phase = lambda rows, k: dict(median=med([p[k] for p in rows]), max=round(max(p[k] for p in rows), 2))
    keys = ("prepare_ms", "warmup_ms", "empty_cache_ms", "host_empty_cache_ms", "record_ms")
    res["phases_at_startup"] = {k: phase(PHASES[:n_start], k) for k in keys}
    res["phases_in_slices"] = {k: phase(PHASES[n_start:], k) for k in keys}
    got = torch.stack(eng.tokens)
    diff = (got != ref_tokens).any(dim=1).nonzero()
    res["tokens_equal_to_eager"] = bool(diff.numel() == 0)
    res["first_divergent_step"] = int(diff[0]) if diff.numel() else None
    res["steps"] = total
    return res


def refit_mode():
    # Every request has emitted tokens hist[1..S1] by step S1 (hist[0] is its first token). After
    # the refit, the right continuation is the new model reading hist[0..S1] from scratch.
    cfg = SMALL
    gen = torch.Generator().manual_seed(7)
    first = torch.randint(0, cfg.vocab, (cfg.serve_bs,), generator=gen).to(DEV)
    S1, S2 = 40, 40
    refit_layers = {"3", "4"}               # the first full-attention layer and the GDN layer after it
    rec, hist = {}, []

    def refit(w):
        w2 = make_weights(cfg, 1)
        for k in w:
            if k.split(".")[0] in refit_layers:
                w[k].copy_(w2[k])           # in place: same addresses, the graph is not recaptured

    def run(kind):
        w = make_weights(cfg, 0)
        eng = Engine(cfg, w, "graph", queue_depth=0)
        eng.add_version(1)
        eng.capture_bs(1, cfg.serve_bs)
        eng.start(first)
        if kind == "fresh":                 # new weights from the start, history teacher-forced
            refit(w)
            for i in range(S1):
                eng.step(feed=hist[i], keep_logits=True)
            eng.step(feed=hist[S1], keep_logits=True)
        else:
            for i in range(S1):
                eng.step(keep_logits=True)
            refit(w)
            if kind == "recompute":         # drop the state the old weights built, rebuild it
                eng.drain()
                t = time.perf_counter()
                for r in eng.reqs:
                    eng.st.reset_slot(r["slot"]); r["L"] = 1
                for i in range(S1):
                    eng.step(feed=hist[i])
                eng.drain(); torch.cuda.synchronize()
                rec["recompute_ms"] = ms(t)
                eng.tokens = eng.tokens[:S1]
                eng.step(feed=hist[S1], keep_logits=True)
            else:                           # stale: carry on with the old KV and GDN state
                eng.step(keep_logits=True)
        for i in range(S2 - 1):
            eng.step(keep_logits=True)
        eng.drain()
        return torch.stack(eng.tokens), eng.logits

    stale_tok, stale_lg = run("stale")
    hist = [first] + [stale_tok[i].to(DEV) for i in range(S1)]
    fresh_tok, fresh_lg = run("fresh")
    recomp_tok, recomp_lg = run("recompute")
    eq = lambda a, b: bool(torch.equal(a, b))
    return dict(
        refit_layers=sorted(refit_layers), steps_before=S1, steps_after=S2,
        stale_tokens_after_refit_equal_fresh=eq(stale_tok[S1:], fresh_tok[S1:]),
        stale_first_step_max_logit_diff=float((stale_lg[S1] - fresh_lg[S1]).abs().max()),
        stale_steps_with_different_tokens=int((stale_tok[S1:] != fresh_tok[S1:]).any(dim=1).sum()),
        recompute_tokens_after_refit_equal_fresh=eq(recomp_tok[S1:], fresh_tok[S1:]),
        recompute_logits_bitwise_equal_fresh=all(eq(a, b) for a, b in zip(recomp_lg[S1:], fresh_lg[S1:])),
        before_refit_stale_equals_recompute=eq(stale_tok[:S1], recomp_tok[:S1]),
        **rec)


if __name__ == "__main__":
    fn = dict(recapture=recapture_mode, refit=refit_mode)[sys.argv[1]]
    print("RESULT " + json.dumps(fn()), flush=True)
