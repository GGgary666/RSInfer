#!/bin/bash
# What a hot-apply session (sm100_session.sh) runs after each apply, unless the commit names another
# check. On the engine as it is, and then with each of the settings below hot-applied to it: a few
# greedy completions through the HTTP API (their texts' digest), the kernels the worker runs, and the
# decode throughput of 1 and of 8 requests at a time. The last setting is the first again, so its
# texts must be the first's. SESSION_URL, SESSION_RUN and SESSION_PYTHON come from the session.
set -u
PY=${SESSION_PYTHON:-python3}
SETTINGS=${SESSION_SETTINGS:-'{"dsv4.backend":"auto","plan.cuda_graph":"auto"} {"plan.cuda_graph":"off"} {"dsv4.backend":"tilelang"} {"plan.cuda_graph":"auto"} {"dsv4.backend":"auto"}'}

measure() {
  "$PY" - "$SESSION_URL" <<'PYEOF'
import hashlib, json, sys, threading, time, urllib.request

url = sys.argv[1]


def complete(prompt: str, tokens: int, ignore_eos: bool = False) -> dict:
    body = json.dumps({"model": "DeepSeek-V4.1-Flash", "prompt": prompt, "max_tokens": tokens, "temperature": 0,
                       "ignore_eos": ignore_eos}).encode()
    request = urllib.request.Request(url + "/v1/completions", body, {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=900))


texts = [complete(p, 16)["choices"][0]["text"] for p in ("The quick brown fox jumps over the lazy dog.", "Hello",
                                                         "写一首关于秋天的诗")]
print("texts digest:", hashlib.md5("\n".join(texts).encode()).hexdigest()[:12])
for streams in (1, 8):
    done = [0] * streams

    def one(i: int) -> None:
        reply = complete("The engine keeps a sliding window of the last tokens.", 64, True)
        done[i] = reply.get("usage", {}).get("completion_tokens", 64)

    complete("warm", 8, True)
    threads = [threading.Thread(target=one, args=(i,)) for i in range(streams)]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - started
    print(f"{streams} at a time, 64 tokens each: {sum(done) / elapsed:.1f} tokens/s, {elapsed / 64 * 1000:.1f} ms a step")
PYEOF
}

kernels() {
  grep -h "core.plan:" -r "$SESSION_RUN"/host-a/snapshots/.run/*/log 2> /dev/null | tail -1 | "$PY" -c "
import json, sys
line = sys.stdin.read()
if '{' in line:
    d = json.loads(line[line.index('{'):])
    for phase in ('prefill', 'decode', 'graph'):
        layers = d.get(phase, {}).get('layers.ds1017@1', {})
        if layers:
            print(phase, 'kernels:', sorted({v.split('.')[0] for k, v in layers.items() if k != 'backend' and v}))
    print('graphs:', d['cuda_graph'].get('buckets'))
"
}

n=0
for setting in $SETTINGS; do
  n=$(( n + 1 ))
  started=$(date +%s)
  "$PY" deploy/rsinfer.py control --run "$SESSION_RUN" apply --config "$setting" > "/tmp/session-apply.$$" 2>&1
  echo "== [$n] apply $setting: exit $? in $(( $(date +%s) - started )) s $(grep -E '"status"' "/tmp/session-apply.$$" | head -1 | tr -d ' ,')"
  kernels
  measure
done
rm -f "/tmp/session-apply.$$"
