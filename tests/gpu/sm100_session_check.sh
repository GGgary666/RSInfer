#!/bin/bash
# What a hot-apply session (sm100_session.sh) runs after each apply, unless the commit names another
# check: a few greedy completions through the engine's HTTP API, their texts' digest, and the kernels
# the worker runs. SESSION_URL and SESSION_RUN come from the session.
set -u
for prompt in "The quick brown fox jumps over the lazy dog." "Hello" "写一首关于秋天的诗"; do
  curl -s -m 300 "$SESSION_URL/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\": \"DeepSeek-V4.1-Flash\", \"prompt\": \"$prompt\", \"max_tokens\": 16, \"temperature\": 0}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['text'] if 'choices' in d else d)"
done > /tmp/session-texts.$$ 2>&1
echo "texts digest: $(md5sum < /tmp/session-texts.$$ | cut -c1-12)"
rm -f /tmp/session-texts.$$
grep -h "core.plan:" -r "$SESSION_RUN"/host-a/snapshots/.run/*/log 2>/dev/null | tail -1 | python3 -c "
import json, sys
line = sys.stdin.read()
if '{' in line:
    d = json.loads(line[line.index('{'):])
    layers = d['decode'].get('layers.ds1017@1', {})
    print('decode backend:', sorted({v.split('.')[0] for k, v in layers.items() if k != 'backend'}), 'graphs:', d['cuda_graph'].get('buckets'))
"
