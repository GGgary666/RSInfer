#!/usr/bin/env bash
# Directly runnable DS1017 official Prefill & Decode benchmark for rsinfer (local via remote-exec, or on-node).
#
# Local commands (run on macOS/dev box; uses artifact-cli + remote_exec.sh / remote-exec):
#   ./run_rsinfer_official.sh stage <TAG>
#   ./run_rsinfer_official.sh remote-prefill <TAG> [BUDGET ...]      # default: 4096 8192 16384 32768 65536
#   ./run_rsinfer_official.sh remote-decode  <TAG> [D ...]           # default: 4 8 16 32 64 128 256
#   ./run_rsinfer_official.sh remote-status  <TAG>
#   ./run_rsinfer_official.sh remote-stop    <TAG>
#
# On-node commands (inside staged directory on n1):
#   ./run_rsinfer_official.sh prepare [prefill|decode]
#   ./run_rsinfer_official.sh serve-prefill
#   ./run_rsinfer_official.sh serve-decode
#   ./run_rsinfer_official.sh prefill [BUDGET ...]
#   ./run_rsinfer_official.sh decode  [D ...]
#   ./run_rsinfer_official.sh status
#   ./run_rsinfer_official.sh stop
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
cmd=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi

OSS_PREFIX="${OSS_PREFIX:-s3://<artifact-bucket>/rsinfer/runs}"
REMOTE_OSS_DIR="${REMOTE_OSS_DIR:-/mnt/artifacts/runs}"
REMOTE_BASE="${REMOTE_BASE:-/workspace/ds1017-bench}"

require_tag() {
    local tag=${1:-}
    if [[ -z "$tag" || ! $tag =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
        echo "error: provide a valid TAG (alphanumeric, ., _, -)" >&2
        exit 2
    fi
}

find_repo_root() {
    local d="$script_dir"
    while [[ "$d" != "/" ]]; do
        if [[ -d "$d/engine" && -d "$d/deploy" ]]; then
            echo "$d"
            return 0
        fi
        d=$(dirname "$d")
    done
    echo "error: could not find repository root containing engine/ and deploy/" >&2
    exit 2
}

stage_to_n1() {
    local tag=$1 mode=${2:-prefill}
    require_tag "$tag"
    local repo scratch object dest
    repo=$(find_repo_root)
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' RETURN
    {
        git -C "$repo" rev-parse HEAD 2>/dev/null || echo "unknown"
        git -C "$repo" status --short -- engine deploy tests/gpu/ds1017_bench 2>/dev/null || true
    } > "$scratch/source-version.txt"
    COPYFILE_DISABLE=1 tar czf "$scratch/source.tgz" --no-xattrs \
        --exclude='__pycache__' --exclude='*.pyc' --exclude='._*' --exclude='extension' \
        -C "$repo" engine deploy tests/gpu/ds1017_bench \
        -C "$scratch" source-version.txt
    object="rsinfer-official-${tag}-source.tgz"
    dest="${REMOTE_BASE}/${tag}"
    artifact-cli cp -f "$scratch/source.tgz" "${OSS_PREFIX}/${object}"
    "$script_dir/remote_exec.sh" "set -e; mkdir -p '${REMOTE_BASE}'; rm -rf '${dest}' && mkdir -p '${dest}'; tar xzf '${REMOTE_OSS_DIR}/${object}' -C '${dest}'; bash '${dest}/tests/gpu/ds1017_bench/run_rsinfer_official.sh' prepare '${mode}'"
    echo "$dest"
}

# ---- Local (remote-exec) subcommands ----
case "$cmd" in
    stage)
        stage_to_n1 "${1:-}" "${2:-prefill}"
        exit 0
        ;;
    remote-prefill)
        tag=${1:-}; require_tag "$tag"; shift
        budgets=("$@"); if [[ ${#budgets[@]} -eq 0 ]]; then budgets=(4096 8192 16384 32768 65536); fi
        dest=$(stage_to_n1 "$tag" prefill | tail -n 1)
        gpus="${N1_GPUS:-0,1,2,3}"
        port="${N1_PORT:-24739}"
        sess="rsinfer-prefill-${tag}"
        "$script_dir/remote_exec.sh" "set -e; cd '${dest}'; tmux new-session -d -s '${sess}' 'N1_GPUS=${gpus} N1_PORT=${port} N1_RUN=${dest}/run-prefill bash tests/gpu/ds1017_bench/run_rsinfer_official.sh prefill ${budgets[*]} >bench.log 2>&1; echo \$? >bench.exit'"
        echo "Started tmux session '${sess}' on ${TARGET_POD:-gpu-node-n1} in ${dest}"
        echo "Check progress: $script_dir/remote_exec.sh \"tail -n 25 '${dest}/bench.log'; cat '${dest}/bench.exit' 2>/dev/null || true\""
        exit 0
        ;;
    remote-decode)
        tag=${1:-}; require_tag "$tag"; shift
        ds=("$@"); if [[ ${#ds[@]} -eq 0 ]]; then ds=(4 8 16 32 64 128 256); fi
        dest=$(stage_to_n1 "$tag" decode | tail -n 1)
        gpus="${N1_GPUS:-0,1,2,3}"
        port="${N1_PORT:-23619}"
        sess="rsinfer-decode-${tag}"
        "$script_dir/remote_exec.sh" "set -e; cd '${dest}'; tmux new-session -d -s '${sess}' 'N1_GPUS=${gpus} N1_PORT=${port} N1_RUN=${dest}/run-decode bash tests/gpu/ds1017_bench/run_rsinfer_official.sh decode ${ds[*]} >bench.log 2>&1; echo \$? >bench.exit'"
        echo "Started tmux session '${sess}' on ${TARGET_POD:-gpu-node-n1} in ${dest}"
        echo "Check progress: $script_dir/remote_exec.sh \"tail -n 25 '${dest}/bench.log'; cat '${dest}/bench.exit' 2>/dev/null || true\""
        exit 0
        ;;
    remote-status)
        tag=${1:-}; require_tag "$tag"
        dest="${REMOTE_BASE}/${tag}"
        "$script_dir/remote_exec.sh" "cd '${dest}' && tail -n 25 bench.log 2>/dev/null || true; echo '--- exit ---'; cat bench.exit 2>/dev/null || echo running; tmux ls 2>/dev/null || true"
        exit 0
        ;;
    remote-stop)
        tag=${1:-}; require_tag "$tag"
        dest="${REMOTE_BASE}/${tag}"
        "$script_dir/remote_exec.sh" "tmux kill-session -t 'rsinfer-prefill-${tag}' 2>/dev/null || true; tmux kill-session -t 'rsinfer-decode-${tag}' 2>/dev/null || true; for r in '${dest}/run-prefill' '${dest}/run-decode'; do if [[ -d \"\$r\" ]]; then N1_RUN=\"\$r\" bash '${dest}/tests/gpu/ds1017_bench/run_rsinfer_official.sh' stop || true; fi; done"
        exit 0
        ;;
esac

# ---- On-node execution ----
src=$(find_repo_root)
cd "$src"
if [[ -f /workspace/gpu-env/cuda-13.2-env.sh ]]; then
    # shellcheck source=/dev/null
    source /workspace/gpu-env/cuda-13.2-env.sh
fi
py="${RSINFER_PY:-/workspace/claude-yoco/venv-opt2/bin/python}"
model="${RSINFER_MODEL:-/models/DeepSeek-V4.1-Flash-1017}"
cli="$src/deploy/rsinfer.py"
bench_dir="$src/tests/gpu/ds1017_bench"
gpus_csv="${N1_GPUS:-0,1,2,3}"
run="${N1_RUN:-$src/run}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
export PYTHONPATH="$src/engine:$src"
export CPATH="/workspace/rsinfer-env/venv-deepinfer/lib/python3.12/site-packages/nvidia/cu13/include${CPATH:+:$CPATH}"

assert_gpus_free() {
    local vals
    vals=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpus_csv")
    awk 'NF{n++; if ($1>1500) busy=1} END{exit !(n>0 && !busy)}' <<< "$vals" || {
        echo "GPUs ${gpus_csv} are busy: ${vals}" >&2
        return 1
    }
}

apply_config() {
    local cfg=$1 out_json=$2
    "$py" "$cli" control --run "$run" apply --source "$src" --config "$cfg" --wait-empty 180 --json > "$out_json"
    "$py" -c 'import json,sys; r=json.load(open(sys.argv[1]))["result"]; assert r["status"]=="committed", r.get("diagnostics", r["status"])' "$out_json"
}

case "$cmd" in
    prepare)
        mode=${1:-prefill}
        if [[ ! -f "$src/source-version.txt" ]]; then
            echo "error: refuse to patch a live git repo directly; run 'stage' first." >&2
            exit 2
        fi
        patchfile="$bench_dir/top5-measurement.patch"
        if ! patch --force -s -R -p1 --dry-run < "$patchfile" >/dev/null 2>&1; then
            patch -p1 < "$patchfile"
        fi
        "$py" -v >/dev/null 2>&1 || py=python3
        "$py" "$bench_dir/admit_probe.py" "$src"
        "$py" "$bench_dir/dsbench_step_probe.py" "$src"
        if [[ "$mode" == "decode" ]]; then
            "$py" "$bench_dir/decode_counts_probe.py" "$src"
        fi
        echo "Prepared ${src} (mode=${mode})"
        cat "$src/source-version.txt"
        ;;
    status)
        exec "$py" "$cli" status --run "$run"
        ;;
    stop)
        exec "$py" "$cli" stop --run "$run" --timeout 90
        ;;
    serve-prefill)
        assert_gpus_free
        exec "$py" "$cli" serve "$model" --source "$src" --run "$run" --detach --port "${N1_PORT:-24739}" \
            --gpus "$gpus_csv" --ds-backend deepinfer --cuda-graph auto --graph-buckets 1 2 4 8 16 32 \
            --max-context 135168 --max-rows 64 --max-slots 320 --token-budget 4096 --overlap --mem-fraction .75 \
            --yoco-clip 140 --timeout 1800 --proc-tunables '{"ipc.shm_capacity":134217728}' \
            --set eplb.extra_experts=32 --set deepinfer.force_random_gate=true --set sampler.force_topk=1024 \
            --set sched.prefill_interval=108 --set sched.prefill_occupancy_target=0
        ;;
    serve-decode)
        assert_gpus_free
        exec "$py" "$cli" serve "$model" --source "$src" --run "$run" --detach --port "${N1_PORT:-23619}" \
            --gpus "$gpus_csv" --ds-backend deepinfer --cuda-graph auto \
            --graph-buckets 1 2 4 8 16 32 64 128 256 \
            --max-context 135168 --max-rows 256 --max-slots 1440 --token-budget 32768 --overlap --mem-fraction .78 \
            --yoco-clip 140 --timeout 1800 --proc-tunables '{"ipc.shm_capacity":16777216}' \
            --set frontend.response_timeout_s=1800.0 --set eplb.extra_experts=32 \
            --set deepinfer.force_random_gate=true --set sampler.force_topk=1024 --set sched.prefill_interval=1
        ;;
    prefill)
        if [[ $# -eq 0 ]]; then set -- 4096 8192 16384 32768 65536; fi
        started_here=0
        if [[ ! -f "$run/engine.json" ]]; then
            bash "$0" serve-prefill
            started_here=1
            trap 'if [[ "${KEEP_ENGINE:-0}" != "1" ]]; then "$py" "$cli" stop --run "$run" --timeout 90 || true; fi' EXIT
        fi
        out="$src/results/prefill-$(date -u +%Y%m%dT%H%M%SZ)-$$"
        mkdir -p "$out"
        nvidia-smi --query-gpu=index,uuid,driver_version --format=csv,noheader -i "$gpus_csv" > "$out/hardware.txt"
        for b in "$@"; do
            case "$b" in 4096|8192|16384|32768|65536) ;; *) echo "Invalid prefill budget: $b" >&2; exit 2 ;; esac
            echo "$(date -Is) RSInfer prefill budget=${b}"
            for cache in false true; do
                cfg="{\"sched.token_budget\":$b,\"alloc.radix_cache\":$cache,\"sched.max_batch\":64,\"batch.max_rows\":64,\"holder.max_slots\":320,\"plan.cuda_graph\":\"auto\",\"plan.graph_buckets\":[1,2,4,8,16,32],\"sched.prefill_interval\":108,\"sched.prefill_occupancy_target\":0,\"eplb.extra_experts\":32,\"deepinfer.force_random_gate\":true,\"sampler.force_topk\":1024,\"dsv4.yoco_mode\":\"compact\"}"
                apply_config "$cfg" "$out/apply-${b}-${cache}.json"
            done
            "$py" "$bench_dir/dsbench_pareto_sweep.py" --run "$run" --model "$model" --mode prefill \
                --context-length 131072 --non-cached-tokens 4096 --yoco-clip 140 --prefill-batch-size "$b" \
                --seed-base 20000 --prefill-measure-steps 24 --prefill-tail-steps 4 --step-stats \
                --result-jsonl "$out/rsinfer-prefill-${b}.jsonl"
            if [[ -d "$REMOTE_OSS_DIR" ]]; then
                obj="rsinfer-prefill-$(basename "$src")-${b}.jsonl"
                cp "$out/rsinfer-prefill-${b}.jsonl" "${REMOTE_OSS_DIR}/${obj}"
                echo "RESULT ${OSS_PREFIX}/${obj}"
            fi
        done
        if [[ $started_here -eq 1 && "${KEEP_ENGINE:-0}" == "1" ]]; then
            echo "Engine left running at ${run}"
        fi
        ;;
    decode)
        if [[ $# -eq 0 ]]; then set -- 4 8 16 32 64 128 256; fi
        if [[ ! -f "$run/engine.json" ]]; then
            bash "$0" serve-decode
            trap 'if [[ "${KEEP_ENGINE:-0}" != "1" ]]; then "$py" "$cli" stop --run "$run" --timeout 90 || true; fi' EXIT
        fi
        out="$src/results/decode-$(date -u +%Y%m%dT%H%M%SZ)-$$"
        mkdir -p "$out"
        nvidia-smi --query-gpu=index,uuid,driver_version --format=csv,noheader -i "$gpus_csv" > "$out/hardware.txt"
        for d in "$@"; do
            echo "$(date -Is) RSInfer decode D=${d}"
            for cache in false true; do
                apply_config "{\"alloc.radix_cache\":$cache}" "$out/apply-d${d}-${cache}.json"
            done
            "$py" "$bench_dir/dsbench_decode.py" --run "$run" --model "$model" --d "$d" \
                --out "$out/rsinfer-decode-${d}.json"
            if [[ -d "$REMOTE_OSS_DIR" ]]; then
                obj="rsinfer-decode-$(basename "$src")-d${d}.json"
                cp "$out/rsinfer-decode-${d}.json" "${REMOTE_OSS_DIR}/${obj}"
                echo "RESULT ${OSS_PREFIX}/${obj}"
            fi
        done
        ;;
    *)
        sed -n '2,18p' "$0" >&2
        exit 2
        ;;
esac
