#!/usr/bin/env bash
# Directly runnable AgentX dataset generation and replay benchmark for agxinfer & DeepInfer (local via remote-exec, or on-node).
#
# Local commands (run on macOS/dev box; uses artifact-cli + remote_exec.sh / remote-exec):
#   ./run_agentx.sh stage          <TAG>
#   ./run_agentx.sh remote-prepare <TAG> <REMOTE_WEKA_TRACES_JSONL> <REMOTE_DATA_DIR>
#   ./run_agentx.sh remote-agx     <TAG> <REMOTE_SUITE_PREFIX> [CONCURRENCIES ...]  # default: 16 24 48
#   ./run_agentx.sh remote-di      <TAG> <REMOTE_SUITE_PREFIX> [CONCURRENCIES ...]  # default: 16 24 48
#   ./run_agentx.sh remote-status  <TAG>
#   ./run_agentx.sh remote-stop    <TAG>
#
# On-node commands (inside staged directory on n1):
#   ./run_agentx.sh prepare   <WEKA_TRACES_JSONL> <OUT_DATA_DIR>
#   ./run_agentx.sh serve-agx
#   ./run_agentx.sh run-agx   <SUITE_PREFIX> [CONCURRENCIES ...]
#   ./run_agentx.sh stop-agx
#   ./run_agentx.sh run-di    <SUITE_PREFIX> [CONCURRENCIES ...]
#
# Notes:
#   <SUITE_PREFIX> refers to <prefix>.json and <prefix>.i32 (e.g. /workspace/.../agentx_eval).
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
cmd=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi

OSS_PREFIX="${OSS_PREFIX:-s3://<artifact-bucket>/agxinfer/runs}"
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
    local tag=$1
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
    object="agentx-${tag}-source.tgz"
    dest="${REMOTE_BASE}/agentx-${tag}"
    artifact-cli cp -f "$scratch/source.tgz" "${OSS_PREFIX}/${object}"
    "$script_dir/remote_exec.sh" "set -e; mkdir -p '${REMOTE_BASE}'; rm -rf '${dest}' && mkdir -p '${dest}'; tar xzf '${REMOTE_OSS_DIR}/${object}' -C '${dest}'; bash '${dest}/tests/gpu/ds1017_bench/run_agentx.sh' prepare-code"
    echo "$dest"
}

# ---- Local (remote-exec) subcommands ----
case "$cmd" in
    stage)
        stage_to_n1 "${1:-}"
        exit 0
        ;;
    remote-prepare)
        tag=${1:-}; traces=${2:-}; out_dir=${3:-}
        require_tag "$tag"
        if [[ -z "$traces" || -z "$out_dir" ]]; then
            echo "usage: $0 remote-prepare <TAG> <REMOTE_WEKA_TRACES_JSONL> <REMOTE_OUT_DATA_DIR>" >&2
            exit 2
        fi
        dest=$(stage_to_n1 "$tag" | tail -n 1)
        "$script_dir/remote_exec.sh" "set -e; cd '${dest}'; bash tests/gpu/ds1017_bench/run_agentx.sh prepare '${traces}' '${out_dir}'"
        exit 0
        ;;
    remote-agx)
        tag=${1:-}; suite_prefix=${2:-}
        require_tag "$tag"
        if [[ -z "$suite_prefix" ]]; then
            echo "usage: $0 remote-agx <TAG> <REMOTE_SUITE_PREFIX> [CONCURRENCIES ...]" >&2
            exit 2
        fi
        shift 2
        concs=("$@"); if [[ ${#concs[@]} -eq 0 ]]; then concs=(16 24 48); fi
        dest=$(stage_to_n1 "$tag" | tail -n 1)
        gpus="${N1_GPUS:-0,1,2,3}"
        port="${N1_PORT:-24919}"
        sess="agentx-agx-${tag}"
        "$script_dir/remote_exec.sh" "set -e; cd '${dest}'; tmux new-session -d -s '${sess}' 'N1_GPUS=${gpus} N1_PORT=${port} N1_RUN=${dest}/agx-run bash tests/gpu/ds1017_bench/run_agentx.sh run-agx \"${suite_prefix}\" ${concs[*]} >agx.log 2>&1; echo \$? >agx.exit'"
        echo "Started tmux session '${sess}' on ${TARGET_POD:-gpu-node-n1} in ${dest}"
        echo "Check progress: $script_dir/remote_exec.sh \"tail -n 25 '${dest}/agx.log'; cat '${dest}/agx.exit' 2>/dev/null || true\""
        exit 0
        ;;
    remote-di)
        tag=${1:-}; suite_prefix=${2:-}
        require_tag "$tag"
        if [[ -z "$suite_prefix" ]]; then
            echo "usage: $0 remote-di <TAG> <REMOTE_SUITE_PREFIX> [CONCURRENCIES ...]" >&2
            exit 2
        fi
        shift 2
        concs=("$@"); if [[ ${#concs[@]} -eq 0 ]]; then concs=(16 24 48); fi
        dest=$(stage_to_n1 "$tag" | tail -n 1)
        gpus="${N1_GPUS:-0,1,2,3}"
        sess="agentx-di-${tag}"
        "$script_dir/remote_exec.sh" "set -e; cd '${dest}'; tmux new-session -d -s '${sess}' 'N1_GPUS=${gpus} bash tests/gpu/ds1017_bench/run_agentx.sh run-di \"${suite_prefix}\" ${concs[*]} >di.log 2>&1; echo \$? >di.exit'"
        echo "Started tmux session '${sess}' on ${TARGET_POD:-gpu-node-n1} in ${dest}"
        echo "Check progress: $script_dir/remote_exec.sh \"tail -n 25 '${dest}/di.log'; cat '${dest}/di.exit' 2>/dev/null || true\""
        exit 0
        ;;
    remote-status)
        tag=${1:-}; require_tag "$tag"
        dest="${REMOTE_BASE}/agentx-${tag}"
        "$script_dir/remote_exec.sh" "cd '${dest}'; for f in agx.log di.log; do if [[ -f \"\$f\" ]]; then echo \"=== \$f ===\"; tail -n 20 \"\$f\"; fi; done; tmux ls 2>/dev/null || true"
        exit 0
        ;;
    remote-stop)
        tag=${1:-}; require_tag "$tag"
        dest="${REMOTE_BASE}/agentx-${tag}"
        "$script_dir/remote_exec.sh" "tmux kill-session -t 'agentx-agx-${tag}' 2>/dev/null || true; tmux kill-session -t 'agentx-di-${tag}' 2>/dev/null || true; if [[ -d '${dest}/agx-run' ]]; then N1_RUN='${dest}/agx-run' bash '${dest}/tests/gpu/ds1017_bench/run_agentx.sh' stop-agx || true; fi"
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
agpy="${AGX_PY:-/workspace/claude-yoco/venv-opt2/bin/python}"
di="${DI_ROOT:-/workspace/agxinfer-env/DeepInfer}"
dipy="${DI_PY:-/workspace/agxinfer-env/venv-deepinfer/bin/python}"
model="${AGX_MODEL:-/models/DeepSeek-V4.1-Flash-1017}"
cli="$src/deploy/agxinfer.py"
bench_dir="$src/tests/gpu/ds1017_bench"
gpus_csv="${N1_GPUS:-0,1,2,3}"
run="${N1_RUN:-$src/agx-run}"
yoco_clip="${YOCO_CLIP:-140}"
max_ctx="${MAX_CONTEXT:-131072}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
export CPATH="/workspace/agxinfer-env/venv-deepinfer/lib/python3.12/site-packages/nvidia/cu13/include${CPATH:+:$CPATH}"

assert_gpus_free() {
    local vals
    vals=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpus_csv")
    awk 'NF{n++; if ($1>1500) busy=1} END{exit !(n>0 && !busy)}' <<< "$vals" || {
        echo "GPUs ${gpus_csv} are busy: ${vals}" >&2
        return 1
    }
}

case "$cmd" in
    prepare-code)
        if [[ ! -f "$src/source-version.txt" ]]; then
            echo "error: refuse to patch a live git repo directly; run 'stage' first." >&2
            exit 2
        fi
        "$agpy" "$bench_dir/admit_probe.py" "$src"
        echo "Prepared staged AgentX source in ${src}"
        ;;
    prepare)
        traces=${1:-}; out_dir=${2:-}
        if [[ -z "$traces" || -z "$out_dir" ]]; then
            echo "usage: $0 prepare <WEKA_TRACES_JSONL> <OUT_DATA_DIR>" >&2
            exit 2
        fi
        PYTHONPATH="$src/engine:$src" "$agpy" "$bench_dir/agentx_driver.py" prepare \
            --traces-jsonl "$traces" --tokenizer "$model" --out-dir "$out_dir" \
            --max-context "$max_ctx" --dev-sessions "${DEV_SESSIONS:-16}" --eval-sessions "${EVAL_SESSIONS:-48}"
        ;;
    serve-agx)
        assert_gpus_free
        export PYTHONPATH="$src/engine:$src"
        exec "$agpy" "$cli" serve "$model" --source "$src" --run "$run" --detach --port "${N1_PORT:-24919}" \
            --gpus "$gpus_csv" --ds-backend deepinfer --cuda-graph auto \
            --max-rows 128 --graph-buckets 1 2 4 8 12 16 24 32 40 48 56 64 80 96 112 128 \
            --max-context "$max_ctx" --token-budget 32768 --overlap --max-slots 512 --mem-fraction .88 \
            --yoco-clip "$yoco_clip" --timeout 2400 --proc-tunables '{"ipc.shm_capacity":134217728}' \
            --set state.kv.pages=65536 --set frontend.response_timeout_s=900.0 \
            --set sampler.force_topk=0 --set deepinfer.force_random_gate=false \
            --set eplb.extra_experts=0 --set dsv4.graph_idle_experts=true --set sched.prefill_occupancy_target=0
        ;;
    stop-agx)
        export PYTHONPATH="$src/engine:$src"
        exec "$agpy" "$cli" stop --run "$run" --timeout 90
        ;;
    run-agx)
        suite_prefix=${1:-}
        if [[ -z "$suite_prefix" ]]; then
            echo "usage: $0 run-agx <SUITE_PREFIX> [CONCURRENCIES ...]" >&2
            exit 2
        fi
        shift
        if [[ $# -eq 0 ]]; then set -- 16 24 48; fi
        export PYTHONPATH="$src/engine:$src"
        if [[ ! -f "$run/engine.json" ]]; then
            bash "$0" serve-agx
            trap 'if [[ "${KEEP_ENGINE:-0}" != "1" ]]; then "$agpy" "$cli" stop --run "$run" --timeout 90 || true; fi' EXIT
        fi
        out_dir="$src/results/agentx-agx-$(date -u +%Y%m%dT%H%M%SZ)-$$"
        mkdir -p "$out_dir"
        for c in "$@"; do
            echo "$(date -Is) AGX AgentX suite=${suite_prefix} C=${c}"
            for enabled in false true; do
                f="$out_dir/apply-c${c}-${enabled}.json"
                "$agpy" "$cli" control --run "$run" apply --source "$src" \
                    --config "{\"alloc.radix_cache\":$enabled}" --wait-empty 120 --json > "$f"
                "$agpy" -c 'import json,sys; d=json.load(open(sys.argv[1]))["result"]; assert d["status"]=="committed", d' "$f"
            done
            "$agpy" "$bench_dir/agentx_driver.py" agx-direct --run "$run" \
                --suite "${suite_prefix}.json" --corpus "${suite_prefix}.i32" \
                --concurrencies "$c" --out "$out_dir/agx-c${c}.json"
            if [[ -d "$REMOTE_OSS_DIR" ]]; then
                obj="agentx-agx-$(basename "$src")-c${c}.json"
                cp "$out_dir/agx-c${c}.json" "${REMOTE_OSS_DIR}/${obj}"
                echo "RESULT ${OSS_PREFIX}/${obj}"
            fi
        done
        ;;
    run-di)
        suite_prefix=${1:-}
        if [[ -z "$suite_prefix" ]]; then
            echo "usage: $0 run-di <SUITE_PREFIX> [CONCURRENCIES ...]" >&2
            exit 2
        fi
        shift
        if [[ $# -eq 0 ]]; then set -- 16 24 48; fi
        assert_gpus_free
        if [[ -f /workspace/gpu-env/deepinfer-env.sh ]]; then
            # shellcheck source=/dev/null
            source /workspace/gpu-env/deepinfer-env.sh
        fi
        export CUDA_VISIBLE_DEVICES="$gpus_csv" DEVICE_OFFSET=0 WORLD_SIZE=1 PROFILE=0
        export DEEPINFER_NUM_STATE_SLOTS_RATIO=4 DEEPINFER_USE_NEXTN_DECODE=0 DEEPINFER_ENABLE_MTP_VARLEN=0
        export DEEPINFER_USE_MULTIMODAL_POOL=0 DEEPINFER_FORCE_LOAD_BALANCE=0 DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=0
        export DEEPINFER_HF_MODEL_DIR="$model" DEEPINFER_ROOT="$di"
        export PYTHONPATH="$bench_dir:$di"
        export MASTER_PORT="${MASTER_PORT:-36897}"
        unset DEEPINFER_ENGRAM_LOCAL
        if ! compgen -G "$bench_dir/extension/ds_prefix_pages*.so" >/dev/null; then
            "$dipy" "$bench_dir/build_di_prefix_ext.py"
        fi
        out_dir="$src/results/agentx-di-$(date -u +%Y%m%dT%H%M%SZ)-$$"
        mkdir -p "$out_dir"
        "$dipy" "$bench_dir/agentx_driver.py" deepinfer \
            --model "$model" --deepinfer "$di" \
            --suite "${suite_prefix}.json" --corpus "${suite_prefix}.i32" \
            --concurrencies "$@" --out "$out_dir/di.json" \
            --mem .88 --token-budget 32768 --max-seqs 128 --max-model-len "$max_ctx" --yoco-clip "$yoco_clip"
        if [[ -d "$REMOTE_OSS_DIR" ]]; then
            obj="agentx-di-$(basename "$src").json"
            cp "$out_dir/di.json" "${REMOTE_OSS_DIR}/${obj}"
            echo "RESULT ${OSS_PREFIX}/${obj}"
        fi
        ;;
    *)
        sed -n '2,20p' "$0" >&2
        exit 2
        ;;
esac
