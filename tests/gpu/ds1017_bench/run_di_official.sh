#!/usr/bin/env bash
# Directly runnable DS1017 official Prefill & Decode benchmark for stock DeepInfer (local via remote-exec, or on-node).
#
# Local commands (run on macOS/dev box; uses artifact-cli + remote_exec.sh / remote-exec):
#   ./run_di_official.sh remote-prefill <TAG> [BUDGET ...]   # default: 4096 8192 16384 32768 65536
#   ./run_di_official.sh remote-decode  <TAG> [D ...]        # default: 4 8 16 32 64 128 256
#   ./run_di_official.sh remote-status  <TAG>
#   ./run_di_official.sh remote-stop    <TAG>
#
# On-node commands (inside staged directory on n1):
#   ./run_di_official.sh prefill [BUDGET ...]
#   ./run_di_official.sh decode  [D ...]
#
# Optional environment variables:
#   N1_GPUS=0,1,2,3           GPU indices (4-GPU DP4/EP4)
#   DI_MODE=resident|cold     Prefill mode: resident (1 model load, default) or cold (upstream per-budget cold start)
#   ENGRAM_QPS=               Optional Engram QP override (empty = stock default)
#   MASTER_PORT=36995         Rendezvous port
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

stage_di_scripts() {
    local tag=$1
    require_tag "$tag"
    local scratch object dest
    scratch=$(mktemp -d)
    trap 'rm -rf "$scratch"' RETURN
    COPYFILE_DISABLE=1 tar czf "$scratch/di-bench.tgz" --no-xattrs --exclude='._*' \
        -C "$script_dir" run_di_official.sh di_official_sweep.py
    object="di-official-${tag}-scripts.tgz"
    dest="${REMOTE_BASE}/di-${tag}"
    artifact-cli cp -f "$scratch/di-bench.tgz" "${OSS_PREFIX}/${object}"
    "$script_dir/remote_exec.sh" "set -e; mkdir -p '${dest}'; tar xzf '${REMOTE_OSS_DIR}/${object}' -C '${dest}'"
    echo "$dest"
}

# ---- Local (remote-exec) subcommands ----
case "$cmd" in
    remote-prefill)
        tag=${1:-}; require_tag "$tag"; shift
        budgets=("$@"); if [[ ${#budgets[@]} -eq 0 ]]; then budgets=(4096 8192 16384 32768 65536); fi
        dest=$(stage_di_scripts "$tag" | tail -n 1)
        gpus="${N1_GPUS:-0,1,2,3}"
        di_mode="${DI_MODE:-resident}"
        qps="${ENGRAM_QPS:-}"
        mport="${MASTER_PORT:-36995}"
        sess="di-prefill-${tag}"
        "$script_dir/remote_exec.sh" "set -e; cd '${dest}'; tmux new-session -d -s '${sess}' 'N1_GPUS=${gpus} DI_MODE=${di_mode} ENGRAM_QPS=${qps} MASTER_PORT=${mport} SWEEP_COPY_TAG=di-prefill-${tag} bash run_di_official.sh prefill ${budgets[*]} >bench.log 2>&1; echo \$? >bench.exit'"
        echo "Started tmux session '${sess}' on ${TARGET_POD:-gpu-node-n1} in ${dest}"
        echo "Check progress: $script_dir/remote_exec.sh \"tail -n 25 '${dest}/bench.log'; cat '${dest}/bench.exit' 2>/dev/null || true\""
        exit 0
        ;;
    remote-decode)
        tag=${1:-}; require_tag "$tag"; shift
        ds=("$@"); if [[ ${#ds[@]} -eq 0 ]]; then ds=(4 8 16 32 64 128 256); fi
        dest=$(stage_di_scripts "$tag" | tail -n 1)
        gpus="${N1_GPUS:-0,1,2,3}"
        qps="${ENGRAM_QPS:-}"
        mport="${MASTER_PORT:-35927}"
        sess="di-decode-${tag}"
        "$script_dir/remote_exec.sh" "set -e; cd '${dest}'; tmux new-session -d -s '${sess}' 'N1_GPUS=${gpus} ENGRAM_QPS=${qps} MASTER_PORT=${mport} SWEEP_COPY_TAG=di-decode-${tag} bash run_di_official.sh decode ${ds[*]} >bench.log 2>&1; echo \$? >bench.exit'"
        echo "Started tmux session '${sess}' on ${TARGET_POD:-gpu-node-n1} in ${dest}"
        echo "Check progress: $script_dir/remote_exec.sh \"tail -n 25 '${dest}/bench.log'; cat '${dest}/bench.exit' 2>/dev/null || true\""
        exit 0
        ;;
    remote-status)
        tag=${1:-}; require_tag "$tag"
        dest="${REMOTE_BASE}/di-${tag}"
        "$script_dir/remote_exec.sh" "cd '${dest}' && tail -n 25 bench.log 2>/dev/null || true; echo '--- exit ---'; cat bench.exit 2>/dev/null || echo running; tmux ls 2>/dev/null || true"
        exit 0
        ;;
    remote-stop)
        tag=${1:-}; require_tag "$tag"
        "$script_dir/remote_exec.sh" "kill_tree() { local p=\$1; for c in \$(pgrep -P \"\$p\" 2>/dev/null || true); do kill_tree \"\$c\"; done; kill -9 \"\$p\" 2>/dev/null || true; }; for s in 'di-prefill-${tag}' 'di-decode-${tag}'; do for p in \$(tmux list-panes -t \"\$s\" -F '#{pane_pid}' 2>/dev/null || true); do kill_tree \"\$p\"; done; tmux kill-session -t \"\$s\" 2>/dev/null || true; done"
        exit 0
        ;;
esac

# ---- On-node execution ----
work=$(pwd)
if [[ -f /workspace/gpu-env/deepinfer-env.sh ]]; then
    # shellcheck source=/dev/null
    source /workspace/gpu-env/deepinfer-env.sh
elif [[ -f /workspace/gpu-env/cuda-13.2-env.sh ]]; then
    # shellcheck source=/dev/null
    source /workspace/gpu-env/cuda-13.2-env.sh
fi
di="${DI_ROOT:-/workspace/agxinfer-env/DeepInfer}"
py="${DI_PY:-/workspace/agxinfer-env/venv-deepinfer/bin/python}"
model="${AGX_MODEL:-/models/DeepSeek-V4.1-Flash-1017}"
gpus_csv="${N1_GPUS:-0,1,2,3}"
device_offset="${DEVICE_OFFSET:-${gpus_csv%%,*}}"

export CUDA_VISIBLE_DEVICES="$gpus_csv" DEVICE_OFFSET="$device_offset" WORLD_SIZE=1 PROFILE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" PYTHONUNBUFFERED=1
export DEEPINFER_HF_MODEL_DIR="$model"
export DEEPINFER_USE_NEXTN_DECODE=0 DEEPINFER_ENABLE_MTP_VARLEN=0
export DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=32 DEEPINFER_USE_MULTIMODAL_POOL=0
export DEEPINFER_ROOT="$di" PYTHONPATH="$di:$di/examples"
export CPATH="/workspace/agxinfer-env/venv-deepinfer/lib/python3.12/site-packages/nvidia/cu13/include${CPATH:+:$CPATH}"
unset DEEPINFER_ENGRAM_LOCAL NVSHMEM_IB_QP_DEPTH NVSHMEM_IBGDA_NUM_RC_PER_PE

assert_gpus_free() {
    local vals
    vals=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpus_csv")
    awk 'NF{n++; if ($1>1500) busy=1} END{exit !(n>0 && !busy)}' <<< "$vals" || {
        echo "GPUs ${gpus_csv} are busy: ${vals}" >&2
        return 1
    }
}

out="$work/results-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$out"
nvidia-smi --query-gpu=index,uuid,driver_version,memory.used --format=csv,noheader -i "$gpus_csv" > "$out/hardware.txt"
git -c safe.directory="$di" -C "$di" rev-parse HEAD > "$out/source.txt" || true

case "$cmd" in
    prefill)
        if [[ $# -eq 0 ]]; then set -- 4096 8192 16384 32768 65536; fi
        assert_gpus_free
        mode="${DI_MODE:-resident}"
        if [[ "$mode" == "cold" ]]; then
            port="${MASTER_PORT:-36995}"
            for b in "$@"; do
                echo "$(date -Is) DeepInfer cold prefill budget=${b}"
                if [[ -n "${ENGRAM_QPS:-}" ]]; then
                    SWEEP_MODE=prefill SWEEP_POINTS="$b" SWEEP_OUT="$out" SWEEP_PREFIX="di-prefill" \
                        SWEEP_COPY_DIR="${REMOTE_OSS_DIR}" MASTER_PORT="$port" \
                        "$py" "$script_dir/di_official_sweep.py" \
                        --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size "$b" \
                        --decode_num_requests 32 --max_model_len 135168 --yoco_clip 140 \
                        --sampler_force_topk 1024 --result_jsonl "$out/init-${b}.jsonl"
                else
                    (cd "$di" && MASTER_PORT="$port" "$py" examples/benchmark.py \
                        --model_name deepseek-ai/DeepSeek-V4.1-Flash-1017 --cp 1 \
                        --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size "$b" \
                        --decode_num_requests 32 --max_model_len 135168 --yoco_clip 140 \
                        --sampler_force_topk 1024 --result_jsonl "$out/di-prefill-${b}.jsonl")
                    if [[ -d "$REMOTE_OSS_DIR" ]]; then
                        cp "$out/di-prefill-${b}.jsonl" "${REMOTE_OSS_DIR}/${SWEEP_COPY_TAG:-di-prefill}-di-prefill-${b}.jsonl"
                    fi
                fi
                port=$((port + 1))
            done
        else
            max_b=0
            points_csv=""
            for b in "$@"; do
                if (( b > max_b )); then max_b=$b; fi
                points_csv="${points_csv:+${points_csv},}${b}"
            done
            echo "$(date -Is) DeepInfer resident prefill points=${points_csv} (init_budget=${max_b})"
            SWEEP_MODE=prefill SWEEP_POINTS="$points_csv" SWEEP_OUT="$out" SWEEP_PREFIX="di-prefill" \
                SWEEP_COPY_DIR="${REMOTE_OSS_DIR}" MASTER_PORT="${MASTER_PORT:-36995}" \
                "$py" "$script_dir/di_official_sweep.py" \
                --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size "$max_b" \
                --decode_num_requests 32 --max_model_len 135168 --yoco_clip 140 \
                --sampler_force_topk 1024 --result_jsonl "$out/init.jsonl"
        fi
        echo "$(date -Is) completed prefill -> ${out}"
        ;;
    decode)
        if [[ $# -eq 0 ]]; then set -- 4 8 16 32 64 128 256; fi
        assert_gpus_free
        # At prefill_batch_size=32768, DeepInfer constructs batch_num_requests=max(248, D).
        # Group D<=248 into one resident engine; run D>248 cold with its exact capacity.
        resident_csv=""
        cold_list=()
        for d in "$@"; do
            if (( d <= 248 )); then
                resident_csv="${resident_csv:+${resident_csv},}${d}"
            else
                cold_list+=("$d")
            fi
        done
        port="${MASTER_PORT:-35927}"
        if [[ -n "$resident_csv" ]]; then
            first_d="${resident_csv%%,*}"
            echo "$(date -Is) DeepInfer resident decode D=${resident_csv}"
            SWEEP_MODE=decode SWEEP_POINTS="$resident_csv" SWEEP_OUT="$out" SWEEP_PREFIX="di-decode" \
                SWEEP_COPY_DIR="${REMOTE_OSS_DIR}" MASTER_PORT="$port" \
                "$py" "$script_dir/di_official_sweep.py" \
                --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size 32768 \
                --decode_num_requests "$first_d" --max_model_len 135168 --yoco_clip 140 \
                --sampler_force_topk 1024 --result_jsonl "$out/init-decode.jsonl"
            port=$((port + 1))
        fi
        for d in "${cold_list[@]}"; do
            echo "$(date -Is) DeepInfer cold decode D=${d}"
            SWEEP_MODE=decode SWEEP_POINTS="$d" SWEEP_OUT="$out" SWEEP_PREFIX="di-decode" \
                SWEEP_COPY_DIR="${REMOTE_OSS_DIR}" MASTER_PORT="$port" \
                "$py" "$script_dir/di_official_sweep.py" \
                --context_length 131072 --non_cached_tokens 4096 --prefill_batch_size 32768 \
                --decode_num_requests "$d" --max_model_len 135168 --yoco_clip 140 \
                --sampler_force_topk 1024 --result_jsonl "$out/init-decode-${d}.jsonl"
            port=$((port + 1))
        done
        echo "$(date -Is) completed decode -> ${out}"
        ;;
    *)
        sed -n '2,18p' "$0" >&2
        exit 2
        ;;
esac
