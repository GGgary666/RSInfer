#!/usr/bin/env bash
# Local one-command launcher and operator wrapper for RSInfer on DS1017 (4x SM100, DP4/EP4).
#
# Automatically aligns CUDA_HOME / nvcc / LD_LIBRARY_PATH / TRITON_PTXAS_PATH with torch.version.cuda
# and adds nvidia.cu13 headers to CPATH, so it works out of the box on multi-CUDA hosts and containers.
#
# Usage:
#   ./deploy/serve_ds1017.sh serve --model <MODEL_DIR> [OPTIONS] [-- EXTRA_RSINFER_ARGS...]
#   ./deploy/serve_ds1017.sh status  [--run <RUN_DIR>]
#   ./deploy/serve_ds1017.sh chat    "<PROMPT>" [--run <RUN_DIR>] [--max-tokens N]
#   ./deploy/serve_ds1017.sh complete "<PROMPT>" [--run <RUN_DIR>] [--max-tokens N]
#   ./deploy/serve_ds1017.sh control <check|apply|status|introspect|logs> [ARGS...] [--run <RUN_DIR>]
#   ./deploy/serve_ds1017.sh sleep   [--run <RUN_DIR>] [--wait-empty [S] | --force]
#   ./deploy/serve_ds1017.sh wake    [--run <RUN_DIR>]
#   ./deploy/serve_ds1017.sh stop    [--run <RUN_DIR>] [--timeout S]
#
# Serve options (all can also be passed via environment variables):
#   --model <DIR>          DS1017 model directory (env: RSINFER_MODEL / MODEL_DIR)
#   --gpus <CSV>           GPU indices for DP/EP (default: 0,1,2,3; env: GPUS)
#   --python <PY>          Python interpreter (default: active python/python3; env: RSINFER_PY)
#   --run <DIR>            Engine run directory (default: run/rsinfer; env: RUN_DIR)
#   --host <ADDR>          HTTP bind address (default: 127.0.0.1; env: HOST)
#   --port <PORT>          HTTP port (default: 8000; env: PORT)
#   --ds-backend <NAME>    DS1017 kernel backend (default: deepinfer; env: DS_BACKEND)
#   --cuda-graph <MODE>    CUDA graph mode: auto, off, required (default: auto; env: CUDA_GRAPH)
#   --max-context <N>      Max sequence length (default: 131072; env: MAX_CONTEXT)
#   --token-budget <N>     Prefill token budget per step (default: 32768; env: TOKEN_BUDGET)
#   --max-rows <N>         Max concurrent rows per rank (default: 128; env: MAX_ROWS)
#   --max-slots <N>        SWA slot budget per rank (default: 512; env: MAX_SLOTS)
#   --mem-fraction <F>     GPU memory fraction for weights + states (default: 0.80; env: MEM_FRACTION)
#   --yoco-clip <N>        YoCo retained suffix tokens, 0 disables (default: 140; env: YOCO_CLIP)
#   --source <DIR>         Workspace whose engine/ tree to serve (default: repo root)
#   --cuda-env <FILE>      Optional shell script to source for CUDA toolchain (env: CUDA_ENV_SCRIPT)
#   --set <KEY=VALUE>      Extra engine config override (repeatable)
#   --foreground           Run in the foreground until Ctrl-C (default: --detach)
#   --dry-run              Write topology/config and print command without starting
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$script_dir/.." && pwd)

usage() {
    sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'
}

cmd=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "$cmd" in
    -h|--help|help)
        usage
        exit 0
        ;;
esac

# ---- Common env setup ----
py="${RSINFER_PY:-}"
cuda_env="${CUDA_ENV_SCRIPT:-}"
run_dir="${RUN_DIR:-$repo/run/rsinfer}"

setup_env() {
    if [[ -z "$py" ]]; then
        py=$(command -v python || command -v python3 || true)
    fi
    if [[ -z "$py" || ! -x "$py" ]]; then
        echo "error: python interpreter not found; activate your venv or set RSINFER_PY / --python" >&2
        exit 2
    fi

    if [[ -n "$cuda_env" ]]; then
        if [[ ! -f "$cuda_env" ]]; then
            echo "error: --cuda-env file not found: $cuda_env" >&2
            exit 2
        fi
        # shellcheck source=/dev/null
        source "$cuda_env"
    else
        local torch_cuda cuda_root=""
        torch_cuda=$("$py" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)
        if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
            if [[ -z "$torch_cuda" ]] || "${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep -q "release ${torch_cuda}"; then
                cuda_root="$CUDA_HOME"
            fi
        fi
        if [[ -z "$cuda_root" && -n "$torch_cuda" && -x "/usr/local/cuda-${torch_cuda}/bin/nvcc" ]]; then
            cuda_root="/usr/local/cuda-${torch_cuda}"
        elif [[ -z "$cuda_root" ]] && ! command -v nvcc >/dev/null 2>&1; then
            for d in /usr/local/cuda-13.2 /usr/local/cuda; do
                if [[ -x "$d/bin/nvcc" ]]; then
                    cuda_root="$d"
                    break
                fi
            done
        fi
        if [[ -n "$cuda_root" ]]; then
            export CUDA_HOME="$cuda_root"
            export CUDA_PATH="$cuda_root"
            export CUDACXX="$cuda_root/bin/nvcc"
            export PATH="$cuda_root/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_root/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            if [[ -n "${TRITON_PTXAS_PATH:-}" && -x "$cuda_root/bin/ptxas" ]]; then
                export TRITON_PTXAS_PATH="$cuda_root/bin/ptxas"
            fi
        fi
    fi

    local cu13_inc
    cu13_inc=$("$py" -c '
import importlib.util, pathlib
spec = importlib.util.find_spec("nvidia.cu13")
if spec and spec.submodule_search_locations:
    p = pathlib.Path(next(iter(spec.submodule_search_locations))) / "include"
    if p.is_dir():
        print(p)
' 2>/dev/null || true)
    if [[ -n "$cu13_inc" ]]; then
        export CPATH="${cu13_inc}${CPATH:+:$CPATH}"
    fi
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" PYTHONUNBUFFERED=1
    export PYTHONPATH="$repo/engine:$repo${PYTHONPATH:+:$PYTHONPATH}"
}

case "$cmd" in
    serve)
        model="${RSINFER_MODEL:-${MODEL_DIR:-}}"
        gpus_csv="${GPUS:-0,1,2,3}"
        host="${HOST:-127.0.0.1}"
        port="${PORT:-8000}"
        ds_backend="${DS_BACKEND:-deepinfer}"
        cuda_graph="${CUDA_GRAPH:-auto}"
        max_ctx="${MAX_CONTEXT:-131072}"
        token_budget="${TOKEN_BUDGET:-32768}"
        max_rows="${MAX_ROWS:-128}"
        max_slots="${MAX_SLOTS:-512}"
        mem_fraction="${MEM_FRACTION:-0.80}"
        yoco_clip="${YOCO_CLIP:-140}"
        source_dir="$repo"
        detach=1
        dry_run=0
        sets=()
        extra=()

        while [[ $# -gt 0 ]]; do
            case "$1" in
                -h|--help) usage; exit 0 ;;
                --model) model=${2:?missing value for --model}; shift 2 ;;
                --gpus) gpus_csv=${2:?missing value for --gpus}; shift 2 ;;
                --python) py=${2:?missing value for --python}; shift 2 ;;
                --run) run_dir=${2:?missing value for --run}; shift 2 ;;
                --host) host=${2:?missing value for --host}; shift 2 ;;
                --port) port=${2:?missing value for --port}; shift 2 ;;
                --ds-backend) ds_backend=${2:?missing value for --ds-backend}; shift 2 ;;
                --cuda-graph) cuda_graph=${2:?missing value for --cuda-graph}; shift 2 ;;
                --max-context) max_ctx=${2:?missing value for --max-context}; shift 2 ;;
                --token-budget) token_budget=${2:?missing value for --token-budget}; shift 2 ;;
                --max-rows) max_rows=${2:?missing value for --max-rows}; shift 2 ;;
                --max-slots) max_slots=${2:?missing value for --max-slots}; shift 2 ;;
                --mem-fraction) mem_fraction=${2:?missing value for --mem-fraction}; shift 2 ;;
                --yoco-clip) yoco_clip=${2:?missing value for --yoco-clip}; shift 2 ;;
                --source) source_dir=${2:?missing value for --source}; shift 2 ;;
                --cuda-env) cuda_env=${2:?missing value for --cuda-env}; shift 2 ;;
                --set) sets+=(--set "${2:?missing value for --set}"); shift 2 ;;
                --foreground) detach=0; shift ;;
                --detach) detach=1; shift ;;
                --dry-run) dry_run=1; shift ;;
                --) shift; extra+=("$@"); break ;;
                -*) extra+=("$1"); shift ;;
                *)
                    if [[ -z "$model" ]]; then
                        model="$1"
                    else
                        extra+=("$1")
                    fi
                    shift
                    ;;
            esac
        done

        if [[ -z "$model" || ! -d "$model" || ! -f "$model/config.json" ]]; then
            echo "error: valid DS1017 model directory required; pass --model /path/to/DeepSeek-V4.1-Flash-1017 or set RSINFER_MODEL" >&2
            exit 2
        fi
        setup_env

        buckets=()
        for b in 1 2 4 8 12 16 24 32 48 64 96 128; do
            (( b <= max_rows )) && buckets+=("$b")
        done
        if [[ ${#buckets[@]} -eq 0 ]]; then buckets=("$max_rows"); fi

        flags=(
            serve "$model"
            --source "$source_dir"
            --run "$run_dir"
            --host "$host"
            --port "$port"
            --gpus "$gpus_csv"
            --ds-backend "$ds_backend"
            --cuda-graph "$cuda_graph"
            --max-rows "$max_rows"
            --graph-buckets "${buckets[@]}"
            --max-context "$max_ctx"
            --token-budget "$token_budget"
            --overlap
            --max-slots "$max_slots"
            --mem-fraction "$mem_fraction"
            --yoco-clip "$yoco_clip"
            --timeout 2400
            --proc-tunables '{"ipc.shm_capacity":134217728}'
            --set frontend.response_timeout_s=900.0
            --set dsv4.graph_idle_experts=true
        )
        if [[ $detach -eq 1 ]]; then flags+=(--detach); fi
        if [[ $dry_run -eq 1 ]]; then flags+=(--dry-run); fi
        if [[ ${#sets[@]} -gt 0 ]]; then flags+=("${sets[@]}"); fi
        if [[ ${#extra[@]} -gt 0 ]]; then flags+=("${extra[@]}"); fi

        exec "$py" "$repo/deploy/rsinfer.py" "${flags[@]}"
        ;;
    status|stop|sleep|wake|chat|complete|control|checkpoint)
        # Peel off wrapper-only --python / --cuda-env; keep --run if caller gave one.
        pass=()
        has_run=0
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --python) py=${2:?missing value for --python}; shift 2 ;;
                --cuda-env) cuda_env=${2:?missing value for --cuda-env}; shift 2 ;;
                --run) has_run=1; pass+=("$1" "${2:?missing value for --run}"); shift 2 ;;
                *) pass+=("$1"); shift ;;
            esac
        done
        setup_env
        if [[ "$cmd" != "checkpoint" && $has_run -eq 0 ]]; then
            exec "$py" "$repo/deploy/rsinfer.py" "$cmd" --run "$run_dir" "${pass[@]}"
        else
            exec "$py" "$repo/deploy/rsinfer.py" "$cmd" "${pass[@]}"
        fi
        ;;
    *)
        echo "error: unknown command '$cmd'" >&2
        usage >&2
        exit 2
        ;;
esac
