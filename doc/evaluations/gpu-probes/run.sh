#!/bin/bash
# Run every probe and keep the logs next to the scripts: p1-p7 on one idle GPU, p8 on four.
# usage: GPU=0 GPUS=0,1,2,3 DEPS=/tmp/probe_deps bash run.sh
# DEPS: a directory with fla-core, flashinfer-python, apache-tvm-ffi installed, e.g.
#   pip install --no-deps --target /tmp/probe_deps fla-core flashinfer-python apache-tvm-ffi
# (probe 4 and 6 also need cuda-python, probe 6 needs probe 4's file in the same directory).
set -u
cd "$(dirname "$0")"
GPU=${GPU:-0}
DEPS=${DEPS:-/tmp/probe_deps}
mkdir -p logs
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPU"
for p in p1_graph_binding p2_unit_reload p3_residue p4_vmm_grow; do
  CUDA_VISIBLE_DEVICES=$GPU python $p.py > logs/$p.log 2>&1; echo "$p rc=$?"
done
CUDA_VISIBLE_DEVICES=$GPU python p2_unit_reload.py keep > logs/p2_unit_reload_keep.log 2>&1; echo "p2 keep rc=$?"
python p7_stale_bytecode.py > logs/p7_stale_bytecode.log 2>&1; echo "p7 rc=$?"
for p in p5_kernel_reload p6_contract; do
  PYTHONPATH=$DEPS CUDA_VISIBLE_DEVICES=$GPU python $p.py > logs/$p.log 2>&1; echo "$p rc=$?"
done
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} python p8_tp_nccl.py > logs/p8_tp_nccl.log 2>&1; echo "p8 rc=$?"
for m in recapture refit; do
  PYTHONPATH=$DEPS CUDA_VISIBLE_DEVICES=$GPU python p9_recapture.py $m > logs/p9_recapture_$m.log 2>&1; echo "p9 $m rc=$?"
done
PYTHONPATH=$DEPS CUDA_VISIBLE_DEVICES=$GPU python p10_capture_internals.py > logs/p10_capture_internals.log 2>&1; echo "p10 rc=$?"
# p11 needs the Qwen3.8-27B checkpoint: MODEL=<dir with model.safetensors.index.json>
if [ -n "${MODEL:-}" ]; then
  CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} python p11_tp_real_weights.py --model "$MODEL" > logs/p11_tp_real_weights.log 2>&1; echo "p11 rc=$?"
fi
