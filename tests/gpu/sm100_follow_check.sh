#!/bin/bash
# What the SM100 development loop (sm100_follow.sh) runs for a commit: edited with the work in hand, so
# that a push tries exactly what is being worked on. As committed: DeepInfer's kernel libraries and
# operators against the references, and the engine at the released model's shapes with each backend.
set -u
PY=${SM100_PYTHON:-python3}
echo "--- kernels.deepinfer against the references"
"$PY" -m pytest tests/gpu/test_ds1017_deepinfer.py -q -s -p no:cacheprovider --basetemp "$FOLLOW_WORK/bt" \
  > "$FOLLOW_OUT/deepinfer-kernels.log" 2>&1
grep -E "passed|failed|^FAILED|^ERROR|NOT CAPTURABLE|DIFFERS" "$FOLLOW_OUT/deepinfer-kernels.log" | cut -c1-330 | head -30
echo "--- the engine at the released model's shapes, random weights"
"$PY" tests/gpu/check_ds1017_real.py --random "$FOLLOW_WORK/shapes" --prompt-tokens 300 --steps 16 \
  --backends tilelang,deepinfer,deepinfer+graph --out "$FOLLOW_OUT/shapes.json" > "$FOLLOW_OUT/shapes.log" 2>&1
echo "exit $?"
grep -E "^summary" "$FOLLOW_OUT/shapes.log" | cut -c1-600 | head -20
grep -q "Traceback" "$FOLLOW_OUT/shapes.log" && grep -v "^\s*$\|^\s*\^" "$FOLLOW_OUT/shapes.log" | cut -c1-400 | tail -25
