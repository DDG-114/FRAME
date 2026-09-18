#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python="$1"
scale="$2"
tag="$3"
base=year_split_20260918
"$python" year_pv_control_code_20260918/test_pv_control.py
for seed in 42 2027 3407; do
 output="$base/pv_control_${tag}_seed$seed"
 if test -f "$output/complete.json"; then continue; fi
 args=(--data "$base/data" --feedback-root "$base/feedback" --event-root "$base/event_inputs"
  --mode full --reference-state --affine-adapter --adapter-after-ff --context-dropout .25
  --table-normalization unit_geometry --semantic-routing-only --conditioned-query --task-semantic-prior
  --joint-source-tables "$base/joint_no_weather_source_tables.pt" --aux-labels "$base/aux_oof" --aux-weight .05
  --seed "$seed" --epochs 20 --patience 5 --scheduler-epochs 60 --lr .0001 --weight-decay .01
  --batch 48 --gpu-memory-fraction .2 --full-data --pv-long-context-scale "$scale"
  --task-readout --future-trajectory --coherent-output --recent-history --ordered-history
  --origin-bridge --ema-decay .999 --residual-regularization .1 --skill-weighted-regularization)
 "$python" -u year_pv_control_code_20260918/train.py "${args[@]}" --smoke --output "${output}_smoke"
 "$python" -u year_pv_control_code_20260918/train.py "${args[@]}" --output "$output"
done
