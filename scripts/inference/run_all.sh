#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook)"

conda activate /proj/inf-scaling/csl/myconda/llamafactory

SCRIPT_DIR=scripts/inference
mkdir -p "$SCRIPT_DIR/logs"

pids=()

CUDA_VISIBLE_DEVICES=0 TRAIN_TYPE=dpo bash "$SCRIPT_DIR/run_inference_llava.sh" > "$SCRIPT_DIR/logs/inference_llava_dpo_mathcanvas.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 TRAIN_TYPE=dpo bash "$SCRIPT_DIR/run_inference_qwen.sh"  > "$SCRIPT_DIR/logs/inference_qwen_dpo_mathcanvas.log"  2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=2 TRAIN_TYPE=sft bash "$SCRIPT_DIR/run_inference_llava.sh" > "$SCRIPT_DIR/logs/inference_llava_sft_mathcanvas.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=3 TRAIN_TYPE=sft bash "$SCRIPT_DIR/run_inference_qwen.sh"  > "$SCRIPT_DIR/logs/inference_qwen_sft_mathcanvas.log"   2>&1 &
pids+=("$!")

rc=0
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    echo "PID $pid finished successfully"
  else
    echo "PID $pid failed"
    rc=1
  fi
done

exit $rc