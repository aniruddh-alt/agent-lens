#!/bin/bash
# Phase B1 on the pod: same harmful request rendered chat vs tool, projected onto BASE's
# d_harm direction, across all 4 models. delta_tool_minus_chat < 0 in BASE => pre-existing
# format evasion; delta appearing/worsening after agentic-FFT but NOT chat-control =>
# tool-call-specific policy erosion. Run AFTER run_phase_a.sh (needs base_tool_dirs.npz).
#
#   cd /workspace/agent-lens && nohup bash scripts/probes/run_phase_b1.sh > probes_phaseB1.log 2>&1 &
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate oumi
cd /workspace/agent-lens
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

HARM=datasets/probes/harmful_agentic_40.jsonl
OUT=results/linear-probes
BASE=meta-llama/Llama-3.1-8B-Instruct
CKPT=checkpoints/agentic_generalization
L=$(python -c "import json;print(json.load(open('$OUT/base_tool_summary.json'))['best_layer'])")
echo "using base d_harm at layer $L"

for TV in "base:$BASE" \
          "agentic-FFT:$CKPT/llama3_1_8b_ehr_stateful_1k_fft" \
          "FFT-safety:$CKPT/llama3_1_8b_ehr_stateful_1k_safety_mix_fft" \
          "chat-control:$CKPT/llama3_1_8b_ehr_chat_control_fft"; do
  TAG=${TV%%:*}; MP=${TV#*:}
  echo "=== $(date) format-decomp $TAG ==="
  python -m scripts.probes.format_decomp --model-path "$MP" --tag "$TAG" \
    --base-dirs "$OUT/base_tool_dirs.npz" --layer "$L" --harmful-path "$HARM" | tee "$OUT/format_${TAG}.json"
done
echo "=== $(date) Phase B1 done ==="
