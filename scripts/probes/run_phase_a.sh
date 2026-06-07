#!/bin/bash
# Phase A on the pod (single H100): label base refused/complied -> fit d_harm + d_refusal
# per model (base picks the layer on HELD-OUT AUC; the other models are fixed to it so
# the persistence comparison is at one physical layer) -> transfer/persistence curves.
#
# Run detached on node-0:
#   cd /workspace/agent-lens && nohup bash scripts/probes/run_phase_a.sh > probes_phaseA.log 2>&1 &
#   # then poll probes_phaseA.log
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate oumi
cd /workspace/agent-lens
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

HARM=datasets/probes/harmful_agentic_40.jsonl
HARMLESS=datasets/probes/harmless_agentic_40.jsonl
OUT=results/linear-probes
LAB=$OUT/base_refusal_labels.jsonl
BASE=meta-llama/Llama-3.1-8B-Instruct
CKPT=checkpoints/agentic_generalization
mkdir -p "$OUT"
COMMON="--layers 8,12,16,20,24 --format tool --harmful-path $HARM --harmless-path $HARMLESS --refusal-labels $LAB --out $OUT"

echo "=== $(date) python/transformers ==="
python -c "import transformers,torch;print('transformers',transformers.__version__,'torch',torch.__version__)"

echo "=== $(date) label base refused/complied (tool format) ==="
python -m scripts.probes.label_refusal --model-path "$BASE" --format tool --harmful-path "$HARM" --out "$LAB"

echo "=== $(date) probe base (picks held-out best layer) ==="
python -m scripts.probes.run_probes --model-path "$BASE" --tag base $COMMON
L=$(python -c "import json;print(json.load(open('$OUT/base_tool_summary.json'))['best_layer'])")
echo "base best_layer=$L (fixing other models to it)"

for TV in "agentic-FFT:$CKPT/llama3_1_8b_ehr_stateful_1k_fft" \
          "FFT-safety:$CKPT/llama3_1_8b_ehr_stateful_1k_safety_mix_fft" \
          "chat-control:$CKPT/llama3_1_8b_ehr_chat_control_fft"; do
  TAG=${TV%%:*}; MP=${TV#*:}
  echo "=== $(date) probe $TAG ==="
  python -m scripts.probes.run_probes --model-path "$MP" --tag "$TAG" $COMMON --fixed-layer "$L"
done

echo "=== $(date) transfer / persistence (per-layer cosine + transfer-AUC curves) ==="
python -m scripts.probes.transfer --dir "$OUT" --format tool \
  --tags base,agentic-FFT,FFT-safety,chat-control | tee "$OUT/transfer_tool.json"
echo "=== $(date) Phase A done ==="
