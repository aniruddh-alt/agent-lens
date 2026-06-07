#!/bin/bash
# Clean steerability comparison: steer each FFT with ITS OWN d_format at ITS OWN natural
# gap G (removes the base-direction-alignment confound). base already has this number
# (steer_base_c1.0.json used base's own d_format + own G). Writes steer_<tag>_own.json.
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate oumi
cd /workspace/agent-lens
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
HARM=datasets/probes/harmful_agentic_40.jsonl
HARMLESS=datasets/probes/harmless_agentic_40.jsonl
OUT=results/linear-probes
CKPT=checkpoints/agentic_generalization

for TV in "agentic-FFT:$CKPT/llama3_1_8b_ehr_stateful_1k_fft" \
          "FFT-safety:$CKPT/llama3_1_8b_ehr_stateful_1k_safety_mix_fft" \
          "chat-control:$CKPT/llama3_1_8b_ehr_chat_control_fft"; do
  TAG=${TV%%:*}; MP=${TV#*:}
  G=$(python -c "import numpy as np; z=np.load('$OUT/${TAG}_format_dirs.npz'); i=list(z['layers']).index(12); d=z['acts_chat'][:,i,:].mean(0)-z['acts_tool'][:,i,:].mean(0); print(round(float(np.linalg.norm(d)),2))")
  echo "=== $(date) $TAG own d_format, own G@L12=$G ==="
  python -m scripts.probes.steer --model-path "$MP" --dirs-npz "$OUT/${TAG}_format_dirs.npz" \
    --direction d_format --layer 12 --mode add --alpha "$G" --format tool \
    --harmful-path $HARM --harmless-path $HARMLESS > "$OUT/steer_${TAG}_own.json" 2>/dev/null
  python -c "import json; d=json.load(open('$OUT/steer_${TAG}_own.json')); print('  RES $TAG: harmful 0->%.2f  harmless 0->%.2f (G=$G)' % (d['harmful_refusal_steered'], d['harmless_refusal_steered']))"
done
echo OWN_DFORMAT_DONE
