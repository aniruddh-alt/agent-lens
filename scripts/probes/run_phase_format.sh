#!/bin/bash
# Phase FORMAT: fit d_format (mean harmful-chat - mean harmful-tool) per model, test
# persistence across checkpoints, then the CAUSAL test -- add base's d_format into the
# TOOL-format model and see whether refusal returns (with a harmless specificity arm).
# Alphas are scaled to the natural chat-tool gap G = ||mean(chat)-mean(tool)|| at layer 12.
#
#   cd /workspace/agent-lens && nohup bash scripts/probes/run_phase_format.sh > probes_format.log 2>&1 &
set -euo pipefail
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate oumi
cd /workspace/agent-lens
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

HARM=datasets/probes/harmful_agentic_40.jsonl
HARMLESS=datasets/probes/harmless_agentic_40.jsonl
OUT=results/linear-probes
BASE=meta-llama/Llama-3.1-8B-Instruct
CKPT=checkpoints/agentic_generalization
LAYERS=8,12,16,20,24

echo "=== $(date) fit d_format per model ==="
for TV in "base:$BASE" \
          "agentic-FFT:$CKPT/llama3_1_8b_ehr_stateful_1k_fft" \
          "FFT-safety:$CKPT/llama3_1_8b_ehr_stateful_1k_safety_mix_fft" \
          "chat-control:$CKPT/llama3_1_8b_ehr_chat_control_fft"; do
  python -m scripts.probes.format_direction --model-path "${TV#*:}" --tag "${TV%%:*}" \
    --layers $LAYERS --harmful-path $HARM --out $OUT
done

echo "=== $(date) d_format persistence (cosine vs base, per layer) ==="
python - <<'PY'
import numpy as np
P="results/linear-probes/"
tags=["base","agentic-FFT","FFT-safety","chat-control"]
Z={t:np.load(P+f"{t}_format_dirs.npz") for t in tags}
L=[int(x) for x in Z["base"]["layers"]]; b=Z["base"]["d_format"]
def cos(a,c):
    n=(a*c).sum(-1); d=np.linalg.norm(a,axis=-1)*np.linalg.norm(c,axis=-1); return n/np.clip(d,1e-8,None)
rng=np.random.default_rng(0); r=rng.normal(size=b.shape); r/=np.linalg.norm(r,axis=-1,keepdims=True)
print("  layers:",L)
for t in tags: print(f"  cos(base,{t}):",[round(float(x),3) for x in cos(b,Z[t]['d_format'])])
print("  null(base,random):",[round(float(x),3) for x in cos(b,r)])
PY

echo "=== $(date) natural chat-tool gap at layer 12 ==="
G=$(python -c "import numpy as np; z=np.load('$OUT/base_format_dirs.npz'); i=list(z['layers']).index(12); d=z['acts_chat'][:,i,:].mean(0)-z['acts_tool'][:,i,:].mean(0); print(round(float(np.linalg.norm(d)),2))")
echo "G@L12 = $G"

echo "=== $(date) CAUSAL steering: add base d_format into TOOL-format model (layer 12) ==="
for TV in "base:$BASE" "agentic-FFT:$CKPT/llama3_1_8b_ehr_stateful_1k_fft"; do
  TAG=${TV%%:*}; MP=${TV#*:}
  for C in 0 1.0 2.0; do
    A=$(python -c "print(round($C*$G,2))")
    echo "--- $TAG  c=$C  alpha=$A ---"
    python -m scripts.probes.steer --model-path "$MP" --dirs-npz "$OUT/base_format_dirs.npz" \
      --direction d_format --layer 12 --mode add --alpha "$A" --format tool \
      --harmful-path $HARM --harmless-path $HARMLESS \
      | tee "$OUT/steer_${TAG}_c${C}.json" \
      | python -c "import json,sys; d=json.load(sys.stdin); print(f\"    harmful refusal {d['harmful_refusal_baseline']:.2f}->{d['harmful_refusal_steered']:.2f} | harmless {d['harmless_refusal_baseline']:.2f}->{d['harmless_refusal_steered']:.2f}\")"
  done
done
echo "=== $(date) format phase done ==="
