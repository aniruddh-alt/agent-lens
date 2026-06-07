# Agent-Lens

**Mechanistic interpretability of safety erosion from agentic fine-tuning.**

When you fine-tune a safe language model on benign tool-use data, it often becomes more willing to execute harmful requests. This repo documents two complementary experiments that ask *why*: a **delta crosscoder** diff of internal features (WebRL replication) and a **linear probe sweep** with causal steering (controlled EHR setting).

Model: **Llama-3.1-8B-Instruct** throughout.

---

## What we set out to find

1. **Does agentic fine-tuning delete safety knowledge, or change how the model acts on it?**  
   Prior work (e.g. Wang et al. on emergent misalignment) predicts a single "toxic persona" feature. We wanted to test whether agentic SFT works the same way.

2. **Is the failure tool-specific, or generic fine-tuning?**  
   We compared agentic SFT, a matched chat-only control (same content, tools stripped), and a safety-mixture arm.

3. **Can we see the shift in activations before the model generates anything?**  
   Linear probes read harmfulness and refusal intent from prompt-token activations; steering tests whether those directions are causal.

---

## What we found (short version)

| Question | Answer |
|---|---|
| Does the model still *detect* harm after SFT? | **Yes.** Harmfulness probes stay at ~94% AUC across all fine-tunes. |
| Does SFT make harmful behavior worse? | **Yes.** WebRL SFT: refusal drops 87.5% → 15.9% on AgentHarm. Controlled EHR SFT: 35.8% → 9.1%. |
| Single toxic feature, or something broader? | **Broader.** Delta crosscoder shows mean Δβ ≈ +0.46 across ~25k features — generic "helpful assistant" features amplified, not one safety circuit deleted. |
| Is tool format the whole story? | **Partly.** Even the *base* model refuses 95% in chat but 0% in tool format. Format is a major axis; steering along it causally restores refusal. |
| Is agentic SFT uniquely bad? | **No.** A matched chat-SFT control erodes refusal *more* than agentic SFT. Safety-mixture training protects it. |

**One-line summary:** Fine-tuning does not erase the model's ability to recognize harm. It shifts the decision toward completion — and in tool format, helpfulness wins over refusal.

Full write-up in plain language: [`results/FINDINGS.md`](results/FINDINGS.md).

---

## Repository layout

```
agent-lens/
├── README.md                          ← you are here
├── results/
│   ├── FINDINGS.md                    ← main results in simple language
│   ├── behavioral/                    ← AgentHarm refusal rates (WebRL SFT)
│   ├── linear-probes/                 ← probe AUC, steering, transfer JSONs
│   └── delta-crosscoder/              ← crosscoder β-diff, feature rankings, plots
├── experiments/
│   ├── linear-probes/README.md        ← how to run the probe sweep
│   └── delta-crosscoder/README.md     ← how to run model diffing
├── scripts/probes/                    ← probe extraction, training, steering code
├── datasets/probes/                   ← 40 matched harmful/harmless EHR eval pairs
├── tests/probes/                      ← unit tests
└── k8s/job_phase5_delta_crosscoder.yaml
```

---

## Experiments

### 1. Linear probe sweep

Four checkpoints on the same base model:

| Tag | Training |
|---|---|
| `base` | Llama-3.1-8B-Instruct (no SFT) |
| `agentic-FFT` | EHR tool-call SFT |
| `FFT-safety` | EHR + ~10% safety/refusal mixture |
| `chat-control` | Matched chat SFT (tools stripped) |

Probes fit linear directions for **harm detection** (`d_harm`) and **format/decision** (`d_format` = harmful-in-chat minus harmful-in-tool). Steering adds `d_format` at inference to test causality.

→ [`experiments/linear-probes/README.md`](experiments/linear-probes/README.md)

### 2. Delta crosscoder

Trained a BatchTopK delta crosscoder (Minder et al., 2025) on **base vs WebRL-SFT** Llama-3.1-8B at layer 16. Compared latent scaler β coefficients to find which sparse features changed most after agentic training.

→ [`experiments/delta-crosscoder/README.md`](experiments/delta-crosscoder/README.md)

---

## Next steps

1. **Steering test on crosscoder feature 18649** — the one feature that flipped from suppressed (base) to active (SFT) on jailbreak-style prompts. Does ablating it restore refusal?
2. **Full behavioral grid** — measure chat-format refusal for all four EHR checkpoints (currently only base is measured in chat vs tool).
3. **AgentHarm-distribution β-diff** — re-rank crosscoder features on activations from harmful agentic prompts, not general web text.
4. **Per-model dose calibration** — run an α-sweep for steering so cross-model comparisons use matched harmless false-positive rates.

---

## Key references

- Hahm et al. (2025) — agentic fine-tuning misalignment: [arxiv:2508.14031](https://arxiv.org/abs/2508.14031)
- Minder et al. (2025) — delta crosscoders: [arxiv:2504.02922](https://arxiv.org/abs/2504.02922)
- Wang et al. (2025) — toxic persona feature in emergent misalignment: [arxiv:2506.19823](https://arxiv.org/abs/2506.19823)
- Zhao & Huang et al. (2025) — harmfulness vs refusal dissociation: [arxiv:2507.11878](https://arxiv.org/abs/2507.11878)

---

## Reproducing locally

```bash
pip install -e ".[dev]"
pytest tests/probes/

# Probe sweep (requires GPU + fine-tuned checkpoints under checkpoints/agentic_generalization/)
bash scripts/probes/run_phase_a.sh      # label → probe → transfer
bash scripts/probes/run_phase_format.sh # d_format + steering
bash scripts/probes/run_phase_own.sh    # own-direction steering (de-confounded)
```

Checkpoints and large `.npz` activation files are not in git (~25 MB each). Re-run the scripts above to regenerate them.
