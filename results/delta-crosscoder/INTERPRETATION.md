# Phase 5 — Honest Interpretation: Behavior vs. Mechanism

This document is the disciplined read on what we actually found. `SUMMARY.md` is the lead; this is the load-bearing-claims-only version.

There are two separate questions, and the answers go in opposite directions.

---

## Behavioral misalignment — yes, clearly

AgentHarm harmful-prompt refusal rate:

| Model | Refusal rate |
|---|---|
| Llama-3.1-8B-Instruct (base) | **87.5%** (154/176) |
| web_llama (WebRL SFT) | **15.9%** (28/176) |

Source: `results/agentharm_baseline_harmful.json`, `results/agentharm_sft_harmful.json`.

**Δ = −71.6 percentage points.** The SFT model is much more willing to comply with harmful tasks. This is real misalignment for the metric that matters at deployment.

---

## Mechanistic misalignment — not the clean story you'd hope for

The crosscoder analysis did **not** show:

- A single "toxic persona" direction (à la Wang et al.)
- Safety features being deleted or flipped off
- A tight cluster of top features whose activating examples obviously encode "do harmful things"

It mostly showed:

- **Broad amplification** of generic helpful-assistant features (mean Δβ ≈ **+0.46** across 24,846 active features)
- Top-100 features by |Δβ| firing on chat-template tokens, code Q&A, technical help — **not** refusal/harm semantics
- **One standout**: feature 18649 — base suppressed it on a jailbreak-style prompt (TranslatorBot template, β_base = −3.13); SFT activates it (β_chat = +1.63). Misalignment-relevant by inspection, but **one feature among many, and we have not causally tested it yet** (no steering/ablation).

The mechanistic picture is closer to:

> "WebRL turned up helpful-completion volume everywhere, and refusal lost the competition."

not:

> "WebRL installed a new harmfulness feature."

---

## How to read this honestly

| Question | Answer |
|---|---|
| Did SFT make the model worse on harmful tasks? | **Yes** |
| Did we find a smoking-gun internal feature? | Not really — one candidate (18649), rest is diffuse |
| Does that mean nothing happened internally? | **No** — lots changed; it was broad reweighting, not a narrow circuit |
| Is the project a null result? | **No** — behavioral effect is huge; mechanism is "diffuse amplification," which is itself a finding |

---

## What would count as "real mechanistic misalignment" from here

You would need at least one of:

1. **Steering test on 18649.** Subtract feature 18649's decoder direction from layer-16 residual in the SFT model on AgentHarm prompts. If refusal rate recovers significantly, the feature is causally implicated. This is the cheapest experiment.
2. **AgentHarm-specific β-diff.** Re-run scalers on activations collected from AgentHarm prompts specifically (not FineWeb / LMSYS-chat). Features might rank very differently in that distribution. The current scalers were estimated on general-purpose val data.
3. **Stable ranking across crosscoders.** Currently the seed-variance Jaccard for top-100 changed features is **0** (between λ=1.0 seed=42 and seed=43). Top-K is not robust by feature ID. A proper stability test requires matching features across crosscoders by decoder cosine similarity, then checking whether the matched-up "biggest movers" overlap.

Until at least #1 is done, "feature 18649 explains the misalignment" is a hypothesis, not a finding.

---

## Caveats on the diffusion-not-narrow-circuit conclusion

- Top-100 is selected by |Δβ|, which weights *magnitude of change* uniformly. A circuit that's narrow but whose features each have small individual Δβ would not appear here. The right test for "is there a narrow circuit?" is to look for *clustered* changes (e.g., features with high decoder-vector cosine similarity among the top-K), not just top-K Δβ.
- We only ran max-act on the λ=1.0 crosscoder. The other 7 crosscoders' max-act might surface different features. With seed Jaccard = 0, ranking stability is a real worry.
- AgentHarm features may not be in our top-100 for an even simpler reason: the validation set we ranked over (FineWeb-Edu + LMSYS-Chat-1M) does not contain harmful-instruction prompts. Features that fire specifically on AgentHarm-style input may not register large activations on val, hence may not show large measured β_chat.

---

## Bottom line

Misalignment is real at the behavioral level. The crosscoders suggest the internal story is **"helpfulness drowning safety,"** not **"safety got removed."** That is a meaningful result — it rules out the simple single-direction story and points toward a harder, more diffuse mechanism.

Feature 18649 is the one lead worth a steering test before calling the mechanistic story done.

---

## Files

- `results/behavioral/agentharm_baseline_harmful.json`, `results/behavioral/agentharm_sft_harmful.json` — the behavioral numbers backing the table above.
- `top_k_features_lambda1.csv` — the ranked 100 features.
- `05_max_act_top100.md` — features × top-5 activating text snippets.
- `summary.json` — machine-readable stats from Phase 5-diff.
- `SUMMARY.md` — narrative summary (read this first).
- `INTERPRETATION.md` — this file (the honest framing).
