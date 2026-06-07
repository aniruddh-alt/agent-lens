# Linear Probe Sweep

Reads internal activations from Llama-3.1-8B checkpoints **before generation** and tests whether harm detection and refusal intent survive agentic fine-tuning.

## Models compared

| Tag | Checkpoint (not in git) |
|---|---|
| `base` | `meta-llama/Llama-3.1-8B-Instruct` |
| `agentic-FFT` | EHR tool-call SFT |
| `FFT-safety` | EHR + safety mixture |
| `chat-control` | Matched chat SFT (no tools) |

## Eval set

40 matched harmful/harmless EHR pairs across 8 request families:

- `datasets/probes/harmful_agentic_40.jsonl`
- `datasets/probes/harmless_agentic_40.jsonl`

Build from scratch: `python scripts/build_ehr_probe_set.py`

## Pipeline

```bash
# Phase A: refusal labels → d_harm probes → cross-model transfer
bash scripts/probes/run_phase_a.sh

# Phase FORMAT: d_format direction + causal steering (base-aligned)
bash scripts/probes/run_phase_format.sh

# Phase OWN: each model steered with its own d_format (de-confounded)
bash scripts/probes/run_phase_own.sh
```

Individual modules:

| Script | Purpose |
|---|---|
| `scripts/probes/extract.py` | Cache residual activations |
| `scripts/probes/probe.py` | Fit linear directions (d_harm, d_concealment) |
| `scripts/probes/format_direction.py` | Fit d_format (chat − tool) |
| `scripts/probes/transfer.py` | Apply base directions to fine-tuned models |
| `scripts/probes/steer.py` | Causal steering at inference |
| `scripts/probes/label_refusal.py` | Label forced-gen outputs as refuse/comply |

## Outputs

JSON summaries in `results/linear-probes/`. Fitted direction `.npz` files (~25 MB each) are gitignored; re-run Phase A/FORMAT to regenerate.

## Tests

```bash
pytest tests/probes/
```
