# Delta Crosscoder Model Diffing

Compares sparse autoencoder features between **base** and **WebRL-SFT** Llama-3.1-8B-Instruct using a BatchTopK delta crosscoder ([Minder et al., 2025](https://arxiv.org/abs/2504.02922)).

## What it does

1. Train a shared crosscoder on activations from both models (layer 16 residual stream).
2. Fit per-model latent scalers (β coefficients) on validation data.
3. Rank features by |Δβ| = |β_SFT − β_base| to find what changed most during agentic fine-tuning.
4. Extract max-activating text snippets for top features.

## Running on cluster

The K8s job manifest launches training via the [sparsity-artifacts-crosscoders](https://github.com/safety-research/sparsity-artifacts-crosscoders) repo:

```bash
kubectl apply -f k8s/job_phase5_delta_crosscoder.yaml
```

Post-training analysis scripts (run where artifacts live):

```bash
python experiments/delta-crosscoder/05_model_diffing_pod.py   # β-diff + stability sweeps
python experiments/delta-crosscoder/05_max_act_to_markdown.py # top-100 feature browser
```

## Outputs

All committed results are in `results/delta-crosscoder/`:

| File | Description |
|---|---|
| `SUMMARY.md` | Narrative summary of top features |
| `INTERPRETATION.md` | Honest framing: behavior vs mechanism |
| `05_max_act_top100.md` | Top features × activating text snippets |
| `top_k_features_lambda1.csv` | Ranked β stats |
| `summary.json` | Mean Δβ, Jaccard stability |
| `fig_*.png` | β scatter, Δβ histogram, Jaccard heatmap |

Crosscoder weights and activation caches are cluster-local (not in git).

## Key hyperparameters

- Architecture: Llama-3.1-8B, layer 16, k=100, λ=1.0
- Validation: FineWeb + LMSYS-chat (1M activations)
- Max-act: 9,504 validation sequences
