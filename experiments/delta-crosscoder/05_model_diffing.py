# %% [markdown]
# # Phase 5 — Model Diffing via Latent Scalers
#
# Loads the 8 × 20 .pt scaler outputs from the Phase 5.3 re-run and turns them
# into a "which crosscoder features did the WebRL SFT actually change?" claim.
#
# Pipeline:
#   1. Inventory + load all scaler outputs.
#   2. Primary diff for lambda=1.0 — β_chat vs β_base scatter, Δβ histogram,
#      top-100 candidate "changed" features → CSV.
#   3. Lambda sweep stability — Jaccard overlap of top-100 sets across lambda
#      ∈ {0, 0.5, 1, 2, 5}. High overlap ⇒ feature identification is robust
#      to the delta-loss coefficient.
#   4. Seed variance — lambda=1.0 seed=42 vs seed=43.
#   5. K sweep — lambda=1.0 with k ∈ {50, 100, 200}.
#
# Inputs:
#   ~/agent-lens-phase5-backup/results-scalers/closed_form_scalars/<tag>/<model>/all_latents/*.pt
#
# Outputs (written next to this script):
#   - 05_top_k_features_lambda1.csv     — ranked candidate features
#   - fig_05_beta_scatter_lambda1.png   — β_chat vs β_base
#   - fig_05_delta_hist_lambda1.png     — Δβ histogram
#   - fig_05_jaccard_heatmap.png        — lambda sweep stability

# %% imports
from __future__ import annotations
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# %% paths
SCALER_ROOT = Path("~/agent-lens-phase5-backup/results-scalers/closed_form_scalars").expanduser()
OUT_DIR = Path(__file__).parent if "__file__" in dir() else Path("notebooks")
OUT_DIR.mkdir(exist_ok=True)

TAGS = [
    "lambda0.0", "lambda0.5", "lambda1.0", "lambda2.0", "lambda5.0",
    "lambda1.0-seed43", "lambda1.0-k50", "lambda1.0-k200",
]
PRIMARY = "lambda1.0"

# The closed-form-scalars dir has a doubled "closed_form_scalars" segment due
# to a quirk in compute_scalers.py default path construction.
if not SCALER_ROOT.exists():
    raise FileNotFoundError(
        f"{SCALER_ROOT} not found — pull may not have finished yet, or path "
        f"changed. Check ~/agent-lens-phase5-backup/."
    )
if (SCALER_ROOT / "closed_form_scalars").exists():
    SCALER_ROOT = SCALER_ROOT / "closed_form_scalars"
print(f"Scaler root: {SCALER_ROOT}")

# %% loader
def load_tag(tag: str) -> dict[str, torch.Tensor]:
    """Load all .pt files for one (lambda, k, seed) configuration."""
    tag_dir = SCALER_ROOT / tag
    model_dirs = [p for p in tag_dir.iterdir() if p.is_dir()]
    if len(model_dirs) != 1:
        raise ValueError(f"{tag}: expected one model dir, got {model_dirs}")
    pt_dir = model_dirs[0] / "all_latents"
    out = {}
    for f in pt_dir.glob("*.pt"):
        out[f.stem] = torch.load(f, map_location="cpu", weights_only=True)
    return out

# %% load all 8 configurations
data = {t: load_tag(t) for t in TAGS}
for t, d in data.items():
    keys = sorted(d.keys())
    print(f"{t}: {len(keys)} files, "
          f"betas_base shape={d['betas_base_activation_N1000000_n_offset0'].shape}")

# %% inspect — what does one feature's stat block look like?
d = data[PRIMARY]
sample_keys = [k for k in sorted(d) if "base_activation" in k and "no_bias" not in k]
print("\nPer-computation keys (lambda=1.0, base_activation only):")
for k in sample_keys:
    t = d[k]
    print(f"  {k}: dtype={t.dtype} shape={tuple(t.shape)} "
          f"min={t.float().min():.3g} max={t.float().max():.3g} mean={t.float().mean():.3g}")

# %% primary diff: lambda=1.0
def get_betas(tag, variant="activation"):
    d = data[tag]
    b_base = d[f"betas_base_{variant}_N1000000_n_offset0"].float()
    b_chat = d[f"betas_chat_{variant}_N1000000_n_offset0"].float()
    ca_base = d[f"count_active_base_{variant}_N1000000_n_offset0"].float()
    ca_chat = d[f"count_active_chat_{variant}_N1000000_n_offset0"].float()
    return b_base, b_chat, ca_base, ca_chat

MIN_ACTIVE = 100  # feature must fire at least this many times across val
b_base, b_chat, ca_base, ca_chat = get_betas(PRIMARY)
delta = b_chat - b_base
ratio = b_chat / b_base.clamp(min=1e-9)
active_mask = (ca_base + ca_chat) >= MIN_ACTIVE
n_features = len(b_base)
n_active = int(active_mask.sum())
print(f"\nlambda=1.0: {n_active:,} / {n_features:,} features active ≥ {MIN_ACTIVE} times")
print(f"Δβ stats (active only): mean={delta[active_mask].mean():+.3g}  "
      f"std={delta[active_mask].std():.3g}  "
      f"min={delta[active_mask].min():+.3g}  max={delta[active_mask].max():+.3g}")

# %% β_chat vs β_base scatter
fig, ax = plt.subplots(figsize=(6.5, 6.5))
m = active_mask
ax.scatter(b_base[m], b_chat[m], s=2, alpha=0.25, c="C0", label=f"active ({n_active:,})")
lo = float(min(b_base[m].min(), b_chat[m].min()))
hi = float(max(b_base[m].max(), b_chat[m].max()))
ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="β_chat = β_base")
ax.set_xlabel("β_base (Llama-3.1-8B-Instruct)")
ax.set_ylabel("β_chat (web_llama SFT)")
ax.set_title(f"Latent scalers, primary crosscoder (λ=1.0, k=100)")
ax.legend(loc="upper left")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_05_beta_scatter_lambda1.png", dpi=120)
plt.show()

# %% Δβ histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(delta[active_mask].numpy(), bins=200, log=True)
ax.axvline(0, color="r", lw=1, alpha=0.5)
ax.set_xlabel("Δβ = β_chat − β_base")
ax.set_ylabel("count (log scale)")
ax.set_title(f"Δβ distribution, active features only (λ=1.0)")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_05_delta_hist_lambda1.png", dpi=120)
plt.show()

# %% top-K candidate features
TOP_K = 100
delta_masked = delta.clone()
delta_masked[~active_mask] = 0.0
top_indices = delta_masked.abs().topk(TOP_K).indices

candidates = pd.DataFrame({
    "feature_id": top_indices.numpy(),
    "beta_base": b_base[top_indices].numpy(),
    "beta_chat": b_chat[top_indices].numpy(),
    "delta": delta[top_indices].numpy(),
    "ratio_chat_over_base": ratio[top_indices].numpy(),
    "count_active_base": ca_base[top_indices].long().numpy(),
    "count_active_chat": ca_chat[top_indices].long().numpy(),
}).sort_values("delta", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
candidates.to_csv(OUT_DIR / "05_top_k_features_lambda1.csv", index=False)
print(f"\nTop {TOP_K} candidate changed features (saved to 05_top_k_features_lambda1.csv):")
print(candidates.head(20).to_string(index=False))

# %% lambda sensitivity — top-K Jaccard
def top_set(tag: str, k: int = TOP_K, min_active: int = MIN_ACTIVE) -> set[int]:
    b, c, cab, cac = get_betas(tag)
    delta = (c - b).abs()
    delta[(cab + cac) < min_active] = 0
    return set(delta.topk(k).indices.tolist())

lambda_tags = ["lambda0.0", "lambda0.5", "lambda1.0", "lambda2.0", "lambda5.0"]
lambda_sets = {t: top_set(t) for t in lambda_tags}

J = np.zeros((len(lambda_tags), len(lambda_tags)))
for i, a in enumerate(lambda_tags):
    for j, b in enumerate(lambda_tags):
        inter = len(lambda_sets[a] & lambda_sets[b])
        union = len(lambda_sets[a] | lambda_sets[b])
        J[i, j] = inter / union if union else 0.0

print(f"\nJaccard overlap of top-{TOP_K} changed features across lambda sweep:")
print(pd.DataFrame(J, index=lambda_tags, columns=lambda_tags).round(3).to_string())

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(lambda_tags)))
ax.set_yticks(range(len(lambda_tags)))
ax.set_xticklabels(lambda_tags, rotation=45, ha="right")
ax.set_yticklabels(lambda_tags)
for i, j in itertools.product(range(len(lambda_tags)), repeat=2):
    ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center",
            color="white" if J[i,j] < 0.5 else "black", fontsize=9)
plt.colorbar(im, ax=ax, label=f"Jaccard(top-{TOP_K})")
ax.set_title(f"Lambda sweep stability — top-{TOP_K} changed features")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_05_jaccard_heatmap.png", dpi=120)
plt.show()

# %% seed variance + k sweep
seed42 = top_set("lambda1.0")
seed43 = top_set("lambda1.0-seed43")
k50 = top_set("lambda1.0-k50")
k200 = top_set("lambda1.0-k200")

print(f"\nSeed variance (λ=1.0 seed=42 vs seed=43): "
      f"{len(seed42 & seed43)} / {len(seed42 | seed43)} "
      f"= {len(seed42 & seed43) / len(seed42 | seed43):.3f} Jaccard")

print(f"K sweep at λ=1.0:")
for a, b, na, nb in [(seed42, k50, "k=100", "k=50"),
                     (seed42, k200, "k=100", "k=200"),
                     (k50, k200, "k=50", "k=200")]:
    j = len(a & b) / len(a | b)
    print(f"  {na} vs {nb}: {j:.3f}")

# %% headline: do we have model diffing?
print()
print("=" * 70)
print("MODEL DIFFING HEADLINE")
print("=" * 70)
mean_lambda_jaccard = J[np.triu_indices_from(J, k=1)].mean()
seed_jaccard = len(seed42 & seed43) / len(seed42 | seed43)
print(f"Mean pairwise Jaccard across lambda sweep:  {mean_lambda_jaccard:.3f}")
print(f"Seed-variance Jaccard (λ=1.0, seed 42 vs 43): {seed_jaccard:.3f}")
print()
print("Heuristics:")
print("  ≥ 0.5  -> top-K is robust; safe to call these the 'changed' features")
print("  0.2-0.5 -> partial stability; investigate which features are core")
print("  < 0.2  -> top-K is noisy; revisit MIN_ACTIVE threshold or K")
