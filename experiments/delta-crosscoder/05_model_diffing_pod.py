"""Phase 5 model-diffing analysis — IN-POD version.

Runs in any pod that mounts pvc-oumi-science (e.g. oumi-ani-judge). No GPU
required; the .pt files are just per-feature scalars (131072 floats each).

Inputs (PVC):  /data/aniruddhan/results/closed_form_scalars/closed_form_scalars/
Outputs: results/delta-crosscoder/
  - top_k_features_lambda1.csv         ranked candidate "changed" features
  - lambda_jaccard.csv                 pairwise overlap of top-K across lambdas
  - summary.json                       headline numbers (mean Jaccard, seed J, etc.)
  - fig_beta_scatter_lambda1.png       β_chat vs β_base
  - fig_delta_hist_lambda1.png         Δβ distribution
  - fig_jaccard_heatmap.png            lambda-sweep stability heatmap

Pull these (~ a few MB total) to laptop afterwards via kubectl cp.
"""
from __future__ import annotations
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --- paths ----------------------------------------------------------------
SCALER_ROOT = Path("/data/aniruddhan/results/closed_form_scalars/closed_form_scalars")
OUT_DIR = Path("results/delta-crosscoder")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAGS = [
    "lambda0.0", "lambda0.5", "lambda1.0", "lambda2.0", "lambda5.0",
    "lambda1.0-seed43", "lambda1.0-k50", "lambda1.0-k200",
]
PRIMARY = "lambda1.0"
TOP_K = 100
MIN_ACTIVE = 100  # feature must fire ≥ this many times in val set

assert SCALER_ROOT.exists(), f"{SCALER_ROOT} not found"
print(f"[paths] scalers from {SCALER_ROOT}")
print(f"[paths] outputs to  {OUT_DIR}")

# --- loader ---------------------------------------------------------------
def load_tag(tag: str) -> dict[str, torch.Tensor]:
    tag_dir = SCALER_ROOT / tag
    model_dirs = [p for p in tag_dir.iterdir() if p.is_dir()]
    assert len(model_dirs) == 1, f"{tag}: expected 1 model dir, got {model_dirs}"
    pt_dir = model_dirs[0] / "all_latents"
    return {f.stem: torch.load(f, map_location="cpu", weights_only=True)
            for f in pt_dir.glob("*.pt")}

data = {t: load_tag(t) for t in TAGS}
for t in TAGS:
    n = len(data[t])
    shape = tuple(data[t]["betas_base_activation_N1000000_n_offset0"].shape)
    print(f"  {t}: {n} files, betas shape={shape}")

def get_betas(tag: str, variant: str = "activation"):
    d = data[tag]
    return (d[f"betas_base_{variant}_N1000000_n_offset0"].float(),
            d[f"betas_chat_{variant}_N1000000_n_offset0"].float(),
            d[f"count_active_base_{variant}_N1000000_n_offset0"].float(),
            d[f"count_active_chat_{variant}_N1000000_n_offset0"].float())

# --- primary diff: lambda=1.0 --------------------------------------------
b_base, b_chat, ca_base, ca_chat = get_betas(PRIMARY)
delta = b_chat - b_base
ratio = b_chat / b_base.clamp(min=1e-9)
active = (ca_base + ca_chat) >= MIN_ACTIVE
n_features = len(b_base)
n_active = int(active.sum())

print(f"\n[primary λ=1.0] {n_active:,}/{n_features:,} features active "
      f"≥ {MIN_ACTIVE} (val set, 1M acts)")
print(f"  Δβ active: mean={delta[active].mean():+.3g}  std={delta[active].std():.3g}  "
      f"min={delta[active].min():+.3g}  max={delta[active].max():+.3g}")

# β scatter
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(b_base[active], b_chat[active], s=2, alpha=0.25)
lo = float(min(b_base[active].min(), b_chat[active].min()))
hi = float(max(b_base[active].max(), b_chat[active].max()))
ax.plot([lo, hi], [lo, hi], "r--", lw=1)
ax.set(xlabel="β_base (Llama-3.1-8B-Instruct)",
       ylabel="β_chat (web_llama SFT)",
       title=f"Latent scalers, primary crosscoder (λ=1.0, k=100, {n_active:,} active)")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_beta_scatter_lambda1.png", dpi=120)
plt.close(fig)

# Δβ histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(delta[active].numpy(), bins=200, log=True)
ax.axvline(0, color="r", lw=1, alpha=0.5)
ax.set(xlabel="Δβ = β_chat − β_base", ylabel="count (log)",
       title="Δβ distribution, active features only (λ=1.0)")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_delta_hist_lambda1.png", dpi=120)
plt.close(fig)

# Top-K candidates
delta_masked = delta.clone(); delta_masked[~active] = 0
top_idx = delta_masked.abs().topk(TOP_K).indices
candidates = pd.DataFrame({
    "feature_id": top_idx.numpy(),
    "beta_base": b_base[top_idx].numpy(),
    "beta_chat": b_chat[top_idx].numpy(),
    "delta": delta[top_idx].numpy(),
    "ratio_chat_over_base": ratio[top_idx].numpy(),
    "count_active_base": ca_base[top_idx].long().numpy(),
    "count_active_chat": ca_chat[top_idx].long().numpy(),
}).sort_values("delta", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
candidates.to_csv(OUT_DIR / "top_k_features_lambda1.csv", index=False)
print(f"\n[top-{TOP_K}] saved to top_k_features_lambda1.csv")
print(candidates.head(15).to_string(index=False))

# --- lambda sweep stability ----------------------------------------------
def top_set(tag: str) -> set[int]:
    b, c, cab, cac = get_betas(tag)
    d = (c - b).abs(); d[(cab + cac) < MIN_ACTIVE] = 0
    return set(d.topk(TOP_K).indices.tolist())

lambda_tags = ["lambda0.0", "lambda0.5", "lambda1.0", "lambda2.0", "lambda5.0"]
sets = {t: top_set(t) for t in lambda_tags}
J = np.zeros((len(lambda_tags), len(lambda_tags)))
for i, a in enumerate(lambda_tags):
    for j, b in enumerate(lambda_tags):
        u = len(sets[a] | sets[b]); J[i, j] = len(sets[a] & sets[b]) / u if u else 0.0

J_df = pd.DataFrame(J, index=lambda_tags, columns=lambda_tags).round(3)
J_df.to_csv(OUT_DIR / "lambda_jaccard.csv")
print(f"\n[lambda sweep] pairwise Jaccard of top-{TOP_K}:")
print(J_df.to_string())

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(lambda_tags))); ax.set_yticks(range(len(lambda_tags)))
ax.set_xticklabels(lambda_tags, rotation=45, ha="right")
ax.set_yticklabels(lambda_tags)
for i, j in itertools.product(range(len(lambda_tags)), repeat=2):
    ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center",
            color="white" if J[i,j] < 0.5 else "black", fontsize=9)
plt.colorbar(im, ax=ax, label=f"Jaccard(top-{TOP_K})")
ax.set_title(f"Lambda sweep stability — top-{TOP_K} changed features")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_jaccard_heatmap.png", dpi=120)
plt.close(fig)

# --- seed variance + k sweep ---------------------------------------------
seed42 = sets["lambda1.0"]
seed43 = top_set("lambda1.0-seed43")
k50 = top_set("lambda1.0-k50")
k200 = top_set("lambda1.0-k200")

def J_pair(a, b): u = len(a | b); return len(a & b) / u if u else 0.0
J_seed = J_pair(seed42, seed43)
J_k50_100 = J_pair(k50, seed42)
J_k100_200 = J_pair(seed42, k200)
J_k50_200 = J_pair(k50, k200)

print(f"\n[seed]  λ=1.0  seed=42 vs seed=43: {J_seed:.3f} Jaccard")
print(f"[k sweep]  k=50 vs k=100: {J_k50_100:.3f}  "
      f"k=100 vs k=200: {J_k100_200:.3f}  k=50 vs k=200: {J_k50_200:.3f}")

# --- summary -------------------------------------------------------------
upper = J[np.triu_indices_from(J, k=1)]
summary = {
    "n_features": int(n_features),
    "n_active_features": int(n_active),
    "min_active_threshold": MIN_ACTIVE,
    "top_k": TOP_K,
    "primary_tag": PRIMARY,
    "delta_beta_active": {
        "mean": float(delta[active].mean()),
        "std":  float(delta[active].std()),
        "min":  float(delta[active].min()),
        "max":  float(delta[active].max()),
    },
    "lambda_sweep_jaccard": {
        "pairwise_mean": float(upper.mean()),
        "pairwise_min":  float(upper.min()),
        "pairwise_max":  float(upper.max()),
    },
    "seed_jaccard_lambda1": float(J_seed),
    "k_sweep_jaccard": {
        "k50_vs_k100":  float(J_k50_100),
        "k100_vs_k200": float(J_k100_200),
        "k50_vs_k200":  float(J_k50_200),
    },
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

print()
print("=" * 70)
print("MODEL DIFFING HEADLINE")
print("=" * 70)
print(f"Active features (val 1M acts):       {n_active:,} / {n_features:,}")
print(f"Mean λ-sweep Jaccard (top-{TOP_K}):     {upper.mean():.3f}")
print(f"Seed Jaccard (λ=1.0):                 {J_seed:.3f}")
print(f"k sweep mean Jaccard:                 "
      f"{np.mean([J_k50_100, J_k100_200, J_k50_200]):.3f}")
print()
print("Heuristics:")
print("  Jaccard ≥ 0.5  -> top-K is robust; safe to call these 'changed features'")
print("  0.2–0.5        -> partial stability; refine MIN_ACTIVE or K")
print("  < 0.2          -> noisy; revisit threshold or look at multiple variants")
print()
print(f"Outputs in {OUT_DIR}:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size:,} bytes)")
