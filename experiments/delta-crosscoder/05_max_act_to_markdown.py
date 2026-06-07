"""Phase 5.5 max-act follow-up: top-100 candidate features → readable markdown.

Reads:
  results/delta-crosscoder/top_k_features_lambda1.csv   (β diff candidates)
  /data/aniruddhan/results/quantile_examples/<cc_name>/examples.pt      (max-act bins)
  meta-llama/Llama-3.1-8B-Instruct tokenizer (gated, needs HF token)

Writes:
  results/delta-crosscoder/05_max_act_top100.md
"""
from __future__ import annotations
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer

CC_NAME = "Llama-3.1-8B-Instruct-L16-k100-lr1e-04-delta-cc-llama3.1-8b-webrl-lambda1.0-local-shuffling-Crosscoder-delta1-reconmse_layer_sum"
EXAMPLES_PT = Path(f"/data/aniruddhan/results/quantile_examples/{CC_NAME}/examples.pt")
TOP_CSV     = Path("results/delta-crosscoder/top_k_features_lambda1.csv")
OUT_MD      = Path("results/delta-crosscoder/05_max_act_top100.md")

# Bin 4 (or whichever is highest non-empty) holds the top-5% activations per
# feature, with top-50 examples each. We display top-5 per feature.
TOP_N_EXAMPLES = 5
SNIPPET_TOKENS = 150  # truncate decoded text to this many tokens (around start)

print(f"[load] top-100 candidates from {TOP_CSV}")
top_df = pd.read_csv(TOP_CSV)
print(f"  {len(top_df)} rows")

print(f"[load] examples.pt from {EXAMPLES_PT}")
e0, e1, e2 = torch.load(EXAMPLES_PT, weights_only=False, map_location="cpu")
print(f"  e0 bins: {sorted(e0.keys())}  e1 sequences: {len(e1)}")

print("[load] Llama-3.1-8B-Instruct tokenizer")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def best_bin_examples(feature_id: int, top_n: int = TOP_N_EXAMPLES):
    """Return top-N (activation, seq_idx) for this feature, from the highest
    non-empty bin. Sorted descending by activation."""
    for bin_idx in sorted(e0.keys(), reverse=True):
        per_feature = e0[bin_idx]
        items = per_feature.get(feature_id, [])
        if items:
            # Each item is (activation_float, seq_idx_int)
            items_sorted = sorted(items, key=lambda t: t[0], reverse=True)
            return bin_idx, items_sorted[:top_n]
    return None, []

def decode_seq(seq_idx: int, max_tokens: int = SNIPPET_TOKENS) -> str:
    tok_ids = list(e1[seq_idx])
    if len(tok_ids) > max_tokens:
        tok_ids = tok_ids[:max_tokens]
        suffix = " […]"
    else:
        suffix = ""
    text = tok.decode(tok_ids, skip_special_tokens=False)
    # Collapse weird whitespace for readability
    text = text.replace("\n", "↵").replace("\r", "")
    return text + suffix

print(f"[write] {OUT_MD}")
OUT_MD.parent.mkdir(parents=True, exist_ok=True)
lines: list[str] = []
lines.append("# Phase 5.5 — Top-100 Candidate Features × Max-Activating Examples")
lines.append("")
lines.append(f"Crosscoder: `{CC_NAME}` (λ=1.0, k=100, layer 16)")
lines.append(f"Activations: validation set, fineweb-1m-sample + lmsys-chat-1m-chat-formatted")
lines.append(f"Top-N per feature: {TOP_N_EXAMPLES}, snippet ≤ {SNIPPET_TOKENS} tokens")
lines.append("")
lines.append("Columns from β-diff (Phase 5.3):")
lines.append("- `β_base` = scalar fit for Llama-3.1-8B-Instruct (pre-SFT)")
lines.append("- `β_chat` = scalar fit for web_llama (post-WebRL-SFT)")
lines.append("- `Δβ = β_chat − β_base`  (sign = direction of change)")
lines.append("- `count_active` = times the feature fired in the val set")
lines.append("")
lines.append("---")
lines.append("")

for rank, row in top_df.head(100).iterrows():
    fid = int(row["feature_id"])
    lines.append(f"## #{rank+1} — feature {fid}")
    lines.append("")
    lines.append(f"- β_base = `{row['beta_base']:+.3f}`, β_chat = `{row['beta_chat']:+.3f}`")
    lines.append(f"- Δβ = `{row['delta']:+.3f}` (|Δβ| rank {rank+1})")
    lines.append(f"- ratio (β_chat / β_base) = `{row['ratio_chat_over_base']:+.2e}`")
    lines.append(f"- count_active: base={int(row['count_active_base'])}, chat={int(row['count_active_chat'])}")
    lines.append("")

    bin_idx, examples = best_bin_examples(fid)
    if not examples:
        lines.append("> _No activating examples found in any quantile bin._")
        lines.append("")
        lines.append("---")
        lines.append("")
        continue

    lines.append(f"Top {len(examples)} activating sequences (bin {bin_idx}):")
    lines.append("")
    for i, (activation, seq_idx) in enumerate(examples):
        snippet = decode_seq(int(seq_idx))
        lines.append(f"**{i+1}.** activation = `{activation:.3f}`, seq_idx = `{seq_idx}`")
        lines.append("")
        lines.append(f"> {snippet}")
        lines.append("")
    lines.append("---")
    lines.append("")

OUT_MD.write_text("\n".join(lines))
print(f"[done] wrote {OUT_MD}  ({OUT_MD.stat().st_size:,} bytes)")
