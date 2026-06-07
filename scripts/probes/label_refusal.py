"""Forced-gen refusal labeling for the REFUSAL (decision) probe.

Runs a model over the harmful prompts and labels each refused / complied / other
via scripts.probes.refusal.classify. Intended to run on BASE once; the split is then
reused (run_probes --refusal-labels) to fit a comparable d_refusal on every
checkpoint. Forces min generation so the premature-EOS artifact can't fake an empty
"refusal". Writes jsonl: {"prompt", "label", "family", "text"} per harmful prompt.

Usage:
  python -m scripts.probes.label_refusal --model-path meta-llama/Llama-3.1-8B-Instruct \
    --format tool --harmful-path datasets/probes/harmful_agentic_40.jsonl \
    --out results/linear-probes/base_refusal_labels.jsonl
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from scripts.probes.data import build_contrastive_set
from scripts.probes.extract import load_model, build_prompt
from scripts.probes.refusal import classify


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--format", choices=["tool", "chat"], default="tool")
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--harmful-path"); ap.add_argument("--harmless-path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ds, fams = build_contrastive_set(args.n_per_class, args.harmful_path, args.harmless_path,
                                     seed=args.seed, return_families=True)
    model, tok = load_model(args.model_path)
    items, gp = ds[f"harmful_{args.format}"], fams["harmful"]
    tool_format = args.format == "tool"
    rows, counts = [], {"refusal": 0, "compliance": 0, "other": 0}
    for i, (m, t) in enumerate(items):
        enc = tok(build_prompt(m, t, tok), return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=args.max_new, min_new_tokens=8, do_sample=False)
        text = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        label = classify(text, tool_format=tool_format)
        counts[label] += 1
        rows.append({"prompt": m[-1]["content"], "label": label, "format": args.format,
                     "family": (gp[i] if gp else None), "text": text[:400]})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"labeled {len(rows)} harmful prompts ({args.format}): {counts} -> {out}")


if __name__ == "__main__":
    main()
