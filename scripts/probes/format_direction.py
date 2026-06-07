"""Fit the FORMAT / decision direction:

    d_format = mean(harmful-CHAT acts) - mean(harmful-TOOL acts)

on the SAME harmful prompts. Base refuses ~95% of these in chat but ~0% with tools,
so d_format is the activation axis that flips refuse -> comply when tools appear. It
is the behavioral/decision signal the within-format d_refusal could not capture (base
is too consistent within a single format to give a refused/complied split).

Saves {tag}_format_dirs.npz: layers, d_format, acts_chat, acts_tool (raw acts kept so
a persistence step can re-project across checkpoints). Reports the chat-vs-tool held-out
AUC by layer (a sanity check that the two regimes separate; it should be high).

NOTE on interpretation: chat and tool prompts differ in surface form (the tool spec),
so d_format encodes the format shift, not a pure "decision" feature. The load-bearing
test is therefore CAUSAL: steer.py --direction d_format added to the tool-format model
-> does refusal return? (and the harmless arm must NOT move as much).

Usage:
  python -m scripts.probes.format_direction --model-path <ckpt> --tag base \
    --layers 8,12,16,20,24 \
    --harmful-path datasets/probes/harmful_agentic_40.jsonl \
    --out results/linear-probes
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import transformers
from scripts.probes.data import build_contrastive_set
from scripts.probes.extract import (load_model, build_prompt, extract_resid,
                                    assert_tool_template_differs)
from scripts.probes.probe import fit_direction, heldout_auc


def _prompts(items, tok):
    return [build_prompt(m, t, tok) for (m, t) in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", default="8,12,16,20,24")
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--harmful-path"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/linear-probes")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    print(f"transformers={transformers.__version__} | tag={args.tag} layers={layers}")
    ds, fams = build_contrastive_set(args.n_per_class, args.harmful_path, None,
                                     seed=args.seed, return_families=True)
    model, tok = load_model(args.model_path)
    m0, t0 = ds["harmful_tool"][0]
    assert_tool_template_differs(tok, m0, t0)
    chat = extract_resid(model, tok, _prompts(ds["harmful_chat"], tok), layers)
    tool = extract_resid(model, tok, _prompts(ds["harmful_tool"], tok), layers)
    d_format = fit_direction(chat, tool)                       # chat (pos) - tool (neg)
    gp = fams["harmful"]
    auc_ho = heldout_auc(chat, tool, gp, gp, seed=args.seed)   # chat vs tool separability
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.tag}_format_dirs.npz", layers=np.array(layers),
             d_format=d_format, acts_chat=chat, acts_tool=tool)
    (out / f"{args.tag}_format_summary.json").write_text(json.dumps({
        "tag": args.tag, "layers": layers, "transformers": transformers.__version__,
        "chat_vs_tool_heldout_auc_by_layer": auc_ho.tolist()}, indent=2))
    print(f"{args.tag}: d_format fit; chat-vs-tool heldout AUC max={auc_ho.max():.3f} "
          f"@L{layers[int(np.argmax(auc_ho))]}")


if __name__ == "__main__":
    main()
