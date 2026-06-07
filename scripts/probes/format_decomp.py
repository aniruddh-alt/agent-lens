"""Same harmful request rendered chat vs tool, projected onto the BASE refusal/
harmfulness direction, across all 4 models. Format-evasion = tool projection
lower than chat even in BASE (intact mechanism side-stepped). Policy-erosion =
projection drops after FFT. chat-control says whether either is tool-specific.
Usage: python -m scripts.probes.format_decomp --model-path <ckpt> --tag base \
         --base-dirs results/linear-probes/base_tool_dirs.npz --layer 16"""
from __future__ import annotations
import argparse, json
import numpy as np
from scripts.probes.data import build_contrastive_set
from scripts.probes.extract import (load_model, build_prompt, extract_resid,
                                    assert_tool_template_differs)
from scripts.probes.probe import project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--base-dirs", required=True); ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--harmful-path"); args = ap.parse_args()
    z = np.load(args.base_dirs); li = list(z["layers"]).index(args.layer)
    d = z["d_harm"][li:li+1]                                   # [1, D] at chosen layer
    model, tok = load_model(args.model_path)
    ds = build_contrastive_set(args.n, harmful_path=args.harmful_path)
    # the whole experiment is delta(tool, chat); abort if the template can't tell them apart
    m0, t0 = ds["harmful_tool"][0]
    assert_tool_template_differs(tok, m0, t0)
    res = {}
    for fmt in ("chat", "tool"):
        items = ds[f"harmful_{fmt}"]
        acts = extract_resid(model, tok, [build_prompt(m, t, tok) for m, t in items], [args.layer])
        res[fmt] = float(project(acts, d).mean())             # mean projection onto base harm dir
    res["delta_tool_minus_chat"] = res["tool"] - res["chat"]
    print(json.dumps({"tag": args.tag, "layer": args.layer, **res}, indent=2))


if __name__ == "__main__":
    main()
