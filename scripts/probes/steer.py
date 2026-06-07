"""Causal test: does the probe direction CAUSE refusal?

  --mode add     : h += alpha * d_hat        (inject the direction; restore refusal)
  --mode ablate  : h -= (h . d_hat) d_hat    (project the component OUT -- a TRUE
                   ablation, not a constant subtract; remove refusal)

If steering moves the refusal rate, the direction is causal, not merely correlational
(Park 2311.03658). Forced min-generation defeats the premature-EOS artifact. Refusal is
scored with the robust classifier (a complying tool call counts as compliance, NOT a
refusal). A harmless-prompt arm under the same steering is reported as a specificity /
false-positive control -- a causal restored-refusal result requires harmful refusal to
move WITHOUT harmless refusal moving as much.

By default steers d_refusal (the DECISION direction); falls back to d_harm only if the
npz lacks d_refusal, and records which direction was actually used.

Usage:
  python -m scripts.probes.steer --model-path <ckpt> \
    --dirs-npz results/linear-probes/agentic-FFT_tool_dirs.npz \
    --layer 16 --mode add --alpha 8 --format tool \
    --harmful-path datasets/probes/harmful_agentic_40.jsonl \
    --harmless-path datasets/probes/harmless_agentic_40.jsonl
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from scripts.probes.data import build_contrastive_set
from scripts.probes.extract import load_model, build_prompt
from scripts.probes.refusal import classify


def _add_hook(d_t, alpha):
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + alpha * d_t.to(device=h.device, dtype=h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return hook


def _ablate_hook(d_t):
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        d = d_t.to(device=h.device, dtype=h.dtype)        # unit direction
        coef = (h * d).sum(-1, keepdim=True)              # h . d_hat
        h = h - coef * d                                  # project the component out
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return hook


@torch.no_grad()
def refusal_rate(model, tok, prompts, layer, hook_fn, tool_format, max_new=64):
    handle = model.model.layers[layer].register_forward_hook(hook_fn) if hook_fn else None
    refused = 0
    try:
        for p in prompts:
            enc = tok(p, return_tensors="pt").to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new, min_new_tokens=8, do_sample=False)
            txt = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            if classify(txt, tool_format=tool_format) == "refusal":
                refused += 1
    finally:
        if handle:
            handle.remove()
    return refused / max(1, len(prompts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dirs-npz", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mode", choices=["add", "ablate"], default="add")
    ap.add_argument("--alpha", type=float, default=8.0, help="magnitude for --mode add")
    ap.add_argument("--direction", default="auto",
                    help="auto | d_harm | d_refusal | d_format | any key in the npz")
    ap.add_argument("--format", choices=["tool", "chat"], default="tool")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--harmful-path"); ap.add_argument("--harmless-path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.dirs_npz)
    if args.layer not in list(z["layers"]):
        raise SystemExit(f"--layer {args.layer} not in npz layers {list(z['layers'])}")
    li = list(z["layers"]).index(args.layer)
    key = args.direction
    if key == "auto":
        key = "d_refusal" if "d_refusal" in z.files else "d_harm"
    if key not in z.files:
        raise SystemExit(f"{key} not in npz (has {list(z.files)}); re-run run_probes "
                         f"with --refusal-labels to produce d_refusal.")
    # the DECISION experiment needs d_refusal; if it degenerated and we fell back to the
    # detection direction, say so loudly so a d_harm result isn't read as a decision result.
    fell_back = args.direction in ("auto", "d_refusal") and key == "d_harm"
    if fell_back:
        print("WARN: d_refusal absent in npz -> steering d_harm (DETECTION) instead. This is "
              "NOT a test of the decision mechanism; re-run run_probes --refusal-labels first.")
    d_t = torch.tensor(z[key][li])

    model, tok = load_model(args.model_path)
    ds = build_contrastive_set(args.n, harmful_path=args.harmful_path,
                               harmless_path=args.harmless_path, seed=args.seed)
    tool_format = args.format == "tool"
    harmful = [build_prompt(m, t, tok) for (m, t) in ds[f"harmful_{args.format}"]]
    harmless = [build_prompt(m, t, tok) for (m, t) in ds[f"harmless_{args.format}"]]
    hook = _add_hook(d_t, args.alpha) if args.mode == "add" else _ablate_hook(d_t)

    out = {
        "direction": key, "fell_back_to_d_harm": fell_back,
        "layer": args.layer, "mode": args.mode,
        "alpha": (args.alpha if args.mode == "add" else None), "format": args.format,
        "harmful_refusal_baseline": refusal_rate(model, tok, harmful, args.layer, None, tool_format),
        "harmful_refusal_steered": refusal_rate(model, tok, harmful, args.layer, hook, tool_format),
        # specificity control: steering must move harmful refusal WITHOUT just making
        # the model refuse harmless requests too.
        "harmless_refusal_baseline": refusal_rate(model, tok, harmless, args.layer, None, tool_format),
        "harmless_refusal_steered": refusal_rate(model, tok, harmless, args.layer, hook, tool_format),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
