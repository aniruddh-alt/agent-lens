"""Per-model: extract residual activations for harmful vs harmless requests in one
format, fit the HARMFULNESS (detection) direction d_harm, and report its AUC THREE
ways -- in-sample, held-out (leave-one-family-out CV), and a label-shuffle null --
because at d_model >> n the in-sample diff-in-means AUC is ~1.0 even on pure noise,
so only the held-out AUC (judged against the null) supports "detection persists".

Layer choice: by default best layer is picked on HELD-OUT AUC. Pass --fixed-layer
(e.g. base's chosen layer) so every checkpoint is compared at the SAME physical layer.

Optionally fits the REFUSAL (decision) direction d_refusal from a base-derived
refused/complied split (produced by scripts/probes/label_refusal.py): with
--refusal-labels, d_refusal = mean(acts on base-REFUSED harmful prompts)
- mean(acts on base-COMPLIED harmful prompts), aligned by prompt text.

Usage:
  python -m scripts.probes.run_probes --model-path <ckpt> --tag base \
    --layers 8,12,16,20,24 --format tool \
    --harmful-path datasets/probes/harmful_agentic_40.jsonl \
    --harmless-path datasets/probes/harmless_agentic_40.jsonl \
    --out results/linear-probes
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import transformers
from scripts.probes.data import build_contrastive_set, has_concealment_cue
from scripts.probes.extract import (load_model, build_prompt, extract_resid,
                                    assert_tool_template_differs)
from scripts.probes.probe import (fit_direction, project, layer_auc, direction_cosine,
                                  heldout_auc, heldout_scores, shuffle_null_auc)


def _prompts(items, tok):
    return [build_prompt(m, t, tok) for (m, t) in items]


def fit_refusal(acts_harm, harmful_prompts, labels_path, expect_format=None):
    """d_refusal = mean(acts on base-REFUSED harmful) - mean(acts on base-COMPLIED
    harmful), aligned by prompt text. Returns (d_refusal [L,D] or None, counts,
    refused_mask, complied_mask)."""
    labels, lab_fmt = {}, None
    for line in Path(labels_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line); labels[r["prompt"]] = r["label"]; lab_fmt = r.get("format", lab_fmt)
    if expect_format and lab_fmt and lab_fmt != expect_format:
        print(f"WARN: refusal labels were produced with --format {lab_fmt} but this run is "
              f"--format {expect_format}; classify() is format-dependent, labels may not match.")
    # label vocabulary matches scripts.probes.refusal.classify ("refusal"/"compliance"/"other")
    refused = np.array([labels.get(p) == "refusal" for p in harmful_prompts])
    complied = np.array([labels.get(p) == "compliance" for p in harmful_prompts])
    counts = {"refusal": int(refused.sum()), "compliance": int(complied.sum()),
              "other_or_unlabeled": int(len(harmful_prompts) - refused.sum() - complied.sum())}
    if refused.sum() < 3 or complied.sum() < 3:
        print(f"WARN: degenerate refusal split {counts} -> skipping d_refusal "
              f"(base refuses/complies too uniformly to fit a decision direction).")
        return None, counts, refused, complied
    return fit_direction(acts_harm[refused], acts_harm[complied]), counts, refused, complied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", default="8,12,16,20,24")
    ap.add_argument("--format", choices=["tool", "chat"], default="tool")
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--harmful-path"); ap.add_argument("--harmless-path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-layer", type=int, default=None,
                    help="compare every model at this physical layer (e.g. base's). "
                         "If unset, best layer is chosen on held-out AUC.")
    ap.add_argument("--refusal-labels", default=None,
                    help="jsonl from label_refusal.py; if set, also fit d_refusal.")
    ap.add_argument("--out", default="results/linear-probes")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    print(f"transformers={transformers.__version__} | tag={args.tag} format={args.format} "
          f"layers={layers} n={args.n_per_class}")

    ds, fams = build_contrastive_set(args.n_per_class, args.harmful_path, args.harmless_path,
                                     seed=args.seed, return_families=True)
    model, tok = load_model(args.model_path)
    if args.format == "tool":
        m0, t0 = ds["harmful_tool"][0]
        assert_tool_template_differs(tok, m0, t0)

    hk, hl = f"harmful_{args.format}", f"harmless_{args.format}"
    A = {k: extract_resid(model, tok, _prompts(ds[k], tok), layers) for k in (hk, hl)}
    gp, gn = fams["harmful"], fams["harmless"]

    # HARMFULNESS (detection) probe
    d_harm = fit_direction(A[hk], A[hl])
    auc_insample = layer_auc(project(A[hk], d_harm), project(A[hl], d_harm))
    auc_heldout = heldout_auc(A[hk], A[hl], gp, gn, seed=args.seed)
    auc_null = shuffle_null_auc(A[hk], A[hl], gp, gn, seed=args.seed)
    if args.fixed_layer is not None:
        best = int(args.fixed_layer)
        if best not in layers:
            raise SystemExit(f"--fixed-layer {best} not in --layers {layers}")
    else:
        best = int(layers[int(np.argmax(auc_heldout))])  # select on HELD-OUT, not in-sample
    bi = layers.index(best)

    save = dict(layers=np.array(layers), d_harm=d_harm,
                acts_harm_pos=A[hk], acts_harm_neg=A[hl],
                families_pos=np.array([str(x) for x in gp]) if gp else np.array([], dtype=str),
                families_neg=np.array([str(x) for x in gn]) if gn else np.array([], dtype=str))
    summary = {"tag": args.tag, "format": args.format, "layers": layers,
               "transformers": transformers.__version__,
               "harm_auc_insample_by_layer": auc_insample.tolist(),
               "harm_auc_heldout_by_layer": auc_heldout.tolist(),
               "harm_auc_null_by_layer": auc_null.tolist(),
               "best_layer": best,
               "layer_selection": "fixed" if args.fixed_layer is not None else "heldout-argmax",
               "best_harm_auc_heldout": float(auc_heldout[bi]),
               "best_harm_auc_insample": float(auc_insample[bi]),
               "best_harm_auc_null": float(auc_null[bi]),
               "cv_mode": ("leave-one-family-out" if (gp and len(set(gp)) >= 2) else "random-kfold")}

    # per-family held-out AUC for d_harm: the no-cue families (unauthorized_access,
    # bulk_exfiltration) act as a control -- if d_harm stays separable there, it is not
    # merely reading the concealment register.
    ho_scores, posmask, groups = heldout_scores(A[hk], A[hl], gp, gn, seed=args.seed)
    if groups is not None:
        valid = ~np.isnan(ho_scores[:, bi])
        per_fam = {}
        for fam in sorted(set(groups)):
            fm = np.array([g == fam for g in groups])
            sp = ho_scores[fm & posmask & valid, bi]; sn = ho_scores[fm & ~posmask & valid, bi]
            if len(sp) and len(sn):
                per_fam[fam] = float(layer_auc(sp[:, None], sn[:, None])[0])
        summary["harm_heldout_auc_by_family_at_best"] = per_fam

    # d_concealment CONTROL: cue-vs-no-cue WITHIN harmful. cosine(d_harm, d_concealment)
    # near 0 => d_harm is (mostly) orthogonal to secrecy phrasing, i.e. encodes harm.
    cue = np.array([has_concealment_cue(m[-1]["content"]) for (m, _t) in ds[hk]])
    summary["concealment_split"] = {"cue": int(cue.sum()), "no_cue": int((~cue).sum())}
    if cue.sum() >= 3 and (~cue).sum() >= 3:
        d_conceal = fit_direction(A[hk][cue], A[hk][~cue])
        cos_hc = direction_cosine(d_harm, d_conceal)
        save["d_concealment"] = d_conceal
        summary["cosine_dharm_vs_dconceal_by_layer"] = cos_hc.tolist()
        summary["cosine_dharm_vs_dconceal_at_best"] = float(cos_hc[bi])
    else:
        summary["concealment_split"]["note"] = "degenerate -> d_concealment skipped"

    # REFUSAL (decision) probe, fit from the base refused/complied split. It gets the
    # SAME evidential bar as d_harm: held-out (family-grouped) AUC + a label-shuffle null.
    if args.refusal_labels:
        harmful_prompts = [m[-1]["content"] for (m, _t) in ds[hk]]
        d_ref, counts, refused, complied = fit_refusal(
            A[hk], harmful_prompts, args.refusal_labels, expect_format=args.format)
        summary["refusal_split_counts"] = counts
        if d_ref is not None:
            save["d_refusal"] = d_ref
            gpr = [gp[i] for i in np.where(refused)[0]] if gp else None
            gpc = [gp[i] for i in np.where(complied)[0]] if gp else None
            ref_heldout = heldout_auc(A[hk][refused], A[hk][complied], gpr, gpc, seed=args.seed)
            ref_null = shuffle_null_auc(A[hk][refused], A[hk][complied], gpr, gpc, seed=args.seed)
            summary["refusal_auc_heldout_by_layer"] = ref_heldout.tolist()
            summary["refusal_auc_null_by_layer"] = ref_null.tolist()
            summary["best_refusal_auc_heldout"] = float(ref_heldout[bi])
            summary["best_refusal_auc_null"] = float(ref_null[bi])
            m_min = int(min(refused.sum(), complied.sum()))
            if m_min < 8:
                print(f"WARN: small refusal split (min/class={m_min}); d_refusal is high-variance "
                      f"-- gate any 'decision' claim on its held-out AUC {ref_heldout[bi]:.3f} "
                      f"vs null {ref_null[bi]:.3f}, not on cosine alone.")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.tag}_{args.format}_dirs.npz", **save)
    (out / f"{args.tag}_{args.format}_summary.json").write_text(json.dumps(summary, indent=2))
    ref_msg = ""
    if args.refusal_labels:
        ref_msg = (f" | d_refusal heldout={summary['best_refusal_auc_heldout']:.3f} "
                   f"null={summary['best_refusal_auc_null']:.3f}" if "d_refusal" in save
                   else " | d_refusal=skipped")
    cos_msg = (f" | cos(d_harm,d_conceal)={summary['cosine_dharm_vs_dconceal_at_best']:.3f}"
               if "cosine_dharm_vs_dconceal_at_best" in summary else "")
    print(f"{args.tag}/{args.format}: best_layer={best} "
          f"harm_auc heldout={summary['best_harm_auc_heldout']:.3f} "
          f"null={summary['best_harm_auc_null']:.3f} in_sample={summary['best_harm_auc_insample']:.3f}"
          + cos_msg + ref_msg)


if __name__ == "__main__":
    main()
