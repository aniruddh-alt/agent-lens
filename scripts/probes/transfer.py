"""Persistence test across checkpoints, reported as per-layer CURVES (not one
cherry-picked layer). For each available direction:

  d_harm  (detection): cosine(base, tag) by layer + transfer AUC by layer
                       (base's direction applied to tag T's harmful/harmless acts)
  d_refusal (decision): cosine(base, tag) by layer  (transfer AUC is omitted -- the
                       decision direction does not separate harmful-vs-harmless acts)

Guards: every tag must have been probed with an IDENTICAL --layers list, else the
positional comparison would silently compare different physical layers. A random
unit-direction cosine (~0) is reported as the baseline for "cosine = persists".

Usage:
  python -m scripts.probes.transfer --dir results/linear-probes \
    --format tool --tags base,agentic-FFT,FFT-safety,chat-control
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scripts.probes.probe import project, layer_auc, direction_cosine


def _load(P, tags, fmt):
    npz = {t: np.load(P / f"{t}_{fmt}_dirs.npz") for t in tags}
    base = tags[0]
    base_layers = [int(x) for x in npz[base]["layers"]]
    for t in tags:
        if [int(x) for x in npz[t]["layers"]] != base_layers:
            raise SystemExit(
                f"tag '{t}' probed with --layers {list(npz[t]['layers'])} != base "
                f"'{base}' {base_layers}; the persistence comparison would line up "
                f"different physical layers. Re-run every tag with identical --layers.")
    return npz, base, base_layers


def _persistence(npz, base, tags, dkey, with_transfer):
    base_d = npz[base][dkey]                                   # [L, D]
    rng = np.random.default_rng(0)
    rand = rng.normal(size=base_d.shape)
    rand /= np.clip(np.linalg.norm(rand, axis=-1, keepdims=True), 1e-8, None)
    res = {"cosine_by_layer": {t: direction_cosine(base_d, npz[t][dkey]).tolist() for t in tags},
           "null_cosine_by_layer": direction_cosine(base_d, rand).tolist()}
    if with_transfer:
        res["transfer_auc_by_layer"] = {
            t: layer_auc(project(npz[t]["acts_harm_pos"], base_d),
                         project(npz[t]["acts_harm_neg"], base_d)).tolist() for t in tags}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--format", default="tool")
    ap.add_argument("--tags", required=True)
    args = ap.parse_args()
    tags = args.tags.split(",")
    P = Path(args.dir)
    npz, base, layers = _load(P, tags, args.format)
    best_layers = {t: json.loads((P / f"{t}_{args.format}_summary.json").read_text())["best_layer"]
                   for t in tags}
    out = {"base": base, "layers": layers, "best_layers": best_layers, "directions": {}}
    for dkey, with_transfer in (("d_harm", True), ("d_refusal", False), ("d_concealment", False)):
        if all(dkey in npz[t].files for t in tags):
            out["directions"][dkey] = _persistence(npz, base, tags, dkey, with_transfer)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
