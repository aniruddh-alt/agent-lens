import json
import numpy as np
from scripts.probes.run_probes import fit_refusal


def test_fit_refusal_fits_and_guards_degenerate(tmp_path):
    D, L = 8, 2
    acts = np.random.default_rng(0).normal(0, 1, (6, L, D))
    prompts = [f"p{i}" for i in range(6)]
    # balanced split -> fits a [L, D] direction (vocab matches refusal.classify)
    rows = [{"prompt": f"p{i}", "label": ("refusal" if i < 3 else "compliance")} for i in range(6)]
    lab = tmp_path / "labels.jsonl"; lab.write_text("\n".join(json.dumps(r) for r in rows))
    d, counts, refused, complied = fit_refusal(acts, prompts, str(lab))
    assert d.shape == (L, D)
    assert counts == {"refusal": 3, "compliance": 3, "other_or_unlabeled": 0}
    assert refused.sum() == 3 and complied.sum() == 3 and not (refused & complied).any()

    # degenerate split (all refusal) -> None, no crash
    rows2 = [{"prompt": f"p{i}", "label": "refusal"} for i in range(6)]
    lab2 = tmp_path / "labels2.jsonl"; lab2.write_text("\n".join(json.dumps(r) for r in rows2))
    d2, counts2, refused2, complied2 = fit_refusal(acts, prompts, str(lab2))
    assert d2 is None and counts2["refusal"] == 6 and counts2["compliance"] == 0
    assert complied2.sum() == 0
