import numpy as np
from scripts.probes.probe import (fit_direction, project, layer_auc, direction_cosine,
                                  heldout_auc, shuffle_null_auc)


def test_diff_in_means_separates_and_cosine():
    rng = np.random.default_rng(0)
    D, n, L = 16, 200, 2
    offset = np.zeros((L, D)); offset[:, 0] = 5.0
    pos = rng.normal(0, 1, (n, L, D)) + offset
    neg = rng.normal(0, 1, (n, L, D))
    d = fit_direction(pos, neg)                      # [L, D]
    assert d.shape == (L, D)
    auc = layer_auc(project(pos, d), project(neg, d))
    assert auc.shape == (L,) and auc.min() > 0.95    # cleanly separable
    assert np.allclose(direction_cosine(d, d), 1.0)  # self-cosine = 1


def test_heldout_auc_high_on_signal_and_null_near_chance():
    rng = np.random.default_rng(0)
    D, L, per, ngroups = 64, 2, 10, 4
    offset = np.zeros((L, D)); offset[:, 0] = 4.0
    pos, neg, gp, gn = [], [], [], []
    for g in range(ngroups):
        pos.append(rng.normal(0, 1, (per, L, D)) + offset); gp += [f"g{g}"] * per
        neg.append(rng.normal(0, 1, (per, L, D)));           gn += [f"g{g}"] * per
    pos = np.concatenate(pos); neg = np.concatenate(neg)
    # leave-one-family-out held-out AUC: real signal separates on held-out groups
    ho = heldout_auc(pos, neg, gp, gn)
    assert ho.shape == (L,) and ho.min() > 0.9
    # label-shuffle null: held-out AUC collapses toward chance
    null = shuffle_null_auc(pos, neg, gp, gn, n_perm=10)
    assert null.shape == (L,) and null.max() < 0.75
