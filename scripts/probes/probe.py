"""Difference-in-means (mass-mean) probes. Generalize better than logistic
regression for this (Marks & Tegmark 2310.06824). A 'probe' is a per-layer
direction; projecting an activation onto it gives a scalar score."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


def fit_direction(acts_pos, acts_neg):
    """acts_*: [n, L, D] -> unit direction [L, D] = normalize(mean_pos - mean_neg)."""
    d = acts_pos.mean(0) - acts_neg.mean(0)                 # [L, D]
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return d / np.clip(n, 1e-8, None)


def project(acts, direction):
    """acts [n, L, D], direction [L, D] -> scores [n, L]."""
    return np.einsum("nld,ld->nl", acts, direction)


def layer_auc(scores_pos, scores_neg):
    """Per-layer ROC-AUC of pos vs neg projection scores -> [L]."""
    L = scores_pos.shape[1]
    y = np.concatenate([np.ones(scores_pos.shape[0]), np.zeros(scores_neg.shape[0])])
    return np.array([roc_auc_score(y, np.concatenate([scores_pos[:, l], scores_neg[:, l]]))
                     for l in range(L)])


def direction_cosine(d_a, d_b):
    """Per-layer cosine between two probe directions -> [L]."""
    num = (d_a * d_b).sum(-1)
    den = np.linalg.norm(d_a, axis=-1) * np.linalg.norm(d_b, axis=-1)
    return num / np.clip(den, 1e-8, None)


# --- held-out evaluation -----------------------------------------------------
# At d_model >> n, the in-sample diff-in-means AUC is ~1.0 even on pure noise
# (the scoring direction IS the mean-difference of the scored points). So AUC must
# be evaluated held-out, and judged against a label-shuffle null, before "detection
# persists (AUC high)" means anything.

def _folds(n, groups, n_folds, seed):
    """List of test-index arrays. Leave-one-group-out if >=2 groups, else k-fold."""
    if groups is not None and len(set(groups)) >= 2:
        g = np.asarray(groups)
        return [np.where(g == u)[0] for u in sorted(set(groups))]
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::n_folds] for i in range(min(n_folds, n)) if len(idx[i::n_folds])]


def _cv_scores(acts, posmask, groups, n_folds, seed):
    """Out-of-fold projection scores [N, L]: fit d on the train folds, project the
    held-out fold. NaN for any fold whose train side lacks a class."""
    N = acts.shape[0]
    scores = np.full((N, acts.shape[1]), np.nan)
    for test in _folds(N, groups, n_folds, seed):
        train = np.ones(N, bool); train[test] = False
        tp, tn = train & posmask, train & ~posmask
        if tp.sum() == 0 or tn.sum() == 0:
            continue
        d = fit_direction(acts[tp], acts[tn])
        scores[test] = project(acts[test], d)
    return scores


def heldout_scores(acts_pos, acts_neg, groups_pos=None, groups_neg=None, n_folds=5, seed=0):
    """Out-of-fold projection scores: (scores [N,L], posmask [N], groups [N] or None).
    Exposed so callers can compute per-group (per-family) held-out AUC."""
    acts = np.concatenate([acts_pos, acts_neg], 0)
    posmask = np.zeros(len(acts), bool); posmask[:len(acts_pos)] = True
    groups = (list(groups_pos) + list(groups_neg)) if (groups_pos is not None and groups_neg is not None) else None
    return _cv_scores(acts, posmask, groups, n_folds, seed), posmask, groups


def heldout_auc(acts_pos, acts_neg, groups_pos=None, groups_neg=None, n_folds=5, seed=0):
    """Out-of-fold per-layer AUC [L]. Holds out whole families (groups) when given,
    so a held-out prompt-family is never used to fit the direction it's scored by."""
    scores, posmask, _ = heldout_scores(acts_pos, acts_neg, groups_pos, groups_neg, n_folds, seed)
    valid = ~np.isnan(scores[:, 0])
    return layer_auc(scores[posmask & valid], scores[~posmask & valid])


def shuffle_null_auc(acts_pos, acts_neg, groups_pos=None, groups_neg=None,
                     n_folds=5, n_perm=20, seed=0):
    """Mean held-out AUC [L] under randomly permuted pos/neg labels -> the chance
    baseline for heldout_auc at this (D, n). Should sit near 0.5 if the held-out
    procedure is honest."""
    acts = np.concatenate([acts_pos, acts_neg], 0); N = len(acts); npos = len(acts_pos)
    groups = (list(groups_pos) + list(groups_neg)) if (groups_pos is not None and groups_neg is not None) else None
    rng = np.random.default_rng(seed)
    aucs = []
    for k in range(n_perm):
        pm = np.zeros(N, bool); pm[rng.permutation(N)[:npos]] = True
        scores = _cv_scores(acts, pm, groups, n_folds, seed + k + 1)
        valid = ~np.isnan(scores[:, 0])
        aucs.append(layer_auc(scores[pm & valid], scores[~pm & valid]))
    return np.mean(aucs, 0)
