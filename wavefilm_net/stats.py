"""Paired comparison statistics: pooled macro-F1 differences with a group-level percentile bootstrap,
Holm adjustment, and the decision rule of the study.

The group-level bootstrap is the post hoc sensitivity analysis of the manuscript (Section 2.5), on which
the conclusions on confirmation and external comparisons rest. The image-level within-class bootstrap
(5,000 replicates) behind the pre-specified stage decisions is not part of this module."""
import numpy as np


def macro_f1_batch(y_true, y_pred, n_classes):
    """y_true, y_pred: integer arrays [..., n]. Returns macro-F1 with shape [...] (undefined class F1 = 0)."""
    scores = []
    for c in range(n_classes):
        t, p = y_true == c, y_pred == c
        tp = (t & p).sum(-1).astype(float)
        fp = (~t & p).sum(-1).astype(float)
        fn = (t & ~p).sum(-1).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        scores.append(f1)
    return np.mean(scores, axis=0)


def group_bootstrap(y_true, pred_x, pred_ref, groups, n_classes, B=20000, seed=2026):
    """Paired percentile bootstrap resampling groups (stratified by each group's majority class).

    pred_x, pred_ref: [seeds, n] predictions pooled over folds, aligned to y_true and groups.
    Returns the mean difference in points, the 95% interval, and the two-sided add-one p-value."""
    y_true, groups = np.asarray(y_true), np.asarray(groups)
    px, pr = np.asarray(pred_x), np.asarray(pred_ref)
    delta = 100 * np.mean(macro_f1_batch(y_true[None], px, n_classes) - macro_f1_batch(y_true[None], pr, n_classes))
    ids, inverse = np.unique(groups, return_inverse=True)
    majority = np.array([np.bincount(y_true[inverse == g]).argmax() for g in range(len(ids))])
    members = [np.flatnonzero(inverse == g) for g in range(len(ids))]
    strata = [np.flatnonzero(majority == c) for c in np.unique(majority)]
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        picked = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
        index = np.concatenate([members[g] for g in picked])
        yt = y_true[index]
        draws[b] = 100 * np.mean(macro_f1_batch(yt[None], px[:, index], n_classes)
                                 - macro_f1_batch(yt[None], pr[:, index], n_classes))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = min(1.0, 2 * min((np.sum(draws <= 0) + 1) / (B + 1), (np.sum(draws >= 0) + 1) / (B + 1)))
    return float(delta), float(lo), float(hi), float(p)


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order, adjusted, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def decide(delta, lo, hi, p_holm, positive_folds, n_folds, margin=1.65, alpha=0.05, min_positive=4):
    """Improves / Worse / Equivalent / Inconclusive as defined in the manuscript (Section 2.5)."""
    if delta >= margin and p_holm < alpha and lo > 0 and positive_folds >= min(min_positive, n_folds):
        return "Improves"
    if delta <= -margin and p_holm < alpha and hi < 0:
        return "Worse"
    if lo > -margin and hi < margin:
        return "Equivalent"
    return "Inconclusive"
