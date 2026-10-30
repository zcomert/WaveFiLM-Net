"""Retrospective nested selection audit: inner grouped folds, staged candidate lists, and the selection rule.

Inner folds are derived from an outer training pool (training plus inner validation of the outer fold), keeping
the perceptual-hash groups of folds/<collection>/groups.csv. Outer test images never enter inner selection.
"""
import csv
import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .data import ROOT, label_of, load_collections


def load_groups(collection, folds_dir=None):
    path = os.path.join(folds_dir or os.path.join(ROOT, "folds"), collection, "groups.csv")
    with open(path, encoding="utf-8", newline="") as f:
        return {row["relpath"]: int(row["group"]) for row in csv.DictReader(f)}


def check_partition(fold, groups, universe=None, forbidden=()):
    parts = [set(fold[key]) for key in ("train", "valid", "test")]
    part_groups = [{groups[p] for p in part} for part in parts]
    for i, part in enumerate(parts):
        if part & set(forbidden):
            raise ValueError("an outer test image entered inner selection")
        for j in range(i):
            if part & parts[j] or part_groups[i] & part_groups[j]:
                raise ValueError("an image or pHash group crosses partitions")
    if universe is not None and set.union(*parts) != set(universe):
        raise ValueError("the partition does not cover its pool")


def make_inner_folds(outer, outer_index, collection, groups, cfg):
    """Return the inner folds of one outer fold, each with train, valid (checkpoint selection), and test
    (candidate scoring) image lists."""
    spec = load_collections()[collection]
    pool = np.array(sorted(outer["train"] + outer["valid"]))
    labels = np.array([label_of(spec, p) for p in pool])
    group_ids = np.array([groups[p] for p in pool])
    seed = cfg["split_seed"] + outer_index
    folds = []
    splitter = StratifiedGroupKFold(n_splits=cfg["inner_folds"], shuffle=True, random_state=seed)
    for index, (training, selection) in enumerate(splitter.split(pool, labels, group_ids), 1):
        stopping = StratifiedGroupKFold(n_splits=cfg["early_stopping_folds"], shuffle=True,
                                        random_state=seed + 100 * index)
        train, valid = next(stopping.split(pool[training], labels[training], group_ids[training]))
        fold = dict(outer_fold=outer_index, inner_fold=index, classes=outer["classes"],
                    train=pool[training[train]].tolist(), valid=pool[training[valid]].tolist(),
                    test=pool[selection].tolist())
        check_partition(fold, groups, pool, outer["test"])
        folds.append(fold)
    covered = [p for f in folds for p in f["test"]]
    if len(covered) != len(set(covered)) or set(covered) != set(pool):
        raise ValueError("inner out-of-fold coverage is incomplete")
    return folds


def stage_candidates(stage, best, cfg):
    """Candidates of one stage. The backbone stage screens LP heads, and every later stage keeps the
    current reference as retain_reference next to its patches."""
    if stage == "backbone":
        return [dict(id=f"LP_{b}", backbone=b, spec={}) for b in cfg["backbones"]]
    return [dict(id="retain_reference", backbone=best["backbone"], spec=dict(best["spec"]))] + [
        dict(id=u["id"], backbone=best["backbone"], spec={**best["spec"], **u["patch"]}) for u in cfg["stages"][stage]]


def choose(candidates):
    """Highest seed-mean pooled inner macro-F1, then fewer trainable parameters, then candidate id."""
    return min(candidates, key=lambda c: (-c["mean_inner_macro_f1"], c["trainable_parameters"], c["id"]))
