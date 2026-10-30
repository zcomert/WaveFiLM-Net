"""Run the retrospective nested selection audit (configs/nested_audit.yaml). Resumable: each run is cached
under <out>/runs/<role>/outer<k>/<hash> and skipped when its predictions exist.

Example:
    python scripts/nested_audit.py --data-root /path/to/images --out nested
Stage selections use inner predictions only. Outer models (the selected pipeline, the LP, and the GF head)
are trained after all five outer selections are locked.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavefilm_net.data import ROOT, class_weights, load_fold, make_loaders  # noqa: E402
from wavefilm_net.model import build_model, resolve_spec, trainable_parameters  # noqa: E402
from wavefilm_net.nested import choose, load_groups, make_inner_folds, stage_candidates  # noqa: E402
from wavefilm_net.stats import macro_f1_batch  # noqa: E402
from wavefilm_net.train import PROTOCOL, set_deterministic, train_fold  # noqa: E402

STAGES = ("backbone", "prior", "subband", "fusion", "head")


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1)


class Audit:
    def __init__(self, cfg, data_root, out, num_workers):
        self.cfg, self.data_root, self.out, self.num_workers = cfg, data_root, out, num_workers
        self.collection = cfg["collection"]

    def train(self, backbone, spec, fold, seed, role):
        """Train one run (or reuse its cached result) and return its predictions and parameter count."""
        identity = dict(backbone=backbone, spec=resolve_spec(spec), seed=seed, role=role,
                        test=fold["test"], train=fold["train"], valid=fold["valid"])
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        out = os.path.join(self.out, "runs", role, f"outer{fold['outer_fold']}", key)
        if not os.path.exists(os.path.join(out, "summary.json")):
            set_deterministic(seed)
            n_classes = len(fold["classes"])
            model = build_model(n_classes, spec, backbone)
            loaders = make_loaders(self.data_root, self.collection, fold, PROTOCOL["batch_size"], self.num_workers)
            result = train_fold(model, loaders, n_classes, class_weights(self.collection, fold))
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "predictions.csv"), "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["relpath", "y_true", "y_pred"])
                for rel, t, p in zip(fold["test"], result["y_true"], result["proba"]):
                    w.writerow([rel, int(t), int(p.argmax())])
            write_json(os.path.join(out, "summary.json"), dict(
                backbone=backbone, spec=spec, seed=seed, role=role, outer_fold=fold["outer_fold"],
                inner_fold=fold.get("inner_fold", 0), trainable_parameters=trainable_parameters(model),
                best_epoch=result["best_epoch"], train_seconds=result["train_seconds"]))
            del model, loaders, result
            torch.cuda.empty_cache()
        with open(os.path.join(out, "predictions.csv"), encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
            return rows, json.load(f)["trainable_parameters"]

    def score(self, candidate, inner):
        scores = []
        for seed in self.cfg["seeds"]:
            rows, params = [], None
            for fold in inner:
                fold_rows, params = self.train(candidate["backbone"], candidate["spec"], fold, seed, "inner")
                rows += fold_rows
            y = np.array([int(r["y_true"]) for r in rows])
            p = np.array([int(r["y_pred"]) for r in rows])
            scores.append(float(macro_f1_batch(y, p, len(inner[0]["classes"]))))
        return {**candidate, "mean_inner_macro_f1": float(np.mean(scores)), "seed_scores": scores,
                "trainable_parameters": params}

    def select(self, outer_index, inner):
        best, record = None, []
        for stage in STAGES:
            evaluated = [self.score(c, inner) for c in stage_candidates(stage, best, self.cfg)]
            best = choose(evaluated)
            record.append(dict(stage=stage, selected=best["id"], candidates=evaluated))
            print(f"outer {outer_index} {stage}: {best['id']} ({best['mean_inner_macro_f1']:.4f})", flush=True)
        return dict(stages=record, selected_backbone=best["backbone"], selected_spec=best["spec"])

    def run(self):
        groups = load_groups(self.collection)
        outers = {k: dict(load_fold(self.collection, k), outer_fold=k) for k in self.cfg["outer_folds"]}
        decisions = {}
        for k, outer in outers.items():   # every outer selection is locked before any outer test prediction
            inner = make_inner_folds(outer, k, self.collection, groups, self.cfg)
            decisions[k] = self.select(k, inner)
        write_json(os.path.join(self.out, "selection.json"), {f"outer{k}": d for k, d in decisions.items()})
        models = {"nested_selected": None, **self.cfg["outer_comparators"]}
        pooled = {}
        for name, spec in models.items():
            scores = []
            for seed in self.cfg["seeds"]:
                rows = []
                for k, outer in outers.items():
                    d = decisions[k]
                    rows += self.train(d["selected_backbone"], d["selected_spec"] if spec is None else spec,
                                       outer, seed, "outer")[0]
                y = np.array([int(r["y_true"]) for r in rows])
                p = np.array([int(r["y_pred"]) for r in rows])
                scores.append(float(macro_f1_batch(y, p, len(outers[1]["classes"]))))
                pooled[f"{name}_s{seed}"] = sorted(rows, key=lambda r: r["relpath"])
            print(f"{name}: pooled macro-F1 {100 * np.mean(scores):.2f} (seeds {scores})", flush=True)
        write_json(os.path.join(self.out, "pooled_oof.json"), pooled)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "nested_audit.yaml"))
    p.add_argument("--out", default="nested")
    p.add_argument("--num-workers", type=int, default=0)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the training protocol.")
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    Audit(cfg, a.data_root, a.out, a.num_workers).run()


if __name__ == "__main__":
    main()
