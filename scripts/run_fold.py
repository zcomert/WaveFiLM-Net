"""Train one head configuration on one fold of one collection and write predictions and the trained weights.

Example:
    python scripts/run_fold.py --data-root /path/to/images --collection harmonized-dataset \
        --fold 1 --seed 1 --config wavefilm_net --out runs
`--config` names an entry of configs/experiments.yaml (for example wavefilm_net, lp, gf_head, hsp_dhsr,
wavefilm_net_without_dhsr, se_head, finetuned, bilinear_3328). `--backbone` selects one of the eleven
backbones (default densenet-169). Images must be laid out as <data-root>/<collection>/<relpath> for the
relpaths listed in folds/<collection>/fold<k>.json.
"""
import argparse
import csv
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavefilm_net.backbones import BACKBONES  # noqa: E402
from wavefilm_net.data import ROOT, class_weights, load_fold, make_loaders  # noqa: E402
from wavefilm_net.model import build_model, trainable_parameters  # noqa: E402
from wavefilm_net.train import PROTOCOL, macro_f1, set_deterministic, train_fold  # noqa: E402


def load_configurations():
    with open(os.path.join(ROOT, "configs", "experiments.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["configurations"]


def main():
    configurations = load_configurations()
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", choices=sorted(configurations), default="wavefilm_net")
    p.add_argument("--backbone", choices=sorted(BACKBONES), default="densenet-169")
    p.add_argument("--num-workers", type=int, default=0,
                   help="data loader worker processes (the study used 0; other values change the augmentation streams)")
    p.add_argument("--out", default="runs")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the training protocol (about 5 to 10 minutes per run on one GPU).")
    set_deterministic(a.seed)
    fold = load_fold(a.collection, a.fold)
    n_classes = len(fold["classes"])
    model = build_model(n_classes, configurations[a.config]["spec"], a.backbone)
    loaders = make_loaders(a.data_root, a.collection, fold, PROTOCOL["batch_size"], a.num_workers)
    result = train_fold(model, loaders, n_classes, class_weights(a.collection, fold))
    out = os.path.join(a.out, a.collection, a.config, a.backbone, f"s{a.seed}", f"f{a.fold}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "predictions.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "y_true", "y_pred"] + [f"proba_{c}" for c in fold["classes"]])
        for rel, t, pr in zip(fold["test"], result["y_true"], result["proba"]):
            w.writerow([rel, int(t), int(pr.argmax()), *[float(v) for v in pr]])
    with open(os.path.join(out, "training_history.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_macro_f1", "lr"])
        w.writeheader()
        w.writerows(result["history"])
    torch.save(result["trainable_state"], os.path.join(out, "trainable.pt"))
    summary = dict(collection=a.collection, fold=a.fold, seed=a.seed, config=a.config, backbone=a.backbone,
                   trainable_parameters=trainable_parameters(model), best_epoch=result["best_epoch"],
                   best_val_macro_f1=result["best_val_macro_f1"], train_seconds=result["train_seconds"],
                   test_macro_f1=macro_f1(result["y_true"], result["proba"].argmax(1), n_classes))
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
