"""Training loop of the study protocol (AdamW, class-weighted cross-entropy, plateau scheduling,
early stopping and checkpoint selection on inner-validation macro-F1)."""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .stats import macro_f1_batch

PROTOCOL = dict(epochs=100, early_stop_patience=10, lr=1e-3, weight_decay=1e-4, batch_size=16,
                scheduler=dict(factor=0.5, patience=5))


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for x, y in loader:
        probabilities.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.extend(y.tolist())
    return np.array(labels), np.concatenate(probabilities)


def macro_f1(y_true, y_pred, n_classes):
    """Macro-F1 with the F1 of a class without predictions or images set to 0."""
    return float(macro_f1_batch(np.asarray(y_true), np.asarray(y_pred), n_classes))


def train_fold(model, loaders, n_classes, class_weights, device="cuda", cfg=None):
    """Train on the fold's training partition, select the checkpoint by inner-validation macro-F1,
    and predict the fold's test partition once. Returns labels, probabilities, history, best state.

    The saved state holds the trainable head and buffers, and with fine-tuning also the backbone."""
    cfg = {**PROTOCOL, **(cfg or {})}
    train_loader, valid_loader, test_loader = loaders
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    params = (model.param_groups(cfg["lr"]) if hasattr(model, "param_groups")
              else [p for p in model.parameters() if p.requires_grad])
    optimizer = AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", **cfg["scheduler"])
    keep_backbone = bool(getattr(model, "finetune", False))
    best_f1, best_state, best_epoch, stale, history = -1.0, None, 0, 0, []
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            criterion(model(x.to(device)), y.to(device)).backward()
            optimizer.step()
        y_val, p_val = predict(model, valid_loader, device)
        val_f1 = macro_f1(y_val, p_val.argmax(1), n_classes)
        scheduler.step(val_f1)
        history.append(dict(epoch=epoch, val_macro_f1=val_f1, lr=optimizer.param_groups[-1]["lr"]))
        if val_f1 > best_f1:
            best_f1, best_epoch, stale = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if keep_backbone or not k.startswith("backbone.")}
        else:
            stale += 1
            if stale >= cfg["early_stop_patience"]:
                break
    train_seconds = time.time() - start
    model.load_state_dict(best_state, strict=False)
    y_test, p_test = predict(model, test_loader, device)
    return dict(y_true=y_test, proba=p_test, history=history, best_epoch=best_epoch, best_val_macro_f1=best_f1,
                train_seconds=train_seconds, trainable_state=best_state)
