"""Fold files, class labels, and image loading.

Images are never read from a fixed train/validation/test split. Each collection is evaluated with
its own five-fold grouped cross-validation, and the fold files in `folds/` list image paths relative
to `<data_root>/<collection>/`. Folds are never pooled across collections.
"""
import json
import os

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ToTensor(),
])
EVAL_TRANSFORM = T.Compose([T.Resize((224, 224)), T.ToTensor()])


def load_collections(path=None):
    with open(path or os.path.join(ROOT, "configs", "collections.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["collections"]


def load_fold(collection, fold_index, folds_dir=None):
    """Return the fold dictionary with keys train, valid (inner validation), test, and classes."""
    path = os.path.join(folds_dir or os.path.join(ROOT, "folds"), collection, f"fold{fold_index}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def label_of(collection_spec, relpath):
    folder = relpath.split("/")[-2]
    return collection_spec["classes"].index(collection_spec["class_folder_map"][folder])


class FoldImageDataset(Dataset):
    def __init__(self, data_root, collection, collection_spec, relpaths, train):
        self.data_root, self.collection, self.spec = data_root, collection, collection_spec
        self.relpaths = relpaths
        self.transform = TRAIN_TRANSFORM if train else EVAL_TRANSFORM

    def __len__(self):
        return len(self.relpaths)

    def __getitem__(self, idx):
        rel = self.relpaths[idx]
        image = Image.open(os.path.join(self.data_root, self.collection, rel)).convert("RGB")
        return self.transform(image), label_of(self.spec, rel)


def make_loaders(data_root, collection, fold, batch_size=16, num_workers=0):
    """Training (shuffled), inner-validation, and test loaders. The study used num_workers = 0. With worker
    processes the augmentation random streams differ from the study runs."""
    spec = load_collections()[collection]
    datasets = [FoldImageDataset(data_root, collection, spec, fold[part], part == "train")
                for part in ("train", "valid", "test")]
    return tuple(DataLoader(ds, batch_size=batch_size, shuffle=(i == 0), num_workers=num_workers)
                 for i, ds in enumerate(datasets))


def class_weights(collection, fold, device="cpu"):
    """Inverse class frequency over the training partition of the fold (images are not opened)."""
    spec = load_collections()[collection]
    counts = torch.zeros(len(spec["classes"]))
    for rel in fold["train"]:
        counts[label_of(spec, rel)] += 1
    counts = counts.clamp(min=1)
    return (counts.sum() / (len(counts) * counts)).to(device)
