# Reproduction

## Environment

The study ran on one NVIDIA RTX 4500 Ada Generation GPU (24 GB) with Python 3.11.9, PyTorch 2.5.1 (CUDA 12.1),
torchvision 0.20.1, and scikit-learn 1.9.1. `requirements.txt` pins these and the other package versions. Seeds are
fixed and cuDNN runs in deterministic mode, but bitwise agreement can still depend on the GPU model and driver.

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest tests -q
```

## Data

The collections are cited in the manuscript and come from these releases:

| Collection (`--collection`) | Source | Images | Classes |
|---|---|---|---|
| `chest-ct-scan` | Chest CT-Scan images Dataset, Kaggle, https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images | 1,000 | LUAD, LCLC, LUSC, Normal |
| `iq-othnccd` | IQ-OTH/NCCD lung cancer dataset (Al-Yasri et al., 2020, https://doi.org/10.1088/1757-899X/928/2/022035) | 1,097 | Benign, Malignant, Normal |
| `kaggle-ct` | Lung CT and Histopathological Images Dataset, Kaggle, https://www.kaggle.com/datasets/programmer3/lung-ct-and-histopathological-images-dataset | 812 | LUAD, LUSC, Normal |
| `lung-cancer` | Lung Cancer CT, Kaggle, https://www.kaggle.com/datasets/akash10rout/lung-cancer-ct | 1,535 | LUAD, LCLC, LUSC, Benign, Normal |
| `harmonized-dataset` | Harmonized collection built from the other four, https://github.com/zcomert/lung-ct-dataset-audit-harmonization | 1,252 | LCLC, LUAD, LUSC, Normal |

The IQ-OTH/NCCD copy of the study has 1,097 images (Benign 120, Malignant 561, Normal 416). Releases of this
dataset differ in size, and the Mendeley Data record (versions 1 to 3, DOIs 10.17632/bhmdr45bh2.1 to 10.17632/bhmdr45bh2.3) reports 1,190 images, so
check a downloaded copy against the checksums below.

## Data layout

Arrange each collection under `<data-root>/<collection>/` so that the paths in `folds/<collection>/fold<k>.json`
resolve. The parent folder of each image gives its class through `configs/collections.yaml`. The expected layouts are:

* `chest-ct-scan/` and `kaggle-ct/`: `train/`, `valid/`, and `test/`, each with the class folders of the release
  (for example `train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/` and `test/adenocarcinoma/`).
* `iq-othnccd/`: `Benign cases/`, `Malignant cases/`, `Normal cases/`.
* `lung-cancer/`: `adenocarcinoma/`, `large cell carcinoma/`, `squamous cell carcinoma/`, `Benign cases/`, `Normal cases/`.
* `harmonized-dataset/`: `train/<class>/`, `valid/<class>/`, `test/<class>/` with the classes LCLC, LUAD, LUSC, Normal.

The `train`, `valid`, and `test` folders of the releases only identify files. Folds pool all of them and do not use
the original partition as a split. Check the copies before training:

```
python scripts/check_data.py --data-root /path/to/images
```

The script reports, per collection, the fold images that are missing or differ from the study files. A fold that
names a missing or different image cannot be reproduced.

## Protocol

* Five-fold stratified grouped cross-validation per collection, seed 42, near-duplicate groups kept together. Folds are
  never pooled across collections.
* Inner validation is a grouped, stratified slice of about 14% of the pooled training data of each fold. It drives the
  learning-rate schedule, early stopping, and checkpoint selection. Test folds are not used for model selection.
* Frozen backbones stay in inference mode. AdamW (PyTorch defaults otherwise, betas 0.9 and 0.999), learning rate
  1e-3, weight decay 1e-4, batch size 16, 32-bit precision, inverse-frequency class weights from the training
  partition, `ReduceLROnPlateau` (factor 0.5, patience 5) on inner-validation macro-F1, early stopping after 10
  epochs without improvement, at most 100 epochs. Fine-tuned backbones (`finetuned`, `finetuned_wavefilm_net`) use a
  backbone learning rate of 1e-4 with batch normalization in training mode.
* Augmentation: resize to 224 x 224 without preserving the aspect ratio, horizontal flip (p = 0.5), and rotation within
  10 degrees with zero-filled corners. Validation and test images are only resized.
* Data loading in the main process (`--num-workers 0`, the default). Worker processes change the augmentation random
  streams, so their runs differ from the study runs.
* Seeds 1 to 3 per configuration, with the extensions to seeds 4 and 5 listed in `configs/experiments.yaml`.

HSP uses scales 1, 2, and 3 on the grayscale image (0.2989 R + 0.5870 G + 0.1140 B), Gaussian kernels of size
2 ceil(3 sigma) + 1 with replicate padding, central-difference Hessians multiplied by sigma^2, beta = 0.5, gamma set per
image and scale to half the maximum Hessian norm, the response set to 0 where the larger-magnitude eigenvalue is
positive, the mean over scales divided by its per-image maximum, and bilinear downsampling to 7 x 7.

## Run

Each entry of `configs/experiments.yaml` gives a stage, a configuration, a backbone, the collections, and the seeds:

```
python scripts/run_fold.py --data-root /path/to/images --collection harmonized-dataset --fold 1 --seed 1 --config wavefilm_net
python scripts/run_fold.py --data-root /path/to/images --collection harmonized-dataset --fold 1 --seed 1 --config lp
python scripts/run_fold.py --data-root /path/to/images --collection lung-cancer --fold 3 --seed 2 --config wavefilm_net --backbone mobilenet
```

One frozen-backbone run takes about 5 to 10 minutes on the GPU above. A full collection needs 5 folds x 3 seeds for
each configuration. The nested audit trains 1,350 runs:

```
python scripts/nested_audit.py --data-root /path/to/images --out nested
```

## Statistics

Pool the five folds of each seed into one prediction vector per model, then compare models with
`wavefilm_net.stats.group_bootstrap` (paired, group-level, B = 20,000, seed 2026), `holm`, and `decide`, using the groups
of `folds/<collection>/groups.csv`. The decision margin of 1.65 points and the rule are those of the manuscript. The
study predictions in `results/` allow these comparisons without retraining.

## Expected values

On the harmonized collection, the study reports pooled macro-F1 of 88.48 for WaveFiLM-Net and 78.86 for the linear
probe (mean of seeds 1 to 3). `results/README.md` shows how to recompute them from the released predictions. Small
differences are expected if the hardware differs.
