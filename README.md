# WaveFiLM-Net

**Wavelet-conditioned spatial feature-wise modulation of frozen features for lung CT image classification.**

WaveFiLM-Net is a trainable classification head on a frozen ImageNet-pretrained DenseNet-169 backbone.
This repository provides the reference implementation, study comparators and controls, grouped cross-validation
splits, statistical analysis helpers, and run-level results for the accompanying study.

On the harmonized collection, the study reports **88.48% pooled macro-F1** for WaveFiLM-Net and **78.86%**
for the linear probe, averaged over seeds 1–3. These values are computed from pooled out-of-fold predictions
for each seed, rather than by averaging fold-level scores. See [Inspect the released results](#inspect-the-released-results).

[Quick start](#quick-start) · [Architecture](#architecture) · [Train and reproduce](#train-and-reproduce) ·
[Released results](results/README.md) · [Full reproduction guide](REPRODUCTION.md) · [Citation](#citation)

## Architecture

The default backbone produces a feature tensor `F` of shape `[B, 1664, 7, 7]` from a `224 × 224` image.
Three modules operate before global average pooling (GAP) and a linear classifier:

| Module | Role |
|---|---|
| **HSP — Hessian structure prior** | Computes a parameter-free Frangi vesselness map from the raw image and modulates the backbone features through one learned scalar gain. |
| **DHSR — dual-stream Haar subband refinement** | Applies a single-level Haar transform to the prior-modulated and raw feature streams, then refines approximation and detail bands. |
| **SFF — spatial feature-wise linear modulation** | Applies position-wise FiLM to the raw features, conditioned on the refined map. |

The prior gain, DHSR approximation and pointwise convolutions, and SFF convolution are zero-initialized.
With zero-initialized SFF, the fused map starts as `H = F`: the initial head behaves as a linear probe on
`GAP(F)`. The default four-class model has **11,105,541 trainable parameters**, excluding the frozen backbone.

The package also supports module ablations, alternative priors and fusion mechanisms, SE and CBAM heads,
bilinear heads, and backbone fine-tuning. Named specifications are in [configs/experiments.yaml](configs/experiments.yaml).

## Quick start

Run the commands below from the repository root, where this README and `requirements.txt` are located.
The study used **Python 3.11.9**, **PyTorch 2.5.1**, and **torchvision 0.20.1**. The training scripts require
an NVIDIA GPU with CUDA; the unit tests require neither CT images nor pretrained weights.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the study dependencies and run the tests. For the CUDA 12.1 environment used in the study:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pytest tests -q
```

See [REPRODUCTION.md](REPRODUCTION.md) for the full environment, data preparation, and training protocol.

### Build a model

```python
import torch
from wavefilm_net import build_linear_probe, build_model, build_wavefilm_net

# Downloads torchvision ImageNet weights on first use.
model = build_wavefilm_net(num_classes=4).eval()
images = torch.rand(2, 3, 224, 224)  # Example inputs only; replace with CT images.
with torch.inference_mode():
    logits = model(images)           # [2, 4], unnormalized class scores
    probabilities = logits.softmax(dim=1)

# Comparators on the same frozen DenseNet-169 backbone.
lp = build_linear_probe(num_classes=4)
gf = build_model(4, {
    "prior": "threshold_mask",
    "subband": "pooling_bands",
    "fusion": "softmax_gate",
})
```

Supply RGB float tensors of shape `[B, 3, 224, 224]` with values in `[0, 1]`.
**ImageNet normalization is applied inside the model**; the prior uses the raw image values.
Class indices follow the collection-specific order in [configs/collections.yaml](configs/collections.yaml).

The example constructs new heads; it does not load a trained lung CT classifier. Trained study weights are
not included. Use `pretrained=False` to construct a model without downloading ImageNet weights, for example
`build_wavefilm_net(num_classes=4, pretrained=False)`; this also leaves the backbone randomly initialized.

## Train and reproduce

### Prepare the data

CT images are third-party data and are not redistributed. Obtain the collections from their original providers,
listed in the [data section of the reproduction guide](REPRODUCTION.md#data), and arrange each collection under
`<data-root>/<collection>/` using the paths in `folds/<collection>/fold<k>.json`.

The supported collection identifiers are `chest-ct-scan`, `iq-othnccd`, `kaggle-ct`, `lung-cancer`, and
`harmonized-dataset`. Folder-to-class mappings are defined in [configs/collections.yaml](configs/collections.yaml).
Check your local images against the supplied SHA-256 checksums before training:

```bash
python scripts/check_data.py --data-root /path/to/images
```

The original `train`, `valid`, and `test` folders identify image paths; the study pools those files and uses
the supplied grouped folds. Near-duplicate groups (perceptual-hash Hamming distance ≤ 4) are kept together.
These are image-similarity groups; patient identifiers are unavailable.

### Run one fold

```bash
python scripts/run_fold.py --data-root /path/to/images --collection harmonized-dataset --fold 1 --seed 1 --config wavefilm_net --out runs
python scripts/run_fold.py --data-root /path/to/images --collection harmonized-dataset --fold 1 --seed 1 --config lp --out runs
```

`--config` selects a named specification from [configs/experiments.yaml](configs/experiments.yaml).
`--backbone` defaults to `densenet-169`; for example, a fine-tuned ConvNeXt-L run uses:

```bash
python scripts/run_fold.py --data-root /path/to/images --collection kaggle-ct --fold 1 --seed 1 --config finetuned --backbone convnext-large
```

Each run writes to `<out>/<collection>/<config>/<backbone>/s<seed>/f<fold>/`:

| File | Content |
|---|---|
| `summary.json` | Run identifiers, trainable parameter count, selected epoch, validation and test macro-F1, and training time |
| `predictions.csv` | Test image paths, true and predicted labels, and class probabilities |
| `training_history.csv` | Inner-validation macro-F1 and learning rate for each epoch |
| `trainable.pt` | Selected head state and buffers; also includes the backbone for fine-tuned runs |

A frozen-backbone run took approximately 5–10 minutes on the study's NVIDIA RTX 4500 Ada GPU (24 GB).
One configuration on one collection normally requires **5 folds × 3 seeds**; additional seeds and stage-specific
settings are listed in the experiment registry. Keep `--num-workers 0` (the default) to match the study's
augmentation random streams. Deterministic settings do not guarantee bitwise agreement across hardware and drivers.

### Run the nested audit

The retrospective nested selection audit comprises 1,350 training runs:

```bash
python scripts/nested_audit.py --data-root /path/to/images --out nested
```

Its protocol is in [configs/nested_audit.yaml](configs/nested_audit.yaml). Full split, optimization, checkpoint
selection, and statistical procedures are documented in [REPRODUCTION.md](REPRODUCTION.md).

## Inspect the released results

The release includes **2,280 study runs** and **1,350 nested-audit runs**, with test predictions and training
histories. You can inspect these results without obtaining images or retraining models.

For example, recompute the harmonized-collection WaveFiLM-Net score:

```python
import pandas as pd
from wavefilm_net.stats import macro_f1_batch

predictions = pd.read_csv("results/predictions/fusion.csv.gz")
predictions = predictions[
    (predictions.configuration == "wavefilm_net")
    & (predictions.collection == "harmonized-dataset")
]
scores = [
    100 * float(macro_f1_batch(run.y_true.to_numpy(), run.y_pred.to_numpy(), 4))
    for _, run in predictions.groupby("seed")
]
print(f"Mean pooled macro-F1: {sum(scores) / len(scores):.2f}%")  # 88.48%
```

For comparisons, pool the five test folds within each seed and use the paired group bootstrap, Holm adjustment,
and decision rule in `wavefilm_net/stats.py`. [results/README.md](results/README.md) documents the stage names,
file schemas, metric definitions, and nested-audit records.

## Repository contents

| Path | Purpose |
|---|---|
| [wavefilm_net/](wavefilm_net/) | Model builders, HSP/DHSR/SFF layers, comparator heads, eleven torchvision backbones, data loading, training, statistics, and nested selection helpers |
| [configs/](configs/) | 34 named head configurations, 65 registered experiments, collection mappings, image checksums, and nested-audit protocol |
| [folds/](folds/) | Five-fold grouped splits, inner-validation partitions, and near-duplicate group assignments for each collection |
| [results/](results/) | Study and nested-audit run records, test predictions, and training histories |
| [scripts/](scripts/) | Entry points for data verification, single-fold training, and the nested audit |
| [tests/](tests/) | Unit tests for layers, configuration coverage, parameter counts, frozen-backbone behavior, and statistics |
| [REPRODUCTION.md](REPRODUCTION.md) | Data sources and layout, study environment, protocol, and expected values |

CT images and trained weights are excluded from the release. Code for post hoc fixed-weight analyses
(map permutation and counterfactual re-prediction) and the image-level bootstrap of stage decisions is also
excluded; run-level test predictions are available in `results/`.

## Verification against the study code

The following checks were recorded when preparing this implementation:

- Every configuration represented in `results/` and every nested-audit candidate, with its backbone, was built
  using both the study code and this package from the same seed. Parameters, buffers, trainable parameter
  counts, and outputs on fixed inputs matched.
- Two-epoch training on one fold produced identical histories and test probabilities for the linear probe,
  GF head, WaveFiLM-Net, WaveFiLM-Net with a prior stream, and a fine-tuned backbone.
- For the locked model on the harmonized collection (seeds 1–3, folds 1–5; 15 runs), study checkpoints loaded
  into this package reproduced all saved class decisions, with a maximum probability difference of `5e-7`.
  The group bootstrap also reproduced a study interval exactly.

## Intended use and limitations

This implementation is for research. It is not a medical device and has not been validated for clinical use.
The study uses 8-bit two-dimensional CT slices without patient identifiers. Near-duplicate grouping does not
establish patient-level separation; interpretation of the reported results should follow the limitations in the manuscript.

## License

The code, fold definitions, and results are released under the [MIT License](LICENSE). CT images,
ImageNet-pretrained weights downloaded through torchvision, and other third-party materials remain subject
to their respective terms.

## Citation

Citation metadata and authors are provided in [CITATION.cff](CITATION.cff). The associated article is titled
*WaveFiLM-Net: Wavelet-Conditioned Spatial Feature-Wise Modulation of Frozen Features for Lung CT Image Classification*.
The journal reference will be added when the article is published.
