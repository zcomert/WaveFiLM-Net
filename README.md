# WaveFiLM-Net

Reference implementation of WaveFiLM-Net, a classification head for frozen DenseNet-169 features applied to lung CT
slice classification, together with every comparator and control of the accompanying study, the fold definitions,
the statistics code, and the run-level results.

The head has three modules acting on the final 7 x 7 x 1664 feature map before global average pooling:

| Module | Role |
|---|---|
| HSP (Hessian structure prior) | Parameter-free Frangi vesselness map of the raw image, injected through one learned scalar gain |
| DHSR (dual-stream Haar subband refinement) | Single-level Haar transform of the prior-modulated and raw streams, with refined approximation and detail bands |
| SFF (spatial feature-wise linear modulation) | Position-wise FiLM of the raw features, conditioned on the refined map |

The prior gain, the approximation and pointwise convolutions of DHSR, and the SFF convolution are zero-initialized.
Because SFF is zero-initialized, H = F at initialization, and the model starts as a linear probe (LP) on GAP(F).
The head has 11,105,541 trainable parameters for four classes.

## What is in this repository

* `wavefilm_net/` the package:
  * `layers.py` HSP, DHSR, and SFF. `variants.py` the alternatives of each development stage, the gated-fusion (GF)
    head, the controls, the context-pooling, prior-stream, and bilinear heads, and the SE and CBAM blocks.
  * `model.py` one model class built from a specification (prior, subband, fusion, pooling, attention, bilinear
    rank, fine-tuning). `backbones.py` the eleven torchvision backbones.
  * `data.py` and `train.py` data loading and the training protocol, `stats.py` the paired group bootstrap, Holm
    adjustment, and decision rule, `nested.py` the inner folds and selection rule of the nested audit.
* `configs/experiments.yaml` the 34 named head configurations and the 65 registered experiments by stage, with
  their reference configurations. `configs/nested_audit.yaml` the nested audit protocol. `configs/collections.yaml`
  class names and folder-to-class mapping. `configs/image_checksums.sha256` SHA-256 of every image in the folds.
* `folds/` five-fold grouped cross-validation definitions for the five collections. Each fold lists training,
  inner-validation, and test image paths. Near-duplicate groups (perceptual-hash Hamming distance <= 4) are never
  split across folds (`groups.csv`).
* `results/` run-level results of all 2,280 study runs and the 1,350 nested-audit runs: metrics, test-fold
  predictions, and training histories (see `results/README.md`).
* `scripts/run_fold.py` trains one configuration on one fold, `scripts/nested_audit.py` runs the nested audit, and
  `scripts/check_data.py` checks local image copies against the checksums.
* `tests/` unit tests that need no image data.

## What is not in this repository

* **Images.** The CT collections are third-party data with their own licenses and are not redistributed here. Obtain
  them from their original providers and arrange them as described in `REPRODUCTION.md`.
* **Trained weights.** They are not included in this release.
* **Post hoc analyses at fixed weights** (map permutation, counterfactual re-prediction) and the image-level
  bootstrap of the stage decisions. Their inputs are the run-level predictions in `results/`.

## Install and test

```
pip install -r requirements.txt
python -m pytest tests -q
```

## Use

```python
from wavefilm_net import build_model, build_wavefilm_net
model = build_wavefilm_net(num_classes=4)   # downloads torchvision ImageNet weights for DenseNet-169
logits = model(images)                      # images: float tensor [B, 3, 224, 224] in [0, 1]
gf = build_model(4, {"prior": "threshold_mask", "subband": "pooling_bands", "fusion": "softmax_gate"})
```

Train one fold of one configuration:

```
python scripts/run_fold.py --data-root /path/to/images --collection harmonized-dataset --fold 1 --seed 1 --config wavefilm_net
python scripts/run_fold.py --data-root /path/to/images --collection kaggle-ct --fold 1 --seed 1 --config finetuned --backbone convnext-large
```

## Verification against the study code

* Every configuration of `results/` and every candidate of the nested audit (each with its backbone) was built with the
  study code and with this package from the same seed. Parameters and buffers were identical, trainable parameter
  counts matched the run records, and outputs on fixed inputs were identical.
* Two-epoch training on one fold with the study loop and with this package gave identical histories and test
  probabilities for the LP, the GF head, WaveFiLM-Net, WaveFiLM-Net with the prior stream, and a fine-tuned backbone.
* The trained weights of the locked model (harmonized collection, seeds 1 to 3, folds 1 to 5, 15 runs) loaded into
  this package reproduced the saved test predictions with identical class decisions and a maximum probability
  difference of 5e-7. The group bootstrap in `wavefilm_net/stats.py` reproduced a published interval of the study
  exactly.

## Intended use and limitations

This code is for research. It is not a medical device and has not been validated for clinical use. The study uses
8-bit two-dimensional slices without patient identifiers, and its reported results carry the limitations described in
the manuscript.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The license covers the code, fold
definitions, and results only. Images, the ImageNet-pretrained weights downloaded through torchvision, and any
third-party material are subject to their own terms.

## Citation

See `CITATION.cff`. The journal reference will be added when the article is published.
