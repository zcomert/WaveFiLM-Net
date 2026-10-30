# Run-level results

Every training run reported in the manuscript, as recorded by the study code. No images or trained weights
are included. Configuration names refer to `configs/experiments.yaml`, collections to `configs/collections.yaml`.

## Files

| File | Content |
|---|---|
| `runs.csv` | One row per run (2,280 runs) |
| `predictions/<stage>.csv.gz` | Test-fold predictions of every run in that stage |
| `training_history.csv.gz` | Inner-validation macro-F1 and learning rate per epoch and run |
| `nested_audit/` | Retrospective nested selection audit (1,350 runs), described below |

## Stages

| Stage | Content | Manuscript |
|---|---|---|
| `backbone_screening` | LP on eleven frozen backbones | Table 2 |
| `reference_heads` | LP (seeds 1 to 5), fixed-threshold Hessian mask, mask with pooling bands, GF head | Table 3 |
| `prior`, `subband`, `fusion` | Stepwise head development | Table 3, Fig. 3 |
| `head_and_ablation` | Head alternatives, ablations, prior controls | Table 4 |
| `confirmation` | LP, GF head, WaveFiLM-Net, and the stepwise additions on the four further collections | Table 5, Section 3.4 |
| `backbone_generalization` | LP and WaveFiLM-Net head on frozen MobileNetV2 and ConvNeXt-L, five collections | Section 3.4 |
| `standard_models` | Eleven fine-tuned backbones, SE and CBAM heads, five collections | Table 6 |
| `finetuned_head` | WaveFiLM-Net head on fine-tuned DenseNet-169, five collections | Section 3.5 |
| `controls` | No-subband control, replicated Haar padding, bilinear heads | Table 7 |
| `checkpoint_rerun` | Retraining of the locked model that kept its weights | Section 3.4 |

Development-collection results of the locked model are the `fusion` rows of `wavefilm_net`. Rows of the
same configuration in different stages are separate runs.

## Columns of `runs.csv`

`stage`, `configuration`, `backbone`, `collection`, `seed`, `fold` identify the run. `spec` repeats the
non-default specification fields. Test metrics are computed on the outer test fold: `test_macro_f1`,
`test_balanced_accuracy` (mean class recall), `test_mcc` (mean one-versus-rest Matthews correlation),
`test_accuracy`, and `test_macro_auc` (one-versus-rest). Training records are `trainable_parameters`,
`epochs_run`, `best_epoch`, `best_inner_val_macro_f1`, `train_seconds` (training loop with validation), and
`test_ms_per_image` (batched test pass). Times depend on queue and load conditions and are descriptive.

The primary metric of the manuscript is macro-F1 on pooled out-of-fold predictions, not the mean of the
fold-level `test_macro_f1`. Pool the five folds of a seed from the prediction files:

```python
import pandas as pd
from wavefilm_net.stats import macro_f1_batch

p = pd.read_csv("results/predictions/fusion.csv.gz")
p = p[(p.configuration == "wavefilm_net") & (p.collection == "harmonized-dataset")]
f1 = [100 * macro_f1_batch(g.y_true.to_numpy(), g.y_pred.to_numpy(), 4) for _, g in p.groupby("seed")]
print(sum(f1) / len(f1))   # 88.48
```

Prediction files hold `relpath` (relative to `<data-root>/<collection>/`), `y_true`, `y_pred`, and
`proba_<i>`, where `i` indexes the class list of the collection in `configs/collections.yaml`.

## Nested audit

`nested_audit/inner_splits.json.gz` holds the inner folds of each outer fold. `selection.json` holds, per outer
fold and stage, every candidate with its specification, seed scores, mean inner macro-F1, trainable parameters,
the identifiers of its runs, and the selected candidate. `runs.csv`, `predictions.csv.gz`, and
`training_history.csv.gz` hold the inner (`role = inner`, scored on the inner test fold) and outer
(`role = outer`) runs, linked by `run_id`. `outer_runs.csv` lists the outer runs of the nested-selected
pipeline, the LP, and the GF head, and `summary.json` their pooled macro-F1.
