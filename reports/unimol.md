# R3: Uni-Mol and ligand 3D shape (partial report)

**Status:** steps 1–3 implemented with preliminary benchmarks; step 4 (fine-tuning) prototyped only. Steps 5–6 (optional) not run, though a preliminary prediction-correlation check was done.

> **Read this first: all benchmark numbers below are exploratory.**
> They come from a single **random** train/validation split (seed 42; 3,924 train / 981 validation molecules), **one seed**, and a **decision tree** model only.
> They were **not** run on the pinned `split` column and do **not** meet the spec's "pinned split, ≥3 seeds" requirement, so they should not be compared directly with pinned-split results from other tasks. Macro R² is negative for every configuration, i.e. the decision tree does worse than predicting the mean, so differences between representations here are small relative to model noise.

## What was done

| Spec step | Status | Notes |
|---|---|---|
| 1. Shape descriptors | Implemented (`src/representations/shape.py`), registered as `shape` | 11 RDKit `Descriptors3D` features (PMI1–3, NPR1–2, asphericity, eccentricity, inertial shape factor, radius of gyration, spherocity, PBF). Fixed-seed ETKDGv3 conformer (seed 42), MMFF-optimized (max 1,000 iterations). Failures give a NaN row. |
| 2. Conformer determinism | Fixed seed; failure rate measured on training data | Test-set failure rate not yet measured. |
| 3. Frozen Uni-Mol | Implemented (`src/representations/unimol.py`), registered as `unimol` | Uni-Mol v1 via `unimol_tools`, 512-dim embeddings, same fixed-seed MMFF conformers as `shape`. Embedding correlation with `chemeleon` **not yet computed** (prediction correlation reported instead, see below). |
| 4. Fine-tuned Uni-Mol | Implemented as a backend of the `train-model` CLI (`src/models/training/unimol.py`, submitted with `scripts/train_model.qsub`); pipeline tested on a small CPU run (24 train / 6 validation molecules) and, in the final multi-task form, only against a stub `unimol_tools` | One multi-task model over the challenge heads plus any `--datasets` auxiliary heads on the pinned `split` column; writes T4 long-format predictions. Full sweep (3 dataset configs × 3 LRs × 3 seeds) **not yet run**; needs SCC GPU. |
| 5. Test-time augmentation (optional) | Not run | |
| 6. Orthogonality (optional) | Preliminary only | Uni-Mol vs CheMeleon *decision-tree* predictions, not chemprop. |

## Results

### Conformer / embedding reliability

| Run | MMFF max iters | Molecules | Success | Fail | Success rate |
|---|---|---|---|---|---|
| Initial sample | 200 | 100 | 64 | 36 | 64% |
| Improved sample | 1,000 | 100 | 100 | 0 | 100% |
| Full training data (`shape`) | 1,000 | 4,905 | 4,903 | 2 | 99.96% |
| Full training data (`unimol`) | 1,000 | 4,905 | 4,903 | 2 | 99.96% |

All 36 failures in the initial sample were MMFF non-convergence, fixed by raising max iterations to 1,000. Two training molecules still fail (rows 3277 and 3627, 0-based); the failure stage is not yet diagnosed:

- `CC1=CC2=C(CC(C)(C)CC2NC(=O)C2=CNC3=CC=CC=C3C2=O)O1`
- `CC(C)N1C=CC(CN2CCNC(=O)C2CC2=CC=CC=C2)=N1`

Full-dataset Uni-Mol embedding runtime: 4.93 min (100-molecule test: 9.76 s). Test-set failure rate: not yet measured.

### Frozen representations: validation RMSE

Decision tree, random split, seed 42, single run (no ± over seeds).

| Input | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | Macro RMSE | Macro MAE | Macro R² |
|---|---|---|---|---|---|---|---|
| `shape` | 1.1245 | 0.8031 | 0.9502 | 1.0598 | 0.9844 | 0.7365 | −0.0453 |
| `morgan` | 1.1281 | 0.7687 | 0.9876 | 1.0628 | 0.9868 | 0.7464 | −0.0486 |
| `morgan` + `shape` | 1.1016 | 0.7510 | 0.9675 | 1.0409 | **0.9653** | 0.7346 | −0.0034 |
| `chemeleon` | 1.2283 | 0.8122 | 0.9470 | 0.9730 | 0.9901 | 0.7386 | −0.0619 |
| `chemeleon` + `shape` | 1.2034 | 0.7982 | 0.9664 | 0.9694 | 0.9844 | 0.7331 | −0.0503 |
| `unimol` | 1.2278 | 0.8236 | 1.0027 | 1.1408 | 1.0487 | 0.7802 | −0.1784 |
| `unimol` + `morgan` | 1.1052 | 0.8340 | 0.9958 | 1.1383 | 1.0183 | 0.7608 | −0.1219 |
| `unimol` + `chemeleon` | 1.1954 | 0.8168 | 0.9889 | 0.9612 | 0.9906 | 0.7393 | −0.0692 |

Observations (preliminary): adding `shape` to `morgan` lowered macro RMSE from 0.9868 to 0.9653, with gains on CYP1A2, CYP2C9 and CYP3A4 but not CYP2D6. Standalone `unimol` was the weakest input. Per the spec, standalone RMSE is not the right test for Uni-Mol; what matters is what it adds to chemprop, which has not been measured yet.

### Uni-Mol vs CheMeleon prediction correlation and simple average

Pearson r between `unimol` and `chemeleon` decision-tree validation predictions, and RMSE of their simple two-model average:

| Isoform | Matched predictions | Pearson r | Uni-Mol RMSE | CheMeleon RMSE | Average RMSE |
|---|---|---|---|---|---|
| CYP1A2 | 297 | 0.0346 | 1.2278 | 1.2283 | 1.1272 |
| CYP2C9 | 271 | 0.1508 | 0.8236 | 0.8122 | 0.7486 |
| CYP2D6 | 307 | 0.1523 | 1.0027 | 0.9470 | 0.9101 |
| CYP3A4 | 446 | 0.1981 | 1.1408 | 0.9730 | 0.9753 |
| **Macro** | | | 1.0487 | 0.9901 | **0.9403** |

Caveat: these correlations are very low partly because single decision trees produce noisy, high-variance predictions; averaging two noisy models reduces variance regardless of whether the representations carry different information. This is **not** the orthogonality verdict the spec asks for (that needs chemprop predictions on the pinned split), and it is **not** the step-3 embedding correlation.

### Chemprop with `shape` as extra descriptors

Not run.

### Fine-tuned Uni-Mol learning-rate sweep

Not run. Planned: 3 dataset configs (ADMET only, ADMET + auxiliary, auxiliary only) × learning rates 1e-2 / 1e-3 / 1e-4 × seeds 0/1/2 = 27 models, submitted as

```
qsub -l h_rt=48:00:00 scripts/train_model.qsub --model unimol \
    --learning-rate 1e-2 1e-3 1e-4 --n-seeds 3 --seed 0 \
    --datasets admet \
    --datasets admet pubchem:CYP1A2 bindingdb:CYP1A2,CYP2C8 chembl:CYP1A2,CYP2C8 \
    --datasets pubchem:CYP1A2 bindingdb:CYP1A2,CYP2C8 chembl:CYP1A2,CYP2C8
```

The auxiliary-only model has no challenge head, so each challenge head is scored with the same-isoform auxiliary head (a cross-assay proxy): its RMSE includes a calibration offset, and only CYP1A2 and CYP2C9/2D6/3A4 heads with a matching auxiliary isoform can be scored. Pipeline verified on a 24/6-molecule CPU smoke test only.

## Not done yet

- Re-run all benchmarks on the pinned `split` column with ≥3 seeds and the repo's standard models via `evaluate-models`.
- Step-3 embedding correlation between `unimol` and `chemeleon`.
- Conformer failure rate on the test set; diagnose the two failing molecules.
- `shape` as chemprop extra descriptors.
- Full fine-tuning sweep (3 dataset configs × LR 1e-2 / 1e-3 / 1e-4 × seeds 0/1/2 = 27 runs) on SCC; see the command above.
- Optional steps 5 (TTA) and 6 (orthogonality vs chemprop).

## Open questions

- Fine-tuning is now one multi-task model with masked labels: `unimol_tools`' `multilabel_regression` uses a NaN-masked MAE loss, so missing labels are skipped. This has not been confirmed on a real run, and the loss is MAE rather than the MSE of single-target `regression`, which also skips its 3-sigma outlier cleaning.
- 1e-2 is a high learning rate for fine-tuning a pretrained transformer and may diverge; the sweep will show.
- **Early stopping uses training loss.** With `kfold=1`, `unimol_tools` 0.1.6 does not hold out any data: its splitter returns an empty test-index tuple, and indexing with an empty tuple returns the whole array, so the internal "validation" set is the full training set. Early stopping (patience 10) and the saved checkpoint are therefore chosen on training loss. The official `val` rows are never passed to the trainer, so the reported validation RMSE is not leaked, but there is no protection against overfitting. Options: hold out a scaffold-grouped slice of the train rows for early stopping, or use `kfold=5` (5× compute; an earlier local run with that setting produced `model_0.pth` … `model_4.pth`).
- `scripts/train_unimol.qsub` is gone: `scripts/train_model.qsub` forwards its arguments to `train-model`, which merges the datasets and pinned split itself (no `data/union_train.csv` needed). The 27 fits run sequentially in one job; raise `h_rt` for more.
- `fold` is written as `"val"` in the OOF output; confirm this matches what T4 expects.
