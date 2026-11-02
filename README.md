# openadmet-cyp

Code for the OpenADMET CYP Inhibition Blind Challenge.

**Contributing:** read [RULES.md](RULES.md) before opening a PR — how a PR should
look, what must never be committed, the rule that only Denali submits to the live
HuggingFace challenge, and how to use AI tools. Pick a workstream from
[tasks/round1/MANIFEST.md](tasks/round1/MANIFEST.md), which also holds the shared metric, the
pinned split and the join keys.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Data

```bash
data list                  # every known dataset with status, rows, size
data fetch NAME [NAME...]  # fetch by name or alias (or `all`), then derive the model-ready table
data remove NAME           # delete a dataset's raw and derived files
data alias NAME ALIAS      # short alias, stored in data-alias.csv
download-data              # same as `data fetch all`
```

Datasets are recipes in `src/data_tools/datasets.py`: a `fetch` step from the source and an optional `derive` step.
`pubchem_aid1851` yields per-isoform auxiliary heads; `p450_family` (ChEMBL + BindingDB, Pfam PF00067) yields one
`{gene}_pIC50_{source}` head per protein per source.

### External data as auxiliary heads

External CYP measurements are **never pooled into the scored pIC50 columns**.
Each source x isoform x readout becomes its own column, and therefore its own
head at training time, so no cross-assay calibration is required: a shared
D-MPNN encoder sees every molecule and every label while each head keeps its
own scale. Chemprop masks missing targets in the loss, so the sparse block
structure of a multi-source union costs nothing.

This replaces a pretrain-then-finetune split. One model is trained jointly on
all heads at once, which avoids both the pooling problem and catastrophic
forgetting during fine-tuning.

**PubChem AID 1851** contributes the signal a potency-only ingest throws away.
The panel screens five isoforms (1A2, 2C9, 2C19, 2D6, 3A4) in 15-point
dose-response. Of its 85,715 compound x isoform rows only 40,684 have a fitted
`Potency`, but **all** of them have `Max_Response` — including the 42,460 rows
the screen called Inactive, which have no potency at all. Since ChEMBL-style
potency data is truncated from below (no curve, no row), those negatives are
what a model trained on fitted pIC50 never sees. CYP2C19 is carried as an
extra head even though the challenge does not score it.

`Max_Response` is a PERCENT change in luminescence and is mostly negative,
because inhibiting pro-luciferin conversion *decreases* signal. It is emitted
negated, as `{ISO}_pctinhib_aid1851`, so that larger means more inhibition —
matching the direction of pIC50.

### Structural identity

Sources spell the same molecule differently (salt vs free base, explicit vs
implicit stereo), so raw SMILES comparison under-reports overlap. Every merge
and every overlap measurement keys on the **InChIKey connectivity block**
(first 14 characters), which is invariant to stereochemistry, protonation and
isotope labelling. See `data_tools.standardize`.

## Training

`train-model` trains any supported model on exactly the datasets you list.
Pick the model with `--model`:

| `--model` | What it trains |
|---|---|
| `chemprop` | Chemprop D-MPNN from scratch |
| `chemeleon` | Chemprop started from the CheMeleon foundation weights |
| `unimol` | Uni-Mol fine-tuned (needs `unimol_tools`; uses the GPU if available, `--cpu` to force CPU) |

```bash
train-model --list-datasets                     # selectable sources
train-model --model chemprop --datasets admet tdc_hlm pubchem:CYP1A2 bindingdb:CYP1A2,CYP2C8 --loss-function mae
train-model --model chemeleon --datasets admet pubchem:CYP1A2
train-model --model unimol --datasets admet pubchem:CYP1A2 --learning-rate 2e-4
# configs x learning rates x seeds: repeat --datasets, add --n-seeds N (or --seeds 0 1 2) and, for unimol, several
# --learning-rate values; one model per combination
train-model --model unimol --n-seeds 3 --learning-rate 1e-2 1e-3 1e-4 \
    --datasets admet \
    --datasets admet pubchem:CYP1A2 bindingdb:CYP2C8 \
    --datasets pubchem:CYP1A2 bindingdb:CYP2C8
# pretrain on stability, then train on the ADMET data plus auxiliary heads (chemprop/chemeleon)
train-model --model chemprop --stage tdc_hlm tdc_rlm --stage admet pubchem:CYP1A2
# same thing, with per-stage settings and an encoder-size sweep
train-model --model chemprop --pretrain tdc_hlm --finetune admet pubchem:CYP1A2 \
    --stage-epochs 30 50 --stage-freeze-epochs 0 5 --stage-lr-warmup-shape linear cosine \
    --stage-loss mae huber --huber-delta 0.5 --hidden-dims 300 600 --mpn-dropouts 0 0.1
```

`--datasets` takes `admet` (the ADMET labels) and auxiliary sources such as
`pubchem`, `bindingdb`, `tdi` or `chembl`, as `NAME[:ISO,ISO]`; heads are named
`{isoform}-{source}`. Only what you list is trained on: `--datasets admet` is the
ADMET data alone, `--datasets admet pubchem` is both, and `--datasets pubchem`
is the auxiliary data alone. The ADMET data is still loaded in every case,
since it provides the pinned train/val split and the validation molecules; with
no `admet` in the config its labels are just not trained on, and each ADMET head
is scored with the same-isoform auxiliary head, so that RMSE carries a
cross-assay calibration offset. Each head keeps its own scale and missing labels
are masked in the loss. With several runs, outputs go to
`<output-dir>/<config>/seed<N>` (default `results/<model>/...`). Shared flags:
`--target-columns`, `--split-strategy`, `--split-seed`, `--epochs`, `--seed`,
`--output-dir`.

Chemprop training options (`src/models/chemprop_ext.py` patches them into
`chemprop train`; `train-model` runs chemprop through it):

| Flag | Effect |
|---|---|
| `--pretrain A B [--pretrain C] --finetune X Y` | Pretrain stages, then the evaluated final stage (shorthand for repeated `--stage`) |
| `--stage-epochs`, `--stage-warmup-epochs`, `--stage-max-lr`, `--stage-lr-warmup-shape`, `--stage-lr-decay`, `--stage-freeze-epochs`, `--stage-loss` | Per-stage values, one per stage or one for all |
| `--loss-function huber --huber-delta D` | Huber loss; `D` in pIC50 units |
| `--label-noise S`, `--label-shrink A` | Training-only target smoothing: Gaussian jitter, or pull toward the batch mean per head |
| `--lr-warmup-shape linear\|cosine\|exp\|none`, `--lr-decay exp\|cosine\|linear\|constant` | Learning-rate schedule shape (chemprop's `--warmup-epochs`, `--init-lr`, `--max-lr`, `--final-lr` still apply) |
| `--freeze-epochs N` | Keep the message-passing encoder frozen for the first `N` epochs |
| `--mpn-dropout P`, `--ffn-dropout P` | Dropout in message passing / the prediction head separately |
| `--hidden-dims`, `--depths`, `--mpn-dropouts` | Encoder-size sweep: every combination trains, times seeds. Later stages inherit the first stage's encoder size |

Checkpoints trained with Huber need `models.chemprop_ext` importable to load.
Censored losses (`--loss-function bounded-mse`) need `<`/`>` in the label CSV;
none of our datasets carry that yet.

`--split-strategy` defaults to `scaffold`, the pinned split every recorded result
is reported against. `blind_matched` instead makes the held-out slices imitate the
blind set's regime — 75% novel scaffolds rather than 100% — which scores about
0.013 lower and is **not** comparable to the pinned numbers. It is a diagnostic for
how much the pinned split understates performance, not a reporting axis; it ranks
models identically. See [reports/blind_matched_split.md](reports/blind_matched_split.md).

Chemprop-based models pass unknown flags (`--batch-size`, `--accelerator`,
`--task-weights`, ...) through to `chemprop train`. To train on a ready table
instead, `--data-path data/union_train.csv --target-columns <all heads>`.
`train-chemprop` and `train-unimol` remain as aliases for the matching
`--model`. On SCC, `scripts/train_model.qsub` forwards its arguments to
`train-model`. To add a model, write a `Backend` in
`src/models/training/` and register it in `BACKENDS`.

```bash
head-search                # search auxiliary-head subsets on the pinned split (single / greedy / ablation / exhaustive)
evaluate-models            # train/val metrics for the sklearn-style models
generate-results           # predictions for submission
```

## Representations

`evaluate-models` and `generate-results` take `--input` with one or more of the
names below; several names are hstacked into one feature matrix.

| `--input` | Class | Dim | What it is |
|---|---|---:|---|
| `rdkit` | `RDKitDescriptors` | 217 | RDKit 2D physicochemical descriptors |
| `chemeleon` | `CheMeleon` | 2048 | fixed pretrained CheMeleon embedding; checkpoint downloads on first use |
| `morgan` | `MorganFingerprint` | 2048 | Morgan/ECFP4 bits, radius 2 |
| `count_morgan` | `CountMorganFingerprint` | 2048 | same generator, substructure counts |
| `maccs` | `MACCS` | 167 | MACCS structural keys |
| `mordred` | `MordredDescriptors` | 1613 | Mordred 2D descriptors (`mordredcommunity`); missing values encoded as 0 |
| `xtb_solv_v1` | `XTBSolvation` | 9 | GFN2-xTB (`tblite`) ALPB water/octanol solvation, xTB logP, heuristic logD at pH 7.4; slow, see note below |
| `xtb_elec_v1` | `XTBElectronic` | 24 | Boltzmann-averaged GFN2-xTB FMO energies, charges, SASA, shape, flexibility; shares conformers with `xtb_solv_v1` |
| `cyp_mech_v1` | `CYPMechanisticBlock` | 30 | Mechanistic block: N/acid group counts, rule-based pKa and pH 7.4 protonation state, basic-N-to-aryl topological distances, CYP2D6 pharmacophore flags. Ported from Lizard Wizard Gizzard, see Acknowledgements |

The xTB blocks embed up to 10 ETKDG/MMFF conformers per molecule and run three
xTB single points each. Cost is seconds per molecule, so the first run is the
expensive one (results are cached in `data/features/`). Worker processes come from
`XTB_NJOBS`, else `NSLOTS` (SGE), else the CPU count. Request both blocks in one
call (`--input xtb_solv_v1 xtb_elec_v1`) to compute each molecule once.

```bash
evaluate-models --input morgan                          # one representation
evaluate-models --input count_morgan rdkit --split butina --seed 0
evaluate-models --input all                             # every representation on its own, one table
evaluate-models --models ridge --input morgan --predictions-out results/validation/morgan.csv
```

`--predictions-out <path>` optionally saves one UTF-8 long-format CSV containing
validation predictions for every selected model, input set, and target:

```text
smiles,isoform,model,input,split,seed,y_true,y_pred
```

`smiles` preserves the source `SMILES` string; `isoform` is the full target
column name and `model` is its registry name. Combined inputs use their ordered
names joined with ` + ` (for example, `count_morgan + rdkit`); `--input all`
produces separate groups. `split` records the executed strategy (`random`,
`scaffold`, or `butina`), and `seed` is the **split seed only**. This evaluator
does not use the pinned split column or pass that seed to model constructors;
models retain their own defaults. Record the command alongside exports, including
`--val-split`, `--butina-cutoff`, and the data source, which are not CSV columns.

Each model predicts validation once, and those exact values are scored and
exported. The shared filter `np.isfinite(X).all(axis=1) & np.isfinite(y)` selects
features, labels, and SMILES together, retaining their order. Every surviving
row is saved, even if another representation drops that molecule. Duplicate
source SMILES within a `(model, input, isoform, split, seed)` group cause export
to fail with an explicit error before writing; rows are never deduplicated.
Repeated selections that duplicate the same group also fail. Without the flag,
evaluation retains its existing behavior, including duplicate rows.

Use ignored `results/` paths for generated CSVs. Missing parent directories are
created. Existing files are refused (exclusive creation, no overwrite or append).
Write errors report the destination and underlying error and exit unsuccessfully;
an I/O failure after creation can leave a partial file, which must be removed or
given a new destination before retrying. Cross-run SMILES inner joins and
prediction/residual correlations are a separate next task.

### Evaluating an embedding

`representations --evaluate <name>... | all` scores embeddings on a train/val
split (`--split` default `scaffold`, plus `--val-split`, `--seed`,
`--butina-cutoff`, `--targets`, `--max-samples`) using the feature cache. It prints
two tables; label metrics are averaged over the targets that have data.

| Metric | Column | Reads as |
|---|---|---|
| ridge probe val R2 | Probe R2 | higher is better |
| 5-NN regression val R2 | kNN R2 | higher is better |
| Tanimoto neighbour recall@10 | Recall | embedding 10-NN overlap with Morgan 10-NN |
| effective rank | EffRank | low means dimensional collapse |
| roughness (1-NN label diff / random-pair diff) | Rough | below 1 means neighbours share labels |
| val-to-train NN distance vs train leave-one-out | NN ratio, Out | 1.0 and 0.05 mean val looks like train |
| train-vs-val classifier AUROC | AUC | 0.5 indistinguishable, 1.0 separable |
| RBF MMD permutation p-value | MMD p | small means the distributions differ |
| Spearman of val NN distance vs probe error | Err rho | positive means distance predicts failure |

The first row of each distance table is the same NN ratio and Out measured on
Morgan Tanimoto, the baseline for how hard the split is on its own.

A third table, on by default (`--no-test` skips it), measures the blinded
`data/test.csv` against all of `train.csv` with the label-free metrics (NN ratio,
Out, AUC, MMD p), so the real test shift can be compared with the validation shift. NN ratio below 1 means
test sits closer to train than train does to itself, while a high AUC with it
means test is still a different distribution. Features are
standardized with train statistics first, and the pairwise metrics subsample to
`--max-samples` molecules.

Unparsable SMILES give an all-NaN row, which `evaluate-models` drops. Features
are cached per representation in `data/features/{name}.npy`, keyed on SMILES, so
each molecule is featurized once across every split and seed. To add one,
subclass `Representation` in `src/representations/` and append it to
`INPUT_REGISTRY` in `src/data_tools/inputs.py`; see
`reports/t2_solo_representations.md` for how the current six compare.

## Models

Registered models accept precomputed feature matrices through `fit(X, y)` and
`predict(X)`. The evaluator applies `models.utils.drop_nan_rows` before fitting
and scoring. The three ensemble wrappers do not add filtering, feature generation,
scaling or PCA. Select models with `evaluate-models --models <name> ...`.

| Name | Estimator | Default settings |
|---|---|---|
| `decision_tree` | sklearn DecisionTreeRegressor | depth=5, min split=2, min leaf=1, seed=42 |
| `ridge` | sklearn StandardScaler + Ridge | alpha=1.0 |
| `random_forest` | sklearn RandomForestRegressor | 200 trees, unlimited depth, min leaf=1, all features |
| `lightgbm` | LightGBM LGBMRegressor | 200 trees, rate=0.05, 31 leaves, unlimited depth, min child=20 |
| `xgboost` | XGBoost XGBRegressor | 200 trees, rate=0.05, depth=6, hist trees |
| `tabpfn` | TabPFN TabPFNRegressor | n_estimators=auto, temperature=auto, device=auto, seed=42 |

The three ensemble wrappers default to CPU, `random_state=42` and `n_jobs=1`.
Random Forest uses `criterion=squared_error`, `bootstrap=True` and `max_features=1.0`.
LightGBM uses `boosting_type=gbdt`, `objective=regression`, `device_type=cpu`
and `verbosity=-1`. Its unlimited depth is `max_depth=-1`.
XGBoost uses `objective=reg:squarederror`, `booster=gbtree`, `tree_method=hist`,
`device=cpu`, `subsample=1.0` and `colsample_bytree=1.0`.

Constructors expose tree count, complexity controls, seed and worker count;
the boosting wrappers also expose learning rate. The evaluator uses constructor
defaults. These are baseline settings, not tuned CYP results.

TabPFN does no gradient training: `fit` stores the training rows and the
transformer conditions on them at predict time. Its constructor exposes
`n_estimators`, `softmax_temperature`, `ignore_pretraining_limits`, `device`,
`model_path` and `random_state`; it defaults to `device="auto"` rather than CPU,
since the others are tree models and this one is a transformer. Set
`ignore_pretraining_limits=True` to feed it representations wider than the
pretraining feature limit.

`tabpfn` is an optional dependency, registered only when the package is
installed, so a missing install drops it from `--models all` instead of
breaking the CLI. It is untuned and has not been run on CYP data yet.

## Splits

`data_tools.load.load_data` supports `random`, `scaffold` (Bemis-Murcko) and
`butina` (Tanimoto cluster) train/val splits; `evaluate-models` exposes them as
`--split` and `--seed`.

Chemprop multi-task runs use a different mechanism: a pinned `split` column in
`data/union_train.csv`, and `head-search` refuses to run without it.

## Ensembling

`evaluate-models` ensembles across folds, seeds, models and inputs. Everything
below needs `--nfolds` except plain averaging.

```bash
evaluate-models --models ridge random_forest lightgbm --input morgan --input rdkit \
    --nfolds 5 --cv-split scaffold --nseeds 3 --ensemble nnls \
    --folds-out results/folds_scaffold5.csv
```

- `--nfolds N`, `--cv-split random|scaffold|butina`: k-fold CV on the training
  split; `--folds-out` saves each training molecule's fold to a new CSV.
- `--nseeds N`: seed averaging per model (models without `random_state` train once).
- `--ensemble average|min|max|nnls|ridgecv|elasticnetcv`: how models are combined into
  an `ensemble[...]` row. The stackers learn per-target weights from out-of-fold
  predictions (`ridgecv` and `elasticnetcv` need `--nfolds >= 3`), print the
  weights, shares, intercept and chosen hyperparameters, and score the stack
  with leave-one-fold-out refits. Models are still averaged over folds and seeds.
- With several input sets (`--input` repeated), each target also gets an
  `all inputs` ensemble over every (model, input) pair; `--no-input-ensemble`
  turns that off.
- `--cv-filter r2 RAE`: before ensembling, drop models whose out-of-fold R² is below
  0.15 and models whose validation RAE is more than 2x their out-of-fold RAE
  (the leakage check). If none pass, all are kept with a warning. This is the
  default with `--nfolds`; `--cv-filter none` disables it.
- Models whose validation predictions correlate above 0.95 (Pearson, complete
  linkage) are averaged into one before ensembling; `--no-cv-deduplication`
  disables this. It runs after the filters.
- Every filter and deduplication decision is logged to stderr and summarised in an
  `ensemble filters` table with each model's scores, kept/dropped status and cluster.

Stackers are fitted only on out-of-fold predictions, never on predictions the base
models saw in training. Validation informs selection, so validation metrics are
development results.

Chemprop can train jointly on additional auxiliary regression columns listed
in `chemprop_config.target_columns`. Feature models do not automatically use
auxiliary target columns, but can use auxiliary-trained checkpoint embeddings.
Checkpoints must not have been trained on held-out labels. Record their resolved
paths and hashes, keep weights immutable, and use fresh caches when changing them.

## Acknowledgements

`cyp_mech_v1` (`src/representations/cyp_mech.py`) is ported from the open-code
submission of the Lizard Wizard Gizzard team,
[doctawho42/CYPChallenge](https://github.com/doctawho42/CYPChallenge) (MIT License,
Copyright (c) 2026 the CYPChallenge contributors), at commit `bf018b8`. The SMARTS
tables, pKa rule and column definitions are theirs; see `THIRD_PARTY_NOTICES.md`.
