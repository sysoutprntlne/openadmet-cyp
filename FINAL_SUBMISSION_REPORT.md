# Final submission report

IN PROGRESS

## Foundation models

The only foundation-style models in the final submission are **Chemprop D-MPNN encoders that we trained
ourselves** (some started from the published CheMeleon weights). They are not used as predictors. Each
encoder turns a molecule into a feature vector, and a **TabPFN** regressor fits on those features. The
TabPFN models are then combined with a **Caruana** ensemble. No Uni-Mol encoder and no standalone CheMeleon
encoder made the cut.

Source of every number below: `final_training_run.txt` (validation split `blind_matched`, seed 42, plus
3-fold scaffold CV on the training portion).

### What made the cut

Nine input sets went into the final ensemble. Each set is **three Chemprop encoders** (three seeds) whose
features are concatenated before TabPFN sees them.

| Input set (as passed to `--input`) | Encoder recipe | Val RMSE | Val MAE | Val R² | Val ρ |
|---|---|---|---|---|---|
| `grid_both_chem_{0,1,2}` | CheMeleon init, challenge + ADMET + auxiliary CYP data, MAE | 0.490 | 0.335 | 0.692 | 0.849 |
| `grid_adme_chem_{0,1,2}` | CheMeleon init, challenge + ADMET/metabolism data, MAE | 0.509 | 0.340 | 0.668 | 0.830 |
| `grid_aux_chem_{0,1,2}` | CheMeleon init, challenge + auxiliary CYP data, MAE | 0.502 | 0.348 | 0.674 | 0.827 |
| `huber_soft_{0,1,2}_{,_2,_3}` (3 batches) | Huber loss with soft labels | 0.586–0.608 | 0.420–0.437 | 0.526–0.562 | 0.725–0.740 |
| `mae_admet_{0,1,2}_{,_2,_3}` (3 batches) | MAE loss, ADMET data | 0.600–0.626 | 0.427–0.445 | 0.502–0.542 | 0.706–0.729 |

Metrics are the mean over the four isoforms, TabPFN on each input set alone. The `_2` and `_3` suffixes mark
two further batches of three seeds for the same recipe.

The three `grid_*_chem` sets carry most of the performance. The `huber_soft` and `mae_admet` sets are weaker
alone but are kept because their predictions differ from the grid sets (Pearson 0.77–0.96 between the two
groups, depending on isoform, against 0.90–0.99 within a group) and the ensemble gains from that.

### How the sets are combined

1. Encoders: three seeds per set, features concatenated (`--input A B C` stacks them).
2. One TabPFN model per input set.
3. Cross-validation filter: a model is kept only if OOF R² ≥ 0.15 and RAE ratio ≤ 2.0.
4. Deduplication: models whose predictions correlate above Pearson 0.95 are averaged into one cluster.
5. Caruana ensemble over the clusters, fit separately per isoform, intercept 0.

### Final performance

| Isoform | RMSE | MAE | R² | ST-RAE | ρ | τ |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.516 | 0.349 | 0.715 | 0.501 | 0.820 | 0.651 |
| CYP2C9 | 0.401 | 0.278 | 0.726 | 0.482 | 0.875 | 0.702 |
| CYP2D6 | 0.600 | 0.379 | 0.510 | 0.631 | 0.785 | 0.603 |
| CYP3A4 | 0.395 | 0.277 | 0.869 | 0.311 | 0.938 | 0.791 |
| **Mean** | **0.478** | **0.321** | **0.705** | **0.481** | **0.855** | **0.687** |

On 3-fold scaffold CV the ensemble's mean is RMSE 0.521, MAE 0.360, R² 0.691. The best single input set
(`grid_both_chem`) is RMSE 0.529 and MAE 0.374 on the same CV, so the ensemble adds a modest, consistent gain.

### Screened but not in the final ensemble

`chemeleon_*` (plain CheMeleon), `grid_*_scra_*` (the same three data sets trained from scratch),
`huber_adme_*`, `huber_cosi_*`, `tdc_hlm+_rlm+...`, `rdkit`, and `cyp_mech_v1`. In the earlier xgboost screen
(`history2.txt`), `chemeleon_*`, `huber_cosi_*`, `rdkit` and `cyp_mech_v1` sit at the bottom of the ranking
(RMSE 0.738–0.944). `cyp_mech_v1` was dropped by the R² filter (OOF R² −0.023 on CYP1A2).

### Reproducing the models

The Chemprop checkpoints are what is reproduced; TabPFN and the ensemble are refit from them.

**1. The three `grid_*_chem` families.** CheMeleon init, MAE loss, seeds 0 1 2, one `train-model` call
(`--model chemeleon` is chemprop started from the CheMeleon weights). This replaces the older
`scripts/run_model_grid.sh` grid search; the dataset lists are the same ones it used. Every name is from
`train-model --list-datasets`, and only listed datasets are trained on; `challenge` is the competition labels
the old script always included, so it is listed in every config:

```bash
export CHEMPROP_HOME=.chemeleon
ADMET="tdc_hlm tdc_rlm expansionrx tdc_clearance_microsome_az tdc_half_life_obach tdc_cyp2c9_substrate tdc_cyp2d6_substrate tdc_cyp3a4_substrate"
AUX="pubchem chembl bindingdb tdi single_conc"

train-model --model chemeleon --loss-function mae --metrics mae rmse --show-individual-scores \
  --seeds 0 1 2 \
  --datasets challenge $ADMET \
  --datasets challenge $AUX \
  --datasets challenge $ADMET $AUX \
  --output-dir results/chemprop/grid_chem
# or on SCC (extra args pass through to train-model):
qsub scripts/train_model.qsub --model chemeleon --loss-function mae --seeds 0 1 2 \
  --datasets challenge $ADMET --datasets challenge $AUX --datasets challenge $ADMET $AUX \
  --output-dir results/chemprop/grid_chem
```

Output: one `results/chemprop/grid_chem/<config>/seed{0,1,2}/` per config, where the config directory name is the
slug of the dataset list. These map to `grid_adme_chem`, `grid_aux_chem` and `grid_both_chem` in the order
given. Not re-run: the dataset lists and flags come from the grid script and the CLI source, but I have not
checked that this call reproduces the original checkpoints exactly.

**2. `huber_soft` and `mae_admet`.** The recipes below come from the run names and the `train-model` flags; the
repository does not record the dataset arguments used for these runs, so set `--datasets` to the config you
trained on. Run each recipe three times to get the three batches (`_`, `_2`, `_3`):

```bash
# huber_soft: Huber loss plus soft labels (label noise and shrink)
train-model --model chemprop --epochs 60 --n-seeds 3 --seed 0 \
  --loss-function huber --huber-delta 0.5 --label-noise 0.2 --label-shrink 0.1 \
  --datasets challenge --output-dir results/chemprop/huber_soft

# mae_admet: MAE loss on the ADMET data
train-model --model chemprop --epochs 60 --n-seeds 3 --seed 0 \
  --loss-function mae --datasets challenge --output-dir results/chemprop/mae_admet
```

Checkpoints trained with `--loss-function huber` need `models.chemprop_losses` importable to load (fixed in
commit `b181e36`; use a checkout that includes it).

**3. Build the feature caches.** This runs each encoder over the train and test molecules:

```bash
foundation-model --generate-cache
foundation-model --list          # confirm every input name below exists
```

Input names such as `grid_aux_chem_0` are aliases from `foundation-path-alias.csv`, which is local and
git-ignored. `foundation-model --paths` shows what each alias points to.

**4. Fit TabPFN, filter, deduplicate, and ensemble** (the final run; the quotes stop the shell expanding the
brackets):

```bash
evaluate-models \
  --input 'chemprop[grid_aux_chem_0]' 'chemprop[grid_aux_chem_1]' 'chemprop[grid_aux_chem_2]' \
  --input 'chemprop[grid_both_chem_0]' 'chemprop[grid_both_chem_1]' 'chemprop[grid_both_chem_2]' \
  --input 'chemprop[grid_adme_chem_0]' 'chemprop[grid_adme_chem_1]' 'chemprop[grid_adme_chem_2]' \
  --input 'chemprop[huber_soft_0]' 'chemprop[huber_soft_1]' 'chemprop[huber_soft_2]' \
  --input 'chemprop[huber_soft_0_2]' 'chemprop[huber_soft_1_2]' 'chemprop[huber_soft_2_2]' \
  --input 'chemprop[huber_soft_0_3]' 'chemprop[huber_soft_1_3]' 'chemprop[huber_soft_2_3]' \
  --input 'chemprop[mae_admet_0]' 'chemprop[mae_admet_1]' 'chemprop[mae_admet_2]' \
  --input 'chemprop[mae_admet_0_2]' 'chemprop[mae_admet_1_2]' 'chemprop[mae_admet_2_2]' \
  --input 'chemprop[mae_admet_0_3]' 'chemprop[mae_admet_1_3]' 'chemprop[mae_admet_2_3]' \
  --models tabpfn --split blind_matched --seed 42 \
  --nfolds 3 --cv-split scaffold --cv-filter r2 RAE \
  --ensemble caruana --correlation
```

Add `--predict-blind --predictions-out <file>` to write predictions for the blinded test set.

`--split blind_matched` exists on branch `feat/blind-matched-split` (commit `a7a8158`) and is not yet in
`main`. Run the final command from a checkout that has it. On a GPU node, TabPFN needs a torch build that
supports the GPU (the Blackwell cards are not supported by torch 2.5.1).
