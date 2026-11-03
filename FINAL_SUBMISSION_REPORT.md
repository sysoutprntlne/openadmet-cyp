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
(`--model chemeleon` is chemprop started from the CheMeleon weights):

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

The three configurations, all with the challenge heads:

| Family | Data besides the challenge heads | Merged |
|---|---|---|
| `grid_adme_chem` | HLM, RLM, ExpansionRx CLint, AZ clearance, half-life, 3 CYP substrate sets | 20,489 molecules, 13 heads (5 binary) |
| `grid_aux_chem` | PubChem, ChEMBL, BindingDB, TDI, single-concentration | 39,575 molecules, 62 heads |
| `grid_both_chem` | both of the above | |

Output: one `results/chemprop/grid_chem/<config>/seed{0,1,2}/` per config, in the order given.

**2. `huber_soft` and `mae_admet`.** Chemprop, 60 epochs, seeds 0 1 2, trained on the challenge labels plus
three auxiliary configurations. Each configuration is one batch of three seeds: `_` is the first, `_2` the
second, `_3` the third.

```bash
CFG1="--datasets challenge pubchem:CYP1A2 chembl:CYP1A2 bindingdb:CYP1A2,CYP2C8"
CFG2="--datasets challenge pubchem:CYP1A2,CYP2C9 chembl:CYP1A1,CYP2B6 bindingdb:CYP2C8,CYP1A1,CYP2B6"
CFG3="--datasets challenge pubchem:CYP1A2,CYP2D6 chembl:CYP1A2,CYP2D6 bindingdb:CYP1A2,CYP2D6,CYP2C8"

# huber_soft: Huber loss plus soft labels (label noise and shrink)
qsub -N cyp_huber_soft -o cyp_huber_soft.qlog scripts/train_model.qsub --model chemprop --epochs 60 \
  --n-seeds 3 --seed 0 --loss-function huber --huber-delta 0.5 --label-noise 0.2 --label-shrink 0.1 \
  --output-dir results/chemprop_12/huber_soft $CFG1 $CFG2 $CFG3

# mae_admet: MAE loss
qsub -N cyp_mae -o cyp_mae.qlog scripts/train_model.qsub --model chemprop --epochs 60 \
  --n-seeds 3 --seed 0 --loss-function mae \
  --output-dir results/chemprop_12/mae $CFG1 $CFG2 $CFG3
```

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
  --ensemble caruana --correlation --predict-blind --predictions-out results/final.csv
```

## Pharmacophores

Five pharmacophore feature blocks were built and tested as additions to the RDKit descriptors. Four are
hand-written SMARTS or descriptor blocks (`src/representations/cyp_pharmacophore.py`); the fifth is a mechanistic
block ported from another team (`src/representations/cyp_mech.py`). Every motif and descriptor carries a
literature source, and the motif list was fixed before any validation scores were seen. The full motif table
with SMARTS, sources and train/test prevalence is in `reports/pharmacophore_features.md`.

| `--input` | Features | What it encodes |
|---|---:|---|
| `cyp_heme_v2` | 13 | Heme-coordination motifs: counts of heme-ligating nitrogens (imidazole, 1,2,4- and 1,2,3-triazole, tetrazole, pyridine, pyrimidine, thiazole, oxazole), total, hindered and unhindered donor nitrogens, and primary and basic aliphatic amines |
| `cyp_tdi_v2` | 6 | Mechanism-based inactivation alerts: methylenedioxyphenyl, terminal alkyne, furan, thiophene, hydrazine, tertiary alkylamine |
| `cyp_isoform_v2` | 10 | Isoform-preference descriptors: basic amine count (CYP2D6), acidic group count (CYP2C9), aromatic ring count and fraction (CYP1A2), fraction Csp3, molecular weight (CYP3A4), cLogP, TPSA, H-bond acceptors and donors |
| `erg` | 315 | RDKit ErG pharmacophore fingerprint at library defaults, the off-the-shelf comparator for the hand-written blocks |
| `cyp_mech_v1` | 30 | Mechanistic block: 16 functional-group counts (amine classes, amidine, guanidine, pyridine and imidazole nitrogen, morpholine, piperazine, piperidine, carboxylic acid, tetrazole, sulfonamide), 8 protonation-state columns (rule-based pKa of the most basic centre, fraction protonated at pH 7.4, net charge, cation and anion counts), and 6 basic-nitrogen-to-aryl geometry columns including the CYP2D6 pharmacophore flags |

### Sources

- **Heme coordination:** Monk et al. 2020 (azoles), Mohamed et al. 2023 (pyridine and imidazole ligands),
  Podgorski et al. 2020 and Perrin et al. 2009 (steric accessibility of the donor nitrogen). Pyrimidine, thiazole
  and oxazole are extrapolations from those sources and are marked unvalidated in the motif table.
- **Inactivation alerts:** Mao 2025, Mirzaei et al. 2021, Dang et al. 2017.
- **Isoform descriptors:** Paine et al. 2003 (CYP2D6), Wester et al. 2004 (CYP2C9), Sansen et al. 2007 (CYP1A2),
  Yano et al. 2004 (CYP3A4), Steyn et al. 2020 and Kiani et al. 2019 (lipophilicity, size, H-bonding).
- **`cyp_mech_v1`:** ported from the Lizard Wizard Gizzard team's open-code CYP Challenge submission,
  [doctawho42/CYPChallenge](https://github.com/doctawho42/CYPChallenge) (MIT License, Copyright (c) 2026 the
  CYPChallenge contributors), at commit `bf018b8`. The SMARTS tables, pKa class table and penalty weights, column
  definitions and their grouping are theirs; here they are adapted to the `Representation` interface. See
  `THIRD_PARTY_NOTICES.md`.

### Comparison against base RDKit

Each block was added to the RDKit descriptors and fitted with XGBoost, against RDKit alone. The last group stacks
the three hand-written SMARTS blocks. Features within one `--input` group are concatenated, and each group is
scored on its own:

```bash
evaluate-models --models xgboost --nseeds 3 --sort-by RMSE \
  --input rdkit \
  --input rdkit cyp_heme_v2 \
  --input rdkit cyp_tdi_v2 \
  --input rdkit cyp_isoform_v2 \
  --input rdkit erg \
  --input rdkit cyp_mech_v1 \
  --input rdkit cyp_heme_v2 cyp_tdi_v2 cyp_isoform_v2
```
