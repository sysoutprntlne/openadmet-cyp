# T4 — OOF harness and stacking

**Depends on:** T3 for real base models. **Not code-blocked** — build and test
against synthetic OOF predictions, then swap in real ones.
**GPU:** yes, moderate — generating OOF predictions means retraining per fold.
**Owns:** `src/models/oof.py`, `src/models/stack.py` (both new).

---

## Why this matters for the challenge

There is no ensembling in this repo. No out-of-fold prediction storage, no
stacking, not even averaging across seeds. `evaluate.py` fits a model, prints
RMSE/MAE/R², and stops. `predict.py` fits one model per isoform and writes the
submission directly from it.

`reports/PXR-SUMMARY.md` §6 opens: *"Virtually all competitive submissions are
ensembles. The question is how to combine model outputs without overfitting the
combination weights."* Every single named competitor in the top 40 of that
survey submitted a blend.

We already know our models disagree usefully. `reports/head_search_analysis.md`
§3 shows two aux configurations that reach similar macro RMSE by being good at
*different tasks* — `CYP1A2`-aux gets CYP2C9 to 0.5300 while `CYP3A4`-aux gets
it to 0.6082, yet they are within 0.026 macro. That is exactly the error
diversity a stacker converts into a real gain, and we currently throw it away by
picking one configuration and discarding the rest.

This task is also the thing that makes the other seven cumulative rather than
competitive. T1, T2, T3, T6 each produce candidate models; without a stacker,
they are seven alternatives and we ship one. With a stacker, they compose.

## How it fits the current repo

Genuinely new surface — nothing here to extend, which is why the interface
contract matters more than usual.

- The pinned `split` column from `add-split-column` defines the folds. It is
  already designed for this: only molecules carrying at least one *scored*
  label can land in validation, and auxiliary-only molecules always go to
  train, so a fold-averaged metric answers the right question.
- `src/models/head_search.py` already implements the discipline this task needs
  — pinned splits, identical evaluation rows across configurations, repeated
  seeds compared against their spread. Read its module docstring before
  designing anything; the same three rules apply, and it is the closest thing
  in the repo to prior art.
- Base model predictions arrive from `REGISTRY` models (T3) and from chemprop
  runs (`results/chemprop/*/model_*/test_predictions.csv`). The harness must
  ingest both.

**Interface contract — agree this with T3 and T6 before generating anything.**
One long-format CSV:

```
model_id, inchikey_block, fold, target, pred
```

`model_id` must encode enough to reconstruct the run (model name, feature set,
aux-head config, seed). Long format rather than wide because models cover
different molecule subsets — sklearn models drop NaN-label rows per isoform,
chemprop does not.

## Scope

1. **OOF generation harness.** For each registered model × feature set, train
   on k−1 folds, predict the held-out fold, write the long CSV. Must be
   resumable — these runs are long and the scheduler will kill some of them.
2. **NNLS stacking first.** Non-negative least squares is the survey's most
   widely adopted method for good reasons: non-negativity prevents
   destabilizing negative weights, it naturally produces sparse solutions so
   only strong and diverse models get weight, and it has no hyperparameters to
   tune. `scipy.optimize.nnls`. Per-isoform weights, not one global set.
3. **Ridge and ElasticNetCV as alternatives.** discoverybytes used Ridge
   (final coefficients: chemprop 0.706, MolE 0.206, UniMol 0.166); ElasticNet's
   L1 term zeroes redundant models.
4. **Port the anti-leakage filters.** ldbc1999 filtered 147 trained models to 5
   survivors, and the filters are the useful part:
   - quality: drop models with R² < 0.3 on held-out data;
   - anti-leakage: drop models where `heldout_RAE / OOF_RAE > 2.0`;
   - deduplication: cluster at Pearson ρ > 0.95 and replace each cluster with
     its mean.
5. **Seed averaging**, which is nearly free and which we currently do not do.

## What not to do

**Do not distill the ensemble into a single model.** `reports/PXR-SUMMARY.md`
§6 documents this backfiring precisely: discoverybytes trained a new chemprop on
the ensemble's predictions as pseudo-labels, expecting it to absorb what the
other models knew. When Ridge then stacked the distilled model with the
originals, it found them redundant — the distilled model had already absorbed
their signal — and slashed their weights. Ensemble diversity was destroyed,
ΔRAE +0.0126. Training any model on another model's predictions makes it
non-independent and removes the only thing that justified combining them.

**Do not fit the stacker on predictions the base models saw in training.** That
is the leakage the `heldout_RAE / OOF_RAE` filter exists to catch.

## Tools

- `scipy.optimize.nnls`; `sklearn.linear_model` (`RidgeCV`, `ElasticNetCV`).
- `polars` for the OOF tables.
- `src/models/head_search.py` as the reference for split discipline and
  multi-seed comparison.

## Done when

- OOF predictions exist for every T3 model and at least the top three chemprop
  aux configurations from `reports/head_search_analysis.md`.
- NNLS stack beats the best single model by more than the seed spread (~0.004).
  If it does not, that is a real and reportable result — say so rather than
  tuning until it does.
- Learned weights are written out and interpreted: which models earned weight
  and which were dropped as redundant is the interesting output, not just the
  final number.

## SCC note

**Contact Denali with your code to run this on the BU SCC.**

Generating OOF predictions multiplies every base model by the fold count, and
for chemprop members by the seed count on top of that. This is the second
largest GPU consumer after T5. Reuse `scripts/head_search.qsub`, keep
`#$ -l gpu_c=7.0`, and make the harness resumable before submitting — a
48-hour job that loses everything on a node failure will cost you a week.

The stacker itself is CPU-trivial; it is only the OOF generation that needs the
cluster. Develop the stacking code locally against synthetic predictions.
