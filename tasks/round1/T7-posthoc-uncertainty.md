# T7 — Post-hoc processing and label uncertainty

**Depends on:** nothing.
**GPU:** light. Weighting experiments need chemprop reruns; clipping is free.
**Owns:** `src/models/predict.py`, `src/data_tools/weights.py` (new).

---

## Why this matters for the challenge

Two cheap wins sitting untaken, one of which costs nothing at all.

**Prediction clipping.** `src/models/predict.py` writes raw model output to the
submission. `reports/PXR-SUMMARY.md` §7 opens by calling clipping to the
observed training range "nearly universal and uncontroversially useful — it
prevents physically unreasonable extrapolation at the extremes with no
downside." It is a handful of lines and it strictly cannot hurt: a predicted
pIC50 of 11 for a CYP assay is not a bold prediction, it is an error, and
clipping it to the observed ceiling reduces RMSE for free.

**Label uncertainty.** `data/train.csv` carries, for every isoform,
`_std`, `_conf_high` and `_conf_low` columns. Nothing in this repo reads any of
them. We currently treat a pIC50 measured with a tight confidence interval and
one fitted from a barely-converged curve as equally authoritative.

The survey's §1 covers this directly: ranga_31489 used sample weights
`w = 1/(SE² + σ_model²)` with σ_model = 0.60 to account for heteroscedastic
measurement error, and discoverybytes downweighted 474 high-SE compounds to
0.3–0.5. Neither reports a dramatic gain on its own, but both are the kind of
small, safe, compounding improvement that separates a good submission from a
mediocre one — and we have the columns already.

There is a third, larger prize with a sharper edge — see the inactive
overestimation section below.

## How it fits the current repo

- `src/models/predict.py` already has the four scored targets and the per-isoform
  fitting loop. Clipping is a post-prediction transform right before the
  submission CSV is written. Compute the bounds from the training labels, per
  isoform, not globally.
- `src/models/train_chemprop.py` passes arbitrary flags through to the
  `chemprop train` CLI, so per-row weighting needs a weights *column* in the
  data file, not new CLI plumbing. Check the current chemprop weight-column
  flag name against the installed version before building around it.
- `src/data_tools/build_union.py` is where a weights column would be emitted.
  Coordinate with T1 — that task also modifies the union's column set, and you
  both want the Hill-fit R² values from its single-concentration work as a
  natural weight source.
- Sklearn models in `REGISTRY` take `sample_weight` in `fit()`; the `CYPModel`
  interface in `src/models/base.py` does not currently expose it. Extending the
  interface touches T3's files — agree the signature change with that owner
  first.

## Scope

1. **Clipping.** Per-isoform bounds from observed training labels. Do this
   first; it is an afternoon and it is pure downside protection.
2. **Sample weights from measurement error.** `w = 1/(SE² + σ²)`, with SE from
   the `_std` columns and σ tuned as a single hyperparameter. Emit as a column
   in the union. Evaluate over ≥3 seeds.
3. **Confidence-interval width as a filter.** Identify compounds whose
   `conf_high − conf_low` is pathological and test excluding versus
   downweighting them. Prefer downweighting — see the warning below.
4. **The inactive overestimation problem.** §7 calls this "the most important
   structural post-processing finding." ldbc1999 found that 37 of 253 revealed
   PXR test compounds were truly inactive (true mean pEC50 2.670) while the
   ensemble predicted them near the training mean (~4.6) — errors of 1.3–1.9
   log units. Their fix, a binary classifier gating predictions to a specialist
   inactive-space model, cut revealed-RAE from 0.5424 to 0.5259, the largest
   post-hoc gain in the survey.

   **Check whether we have the same pathology before building the machinery.**
   Plot our predicted-vs-actual in the low-pIC50 tail on the pinned split. The
   CYP deck's inactive compounds may be censored differently than PXR's were.
   If the pathology is absent, say so and stop — that is a one-day result.

## Approach with caution

Three post-hoc moves have strong negative prior art in `reports/PXR-SUMMARY.md`.
None of them are off the table, but the evidence is bad enough that each needs a
specific reason and a success criterion agreed *before* you spend time on it —
and worth noting, all three failures came from people who had good reasons to
expect them to work.

- **Variance rescaling / dynamic-range recalibration.** Real phenomenon, and the
  obvious fix appears to be the wrong one. discoverybytes tested k=1.10 variance
  scaling and test RAE *worsened* by +0.01, with Spearman also down. firstpass
  reported variance-matching and quartile mapping "dramatically hurt blind-test
  performance." *If you pursue it:* the compression is genuinely there, so the
  case would have to be for a better-targeted correction than global scaling,
  validated against a calibration set matched to the test distribution.
- **Isotonic calibration.** discoverybytes and firstpass both tried it, found it
  harmful or neutral, and abandoned it. But RyeCatcher made it work, and the
  difference is the protocol: fit isotonic *only* on left-out fold predictions,
  never on full-training OOFs, and do not compound fits. Fitting on
  full-training OOFs overestimates the improvement by ~0.009 RAE per step and
  the bias accumulates. *If you pursue it:* use the honest per-fold version or
  not at all — the failures above are all the naive variant.
- **Nearest-neighbour activity transfer.** The single worst result in the
  survey. RyeCatcher copied pIC50 from high-Tanimoto training actives; the test
  set contained SAR misses — compounds at Tanimoto 0.50–0.58 that are
  biologically inactive because the binding mode is incompatible with the
  pocket. OOF 0.4536 → LB 0.658, rank ~40 → ~87. *If you pursue it:* this one
  has the most asymmetric downside of the three, and the precondition is T8
  showing our test set lacks that SAR-miss structure. Treat T8's answer as the
  gate rather than assuming either way.

The common thread is that all three move predictions after the model has
spoken, trading a small expected gain against a large tail risk. Suggested
ordering: get clipping and weighting done first, then revisit these if there is
budget left and T8 has given you a reason to.

**On filtering generally:** discoverybytes evaluated PAINS/REOS filtering and
*rejected* it. Removing 841 flagged compounds degraded LightGBM from OOF RAE
0.609 to 0.724, because those structurally "problematic" compounds anchor the
inactive end of the range and calibrate the dynamic range. Remove measurement
noise; keep structural diversity even when it looks ugly.

## Tools

- `polars`, `numpy` — weight computation and clipping.
- `matplotlib` — the predicted-vs-actual tail plot for step 4. Already a
  dependency and already used in `predict.py`.
- `head-search` / `evaluate-models` for evaluating weighted runs.

## Done when

- Clipping is in `predict.py` with per-isoform bounds.
- Weighting scheme evaluated over ≥3 seeds against unweighted baseline; kept
  only if the gain exceeds the ~0.004 seed spread, reported either way.
- A written verdict on whether the inactive-overestimation pathology exists in
  our data, with the plot.

## SCC note

Clipping and the diagnostic plots are local work.

The weighting experiments are chemprop reruns, so **contact Denali with your
code** if you need more than a couple of configurations. Reuse
`scripts/head_search.qsub` and keep `#$ -l gpu_c=7.0`.
