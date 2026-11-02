# Prediction correlation across representations

How much do two models actually disagree, per molecule, when the only thing
that changed is the representation? Spearman ρ between their validation
predictions answers it. Low ρ between two arms that score similarly is the
precondition for an ensemble gain; high ρ means a second arm costs compute and
buys nothing.

## Why this matters for the challenge

`reports/t2_solo_representations.md` already carries a Spearman matrix, and
flags that the source workbook "does not specify the underlying vectors or how
the four isoform correlations were aggregated, so this report preserves the
workbook's label without interpreting them as raw-feature or prediction
correlations." That ambiguity is the gap. A correlation between *raw feature
columns* and a correlation between *model outputs* are different quantities
answering different questions, and only the second one predicts whether
blending two arms helps. This brief defines the second and writes down the
aggregation, so the number is reproducible.

T4 (ensembling) needs this as its input: it decides what to blend, and blending
two arms with ρ ≈ 0.95 is wasted work.

## How it fits the current repo

Two things in the current code have to change first.

**1. `evaluate-models` discards predictions.** `_evaluate_model` in
`src/models/evaluate.py` fits, predicts, computes RMSE/MAE/R2 and returns only
the metrics dict. The per-molecule vector — the thing being correlated — is
never persisted. Add a `--predictions-out <path>` flag that writes one tidy
long-format CSV, with the columns needed to pivot later:

```
smiles,isoform,model,input,split,seed,y_true,y_pred
```

Write SMILES, not a positional index. The next point is why.

**2. The surviving rows differ per representation.** `drop_nan_rows` masks on
`np.isfinite(X).all(axis=1) & np.isfinite(y)`. The `y` half is identical across
representations for a given isoform, but the `X` half is not. Every
representation leaves an all-NaN row for an unparsable SMILES, so those drop
everywhere — but they diverge on molecules that *do* parse. `mordred.transform`
calls `fill_missing(0.0)` and then `row[~np.isfinite(row)] = 0.0`, so a
parseable molecule always survives; `rdkit_descriptors.transform` writes raw
descriptor values with no finite check, so a descriptor returning inf or NaN
(Ipc overflow, partial charges on exotic atoms) drops that row. The
fingerprints are always finite. Two prediction vectors from different `--input`
values are therefore **not row-aligned, and may not even share a length**.
Correlating them positionally silently produces a meaningless number.

So: key every prediction by SMILES, then inner-join on SMILES before
correlating, and report `n` alongside each ρ. If the intersection is much
smaller than either arm's val set, say so — the ρ describes only the shared
molecules.

## Method

For a fixed (isoform, model, split, seed), with one prediction vector per
representation:

1. Inner-join the arms on SMILES. Keep `n` for the report.
2. Rank-transform each vector, averaging ranks within ties — Morgan-based arms
   produce genuine ties, and dropping tie correction biases ρ upward.
3. ρ = Pearson on the ranks. `scipy.stats.spearmanr` does both steps and is
   already available transitively through scikit-learn; add `scipy` to
   `requirements.txt` explicitly if you import it directly, since relying on
   another package's transitive dep is how it breaks later.
4. Repeat per isoform. Report **all four isoform values and their mean**, and
   state that the aggregate is an unweighted mean of per-isoform ρ. Do not
   report the mean alone — that is precisely the ambiguity in the T2 workbook.
5. Repeat over the ≥3 seeds the round-1 contract requires, and report spread.
   A ρ that moves more between seeds than between representation pairs is not
   evidence of anything.

Run the same procedure on **residuals** (`y_true - y_pred`) as well as on raw
predictions, and report both. Raw predictions correlate highly almost by
construction, because every arm is mostly recovering the same target ordering;
residual correlation is what actually tells you whether two arms fail on the
same molecules, which is what determines ensemble benefit.

## What not to do

- Do not correlate across isoforms, or pool the four into one vector. Different
  targets have different scales and different row counts.
- Do not read ρ as a performance claim. A weak arm can be decorrelated simply
  by being noisy. Always report ρ next to each arm's RMSE.
- Do not quote a single-seed ρ as a finding. Same rule as every other number in
  this repo.

## Done when

A table, per isoform and averaged, of pairwise ρ over the six registered
representations for one fixed model — plus the residual-correlation counterpart,
`n` per cell, and seed spread. Enough for T4 to pick blend candidates without
re-running anything.

One caveat to state in the write-up: `evaluate-models` exposes only `random`,
`scaffold` and `butina`, not the pinned `split` column, so until the evaluation
harness in `MANIFEST.md` ("Not yet written") exists, these correlations are
descriptive single-contract results and are not comparable to the
`0.7557` / `0.7193` reference numbers.

## Sources

- `reports/t2_solo_representations.md` § "Pairwise representation correlation" —
  the undocumented matrix this brief replaces.
- `reports/PXR-SUMMARY.md` § on ensembling — correlated arms contributing
  nothing to a blend is a repeated finding in the surveyed entries.
