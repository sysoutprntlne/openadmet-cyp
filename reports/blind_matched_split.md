# Can we validate on a split that looks like the blind set?

Generated 2026-11-02. Code: `data_tools.split_column.assign_blindlike_splits`,
`data_tools.shift_weights`. Reachable as `train-model --split-strategy blind_matched`.

**Short answer: partly, and it does not change which model you would pick.** The pinned
scaffold split is pessimistic by about 0.013 macro RMSE, it ranks models the same way a
blind-matched split does, and the blind set is not distribution-shifted in any of the
usual senses. The practical consequence is that post-hoc recalibration aimed at
distribution shift is unlikely to pay here.

---

## 1. What the blind set actually is

`data/test.csv` (750 molecules) against `data/train.csv` (4,905), Morgan r2/2048, Tanimoto.

**It is not shifted in property space.** Every bulk property sits within 0.35 sd of train:
MW −0.18 sd, cLogP +0.04, TPSA +0.06, HBD +0.32, rotatable bonds +0.15, aromatic rings
+0.11, heavy atoms −0.21.

**It is not far from train.** Exactly **1 of 750** molecules has a nearest training
neighbour below 0.4. Deciles of nearest-neighbour similarity to train:

| 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|
| 0.516 | 0.532 | 0.556 | 0.571 | 0.587 | 0.604 | 0.627 | 0.649 | 0.702 |

**It is scaffold-novel.** Only **24.5%** of blind molecules (184/750) stand on a Murcko
scaffold that appears in train; there are 546 distinct scaffolds among 750 molecules, and
Butina clustering at 0.4 gives 352 clusters with 232 singletons. So it is not an
analog-series set in the way the PXR test set was (513 analogs of ~63 scaffolds).

**The gap that matters is per-head, and pooling hides it.** All four heads must be
predicted for all 750 molecules, but each head's labelled training subset is a fraction of
the table. Measured against *that head's* labelled molecules only:

| head | train labels | blind NN median | fraction < 0.4 |
|---|---|---|---|
| CYP3A4 | 2,335 | 0.538 | 21% |
| CYP1A2 | 1,412 | 0.517 | 29% |
| CYP2C9 | 1,285 | 0.518 | 35% |
| **CYP2D6** | 1,493 | **0.471** | **42%** |

The pooled figure (median 0.571) is an artifact of four heads' molecules being counted
together. CYP2D6 is also the head whose binding determinant the blind set is depleted in:
a basic nitrogen is present in 44.9% of train but 33.7% of blind molecules. CYP2D6 is
therefore doubly exposed — worst neighbour coverage and a composition shift in the one
feature that drives its SAR.

## 2. A split matching the blind regime: one axis works, one is unreachable

`assign_blindlike_splits` holds out whole Murcko families (making those molecules
scaffold-novel) plus part of other families (leaving their scaffold in train), choosing
which families by greedily centring the held-out similarity median. Targets are measured
from `data/test.csv` at runtime rather than hardcoded.

| | novel scaffolds | median NN to train |
|---|---|---|
| pinned scaffold split | 100% | 0.449 |
| **blind-matched split** | **75.4%** | 0.469 |
| blind set (target) | 75.5% | 0.587 |

Scaffold novelty matches to within 0.1 pp. Similarity does not, and **it cannot**:

| | median NN |
|---|---|
| train to itself, leave-one-out | 0.450 |
| train, excluding own scaffold family | 0.446 |
| best possible 981-molecule holdout | **0.580** |
| blind set to train | **0.587** |

The blind set is closer to train than train is to itself. Only **8.8%** of training
molecules (434) have a cross-family neighbour at 0.587 or above, where half the blind set
does by definition. The 0.580 ceiling is reached only by taking the single densest 981
molecules (minimum 0.525), which would strip train of exactly its crowded regions. No
representative subset of the training data reproduces the blind set's density.

## 3. Importance weighting covers the axis the split cannot

`shift_weights` re-weights a held-out slice so its (scaffold novelty, similarity) mix
matches the reference, by density ratio over cells with similarity bin edges taken from
the reference's own quantiles. Weights are not bounded by what train contains, so they can
reach a regime no subset can.

| held-out slice | novel scaffolds | median NN | blind mass with no support | ESS (of ~980) |
|---|---|---|---|---|
| pinned | 100% | 0.439 | — | 980 |
| pinned + weights | 100% | 0.557 | **19.5%** | 302–434 |
| blind-matched | 75.4% | 0.469 | — | 980 |
| **blind-matched + weights** | **80.1%** | **0.572** | **0.0%** | 374–424 |

The two are complementary. The pinned split holds out whole families, so every held-out
molecule is scaffold-novel and no weighting can bring 100% down to the blind set's 79.2% —
that is where the 19.5% unsupported mass comes from. On the blind-matched split the weights
have support everywhere and unsupported mass goes to zero.

**But the weighted metric is not usable as a headline number.** Effective sample size falls
from 980 to ~300–420, seed spread roughly triples (±0.017 → ±0.048), and the answer depends
on an arbitrary bin count: the similarity correction is −0.026 at 3 bins and −0.004 at
8 bins. The 8-bin figure is the stable one (±0.002), so once novelty is matched by the
split, the residual similarity correction is small.

## 4. Does it change any decision? No

Three models on Morgan r2/2048, per-head RandomForest-style fits, macro RMSE over the four
scored heads, **5 split seeds**, paired within each seed so the split draw cancels.

| model | pinned | blind-matched | shift |
|---|---|---|---|
| hist_gbm | **0.8565 ± 0.0158** | **0.8470 ± 0.0120** | −0.0095 |
| random_forest | 0.8693 ± 0.0148 | 0.8490 ± 0.0166 | −0.0204 |
| ridge | 0.8703 ± 0.0094 | 0.8599 ± 0.0152 | −0.0104 |

Both splits rank them `hist_gbm < random_forest < ridge`. Paired differences:

| pair | pinned | blind-matched |
|---|---|---|
| ridge − hist_gbm | +0.0138 ± 0.0069 (same sign, 5/5) | +0.0129 ± 0.0098 (same sign, 5/5) |
| ridge − random_forest | +0.0010 ± 0.0074 (sign flips) | +0.0109 ± 0.0164 (sign flips) |
| random_forest − hist_gbm | +0.0128 ± 0.0087 (same sign, 5/5) | +0.0020 ± 0.0121 (sign flips) |

The only difference between the axes runs the wrong way: the blind-matched split *loses* a
separation the pinned split resolves, because random_forest gains twice as much from the
easier regime as the other two. It is less discriminating, not more.

**Optimism is ~0.013, not the 0.02 a single model suggests.** The shift is −0.0095 to
−0.0204 depending on the model, averaging −0.0134 — about 1.5% of RMSE. There is no single
constant to quote, and random_forest is the most shift-sensitive of the three.

## 5. What this says about recalibration

Three independent measurements point the same way: the blind set is not shifted in property
or fingerprint space, the pinned split's bias is ~1.5%, and a blind-matched validation axis
changes no model ranking. Affine recalibration, variance rescaling and related post-hoc
corrections are aimed at a distribution shift this data does not show, which is consistent
with `reports/PXR-SUMMARY.md` §7, where k=1.10 variance scaling worsened RAE and
variance-matching "dramatically hurt blind-test performance".

The per-head analysis in §1 points somewhere else: CYP2D6 and CYP1A2 have the thinnest
labelled coverage, the worst neighbour support, and the largest per-head gaps throughout.
That looks like ordinary data starvation on two heads rather than distribution shift.

## 6. Caveats, and what was not tested

- **One fingerprint** (Morgan r2/2048). Other fingerprints shift the absolute similarities.
- **sklearn models on a single representation, not the real pipeline.** The ranking test
  used ridge / random_forest / hist_gbm, not a multi-task D-MPNN with auxiliary heads.
- **Model families were tested; head subsets were not.** The decisions this repo actually
  makes are about auxiliary head subsets and representations. A split could reorder those
  without reordering model families. This is the most important untested case.
- **Evaluation-side only.** Selecting or up-weighting *training* molecules toward test-like
  chemistry is a different intervention that changes the model rather than the measurement.
  Nothing here tests it, and it is not ruled out by these results.
- **Five seeds**, against a split-to-split spread of ~0.015 — four times the ~0.004
  training-seed noise floor the manifest quotes. Anything compared across split
  realisations needs more seeds than a head-search result does.
- `extra_trees` scored 1.12–1.14 in an earlier 4-model pass, far outside the others, and was
  dropped as misconfigured for 2048-bit input rather than reported as a result.
- Absolute RMSE here (~0.85) is **not** comparable to the chemprop reference numbers
  (no-aux `0.7557 ± 0.0036`, best `CYP1A2` `0.7193 ± 0.0065`). Only the deltas transfer.

## 7. Reproducing

```bash
# the split, as a column on any table carrying SMILES + the scored heads
python -c "
import polars as pl, sys; sys.path.insert(0, 'src')
from data_tools.split_column import assign_blindlike_splits
train = pl.read_csv('data/train.csv', infer_schema_length=None)
reference = pl.read_csv('data/test.csv', infer_schema_length=None)['SMILES'].to_list()
scored = [f'CYP{i}_pIC50_direct_inhibition' for i in ('1A2','2C9','2D6','3A4')]
splits, report = assign_blindlike_splits(train, scored, reference, seed=0)
print(report)"

# or during training, opt-in; the default stays the pinned scaffold split
train-model --model chemprop --datasets admet --split-strategy blind_matched
```
