# Workstream Manifest — OpenADMET CYP Inhibition Challenge

Eight parallel tasks, scoped for disjoint file ownership so several people can
work at once without stepping on each other. Each has its own file in this
directory with full context.

Source of the gap analysis: `reports/PXR-SUMMARY.md` (survey of competitor
methods from the sibling PXR challenge) and `reports/head_search_analysis.md`
(our own auxiliary-head results).

---

## Checklist

- [ ] **[T1 — Wire the unused challenge assays into the union](T1-assay-heads.md)**
      TDI, Emax and single-concentration are on disk and reach nothing. *No deps.*
- [x] **[T2 — Representation portfolio](T2-representations.md)**
      Morgan/ECFP4, count-Morgan, MACCS, Mordred. `INPUT_REGISTRY` now has six featurizers. *No deps.*
- [ ] **[T3 — Model zoo](T3-model-zoo.md)**
      LightGBM/XGBoost/Ridge/TabPFN. The registry contains one decision tree. *No deps.*
- [ ] **[T4 — OOF harness and stacking](T4-ensembling.md)**
      No ensembling exists at all. *Interface-blocked on T3, not code-blocked.*
- [ ] **[T5 — Structure generation: co-folding and docking](T5-structure-generation.md)**
      IntelliFold-2 / Protenix / Uni-Mol Docking V2. **Heavy GPU — SCC.** *No deps.*
- [ ] **[T6 — Structure-derived features](T6-structure-features.md)**
      Heme-iron coordination geometry and pocket contacts. *Needs T5's manifest.*
- [ ] **[T7 — Post-hoc processing and label uncertainty](T7-posthoc-uncertainty.md)**
      Prediction clipping; the unused `_std` columns. *No deps.*
- [ ] **[T8 — Chemical space exploration](T8-chemical-space.md)**
      What chemistry is in play and where the model is blind. *No deps.*

**Critical path is T5 → T6.** Start it first even though it is the riskiest arm;
everything else is bounded work and that one is not.

---

## Shared contract — read before starting anything

**One metric.** Macro RMSE over the four scored heads
(`CYP{1A2,2C9,2D6,3A4}_pIC50_direct_inhibition`), on the pinned split.

| Reference point | Macro RMSE |
|---|---|
| No auxiliary heads (baseline) | **0.9430 ± 0.0018** |
| Current best (`CYP1A2+CYP2C8`) | **0.7184 ± 0.0038** |

**A gain under ~0.004 is seed noise.** Report at least 3 seeds and their spread.
A single-seed improvement is not a result. See `reports/head_search_analysis.md`
for the full 55-config table these numbers come from.

**Never re-split.** Use the pinned `split` column produced by `add-split-column`.
Re-splitting per experiment makes split noise the dominant term and renders the
eight workstreams mutually incomparable. `head-search` refuses to run without it.

**Join on `inchikey_block`; dedupe within a source on the full InChIKey.**
Use `data_tools.standardize`. Anything keyed on raw SMILES silently under-merges —
salts, tautomers and stereo variants are spelled differently by every source.
Stereoisomers must *not* collapse: quinidine and quinine share a connectivity
block and differ by 2.56 log units on CYP2D6.

**`data/test.csv` is untouchable.** No new data source may contain a molecule
overlapping it. `build_union` already enforces this; new ingesters must call the
same check (`check-overlap`).

**Shared files.** Only two files are touched by more than one task:

| File | Touched by | Rule |
|---|---|---|
| `src/data_tools/inputs.py` (`INPUT_REGISTRY`) | T2, T6 | Append one line. Never reorder. |
| `src/models/__init__.py` (`REGISTRY`) | T3 | Append one line. Never reorder. |

Everything else is owned by exactly one task.

**`data/` and `results/` are gitignored.** Commit code and reports, not
artifacts. Do not commit results to GitHub.

---

## Running on the BU SCC

Several tasks need GPU time beyond what a laptop provides. The existing pattern
is `scripts/head_search.qsub` + `scripts/run_head_search.sh`.

**If your task needs the SCC, contact Denali with your code** — he has the
project allocation and will get it queued. Do not burn days fighting the
scheduler yourself.

Tasks flagged **Heavy GPU**: T5 (by far the largest), T6, T1, T4.
Tasks with moderate GPU needs: T3 (TabPFN inference), T7.
CPU-only: T2, T8.
