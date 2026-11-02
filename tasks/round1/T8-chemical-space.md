# T8 — Chemical space exploration

**Depends on:** nothing. Feeds T1, T5 and T7.
**GPU:** no. CPU-only.
**Owns:** `scripts/chemspace.py`, `reports/chemical_space.md`.

---

## Why this matters for the challenge

This is not bookkeeping. The question is *what chemistry is actually in play,
and where is the model flying blind* — and right now nobody on this project can
answer either.

**We do not know what kind of problem the test set is.** `data/test.csv` is 750
compounds and no analysis in this repo characterizes them. This matters more
than it sounds, because nearly every methodological lesson in
`reports/PXR-SUMMARY.md` is contingent on test-set structure. The PXR test set
was 513 analogs of ~63 potent training scaffolds, and that single fact drove:

- why random KFold severely overestimates and Butina clustering became the
  dominant CV (§5),
- why TabPFN works so well — it sees all training labels at prediction time and
  captures analog-series patterns by direct comparison (§3),
- why the pairwise/delta model failed catastrophically — 90.8% of test
  compounds had no training neighbour within Tanimoto 0.7, so it extrapolated
  outside its intended regime and returned Spearman −0.071 (§3),
- why nearest-neighbour activity transfer destroyed RyeCatcher's rank (§7).

If our test set is scaffold-novel rather than analog-dense, half that survey
does not transfer and we should stop treating it as a playbook. If it *is*
analog-dense, then TabPFN (T3) should be prioritized and the cliff-handling
problem becomes central. **Either answer redirects several other tasks**, which
is why this is worth doing early despite producing no model.

**Our own report has an open question waiting on this.**
`reports/head_search_analysis.md` §4 and §6 found that row count does not
predict which auxiliary head helps — the pattern is bimodal, and the report's
own hypothesis is that chemical-space overlap with the challenge set is the
real driver. It explicitly parks this as untested and names the experiment:
*"compute Tanimoto similarity between each aux head's molecules and the
challenge molecule set, and re-plot solo-head gain against that instead of row
count."*

That experiment either confirms or kills the repo's current working theory of
why auxiliary heads work. Concretely it should explain the two anomalies in
that report: why CYP3A4 (40,432 rows) is the *worst* of the five qHTS heads
while CYP1A2 (30,958) is the best, and why CYP2C8 (888 rows, useless alone at
0.9447) is the single best addition on top of CYP1A2 (0.7313 → 0.7184).

## How it fits the current repo

Analysis task — reads everything, changes no model code.

- Molecules from `data/train.csv`, `data/test.csv`, `data/union_train.csv`
  (20,871 molecules, 14 heads) and the family sources.
- **Reuse the cached embeddings.** `data_tools.inputs.featurize` already caches
  CheMeleon and RDKit features content-addressed by SMILES in `data/features/`.
  Do not recompute them. T2 has landed, so ECFP4 is there too as `morgan`
  (2048-bit, radius 2 — the same parameters `load.py` uses, so your
  clustering matches the splits).
- Similarity and clustering machinery already exists in
  `src/data_tools/load.py::_butina_split` and
  `src/data_tools/split_column.py::_butina_groups` — same fingerprint
  parameters, so reuse rather than reinvent and your numbers stay comparable to
  the splits everything else is evaluated on.
- `src/data_tools/overlap.py` (`check-overlap`) already measures structural
  duplication within and between datasets on the InChIKey connectivity block.
  Build on it.
- Aux-head gains to regress against are in `reports/head_search_analysis.md`
  §2 — 20 candidates with solo macro RMSE, baseline 0.9430.

**A provenance problem to fix while you are here.** That report sources
`results/data_counts.csv` and `results/head_search_results.csv`. `results/` is
gitignored and currently holds only `chemprop/smoke/`. Neither file is on disk,
so the report's 55-config table cannot be regenerated or checked. Decide what
gets committed — probably the summary CSVs, not the checkpoints — and restore
them. `scripts/data_counts.py` regenerates one of them.

## Scope

1. **Map the space.** UMAP over CheMeleon embeddings *and* over ECFP4
   separately. They will disagree, and the disagreement is itself informative —
   one is a learned metric, the other a substructure metric, and the survey's
   activity-cliff finding lives exactly in that gap. Overlay train / test /
   each aux block / each qHTS isoform.
2. **Characterize the test set.** Tanimoto-to-nearest-train distribution,
   Murcko scaffold overlap with train, number of distinct scaffolds, cluster
   size distribution. Answer plainly: analog series or novel scaffolds? Report
   the fraction of test compounds with no training neighbour above Tanimoto 0.7
   — the statistic that predicted the delta-model disaster.
3. **Test the overlap hypothesis.** For each of the 20 aux-head candidates,
   compute overlap with the challenge set (mean/max Tanimoto to nearest
   challenge molecule, fraction of shared scaffolds, UMAP-region occupancy).
   Regress the 20 solo-head gains against these metrics and against row count.
   Report which predicts better. Redo `reports/data_volume_vs_gain.png` with
   overlap on the x-axis — `scripts/plot_volume_vs_gain.py` is the template.
4. **Quantify qHTS redundancy.** §6 of the head-search report argues the five
   qHTS isoforms are "five views of mostly the same molecule set" because they
   share the AID 1851 compound library, which would explain why they do not
   stack additively. That is directly measurable — compute the pairwise
   molecule-set overlap and settle it.
5. **Activity cliffs and applicability domain.** Find near-identical pairs
   (Tanimoto ≥ 0.9) with large ΔpIC50, per isoform. Identify test regions with
   no training support. Hand both to T7 — the applicability-domain map is what
   tells us where predictions should be shrunk toward the mean.

## Tools

- `rdkit` — `rdFingerprintGenerator`, `DataStructs.BulkTanimotoSimilarity`,
  `MurckoScaffold`, `rdkit.ML.Cluster.Butina`.
- `umap-learn` for the embedding; `scikit-learn` for PCA and clustering.
  Set and record a random seed — UMAP layouts are not reproducible otherwise
  and a figure nobody can regenerate is not evidence.
- `matplotlib` (already a dependency), `polars`.
- `scripts/plot_volume_vs_gain.py` as the plotting template.

## Done when

`reports/chemical_space.md` exists and answers, in order:

1. Is the test set analog-dense or scaffold-novel, and which survey lessons
   therefore transfer?
2. Does chemical-space overlap predict auxiliary-head gain better than row
   count does? (The direct follow-up `head_search_analysis.md` asked for.)
3. Are the five qHTS heads as redundant as we assumed?
4. Where is the applicability domain thin, and where are the cliffs?

Plus: the missing `results/` CSVs restored or regenerated, and a decision
recorded about what belongs in version control.

## SCC note

Not needed — this is CPU analysis over ~21k molecules, comfortably laptop-scale
if you reuse the cached embeddings rather than recomputing them.

If a full pairwise Tanimoto matrix over the union becomes the bottleneck
(21k × 21k is fine; the family sources could push it further), chunk it or ask
Denali for a batch node. No GPU required either way.
