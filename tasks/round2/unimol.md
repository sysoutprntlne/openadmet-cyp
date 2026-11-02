# Uni-Mol and ligand 3D shape

**Depends on:** nothing. Feeds round-1 T4 (ensembling).
**GPU:** yes for Uni-Mol (embedding is light, fine-tuning is moderate). The
shape-descriptor step is CPU.
**Owns:** `src/representations/shape.py`, `src/representations/unimol.py`,
`src/models/train_unimol.py` (all new); one line per featurizer in
`src/data_tools/inputs.py`; `requirements.txt`; `reports/unimol.md`.

---

## Why this matters for the challenge

Everything we compute today reads the 2D bond graph. Round-1 T5 and T6 attack
3D through the protein — docking and co-folding — which is the expensive route
and the one with a real chance of not finishing. There is a cheaper 3D track
that needs no protein at all, and the PXR survey reports it as one of the few
sources of signal that did not saturate (citations are to
`reports/PXR-SUMMARY.md`):

- **Uni-Mol** is a transformer pretrained on 209 million 3D conformers,
  encoding pairwise inter-atomic distances (Zhou et al., ICLR 2023; §2).
- discoverybytes (rank 11) reported a standalone OOF RAE of ~0.645 — weaker
  than their chemprop — yet it earned Ridge coefficient 0.166 in their final
  three-model stack (§2, §6). Its value was in being wrong differently.
- PeterBloomingdale (rank 33) had been stuck at CV RAE ~0.527 and broke
  through when Uni-Mol was added. Their best variant took 24.8% of the final
  SLSQP blend (§2, §6).
- **Learning rate mattered.** PeterBloomingdale tried 5e-5, 2e-4 and 5e-4; the
  2e-4 model carried the weight above, and a 1e-3 variant was assigned zero
  weight as too noisy (§2).
- **Test-time augmentation helped.** discoverybytes averaged predictions over
  10 randomized SMILES orderings, because different atom orderings yield
  different RDKit conformers; this reduced prediction variance (§2).
- **Shape descriptors were a cheap companion.** discoverybytes appended 19
  principal-moment-of-inertia descriptors to the chemprop FFN head as
  molecule-level features, capturing rod / disc / sphere shape orthogonally to
  both 2D topology and Uni-Mol's pairwise distances (§2).

**Two reasons for caution, both from the same survey.**

- RyeCatcher (rank 67) tested UniMolV2-310M among five extra encoders and
  found all of them converged to Spearman > 0.88 against their primary
  ensemble (§2). So Uni-Mol was orthogonal for two teams and redundant for a
  third. Which one we are is an empirical question.
- PeterBloomingdale attributed the gain to PXR's "large, flexible buried
  binding pocket" where shape complementarity matters (§2). That is an
  argument about PXR. Whether it carries to four CYP isoforms is not something
  the survey can tell us.

The survey also records **MolE** as the weakest standalone model in
discoverybytes' stack (OOF RAE ~0.705) that still earned coefficient 0.206
because it generalized to novel scaffolds (§2, §6). It is a stretch goal here,
not core scope.

## How it fits the current repo

Two integration routes, and they are different deliverables.

- **Frozen embeddings as a `Representation`.** Embed each molecule once with
  the pretrained Uni-Mol encoder and register it in `INPUT_REGISTRY`, exactly
  as `representations/chemeleon.py` does for CheMeleon: weights download at
  runtime, `transform` returns a float64 matrix, invalid SMILES give a NaN
  row. Caching, stacking, `evaluate-models --input all`, T3's models and
  TabPFN all work immediately. This is the cheap route and it comes first.
- **Fine-tuning as a model.** This does not fit `CYPModel` — it has its own
  trainer and consumes SMILES, not a feature matrix. Wrap it the way
  `train_chemprop.py` wraps chemprop: a thin CLI that trains on the pinned
  `split` column and writes predictions in T4's long OOF format
  (`model_id, inchikey_block, fold, target, pred`).
- **Shape descriptors** are another `Representation`. RDKit's `Descriptors3D`
  module provides PMI1–3, NPR1–2, asphericity, eccentricity, inertial shape
  factor, radius of gyration, spherocity and PBF from an embedded conformer.
  They can also go to chemprop through `--descriptors-path`.

## Scope

1. **Shape descriptors first.** One afternoon, no GPU, no new dependency.
   Fixed-seed ETKDG conformer, MMFF-optimized, `Descriptors3D` on the result.
   Register as `shape`. Benchmark alone, stacked on `morgan` and `chemeleon`,
   and as chemprop extra descriptors. This also tells us how noisy the
   conformer step is before anything expensive depends on it.
2. **Conformer determinism.** A featurizer whose output changes between runs
   breaks the content-addressed cache silently. Fix the embedding seed, decide
   what happens when embedding fails (NaN row, per repo convention), and
   report the failure rate on train and test.
3. **Frozen Uni-Mol embeddings** as `unimol`. Benchmark through
   `evaluate-models`. Report the pairwise correlation with `chemeleon` — this
   is the RyeCatcher check, and if it is above ~0.9 say so plainly.
4. **Fine-tuned Uni-Mol**, multi-task over the four scored heads on the pinned
   split, ≥3 seeds. Sweep learning rate over the survey's range (5e-5, 2e-4,
   5e-4). Check how the library handles our sparse label matrix before
   starting — most compounds are measured on a subset of isoforms, and a
   trainer that cannot mask missing targets needs one model per isoform.
Steps 1–4 are the core scope. Steps 5 and 6 are **optional extensions** — do
them if time and GPU budget allow, and say in the report which were run.

5. **Test-time augmentation** *(optional extension)*. Average over ~10
   conformers / SMILES orderings at inference and report the variance
   reduction.
6. **Orthogonality, measured** *(optional extension)*. Correlation of
   Uni-Mol's validation predictions with the best chemprop config's, per
   isoform, and the macro RMSE of a simple two-model average. That number is
   what T4 would use.

## What not to do

- **Do not judge it on standalone RMSE.** discoverybytes' Uni-Mol was their
  second-weakest model and still earned stack weight. The question is what it
  adds to chemprop.
- **Do not confuse this with T5.** T5 uses *Uni-Mol Docking V2*, a
  protein–ligand docking model. This task is the ligand-only molecular
  encoder. Different checkpoints, different outputs.
- **Do not commit weights or conformers.** Weights download at runtime;
  conformers and embeddings live under `data/`.
- **Do not let auxiliary heads in yet.** Four scored heads first. Whether
  Uni-Mol benefits from our 20 auxiliary heads the way chemprop does is a
  follow-up.

## Tools

- `rdkit` — already a dependency. `AllChem.EmbedMolecule`, `Descriptors3D`.
- `unimol_tools` — new dependency, not currently installed. Check its current
  API, supported task types and checkpoint names against its documentation
  before designing around it, and confirm it runs on the SCC's CUDA version.
- `torch` — already a dependency.

## Done when

- `shape` and `unimol` are registered and benchmarked on the pinned split,
  ≥3 seeds, alone and stacked.
- Fine-tuned Uni-Mol predictions exist in T4's OOF format for the pinned
  split.
- `reports/unimol.md` gives standalone macro and per-isoform RMSE, the
  learning-rate sweep, and the embedding correlation with `chemeleon` from
  step 3.

Optional, if steps 5 and 6 were run:

- The variance reduction from test-time augmentation.
- The correlation with chemprop predictions and the two-model average.
- An explicit verdict: orthogonal (the discoverybytes / PeterBloomingdale
  outcome) or redundant (the RyeCatcher outcome). **A clean "redundant" is a
  useful result** — together with the pharmacophore task it tells us whether
  T5 and T6 are worth their GPU budget. Without step 6, report the step-3
  correlation and leave the verdict open.

## SCC note

Steps 1–2 are laptop work. Embedding ~21k molecules and fine-tuning both want
a GPU. **Contact Denali with your code** to get jobs queued; reuse
`scripts/head_search.qsub` as the template.

## Sources

Findings are quoted from our survey, `reports/PXR-SUMMARY.md` §2 and §6; the
primary write-ups are:

- discoverybytes (rank 11) — Uni-Mol in the final stack, SMILES-order
  augmentation, PMI shape descriptors, MolE:
  https://github.com/discoverybytes/openadmet-pxr-blind-challenge/tree/main/activity-prediction
- PeterBloomingdale (rank 33) — plateau broken by Uni-Mol, learning-rate
  variants, SLSQP weights: https://github.com/PeterBloomingdale/openadmet-pxr
- RyeCatcher (rank 67) — UniMolV2-310M converging with the 2D ensemble:
  https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026
- Zhou et al. (2023), Uni-Mol, ICLR 2023:
  https://doi.org/10.26434/chemrxiv-2022-jjm0j
- MolE: https://doi.org/10.1038/s42256-024-00860-8
