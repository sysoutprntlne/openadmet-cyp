# T6 — Structure-derived features

**Depends on:** T5's manifest. Develop against ~50 hand-made structures so you
are not blocked.
**GPU:** moderate — feature extraction is CPU, but evaluating the features means
chemprop runs. SCC likely.
**Owns:** `src/representations/structure.py`; one line in `src/data_tools/inputs.py`.

---

## Why this matters for the challenge

T5 produces structures. Structures are not a model input. This task is where
the co-folding bet either pays or is shown not to — and it is worth being clear
that a pile of PDB files with no feature extraction is worth exactly zero on
the leaderboard.

The specific hypothesis being tested: **there is mechanistic information about
CYP inhibition that is visible in a 3D complex and invisible to any 2D
fingerprint.** The strongest version of that claim is heme-iron coordination.
Type II inhibitors bind by donating a nitrogen lone pair into the Fe axial
position, and they are the potent end of CYP inhibition. ECFP4 sees that a
molecule contains an imidazole. It cannot see whether that imidazole, on this
scaffold, in this pocket, can present its nitrogen to the iron — which is the
difference between a nanomolar inhibitor and an inactive compound with the same
substructure.

This also directly addresses the activity-cliff problem that
`reports/PXR-SUMMARY.md` §7 describes: structurally near-identical compounds
with wildly different potency, which destroyed RyeCatcher's nearest-neighbour
approach (OOF 0.4536 → LB 0.658). Cliffs are often conformational or steric,
which is to say they are 3D. If structure features help anywhere, it is here.

## How it fits the current repo

Registers as an ordinary featurizer and then flows through everything.

- Subclass `Representation` (`src/representations/base.py`), same as
  `rdkit_descriptors.py`. Append one line to `INPUT_REGISTRY` in
  `src/data_tools/inputs.py`. **Append only, never reorder** — T2 also touches
  that file.
- Once registered you inherit the content-addressed cache in
  `inputs.py::_featurize_one`, and `evaluate-models --input structure` works
  immediately.
- **Feed it two ways**, because they test different things:
  1. As a feature block for the T3 sklearn-style models and the T4 stack.
  2. As molecule-level descriptors on the chemprop FFN head via
     `--descriptors-path` (passthrough already works in
     `src/models/train_chemprop.py`). This is the pattern discoverybytes used
     for PMI shape descriptors in the PXR challenge — 3D features as `x_d`
     alongside the learned graph representation, rather than replacing it.
- Join to the union on `inchikey_block`. Note the shape mismatch you must
  handle: T5 emits one row per *molecule × isoform*, while the union is one row
  per molecule with per-isoform columns. Pivot accordingly —
  `CYP3A4_fe_distance`, `CYP2D6_fe_distance`, and so on, so each becomes its own
  column the way every other head in this repo does.

## Scope — features in priority order

1. **Heme-iron coordination geometry.** The headline feature.
   - Distance from the nearest ligand sp²/sp³ nitrogen to Fe.
   - Approach angle relative to the heme plane normal — a nitrogen at 3 Å
     off-axis is not coordinating.
   - Identity of the coordinating atom's environment (imidazole / triazole /
     pyridine / aliphatic amine / none).
2. **Minimum heavy-atom distance to Fe**, for ligands that block the site
   sterically without coordinating.
3. **Buried SASA** of the ligand in the complex — how deeply it sits.
4. **SRS contact fingerprint.** Substrate recognition site residue contacts,
   per isoform. A fixed-length vector over the pocket-lining residues.
5. **The co-folder's own affinity prediction** as a single scalar feature
   (IntelliFold's affinity head, PCC 0.53 on benchmark targets).
6. **Confidence** — ipTM / pocket pLDDT — as both a feature and a gate.

## The masking requirement — do not skip this

Given the completion rates in T5's benchmark note (co-folders finished only
~23–25 of 44 systems on hard targets), **failed and low-confidence structures
must be masked, not imputed.**

A feature column that silently encodes "the structure predictor gave up" will
correlate with molecular size, flexibility and pocket difficulty — all of which
correlate with the label — and will leak a nonsense signal straight into the
shared encoder. It will look like a gain on validation and evaporate on test.

Concretely:
- Emit NaN for failed folds and let chemprop's masking handle it, matching the
  repo's existing convention for sparse heads.
- Report solo-feature gain **split by confidence tier and by the `completed`
  flag**. If the gain only exists in the high-confidence subset, that is the
  real result and it is still useful — it tells us to gate structure features
  on confidence at prediction time.

## Tools

- `biotite` or `MDAnalysis` for structure parsing and geometry. `biotite` is
  lighter and reads mmCIF cleanly, which is what the co-folders emit.
- `rdkit` for ligand atom typing — identifying which nitrogens are plausible
  coordinating atoms.
- `freesasa` or MDAnalysis for buried SASA.
- `numpy`, `polars` for the feature matrix.
- `evaluate-models`, `head-search`, and `train-chemprop --descriptors-path` for
  evaluation.

## Done when

- Solo-feature gain measured against the 0.9430 baseline over ≥3 seeds, split
  by confidence tier and completion status.
- Compared head-to-head across T5's three methods — a feature set that only
  works on docked poses and not co-folded ones is a finding worth reporting,
  and it decides where the remaining GPU budget goes.
- An explicit verdict written to `reports/`: does 3D structure add signal over
  CheMeleon + ECFP4 on this dataset, yes or no. **A clean negative is a
  legitimate and valuable deliverable here** — it saves the project weeks of
  A100 time and closes the most expensive open question we have.

## SCC note

Feature extraction from existing structures is CPU work and parallelizes
trivially — do that locally or on a batch node.

Evaluating the features is chemprop training, so **contact Denali with your code**
for the SCC runs. Reuse `scripts/head_search.qsub` and keep `#$ -l gpu_c=7.0`.

You will also need read access to wherever T5 wrote `data/structures/` — sort
that out with T5's owner and Denali early, because these files are large and
copying them around the filesystem will waste more time than the compute does.
