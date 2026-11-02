# T5 — Structure generation: co-folding and docking

**Depends on:** nothing. **Start this first — it is the critical path.**
**GPU: heavy. This is the largest compute consumer in the project. SCC required.**
**Owns:** `src/structure/cofold.py`, `src/structure/dock.py`,
`scripts/structure.qsub`, `data/structures/` and its manifest.

---

## Why this matters for the challenge

Every representation in this repo is topological. RDKit descriptors, CheMeleon
embeddings, and whatever T2 adds all read the 2D bond graph. None of them can
see whether a molecule physically reaches the catalytic center.

For CYP inhibition specifically, that is a real blind spot with a known
mechanism attached. **Type II inhibitors coordinate the heme iron directly** —
an sp²/sp³ nitrogen from an imidazole, triazole or pyridine donates into the
Fe axial position, and these are the potent end of CYP inhibition (ketoconazole
and the azoles generally). A fingerprint sees the azole. Only a structure sees
whether that nitrogen can actually reach the iron in this particular pocket
given the rest of the molecule.

`reports/PXR-SUMMARY.md` §2 supports the general shape of this bet:
PeterBloomingdale broke through a persistent CV plateau at ~0.527 only when 3D
representations were added, attributing it to a large flexible buried pocket
where shape complementarity matters in ways invisible to bond-graph topology.
CYP3A4 is the textbook example of exactly that pocket.

The survey's 3D work was UniMol embeddings and docking scores. We can do better
than both: co-folding models now predict the holo complex directly, and the
current open-source generation has surpassed AlphaFold3 on protein–ligand
interfaces.

## Model choice

Use the open-source Chinese models — they currently lead this specific task.

**IntelliFold-2 / IntFold (IntelliGen AI) — primary co-folder.**
On a post-training-cutoff FoldBench subset, IntFold reaches **58.17% success on
protein–ligand interactions versus Boltz-2's 53.90%**, and its affinity head
correlates better with experiment (**PCC 0.53 vs Boltz-2's 0.47**).
IntelliFold-2 (Feb 2026) is among the first open models to beat AlphaFold3 on
FoldBench, with gains concentrated in protein–ligand co-folding. Three variants
ship — Flash, v2, Pro — so there is a speed/accuracy dial.
Repo: `github.com/IntelliGen-AI/IntelliFold`, weights on HuggingFace at
`intelligenAI/intellifold`.

**Protenix (ByteDance) — secondary, and the better-engineered input path.**
Its inference JSON takes ligands as CCD codes, SMILES, or file paths, which
means **the heme goes in as `CCD_HEM` directly**, and it supports an explicit
`covalent_bonds` block for the proximal cysteine thiolate–Fe linkage. That
matters more than it sounds (see below). Protenix-Mini exists for throughput,
and there is a Docker image if the SCC module environment fights you.

```bash
protenix pred -i input.json -o ./output -n protenix_base_default_v1.0.0
```

**Uni-Mol Docking V2 (DeepModeling / DP Technology) — the high-throughput arm.**
**77%+ of PoseBusters ligands under 2.0 Å RMSD, 75%+ passing all quality
checks**, up from 62% for V1, and it specifically avoids the chirality
inversions and steric clashes that plague ML docking.

### Why docking is not the fallback but a first-class arm

We already have high-resolution holo crystal structures for all four scored
isoforms, heme resolved: **1A2 → 2HI4, 2C9 → 1OG5, 2D6 → 4WNV, 3A4 → 4NY4.**

Co-folding re-predicts a protein we already know. The actual open question is
the *ligand pose in a known pocket containing a known cofactor*, which is the
docking problem. Docking into the crystal structures is far cheaper, and cheap
matters at our scale.

Reserve co-folding for the induced-fit question — CYP3A4's pocket is famously
plastic and a rigid crystal receptor may be the wrong receptor for half our
compounds.

### The caveat that shapes the plan

A 2026 co-folding benchmark across Nav/Cav/Kv channels found median top-ranked
ligand RMSDs of **19.85 Å (AF3), 22.12 Å (Boltz-2), 20.25 Å (Protenix-v2)** —
and Boltz-2 and Protenix-v2 **completed only 23 and 25 of 44 systems** while AF3
completed all 44.

Large flexible pockets break these models, and they break by *failing to
produce output*, not by producing obviously bad output. CYP3A4 is a large
flexible pocket. Hence the bake-off gate below, and hence the `completed` flag
in the output contract — T6 must be able to distinguish "no interaction" from
"the folder gave up."

## How it fits the current repo

New top-level module `src/structure/`. Nothing existing needs to change.

- Molecules come from `data/train.csv` and `data/test.csv` via
  `data_tools.load`. Key everything on `inchikey_block`
  (`data_tools.standardize`) so T6's features join to the union table cleanly.
- `scripts/head_search.qsub` is the qsub template — copy its resource block.
- Structures are large; `data/` is gitignored. Commit the manifest schema and
  the code, never the structure files.

## Scope

1. **Receptor prep.** Pull the four PDB structures, retain HEM, strip
   crystallographic waters and non-cofactor heteroatoms, protonate. Document
   every choice — this is the step that silently invalidates everything
   downstream.
2. **Docking arm (do this first).** Uni-Mol Docking V2 into the four prepared
   pockets, all 4,905 train + 750 test compounds × 4 isoforms.
3. **Co-folding arm.** IntelliFold-2 primary, Protenix cross-check. Heme as
   `CCD_HEM` alongside the substrate; `covalent_bonds` for the Cys–Fe bond.
4. **Bake-off gate — do not skip this.** Before any full fan-out, run all three
   methods on ~200 compounds spanning the pIC50 range on 3A4 and 2D6. Score on:
   - pose plausibility (PoseBusters checks),
   - **completion rate**, not just accuracy on the ones that finished,
   - whether T6's derived features move macro RMSE.

   Only then spend real GPU on the full set.

**Budget honestly.** 5,655 compounds × 4 isoforms ≈ 22,600 complexes per
method. At tens of seconds to minutes each for co-folding, the full grid is
weeks of A100 time. Do not queue it before the gate says the features work.

## Output contract for T6

```
inchikey_block, isoform, method, structure_path, confidence,
predicted_affinity, completed
```

`completed` is load-bearing given the failure rates above. `confidence` is
ipTM/pLDDT for co-folders, the model's own score for docking.

## Tools

- IntelliFold-2, Protenix, Uni-Mol Docking V2 (all open source, all GPU).
- `rdkit` for ligand prep and conformer seeding.
- PDBFixer / OpenMM or `reduce` for receptor protonation.
- `posebusters` for pose validity checks in the bake-off.

## SCC note

**⚠️ This task cannot be done on a laptop. Contact Denali with your code to get
it running on the BU SCC.** He has the project allocation.

Practical guidance:

- Copy `scripts/head_search.qsub`. **Keep `#$ -l gpu_c=7.0`** — older SCC GPUs
  lack the kernels for our torch build and die with
  `cudaErrorNoKernelImageForDevice`. If you hit it anyway, raise to 8.0; do not
  delete the line. These models may need a higher floor regardless.
- Run the bake-off (~200 compounds) as one short job and report back *before*
  requesting a large allocation. Do not ask for weeks of GPU on an unvalidated
  hypothesis.
- Array jobs, not one monolithic job. Make it resumable and checkpoint the
  manifest per batch — a 48-hour job that loses everything to a node failure is
  the default outcome otherwise.
- Model weights are large. Sort out where they live on the filesystem with
  Denali before the first run rather than re-downloading per job.

## Honest risk

Co-folding is weakest exactly where this dataset lives: promiscuous,
low-affinity, high-flexibility binders in a large induced-fit pocket. This is
the speculative arm of the whole project. The docking arm de-risks it — if
co-folding underperforms, Uni-Mol Docking V2 into the crystal pockets still
delivers the heme-coordination geometry, which is the part with genuine
mechanistic grounding.

Gate further GPU spend on T6's first result, not on how interesting the
structures look.
