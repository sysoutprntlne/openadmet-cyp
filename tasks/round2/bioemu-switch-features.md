# BioEmu-2 conformational-switch features

**Depends on:** nothing to start. Pairs with round-1 T5/T6 and round-2 R2.
**GPU:** yes. Amber MD in phase 1 is the heavy part; BioEmu-2 sampling per
molecule is the second. SCC.
**Owns:** `src/representations/bioemu_switch.py` (new); one line in
`src/data_tools/inputs.py`; `scripts/bioemu_*.py`; `reports/bioemu_switch.md`.

---

## The idea

CYP inhibition is not only "does the ligand fit the pocket". The enzyme has
conformational states — the substrate channel and B′/F/G-helix region open and
close, and the I-helix and active-site residues rearrange — and a ligand that
stabilises one state over another changes how tightly it binds. Call that the
**switch**. If we can measure which state a molecule drives the enzyme into, that
is mechanistic information no 2D fingerprint carries.

MD can show the switch but is far too expensive to run on 4,905 molecules ×
4 isoforms. BioEmu-2 is a generative model that samples ensembles quickly and,
unlike BioEmu v1, accepts a bound ligand. The plan:

1. **Check BioEmu-2 is trustworthy for CYP with a ligand present** against real
   MD, because we do not know whether any CYP is in its training data or whether
   it generalises to one.
2. **Only if it passes**, run it for all train and test molecules.
3. **Extract features from the resulting ensembles and test them** on the
   pinned split.

Each phase has a gate. Failing a gate is a valid result — stop and write it up.

## What we know about BioEmu-2, and what we do not

From the model card (https://huggingface.co/microsoft/bioemu2) and repository
(https://github.com/microsoft/bioemu2), read on 2026-10-21:

- MIT licence, released for research, October 2026. Inputs are sequences plus
  optional MSA, contact conditioning and method conditioning; systems "can include
  proteins, nucleic acids, ligands, and other chemical components", defined in
  system YAML files (the repo mentions conversion from Boltz input YAML) or from
  PDB IDs / structure files.
- CLI: `bioemu2 sample-equilibrium`, `predict-structure`, `propagate`; steering
  with `--system <yaml> --sampler edm_gd`. Linux, NVIDIA GPU of compute capability
  ≥ 7.0 (V100 is the lowest supported), Python 3.12 is the benchmarked version.
- **Out of scope per the model card:** binding-affinity or free-energy
  prediction and de novo ligand design. So BioEmu-2 gives us *conformational
  ensembles in the presence of a ligand*, not a binding score. Do not describe
  any feature as an affinity prediction.
- **Not documented in what I could read:** the exact ligand input format (SMILES,
  CCD code or file), runtime per sample, memory use, and whether CYPs or heme
  cofactors were in training. The implementer must read the repository README and
  its speed-and-capacity guide, and **how the heme cofactor is handled is the first
  thing to settle** — a CYP without a modelled heme is not the same system.
  Record the exact version/commit used.

## Phase 1 — Reference MD on 10 molecules per isoform, with Amber

For **each of the four scored isoforms** (CYP1A2, 2C9, 2D6, 3A4):

- **5 inhibitors:** `pIC50 ≥ 5`.
- **5 non-inhibitors:** `pIC50 ≤ 3`.

The thresholds are on **pIC50, not IC50** — higher pIC50 means more inhibitory.
Molecules between 3 and 5 are excluded on purpose.

Selection rules:

- Pick from the **`train` slice of the pinned split only**, using the
  `CYP*_pIC50_direct_inhibition` columns. Never `val` or `test`.
- Prefer tight measurements (low `_std`, narrow `_conf_high`/`_conf_low`).
- Match the two groups on heavy-atom count and logP as far as the data allows, so
  MD differences are not just a size effect. Report the match.
- Pick for scaffold diversity within each group.
- Where possible include a molecule that is an inhibitor of one isoform and a
  non-inhibitor of another; that is the most informative pair.
- Fix the list and commit it (`data/bioemu/reference_ligands.csv`) **before**
  running anything.

### Starting structures

Checked against the RCSB data API on 2026-10-21:

| Isoform | PDB | Title | Resolution |
|---|---|---|---|
| CYP1A2 | [2HI4](https://www.rcsb.org/structure/2HI4) | Human microsomal P450 1A2 with α-naphthoflavone | 1.95 Å |
| CYP2C9 | [1R9O](https://www.rcsb.org/structure/1R9O) | P450 2C9 with flurbiprofen bound | 2.0 Å |
| CYP2D6 | [3TBG](https://www.rcsb.org/structure/3TBG) | P450 2D6 with two thioridazines in the active site | 2.1 Å |
| CYP3A4 | [1TQN](https://www.rcsb.org/structure/1TQN) | Human microsomal P450 3A4 | 2.05 Å |

These are the starting points, not a final choice. Before use, check each for
missing loops, engineered mutations or truncations, and whether the bound ligand
sits in the same pocket as the molecules we will model. Other entries exist (for
example 1OG2 / 1OG5 for CYP2C9 with warfarin) and may suit a given ligand better;
if you swap one, look it up and record why. **Make sure the BioEmu-2 input
sequence and structure match the construct MD uses.**

### Amber MD

Amber is a reasonable choice and I would keep it: P450 heme parameter sets were
built for Amber, `pmemd.cuda` is fast, and the ligand parameterisation tools
(`antechamber`, GAFF2 with AM1-BCC charges) are standard. Two practical checks:

- `pmemd.cuda` needs the licensed Amber, not the free AmberTools alone. Confirm it
  is installed on the BU SCC before building anything; if it is not, tell Denali
  and fall back to OpenMM rather than `sander` on CPU.
- **Heme parameters are the hard part.** A heme-thiolate with Cys-ligated ferric
  iron needs specific parameters. Published Amber-compatible sets exist
  (Shahrokh et al., J Comput Chem 2012 is the one I know; **verify it and use
  whichever you can cite**). Generic force-field heme is not acceptable.

Setup:

- Docked starting pose per ligand (T5's Uni-Mol Docking V2 or any documented
  docker). The pose is a starting point only; MD relaxes it.
- Per system: ≥3 independent replicas, long enough that the switch region moves
  in the apo system. Also run **apo MD per isoform** (4 more systems). Total: 40
  ligand-bound + 4 apo.
- If plain MD does not cross the switch on affordable timescales, say so and
  consider one enhanced-sampling arm instead of pretending convergence.

Contact Denali with the system-building scripts before queuing the full set.

## Phase 2 — Define the switch, then compare BioEmu-2 to MD

### Define the switch coordinates first

Before looking at any BioEmu-2 output, write down from the **literature** which
structural coordinates define the switch, and commit them with citations.
Candidates: B′-helix / F–G loop opening (substrate-channel width), I-helix
distortion, heme-pocket residue rotamers and pocket volume. Keep it to a small set
per isoform. **A coordinate chosen because it separates inhibitors from
non-inhibitors in the MD is a fitted parameter** — fix them before the comparison.

### Compare

Run BioEmu-2 for the **same 44 systems** the MD covers (ligand-bound and apo).
Record version, sampler, system YAML and sample counts.

- Overlay BioEmu-2 marginals on the MD marginals for each system and quantify with
  a divergence or Wasserstein distance, with a bootstrap spread from the MD
  replicas.
- Does BioEmu-2 cover the same states the MD visits? Which are missing?
- **Does BioEmu-2 shift the ensemble when the ligand is added, in the same
  direction as MD?** Apo → bound shift, per ligand. Do inhibitors and
  non-inhibitors differ in BioEmu-2 the way they do in MD? This is the test that
  matters for features.
- Artefact checks: heme position and Cys ligation, ligand position relative to the
  iron, steric clashes, chain breaks, secondary structure.
- Look up whether CYPs are in BioEmu-2's training data; if the documentation does
  not say, say that. If one is available, add a CYP the model could not have seen as
  a generalisation check.

### Gate

Write the pass criteria **before** running the comparison — for example,
BioEmu-2 is within MD replica-to-replica spread on each coordinate, and it
reproduces the sign of the apo → bound shift for most ligands. If no isoform
passes, stop, write `reports/bioemu_switch.md`, and do not run phase 3. If only
some pass, proceed for those only, and say so in every table.

## Phase 3 — Run BioEmu-2 on every molecule

Only after the gate. Run BioEmu-2 with the ligand on all train, validation and
test molecules, for each isoform that passed. The features use no labels, so test
molecules are fine to run; do not skip or special-case any.

Time 200 molecules first and extrapolate. 4,905 molecules × 4 isoforms × a
useful sample count may be large; decide on sample count and any subsampling from
the timed run and record the choice.

## Phase 4 — Features from the simulations

**We need features from these ensembles; which ones is not decided yet.** Do not
fix a feature list now. Once phases 1–3 have produced simulations, look at them
— the MD and the BioEmu-2 ensembles — and propose features then. The brief only
sets rules for how that proposal is made:

- Features come out of the BioEmu-2 ensembles (the MD is the reference that
  checks they mean something, not the feature source for the full set, since MD
  exists for only 40 molecules).
- Write the feature definitions down and commit them **before** scoring them on
  validation or test. A feature added because it helped on validation is a fitted
  parameter.
- Each feature must be computed identically for every molecule and isoform and
  pivoted to per-isoform columns like `CYP3A4_<name>`, the way every other head in
  this repo is.
- Check each feature on the 40 reference systems: does it agree with what the MD
  shows? A feature that does not track MD is not measuring the switch.
- Report features grouped by what they measure (e.g. switch state, pocket
  geometry, ligand–heme geometry) so a block can be dropped as a unit.

## Phase 5 — Benchmark

- Register as `bioemu_switch` in `INPUT_REGISTRY` (append only, never reorder), as
  a `Representation`, following `rdkit_descriptors.py`.
- Benchmark through `evaluate-models` on the **pinned split**, ≥3 seeds, alone and
  stacked on `morgan` and `chemeleon`, and as chemprop `--descriptors-path` extra
  descriptors on the no-aux and best-head configurations. Reference numbers are the
  fixed `0.7557` / `0.7193`.
- **Controls, required:** (a) R2's 2D pharmacophore block, (b) T6's 3D heme
  geometry if it exists, (c) a **shuffled-feature** control — the same block with
  rows permuted across molecules. Report gain over (a) and (b), not over nothing.
- Per-isoform results, not only macro.

## What not to do

- **Do not tune the switch definition or feature set on validation or test
  scores.**
- **Do not select reference molecules using `val` or `test` rows**, or use the test
  set to decide whether BioEmu-2 is "good enough".
- **Do not present any feature as a predicted affinity.** BioEmu-2 does not
  predict affinity.
- **Do not claim BioEmu-2 reproduces MD from one replica or one coordinate.**
  Report spread.
- **Do not fit anything on the 40 reference molecules.** They validate the method.
- **Do not extend this to PXR or other targets.** CYP only.

## Tools

- BioEmu-2 (`bioemu2` CLI, https://github.com/microsoft/bioemu2). Use the `ctx7`
  workflow or the repo README for current usage — the repo has changed since
  BioEmu v1, which is no longer maintained.
- Amber (`tleap`, `antechamber`, `pmemd.cuda`); MDAnalysis or MDTraj for analysis.
- RDKit — already a dependency. New dependencies go in `requirements.txt` with a
  line in the PR saying why.

## Done when

- `data/bioemu/reference_ligands.csv` is committed before MD starts, with the
  selection rule and size/logP match.
- Switch coordinates and gate criteria are committed with citations before the
  comparison runs.
- `reports/bioemu_switch.md` answers, per isoform, whether BioEmu-2 reproduces the
  MD (pass / partial / fail, with figures and spread).
- If the gate passed: committed feature definitions, `bioemu_switch` registered
  and benchmarked with the controls above, per-isoform and macro results.
- A written handoff to T6 and R2 on how much this adds over heme geometry and the
  2D flag.
- A keep or drop recommendation. A null result at either gate is a result.

## SCC note

Phase 1 MD and phase 3 BioEmu-2 runs go through the SCC. Contact Denali with the
setup scripts and a timed 200-molecule BioEmu-2 sample first.
