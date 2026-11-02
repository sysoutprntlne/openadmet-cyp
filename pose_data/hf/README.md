---
license: mit
task_categories:
- graph-ml
tags:
- protein-ligand
- structural-biology
- cytochrome-p450
- cyp3a4
- co-folding
- openadmet
pretty_name: CYP Co-folded Poses and Cross-Engine Selection Benchmark
size_categories:
- 10K<n<100K
---

# CYP co-folded poses — OpenADMET blind challenge, structure track

Protein–HEME–ligand poses for cytochrome P450s, with crystal ground truth where it exists,
released so other teams do not have to re-fold what we already folded.

Produced for the **OpenADMET CYP blind challenge** structure-prediction track. We are
**affiliated with the BU AI & ML team** ([buaiml/openadmet-cyp](https://github.com/buaiml/openadmet-cyp)) —
supporting their T5 structure-generation task while running our own track. Not members of
their roster; this dataset is offered to them and to everyone else on the same terms.

---

## What is here

| path | what | size |
|---|---|---|
| `t5_bakeoff/poses/{CYP3A4,CYP2D6}/*.cif` | 400 co-folded complexes: 200 challenge compounds × 2 isoforms | ~134 MB |
| `t5_bakeoff/t5_manifest.csv` | per-pose manifest: completion, physical validity, Fe geometry, confidence | small |
| `t5_bakeoff/t5_gate_report.json` | aggregate completion / validity / coordination rates | small |
| `t5_bakeoff/compounds.csv` | the 200 compounds, stratified across pIC50 deciles, measured on both isoforms | small |
| `p450_poses_scored.csv` | ~15k P450 poses scored against crystal truth (LDDT-PLI, BiSyRMSD) | ~3 MB |
| `reference_set_p450_sweep.npz` | frozen cross-engine reference set, 492 ligands / 3,898 deduped poses | 1.1 MB |
| `pool_scorecard.json` | the measurement log — every headline number with its date and n | small |

Generation: `protenix_v2` on OpenProtein, MSA-backed, protein + `HEM` + ligand as three
chains. `num_recycles=3`, `num_steps=200`, `diffusion_samples=1`.

## Read this before using it

Seven things we measured the expensive way. They are not opinions; each cost real compute.

1. **Model confidence does not rank poses within a ligand.** Boltz `complex_ipde` vs true
   LDDT-PLI is within-ligand ρ = −0.033 / −0.092; Protenix-v2's seven confidence fields sit
   at frac(ρ>0) = 0.50 on four of five, every p > 0.6. **Selecting the most confident pose
   is worse than selecting at random.** If you weight poses by confidence, you weight by noise.
   *(Between compounds is a different question — see point 7.)*
2. **`diffusion_samples` does not diversify the ligand.** Every model inside one job shares
   a single ligand conformation, on every engine we tested.
3. **Replicate jobs no longer diversify either.** As of 2026-09-15 both `protenix_v2` and
   `esmfold2` return a byte-identical pose per input. A 12→24 replicate doubling produced
   5,892 poses and moved the per-pair oracle on **0 of 489 pairs**.
4. **What does diversify: the sampler.** Varying `num_recycles` / `num_steps` gives 4
   distinct placements from 4 settings at 1.2% catastrophic. Use `num_recycles ≥ 2`;
   `num_recycles=1` measured −0.0596 LDDT-PLI against the default.
5. **Selection, not generation, is the bottleneck.** Pool oracle 0.6975 against selection
   0.5706 on our CYP3A4 set — 0.127 LDDT-PLI per ligand sitting unclaimed in a pool
   already paid for.
6. **More diversity does not become findable.** A union of engines adds +0.0375 of oracle
   and selection captures −0.0017 of it. A sampler sweep adds +0.0355 and captures −0.0038.
   Two unrelated mechanisms, same result. Keep a second source as *reference*, never as a
   pool member.
7. **Iron coordination is saturated on crystal ligands and not on challenge compounds.**
   84% of poses for deposited CYP3A4 ligands already sit inside any reasonable coordination
   window, so the feature is nearly inert there (it moved our selection by −0.0002). On the
   challenge compounds it is 41.0% / 28.5% — real variance, because most of them are not
   Type II binders. A feature dead on one set can be alive on the other.

**The noise floor on this kind of work is +0.0138** (95th percentile of a random feature).
Anything under +0.020 is indistinguishable from noise. We checked this against a formal
multiple-comparisons correction and the two agree to 0.0008.

## Caveats on the poses themselves

- **Ligand atom order differs from the crystal.** Index-for-index comparison halves
  LDDT-PLI silently. Map atoms with graph isomorphism (symmetry-aware) before scoring.
- **Crystals deposit multiple ligand copies.** 8SO1 has three caffeine molecules; only one
  is in the pocket. `(pdb, chain, ligand)` is **not** a unique key — we found 125 duplicate
  rows that way.
- **Residue numbering differs.** Predictions number from 1; crystals use author numbering
  (e.g. 3TK3 starts at 28). Align by residue number, never array position, or whole targets
  score exactly 0.000.
- `t5_manifest.csv` has `predicted_affinity` empty: co-folding with protenix does not
  produce one. `get_affinity` exists on OpenProtein but is boltz-2 only.

## Provenance

Every number above is reproducible from the campaign log in `pool_scorecard.json` and the
findings documents in the source repo. Where a claim was later retracted we say so there
rather than quietly dropping it — several were.

## Citation

If this saves you compute, a note pointing at the OpenADMET challenge and the BU team's
repo is more useful to us than a citation.
