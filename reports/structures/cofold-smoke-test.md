# Co-fold smoke test — ketoconazole × CYP3A4 (IntelliFold-2, local RTX 5090)

**Date:** 2026-09-23 · **Engine:** IntelliFold-2 (JAX/AF3 engine, vendored v3.0.3)
**Hardware:** RTX 5090 (32 GB) under WSL2 Ubuntu, CUDA 12.8 wheels
**Wall time:** 3m14s for 5 diffusion samples (weights pre-cached; first run adds ~1.7 GB HF download)

## Result: SANITY GATE PASSED

The co-fold reproduces the Type II heme-iron coordination signature from
sequence + SMILES alone — no template, no MSA, no crystal input.

| metric | prediction (ranked-best) | crystal ref (4NY4, `2QH`) |
|---|---|---|
| ligand N···Fe distance | **2.13 Å** (atom `N4`, imidazole) | 2.58 Å (atom `N32`) |
| approach angle vs heme normal | **7.7°** | 15.2° |
| min heavy atom ···Fe | 2.13 Å | 2.58 Å |

Both distances sit squarely in the Fe–N coordination regime (~2.0–2.5 Å) and
both approach vectors are near-axial. The prediction is, if anything,
*tighter* than the crystal — plausibly the model over-coordinates, a known
bias of co-folding models on metal centers.

## Confidence (AF3-style summary, ranked-best)

| field | value |
|---|---|
| ipTM | 0.77 |
| pTM | 0.48 |
| ranking score | 0.71 |
| has_clash | 0.0 |
| fraction_disordered | 0.0 |

Note the confidence split: chain-level ipTM is healthy (0.77) while pTM is
low (0.48) — expected, since CYP3A4 is a membrane-associated domain with
flexible regions and we fed no templates or MSA. For T6, the pocket-local
confidence (pLDDT around the heme) is the relevant gate, not global pTM.

## Why this validates the pipeline

1. **Input path:** full-length UniProt P08684 (503 aa) + `CCD HEM` + ligand
   SMILES, AF3-dialect JSON with empty MSA fields and no templates. The
   heme's Fe arrives through the CCD entry; the Cys442 thiolate coordination
   is learned, not specified. (IntelliFold's schema has no cross-entity
   covalent bonds; Protenix's `covalent_bonds` block is the production
   variant for that — see tasks/T5-structure-generation.md.)
2. **Output contract:** manifest row written to
   `data/structures/manifest.csv` with `inchikey_block` (`XMAYWYJOQHXEEK`),
   `method=intellifold2`, `confidence` (ipTM), `completed=True`.
3. **Metrics:** [`src/structure/metrics.py`](../../src/structure/metrics.py)
   computes the T6 headline features (N···Fe distance, approach angle,
   min heavy-atom distance) identically on predicted and crystal cif via
   gemmi — the calibration path works.
4. **Throughput:** 3m14s for one complex × 5 samples on a single 5090
   (~39 s/sample including featurisation). Feasible bake-off scale:
   ~200 compounds × 2 isoforms ≈ 5–6 h on one 5090, or under 2 h across
   3 GPUs (the wrapper batches per-GPU natively).

## Environment learnings (for the SCC hand-off)

- IntelliFold-2 now runs on the AF3 JAX engine and needs Linux; on Windows,
  WSL2 with CUDA passthrough works as-is (`jax.devices()` → `[CudaDevice(id=0)]`).
  No Docker needed.
- The AF3 input validator *requires* `dialect: alphafold3`, `version: 1`,
  and per-chain `unpairedMsa: ""`, `pairedMsa: ""`, `templates: []` even with
  `--norun_data_pipeline`. Empty strings = MSA-free inference.
- Output layout: `output/<name>/<name>_model.cif` (ranked-best) plus
  `seed-N_sample-M/` subdirectories with per-sample cifs and confidences;
  `*_ranking_scores.csv` holds the sample ranking.
- SMILES ligands are written to the cif under `LIG_<id>` (here `LIG_K`).
- Benign noise: `cuda_timer.cc:87` "Delay kernel timed out" warnings during
  sampling on consumer GPUs.

## Files

- Predicted complex: `data/structures/runs/cyp3a4_ketoconazole/output/cyp3a4_hem_lig/cyp3a4_hem_lig_model.cif`
- Interactive view: [`cyp3a4_ketoconazole.html`](cyp3a4_ketoconazole.html) (browser)
- Pocket view: [`cyp3a4_ketoconazole.png`](cyp3a4_ketoconazole.png)
- Manifest: `data/structures/manifest.csv`

## What this does NOT establish

- One complex, one isoform, one ligand. The bake-off gate (T5 §Scope.4)
  still requires ~200 compounds spanning pIC50 on 3A4 + 2D6 before any
  full fan-out — this run only proves the pipeline and the mechanism.
- Pose *accuracy* beyond the Fe-coordination geometry (no pocket RMSD check
  against 4NY4 yet; that needs alignment and is the next cheap win).
- The affinity head did not run — `properties: [affinity]` is a Boltz-side
  feature; IntelliFold's affinity comes from its own head and needs a flag
  check on the wrapper. `predicted_affinity` is null in the manifest.
