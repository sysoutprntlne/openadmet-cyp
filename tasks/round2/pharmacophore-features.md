# CYP pharmacophore features

**Depends on:** nothing. Pairs with round-1 T6 (it is T6's 2D control).
**GPU:** no. CPU-only, laptop-friendly.
**Owns:** `src/representations/cyp_pharmacophore.py` (new); one line per
featurizer in `src/data_tools/inputs.py`; `reports/pharmacophore_features.md`.

---

## Why this matters for the challenge

All six registered representations are target-agnostic. None of them knows the
molecule is going to meet a heme iron.

The PXR survey records two ways competitors put target knowledge into features
(citations are to `reports/PXR-SUMMARY.md` §2):

- **Target-specific SMARTS.** duoduo6 (rank 168), wuhicky (rank 148), duod
  (rank 205) and cat554 (rank 118) added 17 pharmacophore motifs and
  physicochemical rules encoding known PXR pocket preferences — sulfonamides,
  trifluoromethyl, aromatic surface area, hydrophobic volume.
- **Similarity to known binders.** ldbc1999 (rank 65) computed ECFP4 Tanimoto
  similarity against 56 PXR co-crystal ligands from the PDB (max, mean, std,
  top-2, top-3), on the argument that experimentally confirmed binders encode
  proximity to the active pharmacophore in a way within-training similarity
  does not.
- auP7s (rank 27) included generic pharmacophore fingerprints, via
  scikit-fingerprints, in a wide classical-descriptor set.

**Read the ranks.** The SMARTS users sit at 118–205. The survey does not report
an ablation isolating the SMARTS contribution, and nobody in the top decile is
recorded as relying on it. That is weak evidence, and the task should be sized
accordingly: a few days, not a workstream.

**One more reason to run the cheap control first.** jjaybird606 reached rank 3 in
the PXR challenge with an agent-driven blend and reported structure-based scoring
— docking and Boltz-2 both — as redundant to their 2D stack
(https://github.com/jjaybird606-blip/pxr). One team's negative result on a
different target is not a verdict on T5 and T6, but it is a second reason to know
what a substructure flag buys before GPU-weeks are committed.

The reason to do it anyway is that the CYP case is mechanistically sharper than
the PXR one. PXR has a large promiscuous pocket; CYP inhibition has a specific,
well-known interaction — a ligand nitrogen coordinating the heme iron.
Round-1 T6 makes that its headline feature and spends GPU-weeks of docking to
measure it in 3D. Whether the molecule *has* an unhindered azole or pyridine
nitrogen at all is a substructure question that costs milliseconds. If the 2D
flag captures most of the signal, T6's verdict changes; if it captures none,
T6's case gets stronger. Either way T6 needs this as its control.

## How it fits the current repo

- Subclass `Representation` (`src/representations/base.py`), follow
  `rdkit_descriptors.py`, register by appending to `INPUT_REGISTRY`. Caching,
  stacking and `evaluate-models --input` come for free.
- `evaluate-models --input all` benchmarks it against the other six in one
  call. Stack it: `--input cyp_smarts morgan`, `--input cyp_smarts chemeleon`.
- For chemprop, molecule-level features go in through `--descriptors-path`
  (confirmed present in chemprop 2.3.1), the same route discoverybytes used to
  append shape descriptors to the FFN head (§2).

## Scope

1. **Heme-coordination motifs** — the core of the task. SMARTS counts for
   aromatic sp² nitrogens able to ligate iron: imidazole, 1,2,4- and
   1,2,3-triazole, pyridine, pyrimidine, thiazole, oxazole; plus primary and
   unhindered aliphatic amines. Add a steric-accessibility term (ortho
   substitution next to the nitrogen), since a hindered nitrogen does not
   coordinate. These are the same atom environments T6 lists for its 3D
   feature — use the same categories so the two are comparable.
2. **Isoform-preference descriptors.** A small set of physicochemical features
   chosen for the four scored isoforms: basic-nitrogen count and a pKa proxy,
   acidic-group count, aromatic ring count and a planarity measure, size and
   lipophilicity. **Each one must be justified from a cited medicinal-chemistry
   source in the report** — do not encode isoform folklore from memory, and do
   not take an AI tool's word for it (RULES.md §5 on chemistry conventions).
3. **Mechanism-based inactivation alerts** — methylenedioxyphenyl, terminal
   alkynes, furans, thiophenes and similar. These are about time-dependent
   inhibition rather than direct inhibition, so expect little on the scored
   heads; they are here because round-1 T1 adds TDI heads and will want them.
   Emit them as a separate block so they can be left out.
4. **Similarity to co-crystallized CYP ligands.** ldbc1999's feature, ported:
   collect ligands from PDB entries for CYP1A2, 2C9, 2D6 and 3A4, and compute
   max / mean / top-k Tanimoto per isoform. Run `check-overlap` on the ligand
   list against `data/test.csv` first.
5. **A generic pharmacophore fingerprint as the comparator.** RDKit ships ErG
   (`rdReducedGraphs.GetErGFingerprint`) and Gobbi 2D pharmacophores
   (`Chem.Pharm2D`). Register one. If hand-written CYP SMARTS do not beat an
   off-the-shelf pharmacophore fingerprint, the hand-written ones are not
   worth maintaining.
6. **Benchmark** each block alone and stacked on `morgan` and `chemeleon`, and
   as chemprop extra descriptors on the no-aux and best-head configurations.
   Report pairwise correlation with the existing representations, as T2 did.

## What not to do

- **Do not grow the SMARTS list until validation improves.** Fix the list from
  the literature before looking at scores. A motif added because it helped on
  the validation set is a fitted parameter.
- **Do not derive motifs from `data/test.csv`.**
- **Do not add a fingerprint zoo.** The survey's saturation finding (§2:
  RyeCatcher's five extra 2D encoders all converged to Spearman > 0.88 with
  the primary ensemble) applies here. One hand-built block, one generic
  comparator, stop.

## Tools

- `rdkit` — already a dependency. `Chem.MolFromSmarts`, `rdReducedGraphs`,
  `Chem.Pharm2D`, `Descriptors`.
- No new dependencies. `scikit-fingerprints` is not needed for this scope.

## Done when

- Featurizers registered and benchmarked through `evaluate-models` on the
  pinned split, ≥3 seeds, alone and stacked.
- `reports/pharmacophore_features.md` lists every motif with its SMARTS, its
  literature source, and its prevalence in train and test.
- Per-isoform results, not only macro: the heme-coordination block should
  matter differently for different isoforms, and that pattern is the result.
- A written handoff to T6: how much of the gain a 3D coordination feature has
  left to find after the 2D flag is in.
- A keep or drop recommendation for each block. A null result is fine.

## SCC note

Not needed for the featurizers or the sklearn benchmarks. The chemprop
extra-descriptor runs in item 6 go through the SCC — contact Denali.

## Sources

Findings are quoted from our survey, `reports/PXR-SUMMARY.md` §2; the primary
write-ups are:

- ldbc1999 (rank 65) — similarity to PXR co-crystal ligands:
  https://github.com/lizyurkewych-git/pxr-challenge
- cat554 (rank 118): https://github.com/ttll1667/pxr_comp/blob/main/README_pxr_stability_stack_experiment_log.md
- wuhicky (rank 148): https://github.com/wuhike61-cmd/pxr
- duoduo6 (rank 168): https://github.com/duoduo6660/PXR
- duod (rank 205): https://github.com/duoyou666/PXR_report
- auP7s (rank 27) — pharmacophore fingerprints in a wide descriptor set:
  https://gist.github.com/chemotica/a49b002eda2f7fd5eef2dca4f98f8ad7
- discoverybytes (rank 11) — molecule-level descriptors into the chemprop FFN:
  https://github.com/discoverybytes/openadmet-pxr-blind-challenge/tree/main/activity-prediction
- RyeCatcher (rank 67) — 2D saturation finding:
  https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026
