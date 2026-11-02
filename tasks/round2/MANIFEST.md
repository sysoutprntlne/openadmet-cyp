# Workstream Manifest — Round 2

Five tasks, written after round 1 landed `INPUT_REGISTRY`'s six featurizers and
the auxiliary-head search. Read `tasks/round1/MANIFEST.md` first — **its shared
contract still governs everything here** (metric, pinned split, join keys,
seed-noise floor). This file adds only what is specific to round 2.

Source of the gap analysis: `reports/PXR-SUMMARY.md`. Each task cites the
sections and the competitor reports its claims come from, and round-1 tasks that
are still open are listed below so nobody picks up work that is already claimed.

---

## Checklist

- [ ] **[R1 — Chemprop training, and encoders we train ourselves](chemprop-training.md)**
      Two deliverables. (a) A `train-foundation-model` command that pretrains a
      chemprop encoder on *auxiliary* data — never the scored challenge heads,
      never a row outside the pinned `train` slice — and any such checkpoint
      usable as a representation, `chemeleon[path/to/model.pt]`, so the tabular
      models in `REGISTRY` can consume it too. (b) The recipe axes: loss,
      epochs, architecture, and a ladder of encoder sources from scratch through
      frozen and fine-tuned CheMeleon to our own checkpoints. On the head set
      the head search already chose (scored heads + the 7 auxiliary groups whose
      solo gain cleared seed noise); does not rerun `head-search` except for the
      top-five/bottom-five re-check. `--from-foundation` already accepts a local
      path and `chemprop train` already writes a loadable `model_0/best.pt`, so
      the chemprop side needs no new code — the new code is the representation
      resolver and the bookkeeping. **Heavy GPU — SCC.** *No deps.*
- [ ] **[R2 — CYP pharmacophore features](pharmacophore-features.md)**
      Heme-coordination SMARTS and co-crystal ligand similarity. Cheapest task
      here, and it is T6's 2D control. *No deps. CPU.*
- [ ] **[R3 — Uni-Mol and ligand 3D shape](unimol.md)**
      Ligand-only 3D encoder plus RDKit shape descriptors. Feeds T4. *No deps.
      GPU for the encoder.*
- [ ] **[R4 — Auto-research agent](auto-research-agent.md)**
      Unattended propose → run → log loop over data, models, hyperparameters and
      representations. Qwen3-Coder-Next in a Docker container; AIDE and
      MLE-agent evaluated as the harness. **Three** surveyed PXR entries were
      agent-driven — rank 3, rank ~40 and Acellera — so this is prior art, not
      speculation. *Blocked on the
      evaluation harness below, and on Denali standing up the container and the
      model server.*
- [ ] **[R5 — BioEmu-2 conformational-switch features](bioemu-switch-features.md)**
      Check that ligand-conditioned BioEmu-2 ensembles match Amber MD (5
      inhibitors + 5 non-inhibitors per isoform, apo MD as comparator), then run
      it on every molecule and extract switch features. Feature list deliberately
      left open until simulations exist. Two gates; a null at either is a result.
      *Soft dep on T5/T6. Heavy GPU — SCC.*

**Start R1 first.** It is the only task that can invalidate the shared
contract: if a better recipe changes the `0.7557` / `0.7193` reference numbers,
every task in both rounds is comparing against stale baselines. Better to find
that out now than after three more workstreams have cited them.

**The reference numbers on `main` are wrong.** `tasks/round1/MANIFEST.md`,
`RULES.md` and `reports/head_search_analysis.md` on `main` still quote
`0.9430 ± 0.0018` / `0.7184 ± 0.0038`, which came from a run whose baseline
went NaN at epoch 1 (`reports/head_search_baseline_bug.md`). The fixed
2026-09-22 run gives no-aux `0.7557 ± 0.0036` and best (`CYP1A2`)
`0.7193 ± 0.0065`. The fix and the corrected docs are on
`feat/chemical-space` (commits `72617aa`, `2c0c9a4`) and have not been merged;
`main`'s `head_search.py` still has the bug. Round 2 uses the fixed pair
throughout.

**R2 is the cheapest and gates the most expensive thing we own.** Round-1 T6
plans to spend GPU-weeks measuring heme coordination in 3D. R2 measures the
same chemistry as a substructure flag in an afternoon. T6's gain has to be
reported over R2's, so R2 should land before T6's fan-out is queued.

---

## Not yet written

- **Evaluation harness.** One command that returns macro RMSE over the four
  scored heads on the **pinned** `split` column, mean and spread over ≥3 seeds,
  with train metrics alongside validation, written to CSV. This does not exist:
  `evaluate-models` offers only `random`, `scaffold` and `butina`, so no
  sklearn-path task can currently comply with the round-1 contract, and
  `reports/t2_solo_representations.md` is single-seed for exactly this reason.
  R4 cannot start without it. Needs an owner.
- **Curation audit.** Reactive-group, structural-alert and label-conflict
  flagging, per `reports/PXR-SUMMARY.md` §1. Brief not written yet.
- **Two corrections to `reports/PXR-SUMMARY.md`**, found while writing R4 and not
  yet applied. (1) RyeCatcher's entry was an autonomous Claude loop; the survey
  cites it more than any other team and does not record that. (2) The survey
  labels RyeCatcher rank 67, while the author's own write-up says ~40 of 211 —
  and the survey's own §7 says v50 regressed "rank ~40 to ~87", which agrees with
  the author. Probably two different boards; needs one person to check and fix
  the label, since it appears in several briefs.
- **Single-concentration auxiliary heads.** `data/single_concentration_train.csv`
  is 17,504 weak log2fc measurements on 4,376 molecules, all four scored enzymes,
  zero overlap with `data/test.csv` — and **11,505 of those cells have no pIC50
  at all**, against 6,525 pIC50 cells in `data/train.csv`. Nothing ingests it;
  `build-union` carries only the AID 1851 columns. `emax_train.csv` and
  `tdi_train.csv` are likewise unused. Highest value per hour on the board.
  Brief not written yet. **Whoever writes it must carry R1's filter rule:** the
  file's molecules land 3,484 `train` / 446 `val` / 445 `test` in the pinned
  split, so any use of it — as auxiliary heads or as pretraining data — is
  filtered to the train slice first, and `emax_train.csv` / `tdi_train.csv` are
  all 4,905 challenge molecules and need the same filter.

---

## Round-1 tasks still open

T1, T3, T4, T5, T6, T7 and T8 are all unticked in
`tasks/round1/MANIFEST.md`. T3 (model zoo) was attempted in PR #8 and closed;
it should be redone, and it pairs with the evaluation harness above. Round 2
does not supersede any of them.

---

## Shared files — both rounds

The round-1 table lists only T2 and T6 against `INPUT_REGISTRY`, and only T3
against `requirements.txt`. Both are now wider. **This table is the current
one.**

| File | Touched by | Rule |
|---|---|---|
| `src/data_tools/inputs.py` (`INPUT_REGISTRY`) | T2 (done), T6, R2, R3 | Append one line per featurizer. Never reorder. |
| `src/data_tools/inputs.py` (`featurize` / name resolution) | R1 | R1 alone replaces the unknown-name check with a resolver for parameterized names like `chemeleon[<path>]`, and changes the cache stem for those. It adds no `INPUT_REGISTRY` entry, so the conflict surface against T6/R2/R3 is the import block only. |
| `src/models/__init__.py` (`REGISTRY`) | T3, R3 | Append one line. Never reorder. |
| `requirements.txt` | T2 (done), T3, R3 | Append only; say in the PR why the dep is worth it. |
| `tasks/round1/MANIFEST.md` (reference numbers) | R1 | Only R1 may change `0.7557` / `0.7193`, and only in the PR that measures the new recipe. |
| `reports/head_search_analysis.md` | T1 | Append. R1 takes its head set from this report and does not edit it. |

`src/representations/` holds one new file per featurizer — `structure.py` (T6),
`cyp_pharmacophore.py` (R2), `shape.py` and `unimol.py` (R3), `chemprop_encoder.py`
(R1) — so the directory is shared but no file is. `[project.scripts]` in
`pyproject.toml` gains one line from R1 (`train-foundation-model`) and one from
R4 (`auto-research`); append, never reorder.

---

## Where round 2 touches round 1

There is no duplicated work here, but there are five couplings worth knowing
about before you start.

**1. "Uni-Mol" means two different things.** T5 uses **Uni-Mol Docking V2**, a
protein–ligand docking model that produces poses in a prepared CYP pocket. R3
uses the **Uni-Mol molecular encoder**, which takes one ligand and returns an
embedding with no protein involved. Different checkpoints, different outputs,
different dependencies. Do not assume one gives you the other.

**2. R2 and T6 measure the same chemistry at different fidelity.** T6's
headline feature is the geometric one: distance from the nearest ligand nitrogen
to the heme iron, and the approach angle. R2 asks the substructure question —
does the molecule have an unhindered azole or pyridine nitrogen at all. R2 uses
T6's atom-environment categories deliberately so the two are comparable. The
consequence is a reporting requirement, not a merge conflict: **T6's gain must
be measured over R2's features**, or 3D will be credited with signal a SMARTS
pattern already supplies.

**3. R1 can move the goalposts.** T1, T4, T6 and T7 all score against the
`0.7557` baseline and the `0.7193` best config. R1 is explicitly trying to
change how the model behind those numbers is trained. Hence the ordering above.

**4. R4's reading list overlaps R2's.** The rank-3 agent team found
structure-based scoring (docking, Boltz-2) redundant to a 2D stack, which bears on
T5 and T6; and they put a 20-task nuclear-receptor auxiliary panel in their final
blend, which bears on how widely the data axis is worth searching. Both findings
are written up in R4 and referenced from R2.

**5. R4 writes into files other tasks own.** The agent's editable surface is
`src/models/`, `src/representations/` and the two registries — the same files
T3, T6, R2 and R3 work in. It is fenced by construction: it runs in its own
`git worktree` on a scratch branch and never pushes, and a human opens any PR
that comes out of it. If you are running the agent while someone else is
mid-task in those files, say so in the group chat.

---

## Running on the BU SCC

Same as round 1: **contact Denali with your code** rather than fighting the
scheduler. Reuse `scripts/head_search.qsub` + `scripts/run_head_search.sh` and
keep `#$ -l gpu_c=7.0`.

Heavy GPU: R1 (every arm is a training run), R3 (encoder embedding and
fine-tuning).
CPU-only: R2, and phase 1 of R4.
R4 phase 2 writes job files but does not submit them — a human does.
