# Chemprop training recipe, and encoders we train ourselves

**Depends on:** nothing. Uses the pinned split and `head-search` as they are.
**GPU:** yes — every run is a chemprop training. SCC.
**Owns:** `reports/chemprop_recipe.md` (new); `src/models/train_foundation.py`
(new) and its `[project.scripts]` line; `src/representations/chemprop_encoder.py`
(new); the name resolver in `src/data_tools/inputs.py`; a
`scripts/recipe_*.qsub` job file; `src/models/train_chemprop.py` only if a new
flag is genuinely needed.

---

## What this task is for

Two things, and the second is the new one:

1. **Tune how chemprop is trained.** Every chemprop number we have — the
   `0.9430 ± 0.0018` no-aux baseline and the `0.7184 ± 0.0038` best config in
   `tasks/round1/MANIFEST.md` — came from `head-search` with chemprop's
   defaults: 30 epochs, default loss, default depth and width, encoder trained
   from scratch. We tuned *which heads* to train on and never touched *how* the
   model is trained.
2. **Train our own encoders on auxiliary data and use them as
   representations.** We currently have exactly one pretrained encoder,
   CheMeleon, and we use it two ways: frozen, as the `chemeleon` entry in
   `INPUT_REGISTRY`, and as an initialization via `--from-foundation chemeleon`.
   Both are somebody else's weights trained on somebody else's objective. This
   task adds a `train-foundation-model` command that produces *our* checkpoints
   from *our* auxiliary data, and makes any such checkpoint usable as a
   representation — `chemeleon[path/to/model.pt]` — so every tabular model in
   `REGISTRY` can consume it, not just chemprop.

The pretraining data must be **auxiliary data, not the challenge data.** That
constraint is the spine of the whole design; the section below says why it is
not merely tidy but load-bearing.

## Why this matters for the challenge

All citations are to `reports/PXR-SUMMARY.md` unless a primary link is given.

- **Foundation initialization was the largest single gain in the field.**
  discoverybytes (rank 11) measured test RAE dropping from ~0.62 to ~0.59 when
  the D-MPNN was initialized from CheMeleon pretrained weights instead of
  random (§2). Whitebox (rank 175) fine-tuned CheMeleon end-to-end with
  scaffold 5-fold CV and averaged the five fold models (§2, §3).
- **Moving the foundation encoder's weights beat freezing them.** Josifovski's
  write-up (August 2026) compared a frozen CheMeleon encoder against CheMeleon
  fine-tuned end-to-end, five shared seeds with Dunnett-adjusted comparisons:
  held-out MAE 0.479 → 0.417 on an 80/20 holdout and 0.527 → 0.477 on 513
  challenge analogues. Read it carefully — the fine-tuning there also consumed
  extra weak single-concentration screen labels, so it is not a clean
  frozen-versus-fine-tuned contrast; the extra data and the unfrozen weights are
  confounded. What it does establish is that the frozen embedding is leaving
  something on the table.
- **Pretraining on auxiliary data worked, and the size of the gain was
  modest.** mp-alex (rank 64) is the most complete staged-pretraining record in
  the field. Their own table:

  | encoder | pretraining data | OOF RAE |
  |---|---|---|
  | Chemprop-vanilla | none, from scratch | 0.544 |
  | Chemprop-SC | 21K single-concentration pseudo-pEC50 | **0.517** |
  | Chemprop-NR | ChEMBL nuclear-receptor EC50 (PXR, FXR, LXRa, LXRb) | **0.549** |
  | GPS pretrained | 3.08M ChEMBL+ZINC, then 21K single-conc, then pEC50 | 0.534 |
  | TorchMD-NET | PCQM4MV2 3.4M denoising, then ChEMBL PXR, then pEC50 | 0.591 |
  | ChemBERTa-77M-MLM | PubChem masked LM | 0.614 |

  Two things to take from it. The **in-domain** auxiliary data (the same assay,
  weaker readout) bought ~0.027 RAE. The **off-domain** auxiliary data (related
  receptors from ChEMBL) came out *worse than scratch* — 0.549 versus 0.544.
  And mp-alex's two chemprop rows were run on different seeds (2 versus 0), so
  even the positive result is seed-confounded by their own description. The
  honest summary of the field's best pretraining evidence is: in-domain
  auxiliary data helps a little, off-domain probably does not, and nobody
  measured it cleanly. We can measure it cleanly, because we have a pinned
  split and a seed-noise floor.
- **Encoder-only transfer is the mechanism.** Copy the message-passing weights,
  reinitialize the FFN (§4). This avoids a size mismatch when the number of
  heads differs between pretraining and fine-tuning — which it always will,
  since the pretraining table has different columns. chemprop 2.3.1 does
  exactly this for us; see the next section.
- **MAE loss beat MSE loss.** discoverybytes measured ~0.01 RAE, attributed to
  robustness to outliers on noisy labels (§3).
- **Multitask beat single-task** by 0.070 RAE in discoverybytes' four-task
  versus two-task comparison (§3). This is the one finding we have already
  reproduced — it is the whole of `reports/head_search_analysis.md`.
- **Competitive settings were deeper and longer than ours.** 3–5
  message-passing layers, hidden dim 250–600, dropout 0.1–0.3, 60–100 epochs on
  scaffold-split CV (§3). We train 30.

**What failed, and each one is a trap this task could walk into.**

- **Same-family encoders saturate.** RyeCatcher tested ChemBERTa, GIN-ACtriplet,
  UniMolV2-310M, MaskMol and an ImageNet ViT on molecular images; all converged
  to Spearman > 0.88 against their primary ensemble (§2). A chemprop encoder we
  pretrain ourselves is, architecturally, the same family as CheMeleon. If it
  does not *decorrelate* from `chemeleon`, it adds compute and not signal — so
  correlation against the existing representation is a required output of this
  task, not a nice-to-have.
- **MolFormer-XL produced Spearman ρ = −0.007** in discoverybytes' hands (§2).
  Sequence-model embeddings were the field's clearest dead end.
- **Embeddings from a per-fold-trained encoder shift distribution.**
  discoverybytes fed TabPFN CheMeleon embeddings produced by OOF-fold models,
  which differ from all-data embeddings, and the shift degraded test
  performance enough that they dropped TabPFN from the final stack (§3). This
  is the precise failure mode that the "pretrain on auxiliary data only" rule
  immunizes us against: an encoder that never saw a challenge label produces
  the same embedding for a train row and a test row, so it is a fixed
  featurizer and nothing downstream has to reason about fold provenance.
- **External data is useful for pretraining, dangerous as a calibration
  anchor.** AuP7s (rank 27) calibrated an OOF ensemble against a public
  ChEMBL-adjacent set and inflated portal MAE substantially (§4).
- **Distillation destroys diversity.** discoverybytes trained a chemprop on the
  ensemble's own predictions; Ridge then found UniMol and MolE redundant and
  cut their weights, ΔRAE = +0.0126 (§6). Do not pretrain on anything that is
  itself a model output.

**One reason to distrust the whole transfer.** The PXR leaderboard was scored on
MAE (RAE = MAE / MAD). We are scored on RMSE. Part of the MAE-loss gain there is
simply training on the metric being reported, and the auxiliary-data gains were
measured on a different protein with a different assay. Everything above is a
hypothesis list, not a recipe.

## What chemprop 2.3.1 already does for us

Checked against the installed source, not the docs:

- **`--from-foundation` accepts a local file path**, not just `CHEMELEON`. With
  a path it calls `MPNN.load_from_file(...)` and takes that model's
  `message_passing` block and its `agg`, discarding the predictor
  (`chemprop/cli/train.py:1448-1466`). That *is* encoder-only transfer, built
  in. The FFN is constructed fresh for the new target columns.
- **`chemprop train` writes `<output-dir>/model_0/best.pt`**
  (`cli/train.py:1795`, `:1993`), and that file is what
  `MPNN.load_from_file` reads. So the output of one training run is directly
  the input to `--from-foundation` of the next. **No new export format, no
  checkpoint surgery, no new chemprop code.**
- **`train-chemprop` already passes a path straight through** — the
  `--from-foundation` handling in `src/models/train_chemprop.py` only special-cases
  the literal string `chemeleon` and forwards anything else verbatim. And
  `head-search --extra` forwards unknown flags. So the chemprop consumption
  side of this task is **already finished**; it just has never been run.
- **Architecture flags are ignored when a foundation is used.**
  `--message-hidden-dim`, `--depth`, `--dropout`, `--aggregation`,
  `--atom-messages` and friends are logged as ignored
  (`cli/train.py:707-725`). The checkpoint fixes the architecture. This settles
  the open question the previous version of this brief left: do not try to vary
  depth or width in any foundation-initialized arm; vary them only in the
  from-scratch arm.
- **`--checkpoint` + `--freeze-encoder`** is the other half: load a checkpoint
  and freeze the message-passing block, training only the FFN. That is a third
  behaviour, distinct from both `--from-foundation` (initialize and keep
  training) and from embedding-then-sklearn. `--model-frzn` does the same thing
  and is deprecated; use `--checkpoint --freeze-encoder`.
- **Featurizer mode is a silent-corruption risk.** CheMeleon requires
  `--multi-hot-atom-featurizer-mode V2` and chemprop enforces that for the
  named foundation — but **it does not enforce it for a local path**. A
  checkpoint that was itself CheMeleon-initialized must be reused with V2 or the
  atom features will not mean what the weights expect, and nothing will error.
  Record the featurizer mode in the checkpoint's metadata and assert it on
  reuse.

## New deliverable 1: `train-foundation-model`

A thin wrapper, deliberately. The training itself is `train-chemprop` on an
auxiliary CSV; everything of value this command adds is bookkeeping that makes a
checkpoint trustworthy six weeks later.

```
train-foundation-model \
  --model chemeleon \                     # starting point: chemeleon | scratch | <path>
  --data data/union_train.csv \
  --target-columns CYP1A2_pctinhib_aid1851 CYP1A2_pIC50_aid1851 ... \
  --split-column split --pretrain-on train \
  --epochs 60 --loss-function mae --seed 0 \
  --name aid1851_allheads_v1
```

It must:

1. **Refuse to train on rows outside the pinned pretraining slice.** Read the
   `split` column, keep only `train`, and fail loudly — not warn — if asked to
   include `val` or `test`. This is the one hard behaviour in the command. See
   the data section for why.
2. Delegate to the existing `train-chemprop` path so every chemprop flag stays
   available.
3. Write `results/foundation/<name>/model_0/best.pt` plus a
   `metadata.json` sidecar beside it recording: source data file and its sha256,
   the exact target columns, row and molecule counts after filtering, the split
   column and the slice kept, the starting point (`chemeleon` / `scratch` /
   path), epochs, seed, atom featurizer mode, chemprop version, the repo git
   SHA, and the full argv. A checkpoint without this sidecar is not usable as a
   representation — the loader should refuse it.
4. Print the resulting representation string to copy, e.g.
   `chemeleon[results/foundation/aid1851_allheads_v1/model_0/best.pt]`.

Do not build a training loop, a config system, or a sweep runner inside it. If
you want five checkpoints, write five lines in a qsub script.

## New deliverable 2: a checkpoint as a representation

`INPUT_REGISTRY` maps a fixed name to a featurizing callable. A checkpoint
representation needs a *parameterized* name, which the registry cannot express.
Add a resolver, keep the registry append-only:

- **Name syntax** `chemeleon[<path>]` — a base name and a bracketed argument.
  `featurize()` currently rejects any name not in `INPUT_REGISTRY`
  (`src/data_tools/inputs.py:106`); replace that check with a `resolve(name)`
  that returns the registry entry for a bare name and constructs a
  `ChempropEncoder(path)` for a bracketed one. Existing names behave exactly as
  before, and nothing in `INPUT_REGISTRY` moves or is renamed.
- **`ChempropEncoder(Representation)`** in
  `src/representations/chemprop_encoder.py`: `MPNN.load_from_file(path)`,
  `.eval()`, then `model.fingerprint(bmg)` per batch — the same shape as
  `src/representations/chemeleon.py`, which is the model to copy. Invalid SMILES
  give an all-NaN row, as there. Assert the featurizer mode from
  `metadata.json` matches the one used to build the `MolGraph`s.
- **Cache the embeddings by checkpoint content hash, not by path.** The cache
  writes `{input_name}.npy` / `.idx` into `data/features`
  (`inputs.py:70-71`), so a name containing `/` is not a legal filename, and —
  much worse — keying on the path means retraining a checkpoint to the same
  location would silently serve stale embeddings from the old weights. Use
  `f"chemeleon-{sha256(checkpoint)[:12]}"` as the cache stem and record the full
  path in a small sidecar next to the cache files. The existing
  content-addressed-by-SMILES design then works unchanged.
- **Note the two scale differences** before comparing a checkpoint embedding to
  the `chemeleon` one. `MPNN.fingerprint` applies the model's batch-norm layer
  (`chemprop/models/model.py:126-134`); our `CheMeleon` representation builds a
  fresh `MPNN` around the raw message-passing block and so does not. And a
  checkpoint's dimensionality is whatever that encoder's hidden size is, not
  necessarily 2048. Do not assume the two are interchangeable column-for-column.
- **`--input all` will not pick these up**, since `evaluate-models` enumerates
  `INPUT_REGISTRY`. That is correct — an arbitrary path is not a default — and
  should be stated in the report so nobody concludes the checkpoint was
  evaluated when it was not.

## The auxiliary data, and the line not to cross

We have four auxiliary sources on disk. None of them may be used wholesale,
and the reason is specific rather than hygienic.

The pinned `split` column covers all 20,870 rows of `data/union_train.csv`, and
grouping is clean: 20,866 `inchikey_block` groups, **zero** of which span two
splits. Counts by source and slice, from the union table:

| | train | val | test |
|---|---|---|---|
| challenge pIC50 only | 3,782 | 480 | 467 |
| challenge **and** AID 1851 | 143 | 10 | 23 |
| AID 1851 only | **15,965** | 0 | 0 |

- **`data/pubchem_aid1851.csv` (16,139 molecules, CYP1A2/2C9/2C19/2D6/3A4
  percent-inhibition and pIC50) is the clean pretraining set.** All 15,965
  AID-1851-only rows are on the train side of the pinned split, and zero of its
  molecules appear in `data/test.csv`. It is also the right *kind* of data by
  mp-alex's evidence: same enzymes, weaker readout, in-domain. But 33 of its
  molecules are val or test rows of our split (10 + 23 above), so pretraining on
  the file as it sits on disk shows the encoder qHTS labels for 33 lockbox
  molecules. Filter by `split == "train"`; that is what deliverable 1 item 1 is
  for.
- **`data/single_concentration_train.csv` (17,504 rows, 4,376 molecules) must be
  filtered, hard.** Its molecules are challenge molecules — 4,375 of 4,376 are
  in `data/train.csv` — and they land **3,484 train / 446 val / 445 test** in
  the pinned split. Pretraining an encoder on all of it and then using that
  encoder as a representation puts weak activity labels for 445 lockbox
  molecules inside the featurizer. Not the scored pIC50, but close enough that
  no val→test gap measured afterwards would mean anything. Filtered to the
  train slice it is 3,484 molecules, which is the in-domain analogue of
  mp-alex's best-performing Chemprop-SC and worth doing properly — it is big
  enough to be its own brief, and the manifest lists it as one.
- **`data/emax_train.csv` and `data/tdi_train.csv` (6,145 rows each)** are
  entirely challenge molecules (all 4,905 of `train.csv`) with different
  readouts — Emax versus positive control, and time-dependent inhibition. Same
  filter, same reasoning. Both are completely unused today.
- **External CYP data we do not have yet.** Two candidates, in order of
  promise:
  - **OpenADMET's own ChEMBL baseline**,
    `openadmet/cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1` — a multi-task
    CheMeleon model trained on ChEMBL 37 pIC50 for *exactly our four isoforms*,
    Apache-2.0. If its weights are a chemprop MPNN checkpoint it is a
    ready-made `--from-foundation` target and a ready-made representation, for
    the cost of a download. Two unknowns to resolve first, both cheap: the card
    does not name the weight files or format (it ships through the `openadmet`
    / Anvil tooling with `git lfs`), and it makes **no statement about overlap
    with the challenge test set**. A ChEMBL-trained CYP model may well have seen
    some of our 750 test molecules. Check the overlap against `data/test.csv`
    before trusting any number it produces, and say in the report what you
    could and could not verify. Treating it as the *first* experiment rather
    than the last is the right call on expected value.
  - **ChEMBL CYP IC50 directly.** The isoform-specific PubChem assays (AIDs
    410, 883, 884, 891, 899) overlap AID 1851 heavily and are not independent
    data. ChEMBL is where genuinely new potency records are. Curated
    compilations in the literature reach ~12K compounds across seven isoforms
    after filtering. This is a data-engineering task of its own; do not start
    here.

**The rule, stated once:** an encoder used as a representation must never have
seen a label for a molecule in the pinned `val` or `test` slice. Pretraining
data is filtered to `split == "train"` and the filter is enforced in code, not
in a comment.

## Scope

Change one thing at a time against a fixed reference, then combine the winners.
Run every recipe arm on **two** configurations: no auxiliary heads, and the
current best (`CYP1A2+CYP2C8`). A recipe that only helps one of them is a
different finding from one that helps both.

### Part A — the encoder ladder (the headline)

1. **Build the two new pieces** — `train-foundation-model` and
   `ChempropEncoder` + the resolver — and prove them with the cheapest possible
   check: train a foundation checkpoint for two epochs on the filtered AID 1851
   heads, load it as `chemeleon[...]`, and confirm `evaluate-models` runs a
   Ridge on its embeddings end to end. Correctness first, numbers second.
2. **Score the ladder of encoder sources** on the same pinned split and the same
   seed list:
   - **from scratch** — chemprop's default, our current reference;
   - **`--from-foundation chemeleon`** — CheMeleon fine-tuned end-to-end;
   - **frozen CheMeleon** — the existing `chemeleon` representation, embed once
     and fit a head. No training run; the cheap third point that makes the
     others interpretable;
   - **`--from-foundation <our AID 1851 checkpoint>`** — our own encoder,
     pretrained on filtered external data, then fine-tuned on the scored heads;
   - **our checkpoint frozen**, as `chemeleon[path]` through `featurize`, fed to
     the tabular models in `REGISTRY`;
   - **`--checkpoint <ours> --freeze-encoder`** — frozen inside chemprop, FFN
     only. Cheap, and it separates "the embedding is good" from "sklearn on the
     embedding is good."
   - and, if the download resolves, the **OpenADMET ChEMBL baseline** in both
     the initialize and frozen roles.

   CheMeleon is a featurizer in this repo and it stays one: everything above is
   either a `Representation` or a `train-chemprop` flag. **Nothing in this task
   appends to `REGISTRY` in `src/models/__init__.py`.**
3. **Report the correlation, not only the score.** For every frozen encoder,
   give the Spearman correlation of its predictions against the `chemeleon`
   baseline's on `val`. RyeCatcher's saturation result says a same-family
   encoder can score well and still be worthless in a blend. An encoder that
   scores slightly worse but correlates at 0.75 is a better outcome for this
   project than one that scores slightly better at 0.97, and the report should
   say so in those terms.
4. **Then score the single winner on the held-out `test` slice**, reported next
   to its `val` number. Every other arm in this task is tuned on `val` alone.
   This is the one place we spend the lockbox, and the val→test gap is as much
   the result as the ranking is — §5 of the survey is a list of teams whose
   validation numbers did not survive the blinded set. Score it once.

### Part B — the recipe axes

5. **Loss.** `mse` (reference) versus `mae`. Report per isoform, not just
   macro — a robust loss should help most where labels are noisiest.
6. **Training length.** 30 versus 60 versus 100 epochs. Check the validation
   curve, not just the endpoint — if 30 was already past the minimum, say so.
   Run this for the pretraining runs too; 60 epochs on 16K rows is a different
   question from 60 on 4.9K.
7. **Architecture, from-scratch arm only.** Depth 3–5, hidden 300–600, dropout
   0–0.3. A small grid or a handful of random draws. Foundation arms ignore
   these flags outright, as established above.
8. **Combine** the per-axis winners and confirm the gains add. They often do
   not.
9. **Does the head ranking survive?** Rerun `head-search --strategy single` for
   the top five and bottom five heads under the winning recipe. The 0.22 gap
   between baseline and best was measured with a weak encoder; a pretrained
   encoder may already contain much of what the auxiliary heads were teaching —
   and if our own encoder was pretrained *on* the AID 1851 heads, it certainly
   does, which makes this the sharpest version of the question anyone has
   asked. If the ranking changes, `reports/head_search_analysis.md` needs a
   correction, and that is the most important output of this task.

**Budget before queueing.** Part A is roughly 8 encoder sources × 2 configs × 3
seeds plus the pretraining runs themselves. Arms 5–7 on two configs at three
seeds is about 30 runs; arm 9 is that again. Write the run count down and agree
it before submitting.

## What not to do

- **Do not pretrain on unfiltered auxiliary files.** Covered above. If a
  checkpoint's `metadata.json` does not record the split filter, the checkpoint
  is not usable and neither is anything downstream of it.
- **Do not re-split or re-seed between arms.** Same pinned split, same seed
  list for every arm, or the comparison is measuring noise.
- **Do not pick the recipe on one seed.** A difference under ~0.004 is seed
  spread (manifest).
- **Do not pretrain per fold.** One checkpoint, trained once on the train
  slice, reused for every seed and every downstream model. Per-fold encoders
  are the discoverybytes distribution-shift failure, and they also make the
  embedding cache meaningless.
- **Do not reintroduce pretrain-then-finetune on our own *scored* heads.** The
  README explains why joint multi-task training replaced it (`README.md:39`).
  Pretraining on an external or weaker-readout source and then fine-tuning is a
  different thing: the pretraining objective is not the scored objective.
- **Do not pretrain on any model's predictions**, ours or anyone's. That is the
  distillation backfire.
- **Do not fine-tune the encoder per fold and then feed its embeddings to
  TabPFN.** Already flagged in `tasks/round1/T3-model-zoo.md`.
- **Do not add a sequence model.** MolFormer-XL's ρ = −0.007 and ChemBERTa's
  0.614 are the field's evidence; if someone wants to argue otherwise, that is
  a separate brief with a separate budget.

## Tools

- `chemprop` 2.3.1 — already a dependency. Confirmed flags:
  `--loss-function {mse,mae,...}`, `--from-foundation`, `--checkpoint`,
  `--freeze-encoder`, `--frzn-ffn-layers`, `--depth`,
  `--message-hidden-dim`, `--dropout`, `--ffn-hidden-dim`,
  `--ffn-num-layers`, `--epochs`, `--warmup-epochs`, `--max-lr`,
  `--ensemble-size`, `--multi-hot-atom-featurizer-mode`.
- `head-search`, `train-chemprop`, `evaluate-models` — existing commands.
- `scripts/head_search.qsub` + `scripts/run_head_search.sh` are the job
  template; `run_head_search.sh` appends whatever arguments it is given.
- `git lfs` and possibly the `openadmet-models` package, only if the ChEMBL
  baseline download is attempted. No credentials or tokens in the PR.
- No other new dependencies.

## Done when

- `train-foundation-model` exists, refuses out-of-slice rows with a test that
  proves it, and writes a `metadata.json` next to every checkpoint.
- `chemeleon[<path>]` resolves through `featurize`, caches by checkpoint hash,
  and the bare existing names still behave identically.
- `reports/chemprop_recipe.md` has: one table for the encoder ladder (source,
  macro RMSE ± seed spread, per-isoform RMSE, Spearman against the `chemeleon`
  baseline, embedding dimension, both reference configurations) and one table
  per recipe axis, with the exact commands.
- The winner's held-out `test` score and its val→test gap, with a note of how
  many times the lockbox was looked at.
- A verdict in one sentence on each of: does an encoder we pretrained beat
  CheMeleon; does it *decorrelate* from CheMeleon; does the auxiliary-head
  ranking hold under the winning recipe.
- Every checkpoint that gets a number in the report is reproducible from its
  `metadata.json` alone.
- If the recipe changes, the two reference numbers in the manifest are updated
  in the same PR, since every other task compares against them.

## SCC note

All of it. **Contact Denali with your code** to get the jobs queued. Reuse
`scripts/head_search.qsub` and keep `#$ -l gpu_c=7.0`. Checkpoints go under
`results/`, which is gitignored — the report cites paths, never commits weights.
If the ChEMBL baseline download is attempted, remember the compute nodes have no
internet: download on a login node, then queue.

## Sources

Findings are quoted from our survey, `reports/PXR-SUMMARY.md`; the primary
write-ups are:

- discoverybytes (rank 11) — CheMeleon init, MAE loss, multitask, the TabPFN
  distribution-shift and distillation failures:
  https://github.com/discoverybytes/openadmet-pxr-blind-challenge/tree/main/activity-prediction
- mp-alex (rank 64) — the staged-pretraining table quoted above, including
  Chemprop-SC 0.517 / vanilla 0.544 / Chemprop-NR 0.549:
  https://github.com/mirror-physics/pxr_openadmet_blind_challenge/blob/main/reports/track1_activity/submission_6model_adaptive_elasticnet.md
- RyeCatcher (rank ~40) — representation saturation across five pretrained
  encoders, and the multi-head external-data model:
  https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026
- ldbc1999 (rank 65) — concentration-aware single-concentration pretraining and
  encoder-only transfer: https://github.com/lizyurkewych-git/pxr-challenge
- Whitebox (rank 175) — end-to-end CheMeleon fine-tuning with fold averaging:
  https://huggingface.co/Whitebox2026/openadmet-pxr-challenge
- auP7s (rank 27) — external data as a calibration anchor, and why not to:
  https://gist.github.com/chemotica/a49b002eda2f7fd5eef2dca4f98f8ad7
- Kalen Josifovski (August 2026) — frozen versus fine-tuned CheMeleon on
  held-out and analogue sets, five seeds, Dunnett-adjusted:
  https://kalenjosifovski.github.io/2026-08-21-when-weak-pxr-labels-teach-a-model-potency/
- OpenADMET's own ChEMBL-trained multi-task CYP baseline (Apache-2.0, ChEMBL
  37, our four isoforms):
  https://huggingface.co/openadmet/cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1
  — and the hERG card, which documents the same format more fully:
  https://huggingface.co/openadmet/herg-chemeleon-baseline
- CheMeleon pretrained weights: https://arxiv.org/abs/2506.15792
- Heid et al. (2024), Chemprop: https://doi.org/10.1021/acs.jcim.3c01250
- Veith et al. (2009) qHTS CYP panel, the source of AID 1851:
  https://pubchem.ncbi.nlm.nih.gov/bioassay/1851
