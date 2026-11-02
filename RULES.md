# Contributing Rules — OpenADMET CYP Inhibition Challenge

Rules for working in this repo as part of the BU AI & ML (BUAIS) team effort.
Read `tasks/round1/MANIFEST.md` first — it holds the scientific contract (metric, split,
join keys, file ownership). This file covers process: what a PR should look like,
what you may and may not submit externally, and how to use AI tooling.

The sibling repo [dschlesinger/openadmet-pxr](https://github.com/dschlesinger/openadmet-pxr)
is the reference for the shape we are aiming at. When in doubt about structure,
copy what it does.

---

## 1. Before you write code

- **Claim a task.** Pick one of T1–T8 from `tasks/round1/MANIFEST.md`, say so in the group
  chat, and read that task's file in full. The tasks are scoped for disjoint file
  ownership; picking one is what keeps merges clean.
- **Branch off `main`**, named `<type>/<short-slug>` — `feat/`, `fix/`, `docs/`,
  `refactor/`. Example: `feat/maccs-representation`.
- **Don't start a ninth workstream.** If you think something is missing from the
  manifest, add it to the manifest in its own PR first.

---

## 2. What a PR should look like

### One task, one PR

A PR does one thing and says what it does in the title. If you touch two
workstreams, open two PRs. If a PR needs a refactor to land cleanly, the refactor
is its own PR merged first.

### Reuse everything you can

This is the rule that matters most, and the thing that makes the PXR repo work.
Before adding code, look for the existing abstraction:

| If you are adding… | Subclass / register this | Do **not** |
|---|---|---|
| A featurizer | `Representation` in `src/representations/`, then append to `INPUT_REGISTRY` in `src/data_tools/inputs.py` | write a bespoke featurize function inside a model |
| A model | `CYPModel` in `src/models/`, set a unique `name`, append to `REGISTRY` in `src/models/__init__.py` | hardcode a training loop in a script |
| A data source | a builder in `src/data_tools/` that emits its own `{source}_{isoform}_{readout}` columns and calls `check-overlap` | pool new measurements into a scored `pIC50` column |
| A user-facing command | a `main()` plus one line in `[project.scripts]` in `pyproject.toml` | leave it only runnable as `python scripts/foo.py` |

Use the shared helpers rather than reimplementing them:
`data_tools.load.load_data` for loading and splitting, `data_tools.standardize`
for any structural comparison or join, the cached `featurize()` path for feature
matrices, and `scripts/head_search.qsub` + `scripts/run_head_search.sh` as the
template for any new SCC job.

If an existing abstraction *almost* fits, extend it. Adding a parameter to
`load_data` is a better PR than a second loader.

### Make it interchangeable

New components must compose with everything already there, in both directions:

- A new representation works with **every** model in `REGISTRY`.
- A new model works with **every** name in `INPUT_REGISTRY`, including stacked
  inputs (`--input morgan rdkit`).
- Nothing takes a hardcoded path, isoform, head list, or dataset. Those are
  arguments with defaults.
- Registry edits are **append-only**. Never reorder or renumber — other people's
  branches and our pinned results depend on the order.

A component that only runs in the one configuration you tested is not done.

### Report numbers honestly

Any PR claiming an improvement must state, in the PR description:

- **Macro RMSE** over the four scored heads, on the **pinned** `split` column
  from `add-split-column`. Never re-split.
- **At least 3 seeds**, with the spread. A gain under ~0.004 is seed noise and is
  not a result.
- The baseline you are comparing against, by name
  (currently: no-aux `0.9430 ± 0.0018`, best `CYP1A2+CYP2C8` `0.7184 ± 0.0038`).

Paste the actual command you ran. A negative or null result is a perfectly good
PR — write it up in `reports/` and say so.

### Document it

- Update `README.md` when you add a command, model, representation, or data
  source — the tables there are the API surface for everyone else.
- Longer findings go in `reports/` as markdown, in the style of
  `reports/head_search_analysis.md`.
- Tick your task's box in `tasks/round1/MANIFEST.md` when the workstream is actually done.

### Keep the diff clean

- No commented-out code, no debug prints, no `.pyc`, no notebook checkpoints.
- Type-annotate public functions. Keep lines under 120 characters.
- Add deps to `requirements.txt` and say in the PR why the dep is worth it.

---

## 3. Never in a PR

- **Anything under `data/` or `results/`.** Both are gitignored. Code and reports
  go in git; artifacts, checkpoints and predictions stay out. If a reviewer needs
  an artifact, share it out of band.
- **Any molecule overlapping `data/test.csv`.** New ingesters must call
  `check-overlap`. `test.csv` is untouchable.
- **Credentials or tokens** of any kind, including HuggingFace tokens.
- **Re-splits.** If your code re-splits, it will be rejected — it makes the eight
  workstreams mutually incomparable.
- **Large binaries.** Weights download at runtime, as `chemeleon` already does.

---

## 4. The live challenge and HuggingFace — Denali submits

**Denali submits on behalf of BUAIS. Nobody else submits to the live OpenADMET
CYP challenge on HuggingFace — not for BUAIS, and not in any way that could be
read as a second BUAIS entry.**

The challenge rules forbid one team from using multiple HuggingFace accounts. If
several of us submit individually while also working on this repo, we put the
whole team's entry at risk of disqualification — this is not a formality.

Concretely:

- **One team, one account.** BUAIS submits through a single account, handled by
  Denali. Nobody else uploads a submission for this project.
- Do not create a second HuggingFace account, team, or org for challenge
  submissions. Do not submit under a personal account using work from this repo.
- If you already have a personal submission or account for this challenge, say so
  **before** contributing, so we can sort out eligibility rather than discover it
  later.
- Leaderboard probing — repeated submissions to read the test distribution — is
  out of bounds regardless of which account it happens from.
- Nothing here restricts what you learn or publish afterwards; it restricts
  *submitting*.

When your work is ready to go into a submission, open a PR and flag it. Denali
runs the submission.

---

## 5. AI tools

**Using Claude Code or other AI tools is encouraged.** There is no penalty, no
disclosure ritual, and no expectation that you hand-write boilerplate. Most of
this repo was built that way.

The one hard requirement: **you must be able to explain the reasoning behind
every choice in your PR — yours or the model's.** If a reviewer asks why you used
a Butina split here, why that loss is masked, why this head was included, "the
model wrote it" is not an answer. You own the diff you open.

That implies a few habits:

- Read what you are about to commit, all of it.
- Verify API calls and library behavior against real docs, not the model's memory.
  Plausible-looking calls to functions that don't exist are the standard failure.
- **Never let a tool report numbers you did not run.** Every metric in a PR
  description must come from a command you actually executed, on the pinned split.
  Fabricated or hallucinated results are the one thing that will get a PR closed
  on sight.
- Be extra skeptical where the model can't check itself: chemistry conventions,
  sign conventions (see `Max_Response` in the README), and whether a join
  silently under-merged.
- If a tool suggests a new abstraction where one already exists, prefer the
  existing one. AI tools default to writing something new from scratch rather
  than reading the repo and fitting into it; §2 does not.

`CLAUDE.md`-style repo guidance is welcome — add it in its own PR.
