# Auto-research agent

**Depends on:** the round-2 evaluation harness (a single frozen command that
returns macro RMSE on the pinned split). Denali stands up the container and the
model server before the agent is let loose.
**GPU:** none for the agent's own experiments in phase 1. Serving
Qwen3-Coder-Next needs its own GPU allocation. Phase 2 queues chemprop runs.
**Owns:** `src/auto_research/` (new); `docker/auto-research/` (new); one line in
`[project.scripts]` in `pyproject.toml`; `reports/auto_research.md`.

---

## The objective

**Minimize macro RMSE — the mean of the four scored heads' RMSE — on held-out
data.** Stated that way deliberately: RMSE is an error, so the agent's goal
string must say *minimize*, and whoever writes the task description for the
harness should read that line twice. An agent handed "maximize RMSE" will
cheerfully do it.

"Held-out" means two different things and the distinction is the whole design:

- The **`val`** slice of the pinned `split` column (490 rows in
  `data/union_train.csv`) is what the agent sees. Every proposal it scores, it
  scores here.
- The **`test`** slice (also 490 rows) is the lockbox. The agent never reads it.
  Denali scores the agent's recommended configurations on it at a fixed
  cadence, and the `val`→`test` gap is a headline deliverable, not a footnote.
- `data/test.csv` is the blind challenge set: **750 molecules, SMILES only, no
  labels.** There is nothing in it to leak, which is worth knowing because it
  means the pinned `test` slice is the only lockbox we have.

490 rows × 4 heads is a small lockbox. Scored twenty times it stops being one.
Agree the scoring cadence up front and log every look.

## Why this matters for the challenge

After T2 we have six representations, which is 63 non-empty input stacks before
any model choice. T3 adds four or five model families, each with
hyperparameters. The chemprop side has 20 auxiliary-head candidates plus the
training-recipe axes in R1. And as of this round we know there are three unused
label files in `data/` (`single_concentration_train.csv`, `emax_train.csv`,
`tdi_train.csv`) that nothing ingests. Nobody is going to walk that grid by
hand, and the T2 benchmark showed what happens when someone tries: 21 runs, one
seed, results transcribed into a spreadsheet.

**Two teams in our own competition family did this, and one of them came
third.** This is no longer speculative.

**jjaybird606 finished rank 3 at RAE 0.5076** with agents doing the modelling
and the human supplying infrastructure and curated data. Their phase 1 was
headless Claude Code on cloud GPUs driven from a phone, running a DSPy-guardrailed
autonomous loop for roughly **2,500 train-and-submit cycles** under a 4-hour
per-cycle limit, shipping a candidate only when it passed a gate (CV score plus
further checks), with a continuously running "autoresearch" daemon alongside.
What the loop converged on is worth reading closely, because it is our stack:
a uniform blend of multitask chemprop networks varied by depth, aggregation and
task set, an RDKit 2D descriptor model, CheMeleon, and a graph transformer, with
auxiliary multitask heads — a counter-assay and a **20-task nuclear-receptor
panel** — and task-specific weighting. Recipe and architecture search were
autonomous. Validation was 10-fold CV plus a held-out Set 1, combined by greedy
ensemble selection.

**RyeCatcher — Justin Johnson — was the third, and this one matters most to us.**
Final rank ~40 of 211 teams on 513 blinded molecules, top 19%, by an author with
no ADMET or molecular-modelling background, running an autonomous loop driven
largely by Claude. For about a week he was out of the loop entirely while the
agent made the modelling decisions. His own framing of what he contributed is the
thesis of this task: **the scaffolding that defined what counted as trustworthy
evidence.** Code, weights and methodology are published Apache-2.0.

Note what this means for our own documents. `reports/PXR-SUMMARY.md` cites
RyeCatcher more than any other team — the LOCO cluster splits, the honest
per-fold isotonic calibration, the 2D-saturation finding, the v50 regression —
and those citations are load-bearing in R2, R3 and this brief. The survey never
records that the entry was agent-driven. The validation protocol we treat as the
field's best came *out of* an agent loop, which is the strongest evidence here
that the guardrails below are the deliverable rather than overhead.

His loop structure, which is more detailed than jjaybird606's published account:

- **One gate function.** A candidate is accepted only if it improves the
  *honest* score by a set threshold. One function, one definition of better.
- **Two-tier evaluation.** A fast sanity-check pass, then a slower full
  evaluation only for what survives. This is how a loop affords hundreds of
  proposals.
- **Failed candidates archived with their verdicts**, plus checkpoints recording
  state at each phase — so the ledger is a record of what was rejected and why,
  not just what was kept.
- **Red-team reports that tried to attack each claim.** An adversarial pass over
  every kept result. Nothing else in the survey does this, and it is the step
  that catches the v50 class of error.

**v50 is the worked example of why the agent cannot own the scorer.** A candidate
looked like a breakthrough because its calibration step had seen the test
answers; the honest check showed almost no gain; shipping it would have moved the
entry from ~40th to ~87th. `reports/PXR-SUMMARY.md` §7 records the same event as
nearest-neighbour activity transfer, OOF 0.4536 → LB 0.658, and calls it the most
catastrophic regression in any report surveyed. It was caught by the gate, not by
intuition.

**And he stopped on convergence.** Twenty-seven further candidates correlated
above 0.87 with the leading one, which he read as a real performance ceiling —
and concluded that loops should stop once candidates converge. He argues the
system and the documented ceiling were the real deliverables. That is a
termination criterion we did not have, and item 8 below adds it.

His progression is also informative about where the gains are: the tutorial
baseline ranked 131st, his first model was worse than that, computed chemistry
descriptors took him to ~112th, and **the single largest jump came from blending
in a graph model that learns from bonds — to ~42nd.** A second graph model on
extra public data and a foundation-embedding ensemble finished the job.
Descriptors plateaued; the message-passing network was the step change. Our R1
and T3 ordering already assumes that, and this is a clean independent
confirmation.

**Acellera** reported the same division of labour from the other direction: an
agent sequencing featurization, training, tuning and inference, with the
scientific team keeping responsibility for the dataset, the evaluation design
and the interpretation. Their stated lesson — adding labelled compounds did not
automatically improve transfer to the second analogue series, so "data relevance
and evaluation design matter more than dataset size alone" — is the one to pin
above the desk, since this task hands the agent control over which data is used.

**Four of jjaybird606's findings change what we should build.**

- **The overfit signal is observable.** They report that CV kept improving after
  the leaderboard stopped moving. That divergence is a concrete, implementable
  detector, and it is better than any abstract warning: track the `val`
  improvement curve against each lockbox check, and when `val` keeps falling
  while the lockbox does not, the search has started fitting its validation set.
  Build this into the ledger from day one.
- **Their loop cannot be copied directly, because it submitted.** 2,500
  "train-and-submit" cycles means the leaderboard was part of the objective
  function. RULES.md §4 forbids that for us from any account. Our agent gets a
  490-row lockbox consulted on a fixed cadence instead, which is a far weaker and
  noisier signal — so the keep rule and the gate matter *more* here, not less.
- **A broad off-target auxiliary panel earned its place.** A 20-task
  nuclear-receptor panel was in a rank-3 blend. That is evidence against the
  prior that only same-protein, same-readout heads help, and it is the strongest
  argument we have for letting the agent search the data axis widely rather than
  hand-picking heads.
- **Structure-based scoring was redundant for them** — docking and Boltz-2 both.
  Round-1 T5 and T6 plan GPU-weeks on exactly that. One team's negative result is
  not a verdict, but combined with R2 being the cheap 2D control it is a reason to
  settle the question before the fan-out is queued.

Also recorded as dead ends there: post-hoc calibration worsened held-out RAE;
training on Set 1 transferred poorly to Set 2; and an AutoGluon suite over
scikit-fingerprints and Mordred features was judged redundant — relevant to us,
since `mordred` is already in `INPUT_REGISTRY`. Their phase 0, a weak agent on a
small VPS, got stuck in calibration loops and produced low-parameter models,
which is the failure mode the gate exists to catch.

**The risk is still larger than the opportunity if it is built carelessly.** An
agent that runs 300 experiments against 490 validation rows and reports the best
one has found the luckiest configuration, not the best one — the same failure
`reports/head_search_analysis.md` warns about for seeds, at scale. Most of this
task is the guardrails.

## What the agent may change

Four axes, and nothing else:

1. **Which data is used.** Which auxiliary heads and which label files go into
   the training frame — the `union_train.csv` ingest, the AID 1851 columns, the
   three unused files above, head granularity. This is the axis Acellera's
   lesson is about and it is probably the richest one we have.
2. **Models.** New `CYPModel` subclasses appended to `REGISTRY`.
3. **Hyperparameters.** Of anything already registered, and chemprop flags via
   `--extra`.
4. **Representations.** New `Representation` subclasses appended to
   `INPUT_REGISTRY`.

It may **not** change what "held out" means or what the score is. Concretely,
off limits: the pinned `split` column and `data_tools/split_column.py`, the
scorer, `data_tools/load.py`'s split logic, the `test` slice, and the four
scored target columns. Note the asymmetry — the agent chooses the *training*
data freely, and cannot touch the *evaluation* data at all. Enforce this with
container mounts and file permissions, not with a sentence in the prompt.

## How it fits the current repo

- **The score is not the agent's to define.** One command, owned by the
  evaluation-harness task, takes a model name and input list and returns macro
  RMSE over the four scored heads on the pinned split, mean and spread over ≥3
  seeds. The agent calls it and reads the result. If that harness has not
  landed, the first step here is to agree its interface with whoever owns it —
  do not write a second scorer.
- **Experiments are registry entries, not scripts.** `INPUT_REGISTRY` and
  `REGISTRY` are append-only and never reordered. That is the agent's editable
  surface for axes 2 and 4, and it is the same surface RULES.md §2 gives a human.
- **Data choices are CLI flags, mostly.** `head-search` already derives its
  candidates from column names (`discover_candidates`,
  `src/models/head_search.py:57`), so "use this auxiliary head" is a column in
  the frame, not a code change. `build-union` is where a new label file gets
  ingested.
- **Features are cached.** After one `evaluate-models --input all` run,
  `data/features/` holds all six representations (Mordred takes about 20
  minutes the first time), so every later sklearn-scale experiment costs
  seconds. That is what makes a bounded loop practical.
- **Results stay out of git.** The ledger lives under `results/` (gitignored).
  Only the write-up goes in `reports/`.

## Harness: AIDE, MLE-agent, or ours

Two existing frameworks are worth evaluating rather than writing a loop from
scratch. They are good at different halves of this.

**AIDE** (*AI-Driven Exploration in the Space of Code*, Jiang, Schmidt et al.,
Weco AI, arXiv 2502.13138; reference implementation `WecoAI/aideml`) frames ML
engineering as code optimization and runs **trial-and-error as a tree search**
over candidate solutions: draft, debug the most promising node, improve, repeat,
against a user-defined metric to minimize. It is the strongest published option
on MLE-bench — 16.9% of 75 Kaggle competitions medalled at pass@1 with
o1-preview, roughly 4× the next agent framework — and the paper's explanation is
that keeping each node's code short localizes every LLM call to one problem
(debug this script) instead of restarting from scratch on failure.

- **Why it fits.** Its native task shape *is* ours: a data directory, a metric,
  a goal sentence. The tree search is a better search policy than anything we
  would write in a week, and `head-search --strategy greedy` is already a
  hand-rolled, one-axis version of the same idea.
- **Where it rubs.** AIDE writes standalone scripts and expects to own its
  evaluation. That collides with "the agent does not define the score" and with
  our registry structure. The fix is not to fight it: give it the frozen scorer
  as the only way to get a number, and treat its output as *idea generation* —
  a human ports a winning script into a `Representation` or `CYPModel` before
  anything is merged. That also satisfies RULES.md §5, where a human has to be
  able to explain every line in a PR.

**MLE-agent** (`MLSysOps/MLE-agent`) is a pair-programming CLI: autonomous
baseline building, a Kaggle mode (`mle kaggle --auto`), `mle new` / `mle start` /
`mle chat` / `mle report`, debugger-coder interaction loops, and — the
interesting part — **retrieval from arXiv and Papers with Code** to pull in
current methods. Multi-provider (OpenAI, Anthropic, Ollama), so an
OpenAI-compatible base URL is all it needs.

- **Why it fits.** The literature-retrieval step is the "research" half the user
  asked for: propose from published method, not from the model's priors. That is
  the part AIDE does not do.
- **Where it rubs.** It is positioned as a pairing tool rather than an
  unattended optimizer, and the releases visible in search are 0.4.x from late
  2024 with a PyPI warning that the API may change frequently. **Check whether
  it is still maintained before building on it** — if it is stale, lift the idea
  (a retrieval step in the propose phase, with our own `reports/PXR-SUMMARY.md`
  as the first corpus) rather than the dependency.

**Recommendation.** Timebox one day to run AIDE out of the box on the sklearn
path with the frozen scorer, and write down what it finds. That is a real number
for the cost of a day, and it tells us whether a bespoke loop is worth building
at all. Then: AIDE's tree search as the optimizer, an MLE-agent-style retrieval
step in the propose phase, our frozen scorer and ledger around both. Do not
adopt either framework wholesale before that spike.

## The model and the container

**Model: Qwen3-Coder-Next** (`Qwen/Qwen3-Coder-Next`). 80B total parameters with
3B activated per token, hybrid Gated DeltaNet / Gated Attention MoE, 262,144-token
native context, Apache-2.0, non-thinking (no `<think>` blocks). Stated SWE-bench
Verified 70.6, Terminal-Bench 2.0 36.2. It serves through vLLM ≥ 0.15.0
(`--enable-auto-tool-choice --tool-call-parser qwen3_coder`) or SGLang ≥ 0.5.8,
both exposing an OpenAI-compatible API — which is what AIDE and MLE-agent both
want. Recommended sampling: temperature 1.0, top_p 0.95, top_k 40.

**Resolve serving before anything else is built.** 80B weights at bf16 are
~160 GB, and `scripts/head_search.qsub` pins `#$ -l gpu_c=7.0`. Whether that
hardware can serve this model at a usable context length, quantized or not, is
an empirical question with a short answer — get it before designing around the
model. The alternatives are a quantized checkpoint, a larger allocation, or a
hosted endpoint. Also note the topology problem: the agent needs outbound
internet for docs and the model server needs a GPU, and SCC compute nodes should
be assumed to have no outbound network. Plan on the server on a GPU node and the
agent container elsewhere on the cluster network.

**Container.** The agent gets a shell, so it runs in Docker, not on a laptop:

- A `docker/auto-research/Dockerfile` pinning the repo's dependencies, so the
  agent's environment is reproducible and a broken `pip install` cannot take out
  a human's checkout.
- **Mounts, not prompts, define the fence.** `src/models/`,
  `src/representations/`, the two registry files and a scratch dir read-write.
  The scorer, `split_column.py`, the pinned `split` column and the `test` slice
  read-only or absent. Verify with `git status` on the protected paths after a
  run — if the fence only exists in the system prompt, it does not exist.
- **Egress allowlist, not open internet.** The agent is meant to research, so
  allow documentation, arXiv, Papers with Code, PyPI and model-weight downloads.
  Block the challenge: no HuggingFace token, no write path to any submission
  endpoint, no leaderboard. RULES.md §4 — Denali submits, nobody else, and
  leaderboard probing is out of bounds from any account. An automated loop with
  submit access is that rule's worst case.
- `git worktree` on a scratch branch inside the container. It commits locally
  and **never pushes**.
- A hard spending cap on any hosted API key, set on the provider side.

## Scope

### Phase 1 — CPU loop over the sklearn-style path

1. **A ledger.** Append-only JSONL, one row per experiment: the hypothesis in
   one sentence, the exact command, the git commit of the code that ran, which
   data went in, per-isoform and macro RMSE with seed spread, wall time, and
   keep/discard. The agent reads the ledger at the start of every iteration; it
   has no other memory.
2. **The loop.** `auto-research --budget-experiments N --budget-hours H`:
   propose → edit on the scratch branch → run the frozen scorer → log → decide.
   One experiment per iteration, one commit per kept experiment.
3. **A baseline run before any search.** Reproduce a known number from the
   ledger's first row — the decision tree on `rdkit`, say — and stop if it does
   not match. An agent searching on a broken scorer is worse than no agent.
4. **A noise-aware keep rule, plus a gate.** An experiment is kept only if it
   beats the current best by more than the seed spread (~0.004, per the
   manifest). Anything inside the noise is logged as "no difference", not as a
   win. Separately, nothing is *recommended* without passing a gate —
   jjaybird606's loop shipped only gated candidates, and their ungated phase 0
   produced calibration loops and degenerate models.
5. **The divergence detector.** Record the running best `val` score against
   every lockbox check. If `val` keeps improving while the lockbox does not, stop
   and say so in the report. This is jjaybird606's "CV kept improving after the
   leaderboard stopped moving", turned into an alarm. It is the single most
   valuable thing in this brief.
6. **Two-tier evaluation.** A fast sanity pass — one seed, does it run, is the
   number plausible — then the full ≥3-seed scorer only for what survives.
   RyeCatcher's loop was built this way and it is what makes a few hundred
   proposals affordable.
7. **A red-team pass on every kept result.** Before a configuration is
   recommended, a separate step tries to break the claim: did the gain come from
   a seed, from a row count, from a leak, from a changed column? RyeCatcher ran
   adversarial reports against each claim and that is what caught v50. Archive
   failed candidates with their verdicts, not just the winners.
8. **A convergence stop.** Track the correlation of each new candidate's
   validation predictions against the current best. When new candidates stop
   being different — RyeCatcher's threshold was ~0.87 across 27 of them — halt
   the loop and report the ceiling. A loop that cannot stop will spend its whole
   budget rediscovering the same model.
9. **Let it loose on the data axis.** The three unused label files are the
   obvious first target, and the one where Acellera's lesson gets tested
   directly: does more labelled data help here, or does relevance dominate? The
   nuclear-receptor-panel result above says search this axis widely.

### Phase 2 — chemprop on the SCC

10. **Queue, do not run.** The agent writes a `.qsub` from the
   `scripts/head_search.qsub` template and stops. A human submits it; the agent
   picks the result up from `results/` on its next invocation. The loop becomes
   asynchronous, so the ledger needs a `pending` state.
11. **Search the training recipe**, not just head subsets: loss, foundation
   init, depth, hidden size, dropout. `head-search` already owns head subsets —
   call it, do not reimplement it. Coordinate with R1, which is measuring the
   same axes by hand.

### Reporting

12. **`reports/auto_research.md`.** How many experiments ran, how many were
   kept, the top configurations with seed spread, and — the important line —
   their lockbox `test` score next to their `val` score, with the number of
   times the lockbox was consulted. The gap between those two is the
   measurement of how much the agent overfit.

## What not to do

- **Do not let it define or touch the score.** Not the scorer, not the split,
  not the `test` slice, not the target columns.
- **Do not report the best-of-N validation score as a result.** Report the
  lockbox score, and say how many times the lockbox was looked at.
- **Do not let it push.** A human reads the kept experiments and opens the PR,
  and under RULES.md §5 must be able to explain every one of them. "The agent
  found it" is not an answer.
- **Do not let it tune by re-splitting or re-seeding** until something wins. The
  split is pinned and the seed list is fixed up front.
- **Do not start with chemprop.** A GPU experiment takes long enough that
  mistakes in the loop cost days. Get the ledger, the fence and the keep rule
  right on experiments that take seconds.
- **Do not let it add data without an overlap check.** `check-overlap` against
  `data/test.csv` on anything new, every time.
- **Do not reproduce jjaybird606's submit loop.** Their 2,500 cycles used
  leaderboard feedback as the objective. RULES.md §4 puts that out of bounds for
  us from any account, so the loop is structurally different and the lockbox
  discipline above is the substitute.
- **Do not let it add post-hoc calibration** without measuring on the lockbox.
  It worsened held-out RAE for jjaybird606 and inflated apparent gains by ~0.009
  RAE per step elsewhere in the survey.

## Tools

- **AIDE** (`WecoAI/aideml`) — tree-search optimizer. Evaluate first.
- **MLE-agent** (`MLSysOps/MLE-agent`) — retrieval-augmented proposal; check
  maintenance status before depending on it.
- **vLLM ≥ 0.15.0** or **SGLang ≥ 0.5.8** serving `Qwen/Qwen3-Coder-Next` over
  an OpenAI-compatible API. Check the current serving docs rather than relying
  on these flags surviving a release.
- **Docker**, `git worktree`.
- **DSPy** is worth a look for the gate and the proposal step specifically,
  since it is what jjaybird606 used to guardrail a loop that survived 2,500
  cycles.
- `evaluate-models`, `head-search`, `build-union`, `check-overlap` (today) and
  the evaluation-harness command (once it exists) as the only scorers.

## Done when

- `auto-research` runs unattended for a fixed budget in the container and
  leaves a complete ledger, with no writes outside its fenced area — verified by
  `git status` on the protected paths after a run.
- The baseline-reproduction check passes and is part of every run.
- The divergence detector is implemented and its curve is in the report, even if
  it never fired.
- The gate, the red-team pass and the convergence stop all exist as code, and
  the report says whether the loop stopped on budget or on convergence.
- A documented ceiling, if one is reached. RyeCatcher argues the system and the
  ceiling were the real deliverables, and on our noisier signal that is more
  likely to be the outcome than a new best model.
- The AIDE spike is written up, with its number, whether or not we keep it.
- At least one full phase-1 run is in `reports/auto_research.md` with val
  scores, lockbox scores, and the gap between them.
- An explicit verdict: did the agent find anything a person following T2, T3 and
  R1 would not have? **A null result is a legitimate deliverable** — "an
  unattended search over this space finds nothing beyond the obvious" tells us
  where not to spend time.
- Phase 2 is optional for the round. If attempted, at least one agent-written
  chemprop job has gone through the SCC queue and back into the ledger.

## SCC note

The agent's own phase-1 experiments do not need it. Serving the model does.
Compute nodes should be assumed to have no outbound network, so the agent
container runs where it can reach the internet and only training jobs go to the
queue. **Contact Denali with your code** to get jobs submitted — the agent does
not get its own access to the allocation, and Denali stands up the container and
the model server before the loop runs unattended.

## Sources

**Agents in this competition family.**

jjaybird606 — **rank 3, RAE 0.5076** (checkpoint `cycle0766`; 9th on the combined
intermediate board at MAE 0.426 against the leader's 0.400), with agents doing the
modelling and the human supplying infrastructure and curated data. Headless Claude
Code via `claude remote-control` on cloud GPUs, a DSPy-guardrailed autonomous loop
of ~2,500 gated train-and-submit cycles at a 4-hour limit, plus a Karpathy-style
"autoresearch" daemon. Seven-family uniform blend, 10-fold CV plus held-out Set 1,
greedy ensemble selection, final retrain on ~43,800 rows. Dead ends: calibration
loops, out-of-fold overfitting (CV improving after the leaderboard stalled),
unbounded cost, post-hoc calibration, Set-1→Set-2 transfer, redundant
structure-based scoring (docking, Boltz-2), redundant AutoGluon over
scikit-fingerprints and Mordred. The prompts are not published — only the loop
structure and tooling:
https://github.com/jjaybird606-blip/pxr

RyeCatcher / Justin Johnson — **rank ~40 of 211, top 19%**, autonomous loop driven
largely by Claude, author with no prior ADMET background, out of the loop for
about a week. One gate function on an honest score, two-tier evaluation, archived
failed candidates with verdicts, phase checkpoints, red-team reports attacking
each claim. v50 looked like a breakthrough because its calibration saw the test
answers (~40th → ~87th if shipped). Stopped on convergence after 27 candidates
correlated > 0.87. Apache-2.0 code, weights and methodology report:
https://rundatarun.io/p/pulling-threads ·
https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026
**This is the same RyeCatcher cited throughout `reports/PXR-SUMMARY.md`**, where
the agent-driven provenance is not recorded. The survey labels the entry rank 67
while this post says ~40 of 211 — worth reconciling, since the rank-67 label is
used across several briefs.

Acellera, "Reaching the top tier of the
PXR blind challenge with an AI agent black-box trainer": an agent sequencing
featurization, training, tuning and inference, with the scientific team keeping
the dataset, evaluation design and interpretation; and the finding that adding
labelled compounds did not automatically improve transfer —
https://www.acellera.com/blog/reaching-the-top-tier-of-the-pxr-blind-challenge-with-an-ai-agent-black-box-trainer/
The post does not name the LLM or framework, state a rank, or describe the
agent's permissions, so treat the architectural detail above as the limit of
what it supports.

**Agent frameworks.**

- AIDE — Jiang, Schmidt et al., *AI-Driven Exploration in the Space of Code*,
  arXiv 2502.13138: https://arxiv.org/abs/2502.13138 ·
  https://github.com/WecoAI/aideml
- MLE-agent — https://github.com/MLSysOps/MLE-agent
- MLE-bench (Chan et al., 2024), the benchmark AIDE's medal rate is measured on:
  https://arxiv.org/abs/2410.07095
- Qwen3-Coder-Next model card:
  https://huggingface.co/Qwen/Qwen3-Coder-Next

**Why the guardrails look like this.** Beyond the two agent write-ups above, the
evidence below is about the failure modes an automated search would mechanize. Findings are quoted from
`reports/PXR-SUMMARY.md`:

- **Automated search over many candidates has precedent.** auP7s (rank 27) ran
  NSGA-II Pareto optimization over (OOF MAE, Spearman ρ) to select ensemble
  subsets (§6). RyeCatcher (rank 67) cascaded through seven intermediate
  ensemble versions (§6). Both are hand-driven searches over a large space —
  the same shape of work, without the loop.
- **Mass experimentation needs survivor filters.** ldbc1999 (rank 65) trained
  147 models and filtered to 5 using quality (R² > 0.3 held out), anti-leakage
  (`heldout_RAE / OOF_RAE > 2.0`) and correlation deduplication (§6). An agent
  generating hundreds of configurations inherits that problem exactly, and the
  anti-leakage ratio is the measurement that catches a search which has started
  fitting its own validation set.
- **Honest validation is the central difficulty.** §5: random K-fold severely
  overestimated leaderboard performance in every report that tried it, and
  RyeCatcher's per-fold protocol was the only one whose CV→LB shifts matched
  outcomes. Hence `val` for the agent, `test` for a human.
- **Fitting a calibration step on data the model saw inflates the apparent
  gain** by ~0.009 RAE per step, and it compounds (§5, §7, RyeCatcher). The same
  arithmetic applies to best-of-N selection.
- **Optimizing against the wrong signal destroys the result.** The distillation
  backfire (§6, discoverybytes, ΔRAE +0.0126) and nearest-neighbour activity
  transfer (§7, RyeCatcher, OOF 0.4536 → LB 0.658, rank ~40 → ~87) are both
  cases where a step that improved the local objective was badly wrong on the
  blinded set. An unsupervised loop optimizing a single number is the mechanized
  version of that risk.
- Our own `reports/head_search_analysis.md` supplies the seed-noise floor
  (~0.004) the keep rule uses.

Primary write-ups cited above:

- auP7s (rank 27): https://gist.github.com/chemotica/a49b002eda2f7fd5eef2dca4f98f8ad7
- ldbc1999 (rank 65): https://github.com/lizyurkewych-git/pxr-challenge
- RyeCatcher (rank 67): https://huggingface.co/RyeCatcher/openadmet-pxr-challenge-2026
- discoverybytes (rank 11):
  https://github.com/discoverybytes/openadmet-pxr-blind-challenge/tree/main/activity-prediction
