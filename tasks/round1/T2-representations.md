# T2 — Representation portfolio

**Depends on:** nothing.
**GPU:** no. CPU-only, laptop-friendly.
**Owns:** `src/representations/morgan.py`, `maccs.py`, `mordred.py` (all new);
one line each in `src/data_tools/inputs.py`.
**Status:** done. `morgan`, `count_morgan`, `maccs` and `mordred` are registered;
results are in `reports/t2_solo_representations.md`. The text below is the
original brief and describes the repo as it was before this task landed.

---

## Why this matters for the challenge

`INPUT_REGISTRY` currently contains two entries: `rdkit` (217 2D descriptors)
and `chemeleon` (2048-d pretrained embedding).

**Morgan/ECFP4 does not exist as a feature anywhere in this repo.** It appears
only inside `_butina_split` and `split_column._butina_groups`, where it is used
to compute clustering distances and then thrown away.

`reports/PXR-SUMMARY.md` §2 opens by calling ECFP4 "the universal starting
fingerprint — every submission uses it." Not having it is a conspicuous hole,
and more importantly it means we have never measured how much of our CheMeleon
gain is real versus what a cheap fingerprint would have given us. Without that
baseline we cannot tell whether the foundation model is earning its cost.

The survey's structural finding is that competitive entries combine at least
two *orthogonal* representation tracks, and that the orthogonality is what pays:

- ldbc1999 found MACCS and ECFP4 identify **~90% non-overlapping activity cliff
  pairs** — the combination is genuinely richer than either alone, not
  redundant.
- Mordred extends descriptor coverage but carries real overfitting risk;
  discoverybytes measured OOF RAE 0.574 against test RAE 0.67, a gap that
  indicates scaffold overfitting. Worth having, worth distrusting.
- The negative result is equally useful: RyeCatcher tested five additional 2D
  neural representations and found they all converged to Spearman > 0.88
  against their primary ensemble. **The 2D graph space saturates.** So this task
  is about covering the cheap orthogonal axes properly and then stopping — not
  about stacking more neural encoders. That is what T5/T6 (3D) is for.

This task is also a hard dependency in practice for T3 and T4: tree models and
TabPFN need feature matrices, and right now there are two to choose from.

## How it fits the current repo

Clean plug-in point, minimal blast radius.

- `src/representations/base.py` defines the `Representation` interface. Subclass
  it, set a unique `name` class variable, implement `transform(smiles) -> np.ndarray`.
- Follow `src/representations/rdkit_descriptors.py` as the template — it is 37
  lines and shows the expected shape, including the NaN-row convention for
  unparsable SMILES.
- Register in `src/data_tools/inputs.py` by appending to `INPUT_REGISTRY`.
  **One line per featurizer, appended, never reordering the dict** — T6 also
  touches this file.
- **Do not build your own caching.** `inputs.py::_featurize_one` already does
  content-addressed caching keyed on SMILES, writing
  `data/features/{name}.npy` + `.idx`, and correctly reuses the cache across
  split types and seeds because a molecule's representation does not depend on
  how the data was split. You get this for free by registering.
- `featurize()` hstacks multiple named inputs, so combinations
  (`--input morgan maccs`) work immediately with no extra code.
- `evaluate-models` and `generate-results` both take `--input`, so every new
  featurizer is benchmarkable the moment it is registered.

## Scope

1. **Morgan/ECFP4** — 2048-bit, radius 2. Use `GetMorganGenerator`, matching
   the parameters already used in `load.py` so the feature and the split
   clustering agree.
2. **Count-based Morgan** — same generator, `GetCountFingerprint`. Preserves
   repeated-substructure frequency, which the binary version discards.
3. **MACCS** — 167 structural keys, `rdkit.Chem.MACCSkeys`.
4. **Mordred** — the wide descriptor set. Gate it: report train/val gap
   explicitly, and flag it as suspect if you see the survey's overfitting
   signature. Adds a dependency, so justify it with numbers.
5. Benchmark each alone and in combination against the existing `rdkit` and
   `chemeleon` baselines via `evaluate-models`.
6. **Report the pairwise correlation between representations**, not just their
   individual scores. The survey's saturation finding means a new featurizer
   that correlates > 0.9 with CheMeleon is not worth its compute — say so in
   the writeup rather than quietly adding it.

## Tools

- `rdkit` — already a dependency. `rdFingerprintGenerator`, `MACCSkeys`.
- `mordred` or `mordredcommunity` — new dependency, add to `requirements.txt`
  only if step 4 justifies it. The original `mordred` is unmaintained against
  recent numpy; `mordredcommunity` is the live fork.
- `numpy`, `polars`, `tqdm` — already in use, follow existing style.
- `evaluate-models --input <name>` for benchmarking.

## Done when

- All featurizers registered, cached, and benchmarked through `evaluate-models`
  on the pinned split.
- A short table in `reports/` giving each representation's solo performance,
  its best pairing, and its correlation with CheMeleon.
- An explicit recommendation on Mordred: keep or drop, with the train/val gap
  as the evidence.

## SCC note

Not needed. This is CPU featurization and sklearn-scale evaluation.

One caveat: Mordred over ~21k union molecules is slow and single-threaded by
default. If it becomes the bottleneck, parallelize locally before reaching for
the cluster. If you do end up wanting a batch node for a one-off featurization
sweep, contact Denali.
