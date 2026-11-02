# T1 — Wire the unused challenge assays into the union

**Depends on:** nothing. Start here.
**GPU:** yes, moderate — every evaluation is a multi-seed chemprop run. See SCC note.
**Owns:** `src/data_tools/assay_heads.py` (new), the argument surface of `src/data_tools/build_union.py`.

---

## Why this matters for the challenge

The challenge shipped four training files. We use one.

`data/train.csv` (direct inhibition) is in the union. `data/tdi_train.csv`,
`data/emax_train.csv` and `data/single_concentration_train.csv` are downloaded,
have loaders written for them in `src/data_tools/load.py`, and reach nothing.
`data/union_train.csv` contains exactly 14 heads: the 4 scored columns plus 10
from PubChem AID 1851.

This is free signal on measurements of *the same molecules on the same enzymes*.
Every other auxiliary source we use is external chemistry that needs
cross-source reconciliation; these three do not — they are the challenge's own
deck, same compounds, same assay platform, different readout.

`reports/PXR-SUMMARY.md` §4 is entirely about this move and the top-decile PXR
competitors all made it:

- **Single-concentration → pseudo-labels.** The standard play was Hill-fitting
  the dose points into pseudo-pEC50 and pretraining on them (Schnappi, mp-alex,
  ldbc1999). The better variant, ldbc1999's Sub 6, kept *all* concentration
  points as separate rows with `log10[concentration]` as a molecule-level
  descriptor — roughly 4× the rows of Hill-fitting, and it does not discard the
  borderline compounds that fail an R² filter.
- **Emax as a joint target.** ldbc1999's Sub 8 predicted potency and Emax
  together, on the argument that partial agonists carry SAR information that
  potency alone misses.
- **A specificity/artifact channel.** In PXR this was the counter-assay; the
  CYP analogue is the TDI-vs-direct contrast, which separates
  mechanism-based inactivation from reversible inhibition.

Our own results say multitask transfer works on this dataset: the best aux
configuration moves macro RMSE from 0.9430 to 0.7184. These three assays are
the most obviously relevant heads we have not yet tried.

## How it fits the current repo

The machinery already exists and is the repo's central design bet. From the
README: external measurements are never pooled into the scored pIC50 columns;
each source × isoform × readout becomes its own column and therefore its own
head, so no cross-assay calibration is needed. Chemprop masks missing targets
in the loss, so a sparse block costs nothing.

So this task is *not* building a new pipeline. It is producing correctly shaped
columns and letting the existing pipeline consume them.

- `src/data_tools/load.py` already has `load_tdi_train`, `load_emax_train`,
  `load_single_concentration_train`. Use them.
- `src/data_tools/build_union.py` takes auxiliary sources as `NAME=PATH` pairs
  (`--aux-sources`, default `pubchem=data/pubchem_aid1851.csv`). Your output
  should be CSVs that slot straight into that flag.
- Follow the `src/data_tools/pubchem.py` precedent for pivoting a long assay
  table into per-head columns, and `src/data_tools/family_heads.py` for the
  naming convention `{gene}_{readout}_{source}`.
- `head-search --strategy single` then ranks each new block against baseline
  without you writing any evaluation code.

**Watch the row shape.** `tdi_train.csv` and `emax_train.csv` are ~6,145 rows
and already one-row-per-molecule — a straightforward pivot. But
`single_concentration_train.csv` is ~17,500 long-format rows with `enzyme`,
`concentration_M`, `log2fc_estimate` and a per-row standard error, so it needs
a real decision about shape, not a pivot.

## Scope

1. **TDI heads.** `CYP{iso}_pIC50_TDI_condition` as four heads, plus the
   `is_TDI` booleans. Note `tdi_train.csv` also carries its own copy of the
   direct-inhibition columns — do not let those collide with the scored
   columns from `train.csv`. They must either be dropped or verified identical.
2. **Emax heads.** Both `EmaxVsPosCtrl_direct_inhibition` and
   `EmaxVsPosCtrl_TDI_condition`, four isoforms each.
3. **Single-concentration, two variants, benchmarked against each other:**
   - *(a)* Hill-fit per compound × enzyme → pseudo-pIC50, R² ≥ 0.5 filter.
   - *(b)* Keep all concentration points as rows, pass
     `log10(concentration_M)` as a molecule-level descriptor to the chemprop
     FFN head via `--descriptors-path`. This is ldbc1999's variant and the one
     the survey reports as stronger.

   Variant (b) changes the row cardinality of the union, so build it as a
   separate output file rather than mutating the main union.
4. Run `head-search --strategy single` over every new block, then a greedy pass
   including them alongside the existing aux candidates.

## Tools

- `polars` for all table work — the repo is polars throughout, not pandas.
- `scipy.optimize.curve_fit` for the four-parameter Hill fit in variant (a).
  Record the fitted R² per compound; it is the filter and it is also a
  reasonable per-row weight for T7.
- `data_tools.standardize` for keys. `check-overlap` before you ship anything.
- `head-search`, `build-union`, `add-split-column` — existing entry points.

## Done when

- The new heads are in a union table and `head-search --strategy single` has
  scored each block against the 0.9430 baseline over ≥3 seeds.
- Single-concentration variants (a) and (b) have been compared head to head.
- Results appended to `reports/head_search_analysis.md` in the existing table
  format, with the seed spread reported.

## SCC note

**Contact Denali with your code to run this on the BU SCC.**

Building the tables is CPU work you can do locally, but every evaluation is a
chemprop training run and the greedy search multiplies that by configs × seeds.
Reuse `scripts/head_search.qsub`; keep `#$ -l gpu_c=7.0` (older SCC GPUs fail
with `cudaErrorNoKernelImageForDevice`), and pad `h_rt` — the existing job asks
for 48 hours.
