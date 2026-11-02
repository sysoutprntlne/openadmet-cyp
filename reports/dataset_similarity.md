# Pretraining candidates: similarity to the challenge train and test sets

Generated 2026-10-19 by `dataset-similarity --extra pubchem_aid1851=data/pubchem_aid1851.csv`
(`src/data_tools/dataset_similarity.py`). Raw numbers: `results/dataset_similarity.csv`.

Datasets come from `download-data` (`data/metabolism/`). `pubchem_aid1851` is the auxiliary
source already used as heads, included as a reference point.

## Method

Morgan fingerprints (radius 2, 2048 bits), Tanimoto similarity. Molecules are reduced to their
largest fragment and deduplicated on the InChIKey connectivity block, so counts are unique
skeletons (train 4,901, test 748).

- `->ref NN`: mean nearest-neighbour similarity from each dataset molecule to the reference set.
  High means the dataset contains the kind of molecules the reference contains.
- `ref-> NN`: mean nearest-neighbour similarity from each reference molecule to the dataset.
  High means the dataset **covers** the reference chemistry. This is the direction that matters
  for pretraining.
- `>=0.4`, `>=0.7`: fraction of molecules whose nearest neighbour is at least that similar.
- `shared`: dataset skeletons also present in the reference set.
- The `train` and `test` rows compare each set to itself leave-one-out (a molecule never matches itself).

## Table

| dataset | n | ->train NN | ->train >=0.4 | ->train >=0.7 | shared train | train-> NN | train->  >=0.4 | ->test NN | ->test >=0.4 | ->test >=0.7 | shared test | test-> NN | test->  >=0.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 4901 | 0.463 | 0.736 | 0.039 | 0 | - | - | 0.314 | 0.120 | 0.009 | 0 | 0.598 | 0.999 |
| test | 748 | 0.598 | 0.999 | 0.103 | 0 | 0.314 | 0.120 | 0.663 | 0.989 | 0.345 | 0 | - | - |
| expansionrx | 4969 | 0.338 | 0.142 | 0.000 | 0 | 0.281 | 0.028 | 0.270 | 0.028 | 0.000 | 0 | 0.297 | 0.044 |
| tdc_clearance_microsome_az | 1088 | 0.353 | 0.174 | 0.028 | 30 | 0.292 | 0.047 | 0.274 | 0.034 | 0.001 | 0 | 0.310 | 0.107 |
| tdc_cyp2c9_substrate | 665 | 0.491 | 0.445 | 0.226 | 130 | 0.287 | 0.058 | 0.251 | 0.047 | 0.005 | 0 | 0.275 | 0.035 |
| tdc_cyp2d6_substrate | 663 | 0.491 | 0.446 | 0.226 | 129 | 0.287 | 0.058 | 0.252 | 0.048 | 0.005 | 0 | 0.276 | 0.036 |
| tdc_cyp3a4_substrate | 666 | 0.490 | 0.443 | 0.224 | 130 | 0.287 | 0.058 | 0.251 | 0.048 | 0.005 | 0 | 0.276 | 0.036 |
| tdc_half_life_obach | 658 | 0.445 | 0.389 | 0.170 | 99 | 0.277 | 0.048 | 0.239 | 0.036 | 0.002 | 0 | 0.274 | 0.043 |
| tdc_hlm | 5871 | 0.317 | 0.106 | 0.007 | 20 | 0.318 | 0.081 | 0.260 | 0.034 | 0.000 | 0 | 0.335 | 0.148 |
| tdc_rlm | 5527 | 0.335 | 0.171 | 0.008 | 32 | 0.331 | 0.115 | 0.273 | 0.035 | 0.000 | 0 | 0.342 | 0.166 |
| pubchem_aid1851 | 16139 | 0.342 | 0.180 | 0.015 | 173 | 0.385 | 0.274 | 0.273 | 0.041 | 0.000 | 1 | 0.390 | 0.362 |

## Reading it

- **Test sits close to train** (test->train NN 0.598, 99.9% at >=0.4). Train to itself is 0.463,
  so test is an interpolation problem, not a scaffold shift.
- **Every metabolism set is chemically far from the challenge.** Typical nearest neighbours are
  0.27 to 0.34.
- **TDC RLM and HLM cover the challenge best** among the metabolism sets: 11.5% and 8.1% of train
  molecules, and 16.6% and 14.8% of test molecules, have a neighbour at >=0.4. ExpansionRx covers
  least (2.8% of train) despite being the largest regression CLint set.
- **The substrate sets and half-life are small drug lists.** They share molecules with train
  (about 130 and 99) but cover almost none of its chemistry. The three substrate sets are
  near-identical in every column.
- **PubChem AID 1851 covers better than every metabolism set** (27.4% of train, 36.2% of test).
  That is the source the head search found useful, so the stability sets are weaker candidates
  than data the joint model already uses.
- **No dataset shares a skeleton with test**, except one PubChem molecule, which `build-union` and
  `train-chemprop --datasets` already drop.

## Caveats

- Similarity only. High overlap does not guarantee transfer.
- TDC HLM and RLM labels are binary stable/unstable, not CLint.
- One fingerprint type (Morgan r2); other fingerprints would shift the absolute numbers.
- Row counts are unique skeletons after deduplication, so they differ slightly from the raw files.
