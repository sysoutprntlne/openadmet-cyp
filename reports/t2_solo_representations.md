# T2 — Representation portfolio evaluation

## Summary

This report compares all six registered representations alone and in all 15 two-input combinations. It reports decision-tree validation results for the four scored CYP isoforms, plus a pairwise Spearman correlation matrix. **Count-Morgan + RDKit** has the lowest reported macro RMSE (0.8848). **Mordred** has the lowest solo macro RMSE (0.9202).

These are descriptive results from one random split at seed 42. The workbook does not document a saved pinned split, repeated model seeds, training scores, or the underlying values used for the correlations. They therefore do not establish a stable improvement under the project's shared evaluation contract.

## Source and metric

The source is the supplied `CompBio T2-representations.xlsx` workbook (done by Tony): `Ind. runs` (solo metrics), `Combinations` and `Summary` (two-input metrics), and `Pairwise Spearman Correlation` (correlations). The workbook's chart captions identify a decision tree, random split, and seed 42. No prior locally generated results are used in this report.

The primary metric is **macro RMSE**, the unweighted average of validation RMSE for CYP1A2, CYP2C9, CYP2D6, and CYP3A4 direct inhibition. Lower is better. The workbook also records MAE and R² by isoform; the tables below focus on the Task 2 comparison metric. A positive change in macro RMSE below means the pair scored lower than the stated solo comparator on this one run.

## Solo performance and best pairing

| Representation | Solo macro RMSE | Best partner | Pair macro RMSE | Solo minus pair | Mean Spearman ρ with CheMeleon |
|---|---:|---|---:|---:|---:|
| Morgan | 0.9868 | RDKit | 0.8930 | +0.0938 | 0.2405 |
| Count-Morgan | 0.9354 | RDKit | **0.8848** | +0.0506 | 0.4278 |
| MACCS | 0.9245 | RDKit | 0.9076 | +0.0169 | 0.3328 |
| Mordred | **0.9202** | Morgan | 0.8991 | +0.0211 | 0.4871 |
| RDKit | 0.9275 | Count-Morgan | **0.8848** | +0.0427 | 0.3955 |
| CheMeleon | 0.9901 | Mordred | 0.9274 | +0.0627 | — |

The pairing is the lowest macro RMSE among the five pairs containing that representation. “Solo minus pair” uses that representation's solo score, so it is not a comparison with the stronger member of the pair. The full pair table below makes that comparison explicit.

### Solo validation RMSE by isoform

| Representation | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | Macro RMSE |
|---|---:|---:|---:|---:|---:|
| Morgan | 1.1281 | 0.7687 | 0.9876 | 1.0628 | 0.9868 |
| Count-Morgan | 1.1095 | 0.7323 | 0.9486 | 0.9513 | 0.9354 |
| MACCS | 1.0849 | 0.7442 | **0.8717** | 0.9970 | 0.9245 |
| Mordred | 1.1754 | 0.7099 | 0.9749 | **0.8205** | **0.9202** |
| RDKit | 1.1270 | **0.7048** | 1.0357 | 0.8426 | 0.9275 |
| CheMeleon | 1.2283 | 0.8122 | 0.9470 | 0.9730 | 0.9901 |

No solo representation leads on every isoform. MACCS has the lowest CYP1A2 and CYP2D6 RMSE, RDKit the lowest CYP2C9 RMSE, and Mordred the lowest CYP3A4 RMSE. All values above are transcribed from `Ind. runs`; the macro values agree with the four displayed target RMSEs to workbook precision.

## Two-input performance

Each pair concatenates two representations. The change column compares its macro RMSE with the **better solo member** of that pair: positive means the pair scored lower in the workbook's run.

| Pair | Macro RMSE | Change vs better solo |
|---|---:|---:|
| Count-Morgan + RDKit | **0.8848** | +0.0427 |
| Morgan + RDKit | 0.8930 | +0.0345 |
| Morgan + Mordred | 0.8991 | +0.0211 |
| Mordred + RDKit | 0.9048 | +0.0154 |
| MACCS + RDKit | 0.9076 | +0.0169 |
| Count-Morgan + Mordred | 0.9104 | +0.0098 |
| MACCS + Mordred | 0.9226 | −0.0024 |
| Mordred + CheMeleon | 0.9274 | −0.0072 |
| Morgan + Count-Morgan | 0.9303 | +0.0051 |
| Count-Morgan + MACCS | 0.9307 | −0.0062 |
| RDKit + CheMeleon | 0.9515 | −0.0240 |
| Morgan + MACCS | 0.9591 | −0.0346 |
| Morgan + CheMeleon | 0.9816 | +0.0052 |
| MACCS + CheMeleon | 0.9844 | −0.0599 |
| Count-Morgan + CheMeleon | 0.9849 | −0.0495 |

Count-Morgan + RDKit scores 0.0354 lower than the best *solo across all six* (Mordred, 0.9202). Eight of the 15 pairs score lower than their better solo member; seven score higher. In particular, adding CheMeleon to Count-Morgan, MACCS, or RDKit raises macro RMSE in this run. A pair's benefit should be judged from measured performance, not from its feature count alone.

## Pairwise representation correlation

The workbook labels these values “Mean Pairwise Spearman Correlation Across CYP Isoforms.” It does not specify the underlying vectors or how the four isoform correlations were aggregated, so this report preserves the workbook's label without interpreting them as raw-feature or prediction correlations.

| ρ | Morgan | Count-Morgan | MACCS | Mordred | RDKit | CheMeleon |
|---|---:|---:|---:|---:|---:|---:|
| Morgan | 1.0000 | 0.4355 | 0.2767 | 0.2261 | 0.2522 | 0.2405 |
| Count-Morgan | 0.4355 | 1.0000 | 0.3539 | 0.4368 | 0.4094 | 0.4278 |
| MACCS | 0.2767 | 0.3539 | 1.0000 | 0.3716 | 0.3367 | 0.3328 |
| Mordred | 0.2261 | 0.4368 | 0.3716 | 1.0000 | **0.5873** | 0.4871 |
| RDKit | 0.2522 | 0.4094 | 0.3367 | **0.5873** | 1.0000 | 0.3955 |
| CheMeleon | 0.2405 | 0.4278 | 0.3328 | 0.4871 | 0.3955 | 1.0000 |

Every non-identical pair is below the Task 2 brief's 0.9 saturation threshold. Correlation with CheMeleon ranges from 0.2405 (Morgan) to 0.4871 (Mordred). Low correlation alone does not ensure a useful combination: Count-Morgan + CheMeleon has ρ = 0.4278, yet its macro RMSE (0.9849) is higher than Count-Morgan alone (0.9354) in this run.

## Mordred recommendation and remaining checks

**Keep Mordred as a candidate representation for now.** It has the best solo macro RMSE in the supplied results and participates in the third-ranked pair, Morgan + Mordred (0.8991). The workbook contains validation metrics but no training metrics, so the train/validation gap required by the Task 2 brief **cannot be calculated**. These results therefore cannot rule out the specified overfitting risk or support a final keep/drop decision. Measuring that gap is the next required check for Mordred.

The workbook records only one random-split seed. The project rules call for a pinned split and at least three seeds with spread before claiming an improvement. The exact executed commands, saved split assignment, target sample counts, correlation inputs, and training scores are not included in the workbook. The tables document the supplied results; they should not be compared directly with the Chemprop auxiliary-head reference numbers in `tasks/round1/MANIFEST.md`.
