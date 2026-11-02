"""Filter and cluster candidate predictions, fit a stacker on OOF
predictions, and apply it to validation predictions.
"""

from collections.abc import Callable

import polars as pl

from models.stack import (
    fit_elasticnetcv_stack,
    fit_nnls_stack,
    fit_ridgecv_stack,
    predict_stack,
)
from models.validation import (
    deduplication_filter,
    heldout_rae_filter,
    r_squared_filtering,
)

def fit_filtered_stack(
    validation_predictions: pl.DataFrame,
    validation_truth: pl.DataFrame,
    oof_predictions: pl.DataFrame,
    oof_truth: pl.DataFrame,
    method: str = "nnls",
    *,
    on_filter_decisions: Callable[[pl.DataFrame], None] | None = None,
) -> dict:
    """Inputs are already seed-averaged, use matching model IDs across populations.
    
    The function returns filter decisions, membership, clustered predictions, and 
    fitted stack results. The optional callback receives decisions before
    checking for empty targets, so callers can save rejected-run diagnostics.
    """

    stackers = {
        "nnls": fit_nnls_stack, 
        "ridgecv": fit_ridgecv_stack,
        "elasticnetcv": fit_elasticnetcv_stack,
    }

    if method not in stackers:
        raise ValueError(f"Unknown stacking method: {method}")

    quality_filter = r_squared_filtering(validation_predictions, validation_truth)
    leakage_filter = heldout_rae_filter(
        validation_predictions,
        validation_truth,
        oof_predictions,
        oof_truth,
    )

    keys = ["model_id", "target"]

    decisions = quality_filter.rename({"keep": "quality_keep"}).join(
        leakage_filter.rename({"keep": "leakage_keep"}),
        on=keys,
        validate="1:1",
    ).with_columns(
        (pl.col("quality_keep") & pl.col("leakage_keep")).alias("keep")
    )

    if on_filter_decisions is not None:
        on_filter_decisions(decisions)

    survivors = decisions.filter(pl.col("keep")).select(keys)

    filtered_validation = validation_predictions.join(
        survivors, on=keys, how="semi"
    )
    filtered_oof = oof_predictions.join(
        survivors, on=keys, how="semi"
    )

    targets = validation_truth["target"].unique().to_list()
    for target in targets:
        if survivors.filter(pl.col("target") == target).is_empty():
            raise ValueError(f"No surviving models for {target}")

    clustered_validation, clustered_oof, membership = deduplication_filter(
        filtered_validation, 
        validation_truth,
        filtered_oof, 
        oof_truth,
    )
    
    stack_results = stackers[method](clustered_oof, oof_truth)

    ensemble_validation = predict_stack(clustered_validation, stack_results)

    return {
        "stack": stack_results,
        "filter_decisions": decisions,
        "cluster_membership": membership,
        "validation_predictions": clustered_validation,
        "ensemble_validation_predictions": ensemble_validation,
        "oof_predictions": clustered_oof,
    }
