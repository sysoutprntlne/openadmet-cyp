"""Validate and seed-average OOF predictions, fit per-target NNLS, RidgeCV,
or ElasticNetCV stackers, and apply fitted weights to aligned predictions.
"""

import numpy as np
import polars as pl
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import PredefinedSplit


_SCORED_DEFAULT = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
]


def _validate_inputs(predictions: pl.DataFrame, truth: pl.DataFrame) -> None:
    """Raise ValueError when OOF predictions and truth cannot be safely aligned."""

    # 1. Required headers
    prediction_required = {"model_id", "inchikey_full", "fold", "target", "pred"}
    prediction_missing = prediction_required - set(predictions.columns)
    if prediction_missing:
        raise ValueError(f"predictions is missing required columns: {sorted(prediction_missing)}")
    
    truth_required = {"inchikey_full", "fold", "target", "truth"}
    truth_missing = truth_required - set(truth.columns)
    if truth_missing:
        raise ValueError(f"truth is missing required columns: {sorted(truth_missing)}")

    # 2. Empty tables, nulls, identifier validity, and numeric validity
    if len(predictions) == 0:
        raise ValueError("predictions table is empty")
    if len(truth) == 0:
        raise ValueError("truth table is empty")

    prediction_null_columns = [
        c for c in prediction_required
        if predictions[c].null_count() > 0
    ]
    if prediction_null_columns:
        raise ValueError(f"There are null values in predictions: {sorted(prediction_null_columns)}")
    
    truth_null_columns = [
        c for c in truth_required
        if truth[c].null_count() > 0
    ]
    if truth_null_columns:
        raise ValueError(f"There are null values in truth: {sorted(truth_null_columns)}")

    pred_dtype = predictions["pred"].dtype
    if not (pred_dtype.is_integer() or pred_dtype.is_float()):
        raise ValueError("pred must contain integers or floats")
    truth_dtype = truth["truth"].dtype
    if not (truth_dtype.is_integer() or truth_dtype.is_float()):
        raise ValueError("truth must contain integers or floats")

    if predictions["model_id"].dtype != pl.String:
        raise ValueError("predictions.model_id must contain strings")
    if predictions["inchikey_full"].dtype != pl.String:
        raise ValueError("predictions.inchikey_full must contain strings")
    if truth["inchikey_full"].dtype != pl.String:
        raise ValueError("truth.inchikey_full must contain strings")
    if predictions["target"].dtype != pl.String:
        raise ValueError("predictions.target must contain strings")
    if truth["target"].dtype != pl.String:
        raise ValueError("truth.target must contain strings")
    
    if not predictions["fold"].dtype.is_integer():
        raise ValueError("predictions.fold must contain integers")
    if not truth["fold"].dtype.is_integer():
        raise ValueError("truth.fold must contain integers")
    
    if (predictions["model_id"].str.strip_chars() == "").any():
        raise ValueError("predictions.model_id contains blank identifiers")
    if (predictions["inchikey_full"].str.strip_chars() == "").any():
        raise ValueError("predictions.inchikey_full contains blank identifiers")
    if (truth["inchikey_full"].str.strip_chars() == "").any():
        raise ValueError("truth.inchikey_full contains blank identifiers")
    
    if not np.isfinite(predictions["pred"].to_numpy()).all():
        raise ValueError("pred contains non-finite values")
    if not np.isfinite(truth["truth"].to_numpy()).all():
        raise ValueError("truth contains non-finite values")

    # 3. Unique identity keys
    prediction_keys = ["model_id", "inchikey_full", "fold", "target"]
    prediction_counts = predictions.group_by(prediction_keys).len()
    prediction_dup_keys = prediction_counts.filter(pl.col("len") > 1)
    if len(prediction_dup_keys) > 0:
        raise ValueError(f"There are {len(prediction_dup_keys)} duplicates in prediction.")

    truth_keys_columns = ["inchikey_full", "fold", "target"]
    truth_counts = truth.group_by(truth_keys_columns).len()
    truth_dup_keys = truth_counts.filter(pl.col("len") > 1)
    if len(truth_dup_keys) > 0:
        raise ValueError(f"There are {len(truth_dup_keys)} duplicates in truth.")

    # 4. Canonical target set
    predictions_missing_target = set(_SCORED_DEFAULT) - set(predictions["target"].to_list())  
    if len(predictions_missing_target) > 0:
        raise ValueError(f"Predictions missing target: {sorted(predictions_missing_target)}")

    truth_missing_target = set(_SCORED_DEFAULT) - set(truth["target"].to_list())
    if len(truth_missing_target) > 0:
        raise ValueError(f"Truth missing target: {sorted(truth_missing_target)}")

    predictions_unexpected_target = set(predictions["target"].to_list()) - set(_SCORED_DEFAULT) 
    if len(predictions_unexpected_target) > 0:
        raise ValueError(f"Predictions unexpected target: {sorted(predictions_unexpected_target)}")

    truth_unexpected_target = set(truth["target"].to_list()) - set(_SCORED_DEFAULT) 
    if len(truth_unexpected_target) > 0:
        raise ValueError(f"Truth unexpected target: {sorted(truth_unexpected_target)}")

    # 5. Fold-specific mismatch diagnostics
    prediction_folds = predictions.select(
        "model_id",
        "inchikey_full",
        "target",
        pl.col("fold").alias("prediction_fold"),
    )

    truth_folds = truth.select(
        "inchikey_full",
        "target",
        pl.col("fold").alias("truth_fold"),
    )

    truth_fold_counts = truth.group_by("inchikey_full").agg(
        pl.col("fold").n_unique().alias("n_folds")
    )
    inconsistent_truth = truth_fold_counts.filter(pl.col("n_folds") > 1)
    if len(inconsistent_truth) > 0:
        raise ValueError(
            f"Truth molecules assigned to multiple folds: "
            f"{inconsistent_truth.head(5).to_dicts()}"
        )

    prediction_fold_counts = predictions.group_by(
        ["model_id", "inchikey_full"]
    ).agg(
        pl.col("fold").n_unique().alias("n_folds")
    )   
    inconsistent_prediction = prediction_fold_counts.filter(pl.col("n_folds") > 1)
    if len(inconsistent_prediction) > 0:
        raise ValueError(
            f"Prediction molecules assigned to multiple folds: "
            f"{inconsistent_prediction.head(5).to_dicts()}"
        )

    fold_mismatch = (
        prediction_folds.join(
            truth_folds, 
            on = ["inchikey_full", "target"],
            how = "inner",
        )
        .filter(pl.col("prediction_fold") != pl.col("truth_fold"))
    )
    if len(fold_mismatch) > 0:
        raise ValueError(
            "Fold mismatch error. "
            f"Sample of fold mismatch: {fold_mismatch.head(5).to_dicts()}"
        )

    # 6. Exact per-model key coverage
    model_ids = predictions["model_id"].unique().to_list()
    key_columns = ["inchikey_full", "fold", "target"]
    truth_keys = truth.select(key_columns)

    for model_id in model_ids:
        model_keys = predictions.filter(pl.col("model_id") == model_id).select(key_columns)

        extra = model_keys.join(truth_keys, on = key_columns, how = "anti")
        if len(extra) > 0:
            raise ValueError(
                f"model {model_id!r} has {len(extra)} OOF prediction key(s) with no matching truth row: "
                f"{extra.head(3).to_dicts()}"
            )

        model_predictions = predictions.filter(pl.col("model_id") == model_id)
        for target in model_predictions["target"].unique().to_list():
            expected = truth_keys.filter(pl.col("target") == target)
            actual = model_keys.filter(pl.col("target") == target)

            missing = expected.join(actual, on=key_columns, how="anti")
            if not missing.is_empty():
                raise ValueError(
                    f"model {model_id!r} is missing prediction keys for {target}"
                )


def _seed_averaging(
    prediction_by_seed: dict[int, pl.DataFrame],
    truth: pl.DataFrame,
    model_id: str,
) -> pl.DataFrame:
    """Average seeds for models that have same configuration and differ only in seed.
    
    Inputs must be runs of the same configuration with distinct training seeds, as 
    supplied by the multi-seed OOF runner.
    """

    for seed in prediction_by_seed.keys():
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed keys must be integers")

    if not isinstance(model_id, str):
        raise ValueError("model_id must be a string")

    if model_id.strip() == "":
        raise ValueError("model_id cannot be blank")

    if len(prediction_by_seed) < 3:
        raise ValueError("There must be at least three seed entries")

    for table in prediction_by_seed.values():
        _validate_inputs(table, truth)

        if table["model_id"].n_unique() > 1:
            raise ValueError("Each prediction table must contain exactly one model_id.")

    combined_predictions = pl.concat(list(prediction_by_seed.values()))

    average_prediction = combined_predictions.group_by(
        pl.col("inchikey_full"), 
        pl.col("fold"), 
        pl.col("target"),
    ).agg(pl.col("pred").mean())

    average_prediction = average_prediction.with_columns(pl.lit(model_id).alias("model_id"))

    return average_prediction.select(
        "model_id", "inchikey_full", "fold", "target", "pred"
    ).sort(["inchikey_full", "fold", "target"])
    

def _align_target(
    predictions: pl.DataFrame,
    truth: pl.DataFrame, 
    target: str,
    model_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return truth values and an aligned prediction matrix for one target."""

    target_truth = truth.filter(pl.col("target") == target)
    target_truth = target_truth.sort(["inchikey_full", "fold"])
    y = target_truth["truth"].to_numpy()

    prediction_columns = []

    for model_id in model_ids:
        target_prediction = predictions.filter(
            (pl.col("model_id") == model_id) 
            & (pl.col("target") == target)
        ).sort(["inchikey_full", "fold"])

        prediction_columns.append(target_prediction["pred"].to_numpy())

    X = np.column_stack(prediction_columns)

    return y, X


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute RMSE"""
    a = [float(v) for v in actual]
    p = [float(v) for v in predicted]
    n = len(a)
    ss_res = sum((ai - pi) ** 2 for ai, pi in zip(a, p))
    rmse = (ss_res / n) ** 0.5
    return rmse


def fit_nnls_stack(predictions: pl.DataFrame, truth: pl.DataFrame) -> dict[str, dict]:
    """Fit separate non-negative ensemble weights for the four scored CYP targets.

    Parameters
    ----------
    predictions : pl.DataFrame
        Long-format OOF predictions with columns:
        model_id, inchikey_full, fold, target, pred.
    truth : pl.DataFrame
        Corresponding observed labels with columns:
        inchikey_full, fold, target, truth.

    Input requirements
    ------------------
    Model IDs, molecule IDs, and target names must be strings. Model and
    molecule IDs must not be blank. Folds must be integers, and each molecule
    must have one consistent fold assignment across targets and models.
    Required fields must not be null; pred and truth must be finite integers
    or floats.

    Both tables must contain exactly the four scored targets. Truth keys
    (inchikey_full, fold, target) must be unique. Prediction keys must be
    unique within each model. Input row order does not matter.

    Different targets may use different sets of models; each scored target 
    must have at least one model.

    Returns
    -------
    dict[str, dict]
        Results keyed by target. Each target contains:
        - weights: model ID to fitted coefficient, including zero weights.
        - model_rmse: model ID to RMSE on that target's truth rows.
        - ensemble_rmse: RMSE of the weighted ensemble.
        - included_models: model IDs with positive weights.
        - dropped_models: model IDs with zero weights.

    Raises
    ------
    ValueError
        If the inputs violate the required schema, value, or alignment rules.

    Notes
    -----
    Weights are non-negative but are not constrained to sum to one.
    Ensemble RMSE is measured on the rows used to fit the weights; it is a
    fit diagnostic, not an independent estimate of generalization.
    Callers must ensure base predictions were genuinely produced out of fold;
    table validation cannot verify how the base models were trained.
    """

    _validate_inputs(predictions, truth)

    results = {}

    for target in _SCORED_DEFAULT:
        model_ids = sorted(
            predictions.filter(pl.col("target") == target)["model_id"]
            .unique()
            .to_list()
        )

        y, X = _align_target(predictions, truth, target, model_ids)

        weights, _residual_norm = nnls(X, y)

        model_rmse = {}
        model_weights = {}
        included_models = []
        dropped_models = []

        for column, model_id in enumerate(model_ids):
            model_rmse[model_id] = _rmse(y, X[:, column])
            model_weights[model_id] = float(weights[column])

            if weights[column] > 0:
                included_models.append(model_id)
            else:
                dropped_models.append(model_id)

        ensemble_predictions = X @ weights
        ensemble_rmse = _rmse(y, ensemble_predictions)
        
        results[target] = {
            "weights": model_weights,
            "model_rmse": model_rmse,
            "ensemble_rmse": ensemble_rmse,
            "included_models": included_models,
            "dropped_models": dropped_models,
        }

    return results


def fit_ridgecv_stack(predictions: pl.DataFrame, truth: pl.DataFrame) -> dict[str, dict]:
    """Fit per-target ridge stackers on OOF predictions, selecting alpha
    using the saved folds. Return weights, intercepts, and fit metrics.
    """

    _validate_inputs(predictions, truth)

    results = {}
    
    for target in _SCORED_DEFAULT:
        model_ids = sorted(
            predictions.filter(pl.col("target") == target)["model_id"]
            .unique()
            .to_list()
        )

        y, X = _align_target(predictions, truth, target, model_ids)

        target_truth = truth.filter(
            pl.col("target") == target
        ).sort(["inchikey_full", "fold"])

        if (target_truth["fold"] < 0).any():
            raise ValueError("Fold labels must be non-negative")
    
        if target_truth["fold"].n_unique() < 2:
            raise ValueError("There must be at least 2 distinct folds per target")
            
        cv = PredefinedSplit(test_fold=target_truth["fold"].to_numpy())

        ridge = RidgeCV(
            alphas=[0.1, 1.0, 10.0],
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        ridge.fit(X, y)
        weights = ridge.coef_        

        model_rmse = {}
        model_weights = {}
        included_models = []
        dropped_models = []

        for column, model_id in enumerate(model_ids):
            model_rmse[model_id] = _rmse(y, X[:, column])
            model_weights[model_id] = float(weights[column])

            if weights[column] != 0:
                included_models.append(model_id)
            else:
                dropped_models.append(model_id)

        ensemble_predictions = ridge.predict(X)
        ensemble_rmse = _rmse(y, ensemble_predictions)
        
        results[target] = {
            "weights": model_weights,
            "model_rmse": model_rmse,
            "ensemble_rmse": ensemble_rmse,
            "included_models": included_models,
            "dropped_models": dropped_models,
            "intercept": float(ridge.intercept_),
            "alpha": float(ridge.alpha_),
        }

    return results


def fit_elasticnetcv_stack(predictions: pl.DataFrame, truth: pl.DataFrame) -> dict[str, dict]:
    """Fit per-target elastic-net stackers on OOF predictions, selecting alpha
    and l1_ratio using saved folds. Return weights, intercepts, and fit metrics.
    """

    _validate_inputs(predictions, truth)

    results = {}
    
    for target in _SCORED_DEFAULT:
        model_ids = sorted(
            predictions.filter(pl.col("target") == target)["model_id"]
            .unique()
            .to_list()
        )

        y, X = _align_target(predictions, truth, target, model_ids)

        target_truth = truth.filter(
            pl.col("target") == target
        ).sort(["inchikey_full", "fold"])

        if (target_truth["fold"] < 0).any():
            raise ValueError("Fold labels must be non-negative")
    
        if target_truth["fold"].n_unique() < 2:
            raise ValueError("There must be at least 2 distinct folds per target")
            
        cv = PredefinedSplit(test_fold=target_truth["fold"].to_numpy())

        elastic = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            cv=cv,
        )
        elastic.fit(X, y)
        weights = elastic.coef_        

        model_rmse = {}
        model_weights = {}
        included_models = []
        dropped_models = []

        for column, model_id in enumerate(model_ids):
            model_rmse[model_id] = _rmse(y, X[:, column])
            model_weights[model_id] = float(weights[column])

            if weights[column] != 0:
                included_models.append(model_id)
            else:
                dropped_models.append(model_id)

        ensemble_predictions = elastic.predict(X)
        ensemble_rmse = _rmse(y, ensemble_predictions)
        
        results[target] = {
            "weights": model_weights,
            "model_rmse": model_rmse,
            "ensemble_rmse": ensemble_rmse,
            "included_models": included_models,
            "dropped_models": dropped_models,
            "intercept": float(elastic.intercept_),
            "alpha": float(elastic.alpha_),
            "l1_ratio": float(elastic.l1_ratio_),
        }

    return results


def predict_stack(
    predictions: pl.DataFrame,
    stack_results: dict,
) -> pl.DataFrame:
    """Align model predictions to fitted per-target weights and apply the
    weighted sum plus intercept to return ensemble predictions.
    """

    keys = ["inchikey_full"]
    if "fold" in predictions.columns:
        keys.append("fold")

    output_tables = []

    for target, fitted in stack_results.items():
        model_ids = list(fitted["weights"])

        target_predictions = predictions.filter(
            pl.col("target") == target
        )

        molecule_ids = (
            target_predictions
            .filter(pl.col("model_id") == model_ids[0])
            .select(keys)
            .sort(keys)
        )

        if molecule_ids.is_empty():
            raise ValueError(f"No predictions for {target}")

        columns = []

        for model_id in model_ids:
            model_predictions = target_predictions.filter(
                pl.col("model_id") == model_id
            ).select([*keys, "pred"])

            extra = model_predictions.join(
                molecule_ids, on=keys, how="anti"
            )
            if not extra.is_empty():
                raise ValueError(f"Molecule IDs differ for {model_id}")

            aligned = molecule_ids.join(
                model_predictions,
                on=keys,
                how="left",
                validate="1:1",
            ).sort(keys)

            if aligned["pred"].null_count() > 0:
                raise ValueError(f"Missing predictions for {model_id}")
            if not aligned["pred"].is_finite().all():
                raise ValueError(f"Nonfinite predictions for {model_id}")

            columns.append(aligned["pred"].to_numpy())

        X = np.column_stack(columns)
        weights = np.array([fitted["weights"][model_id] for model_id in model_ids])
        ensemble_predictions = X @ weights + fitted.get("intercept", 0.0)

        output_tables.append(
            molecule_ids.with_columns(
                pl.lit(target).alias("target"),
                pl.Series("pred", ensemble_predictions),
            )
        )

    return pl.concat(output_tables)