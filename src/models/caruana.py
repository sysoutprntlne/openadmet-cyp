"""Ensemble selection from OOF arrays or aligned prediction tables."""
from typing import Literal

import numpy as np
import polars as pl

from models.stack import _align_target, _validate_inputs

def _selection_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
    metric: str,
) -> float:
    """Compute the selection metric."""

    errors = actual - predicted

    if metric == "rmse":
        return float(np.sqrt(np.mean(errors**2)))

    if metric == "mae":
        return float(np.mean(np.abs(errors)))

    if metric == "rae":
        denominator = np.sum(np.abs(actual - actual.mean()))
        if denominator == 0:
            raise ValueError("RAE is undefined for constant target labels.")
        return float(np.sum(np.abs(errors)) / denominator)

    raise ValueError(f"Unknown metric: {metric!r}")


def fit_caruana_weights(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iterations: int = 100,
    metric: Literal["rmse", "mae", "rae"] = "rmse",
) -> tuple[np.ndarray, dict]:
    """Fit one target's ensemble weights from aligned OOF predictions.

    Selection is with replacement. Returned weights are nonnegative
    and sum to one, with no intercept. Diagnostics contain selection_counts
    in column order, best_iteration, selection_metric, selection_loss, and the
    full history (using zero-based model_index). 
    """
    _validate_selection_options(max_iterations, metric)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("X must be two-dimensional and y must be one-dimensional")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must contain at least one row and one model")
    if X.shape[0] != len(y):
        raise ValueError("X and y must have the same number of rows")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")

    prediction_sum = np.zeros(len(y), dtype=float)
    selection_counts = np.zeros(X.shape[1], dtype=int)

    best_loss = float("inf")
    best_counts = selection_counts.copy()
    history = []

    for iteration in range(1, max_iterations + 1):
        candidate_losses = np.array([
            _selection_loss(
                y,
                (prediction_sum + X[:, column]) / iteration,
                metric,
            )
            for column in range(X.shape[1])
        ])

        selected = int(np.argmin(candidate_losses))
        selected_loss = float(candidate_losses[selected])

        prediction_sum += X[:, selected]
        selection_counts[selected] += 1

        history.append({
            "iteration": iteration,
            "model_index": selected,
            "loss": selected_loss,
        })

        if selected_loss < best_loss:
            best_loss = selected_loss
            best_counts = selection_counts.copy()

    weights = best_counts / best_counts.sum()

    return weights, {
        "selection_metric": metric,
        "selection_loss": best_loss,
        "selection_counts": best_counts,
        "best_iteration": int(best_counts.sum()),
        "history": history,
    }


def _validate_selection_options(max_iterations: int, metric: str) -> None:
    """Validate options shared by the array and table interfaces."""
    if metric not in {"rmse", "mae", "rae"}:
        raise ValueError(f"Unknown metric: {metric!r}")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")


def fit_caruana_stack(
    predictions: pl.DataFrame,
    truth: pl.DataFrame,
    *,
    max_iterations: int = 100,
    metric: Literal["rmse", "mae", "rae"] = "rmse"
) -> dict[str, dict]:
    """Align OOF tables and fit separate weights for each scored target."""

    _validate_selection_options(max_iterations, metric)

    _validate_inputs(predictions, truth)

    results = {}

    for target in sorted(truth["target"].unique().to_list()):
        model_ids = sorted(
            predictions.filter(pl.col("target") == target)["model_id"]
            .unique()
            .to_list()
        )

        y, X = _align_target(predictions, truth, target, model_ids)

        weights, diagnostics = fit_caruana_weights(
            X, y, max_iterations=max_iterations, metric=metric,
        )
        best_counts = diagnostics["selection_counts"]
        ensemble_predictions = X @ weights

        results[target] = {
            "weights": {
                model_id: float(weights[column])
                for column, model_id in enumerate(model_ids)
            },
            "model_rmse": {
                model_id: _selection_loss(y, X[:, column], "rmse")
                for column, model_id in enumerate(model_ids)
            },
            "ensemble_rmse": _selection_loss(y, ensemble_predictions, "rmse"),
            "included_models": [
                model_id
                for column, model_id in enumerate(model_ids)
                if best_counts[column] > 0
            ],
            "dropped_models": [
                model_id
                for column, model_id in enumerate(model_ids)
                if best_counts[column] == 0
            ],
            "selection_metric": metric,
            "selection_loss": diagnostics["selection_loss"],
            "selection_counts": {
                model_id: int(best_counts[column])
                for column, model_id in enumerate(model_ids)
            },
            "best_iteration": int(best_counts.sum()),
            "history": [
                {"iteration": step["iteration"], "model_id": model_ids[step["model_index"]], "loss": step["loss"]}
                for step in diagnostics["history"]
            ],
        }

    return results


def bagging_wrapper(
    predictions: pl.DataFrame,
    truth: pl.DataFrame,
    *,
    max_iterations: int = 100,
    metric: Literal["rmse", "mae", "rae"] = "rmse",
    seed: int = 37,
    subset_size: int,
    n_bags: int,
) -> dict[str, dict]:
    """Average Caruana weights over random candidate subsets per target."""

    _validate_inputs(predictions, truth)

    for name, value in [("n_bags", n_bags), ("subset_size", subset_size)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    for target in sorted(truth["target"].unique().to_list()):
        available = (
            predictions.filter(pl.col("target") == target)["model_id"]
            .n_unique()
        )
        if subset_size > available:
            raise ValueError(
                f"subset_size={subset_size} exceeds {available} models for {target}"
            )

    rng = np.random.default_rng(seed)

    bag_results = []

    for bag in range(n_bags):
        bag_tables = []

        for target in sorted(truth["target"].unique().to_list()):
            target_predictions = predictions.filter(pl.col("target") == target)
            model_ids = sorted(target_predictions["model_id"].unique().to_list())

            subset_ids = rng.choice(
                model_ids,
                size=subset_size,
                replace=False,
            ).tolist()

            bag_tables.append(
                target_predictions.filter(pl.col("model_id").is_in(subset_ids))
            )

        bag_predictions = pl.concat(bag_tables)

        bag_results.append(
            fit_caruana_stack(
                bag_predictions,
                truth,
                max_iterations=max_iterations,
                metric=metric,
            )
        )

    results = {}

    for target in sorted(truth["target"].unique().to_list()):
        model_ids = sorted(
            predictions.filter(pl.col("target") == target)["model_id"]
            .unique()
            .to_list()
        )

        weights = {
            model_id: float(np.mean([
                bag[target]["weights"].get(model_id, 0.0)
                for bag in bag_results
            ]))
            for model_id in model_ids
        }

        results[target] = {
            "weights": weights,
            "included_models": [
                model_id for model_id, weight in weights.items() if weight > 0
            ],
            "dropped_models": [
                model_id for model_id, weight in weights.items() if weight == 0
            ],
        }

    return results

