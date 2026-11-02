import numpy as np
import polars as pl
import pytest

from models.stack import (
    fit_nnls_stack,
    _seed_averaging,
    fit_ridgecv_stack,
    fit_elasticnetcv_stack,
    predict_stack
)
from polars.testing import assert_frame_equal
from models.validation import deduplication_filter


def test_recovers_expected_weights(stack_tables) -> None:
    predictions, truth = stack_tables

    results = fit_nnls_stack(predictions, truth)

    expected = {
        "CYP1A2_pIC50_direct_inhibition": [0.2, 0.8, 0.0],
        "CYP2C9_pIC50_direct_inhibition": [0.4, 0.6, 0.0],
        "CYP2D6_pIC50_direct_inhibition": [0.6, 0.4, 0.0],
        "CYP3A4_pIC50_direct_inhibition": [0.8, 0.2, 0.0],
    }

    for target, expected_weights in expected.items():
        actual_weights = [
            results[target]["weights"][model_id]
            for model_id in ["model_a", "model_b", "model_c"]
        ]

        np.testing.assert_allclose(
            actual_weights, 
            expected_weights, 
            rtol=0, 
            atol=1e-10,
        )

        assert results[target]["ensemble_rmse"] < 1e-10
        assert results[target]["included_models"] == ["model_a", "model_b"]
        assert results[target]["dropped_models"] == ["model_c"]


def test_shuffled_rows_preserve_results(stack_tables) -> None:
    predictions, truth = stack_tables

    expected = fit_nnls_stack(predictions, truth)

    shuffled_predictions = predictions.sample(
        fraction=1.0, shuffle=True, seed =42
    )
    shuffled_truth = truth.sample(
        fraction=1.0, shuffle=True, seed=17
    )

    actual = fit_nnls_stack(shuffled_predictions, shuffled_truth)

    assert actual == expected


# Table validation and alignment
def test_rejects_duplicate_predictions(stack_tables) -> None:
    predictions, truth = stack_tables

    duplicated = pl.concat([predictions, predictions.head(1)])

    with pytest.raises(ValueError, match="duplicates in prediction"):
        fit_nnls_stack(duplicated, truth)


def test_rejects_missing_prediction(stack_tables) -> None:
    predictions, truth = stack_tables

    incomplete = predictions.slice(1)

    with pytest.raises(ValueError, match="missing prediction keys"):
        fit_nnls_stack(incomplete, truth)


def test_rejects_mismatched_folds(stack_tables) -> None:
    predictions, truth = stack_tables

    wrong_folds = predictions.with_columns(
        (1 - pl.col("fold")).alias("fold")
    )

    with pytest.raises(ValueError, match="Fold mismatch"):
        fit_nnls_stack(wrong_folds, truth)


def test_rejects_duplicate_truth(stack_tables) -> None:
    predictions, truth = stack_tables

    duplicated_truth = pl.concat([truth, truth.head(1)])

    with pytest.raises(ValueError, match="duplicates in truth"):
        fit_nnls_stack(predictions, duplicated_truth)


def test_rejects_extra_prediction(stack_tables) -> None: 
    predictions, truth = stack_tables

    extra_row = predictions.head(1).with_columns(
        pl.lit("SYNTHETIC_EXTRA").alias("inchikey_full")
    )
    extended = pl.concat([predictions, extra_row])

    with pytest.raises(ValueError, match="no matching truth row"):
        fit_nnls_stack(extended, truth)


# Reported metrics
def test_per_model_rmse(stack_tables) -> None:
    predictions, truth = stack_tables

    results = fit_nnls_stack(predictions, truth)

    expected_weights = {
        "CYP1A2_pIC50_direct_inhibition": (0.2, 0.8),
        "CYP2C9_pIC50_direct_inhibition": (0.4, 0.6),
        "CYP2D6_pIC50_direct_inhibition": (0.6, 0.4),
        "CYP3A4_pIC50_direct_inhibition": (0.8, 0.2),
    }

    for target, (a, b) in expected_weights.items():
        expected_rmse = {
            "model_a": (((1 - a) ** 2 + b ** 2) / 4) ** 0.5,
            "model_b": ((a ** 2 + (1 - b) ** 2) / 4) ** 0.5,
            "model_c": ((a ** 2 + b ** 2 + 1) / 4) ** 0.5,
        }

        for model_id, expected in expected_rmse.items():
            assert results[target]["model_rmse"][model_id] == pytest.approx(
                expected, abs=1e-10
            )


# Value validation
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("column", ["pred", "truth"])
def test_rejects_nonfinite_values(stack_tables, column: str, bad_value: float) -> None:
    predictions, truth = stack_tables

    if column == "pred":
        predictions = predictions.with_columns(
            pl.lit(bad_value).alias("pred")
        )
    else:
        truth = truth.with_columns(
            pl.lit(bad_value).alias("truth")
        )

    with pytest.raises(ValueError, match="contains non-finite values"):
        fit_nnls_stack(predictions, truth)


@pytest.mark.parametrize(
    "table_name, column",
    [
        ("predictions", "model_id"),
        ("predictions", "inchikey_full"),
        ("predictions", "fold"),
        ("predictions", "target"),
        ("predictions", "pred"),
        ("truth", "inchikey_full"),
        ("truth", "fold"),
        ("truth", "target"),
        ("truth", "truth"),
    ],
)
def test_rejects_null_values(stack_tables, table_name: str, column: str) -> None:
    predictions, truth = stack_tables

    if table_name == "predictions":
        predictions = predictions.with_columns(
            pl.lit(None, dtype=predictions[column].dtype).alias(column)
        )
    else:
        truth = truth.with_columns(
            pl.lit(None, dtype=truth[column].dtype).alias(column)
        )

    with pytest.raises(ValueError, match=f"null values in {table_name}"):
        fit_nnls_stack(predictions, truth)


# Fold integrity
@pytest.mark.parametrize("table_name", ["predictions", "truth"])
def test_rejects_multiple_folds_per_molecule(stack_tables, table_name: str) -> None:
    predictions, truth = stack_tables

    changed_fold = (
        pl.when(
            (pl.col("inchikey_full") == "SYNTHETIC_A")
            & (pl.col("target") == "CYP1A2_pIC50_direct_inhibition")
        )
        .then(pl.lit(1))
        .otherwise(pl.col("fold"))
        .alias("fold")
    )

    if table_name == "predictions":
        predictions = predictions.with_columns(changed_fold)
        message = "Prediction molecules assigned to multiple folds"
    else:
        truth = truth.with_columns(changed_fold)
        message = "Truth molecules assigned to multiple folds"

    with pytest.raises(ValueError, match=message):
        fit_nnls_stack(predictions, truth)

@pytest.fixture
def seed_averaging_data(stack_tables):
    predictions, truth = stack_tables
    base = predictions.filter(pl.col("model_id") == "model_a")

    predictions_by_seed = {
        seed: base.with_columns(
            pl.lit(f"model_a_seed{seed}").alias("model_id"),
            (pl.col("pred") + offset).alias("pred"),
        )
        # This creates three prediction tables
        # Seed 0: subtract 1, Seed 1: unchanged, Seed 2: add 1
        for seed, offset in enumerate([-1.0, 0.0, 1.0])
    }

    expected = base.with_columns(
        pl.lit("model_a_averaged").alias("model_id")
    ).select(
        "model_id", "inchikey_full", "fold", "target", "pred"
    ).sort(["inchikey_full", "fold", "target"])

    return predictions_by_seed, truth, expected

def test_seed_averaging_mean_and_output(seed_averaging_data):
    predictions_by_seed, truth, expected = seed_averaging_data

    actual = _seed_averaging(
        predictions_by_seed,
        truth,
        model_id="model_a_averaged",

    )

    assert_frame_equal(actual, expected)

def test_seed_averaging_row_order_invariant(seed_averaging_data): 
    predictions_by_seed, truth, expected = seed_averaging_data

    predictions_by_seed = {
        seed: table.sample(fraction=1.0, shuffle=True, seed=seed + 10)
        for seed, table in predictions_by_seed.items()
    }
    truth = truth.sample(fraction=1.0, shuffle=True, seed=20)

    actual = _seed_averaging(
        predictions_by_seed,
        truth,
        model_id="model_a_averaged",

    )

    assert_frame_equal(actual, expected)

def test_seed_averaging_rejects_too_few_seeds(seed_averaging_data):
    predictions_by_seed, truth, expected = seed_averaging_data

    del predictions_by_seed[2]

    with pytest.raises(ValueError, match = "There must be at least three seed entries"): 
        _seed_averaging(
            predictions_by_seed,
            truth,
            model_id="model_a_averaged",
        )

@pytest.mark.parametrize("bad_seed", ["0", 0.5, True])
def test_seed_averaging_rejects_invalid_seed_keys(
    seed_averaging_data, bad_seed
):
    predictions_by_seed, truth, expected = seed_averaging_data
    table = predictions_by_seed.pop(1)
    predictions_by_seed[bad_seed] = table

    with pytest.raises(ValueError, match="seed keys must be integers"):
        _seed_averaging(predictions_by_seed, truth, "model_a_averaged")

@pytest.mark.parametrize("bad_model_id", ["", "   ", None, 123])
def test_seed_averaging_rejects_invalid_model_id(
    seed_averaging_data, bad_model_id
):
    predictions_by_seed, truth, expected = seed_averaging_data

    with pytest.raises(ValueError, match="model_id"):
        _seed_averaging(predictions_by_seed, truth, bad_model_id)

@pytest.mark.parametrize("case", ["missing", "duplicate", "multiple_models"])
def test_seed_averaging_rejects_invalid_prediction_tables(
    seed_averaging_data, case
): 
    predictions_by_seed, truth, expected = seed_averaging_data

    if case == "missing":
        predictions_by_seed[0] = predictions_by_seed[0].slice(1)

        with pytest.raises(ValueError, match="missing"):
            _seed_averaging(predictions_by_seed, truth, model_id="model_a_averaged")
    
    elif case == "duplicate":
        table = predictions_by_seed[0]
        predictions_by_seed[0] = pl.concat([table, table.head(1)])

        with pytest.raises(ValueError, match="duplicate"):
            _seed_averaging(predictions_by_seed, truth, model_id="model_a_averaged")

    elif case == "multiple_models":
        table = predictions_by_seed[0]
        other_model = table.with_columns(
            pl.lit("other_model").alias("model_id")
        )
        predictions_by_seed[0] = pl.concat([table, other_model])

        with pytest.raises(ValueError, match="Each prediction table must contain exactly one model_id."):
            _seed_averaging(predictions_by_seed, truth, model_id="model_a_averaged")

@pytest.fixture
def ridge_stack_data():
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]
    prediction_rows = []
    truth_rows = []

    for target in targets:
        for fold in range(3):
            for position, (a, b) in enumerate([
                (-1.0, -1.0),
                (-1.0, 1.0),
                (1.0, -1.0),
                (1.0, 1.0),
            ]):
                molecule = f"molecule_{fold}_{position}"
                key = {
                    "inchikey_full": molecule,
                    "fold": fold,
                    "target": target,
                }
                truth_rows.append({
                    **key,
                    "truth": 5.0 + 2.0 * a - b,
                })
                for model_id, prediction in [("model_a", a), ("model_b", b)]:
                    prediction_rows.append({
                        **key, 
                        "model_id": model_id, 
                        "pred": prediction,
                    })

    return pl.DataFrame(prediction_rows), pl.DataFrame(truth_rows)

def test_ridge_stack_weights_intercept_and_metrics(ridge_stack_data):
    predictions, truth = ridge_stack_data

    results = fit_ridgecv_stack(predictions, truth)

    assert set(results) == set(truth["target"].unique().to_list())

    for result in results.values():
        alpha = result["alpha"]
        assert alpha in [0.1, 1.0, 10.0]

        shrinkage = 12.0 / (12.0 + alpha)

        assert result["weights"]["model_a"] == pytest.approx(2.0 * shrinkage)
        assert result["weights"]["model_b"] == pytest.approx(-shrinkage)
        assert result["intercept"] == pytest.approx(5.0)

        assert result["model_rmse"]["model_a"] == pytest.approx(np.sqrt(27.0))
        assert result["model_rmse"]["model_b"] == pytest.approx(np.sqrt(33.0))

        expected_rmse = np.sqrt(5.0) * (1.0 - shrinkage)
        assert result["ensemble_rmse"] == pytest.approx(expected_rmse)

        assert result["included_models"] == ["model_a", "model_b"]
        assert result["dropped_models"] == []


@pytest.mark.parametrize(
    "fit_stack",
    [fit_ridgecv_stack, fit_elasticnetcv_stack],
    ids=["ridge", "elasticnet"],
)
def test_stack_row_order_invariant(ridge_stack_data, fit_stack):
    predictions, truth = ridge_stack_data

    shuffled_predictions = predictions.sample(
        fraction=1.0, shuffle=True, seed =42
    )

    shuffled_truth = truth.sample(
        fraction=1.0, shuffle=True, seed=42
    )

    shuffled_results = fit_stack(shuffled_predictions, shuffled_truth)
    results = fit_stack(predictions, truth)

    assert shuffled_results == results

@pytest.mark.parametrize(
    "fit_stack",
    [fit_ridgecv_stack, fit_elasticnetcv_stack],
    ids=["ridge", "elasticnet"],
)
def test_stack_includes_negative_weights(ridge_stack_data, fit_stack): 
    predictions, truth = ridge_stack_data
    
    results = fit_stack(predictions, truth)

    for result in results.values():
        assert result["weights"]["model_b"] < 0
        assert "model_b" in result["included_models"]
        assert "model_b" not in result["dropped_models"]

@pytest.mark.parametrize(
    "fit_stack, estimator_name",
    [
        (fit_ridgecv_stack, "RidgeCV"),
        (fit_elasticnetcv_stack, "ElasticNetCV"),
    ],
    ids=["ridge", "elasticnet"],
)
def test_stack_uses_supplied_folds(
    ridge_stack_data, monkeypatch, fit_stack, estimator_name
):
    predictions, truth = ridge_stack_data

    shuffled_predictions = predictions.sample(
        fraction=1.0, shuffle=True, seed =42
    )

    shuffled_truth = truth.sample(
        fraction=1.0, shuffle=True, seed=42
    )

    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    fit_calls = []

    class FakeEstimator:
        def __init__(self, *, cv, **kwargs):
            self.cv = cv

        def fit(self, X, y):
            expected_folds = expected_folds_by_target[len(fit_calls)]
            fit_calls.append(1)

            splits = list(self.cv.split(X, y))
            assert len(splits) == 3

            for fold, (train_indices, val_indices) in enumerate(splits):
                np.testing.assert_array_equal(
                    val_indices, np.flatnonzero(expected_folds == fold)
                )
                np.testing.assert_array_equal(
                    train_indices, np.flatnonzero(expected_folds != fold)
                )

            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0.0
            self.alpha_ = 0.1
            if estimator_name == "ElasticNetCV":
                self.l1_ratio_ = 0.5

            return self

        def predict(self, X):
            return np.zeros(X.shape[0])

    expected_folds_by_target = [
        truth.filter(pl.col("target") == target)
        .sort(["inchikey_full", "fold"])["fold"]
        .to_numpy()
        for target in targets
    ]
       
    monkeypatch.setattr(f"models.stack.{estimator_name}", FakeEstimator)
    fit_stack(shuffled_predictions, shuffled_truth)
    assert len(fit_calls) == 4

@pytest.mark.parametrize(
    "fit_stack",
    [fit_ridgecv_stack, fit_elasticnetcv_stack],
    ids=["ridge", "elasticnet"],
)
@pytest.mark.parametrize(
    "case, message",
    [
        ("negative", "Fold labels must be non-negative"),
        ("single_fold", "There must be at least 2 distinct folds per target"),
    ],
)
def test_stack_rejects_invalid_folds(
    ridge_stack_data, fit_stack, case, message):
    predictions, truth = ridge_stack_data

    if case == "negative":
        changed_fold = (
            pl.when(pl.col("fold") == 0)
            .then(pl.lit(-1))
            .otherwise(pl.col("fold"))
            .alias("fold")
        )
        predictions = predictions.with_columns(changed_fold)
        truth = truth.with_columns(changed_fold)

    if case == "single_fold":
        predictions = predictions.with_columns(fold=pl.lit(0))
        truth = truth.with_columns(fold=pl.lit(0))

    with pytest.raises(ValueError, match=message):
        fit_stack(predictions, truth)

def test_elastic_stack_selected_parameters(ridge_stack_data): 
    predictions, truth = ridge_stack_data

    results = fit_elasticnetcv_stack(predictions, truth)

    assert set(results) == set(truth["target"].unique().to_list())

    for result in results.values():
        alpha = result["alpha"]
        assert alpha > 0

        assert result["l1_ratio"] in [0.1, 0.5, 0.9]

def test_elastic_stack_intercept_and_metrics(ridge_stack_data):
    predictions, truth = ridge_stack_data
    
    results = fit_elasticnetcv_stack(predictions, truth)

    assert set(results) == set(truth["target"].unique().to_list())

    for result in results.values():
        assert result["intercept"] == pytest.approx(5.0)
        assert result["model_rmse"]["model_a"] == pytest.approx(np.sqrt(27.0))
        assert result["model_rmse"]["model_b"] == pytest.approx(np.sqrt(33.0))

        w_a = result["weights"]["model_a"]
        w_b = result["weights"]["model_b"]
        intercept = result["intercept"]

        expected_rmse = np.sqrt(
            (intercept - 5.0) ** 2
            + (w_a - 2.0) ** 2
            + (w_b + 1.0) ** 2
        )

        assert result["ensemble_rmse"] == pytest.approx(expected_rmse)

def test_elastic_stack_drops_zero_weight_model(ridge_stack_data): 
    predictions, truth = ridge_stack_data

    zero_model = predictions.filter(
        pl.col("model_id") == "model_a"
    ).with_columns(
        pl.lit("model_zero").alias("model_id"),
        pl.lit(0.0).alias("pred"),
    )

    predictions = pl.concat([predictions, zero_model])
    results = fit_elasticnetcv_stack(predictions, truth)

    for result in results.values():
        assert result["weights"]["model_zero"] == 0.0
        assert "model_zero" in result["dropped_models"]
        assert "model_zero" not in result["included_models"]


@pytest.mark.parametrize(
    "stacker",
    [fit_nnls_stack, fit_ridgecv_stack, fit_elasticnetcv_stack],
    ids=["nnls", "ridge", "elasticnet"],
)
def test_stack_accepts_different_models_per_target(ridge_stack_data, stacker):
    predictions, truth = ridge_stack_data
    target = sorted(truth["target"].unique().to_list())[0]

    predictions = predictions.filter(
        ~(
            (pl.col("target") == target)
            & (pl.col("model_id") == "model_b")
        )
    )

    results = stacker(predictions, truth)

    for name, result in results.items():
        expected_models = (
            {"model_a"} if name == target else {"model_a", "model_b"}
        )
        assert set(result["model_rmse"]) == expected_models
        assert set(result["weights"]) == expected_models
        assert np.isfinite(result["ensemble_rmse"])


def test_validate_rejects_partial_target_coverage(ridge_stack_data):
    predictions, truth = ridge_stack_data

    # Remove one molecule, not the whole model-target combination.
    incomplete = predictions.slice(1)

    with pytest.raises(ValueError, match="missing prediction keys"):
        fit_nnls_stack(incomplete, truth)


def test_validate_rejects_target_without_models(ridge_stack_data):
    predictions, truth = ridge_stack_data
    target = truth["target"][0]

    predictions = predictions.filter(pl.col("target") != target)

    with pytest.raises(ValueError, match="Predictions missing target"):
        fit_nnls_stack(predictions, truth)


@pytest.mark.parametrize(
    "stacker",
    [fit_nnls_stack, fit_ridgecv_stack, fit_elasticnetcv_stack],
    ids=["nnls", "ridge", "elasticnet"],
)
def test_deduplicated_oof_feeds_stack(ridge_stack_data, stacker):
    oof_predictions, oof_truth = ridge_stack_data
    targets = sorted(oof_truth["target"].unique().to_list())
    merged_target = targets[0]

    # Synthetic validation population with distinct molecule IDs.
    validation_predictions = oof_predictions.drop("fold").with_columns(
        (pl.lit("val_") + pl.col("inchikey_full")).alias("inchikey_full")
    )
    validation_truth = oof_truth.drop("fold").with_columns(
        (pl.lit("val_") + pl.col("inchikey_full")).alias("inchikey_full")
    )

    # On one target, make B identical to A so they form one cluster.
    a_predictions = (
        validation_predictions
        .filter(
            (pl.col("target") == merged_target)
            & (pl.col("model_id") == "model_a")
        )
        .with_columns(pl.lit("model_b").alias("model_id"))
    )
    validation_predictions = pl.concat([
        validation_predictions.filter(
            ~(
                (pl.col("target") == merged_target)
                & (pl.col("model_id") == "model_b")
            )
        ),
        a_predictions,
    ])

    _, clustered_oof, membership = deduplication_filter(
        validation_predictions,
        validation_truth,
        oof_predictions,
        oof_truth,
    )

    results = stacker(clustered_oof, oof_truth)

    assert set(results) == set(targets)

    for target in targets:
        cluster_ids = set(
            membership.filter(pl.col("target") == target)["cluster_id"]
        )
        assert len(cluster_ids) == (1 if target == merged_target else 2)
        assert set(results[target]["model_rmse"]) == cluster_ids
        assert set(results[target]["weights"]) == cluster_ids
        assert np.isfinite(results[target]["ensemble_rmse"])


@pytest.fixture
def stack_prediction_data():
    predictions = pl.DataFrame({
        "model_id": ["a", "a", "b", "b"],
        "inchikey_full": ["A", "B", "A", "B"],
        "target": ["t1"] * 4,
        "pred": [2.0, 4.0, 6.0, 8.0],
    })
    # Deliberately use a different order from the prediction rows.
    fitted = {"t1": {"weights": {"b": 0.75, "a": 0.25}}}
    return predictions, fitted


@pytest.mark.parametrize("intercept", [None, 1.5])
def test_predict_stack_weights_and_intercept(stack_prediction_data, intercept):
    predictions, fitted = stack_prediction_data
    if intercept is not None:
        fitted["t1"]["intercept"] = intercept

    result = predict_stack(predictions, fitted)

    expected = pl.DataFrame({
        "inchikey_full": ["A", "B"],
        "target": ["t1", "t1"],
        "pred": [5.0 + (intercept or 0.0), 7.0 + (intercept or 0.0)],
    })
    assert_frame_equal(result, expected)


def test_predict_stack_ignores_row_order(stack_prediction_data):
    predictions, fitted = stack_prediction_data
    expected = predict_stack(predictions, fitted)

    result = predict_stack(
        predictions.sample(fraction=1.0, shuffle=True, seed=42),
        fitted,
    )

    assert_frame_equal(result, expected)


@pytest.mark.parametrize("with_fold", [False, True])
def test_predict_stack_preserves_optional_fold(stack_prediction_data, with_fold):
    predictions, fitted = stack_prediction_data

    if with_fold:
        predictions = predictions.with_columns(
            pl.Series("fold", [0, 1, 0, 1])
        )

    result = predict_stack(predictions, fitted)

    assert ("fold" in result.columns) == with_fold
    if with_fold:
        assert result["fold"].to_list() == [0, 1]

    np.testing.assert_allclose(result["pred"].to_numpy(), [5.0, 7.0])


def test_predict_stack_supports_target_specific_models(stack_prediction_data):
    predictions, fitted = stack_prediction_data

    extra = pl.DataFrame({
        "model_id": ["c", "c"],
        "inchikey_full": ["A", "C"],
        "target": ["t2", "t2"],
        "pred": [3.0, 5.0],
    })
    predictions = pl.concat([predictions, extra])
    fitted["t2"] = {"weights": {"c": -2.0}, "intercept": 1.0}

    result = predict_stack(predictions, fitted)

    expected = pl.DataFrame({
        "inchikey_full": ["A", "B", "A", "C"],
        "target": ["t1", "t1", "t2", "t2"],
        "pred": [5.0, 7.0, -5.0, -9.0],
    })
    keys = ["target", "inchikey_full"]
    assert_frame_equal(result.sort(keys), expected.sort(keys))


@pytest.mark.parametrize(
    "case, error, match",
    [
        ("missing_row", ValueError, "Missing predictions"),
        ("missing_model", ValueError, "Missing predictions"),
        ("extra_row", ValueError, "Molecule IDs differ"),
        ("wrong_fold", ValueError, "Molecule IDs differ"),
        ("duplicate", pl.exceptions.ComputeError, "1:1"),
        ("null", ValueError, "Missing predictions"),
        ("nonfinite", ValueError, "Nonfinite predictions"),
    ],
)
def test_predict_stack_rejects_misaligned_predictions(
    stack_prediction_data,
    case,
    error,
    match,
):
    predictions, fitted = stack_prediction_data

    # B is the reference model because it is first in fitted["weights"].
    affected = (
        (pl.col("model_id") == "a")
        & (pl.col("inchikey_full") == "A")
    )

    if case == "missing_row":
        predictions = predictions.filter(~affected)
    elif case == "missing_model":
        predictions = predictions.filter(pl.col("model_id") != "a")
    elif case == "extra_row":
        extra = predictions.filter(affected).with_columns(
            pl.lit("EXTRA").alias("inchikey_full")
        )
        predictions = pl.concat([predictions, extra])
    elif case == "wrong_fold":
        predictions = predictions.with_columns(
            pl.when(affected).then(1).otherwise(0).alias("fold")
        )
    elif case == "duplicate":
        predictions = pl.concat([
            predictions,
            predictions.filter(affected),
        ])
    else:
        value = None if case == "null" else float("inf")
        predictions = predictions.with_columns(
            pl.when(affected)
            .then(pl.lit(value, dtype=pl.Float64))
            .otherwise(pl.col("pred"))
            .alias("pred")
        )

    with pytest.raises(error, match=match):
        predict_stack(predictions, fitted)