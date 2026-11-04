import numpy as np
import polars as pl
import pytest

from models.caruana import fit_caruana_stack, bagging_wrapper
from models.stack import predict_stack
import models.caruana as caruana


@pytest.fixture
def complementary_predictions():
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]
    truth_rows = []
    prediction_rows = []

    for target in targets:
        for index, actual in enumerate([1.0, 2.0, 3.0, 4.0]):
            identity = {
                "inchikey_full": f"molecule-{index}",
                "fold": index % 2,
                "target": target,
            }
            truth_rows.append({**identity, "truth": actual})

            # Two selections of A and one of B cancel their errors.
            for model_id, offset in [("A", 1.0), ("B", -2.0)]:
                prediction_rows.append({
                    **identity,
                    "model_id": model_id,
                    "pred": actual + offset,
                })

    return pl.DataFrame(prediction_rows), pl.DataFrame(truth_rows)


@pytest.mark.parametrize("metric", ["rmse", "mae", "rae"])
def test_repeated_selection_preserves_best_ensemble(
    complementary_predictions, metric
):
    predictions, truth = complementary_predictions

    results = fit_caruana_stack(
        predictions,
        truth,
        max_iterations=4,
        metric=metric,
    )

    assert set(results) == set(truth["target"].to_list())

    for fitted in results.values():
        assert fitted["selection_counts"] == {"A": 2, "B": 1}
        assert fitted["weights"] == pytest.approx({
            "A": 2 / 3,
            "B": 1 / 3,
        })
        assert fitted["best_iteration"] == 3
        assert fitted["selection_loss"] == pytest.approx(0.0)
        assert fitted["ensemble_rmse"] == pytest.approx(0.0)
        assert len(fitted["history"]) == 4
        assert fitted["history"][-1]["loss"] > 0
        assert np.isclose(sum(fitted["weights"].values()), 1.0)


@pytest.mark.parametrize(
    ("metric", "winner"),
    [("rmse", "A"), ("mae", "B"), ("rae", "B")],
)
def test_metric_changes_selection(complementary_predictions, metric, winner):
    predictions, truth = complementary_predictions
    predictions = (
        predictions.drop("pred")
        .join(truth, on=["inchikey_full", "fold", "target"])
        .with_columns(
            (
                pl.col("truth")
                + pl.when(pl.col("model_id") == "A")
                .then(1.0)
                .otherwise(
                    pl.when(pl.col("inchikey_full") == "molecule-0")
                    .then(3.0)
                    .otherwise(0.0)
                )
            ).alias("pred")
        )
        .drop("truth")
    )
    results = fit_caruana_stack(
        predictions, truth, max_iterations=1, metric=metric
    )
    for fitted in results.values():
        assert fitted["weights"][winner] == 1.0


def test_row_alignment(complementary_predictions):
    predictions, truth = complementary_predictions
    expected = fit_caruana_stack(predictions, truth, max_iterations=4)
    actual = fit_caruana_stack(
        predictions.sample(fraction=1, shuffle=True, seed=11),
        truth.sample(fraction=1, shuffle=True, seed=22),
        max_iterations=4,
    )
    for target in expected:
        assert actual[target]["weights"] == expected[target]["weights"]
        assert actual[target]["selection_loss"] == pytest.approx(
            expected[target]["selection_loss"]
        )


def test_predict_stack_compatibility(complementary_predictions):
    predictions, truth = complementary_predictions
    fitted = fit_caruana_stack(predictions, truth, max_iterations=4)
    output = predict_stack(predictions, fitted).join(
        truth, on=["inchikey_full", "fold", "target"], validate="1:1"
    )
    assert output.height == truth.height
    np.testing.assert_allclose(output["pred"], output["truth"])


def test_single_model_retains_first_iteration(complementary_predictions):
    predictions, truth = complementary_predictions
    results = fit_caruana_stack(
        predictions.filter(pl.col("model_id") == "A"),
        truth,
        max_iterations=4,
    )
    for fitted in results.values():
        assert fitted["weights"] == {"A": 1.0}
        assert fitted["best_iteration"] == 1
        assert len(fitted["history"]) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "unknown"},
        *({"max_iterations": value} for value in [0, -1, 1.5, True]),
    ],
)
def test_invalid_arguments(complementary_predictions, kwargs):
    predictions, truth = complementary_predictions
    message = "Unknown metric" if "metric" in kwargs else "positive integer"
    with pytest.raises(ValueError, match=message):
        fit_caruana_stack(predictions, truth, **kwargs)


@pytest.mark.parametrize("metric", ["rmse", "mae", "rae"])
def test_constant_labels(complementary_predictions, metric):
    predictions, truth = complementary_predictions
    truth = truth.with_columns(pl.lit(2.0).alias("truth"))

    if metric == "rae":
        with pytest.raises(ValueError, match="constant"):
            fit_caruana_stack(predictions, truth, metric=metric)
    else:
        results = fit_caruana_stack(
            predictions, truth, metric=metric, max_iterations=4
        )
        assert all(
            np.isfinite(fitted["selection_loss"])
            for fitted in results.values()
        )


def test_target_independence(complementary_predictions):
    predictions, truth = complementary_predictions
    special_target = sorted(truth["target"].unique().to_list())[0]

    # Swap model identities for one target, reversing its expected weights.
    predictions = predictions.with_columns(
        pl.when(pl.col("target") == special_target)
        .then(
            pl.when(pl.col("model_id") == "A")
            .then(pl.lit("B"))
            .otherwise(pl.lit("A"))
        )
        .otherwise(pl.col("model_id"))
        .alias("model_id")
    )
    results = fit_caruana_stack(predictions, truth, max_iterations=4)
    for target, fitted in results.items():
        expected = (
            {"A": 1 / 3, "B": 2 / 3}
            if target == special_target
            else {"A": 2 / 3, "B": 1 / 3}
        )
        assert fitted["weights"] == pytest.approx(expected)


def test_full_library_matches_stepwise(complementary_predictions):
    predictions, truth = complementary_predictions
    expected = fit_caruana_stack(predictions, truth, max_iterations=4)
    actual = bagging_wrapper(
        predictions, truth, subset_size=2, n_bags=3, max_iterations=4
    )
    for target in expected:
        assert actual[target]["weights"] == pytest.approx(
            expected[target]["weights"]
        )


def test_bagging_reproducible(complementary_predictions):
    predictions, truth = complementary_predictions
    kwargs = dict(subset_size=1, n_bags=10, seed=37, max_iterations=4)
    first = bagging_wrapper(predictions, truth, **kwargs)
    second = bagging_wrapper(predictions, truth, **kwargs)
    assert first == second


def test_bagging_averages_all_bags(complementary_predictions, monkeypatch):
    predictions, truth = complementary_predictions
    recorded = []
    original = caruana.fit_caruana_stack

    def capture(predictions, truth, **kwargs):
        for target in truth["target"].unique():
            candidates = predictions.filter(pl.col("target") == target)
            assert candidates["model_id"].n_unique() == 1

        fitted = original(predictions, truth, **kwargs)
        recorded.append(fitted)
        return fitted

    monkeypatch.setattr(caruana, "fit_caruana_stack", capture)
    actual = bagging_wrapper(
        predictions, truth, subset_size=1, n_bags=10, max_iterations=4
    )

    assert len(recorded) == 10
    for target, fitted in actual.items():
        expected = {
            model: sum(
                bag[target]["weights"].get(model, 0.0)
                for bag in recorded
            ) / 10
            for model in ["A", "B"]
        }
        assert fitted["weights"] == pytest.approx(expected)
        assert sum(fitted["weights"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_bags", 0),
        ("n_bags", True),
        ("n_bags", 1.5),
        ("subset_size", 0),
        ("subset_size", True),
        ("subset_size", 1.5),
        ("subset_size", 3),
    ],
)
def test_invalid_bagging_arguments(complementary_predictions, name, value):
    predictions, truth = complementary_predictions
    kwargs = {"subset_size": 1, "n_bags": 2, name: value}

    with pytest.raises(ValueError, match=name):
        bagging_wrapper(predictions, truth, **kwargs)

@pytest.mark.parametrize("metric", ["rmse", "mae", "rae"])
def test_array_interface_matches_table_interface(complementary_predictions, metric):
    predictions, truth = complementary_predictions
    y = np.array([1., 2., 3., 4.])
    X = np.column_stack([y + 1, y - 2])
    original = X.copy()
    weights, diagnostics = caruana.fit_caruana_weights(X, y, max_iterations=4, metric=metric)
    np.testing.assert_allclose(weights, [2 / 3, 1 / 3])
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(diagnostics["selection_counts"], [2, 1])
    assert diagnostics["best_iteration"] == 3
    assert diagnostics["history"][-1]["loss"] > diagnostics["selection_loss"]
    for fitted in fit_caruana_stack(predictions, truth, max_iterations=4, metric=metric).values():
        np.testing.assert_allclose(weights, [fitted["weights"][name] for name in ["A", "B"]])
        assert diagnostics["selection_loss"] == fitted["selection_loss"]
        assert [step["model_index"] for step in diagnostics["history"]] == [
            ["A", "B"].index(step["model_id"]) for step in fitted["history"]
        ]
    # Apply weights to a different population while preserving candidate order.
    y_new = np.array([5., 8.])
    np.testing.assert_allclose(np.column_stack([y_new + 1, y_new - 2]) @ weights, y_new)


def test_array_ties_choose_first_column_and_earliest_iteration():
    weights, info = caruana.fit_caruana_weights(np.ones((3, 2)), np.arange(3.), max_iterations=4)
    np.testing.assert_array_equal(weights, [1., 0.])
    assert info["best_iteration"] == 1


@pytest.mark.parametrize(("X", "y", "message"), [
    (np.ones(3), np.ones(3), "dimensional"),
    (np.ones((3, 2)), np.ones((3, 1)), "dimensional"),
    (np.empty((0, 2)), np.empty(0), "at least"),
    (np.empty((3, 0)), np.ones(3), "at least"),
    (np.ones((3, 2)), np.ones(2), "same number"),
    (np.array([[np.nan]]), np.ones(1), "finite"),
    (np.array([[np.inf]]), np.ones(1), "finite"),
    (np.ones((1, 1)), np.array([np.nan]), "finite"),
    (np.ones((1, 1)), np.array([np.inf]), "finite"),
])
def test_array_rejects_invalid_inputs(X, y, message):
    with pytest.raises(ValueError, match=message):
        caruana.fit_caruana_weights(X, y)


@pytest.mark.parametrize("kwargs", [
    {"metric": "unknown"},
    *({"max_iterations": value} for value in [0, -1, 1.5, True]),
])
def test_array_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        caruana.fit_caruana_weights(np.ones((3, 1)), np.arange(3.), **kwargs)
