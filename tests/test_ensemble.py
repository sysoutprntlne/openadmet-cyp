import numpy as np
import polars as pl
import pytest

import models.ensemble as ensemble


@pytest.fixture
def ensemble_tables():
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    # Model: (validation error, OOF error).
    # A/B pass both filters and correlate perfectly.
    # Bad quality fails R² only; bad ratio fails the RAE ratio only.
    offsets = {
        "a": (0.2, 0.2),
        "b": (0.4, 0.4),
        "bad_quality": (10.0, 10.0),
        "bad_ratio": (1.0, 0.1),
    }

    tables = {}
    for population, prefix, shift, error_index in [
        ("validation", "V", 0.0, 0),
        ("oof", "O", 20.0, 1),
    ]:
        truth_rows = []
        prediction_rows = []

        for target in targets:
            for i in range(12):
                key = {
                    "inchikey_full": f"{prefix}{i:02d}",
                    "target": target,
                }
                if population == "oof":
                    key["fold"] = i % 3

                value = shift + i
                truth_rows.append({**key, "truth": value})

                for model, errors in offsets.items():
                    prediction_rows.append({
                        **key,
                        "model_id": model,
                        "pred": value + errors[error_index],
                    })

        tables[f"{population}_truth"] = pl.DataFrame(truth_rows)
        tables[f"{population}_predictions"] = pl.DataFrame(prediction_rows)

    return tables


@pytest.mark.parametrize("method", ["nnls", "ridgecv", "elasticnetcv"])
def test_filtered_stack_end_to_end(ensemble_tables, method):
    result = ensemble.fit_filtered_stack(**ensemble_tables, method=method)

    assert set(result) == {
        "stack", "filter_decisions", "cluster_membership",
        "validation_predictions", "ensemble_validation_predictions",
        "oof_predictions",
    }

    membership = result["cluster_membership"]
    targets = set(ensemble_tables["validation_truth"]["target"])

    assert set(result["stack"]) == targets
    assert membership.height == 2 * len(targets)
    assert set(membership["model_id"]) == {"a", "b"}

    for target in targets:
        members = membership.filter(pl.col("target") == target)
        cluster_ids = members["cluster_id"].unique().to_list()

        assert len(cluster_ids) == 1
        cluster_id = cluster_ids[0]
        assert set(members["model_id"]) == {"a", "b"}

        fitted = result["stack"][target]
        assert set(fitted["weights"]) == {cluster_id}
        assert np.isfinite(fitted["ensemble_rmse"])

        for population in ["validation", "oof"]:
            output = result[f"{population}_predictions"].filter(
                pl.col("target") == target
            )
            truth = ensemble_tables[f"{population}_truth"].filter(
                pl.col("target") == target
            )
            keys = ["inchikey_full", "target"]
            if population == "oof":
                keys.append("fold")

            assert output.height == truth.height
            assert set(output["model_id"]) == {cluster_id}

            aligned = truth.join(output, on=keys, validate="1:1")
            assert aligned.height == truth.height
            # Mean of A/B offsets is 0.3; rejected models must not contribute.
            np.testing.assert_allclose(
                aligned["pred"].to_numpy(),
                aligned["truth"].to_numpy() + 0.3,
            )

            final = result["ensemble_validation_predictions"].filter(
                pl.col("target") == target
            )

            clusters = result["validation_predictions"].filter(
                pl.col("target") == target
            )
            fitted = result["stack"][target]

            expected_ids = sorted(
                ensemble_tables["validation_truth"]
                .filter(pl.col("target") == target)["inchikey_full"]
                .to_list()
            )

            assert set(final.columns) == {"inchikey_full", "target", "pred"}
            assert final.height == len(expected_ids)
            assert final["inchikey_full"].n_unique() == len(expected_ids)
            assert sorted(final["inchikey_full"].to_list()) == expected_ids

            expected = np.full(len(expected_ids), fitted.get("intercept", 0.0))

            for cluster_id, weight in fitted["weights"].items():
                cluster = clusters.filter(
                    pl.col("model_id") == cluster_id
                ).sort("inchikey_full")

                assert cluster["inchikey_full"].to_list() == expected_ids
                expected += weight * cluster["pred"].to_numpy()

            np.testing.assert_allclose(
                final.sort("inchikey_full")["pred"].to_numpy(),
                expected,
            )


def test_filtered_stack_requires_both_filters(ensemble_tables):
    result = ensemble.fit_filtered_stack(**ensemble_tables)
    decisions = result["filter_decisions"]

    expected = {
        "a": (True, True, True),
        "b": (True, True, True),
        "bad_quality": (False, True, False),
        "bad_ratio": (True, False, False),
    }

    assert decisions.height == 16
    for row in decisions.iter_rows(named=True):
        assert (
            row["quality_keep"],
            row["leakage_keep"],
            row["keep"],
        ) == expected[row["model_id"]]

    assert set(result["cluster_membership"]["model_id"]) == {"a", "b"}


def test_filtered_stack_rejects_empty_target(ensemble_tables, monkeypatch):
    target = ensemble_tables["validation_truth"]["target"][0]

    # Leave only failing candidates for one target.
    for name in ["validation_predictions", "oof_predictions"]:
        ensemble_tables[name] = ensemble_tables[name].filter(
            ~(
                (pl.col("target") == target)
                & pl.col("model_id").is_in(["a", "b"])
            )
        )

    def fail_if_called(*args, **kwargs):
        pytest.fail("Deduplication must not run without surviving models")

    monkeypatch.setattr(ensemble, "deduplication_filter", fail_if_called)

    with pytest.raises(ValueError, match=f"No surviving models for {target}"):
        ensemble.fit_filtered_stack(**ensemble_tables)


def test_filtered_stack_rejects_unknown_method(ensemble_tables, monkeypatch):
    def fail_if_called(*args, **kwargs):
        pytest.fail("Filtering must not run for an invalid method")

    monkeypatch.setattr(ensemble, "r_squared_filtering", fail_if_called)

    with pytest.raises(ValueError, match="Unknown stacking method"):
        ensemble.fit_filtered_stack(**ensemble_tables, method="unknown")


