import json
from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import models.run_ensemble as runner
from data_tools.prepare_ensemble import SCORED_TARGETS, prepare_ensemble_data
from models.stack import predict_stack


@pytest.fixture
def runner_setup(tmp_path, monkeypatch):
    data = prepare_ensemble_data(pl.DataFrame({
        "SMILES": [
            "C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O",
            "c1ccccc1", "C1CCCCC1", "CCN", "CCCl",
        ],
        "split": ["train"] * 4 + ["val", "test"],
        **{target: [5.0] * 6 for target in SCORED_TARGETS},
    }))
    folds = data.head(4).select("inchikey_full").with_columns(
        pl.Series("fold", [0, 0, 1, 1]),
        pl.Series("oof_group", ["g0", "g0", "g1", "g2"]),
    )

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"
    data.write_csv(data_path)
    folds.write_csv(folds_path)

    config = {
        "data_path": str(data_path),
        "output_dir": str(tmp_path / "run"),
        "folds": {"path": str(folds_path)},
        "seeds": [11, 22, 33],
        "stacker": "nnls",
        "models": [{
            "configuration_id": "rf",
            "model_name": "random_forest",
            "input_names": ["morgan"],
            "model_params": {"n_estimators": 2},
        }],
    }

    def fail_if_called(**kwargs):
        pytest.fail("Invalid inputs must be rejected before training")

    monkeypatch.setattr(runner, "run_oof_seeds", fail_if_called)
    monkeypatch.setattr(runner, "run_validation_seeds", fail_if_called)

    return config, folds


@pytest.mark.parametrize("case", ["empty", "duplicate"])
def test_runner_rejects_invalid_model_list(runner_setup, case):
    config, _ = runner_setup
    if case == "empty":
        config["models"] = []
    else:
        config["models"].append(deepcopy(config["models"][0]))

    with pytest.raises(ValueError):
        runner.run(config)


@pytest.mark.parametrize("case", ["group_across_folds", "block_across_groups"])
def test_runner_rejects_invalid_groups(runner_setup, case):
    config, folds = runner_setup
    groups = (
        ["g0", "g0", "g0", "g2"]  # g0 spans folds 0 and 1
        if case == "group_across_folds"
        else ["g0", "other", "g1", "g2"]  # stereo pair split into groups
    )
    folds.with_columns(
        pl.Series("oof_group", groups)
    ).write_csv(config["folds"]["path"])

    with pytest.raises(ValueError):
        runner.run(config)


def test_runner_preserves_existing_run(runner_setup):
    config, _ = runner_setup
    output = Path(config["output_dir"])
    output.mkdir()
    saved = output / "train.csv"
    saved.write_text("existing run\n")

    with pytest.raises((ValueError, FileExistsError)):
        runner.run(config)

    assert saved.read_text() == "existing run\n"


@pytest.mark.parametrize("method", ["nnls", "ridgecv", "elasticnetcv", "all_rejected"])
@pytest.mark.parametrize("saved_folds", [False, True])
def test_runner_saves_complete_run(tmp_path, monkeypatch, method, saved_folds):
    data = prepare_ensemble_data(pl.DataFrame({
        "SMILES": ["C" * (i + 1) for i in range(13)],
        "split": ["train"] * 6 + ["val"] * 6 + ["test"],
        **{
            target: [
                None if j == 0 and i == 11 else float(i % 6 + j)
                for i in range(13)
            ]
            for j, target in enumerate(SCORED_TARGETS)
        },
    }))
    data_path = tmp_path / "approved.csv"
    data.write_csv(data_path)
    fold_config = {"path": None, "strategy": "random", "n_folds": 3, "seed": 42}
    if saved_folds:
        path = tmp_path / "saved_folds.csv"
        runner.make_oof_folds(data, strategy="random", n_folds=3, seed=42).write_csv(path)
        fold_config["path"] = str(path)

    config = {
        "data_path": str(data_path),
        "output_dir": str(tmp_path / "run"),
        "folds": fold_config,
        "seeds": [11, 22, 33],
        "stacker": "nnls" if method == "all_rejected" else method,
        "models": [
            {"configuration_id": name, "model_name": "random_forest", "input_names": ["morgan"]}
            for name in (["bad"] if method == "all_rejected" else ["a", "b", "bad"])
        ],
    }
    original_config = deepcopy(config)
    calls = []

    def fake_seeds(population, kwargs):
        calls.append((population, kwargs["model_id"]))
        assert kwargs["seeds"] == [11, 22, 33]
        assert kwargs["group_column"] == "oof_group"
        assert kwargs["validation_seed"] == 42
        if population == "oof":
            frame = runner._load_oof_data(kwargs["data_path"], kwargs["folds_path"])
            assert set(frame["split"]) == {"train"}
            keys = ["inchikey_full", "fold"]
        else:
            assert set(kwargs["df"]["split"]) == {"train", "val", "test"}
            assert kwargs["df"].filter(pl.col("split") == "train")["oof_group"].null_count() == 0
            frame = kwargs["df"].filter(pl.col("split") == "val")
            keys = ["inchikey_full"]
        truth = frame.unpivot(
            on=SCORED_TARGETS, index=keys, variable_name="target", value_name="truth",
        ).drop_nulls("truth")
        offset = {"a": 0.2, "b": 0.4, "bad": 10.0}[kwargs["model_id"]]
        predictions = {
            seed: truth.with_columns(
                (pl.col("truth") + offset + shift).alias("pred"),
                pl.lit(f"{kwargs['model_id']}_seed{seed}").alias("model_id"),
            ).drop("truth").reverse()
            for seed, shift in zip(kwargs["seeds"], [-0.03, 0.0, 0.03])
        }
        return predictions, truth.reverse() if kwargs["model_id"] == "b" else truth

    monkeypatch.setattr(runner, "run_oof_seeds", lambda **kw: fake_seeds("oof", kw))
    monkeypatch.setattr(runner, "run_validation_seeds", lambda **kw: fake_seeds("validation", kw))
    if method == "all_rejected":
        with pytest.raises(ValueError, match="No surviving models"):
            runner.run(config)
        output = Path(config["output_dir"])
        decisions = pl.read_csv(output / "filter_decisions.csv")
        assert set(decisions["target"]) == set(SCORED_TARGETS)
        assert decisions.height == len(SCORED_TARGETS)
        assert not decisions["quality_keep"].any()
        assert decisions["leakage_keep"].all()
        assert not decisions["keep"].any()
        assert (decisions["r_squared"] < 0.3).all()
        for row in decisions.iter_rows(named=True):
            expected_ratio = 1.25 if row["target"] == SCORED_TARGETS[0] else 1.0
            assert row["rae_ratio"] == pytest.approx(expected_ratio)
        assert {"heldout_rae", "oof_rae"} <= set(decisions.columns)
        assert not (output / "stack.json").exists()
        return
    runner.run(config)

    output = Path(config["output_dir"])
    expected = {
        "config.json", "prepared.csv",
        "train.csv", "folds.csv", "oof_predictions.csv", "oof_truth.csv",
        "validation_predictions.csv", "validation_truth.csv", "filter_decisions.csv",
        "cluster_membership.csv", "clustered_oof_predictions.csv",
        "clustered_validation_predictions.csv", "ensemble_validation_predictions.csv",
        "stack.json", "validation_metrics.csv",
    }
    assert expected <= {path.name for path in output.iterdir()}
    assert config == original_config
    assert json.loads((output / "config.json").read_text()) == config
    assert_frame_equal(pl.read_csv(data_path), data)
    assert_frame_equal(pl.read_csv(output / "prepared.csv").select(data.columns), data)
    assert len(calls) == 6

    keys = ["inchikey_full", "target"]
    averaged = pl.read_csv(output / "validation_predictions.csv")
    truth = pl.read_csv(output / "validation_truth.csv")
    for model_id in ["a", "b", "bad"]:
        model_dir = output / model_id
        assert (model_dir / "oof_truth.csv").is_file()
        assert (model_dir / "validation_truth.csv").is_file()
        assert (model_dir / "averaged_oof.csv").is_file()
        assert_frame_equal(
            pl.read_csv(model_dir / "averaged_validation.csv").sort(keys),
            averaged.filter(pl.col("model_id") == model_id).sort(keys),
        )
        seed_tables = []
        for seed in config["seeds"]:
            assert (model_dir / "oof" / f"seed-{seed}" / "oof_predictions.csv").is_file()
            table = pl.read_csv(model_dir / "validation" / f"seed-{seed}" / "predictions.csv")
            assert set(table["model_id"]) == {f"{model_id}_seed{seed}"}
            seed_tables.append(table)
        seed_mean = pl.concat(seed_tables).group_by(keys).agg(pl.col("pred").mean())
        assert_frame_equal(
            seed_mean.sort(keys),
            averaged.filter(pl.col("model_id") == model_id).select(*keys, "pred").sort(keys),
        )

    decisions = pl.read_csv(output / "filter_decisions.csv")
    assert set(decisions.filter(pl.col("keep"))["model_id"]) == {"a", "b"}
    membership = pl.read_csv(output / "cluster_membership.csv")
    assert set(membership["model_id"]) == {"a", "b"}
    stack = json.loads((output / "stack.json").read_text())
    clusters = pl.read_csv(output / "clustered_validation_predictions.csv")
    final = pl.read_csv(output / "ensemble_validation_predictions.csv")
    assert_frame_equal(predict_stack(clusters, stack).sort(keys), final.sort(keys))
    assert_frame_equal(final.select(keys).sort(keys), truth.select(keys).sort(keys))
    for target, fitted in stack.items():
        assert set(fitted["weights"]) == set(
            membership.filter(pl.col("target") == target)["cluster_id"]
        )
        if method != "nnls":
            assert "intercept" in fitted and "alpha" in fitted
        if method == "elasticnetcv":
            assert "l1_ratio" in fitted

    metrics = pl.read_csv(output / "validation_metrics.csv")
    assert metrics.height == 4 * len(SCORED_TARGETS)
    assert set(metrics["evaluation"]) == {"development"}
    for row in metrics.iter_rows(named=True):
        assert row["n_rows"] == truth.filter(pl.col("target") == row["target"]).height
        if row["kind"] == "individual":
            assert row["MAE"] == pytest.approx({"a": 0.2, "b": 0.4, "bad": 10.0}[row["model_id"]])
        else:
            aligned = truth.filter(pl.col("target") == row["target"]).join(final, on=keys)
            expected_metrics = runner._compute_metrics(aligned["truth"], aligned["pred"])
            assert row["RMSE"] == pytest.approx(expected_metrics["RMSE"])

