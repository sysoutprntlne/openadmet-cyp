from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models.stack import fit_nnls_stack
import models.oof as oof
from polars.testing import assert_frame_equal

@pytest.fixture
def load_oof_fixture(monkeypatch: pytest.MonkeyPatch):
    """Create synthetic tables and register fake models."""

    molecules = [
        ("SYNTHETIC_A", 0),
        ("SYNTHETIC_B", 0), 
        ("SYNTHETIC_C", 1),
        ("SYNTHETIC_D", 1),
    ]

    model_predictions = {
        "model_a": [1.0, 0.0, 0.0, 1.0],
        "model_b": [0.0, 1.0, 0.0, 1.0],
        "model_c": [0.0, 0.0, 1.0, 1.0],
    }

    target_truth_values = {
        "CYP1A2_pIC50_direct_inhibition": [0.2, 0.8, 0.0, 1.0],
        "CYP2C9_pIC50_direct_inhibition": [0.4, None, 0.0, 1.0],
        "CYP2D6_pIC50_direct_inhibition": [0.6, 0.4, 0.0, 1.0],
        "CYP3A4_pIC50_direct_inhibition": [0.8, 0.2, 0.0, 1.0],
    }

    targets = list(target_truth_values)

    data = pl.DataFrame({
        "inchikey_full": [molecule for molecule, fold in molecules],
        "inchikey_block": [molecule for molecule, fold in molecules],
        "fold": [fold for molecule, fold in molecules],
        "SMILES": ["C", "CC", "CCC", "CCCC"],
        **target_truth_values,
    })

    truth_rows = []

    for target, values in target_truth_values.items():
        for (molecule, fold), value in zip(molecules, values, strict=True):
            if value is None:
                continue
            truth_rows.append({
                "inchikey_full": molecule,
                "fold": fold, 
                "target": target,
                "truth": value,
            })

    truth = pl.DataFrame(truth_rows)

    def fake_featurize(df, input_names, cache_dir):
        return np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64, 
        )

    monkeypatch.setattr(oof, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self, prediction_key):
            self.values = np.array(
                model_predictions[prediction_key], dtype=np.float64
            )
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            assert set(X[:, 0]).isdisjoint(self.X_train[:, 0])
            indices = X[:, 0].astype(int) - 1
            return self.values[indices]
    
    monkeypatch.setitem(oof.REGISTRY, "fake_model", FakeModel)

    return data, truth, targets, model_predictions, instances

def test_oof_to_stack(tmp_path: Path, load_oof_fixture):
    # First run, expected weights, unchanged resume.

    data, truth, targets, model_predictions, instances = load_oof_fixture

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    data.drop("fold").write_csv(data_path)
    data.select("inchikey_full", "fold").write_csv(folds_path)

    model_tables = []

    for model_id in model_predictions:
        predictions, actual_truth = oof.run_oof(
            data_path=data_path,
            folds_path=folds_path,
            model_name="fake_model",
            model_id=model_id,
            model_params={"prediction_key": model_id},
            input_names=["rdkit"],
            output_dir=tmp_path / model_id,
            cache_dir=tmp_path / "features",
        )
        model_tables.append(predictions)

        keys = ["inchikey_full", "fold", "target"]
        assert_frame_equal(
            actual_truth.sort(keys),
            truth.sort(keys),
        )

    predictions = pl.concat(model_tables)
    results = fit_nnls_stack(predictions, actual_truth)

    expected_weights = {
        targets[0]: [0.2, 0.8, 0.0],
        targets[1]: [0.4, 0.6, 0.0],
        targets[2]: [0.6, 0.4, 0.0],
        targets[3]: [0.8, 0.2, 0.0],
    }

    assert truth.height == 15
    assert predictions.height == 45  # Three models × 15 labels.
    assert len(instances) == 24  # Three models × two folds × four targets.
    assert set(results) == set(targets)
    for target, expected in expected_weights.items():
        actual = [
            results[target]["weights"][model_id]
            for model_id in ["model_a", "model_b", "model_c"]
        ]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        assert results[target]["ensemble_rmse"] < 1e-10

    instance_count = len(instances)
    resumed_tables = []

    for model_id in model_predictions:
        resumed, resumed_truth = oof.run_oof(
            data_path=data_path,
            folds_path=folds_path,
            model_name="fake_model",
            model_id=model_id,
            model_params={"prediction_key": model_id},
            input_names=["rdkit"],
            output_dir=tmp_path / model_id,
            cache_dir=tmp_path / "features",
        )
        resumed_tables.append(resumed)

        assert_frame_equal(
            resumed_truth.sort(keys),
            truth.sort(keys),
        )

    assert len(instances) == instance_count

    resumed_predictions = pl.concat(resumed_tables)
    assert_frame_equal(resumed_predictions, predictions)
    assert fit_nnls_stack(resumed_predictions, truth) == results


@pytest.mark.parametrize(
    "changed_input",
    ["model_params", "labels", "folds"],
)
def test_run_oof_invalidates_changed_inputs(
    tmp_path: Path,
    load_oof_fixture,
    changed_input: str,
) -> None:
    data, truth, targets, model_predictions, instances = load_oof_fixture

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    molecule_data = data.drop("fold")
    assignments = data.select("inchikey_full", "fold")

    molecule_data.write_csv(data_path)
    assignments.write_csv(folds_path)

    # Keep the model ID and output directory fixed across calls.
    run_args = {
        "data_path": data_path,
        "folds_path": folds_path,
        "model_name": "fake_model",
        "model_id": "model_a",
        "model_params": {"prediction_key": "model_a"},
        "input_names": ["rdkit"],
        "output_dir": tmp_path / "model_a",
        "cache_dir": tmp_path / "features",
    }

    original_predictions, original_truth = oof.run_oof(**run_args)

    assert len(instances) == 8  # Two folds × four targets.

    if changed_input == "model_params":
        run_args["model_params"] = {"prediction_key": "model_b"}

    elif changed_input == "labels":
        target = targets[0]
        molecule_data = molecule_data.with_columns(
            pl.when(pl.col("inchikey_full") == "SYNTHETIC_A")
            .then(pl.col(target) + 0.5)
            .otherwise(pl.col(target))
            .alias(target)
        )
        molecule_data.write_csv(data_path)

    else:
        # Swap A and C while retaining two valid folds.
        assignments = assignments.with_columns(
            pl.when(pl.col("inchikey_full") == "SYNTHETIC_A")
            .then(pl.lit(1))
            .when(pl.col("inchikey_full") == "SYNTHETIC_C")
            .then(pl.lit(0))
            .otherwise(pl.col("fold"))
            .alias("fold")
        )
        assignments.write_csv(folds_path)

    updated_predictions, updated_truth = oof.run_oof(**run_args)

    # The changed run fingerprint must invalidate both saved folds.
    assert len(instances) == 16

    prediction_keys = ["model_id", "inchikey_full", "fold", "target"]
    truth_keys = ["inchikey_full", "fold", "target"]

    if changed_input == "model_params":
        assert not updated_predictions.sort(prediction_keys).equals(
            original_predictions.sort(prediction_keys)
        )
        assert_frame_equal(
            updated_truth.sort(truth_keys),
            original_truth.sort(truth_keys),
        )
    else:
        assert not updated_truth.sort(truth_keys).equals(
            original_truth.sort(truth_keys)
        )

    # The updated run should now resume without further training.
    resumed_predictions, resumed_truth = oof.run_oof(**run_args)

    assert len(instances) == 16
    assert_frame_equal(
        resumed_predictions.sort(prediction_keys),
        updated_predictions.sort(prediction_keys),
    )
    assert_frame_equal(
        resumed_truth.sort(truth_keys),
        updated_truth.sort(truth_keys),
    )


def test_run_oof_recovers_corrupt_fold(tmp_path: Path, load_oof_fixture):
    data, truth, targets, model_predictions, instances = load_oof_fixture

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    molecule_data = data.drop("fold")
    assignments = data.select("inchikey_full", "fold")

    molecule_data.write_csv(data_path)
    assignments.write_csv(folds_path)

    # Keep the model ID and output directory fixed across calls.
    run_args = {
        "data_path": data_path,
        "folds_path": folds_path,
        "model_name": "fake_model",
        "model_id": "model_a",
        "model_params": {"prediction_key": "model_a"},
        "input_names": ["rdkit"],
        "output_dir": tmp_path / "model_a",
        "cache_dir": tmp_path / "features",
    }

    original_predictions, original_truth = oof.run_oof(**run_args)

    assert len(instances) == 8  

    # Damage only fold 0's saved predictions.
    corrupt_path = tmp_path / "model_a" / "fold-0" / "predictions.csv"
    corrupt_path.write_text("broken csv\n", encoding="utf-8")

    recovered_predictions, recovered_truth = oof.run_oof(**run_args)

    # Only one fold retrains: four additional target models.
    assert len(instances) == 12

    # Fold 0's training molecules are C and D (features 3 and 4).
    # Confirm the new models belong to that fold.
    for instance in instances[8:]:
        assert set(instance.X_train[:, 0]) == {3.0, 4.0}

    prediction_keys = ["model_id", "inchikey_full", "fold", "target"]
    truth_keys = ["inchikey_full", "fold", "target"]

    assert_frame_equal(
        recovered_predictions.sort(prediction_keys),
        original_predictions.sort(prediction_keys),
    )
    assert_frame_equal(
        recovered_truth.sort(truth_keys),
        original_truth.sort(truth_keys),
    )

    # Confirm repaired artifacts are reusable.
    oof.run_oof(**run_args)
    assert len(instances) == 12


def test_run_oof_saved_tables_match_returned_tables(
    tmp_path: Path,
    load_oof_fixture,
) -> None:
    data, truth, targets, model_predictions, instances = load_oof_fixture

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"
    output_dir = tmp_path / "model_a"

    data.drop("fold").write_csv(data_path)
    data.select("inchikey_full", "fold").write_csv(folds_path)

    predictions, actual_truth = oof.run_oof(
        data_path=data_path,
        folds_path=folds_path,
        model_name="fake_model",
        model_id="model_a",
        model_params={"prediction_key": "model_a"},
        input_names=["rdkit"],
        output_dir=output_dir,
        cache_dir=tmp_path / "features",
    )

    saved_predictions = pl.read_csv(output_dir / "oof_predictions.csv")
    saved_truth = pl.read_csv(output_dir / "truth.csv")

    prediction_keys = ["model_id", "inchikey_full", "fold", "target"]
    truth_keys = ["inchikey_full", "fold", "target"]

    assert_frame_equal(
        saved_predictions.sort(prediction_keys),
        predictions.sort(prediction_keys),
    )
    assert_frame_equal(
        saved_truth.sort(truth_keys),
        actual_truth.sort(truth_keys),
    )

    # Also check saved truth against the independent fixture.
    assert_frame_equal(
        saved_truth.sort(truth_keys),
        truth.sort(truth_keys),
    )
