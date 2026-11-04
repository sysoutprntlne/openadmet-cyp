"""Test OOF prediction generation, validation, saved-fold loading,
seed runs, checkpoint-aware resume, and CLI behavior.
"""

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import models.oof as oof
from models.oof import (
    _file_sha256,
    _generate_fold_predictions,
    _make_run_id,
    _run_fold_job,
    _validate_completion,
    _validate_inputs,
    _write_fold_artifacts,
    _load_oof_data,
    _convert_chemprop_predictions,
)


@pytest.mark.parametrize("use_alias", [False, True])
def test_cli_accepts_checkpoint_input(tmp_path, monkeypatch, use_alias):
    from representations.foundation_alias import set_alias

    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fake weights")
    set_alias(checkpoint, "test_encoder")
    name = f"chemprop[{'test_encoder' if use_alias else checkpoint}]"
    calls = []
    monkeypatch.setattr(oof, "run_oof", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", [
        "generate-oof", "--data-path", "data.csv", "--folds-path", "folds.csv",
        "--model-name", "decision_tree", "--model-id", "tree",
        "--input-names", name, "--output-dir", "run", "--cache-dir", "features",
    ])
    oof.main()
    assert len(calls) == 1
    assert calls[0]["input_names"] == [name]


@pytest.mark.parametrize("change", ["unchanged", "contents", "alias"])
def test_oof_checkpoint_resume(tmp_path, monkeypatch, loader_tables, change):
    from representations.foundation_alias import set_alias

    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights A")
    set_alias(checkpoint, "encoder")
    name = "chemprop[encoder]" if change == "alias" else f"chemprop[{checkpoint}]"
    data, folds = loader_tables
    data.write_csv(tmp_path / "data.csv")
    folds.write_csv(tmp_path / "folds.csv")
    calls = []

    class FakeModel:
        def fit(self, X, y):
            calls.append(len(y))

        def predict(self, X):
            return np.ones(len(X), dtype=np.float64)

    monkeypatch.setitem(oof.REGISTRY, "fake_checkpoint_model", FakeModel)
    monkeypatch.setattr(
        oof, "featurize",
        lambda df, input_names, cache_dir: np.ones((df.height, 2)),
    )
    kwargs = dict(
        data_path=tmp_path / "data.csv", folds_path=tmp_path / "folds.csv",
        model_name="fake_checkpoint_model", model_id="test_model",
        input_names=[name], output_dir=tmp_path / "run",
        cache_dir=tmp_path / "features",
    )
    first, first_truth = oof.run_oof(**kwargs)
    first_calls = len(calls)
    assert first_calls == folds["fold"].n_unique() * len(oof._SCORED_DEFAULT)
    completion_path = tmp_path / "run" / "fold-0" / "complete.json"
    first_run_id = json.loads(completion_path.read_text())["run_id"]

    if change == "contents":
        checkpoint.write_bytes(b"weights B")  # Same path and byte count.
    elif change == "alias":
        replacement = tmp_path / "replacement.pt"
        replacement.write_bytes(b"weights B")
        (tmp_path / "foundation-path-alias.csv").unlink()
        set_alias(replacement, "encoder")

    second, second_truth = oof.run_oof(**kwargs)
    second_run_id = json.loads(completion_path.read_text())["run_id"]
    assert len(calls) == first_calls * (1 if change == "unchanged" else 2)
    assert (first_run_id == second_run_id) == (change == "unchanged")
    assert_frame_equal(first, second)
    assert_frame_equal(first_truth, second_truth)


# OOF prediction-table validation
def test_validate_inputs_accepts_valid_fixture(oof_tables) -> None:
    predictions, truth = oof_tables

    assert _validate_inputs(
        truth=truth, 
        heldout_fold=0,
        predictions=predictions,
        model_id="model_a"
    ) is None


def test_shuffled_rows_preserve_results(oof_tables) -> None:
    predictions, truth = oof_tables

    shuffled_predictions = predictions.sample(
        fraction=1.0, shuffle=True, seed=42
    )
    shuffled_truth = truth.sample(
        fraction=1.0, shuffle=True, seed=17
    )

    assert _validate_inputs(
        truth=shuffled_truth,
        heldout_fold=0,
        predictions=shuffled_predictions,
        model_id="model_a",
    ) is None


def test_rejects_duplicate_predictions(oof_tables) -> None:
    predictions, truth = oof_tables

    duplicated = pl.concat([predictions, predictions.head(1)])

    with pytest.raises(ValueError, match="duplicates in prediction"):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=duplicated,
            model_id="model_a",
        ) 


def test_rejects_missing_prediction(oof_tables) -> None:
    predictions, truth = oof_tables

    incomplete = predictions.slice(1)

    with pytest.raises(ValueError, match="missing required prediction"):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=incomplete,
            model_id="model_a",
        )


def test_rejects_mismatched_folds(oof_tables) -> None:
    predictions, truth = oof_tables

    wrong_folds = predictions.with_columns(
        (1 - pl.col("fold")).alias("fold")
    )

    with pytest.raises(ValueError, match="missing required prediction"):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=wrong_folds,
            model_id="model_a",
        )


def test_rejects_duplicate_truth(oof_tables) -> None:
    predictions, truth = oof_tables

    duplicated_truth = pl.concat([truth, truth.head(1)])

    with pytest.raises(ValueError, match="duplicates in truth"):
        _validate_inputs(
            truth=duplicated_truth,
            heldout_fold=0,
            predictions=predictions,
            model_id="model_a",
        )


def test_rejects_extra_prediction(oof_tables) -> None:
    predictions, truth = oof_tables

    extra_row = predictions.head(1).with_columns(
        pl.lit("SYNTHETIC_EXTRA").alias("inchikey_full")
    )
    extended = pl.concat([predictions, extra_row])

    with pytest.raises(ValueError, match="unexpected prediction key"):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=extended,
            model_id="model_a",
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("column", ["pred", "truth"])
def test_rejects_nonfinite_values(oof_tables, column: str, bad_value: float) -> None:
    predictions, truth = oof_tables

    if column == "pred":
        predictions = predictions.with_columns(
            pl.lit(bad_value).alias("pred")
        )
    else:
        truth = truth.with_columns(
            pl.lit(bad_value).alias("truth")
        )

    with pytest.raises(ValueError, match="contains non-finite values"):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=predictions,
            model_id="model_a",
        )

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
def test_rejects_null_values(oof_tables, table_name: str, column: str) -> None:
    predictions, truth = oof_tables

    if table_name == "predictions":
        predictions = predictions.with_columns(
            pl.lit(None, dtype=predictions[column].dtype).alias(column)
        )
    else:
        truth = truth.with_columns(
            pl.lit(None, dtype=truth[column].dtype).alias(column)
        )

    with pytest.raises(ValueError, match=f"null values in {table_name}"):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=predictions,
            model_id="model_a",
        )


@pytest.mark.parametrize("table_name", ["predictions", "truth"])
def test_rejects_multiple_folds_per_molecule(oof_tables, table_name: str) -> None:
    predictions, truth = oof_tables

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
        message = "missing required prediction"
    else:
        conflicting_row = truth.filter(
            (pl.col("inchikey_full") == "SYNTHETIC_A")
            & (pl.col("target") == "CYP1A2_pIC50_direct_inhibition")
        ).with_columns(
            pl.lit(1, dtype=truth["fold"].dtype).alias("fold")
        )

        truth = pl.concat([truth, conflicting_row])
        message = "Truth molecules assigned to multiple folds"

    with pytest.raises(ValueError, match=message):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=predictions,
            model_id="model_a",
        )


def test_rejects_wrong_model_id(oof_tables) -> None:
    predictions, truth = oof_tables

    with pytest.raises(
        ValueError,
        match="Expected all predictions to belong to model 'model_b'",
    ):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=predictions,
            model_id="model_b",
        )


@pytest.mark.parametrize("bad_fold", [None, "0", 0.0, True])
def test_rejects_invalid_heldout_fold(oof_tables, bad_fold) -> None:
    predictions, truth = oof_tables

    with pytest.raises(ValueError, match="heldout_fold must be an integer"):
        _validate_inputs(
            truth=truth,
            heldout_fold=bad_fold,
            predictions=predictions,
            model_id="model_a"
        )


def test_rejects_absent_heldout_fold(oof_tables) -> None:
    predictions, truth = oof_tables

    with pytest.raises(ValueError, match="No truth rows for held-out fold 99"):
        _validate_inputs(
            truth=truth,
            heldout_fold=99,
            predictions=predictions,
            model_id="model_a",
        )


def test_rejects_predictions_belong_to_another_fold(oof_tables) -> None:
    predictions, truth = oof_tables

    with pytest.raises(ValueError, match="missing required prediction"):
        _validate_inputs(
            truth=truth,
            heldout_fold=1,
            predictions=predictions,
            model_id="model_a",
        )


def test_reject_scored_target_absent_from_requested_fold(oof_tables) -> None:
    predictions, truth = oof_tables

    target = "CYP1A2_pIC50_direct_inhibition"

    truth = truth.filter(
        ~((pl.col("fold") == 0) & (pl.col("target") == target))
    )
    predictions = predictions.filter(pl.col("target") != target)

    with pytest.raises(ValueError, match="Fold 0 is missing scored targets"):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=predictions,
            model_id="model_a",
        )


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Int64])
@pytest.mark.parametrize("column", ["pred", "truth"])
def test_reject_float32_and_integers(oof_tables, column, dtype) -> None:
    predictions, truth = oof_tables

    if column == "pred":
        predictions = predictions.with_columns(pl.col("pred").cast(dtype))
    else:
        truth = truth.with_columns(pl.col("truth").cast(dtype))

    with pytest.raises(ValueError, match=f"{column} must contain Float64 values"):
        _validate_inputs(
            truth=truth,
            heldout_fold=0,
            predictions=predictions,
            model_id="model_a",
        )


def test_accepts_molecule_lacking_one_target_label(oof_tables) -> None:
    predictions, truth = oof_tables

    missing_pair = (
        (pl.col("inchikey_full") == "SYNTHETIC_A")
        & (pl.col("target") == "CYP1A2_pIC50_direct_inhibition")
    )

    truth = truth.filter(~missing_pair)
    predictions = predictions.filter(~missing_pair)

    assert _validate_inputs(
        truth=truth,
        heldout_fold=0,
        predictions=predictions,
        model_id="model_a",
    ) is None


# Run identity and file hashing
def test_repeatability() -> None:
    first = {"seed": 42, "model": {"name": "ridge", "alpha": 1.0}}

    assert _make_run_id(first) == _make_run_id(first)

def test_run_id_ignores_dictionary_order() -> None:
    first = {"seed": 42, "model": {"name": "ridge", "alpha": 1.0}}
    reordered = {"model": {"alpha": 1.0, "name": "ridge"}, "seed": 42}

    assert _make_run_id(first) == _make_run_id(reordered)

@pytest.mark.parametrize(
        "field, replacement",
        [
            ("seed", 37),
            ("data_fingerprint", "data-b"),
            ("fold_assignment_fingerprint", "folds-b"),
            ("model", {"name": "ridge", "parameters": {"alpha": 2.0}}),
        ]
)
def test_changing_config_changes_id(field, replacement) -> None:
    original = {
        "seed": 42, 
        "data_fingerprint": "data-a",
        "fold_assignment_fingerprint":"folds-a",
        "model": {"name": "ridge", "parameters": {"alpha": 1.0}},
    }
    changed = deepcopy(original)
    changed[field] = replacement

    assert _make_run_id(original) != _make_run_id(changed)


@pytest.mark.parametrize(
        "bad_value", [float("nan"), float("inf"), float("-inf")]
)
def test_rejects_invalid_values(bad_value: float) -> None:
    config = {
        "seed": 42, 
        "model": {"name": "ridge", "alpha": bad_value},
    } 

    with pytest.raises(ValueError, match = "Out of range float values"):
        _make_run_id(config)


def test_file_hash_ignores_filename(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    first.write_bytes(b"molecule,value\nA,6.2\n")
    second.write_bytes(b"molecule,value\nA,6.2\n")

    assert _file_sha256(first) == _file_sha256(second)


def test_file_hash_changes_with_contents(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_bytes(b"molecule,value\nA,6.2\n")
    original_hash = _file_sha256(path)

    path.write_bytes(b"molecule,value\nA,7.1\n")

    assert _file_sha256(path) != original_hash


def test_file_hash_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _file_sha256(tmp_path / "missing.csv")


# Completion-artifact validation
def test_accepts_valid_path(oof_tables, tmp_path: Path) -> None:
    predictions, truth = oof_tables
    prediction_path = tmp_path / "predictions.csv"
    predictions.write_csv(prediction_path)

    run_id = _make_run_id({"seed": 42})
    completion = {
        "contract_version": 1, 
        "run_id": run_id,
        "model_id": "model_a",
        "fold": 0,
        "prediction_sha256": _file_sha256(prediction_path),
        "n_predictions": predictions.height,
    }

    completion_path = tmp_path / "complete.json"
    completion_path.write_text(
        json.dumps(completion, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is True


@pytest.mark.parametrize("missing_file", ["complete.json", "predictions.csv"])
def test_rejects_missing_files(oof_tables, tmp_path: Path, missing_file: str) -> None:
    predictions, truth = oof_tables
    prediction_path = tmp_path / "predictions.csv"
    predictions.write_csv(prediction_path)

    run_id = _make_run_id({"seed": 42})
    completion = {
        "contract_version": 1, 
        "run_id": run_id,
        "model_id": "model_a",
        "fold": 0,
        "prediction_sha256": _file_sha256(prediction_path),
        "n_predictions": predictions.height,
    }

    completion_path = tmp_path / "complete.json"
    completion_path.write_text(
        json.dumps(completion, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8"
    )

    (tmp_path / missing_file).unlink()

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is False

@pytest.mark.parametrize(
    "filename, contents",
    [
        ("complete.json", '{"run_id":'),  # Incomplete JSON
        ("predictions.csv", ""),         # Empty CSV
    ],
)
def test_rejects_malformed_files(oof_tables, tmp_path: Path, filename, contents) -> None:
    predictions, truth = oof_tables
    prediction_path = tmp_path / "predictions.csv"
    predictions.write_csv(prediction_path)
    
    run_id = _make_run_id({"seed": 42})
    completion = {
        "contract_version": 1, 
        "run_id": run_id,
        "model_id": "model_a",
        "fold": 0,
        "prediction_sha256": _file_sha256(prediction_path),
        "n_predictions": predictions.height,
    }
        
    completion_path = tmp_path / "complete.json"
    completion_path.write_text(
        json.dumps(completion, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    
    (tmp_path / filename).write_text(contents, encoding="utf-8")
    
    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is False

@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("contract_version", 999),
        ("contract_version", True),
        ("run_id", "wrong-run"),
        ("model_id", "wrong-model"),
        ("fold", 1),
        ("fold", False),
        ("n_predictions", 999),
        ("n_predictions", "8"),
    ],
)
def test_rejects_wrong_metadata(oof_tables, tmp_path: Path, field, bad_value) -> None:
    predictions, truth = oof_tables
    prediction_path = tmp_path / "predictions.csv"
    predictions.write_csv(prediction_path)
    
    run_id = _make_run_id({"seed": 42})
    completion = {
        "contract_version": 1, 
        "run_id": run_id,
        "model_id": "model_a",
        "fold": 0,
        "prediction_sha256": _file_sha256(prediction_path),
        "n_predictions": predictions.height,
    }
        
    completion_path = tmp_path / "complete.json"
    completion_path.write_text(
        json.dumps(completion, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    
    completion[field] = bad_value
    (tmp_path / "complete.json").write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    
    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is False


def test_rejects_checksum_mismatches(oof_tables, tmp_path: Path) -> None:
    predictions, truth = oof_tables
    prediction_path = tmp_path / "predictions.csv"
    predictions.write_csv(prediction_path)
    
    run_id = _make_run_id({"seed": 42})
    completion = {
        "contract_version": 1, 
        "run_id": run_id,
        "model_id": "model_a",
        "fold": 0,
        "prediction_sha256": _file_sha256(prediction_path),
        "n_predictions": predictions.height,
    }
        
    completion_path = tmp_path / "complete.json"
    completion_path.write_text(
        json.dumps(completion, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8"
    )
    
    changed = predictions.with_columns(
        (pl.col("pred") + 0.1).alias("pred")
    )
    changed.write_csv(tmp_path / "predictions.csv")
    
    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is False


def test_rejects_invalid_prediction_keys(oof_tables, tmp_path: Path) -> None:
    predictions, truth = oof_tables
    prediction_path = tmp_path / "predictions.csv"
    invalid_predictions = predictions.with_columns(
        pl.when(pl.col("inchikey_full") == "SYNTHETIC_A")
        .then(pl.lit("SYNTHETIC_UNKNOWN"))
        .otherwise(pl.col("inchikey_full"))
        .alias("inchikey_full")
    )
    invalid_predictions.write_csv(prediction_path)
    
    run_id = _make_run_id({"seed": 42})
    completion = {
        "contract_version": 1, 
        "run_id": run_id,
        "model_id": "model_a",
        "fold": 0,
        "prediction_sha256": _file_sha256(prediction_path),
        "n_predictions": predictions.height,
    }
        
    completion_path = tmp_path / "complete.json"
    completion_path.write_text(
        json.dumps(completion),
        encoding="utf-8",
    )
    
    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is False


# Atomic artifact writing
def test_write_fold_artifacts_creates_valid_completion(oof_tables, tmp_path: Path) -> None:
    predictions, truth = oof_tables
    run_id = _make_run_id({"seed": 42})

    _write_fold_artifacts(
        job_dir=tmp_path,
        predictions=predictions,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0
    )

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is True

    assert not (tmp_path / "predictions.tmp.csv").exists()
    assert not (tmp_path / "complete.tmp.json").exists()


def test_write_fold_artifacts_rejects_invalid_predictions(oof_tables, tmp_path: Path) -> None:
    predictions, truth = oof_tables
    run_id = _make_run_id({"seed": 42})

    incomplete = predictions.slice(1)

    with pytest.raises(ValueError, match="missing required prediction"):
        _write_fold_artifacts(
            job_dir=tmp_path,
            predictions=incomplete,
            truth=truth,
            run_id=run_id,
            model_id="model_a",
            heldout_fold=0,
        )

    assert not (tmp_path / "predictions.csv").exists()
    assert not (tmp_path / "complete.json").exists()
    

def test_failure_before_prediction_rename_leaves_job_incomplete(
    oof_tables,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = Path.replace

    def fail_prediction_rename(self, target):
        if self.name == "predictions.tmp.csv":
            raise OSError("Simulated interruption")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_prediction_rename)

    predictions, truth = oof_tables
    run_id = _make_run_id({"seed": 42})

    with pytest.raises(OSError, match="Simulated interruption"):
        _write_fold_artifacts(
            job_dir=tmp_path,
            predictions=predictions,
            truth=truth,
            run_id=run_id,
            model_id="model_a",
            heldout_fold=0,
        )

    assert (tmp_path / "predictions.tmp.csv").exists()
    assert not (tmp_path / "predictions.csv").exists()
    assert not (tmp_path / "complete.json").exists()

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is False


def test_failure_before_completion_rename_leaves_job_incomplete(
    oof_tables,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = Path.replace

    def fail_completion_rename(self, target):
        if self.name == "complete.tmp.json":
            raise OSError("Simulated interruption")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_completion_rename)

    predictions, truth = oof_tables
    run_id = _make_run_id({"seed": 42})

    with pytest.raises(OSError, match="Simulated interruption"):
        _write_fold_artifacts(
            job_dir=tmp_path,
            predictions=predictions,
            truth=truth,
            run_id=run_id,
            model_id="model_a",
            heldout_fold=0,
        )

    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "complete.tmp.json").exists()
    assert not (tmp_path / "complete.json").exists()
    
    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is False


def test_recovers_after_completion_rename_failure(
    oof_tables,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_replace = Path.replace

    def fail_completion(self, target):
        if self.name == "complete.tmp.json":
            raise OSError("Simulated interruption")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_completion)

    predictions, truth = oof_tables
    run_id = _make_run_id({"seed": 42})

    with pytest.raises(OSError, match="Simulated interruption"):
        _write_fold_artifacts(
            job_dir=tmp_path,
            predictions=predictions,
            truth=truth,
            run_id=run_id,
            model_id="model_a",
            heldout_fold=0,
        )

    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "complete.tmp.json").exists()
    assert not (tmp_path / "complete.json").exists()
    
    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is False

    monkeypatch.undo()

    _write_fold_artifacts(
        job_dir=tmp_path,
        predictions=predictions,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    )

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
    ) is True

    assert not (tmp_path / "predictions.tmp.csv").exists()
    assert not (tmp_path / "complete.tmp.json").exists()


# Resumable fold jobs
def test_run_fold_job_generates_and_saves_new_job(oof_tables, tmp_path: Path) -> None:
    predictions, truth = oof_tables
    run_id = _make_run_id({"seed": 42})
    calls = []

    def fake_predictor() -> pl.DataFrame:
        calls.append("call")
        return predictions

    assert _run_fold_job(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
        generate_predictions=fake_predictor,

    ) is True

    assert calls == ["call"]

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0
    ) is True


def test_run_fold_job_skips_completed_job(oof_tables, tmp_path: Path) -> None:
    predictions, truth = oof_tables
    run_id = _make_run_id({"seed": 42})
    calls = []

    def fake_predictor() -> pl.DataFrame:
        calls.append("call")
        return predictions

    assert _run_fold_job(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
        generate_predictions=fake_predictor,

    ) is True

    assert calls == ["call"]

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0
    ) is True

    assert _run_fold_job(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
        generate_predictions=fake_predictor,

    ) is False

    assert calls == ["call"]

    
def test_run_fold_job_regenerates_corrupt_output(oof_tables, tmp_path: Path) -> None:
    predictions, truth = oof_tables
    run_id = _make_run_id({"seed": 42})
    calls = []

    def fake_predictor() -> pl.DataFrame:
        calls.append("call")
        return predictions

    assert _run_fold_job(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
        generate_predictions=fake_predictor,

    ) is True

    assert calls == ["call"]

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0
    ) is True

    (tmp_path / "predictions.csv").write_text("", encoding="utf-8")

    assert _run_fold_job(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
        generate_predictions=fake_predictor,

    ) is True

    assert calls == ["call", "call"]

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0
    ) is True

    
def test_run_fold_job_retries_incomplete_job(oof_tables, tmp_path: Path) -> None:
    predictions, truth = oof_tables
    run_id = _make_run_id({"seed": 42})
    calls = []

    def fake_predictor() -> pl.DataFrame:
        calls.append("call")
        return predictions

    assert _run_fold_job(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
        generate_predictions=fake_predictor,

    ) is True

    assert calls == ["call"]

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0
    ) is True

    (tmp_path / "complete.json").unlink()

    assert _run_fold_job(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
        generate_predictions=fake_predictor,

    ) is True

    assert calls == ["call", "call"]

    assert _validate_completion(
        job_dir=tmp_path,
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0
    ) is True


def test_run_fold_jobs_skip_completed_folds_and_run_incomplete_folds(
    oof_tables,
    tmp_path: Path,
) -> None:
    _, truth = oof_tables
    run_id = _make_run_id({"seed": 42})
    calls = []

    def predictor_for(fold: int):
        def fake_predictor() -> pl.DataFrame:
            calls.append(fold)
            return (
                truth.filter(pl.col("fold") == fold)
                .rename({"truth": "pred"})
                .with_columns(pl.lit("model_a").alias("model_id"))
            )

        return fake_predictor

    # Finish fold 0
    assert _run_fold_job(
        job_dir=tmp_path / "fold-0",
        truth=truth,
        run_id=run_id,
        model_id="model_a",
        heldout_fold=0,
        generate_predictions=predictor_for(0),
    ) is True

    assert calls == [0]

    # Leave fold 1 with an unfinished temporary file
    fold1_dir = tmp_path / "fold-1"
    fold1_dir.mkdir()
    (fold1_dir / "predictions.tmp.csv").write_text(
        "unfinished", encoding="utf-8"
    )

    # Attempt both folds again.
    results = []
    for fold in [0, 1]:
        results.append(
            _run_fold_job(
                job_dir=tmp_path / f"fold-{fold}",
                truth=truth,
                run_id=run_id,
                model_id="model_a",
                heldout_fold=fold,
                generate_predictions=predictor_for(fold),
            )
        )

    assert results == [False, True]
    assert calls == [0, 1]

    for fold in [0, 1]:
        assert _validate_completion(
            job_dir=tmp_path / f"fold-{fold}",
            truth=truth,
            run_id=run_id,
            model_id="model_a",
            heldout_fold=fold,
        ) is True


# Per-fold prediction generation
@pytest.mark.parametrize("shared_block", [False, True])
def test_generate_fold_predictions_valid_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared_block: bool,
) -> None:
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    train_df = pl.DataFrame({
        "inchikey_full": ["TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full"],
        "inchikey_block": ["TRAIN_A", "TRAIN_B", "TRAIN_C"],
        "SMILES": ["C", "CC", "CCC"],
        **{target: [5.0, 6.0, 7.0] for target in targets}
    })

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],
        "inchikey_block": ["HELDOUT_A", "HELDOUT_B"],
        "SMILES": ["CCCC", "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    if shared_block:
        heldout_df = heldout_df.with_columns(
            pl.lit("HELDOUT_SHARED").alias("inchikey_block")
        )

    def fake_featurize(df, input_names, cache_dir):
        return np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

    monkeypatch.setattr(oof, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            return X[:, 0] # Return the fake SMILES-length feature.

    monkeypatch.setitem(oof.REGISTRY, "fake_model", FakeModel)

    result = _generate_fold_predictions(
        train_df=train_df,
        heldout_df=heldout_df,
        model_name="fake_model",
        model_id="fake_model_test_seed42",
        model_params={},
        input_names=["rdkit"],
        targets=targets,
        heldout_fold=0,
        cache_dir=tmp_path / "features",
    )

    assert result.height == 8
    assert set(result.columns) == {
        "model_id", "inchikey_full", "fold", "target", "pred"
    }
    assert result["model_id"].unique().to_list() == ["fake_model_test_seed42"]
    assert result["fold"].unique().to_list() == [0]
    assert result["pred"].dtype == pl.Float64

    expected_keys = {
        (molecule, 0, target)
        for molecule in ["HELDOUT_A-full", "HELDOUT_B-full"]
        for target in targets
    }
    actual_keys = set(
        result.select("inchikey_full", "fold", "target").iter_rows()
    )
    assert actual_keys == expected_keys

    for target in targets:
        rows = result.filter(pl.col("target") == target).sort("inchikey_full")
        np.testing.assert_array_equal(rows["pred"].to_numpy(), [4.0, 5.0])

    # Four distinct models, each fitted on training rows only.
    assert len(instances) == 4
    for model in instances:
        np.testing.assert_array_equal(model.X_train, [[1.0], [2.0], [3.0]])
        np.testing.assert_array_equal(model.y_train, [5.0, 6.0, 7.0])


def test_generate_fold_predictions_omits_missing_heldout_labels(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch
) -> None:
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    train_df = pl.DataFrame({
        "inchikey_full": ["TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full"],
        "inchikey_block": ["TRAIN_A", "TRAIN_B", "TRAIN_C"],
        "SMILES": ["C", "CC", "CCC"],
        **{target: [5.0, 6.0, 7.0] for target in targets}
    })

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],
        "inchikey_block": ["HELDOUT_A", "HELDOUT_B"],
        "SMILES": ["CCCC", "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    heldout_df = heldout_df.with_columns(
        pl.when(pl.col("inchikey_full") == "HELDOUT_A-full")
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(targets[0]))
        .alias(targets[0])
    )

    def fake_featurize(df, input_names, cache_dir):
        return np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

    monkeypatch.setattr(oof, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            return X[:, 0] # Return the fake SMILES-length feature.

    monkeypatch.setitem(oof.REGISTRY, "fake_model", FakeModel)

    result = _generate_fold_predictions(
        train_df=train_df,
        heldout_df=heldout_df,
        model_name="fake_model",
        model_id="fake_model_test_seed42",
        model_params={},
        input_names=["rdkit"],
        targets=targets,
        heldout_fold=0,
        cache_dir=tmp_path / "features",
    )

    assert result.height == 7
    assert set(result.columns) == {
        "model_id", "inchikey_full", "fold", "target", "pred"
    }
    assert result["model_id"].unique().to_list() == ["fake_model_test_seed42"]
    assert result["fold"].unique().to_list() == [0]
    assert result["pred"].dtype == pl.Float64

    expected_keys = {
        (molecule, 0, target)
        for molecule in ["HELDOUT_A-full", "HELDOUT_B-full"]
        for target in targets
    }
    expected_keys.remove(("HELDOUT_A-full", 0, targets[0]))

    actual_keys = set(
        result.select("inchikey_full", "fold", "target").iter_rows()
    )
    assert actual_keys == expected_keys

    for target in targets:
        rows = result.filter(pl.col("target") == target).sort("inchikey_full")
        expected = [5.0] if target == targets[0] else [4.0, 5.0]
        np.testing.assert_array_equal(rows["pred"].to_numpy(), expected)

    # Four distinct models, each fitted on training rows only.
    assert len(instances) == 4
    for model in instances:
        np.testing.assert_array_equal(model.X_train, [[1.0], [2.0], [3.0]])
        np.testing.assert_array_equal(model.y_train, [5.0, 6.0, 7.0])


def test_generate_fold_predictions_excludes_missing_training_labels(
    tmp_path: Path, 
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    train_df = pl.DataFrame({
        "inchikey_full": ["TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full"],
        "inchikey_block": ["TRAIN_A", "TRAIN_B", "TRAIN_C"],
        "SMILES": ["C", "CC", "CCC"],
        **{target: [5.0, 6.0, 7.0] for target in targets}
    })

    train_df = train_df.with_columns(
        pl.when(pl.col("inchikey_block") == "TRAIN_A")
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(targets[0]))
        .alias(targets[0])
    )

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],
        "inchikey_block": ["HELDOUT_A", "HELDOUT_B"],
        "SMILES": ["CCCC", "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    def fake_featurize(df, input_names, cache_dir):
        return np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

    monkeypatch.setattr(oof, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            return X[:, 0] # Return the fake SMILES-length feature.

    monkeypatch.setitem(oof.REGISTRY, "fake_model", FakeModel)

    result = _generate_fold_predictions(
        train_df=train_df,
        heldout_df=heldout_df,
        model_name="fake_model",
        model_id="fake_model_test_seed42",
        model_params={},
        input_names=["rdkit"],
        targets=targets,
        heldout_fold=0,
        cache_dir=tmp_path / "features",
    )

    assert result.height == 8
    assert set(result.columns) == {
        "model_id", "inchikey_full", "fold", "target", "pred"
    }
    assert result["model_id"].unique().to_list() == ["fake_model_test_seed42"]
    assert result["fold"].unique().to_list() == [0]
    assert result["pred"].dtype == pl.Float64

    expected_keys = {
        (molecule, 0, target)
        for molecule in ["HELDOUT_A-full", "HELDOUT_B-full"]
        for target in targets
    }

    actual_keys = set(
        result.select("inchikey_full", "fold", "target").iter_rows()
    )
    assert actual_keys == expected_keys

    for target in targets:
        rows = result.filter(pl.col("target") == target).sort("inchikey_full")
        np.testing.assert_array_equal(rows["pred"].to_numpy(), [4.0, 5.0])

    assert len(instances) == 4

    for target, model in zip(targets, instances, strict=True):
        if target == targets[0]:
            expected_X = [[2.0], [3.0]]
            expected_y = [6.0, 7.0]
        else:
            expected_X = [[1.0], [2.0], [3.0]]
            expected_y = [5.0, 6.0, 7.0]

        np.testing.assert_array_equal(model.X_train, expected_X)
        np.testing.assert_array_equal(model.y_train, expected_y)


def test_generate_fold_predictions_rejects_train_heldout_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
    ]

    train_df = pl.DataFrame({
        "inchikey_full": ["TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full"],
        "inchikey_block": ["TRAIN_A", "TRAIN_B", "TRAIN_C"],
        "SMILES": ["C", "CC", "CCC"],
        **{target: [5.0, 6.0, 7.0] for target in targets}
    })

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],
        "inchikey_block": ["TRAIN_C", "HELDOUT_B"],
        "SMILES": ["CCC", "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    def fake_featurize(df, input_names, cache_dir):
        return np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

    monkeypatch.setattr(oof, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            return X[:, 0] # Return the fake SMILES-length feature.

    monkeypatch.setitem(oof.REGISTRY, "fake_model", FakeModel)

    with pytest.raises(ValueError, match="Training and held-out molecules overlap"):
        _generate_fold_predictions(
            train_df=train_df,
            heldout_df=heldout_df,
            model_name="fake_model",
            model_id="fake_model_test_seed42",
            model_params={},
            input_names=["rdkit"],
            targets=targets,
            heldout_fold=0,
            cache_dir=tmp_path / "features",
        )

    assert instances == []


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_generate_fold_predictions_rejects_nonfinite_heldout_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_value,
) -> None:
    targets = [
    "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4_pIC50_direct_inhibition",
    ]

    train_df = pl.DataFrame({
        "inchikey_full": ["TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full"],
        "inchikey_block": ["TRAIN_A", "TRAIN_B", "TRAIN_C"],
        "SMILES": ["C", "CC", "CCC"],
        **{target: [5.0, 6.0, 7.0] for target in targets}
    })

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],
        "inchikey_block": ["HELDOUT_A", "HELDOUT_B"],
        "SMILES": ["CCCC", "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    def fake_featurize(df, input_names, cache_dir):
        features = np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

        for row, molecule in enumerate(df["inchikey_block"]):
            if molecule == "HELDOUT_A":
                features[row, 0] = bad_value

        return features

    monkeypatch.setattr(oof, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            return X[:, 0] # Return the fake SMILES-length feature.

    monkeypatch.setitem(oof.REGISTRY, "fake_model", FakeModel)

    with pytest.raises(ValueError, match="Non-finite held-out features"):
        _generate_fold_predictions(
            train_df=train_df,
            heldout_df=heldout_df,
            model_name="fake_model",
            model_id="fake_model_test_seed42",
            model_params={},
            input_names=["rdkit"],
            targets=targets,
            heldout_fold=0,
            cache_dir=tmp_path / "features",
        )
    
@pytest.mark.parametrize("bad_output", [
    np.array(4.0),          # Scalar
    np.array([[4.0], [5.0]]),  # 2D instead of 1D
])
def test_generate_fold_predictions_rejects_invalid_prediction_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_output,
) -> None:
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
        ]

    train_df = pl.DataFrame({
        "inchikey_full": ["TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full"],
        "inchikey_block": ["TRAIN_A", "TRAIN_B", "TRAIN_C"],
        "SMILES": ["C", "CC", "CCC"],
        **{target: [5.0, 6.0, 7.0] for target in targets}
    })

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],
        "inchikey_block": ["HELDOUT_A", "HELDOUT_B"],
        "SMILES": ["CCCC", "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    def fake_featurize(df, input_names, cache_dir):
        features = np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )
        return features

    monkeypatch.setattr(oof, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return bad_output

    monkeypatch.setitem(oof.REGISTRY, "fake_model", FakeModel)

    with pytest.raises(ValueError, match="must be one-dimensional"):
        _generate_fold_predictions(
            train_df=train_df,
            heldout_df=heldout_df,
            model_name="fake_model",
            model_id="fake_model_test_seed42",
            model_params={},
            input_names=["rdkit"],
            targets=targets,
            heldout_fold=0,
            cache_dir=tmp_path / "features",
        )

@pytest.mark.parametrize("bad_output", [
    np.array([4.0]),
    np.array([4.0, 5.0, 6.0]),
])
def test_generate_fold_predictions_rejects_wrong_prediction_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_output,
) -> None:
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    train_df = pl.DataFrame({
        "inchikey_full": ["TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full"],
        "inchikey_block": ["TRAIN_A", "TRAIN_B", "TRAIN_C"],
        "SMILES": ["C", "CC", "CCC"],
        **{target: [5.0, 6.0, 7.0] for target in targets}
    })

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],
        "inchikey_block": ["HELDOUT_A", "HELDOUT_B"],
        "SMILES": ["CCCC", "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    def fake_featurize(df, input_names, cache_dir):
        features = np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )
        return features

    monkeypatch.setattr(oof, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return bad_output

    monkeypatch.setitem(oof.REGISTRY, "fake_model", FakeModel)

    with pytest.raises(ValueError, match="Prediction count does not match held-out rows"):
        _generate_fold_predictions(
            train_df=train_df,
            heldout_df=heldout_df,
            model_name="fake_model",
            model_id="fake_model_test_seed42",
            model_params={},
            input_names=["rdkit"],
            targets=targets,
            heldout_fold=0,
            cache_dir=tmp_path / "features",
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_generate_fold_predictions_rejects_nonfinite_predictions(
    tmp_path: Path, 
    monkeypatch: pytest.MonkeyPatch,
    bad_value,
) -> None:
    targets = [
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ]

    train_df = pl.DataFrame({
        "inchikey_full": ["TRAIN_A-full", "TRAIN_B-full", "TRAIN_C-full"],
        "inchikey_block": ["TRAIN_A", "TRAIN_B", "TRAIN_C"],
        "SMILES": ["C", "CC", "CCC"],
        **{target: [5.0, 6.0, 7.0] for target in targets}
    })

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],
        "inchikey_block": ["HELDOUT_A", "HELDOUT_B"],
        "SMILES": ["CCCC", "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    def fake_featurize(df, input_names, cache_dir):
        features = np.array(
            [[float(len(smiles))] for smiles in df["SMILES"]],
            dtype=np.float64,
        )

        return features

    monkeypatch.setattr(oof, "featurize", fake_featurize)

    instances = []

    class FakeModel:
        def __init__(self):
            instances.append(self)

        def fit(self, X, y):
            self.X_train = X.copy()
            self.y_train = y.copy()

        def predict(self, X):
            return np.array([4.0, bad_value], dtype=np.float64)

    monkeypatch.setitem(oof.REGISTRY, "fake_model", FakeModel)

    with pytest.raises(ValueError, match="contain non-finite values"):
        _generate_fold_predictions(
            train_df=train_df,
            heldout_df=heldout_df,
            model_name="fake_model",
            model_id="fake_model_test_seed42",
            model_params={},
            input_names=["rdkit"],
            targets=targets,
            heldout_fold=0,
            cache_dir=tmp_path / "features",
        )


@pytest.fixture
def loader_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    data = pl.DataFrame({
        "inchikey_full": ["A-full", "B-full", "C-full", "D-full"],
        "inchikey_block": ["A", "B", "C", "D"],
        "SMILES": ["C", "CC", "CCC", "CCCC"],
        "CYP1A2_pIC50_direct_inhibition": [5.0, None, 6.0, 7.0],
        "CYP2C9_pIC50_direct_inhibition": [5.0, 6.0, 7.0, 8.0],
        "CYP2D6_pIC50_direct_inhibition": [4.0, 5.0, 6.0, 7.0],
        "CYP3A4_pIC50_direct_inhibition": [6.0, 7.0, 8.0, 9.0],
    })

    folds = pl.DataFrame({
        "inchikey_full": ["A-full", "B-full", "C-full", "D-full"],
        "fold": [0, 0, 1, 1], 
    })

    return data, folds


def test_load_oof_data_accepts_valid_inputs(tmp_path, loader_tables):
    data, folds = loader_tables

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"
    data.write_csv(data_path)
    folds.write_csv(folds_path)

    result = _load_oof_data(data_path, folds_path)

    assert result.height == 4
    assert result["fold"].null_count() == 0


def test_load_oof_data_joins_by_id_with_shuffled_assignments(tmp_path: Path, loader_tables) -> None:
    data, folds = loader_tables
    
    shuffled_data = data.sample(
        fraction=1.0, shuffle=True, seed=42,
    )
    shuffled_folds = folds.sample(
        fraction=1.0, shuffle=True, seed=17,
    )

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    shuffled_data.write_csv(data_path)
    shuffled_folds.write_csv(folds_path)

    result = _load_oof_data(data_path, folds_path)

    assert dict(result.select("inchikey_full", "fold").iter_rows()) == {
        "A-full": 0, "B-full": 0, "C-full": 1, "D-full": 1,
    }


def test_load_oof_data_preserves_missing_target_labels(tmp_path: Path, loader_tables) -> None:
    data, folds = loader_tables

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"
    data.write_csv(data_path)
    folds.write_csv(folds_path)

    result = _load_oof_data(data_path, folds_path)

    target = "CYP1A2_pIC50_direct_inhibition"
    assert result[target].null_count() == 1
    assert result.filter(pl.col("inchikey_block") == "B")[target].item() is None
    assert result.height == 4


def test_load_oof_data_rejects_missing_required_columns(tmp_path: Path, loader_tables):
    data, folds = loader_tables
            
    data = data.drop("inchikey_block")

    data.write_csv(tmp_path / "data.csv")
    folds.write_csv(tmp_path / "folds.csv")

    with pytest.raises(ValueError, match="data is missing required columns"):
        _load_oof_data(
            tmp_path / "data.csv",
            tmp_path / "folds.csv",
        )


@pytest.mark.parametrize("table",
                         ["data", "folds"])
def test_load_oof_data_rejects_duplicate_molecule_ids(
    tmp_path: Path, loader_tables, table) -> None:
    data, folds = loader_tables

    if table == "data":
        data = pl.concat([data, data.head(1)])
    else:
        folds = pl.concat([folds, folds.head(1)])

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    data.write_csv(data_path)
    folds.write_csv(folds_path)

    with pytest.raises(ValueError, match=f"{table} contains duplicate molecules IDs"):
        _load_oof_data(data_path, folds_path)
    

def test_load_oof_data_rejects_missing_fold_assignments(tmp_path: Path, loader_tables):
    data, folds = loader_tables
        
    # Remove one assignment while retaining both fold values.
    folds = folds.filter(pl.col("inchikey_full") != "A-full")

    data.write_csv(tmp_path / "data.csv")
    folds.write_csv(tmp_path / "folds.csv")

    with pytest.raises(ValueError, match="missing fold assignments"):
        _load_oof_data(
            tmp_path / "data.csv",
            tmp_path / "folds.csv",
        )   


def test_load_oof_data_rejects_unknown_assignment_ids(tmp_path: Path, loader_tables) -> None:
    data, folds = loader_tables
            
    folds = pl.concat([
        folds, 
        pl.DataFrame({
            "inchikey_full": ["UNKNOWN"],
            "fold": [0],
        }),
    ])

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    data.write_csv(data_path)
    folds.write_csv(folds_path)

    with pytest.raises(ValueError, match="Assignments reference unknown molecules"):
        _load_oof_data(data_path, folds_path)


@pytest.mark.parametrize(
    "fold_values, message",
    [
        ([0.0, 0.5, 1.0, 1.0], "folds.fold must contain integers"),
        ([0, None, 1, 1], "There are null values in folds"),
        ([0, 0, 0, 0], "At least two distinct folds are required"),
        ([-1, 0, 1, 1], "Fold assignments must be nonnegative"),
    ],
)
def test_load_oof_data_rejects_invalid_folds(
    tmp_path: Path, loader_tables, fold_values, message,
) -> None:
    data, folds = loader_tables
            
    folds = folds.with_columns(
        pl.Series("fold", fold_values)
    )

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    data.write_csv(data_path)
    folds.write_csv(folds_path)

    with pytest.raises(ValueError, match=message):
        _load_oof_data(data_path, folds_path)
    

def test_load_oof_data_rejects_existing_fold_column(tmp_path: Path, loader_tables) -> None:
    data, folds = loader_tables
                
    data = data.with_columns(pl.lit(0).alias("fold"))

    data.write_csv(tmp_path / "data.csv")
    folds.write_csv(tmp_path / "folds.csv")

    with pytest.raises(ValueError, match="data already contains a fold column"):
        _load_oof_data(
            tmp_path / "data.csv",
            tmp_path / "folds.csv",
    )   


def test_load_oof_data_preserves_distinct_full_keys_with_same_block(
    tmp_path, loader_tables,
):
    data, folds = loader_tables
    data = data.with_columns(
        pl.Series("inchikey_block", ["A", "A", "C", "D"])
    )
    data.write_csv(tmp_path / "data.csv")
    folds.write_csv(tmp_path / "folds.csv")

    result = _load_oof_data(
        tmp_path / "data.csv", tmp_path / "folds.csv"
    )

    assert result.height == 4
    shared = result.filter(pl.col("inchikey_block") == "A")
    assert set(shared["inchikey_full"]) == {"A-full", "B-full"}
    assert shared["fold"].to_list() == [0, 0]


def test_load_oof_data_rejects_block_across_folds(
    tmp_path, loader_tables,
):
    data, folds = loader_tables
    data = data.with_columns(
        pl.Series("inchikey_block", ["A", "A", "C", "D"])
    )
    folds = folds.with_columns(
        pl.Series("fold", [0, 1, 1, 1])
    )
    data.write_csv(tmp_path / "data.csv")
    folds.write_csv(tmp_path / "folds.csv")

    with pytest.raises(ValueError, match="must share a fold"):
        _load_oof_data(
            tmp_path / "data.csv", tmp_path / "folds.csv"
        )


@pytest.mark.parametrize("table", ["data", "folds"])
@pytest.mark.parametrize(
    "values, message",
    [
        ([None, "B-full", "C-full", "D-full"], "null values"),
        ([" ", "B-full", "C-full", "D-full"], "blank identifiers"),
        ([1, 2, 3, 4], "must contain strings"),
    ],
    ids=["null", "blank", "numeric"],
)
def test_load_oof_data_rejects_invalid_full_keys(
    tmp_path, loader_tables, table, values, message,
):
    data, folds = loader_tables
    replacement = pl.Series("inchikey_full", values)
    if table == "data":
        data = data.with_columns(replacement)
    else:
        folds = folds.with_columns(replacement)

    data.write_csv(tmp_path / "data.csv")
    folds.write_csv(tmp_path / "folds.csv")

    with pytest.raises(ValueError, match=message):
        _load_oof_data(
            tmp_path / "data.csv", tmp_path / "folds.csv"
        )


def test_cli_forwards_valid_arguments(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run_oof(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)
    monkeypatch.setattr(sys, "argv", [
        "generate-oof",
        "--data-path", str(tmp_path / "data.csv"),
        "--folds-path", str(tmp_path / "folds.csv"),
        "--model-name", "decision_tree",
        "--model-id", "tree_test",
        "--model-params", '{"max_depth": 5}',
        "--input-names", "rdkit",
        "--output-dir", str(tmp_path / "output"),
        "--cache-dir", str(tmp_path / "features"),
    ])

    oof.main()

    assert len(calls) == 1
    assert calls[0]["model_params"] == {"max_depth": 5}
    assert calls[0]["input_names"] == ["rdkit"]
    assert calls[0]["data_path"] == tmp_path / "data.csv"
    assert calls[0]["folds_path"] == tmp_path / "folds.csv"
    assert calls[0]["model_name"] == "decision_tree"
    assert calls[0]["model_id"] == "tree_test"
    assert calls[0]["output_dir"] == tmp_path / "output"
    assert calls[0]["cache_dir"] == tmp_path / "features"


def test_cli_rejects_malformed_model_params_json(monkeypatch, capsys, tmp_path) -> None:
    calls = []
    
    def fake_run_oof(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)
    monkeypatch.setattr(sys, "argv", [
        "generate-oof",
        "--data-path", str(tmp_path / "data.csv"),
        "--folds-path", str(tmp_path / "folds.csv"),
        "--model-name", "decision_tree",
        "--model-id", "tree_test",
        "--model-params", "{'max_depth': 5}",    # JSON requires double quotes.
        "--input-names", "rdkit",
        "--output-dir", str(tmp_path / "output"),
        "--cache-dir", str(tmp_path / "features"),
    ])

    with pytest.raises(SystemExit) as exc:
        oof.main()

    assert exc.value.code == 2
    assert "--model-params must be valid JSON" in capsys.readouterr().err
    assert calls == []  


def test_cli_rejects_non_object_model_params(monkeypatch, capsys, tmp_path) -> None:
    calls = []
        
    def fake_run_oof(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)
    monkeypatch.setattr(sys, "argv", [
        "generate-oof",
        "--data-path", str(tmp_path / "data.csv"),
        "--folds-path", str(tmp_path / "folds.csv"),
        "--model-name", "decision_tree",
        "--model-id", "tree_test",
        "--model-params", "[]",    # non-object
        "--input-names", "rdkit",
        "--output-dir", str(tmp_path / "output"),
        "--cache-dir", str(tmp_path / "features"),
    ])

    with pytest.raises(SystemExit) as exc:
        oof.main()

    assert exc.value.code == 2
    assert "--model-params must be a JSON object" in capsys.readouterr().err
    assert calls == []  

@pytest.mark.parametrize(
    "model_name, feature_name, bad_option",
    [
        ("unknown_model", "rdkit", "--model-name"),
        ("decision_tree", "unknown_feature", "--input-names"),
    ]
)
def test_cli_rejects_unknown_model_or_feature(
    monkeypatch, capsys, tmp_path, 
    model_name, feature_name, bad_option,
) -> None:
    calls = []
        
    def fake_run_oof(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)
    monkeypatch.setattr(sys, "argv", [
        "generate-oof",
        "--data-path", str(tmp_path / "data.csv"),
        "--folds-path", str(tmp_path / "folds.csv"),
        "--model-name", model_name,
        "--model-id", "tree_test",
        "--model-params", "{}",   
        "--input-names", feature_name,
        "--output-dir", str(tmp_path / "output"),
        "--cache-dir", str(tmp_path / "features"),
    ])

    with pytest.raises(SystemExit) as exc:
        oof.main()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    if bad_option == "--model-name":
        assert "argument --model-name: invalid choice" in error
    else:
        assert "--input-names: unknown input(s)" in error
    assert calls == []


def test_cli_forwards_chemprop_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = []
    
    def fake_run_oof(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)

    config = {
        "epochs": 2, 
        "seed": 42, 
        "from_foundation": None, 
        "target_columns": list(oof._SCORED_DEFAULT),
    }
    monkeypatch.setattr(sys, "argv", [
        "generate-oof",
        "--data-path", str(tmp_path / "data.csv"),
        "--folds-path", str(tmp_path / "folds.csv"),
        "--model-name", "chemprop",
        "--model-id", "chemprop_test",
        "--output-dir", str(tmp_path / "output"),
        "--chemprop-config", json.dumps(config),
        "--group-column", "group_id",
        "--validation-fraction", "0.1",
        "--validation-seed", "42",
    ])

    oof.main()

    assert len(calls) == 1
    assert calls[0]["model_name"] == "chemprop"
    assert calls[0]["model_id"] == "chemprop_test"
    assert calls[0]["chemprop_config"] == config
    assert calls[0]["group_column"] == "group_id"
    assert calls[0]["validation_fraction"] == 0.1
    assert calls[0]["validation_seed"] == 42
    assert calls[0]["input_names"] is None
    assert calls[0]["cache_dir"] is None
    assert calls[0]["model_params"] == {}
    
@pytest.mark.parametrize(
    "config_text, expected_error",
    [
        ("{'epochs': 2}", "--chemprop-config must be valid JSON"),
        ('{"epochs": 2', "--chemprop-config must be valid JSON"),
        ("[]", "--chemprop-config must be a JSON object"),
        ("null", "--chemprop-config must be a JSON object"),
    ],
)
def test_cli_rejects_invalid_chemprop_json(
    monkeypatch: pytest.MonkeyPatch, 
    capsys,
    tmp_path,
    config_text, 
    expected_error
):
    calls = []
    
    def fake_run_oof(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)
    monkeypatch.setattr(sys, "argv", [
        "generate-oof",
        "--data-path", str(tmp_path / "data.csv"),
        "--folds-path", str(tmp_path / "folds.csv"),
        "--model-name", "chemprop",
        "--model-id", "chemprop_test",
        "--output-dir", str(tmp_path / "output"),
        "--chemprop-config", config_text,
        "--group-column", "group_id",
        "--validation-fraction", "0.1",
        "--validation-seed", "42",
    ])
    
    with pytest.raises(SystemExit) as exc:
        oof.main()

    assert exc.value.code == 2
    assert expected_error in capsys.readouterr().err
    assert calls == []  

@pytest.mark.parametrize(
    "missing_flag",
    [
        "--chemprop-config",
        "--group-column",
        "--validation-fraction",
        "--validation-seed",
    ],
)
def test_cli_requires_chemprop_flags(
    monkeypatch: pytest.MonkeyPatch, 
    capsys,
    tmp_path,
    missing_flag
):
    calls = []
    
    def fake_run_oof(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)

    config = {
        "epochs": 2, 
        "seed": 42, 
        "from_foundation": None, 
        "target_columns": list(oof._SCORED_DEFAULT),
    }

    argv = [
        "generate-oof",
        "--data-path", str(tmp_path / "data.csv"),
        "--folds-path", str(tmp_path / "folds.csv"),
        "--model-name", "chemprop",
        "--model-id", "chemprop_test",
        "--output-dir", str(tmp_path / "output"),
        "--chemprop-config", json.dumps(config),
        "--group-column", "group_id",
        "--validation-fraction", "0.1",
        "--validation-seed", "42",
    ]

    index = argv.index(missing_flag)
    del argv[index:index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    
    with pytest.raises(SystemExit) as exc:
        oof.main()

    assert exc.value.code == 2
    assert f"{missing_flag} is required for Chemprop" in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize(
    "missing_flag",
    [
        "--input-names",
        "--cache-dir",
    ],
)
def test_cli_requires_feature_model_flags(
    monkeypatch: pytest.MonkeyPatch, 
    capsys,
    tmp_path,
    missing_flag
):
    calls = []
    
    def fake_run_oof(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)

    argv = [
        "generate-oof",
        "--data-path", str(tmp_path / "data.csv"),
        "--folds-path", str(tmp_path / "folds.csv"),
        "--model-name", "decision_tree",
        "--model-id", "tree_test",
        "--model-params", '{"max_depth": 5}',
        "--input-names", "rdkit",
        "--output-dir", str(tmp_path / "output"),
        "--cache-dir", str(tmp_path / "features"),
    ]

    index = argv.index(missing_flag)
    del argv[index:index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    
    with pytest.raises(SystemExit) as exc:
        oof.main()

    assert exc.value.code == 2
    assert f"{missing_flag} is required for feature-based models" in capsys.readouterr().err
    assert calls == []


@pytest.fixture
def chemprop_conversion_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    wide_predictions = pl.DataFrame({
        "inchikey_full": ["A", "B"],
        "CYP1A2_pIC50_direct_inhibition": [5.1, 6.2],
        "CYP2C9_pIC50_direct_inhibition": [4.3, 5.4],
        "CYP2D6_pIC50_direct_inhibition": [6.5, 7.6],
        "CYP3A4_pIC50_direct_inhibition": [3.7, 4.8],
    })

    target_truth_values = {
            "CYP1A2_pIC50_direct_inhibition": [0.2, 0.8, 0.0, 1.0],
            "CYP2C9_pIC50_direct_inhibition": [0.4, 0.6, 0.0, 1.0],
            "CYP2D6_pIC50_direct_inhibition": [0.6, 0.4, 0.0, 1.0],
            "CYP3A4_pIC50_direct_inhibition": [0.8, 0.2, 0.0, 1.0],
        }

    molecules = [
        ("A", 0),
        ("B", 0),
        ("C", 1),
        ("D", 1),
    ]
    
    truth_rows = []

    for target, values in target_truth_values.items():
        for (molecule, fold), value in zip(molecules, values, strict=True):
            truth_rows.append({
                "inchikey_full": molecule, 
                "fold": fold, 
                "target": target, 
                "truth": value,
            })

    truth = pl.DataFrame(truth_rows)

    return wide_predictions, truth


def test_convert_chemprop_predictions_valid_input(
    chemprop_conversion_tables,
) -> None:

    wide_predictions, truth = chemprop_conversion_tables

    result = _convert_chemprop_predictions(
        predictions = wide_predictions, 
        truth=truth,
        model_id="chemprop_test",
        heldout_fold=0,
    )

    assert result.height == 8 # Two molecules x four targets
    assert result["model_id"].unique().to_list() == ["chemprop_test"]
    assert result["fold"].unique().to_list() == [0]

    expected_predictions = pl.DataFrame(
        [
            ("chemprop_test", "A", 0, "CYP1A2_pIC50_direct_inhibition", 5.1),
            ("chemprop_test", "B", 0, "CYP1A2_pIC50_direct_inhibition", 6.2),
            ("chemprop_test", "A", 0, "CYP2C9_pIC50_direct_inhibition", 4.3),
            ("chemprop_test", "B", 0, "CYP2C9_pIC50_direct_inhibition", 5.4),
            ("chemprop_test", "A", 0, "CYP2D6_pIC50_direct_inhibition", 6.5),
            ("chemprop_test", "B", 0, "CYP2D6_pIC50_direct_inhibition", 7.6),
            ("chemprop_test", "A", 0, "CYP3A4_pIC50_direct_inhibition", 3.7),
            ("chemprop_test", "B", 0, "CYP3A4_pIC50_direct_inhibition", 4.8),
        ],
        schema=["model_id", "inchikey_full", "fold", "target", "pred"],
        orient="row"
    )

    keys = ["model_id", "inchikey_full", "fold", "target"]
    assert_frame_equal(
        result.sort(keys),
        expected_predictions.sort(keys),
    )


def test_convert_chemprop_predictions_handles_shuffled_rows(
    chemprop_conversion_tables
) -> None:
    wide_predictions, truth = chemprop_conversion_tables

    shuffled_predictions = wide_predictions.reverse()
    shuffled_truth = truth.reverse()

    result = _convert_chemprop_predictions(
        predictions = shuffled_predictions, 
        truth=shuffled_truth,
        model_id="chemprop_test",
        heldout_fold=0,
    )

    assert result.height == 8 # Two molecules x four targets
    assert result["model_id"].unique().to_list() == ["chemprop_test"]
    assert result["fold"].unique().to_list() == [0]

    expected_predictions = pl.DataFrame(
        [
            ("chemprop_test", "A", 0, "CYP1A2_pIC50_direct_inhibition", 5.1),
            ("chemprop_test", "B", 0, "CYP1A2_pIC50_direct_inhibition", 6.2),
            ("chemprop_test", "A", 0, "CYP2C9_pIC50_direct_inhibition", 4.3),
            ("chemprop_test", "B", 0, "CYP2C9_pIC50_direct_inhibition", 5.4),
            ("chemprop_test", "A", 0, "CYP2D6_pIC50_direct_inhibition", 6.5),
            ("chemprop_test", "B", 0, "CYP2D6_pIC50_direct_inhibition", 7.6),
            ("chemprop_test", "A", 0, "CYP3A4_pIC50_direct_inhibition", 3.7),
            ("chemprop_test", "B", 0, "CYP3A4_pIC50_direct_inhibition", 4.8),
        ],
        schema=["model_id", "inchikey_full", "fold", "target", "pred"],
        orient="row"
    )

    keys = ["model_id", "inchikey_full", "fold", "target"]
    assert_frame_equal(
        result.sort(keys),
        expected_predictions.sort(keys),
    )


def test_convert_chemprop_predictions_omits_missing_truth_labels(
    chemprop_conversion_tables
) -> None:
    wide_predictions, truth = chemprop_conversion_tables
    target = "CYP2C9_pIC50_direct_inhibition"

    # Baseline: all eight molecule/target predictions are required.
    complete = _convert_chemprop_predictions(
        predictions=wide_predictions,
        truth=truth,
        model_id="chemprop_test",
        heldout_fold=0,
    )

    omitted_key = (
        (pl.col("inchikey_full") == "A")
        & (pl.col("fold") == 0)
        & (pl.col("target") == target)
    )
    sparse_truth = truth.filter(~omitted_key)

    result = _convert_chemprop_predictions(
        predictions=wide_predictions,
        truth=sparse_truth,
        model_id="chemprop_test",
        heldout_fold=0,
    )

    assert result.height == 7
    assert result.filter(omitted_key).is_empty()

    keys = ["model_id", "inchikey_full", "fold", "target"]
    assert_frame_equal(
        result.sort(keys),
        complete.filter(~omitted_key).sort(keys),
    )


def test_convert_chemprop_predictions_ignores_auxiliary_columns(
    chemprop_conversion_tables
) -> None:
    wide_predictions, truth = chemprop_conversion_tables
    
    wide_predictions = wide_predictions.with_columns(pl.lit(0).alias("auxiliary"))

    result = _convert_chemprop_predictions(
        predictions = wide_predictions, 
        truth=truth,
        model_id="chemprop_test",
        heldout_fold=0,
    )

    assert result.height == 8 # Two molecules x four targets
    assert result["model_id"].unique().to_list() == ["chemprop_test"]
    assert result["fold"].unique().to_list() == [0]

    expected_predictions = pl.DataFrame(
        [
            ("chemprop_test", "A", 0, "CYP1A2_pIC50_direct_inhibition", 5.1),
            ("chemprop_test", "B", 0, "CYP1A2_pIC50_direct_inhibition", 6.2),
            ("chemprop_test", "A", 0, "CYP2C9_pIC50_direct_inhibition", 4.3),
            ("chemprop_test", "B", 0, "CYP2C9_pIC50_direct_inhibition", 5.4),
            ("chemprop_test", "A", 0, "CYP2D6_pIC50_direct_inhibition", 6.5),
            ("chemprop_test", "B", 0, "CYP2D6_pIC50_direct_inhibition", 7.6),
            ("chemprop_test", "A", 0, "CYP3A4_pIC50_direct_inhibition", 3.7),
            ("chemprop_test", "B", 0, "CYP3A4_pIC50_direct_inhibition", 4.8),
        ],
        schema=["model_id", "inchikey_full", "fold", "target", "pred"],
        orient="row"
    )

    keys = ["model_id", "inchikey_full", "fold", "target"]
    assert_frame_equal(
        result.sort(keys),
        expected_predictions.sort(keys),
    )


def test_convert_chemprop_predictions_rejects_duplicate_ids(
    chemprop_conversion_tables
) -> None:
    wide_predictions, truth = chemprop_conversion_tables

    duplicated = pl.concat([wide_predictions, wide_predictions.head(1)])

    with pytest.raises(ValueError, match="duplicate molecules IDs"):
        _convert_chemprop_predictions(
            predictions = duplicated, 
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
        )


def test_convert_chemprop_predictions_rejects_missing_molecules(
    chemprop_conversion_tables
) -> None:
    wide_predictions, truth = chemprop_conversion_tables

    incomplete = wide_predictions.slice(1)
    
    with pytest.raises(ValueError, match="Missing molecules"):
        _convert_chemprop_predictions(
                predictions = incomplete, 
                truth=truth,
                model_id="chemprop_test",
                heldout_fold=0,
            )


def test_convert_chemprop_predictions_rejects_extra_molecules(
    chemprop_conversion_tables
) -> None:
    wide_predictions, truth = chemprop_conversion_tables
    
    extra_row = wide_predictions.head(1).with_columns(
        pl.lit("SYNTHETIC_EXTRA").alias("inchikey_full")
    )
    extended = pl.concat([wide_predictions, extra_row])

    with pytest.raises(ValueError, match="Unexpected molecules"):
        _convert_chemprop_predictions(
            predictions = extended, 
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
        )

@pytest.mark.parametrize(
    "missing_column",
    [
        "inchikey_full",
        "CYP1A2_pIC50_direct_inhibition",
        "CYP2C9_pIC50_direct_inhibition",
        "CYP2D6_pIC50_direct_inhibition",
        "CYP3A4_pIC50_direct_inhibition",
    ],
)
def test_convert_chemprop_predictions_rejects_missing_columns(
    chemprop_conversion_tables,
    missing_column: str,
) -> None:
    wide_predictions, truth = chemprop_conversion_tables
    incomplete = wide_predictions.drop(missing_column)

    with pytest.raises(ValueError, match="Missing prediction columns") as exc:
        _convert_chemprop_predictions(
            predictions = incomplete, 
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
        )

    assert missing_column in str(exc.value)

@pytest.mark.parametrize(
    "values, message",
    [
        pytest.param(
            ["invalid", "6.2"],
            "must be numeric",
            id="nonnumeric",
        ),
        pytest.param(
            [None, 6.2],
            "null values in predictions",
            id="null",
        ),
        pytest.param(
            [float("nan"), 6.2],
            "pred contains non-finite values",
            id="nan",
        ),
        pytest.param(
            [float("inf"), 6.2],
            "pred contains non-finite values",
            id="positive-infinity",
        ),
        pytest.param(
            [float("-inf"), 6.2],
            "pred contains non-finite values",
            id="negative-infinity",
        ),
    ],
)
def test_convert_chemprop_predictions_rejects_invalid_prediction_values(
    chemprop_conversion_tables,
    values,
    message: str,
) -> None:
    wide_predictions, truth = chemprop_conversion_tables
    target = "CYP1A2_pIC50_direct_inhibition"

    invalid = wide_predictions.with_columns(
        pl.Series(target, values)
    )

    with pytest.raises(ValueError, match=message):
        _convert_chemprop_predictions(
            predictions=invalid,
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
        )
    

@pytest.mark.parametrize(
    "model_name, config_arg, seed_key, config",
    [
        (
            "decision_tree",
            "model_params",
            "random_state",
            {"max_depth": 5, "random_state": 42},
        ),
        (
            "chemprop",
            "chemprop_config",
            "seed",
            {"epochs": 2, "seed": 42},
        ),
    ],
)
def test_run_oof_seeds_calls_and_outputs(
    tmp_path, monkeypatch, model_name, config_arg, seed_key, config
):
    calls = []
    truth = pl.DataFrame({"truth": [5.0]})
    returned_predictions = {}

    def fake_run_oof(**kwargs):
        calls.append(kwargs)
        seed = kwargs[config_arg][seed_key]
        predictions = pl.DataFrame({
            "model_id": [kwargs["model_id"]],
            "pred": [5.0 + seed],
        })
        returned_predictions[seed] = predictions
        return predictions, truth

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)

    original_config = config.copy()

    predictions_by_seed, actual_truth = oof.run_oof_seeds(
        data_path=tmp_path / "data.csv",
        folds_path=tmp_path / "folds.csv",
        model_name=model_name,
        model_id="model_test",
        input_names=["rdkit"],
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "features",
        validation_seed=42,
        seeds=[0, 1, 2],
        **{config_arg: config},
    )

    assert len(calls) == 3
    assert set(predictions_by_seed) == {0, 1, 2}
    assert actual_truth.equals(truth)
    assert config == original_config

    for seed, call in enumerate(calls):
        expected_config = {**original_config, seed_key: seed}
        assert call[config_arg] == expected_config
        assert call[config_arg] is not config
        assert call["model_id"] == f"model_test_seed{seed}"
        assert call["output_dir"] == tmp_path / "output" / f"seed-{seed}"
        assert call["data_path"] == tmp_path / "data.csv"
        assert call["folds_path"] == tmp_path / "folds.csv"
        assert call["model_name"] == model_name
        assert call["validation_seed"] == 42
        assert call["input_names"] == ["rdkit"]
        assert call["cache_dir"] == tmp_path / "features"
        assert predictions_by_seed[seed].equals(returned_predictions[seed])

@pytest.mark.parametrize(
    "seeds, message",
    [
        ([0, 1], "at least 3 seeds"),
        ([0, 1, 1], "seeds must be unique"),
        ([0, 1, "2"], "seeds must be integers"),
        ([0, 1, 2.5], "seeds must be integers"),
        ([0, 1, True], "seeds must be integers"),
    ],
)
def test_run_oof_seeds_rejects_invalid_seeds(
    tmp_path, monkeypatch, seeds, message
):
    def fake_run_oof(**kwargs):
        pytest.fail("run_oof should not be called for invalid seeds")

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)

    with pytest.raises(ValueError, match=message):
        oof.run_oof_seeds(
            data_path=tmp_path / "data.csv",
            folds_path=tmp_path / "folds.csv",
            model_name="decision_tree",
            model_id="model_test",
            input_names=["rdkit"],
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "features",
            validation_seed=42,
            seeds=seeds,
        )

@pytest.mark.parametrize(
    "model_name, config_arg, config, message",
    [
        ("decision_tree", "model_params", [], "model_params must be dict"),
        ("decision_tree", "model_params", "bad", "model_params must be dict"),
        ("chemprop", "chemprop_config", None, "chemprop_config must be dict"),
        ("chemprop", "chemprop_config", [], "chemprop_config must be dict"),
        ("chemprop", "chemprop_config", "bad", "chemprop_config must be dict"),
    ],
)
def test_run_oof_seeds_rejects_invalid_config(
    tmp_path, monkeypatch, model_name, config_arg, config, message
):
    def fake_run_oof(**kwargs):
        pytest.fail("run_oof should not be called for invalid config")
    
    monkeypatch.setattr(oof, "run_oof", fake_run_oof)

    with pytest.raises(ValueError, match=message):
        oof.run_oof_seeds(
            data_path=tmp_path / "data.csv",
            folds_path=tmp_path / "folds.csv",
            model_name=model_name,
            model_id="model_test",
            input_names=["rdkit"],
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "features",
            validation_seed=42,
            seeds=[0, 1, 2],
            **{config_arg: config},
        )

def test_run_oof_seeds_rejects_changed_truth(tmp_path, monkeypatch): 
    calls = []

    def fake_run_oof(**kwargs):
        calls.append(kwargs)
        predictions = pl.DataFrame({
            "model_id": [kwargs["model_id"]],
            "pred": [5.0],
        })
        truth_value = 5.0 if len(calls) == 1 else 6.0
        truth = pl.DataFrame({"truth": [truth_value]})
        return predictions, truth

    monkeypatch.setattr(oof, "run_oof", fake_run_oof)

    with pytest.raises(ValueError, match="truth is different"):
        oof.run_oof_seeds(
            data_path=tmp_path / "data.csv",
            folds_path=tmp_path / "folds.csv",
            model_name="decision_tree",
            model_id="model_test",
            input_names=["rdkit"],
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "features",
            validation_seed=42,
            seeds=[0, 1, 2],
        )

    assert len(calls) == 2

