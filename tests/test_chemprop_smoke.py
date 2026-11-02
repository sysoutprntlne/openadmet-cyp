from pathlib import Path

import numpy as np
import polars as pl
import pytest

import subprocess
import sys

from models.training.chemprop import build_command
from polars.testing import assert_frame_equal
from models.oof_chemprop import _generate_chemprop_fold_predictions
from models.oof import _SCORED_DEFAULT


@pytest.fixture
def chemprop_fixture():
    """Create chemprop tables."""

    molecules = [
        ("SYNTHETIC_A", "train"),
        ("SYNTHETIC_B", "train"), 
        ("SYNTHETIC_C", "train"),
        ("SYNTHETIC_D", "train"),
        ("SYNTHETIC_E", "train"),
        ("SYNTHETIC_F", "train"),
        ("SYNTHETIC_G", "train"),
        ("SYNTHETIC_H", "train"),
        ("SYNTHETIC_I", "val"),
        ("SYNTHETIC_J", "val"),
        ("SYNTHETIC_K", "val"),
        ("SYNTHETIC_L", "val"),
        ("SYNTHETIC_M", "heldout"),
        ("SYNTHETIC_N", "heldout"),
    ]

    target = "CYP1A2_pIC50_direct_inhibition"

    data = pl.DataFrame({
        "inchikey_full": [molecule for molecule, split in molecules],
        "inchikey_block": [molecule for molecule, split in molecules],
        "split": [split for molecule, split in molecules],
        "SMILES": [
            "C", "CC", "CCC", "CCCC", "CCO", "CCN", "CCCO", "CCCN",
            "CCOC", "CCNC", "CC(=O)O", "CC(=O)N",
            "c1ccccc1", "Cc1ccccc1",
        ],
        target: [4.1, 4.5, 5.0, 5.2, 4.8, 5.5, 5.1, 5.8,
                5.3, 5.7, 4.6, 5.4, 6.0, 6.2],
    })

    training_data = data.filter(pl.col("split").is_in(["train", "val"]))

    prediction_data = data.filter(
        pl.col("split") == "heldout"
    ).select("inchikey_full", "SMILES")

    return training_data, prediction_data, target

def test_chemprop_train_predict_smoke(tmp_path: Path, chemprop_fixture):
    training_data, prediction_data, target = chemprop_fixture

    training_path = tmp_path / "training.csv"
    heldout_path = tmp_path / "heldout.csv"
    output_dir = tmp_path / "chemprop"
    predictions_path = tmp_path / "predictions.csv"

    training_data.write_csv(training_path)
    prediction_data.write_csv(heldout_path)

    cmd = build_command(
        data_path=training_path,
        smiles_column="SMILES",
        target_columns=[target],
        output_dir=output_dir,
        epochs=2,
        seed=42,
        from_foundation=None,
        extra_args=[
            "--splits-column", "split",
            "--accelerator", "cpu",
            "--devices", "1",
            "--num-workers", "0",
            "--batch-size", "4",
            "--warmup-epochs", "0",
        ],
    )

    # Use Chemprop from the same environment as the test's Python.
    cmd[0] = str(Path(sys.executable).parent / "chemprop")

    result = subprocess.run(
        cmd, 
        capture_output=True,
        text=True, 
        timeout=300,
    )

    assert result.returncode == 0, (
        f"Chemprop training failed:\n{result.stdout}\n{result.stderr}"
    )

    model_path = output_dir / "model_0" / "best.pt"
    assert model_path.is_file()

    predict_cmd = [
        str(Path(sys.executable).parent / "chemprop"),
        "predict",
        "--test-path", str(heldout_path),
        "--model-path", str(model_path),
        "--preds-path", str(predictions_path),
        "--smiles-columns", "SMILES",
        "--accelerator", "cpu",
        "--devices", "1",
        "--num-workers", "0",
        "--batch-size", "4",
    ]

    prediction_result = subprocess.run(
        predict_cmd, 
        capture_output=True,
        text=True, 
        timeout=300,
    )

    assert prediction_result.returncode == 0, (
        f"Chemprop prediction failed:\n"
        f"{prediction_result.stdout}\n{prediction_result.stderr}"
    )

    predictions = pl.read_csv(predictions_path)
    heldout = pl.read_csv(heldout_path)

    assert predictions.height == 2, (
        f"Expected 2 prediction rows, got {predictions.height}"
    )

    required = {"inchikey_full", "SMILES", target}
    missing = required - set(predictions.columns)
    assert not missing, f"Missing prediction columns: {sorted(missing)}"

    # Check that both identifiers and structures survive prediction unchanged.
    assert_frame_equal(
        predictions.select("inchikey_full", "SMILES"),
        heldout.select("inchikey_full", "SMILES"),
    )

    assert predictions[target].dtype.is_numeric()
    assert predictions[target].null_count() == 0
    assert np.isfinite(predictions[target].to_numpy()).all()

@pytest.fixture
def four_target_chemprop_fixture():
    """Create chemprop tables."""

    molecules = [
        ("SYNTHETIC_A", "train"),
        ("SYNTHETIC_B", "train"), 
        ("SYNTHETIC_C", "train"),
        ("SYNTHETIC_D", "train"),
        ("SYNTHETIC_E", "train"),
        ("SYNTHETIC_F", "train"),
        ("SYNTHETIC_G", "train"),
        ("SYNTHETIC_H", "train"),
        ("SYNTHETIC_I", "val"),
        ("SYNTHETIC_J", "val"),
        ("SYNTHETIC_K", "val"),
        ("SYNTHETIC_L", "val"),
        ("SYNTHETIC_M", "heldout"),
        ("SYNTHETIC_N", "heldout"),
    ]

    data = pl.DataFrame({
        "inchikey_full": [molecule for molecule, split in molecules],
        "inchikey_block": [molecule for molecule, split in molecules],
        "split": [split for molecule, split in molecules],
        "SMILES": [
            "C", "CC", "CCC", "CCCC", "CCO", "CCN", "CCCO", "CCCN",
            "CCOC", "CCNC", "CC(=O)O", "CC(=O)N",
            "c1ccccc1", "Cc1ccccc1",
        ],
    })

    for i, target in enumerate(_SCORED_DEFAULT):
        data = data.with_columns(
            pl.Series(
                target,
                [4.0 + 0.1 * j + 0.2 * i for j in range(data.height)],
            )
        )

    train_df = data.filter(pl.col("split").is_in(["train"]))
    val_df = data.filter(pl.col("split").is_in(["val"]))
    heldout_df = data.filter(pl.col("split").is_in(["heldout"]))

    truth_rows = []

    for row in heldout_df.iter_rows(named=True):
        for target in _SCORED_DEFAULT:
            value = row[target]
            if value is None:
                continue

            truth_rows.append({
                "inchikey_full": row["inchikey_full"],
                "fold": 0,
                "target": target,
                "truth": value,
            })

    truth = pl.DataFrame(truth_rows)

    return train_df, val_df, heldout_df, truth

def test_chemprop_fold_adapter_smoke(tmp_path: Path, four_target_chemprop_fixture) -> None:
    train_df, val_df, heldout_df, truth = four_target_chemprop_fixture

    predictions = _generate_chemprop_fold_predictions(
        train_df=train_df,
        val_df=val_df,
        heldout_df=heldout_df,
        truth=truth,
        model_id="chemprop_smoke",
        heldout_fold=0,
        work_dir=tmp_path / "adapter",
        target_columns=list(_SCORED_DEFAULT),
        epochs=2,
        seed=42,
    )

    assert predictions.height == 8

    assert set(predictions.columns) == {
        "model_id", "inchikey_full", "fold", "target", "pred"
    }

    assert predictions["model_id"].to_list() == ["chemprop_smoke"] * 8

    # Exact molecule/fold/target coverage, including no duplicate rows.
    keys = ["inchikey_full", "fold", "target"]
    assert_frame_equal(
        predictions.select(keys).sort(keys),
        truth.select(keys).sort(keys),
    )

    assert predictions["pred"].dtype.is_numeric()
    assert predictions["pred"].null_count() == 0
    assert np.isfinite(predictions["pred"].to_numpy()).all()
