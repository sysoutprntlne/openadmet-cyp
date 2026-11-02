"""Test Chemprop OOF commands, internal splits, configuration, and resume behavior."""

import random
import re
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from models.oof import _SCORED_DEFAULT, run_oof
from models.oof_chemprop import (
    _generate_chemprop_fold_predictions,
    _resolve_chemprop_config,
    _split_internal_validation,
)


def test_chemprop_excludes_auxiliary_only_heldout_rows(tmp_path, monkeypatch):
    def population(prefix, smiles, labels):
        return pl.DataFrame({
            "inchikey_full": [f"{prefix}_{i}-full" for i in range(len(smiles))],
            "inchikey_block": [f"{prefix}_{i}" for i in range(len(smiles))],
            "SMILES": smiles,
            **{target: labels for target in _SCORED_DEFAULT},
            "auxiliary": [2.0] * len(smiles),
        })

    train = population("train", ["C", "CC"], [5.0, None])
    val = population("val", ["CCC"], [6.0])
    heldout = population("heldout", ["CCCC", "CCCCC"], [7.0, None])
    truth = pl.DataFrame({
        "inchikey_full": ["heldout_0-full"] * 4,
        "fold": [0] * 4,
        "target": _SCORED_DEFAULT,
        "truth": [7.0] * 4,
    })
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1])
        if cmd[1] == "train":
            table = pl.read_csv(cmd[cmd.index("--data-path") + 1])
            fitting = table.filter(pl.col("split") == "train")
            assert set(fitting["inchikey_full"]) == set(train["inchikey_full"])
            aux_only = fitting.filter(pl.col("inchikey_full") == "train_1-full")
            assert aux_only["auxiliary"].item() == 2.0
            assert all(aux_only[target].item() is None for target in _SCORED_DEFAULT)
            model = Path(cmd[cmd.index("--output-dir") + 1]) / "model_0" / "best.pt"
            model.parent.mkdir(parents=True)
            model.touch()
        elif cmd[1] == "predict":
            table = pl.read_csv(cmd[cmd.index("--test-path") + 1])
            assert table["inchikey_full"].to_list() == ["heldout_0-full"]
            table.with_columns(
                *[pl.lit(6.5).alias(target) for target in _SCORED_DEFAULT]
            ).write_csv(cmd[cmd.index("--preds-path") + 1])
        else:
            pytest.fail(f"Unexpected command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("models.oof_chemprop.subprocess.run", fake_run)
    predictions = _generate_chemprop_fold_predictions(
        train_df=train, val_df=val, heldout_df=heldout, truth=truth,
        model_id="aux_model", heldout_fold=0, work_dir=tmp_path / "job",
        target_columns=[*_SCORED_DEFAULT, "auxiliary"], epochs=1, seed=42,
    )
    assert calls == ["train", "predict"]
    keys = ["inchikey_full", "fold", "target"]
    assert_frame_equal(predictions.select(keys).sort(keys), truth.select(keys).sort(keys))
    assert predictions["pred"].to_list() == [6.5] * 4


@pytest.mark.parametrize("table_name", ["train_df", "val_df", "heldout_df"])
@pytest.mark.parametrize(
    "case, error_template",
    [
        ("missing_id", "{table} is missing required columns"),
        ("missing_smiles", "{table} is missing required columns"),
        ("duplicate_id", "{table} contains duplicate molecule IDs"),
        ("null_id", "{table}.inchikey_full contains null values"),
        ("blank_id", "{table}.inchikey_full contains blank values"),
        ("empty", "{table} is empty"),
    ],
)
def test_adapter_rejects_invalid_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    table_name: str,
    case: str,
    error_template: str,
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
        **{target: [5.0, 6.0, 7.0] for target in targets},
    })

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],
        "inchikey_block": ["HELDOUT_A", "HELDOUT_B"],
        "SMILES": ["CCCC", "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    val_df = pl.DataFrame({
        "inchikey_full": ["VAL_A-full"],
        "inchikey_block": ["VAL_A"],
        "SMILES": ["CCO"],
        **{target: [5.8] for target in targets},
    })

    truth = pl.DataFrame([
        {
            "inchikey_full": row["inchikey_full"],
            "fold": 0, 
            "target": target,
            "truth": row[target],
        }
        for row in heldout_df.iter_rows(named=True)
        for target in targets
    ])

    tables = {
        "train_df": train_df,
        "val_df": val_df,
        "heldout_df": heldout_df
    }

    df = tables[table_name]

    if case == "missing_id":
        df = df.drop("inchikey_full")
    elif case == "missing_smiles":
        df = df.drop("SMILES")
    elif case == "duplicate_id":
        df = pl.concat([df, df.head(1)])
    elif case == "null_id":
        df = df.with_columns(
            pl.lit(None, dtype=pl.String).alias("inchikey_full")
        )
    elif case == "blank_id":
        df = df.with_columns(
            pl.lit(" ").alias("inchikey_full")
        )
    elif case == "empty":
        df = df.head(0)

    tables[table_name] = df

    def fail_if_called(*args, **kwargs):
        pytest.fail("Chemprop must not run for invalid inputs")

    monkeypatch.setattr(
        "models.oof_chemprop.subprocess.run",
        fail_if_called,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(error_template.format(table=table_name)),
    ):
        _generate_chemprop_fold_predictions(
            train_df=tables["train_df"],
            val_df=tables["val_df"],
            heldout_df=tables["heldout_df"],
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
            work_dir=tmp_path / "adapter",
            target_columns=targets,
            epochs=2,
            seed=42,
        )

@pytest.mark.parametrize(
    "val_id, heldout_id, expected_error",
    [
        ("VAL_A", "TRAIN_C", "Training and heldout molecules overlap"),
        ("TRAIN_C", "HELDOUT_A", "Training and validation molecules overlap"),
        ("VAL_A", "VAL_A", "Validation and heldout molecules overlap"),
    ],
    ids=["train-heldout", "train-val", "val-heldout"],
)
def test_adapter_rejects_overlapping_populations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    val_id: str,
    heldout_id: str,
    expected_error: str,
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

    # Make shared IDs represent the same structure.
    smiles_by_id = {
        "TRAIN_C": "CCC",
        "VAL_A": "CCO",
        "HELDOUT_A": "CCCC",
    }

    heldout_df = pl.DataFrame({
        "inchikey_full": ["HELDOUT_A-full", "HELDOUT_B-full"],  
        "inchikey_block": [heldout_id, "HELDOUT_B"],      
        "SMILES": [smiles_by_id[heldout_id], "CCCCC"],
        **{target: [5.5, 6.5] for target in targets},
    })

    val_df = pl.DataFrame({
        "inchikey_full": ["VAL_A-full"],
        "inchikey_block": [val_id],
        "SMILES": [smiles_by_id[val_id]],
        **{target: [5.8] for target in targets},
    })

    truth = pl.DataFrame([
        {
            "inchikey_full": row["inchikey_full"],
            "fold": 0, 
            "target": target,
            "truth": row[target],
        }
        for row in heldout_df.iter_rows(named=True)
        for target in targets
    ])

    def fail_if_called(*args, **kwargs):
        pytest.fail("Chemprop must not run for invalid inputs")

    monkeypatch.setattr(
        "models.oof_chemprop.subprocess.run",
        fail_if_called,
    )

    with pytest.raises(
        ValueError,
        match=expected_error,
    ):
        _generate_chemprop_fold_predictions(
            train_df=train_df,
            val_df=val_df,
            heldout_df=heldout_df,
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
            work_dir=tmp_path / "adapter",
            target_columns=targets,
            epochs=2,
            seed=42
        )

def test_adapter_rejects_nonempty_work_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
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

    val_df = pl.DataFrame({
        "inchikey_full": ["VAL_A-full"],
        "inchikey_block": ["VAL_A"],
        "SMILES": ["CCCCCCCC"],
        **{target: [5.8] for target in targets},
    })

    truth = pl.DataFrame([
        {
            "inchikey_full": row["inchikey_full"],
            "fold": 0, 
            "target": target,
            "truth": row[target],
        }
        for row in heldout_df.iter_rows(named=True)
        for target in targets
    ])

    def fail_if_called(*args, **kwargs):
        pytest.fail("Chemprop must not run for invalid inputs")

    monkeypatch.setattr(
        "models.oof_chemprop.subprocess.run",
        fail_if_called,
    )

    work_dir = tmp_path/ "adapter"
    work_dir.mkdir()

    existing_file = work_dir / "existing.txt"
    existing_file.write_text("previous run", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="work_dir must be empty for a new training attempt",
    ):
        _generate_chemprop_fold_predictions(
            train_df=train_df,
            val_df=val_df,
            heldout_df=heldout_df,
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
            work_dir=work_dir,
            target_columns=targets,
            epochs=2,
            seed=42
        )

    assert existing_file.read_text(encoding="utf-8") == "previous run"
    assert not (work_dir / "training.csv").exists()

def test_adapter_raises_when_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
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

    val_df = pl.DataFrame({
        "inchikey_full": ["VAL_A-full"],
        "inchikey_block": ["VAL_A"],
        "SMILES": ["CCCCCCCC"],
        **{target: [5.8] for target in targets},
    })

    truth = pl.DataFrame([
        {
            "inchikey_full": row["inchikey_full"],
            "fold": 0, 
            "target": target,
            "truth": row[target],
        }
        for row in heldout_df.iter_rows(named=True)
        for target in targets
    ])

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1])
        assert cmd[1] == "train", "Prediction must not start without a model"

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )
    
    monkeypatch.setattr(
        "models.oof_chemprop.subprocess.run",
        fake_run,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Chemprop training did not produce a model",
    ):
        _generate_chemprop_fold_predictions(
            train_df=train_df,
            val_df=val_df,
            heldout_df=heldout_df,
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
            work_dir=tmp_path / "adapter",
            target_columns=targets,
            epochs=2,
            seed=42
        )

    assert calls == ["train"]

def test_adapter_raises_on_prediction_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
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


    val_df = pl.DataFrame({
        "inchikey_full": ["VAL_A-full"],
        "inchikey_block": ["VAL_A"],
        "SMILES": ["CCCCCCCC"],
        **{target: [5.8] for target in targets},
    })

    truth = pl.DataFrame([
        {
            "inchikey_full": row["inchikey_full"],
            "fold": 0, 
            "target": target,
            "truth": row[target],
        }
        for row in heldout_df.iter_rows(named=True)
        for target in targets
    ])

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1])

        if cmd[1] == "train":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            model_path = output_dir / "model_0" / "best.pt"
            model_path.parent.mkdir(parents=True)
            model_path.touch()

        elif cmd[1] == "predict":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="Simulated prediction failure",
            )

        else:
            pytest.fail(f"Unexpected command: {cmd}")

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        "models.oof_chemprop.subprocess.run",
        fake_run,
    )

    with pytest.raises(RuntimeError, match="Chemprop prediction failed"):
        _generate_chemprop_fold_predictions(
            train_df=train_df,
            val_df=val_df,
            heldout_df=heldout_df,
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
            work_dir=tmp_path / "adapter",
            target_columns=targets,
            epochs=2,
            seed=42
        )

    assert calls == ["train", "predict"]

def test_adapter_raises_when_predictions_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
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


    val_df = pl.DataFrame({
        "inchikey_full": ["VAL_A-full"],
        "inchikey_block": ["VAL_A"],
        "SMILES": ["CCCCCCCC"],
        **{target: [5.8] for target in targets},
    })

    truth = pl.DataFrame([
        {
            "inchikey_full": row["inchikey_full"],
            "fold": 0, 
            "target": target,
            "truth": row[target],
        }
        for row in heldout_df.iter_rows(named=True)
        for target in targets
    ])

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1])

        if cmd[1] == "train":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            model_path = output_dir / "model_0" / "best.pt"
            model_path.parent.mkdir(parents=True)
            model_path.touch()

        elif cmd[1] == "predict":
            pass

        else:
            pytest.fail(f"Unexpected command: {cmd}")

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        "models.oof_chemprop.subprocess.run",
        fake_run,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Chemprop did not produce predictions",
    ):
        _generate_chemprop_fold_predictions(
            train_df=train_df,
            val_df=val_df,
            heldout_df=heldout_df,
            truth=truth,
            model_id="chemprop_test",
            heldout_fold=0,
            work_dir=tmp_path / "adapter",
            target_columns=targets,
            epochs=2,
            seed=42
        )

    assert calls == ["train", "predict"]
    
def test_adapter_returns_converted_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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


    val_df = pl.DataFrame({
        "inchikey_full": ["VAL_A-full"],
        "inchikey_block": ["VAL_A"],
        "SMILES": ["CCCCCCCC"],
        **{target: [5.8] for target in targets},
    })

    truth = pl.DataFrame([
        {
            "inchikey_full": row["inchikey_full"],
            "fold": 0, 
            "target": target,
            "truth": row[target],
        }
        for row in heldout_df.iter_rows(named=True)
        for target in targets
    ])

    calls = []

    native_predictions = heldout_df.select(
        "inchikey_full", "SMILES"
    ).with_columns([
        pl.Series(target, [1.0 +i, 2.0+i])
        for i, target in enumerate(targets)
    ])

    def fake_run(cmd, **kwargs):
        calls.append(cmd[1])

        if cmd[1] == "train":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            model_path = output_dir / "model_0" / "best.pt"
            model_path.parent.mkdir(parents=True)
            model_path.touch()

        elif cmd[1] == "predict":
            predictions_path = Path(cmd[cmd.index("--preds-path") + 1])
            native_predictions.write_csv(predictions_path)

        else:
            pytest.fail(f"Unexpected command: {cmd}")

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        "models.oof_chemprop.subprocess.run",
        fake_run,
    )

    predictions = _generate_chemprop_fold_predictions(
        train_df=train_df,
        val_df=val_df,
        heldout_df=heldout_df,
        truth=truth,
        model_id="chemprop_test",
        heldout_fold=0,
        work_dir=tmp_path / "adapter",
        target_columns=targets,
        epochs=2,
        seed=42
    )

    assert calls == ["train", "predict"]

    expected_values = {
        (molecule, target): value
        for i, target in enumerate(targets)
        for molecule, value in [
            ("HELDOUT_A-full", 1.0 + i),
            ("HELDOUT_B-full", 2.0 + i),
        ]
    }

    actual_values = {
        (row["inchikey_full"], row["target"]): row["pred"]
        for row in predictions.iter_rows(named=True)
    }

    assert actual_values == expected_values

    assert predictions.height == 8
    
    assert set(predictions.columns) == {
        "model_id", "inchikey_full", "fold", "target", "pred"
    }

    assert predictions["model_id"].to_list() == ["chemprop_test"] * 8

    # Exact molecule/fold/target coverage, including no duplicate rows.
    keys = ["inchikey_full", "fold", "target"]
    assert_frame_equal(
        predictions.select(keys).sort(keys),
        truth.select(keys).sort(keys),
    )

    assert predictions["pred"].dtype.is_numeric()
    assert predictions["pred"].null_count() == 0
    assert np.isfinite(predictions["pred"].to_numpy()).all()


@pytest.fixture
def training_pool():
    return pl.DataFrame({
        "inchikey_full": ["A-full", "B-full", "C-full", "D-full", "E-full", "F-full"],
        "inchikey_block": ["A", "A", "C", "D", "E", "F"],
        "group_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
        **{
            target: [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]
            for target in _SCORED_DEFAULT
        },
    })

def test_split_internal_validation_preserves_rows_and_groups(training_pool):
    train_df, val_df = _split_internal_validation(
        training_pool, 
        group_column="group_id",
        validation_fraction=0.3,
        seed=42,
    )

    train_groups = set(train_df["group_id"].to_list())
    val_groups = set(val_df["group_id"].to_list())

    assert train_groups.isdisjoint(val_groups)

    combined = pl.concat([train_df, val_df])

    assert combined.height == training_pool.height
    assert combined["inchikey_full"].n_unique() == combined.height
    assert set(combined["inchikey_full"]) == set(training_pool["inchikey_full"])
    assert set(train_df["inchikey_block"]).isdisjoint(
        set(val_df["inchikey_block"])
)

    assert not train_df.is_empty()
    assert not val_df.is_empty()

def test_split_internal_validation_is_reproducible(training_pool):
    train_1, val_1 = _split_internal_validation(
        training_pool, 
        group_column="group_id",
        validation_fraction=0.3,
        seed=42,
    )

    train_2, val_2 = _split_internal_validation(
        training_pool, 
        group_column="group_id",
        validation_fraction=0.3,
        seed=42,
    )

    assert_frame_equal(train_1, train_2)
    assert_frame_equal(val_1, val_2)

def test_split_internal_validation_ignores_input_row_order(training_pool):
    train_1, val_1 = _split_internal_validation(
        training_pool, 
        group_column="group_id",
        validation_fraction=0.3,
        seed=42,
    )

    shuffled_pool = training_pool.sample(fraction=1.0, shuffle=True, seed=123)

    train_2, val_2 = _split_internal_validation(
        shuffled_pool, 
        group_column="group_id",
        validation_fraction=0.3,
        seed=42,
    )

    assert_frame_equal(
        train_1.sort("inchikey_full"),
        train_2.sort("inchikey_full"),
    )
    assert_frame_equal(
        val_1.sort("inchikey_full"),
        val_2.sort("inchikey_full"),
    )

def test_split_internal_validation_selects_closest_shuffled_prefix():
    training_pool = pl.DataFrame({
        "inchikey_full": [f"{letter}-full" for letter in "ABCDEFGHIJ"],
        "inchikey_block": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        "group_id": ["g1"] * 5 + ["g2"] * 2 + ["g3"] * 3, 
        **{
            target: [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
            for target in _SCORED_DEFAULT
        },
    })

    train_df, val_df = _split_internal_validation(
        training_pool, 
        group_column="group_id",
        validation_fraction=0.3,
        seed=42,
    )

    assert set(val_df["group_id"].to_list()) == {"g2"}
    assert val_df.height == 2


@pytest.mark.parametrize(
    "case, bad_value, expected_error",
    [
        ("fraction", 0.0, "validation_fraction must"),
        ("fraction", 1.0, "validation_fraction must"),
        ("fraction", -0.1, "validation_fraction must"),
        ("fraction", float("nan"), "validation_fraction must"),
        ("fraction", float("inf"), "validation_fraction must"),
        ("fraction", True, "validation_fraction must"),
        ("seed", True, "seed must be an integer"),
        ("seed", 1.5, "seed must be an integer"),
        ("seed", "42", "seed must be an integer"),
        ("missing_column", "group_id", "Missing required columns"),
        ("missing_column", "inchikey_full", "Missing required columns"),
        ("null_id", "inchikey_full", "contains null values"),
        ("null_id", "group_id", "contains null values"),
        ("blank_id", "inchikey_full", "contains blank values"),
        ("blank_id", "group_id", "contains blank values"),
        ("duplicate", None, "duplicate molecule IDs"),
        ("one_group", None, "At least two groups"),
        ("empty", None, "training_pool is empty"),
    ],
)
def test_split_internal_validation_rejects_invalid_inputs(
    training_pool, 
    case, 
    bad_value, 
    expected_error
):
    validation_fraction = 0.3
    seed = 42

    if case == "fraction":
        validation_fraction = bad_value
    elif case == "seed":
        seed = bad_value
    elif case == "missing_column":
        training_pool = training_pool.drop(bad_value)
    elif case == "null_id":
        training_pool = training_pool.with_columns(
            pl.lit(None, dtype=pl.String).alias(bad_value)
        )
    elif case == "blank_id":
        training_pool = training_pool.with_columns(
            pl.lit(" ").alias(bad_value)
        )
    elif case == "duplicate":
        training_pool = pl.concat([training_pool, training_pool.head(1)])
    elif case == "one_group":
        training_pool = training_pool.with_columns(
            pl.lit("g1").alias("group_id")
        )
    elif case == "empty":
        training_pool = training_pool.head(0)

    with pytest.raises(ValueError, match=expected_error):
        _split_internal_validation(
            training_pool,
            group_column="group_id",
            validation_fraction=validation_fraction,
            seed=seed,
        )

@pytest.mark.parametrize(
    "subset, expected_error",
    [
        ("train", "No training labels for scored target"),
        ("val", "Validation data contains no observed target labels"),
    ],
)
@pytest.mark.parametrize(
    "missing_value",
    [None, float("nan")],
    ids=["null", "nan"],
)
def test_split_internal_validation_rejects_unusable_label_coverage(
    training_pool,
    subset,
    expected_error,
    missing_value,
):
    if subset == "train":
        affected_rows = pl.col("group_id") != "g2"
        affected_targets = [_SCORED_DEFAULT[0]]
    else:
        affected_rows = pl.col("group_id") == "g2"
        affected_targets = _SCORED_DEFAULT

    training_pool = training_pool.with_columns([
        pl.when(affected_rows)
        .then(pl.lit(missing_value, dtype=pl.Float64))
        .otherwise(pl.col(target))
        .alias(target)
        for target in affected_targets
    ])

    with pytest.raises(ValueError, match=expected_error):
        _split_internal_validation(
            training_pool, 
            group_column="group_id",
            validation_fraction=0.3,
            seed=42,
        )

@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf")])
def test_split_internal_validation_rejects_infinite_labels(
    training_pool,
    bad_value,
):
    training_pool = training_pool.with_columns(
        pl.lit(bad_value).alias(_SCORED_DEFAULT[0])
    )
    
    with pytest.raises(ValueError, match="contains infinite values"):
        _split_internal_validation(
            training_pool, 
            group_column="group_id",
            validation_fraction=0.3,
            seed=42,
        )


@pytest.fixture
def chemprop_config():
    return {
        "epochs": 20,
        "seed": 42,
        "from_foundation": None,
        "target_columns": list(_SCORED_DEFAULT),
    }

def test_resolve_chemprop_config_valid(chemprop_config) -> None:
    config = chemprop_config

    original = deepcopy(config)

    resolved = _resolve_chemprop_config(config)

    assert resolved == {
        **original, 
        "training_timeout": 300,
        "prediction_timeout": 300,
        "accelerator": "cpu",
        "devices": 1,
        "num_workers": 0,
        "batch_size": 4,
        "warmup_epochs": 0,
    }

    assert config == original
    assert resolved["target_columns"] is not config["target_columns"]

    overridden = {
        **original,
        "training_timeout": 1200,
        "prediction_timeout": 600,
    }
    
    assert _resolve_chemprop_config(overridden) == {
        **resolved,
        **overridden,
    }


@pytest.mark.parametrize(
    "case, bad_value, expected_error",
    [
        ("config", None, "config must be a dictionary"),
        ("config", [], "config must be a dictionary"),
        ("key", 123, "config keys must be strings"),
        ("unknown", "epohcs", "config includes unexpected keys"),

        ("missing", "epochs", "config must include"),
        ("missing", "seed", "config must include"),
        ("missing", "from_foundation", "config must include"),
        ("missing", "target_columns", "config must include"),

        ("epochs", 0, "epochs must"),
        ("epochs", -1, "epochs must"),
        ("epochs", True, "epochs must"),
        ("epochs", 1.5, "epochs must"),
        ("epochs", "20", "epochs must"),

        ("seed", True, "Seed must"),
        ("seed", 1.5, "Seed must"),
        ("seed", "42", "Seed must"),

        ("training_timeout", 0, "training_timeout must"),
        ("training_timeout", -1, "training_timeout must"),
        ("training_timeout", True, "training_timeout must"),
        ("training_timeout", "300", "training_timeout must"),
        ("training_timeout", float("nan"), "training_timeout must"),
        ("training_timeout", float("inf"), "training_timeout must"),

        ("prediction_timeout", 0, "prediction_timeout must"),
        ("prediction_timeout", -1, "prediction_timeout must"),
        ("prediction_timeout", True, "prediction_timeout must"),
        ("prediction_timeout", "300", "prediction_timeout must"),
        ("prediction_timeout", float("nan"), "prediction_timeout must"),
        ("prediction_timeout", float("inf"), "prediction_timeout must"),

        ("target_columns", None, "must be a nonempty list"),
        ("target_columns", "CYP1A2", "must be a nonempty list"),
        ("target_columns", [], "must be a nonempty list"),
        ("target_columns", [*_SCORED_DEFAULT, None], "nonblank strings"),
        ("target_columns", [*_SCORED_DEFAULT, " "], "nonblank strings"),
        (
            "target_columns",
            [*_SCORED_DEFAULT, _SCORED_DEFAULT[0]],
            "duplicate names",
        ),
        (
            "target_columns",
            [*_SCORED_DEFAULT, "SMILES"],
            "conflict with metadata",
        ),
        ("target_columns", list(_SCORED_DEFAULT[1:]), "Missing scored targets"),
    ],
)
def test_resolve_chemprop_config_rejects_invalid_inputs(
    chemprop_config,
    case, 
    bad_value,
    expected_error,
) -> None:
    config = deepcopy(chemprop_config)

    if case == "config":
        config = bad_value
    elif case in ("key", "unknown"):
        config[bad_value] = 123
    elif case == "missing":
        del config[bad_value]
    else:
        config[case] = bad_value

    with pytest.raises(ValueError, match=expected_error):
        _resolve_chemprop_config(config)

  
@pytest.mark.parametrize(
    "case",
    [
        "none",
        "chemeleon",
        "existing_file",
        "missing_file",
        "directory",
        "blank",
        "wrong_type",
    ],
)
def test_resolve_chemprop_config_foundation(
    chemprop_config,
    case,
    tmp_path: Path,
) -> None:
    config = chemprop_config.copy()
    expected_error = None

    if case == "none":
        foundation = None
        expected = None

    elif case == "chemeleon":
        foundation = "ChEmElEoN"
        expected = "chemeleon"

    elif case == "existing_file":
        checkpoint = tmp_path / "foundation.pt"
        checkpoint.touch()
        foundation = str(checkpoint)
        expected = str(checkpoint.resolve())

    elif case == "missing_file":
        foundation = str(tmp_path / "missing.pt")
        expected_error = FileNotFoundError

    elif case == "directory":
        foundation = str(tmp_path)
        expected_error = FileNotFoundError

    elif case == "blank":
        foundation = " "
        expected_error = ValueError

    elif case == "wrong_type":
        foundation = 123
        expected_error = ValueError

    else: 
        pytest.fail(f"Unexpected case: {case}")

    config["from_foundation"] = foundation

    if expected_error is None:
        resolved = _resolve_chemprop_config(config)
        assert resolved["from_foundation"] == expected
    else:
        message = (
            "Foundation checkpoint does not exist"
            if expected_error is FileNotFoundError
            else "from_foundation must be"
        )
        with pytest.raises(expected_error, match=message):
            _resolve_chemprop_config(config)



@pytest.fixture
def loader_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    data = pl.DataFrame({
        "inchikey_full": ["A-full", "B-full", "C-full", "D-full", "E-full", "F-full"],
        "inchikey_block": ["A", "B", "C", "D", "E", "F"],
        "SMILES": ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"],
        "CYP1A2_pIC50_direct_inhibition": [5.0, None, 6.0, 7.0, 7.5, 8.0],
        "CYP2C9_pIC50_direct_inhibition": [5.0, 6.0, 7.0, 8.0, 8.5, 9.0],
        "CYP2D6_pIC50_direct_inhibition": [4.0, 5.0, 6.0, 7.0, 8.0, 8.5],
        "CYP3A4_pIC50_direct_inhibition": [6.0, 7.0, 8.0, 9.0, 9.5, 10.0],
        "group_id" : ["g1", "g2", "g3", "g4", "g5", "g6"]
    })

    folds = pl.DataFrame({
        "inchikey_full": ["A-full", "B-full", "C-full", "D-full", "E-full", "F-full"],
        "fold": [0, 0, 1, 1, 2, 2], 
    })

    return data, folds

@pytest.fixture
def fake_chemprop(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "train":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            model_path = output_dir / "model_0" / "best.pt"
            model_path.parent.mkdir(parents=True)
            model_path.touch()

        elif cmd[1] == "predict":
            heldout_path = Path(cmd[cmd.index("--test-path") + 1])
            heldout = pl.read_csv(heldout_path)

            native_predictions = heldout.select(
                "inchikey_full", "SMILES"
            ).with_columns([
                pl.lit(5.0 + i).alias(target)
                for i, target in enumerate(_SCORED_DEFAULT)
            ])

            predictions_path = Path(cmd[cmd.index("--preds-path") + 1])
            native_predictions.write_csv(predictions_path)

        else:
            pytest.fail(f"Unexpected command: {cmd}")

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        "models.oof_chemprop.subprocess.run",
        fake_run,
    )

    return calls

def test_run_oof_chemprop_combines_folds_without_leakage(
    tmp_path,
    loader_tables,
    chemprop_config,
    fake_chemprop,
):
    data, folds = loader_tables

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    data.write_csv(data_path)
    folds.write_csv(folds_path)

    predictions, truth = run_oof(
        data_path=data_path,
        folds_path=folds_path,
        model_name="chemprop",
        model_id="chemprop_test",
        output_dir=tmp_path / "oof",
        chemprop_config=chemprop_config,
        group_column="group_id",
        validation_fraction=0.3,
        validation_seed=42,
    )


    assert predictions.height == 23
    assert truth.height == 23

    keys = ["inchikey_full", "fold", "target"]
    assert_frame_equal(
        predictions.select(keys).sort(keys),
        truth.select(keys).sort(keys),
    )
    
    assert [cmd[1] for cmd in fake_chemprop] == [
        "train", "predict",
        "train", "predict",
        "train", "predict"
    ]

    all_ids = set(data["inchikey_full"].to_list())
    fold_ids = folds["fold"].unique().sort().to_list()

    for i, fold in enumerate(fold_ids): 
        # Commands alternate: train, predict, train, predict
        train_cmd = fake_chemprop[2 * i]
        predict_cmd = fake_chemprop[2 * i + 1]

        training_path = Path(
            train_cmd[train_cmd.index("--data-path") + 1]
        )
        heldout_path = Path(
            predict_cmd[predict_cmd.index("--test-path") + 1]
        )

        training = pl.read_csv(training_path)
        heldout = pl.read_csv(heldout_path)

        train_ids = set(
            training.filter(pl.col("split") == "train")
            ["inchikey_full"].to_list()
        )
        val_ids = set(
            training.filter(pl.col("split") == "val")
            ["inchikey_full"].to_list()
        )
        heldout_ids = set(heldout["inchikey_full"].to_list())

        expected_fold_ids = set(
            folds.filter(pl.col("fold") == fold)
            ["inchikey_full"].to_list()
        )

        assert train_ids
        assert val_ids
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(heldout_ids)
        assert val_ids.isdisjoint(heldout_ids)

        assert heldout_ids == expected_fold_ids
        assert train_ids | val_ids == all_ids - expected_fold_ids
        

def test_run_oof_chemprop_resumes_completed_folds(
    tmp_path,
    loader_tables,
    chemprop_config,
    fake_chemprop,
):
    data, folds = loader_tables
    
    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"
    output_dir = tmp_path / "oof"


    data.write_csv(data_path)
    folds.write_csv(folds_path)

    predictions_1, truth_1 = run_oof(
        data_path=data_path,
        folds_path=folds_path,
        model_name="chemprop",
        model_id="chemprop_test",
        output_dir=output_dir,
        chemprop_config=chemprop_config,
        group_column="group_id",
        validation_fraction=0.3,
        validation_seed=42,
    )

    calls_before = len(fake_chemprop)
    attempts_before = set(output_dir.glob("fold-*/attempt-*"))

    predictions_2, truth_2 = run_oof(
        data_path=data_path,
        folds_path=folds_path,
        model_name="chemprop",
        model_id="chemprop_test",
        output_dir=output_dir,
        chemprop_config=chemprop_config,
        group_column="group_id",
        validation_fraction=0.3,
        validation_seed=42,
    )

    assert len(fake_chemprop) == calls_before
    assert set(output_dir.glob("fold-*/attempt-*")) == attempts_before

    assert_frame_equal(predictions_1, predictions_2)
    assert_frame_equal(truth_1, truth_2)

@pytest.mark.parametrize(
    "case, replacement",
    [
        ("epochs", 21),
        ("seed", 43),
        ("validation_fraction", 0.5),
        ("validation_seed", 43),
        ("checkpoint_contents", b"updated checkpoint"),
    ],
    ids=[
        "training-epochs",
        "training-seed",
        "validation-fraction",
        "validation-seed",
        "checkpoint-contents",
    ],
)
def test_run_oof_chemprop_invalidates_changed_inputs(
    tmp_path,
    loader_tables,
    chemprop_config,
    fake_chemprop,
    case,
    replacement,
):
    data, folds = loader_tables

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    data.write_csv(data_path)
    folds.write_csv(folds_path)

    config = deepcopy(chemprop_config)

    if case == "checkpoint_contents":
        checkpoint = tmp_path / "foundation.pt"
        checkpoint.write_bytes(b"original checkpoint")
        config["from_foundation"] = str(checkpoint)

    validation_fraction = 0.3
    validation_seed = 42

    predictions_1, truth_1 = run_oof(
        data_path=data_path,
        folds_path=folds_path,
        model_name="chemprop",
        model_id="chemprop_test",
        output_dir=tmp_path / "oof",
        chemprop_config=config,
        group_column="group_id",
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
    )

    output_dir = tmp_path / "oof"
    calls_before = len(fake_chemprop)
    attempts_before = {
        fold: set((output_dir / f"fold-{fold}").glob("attempt-*"))
        for fold in range(3)
    }

    assert calls_before == 6

    if case in {"epochs", "seed"}:
        config[case] = replacement
    elif case == "validation_fraction":
        validation_fraction = replacement
    elif case == "validation_seed":
        validation_seed = replacement
    elif case == "checkpoint_contents":
        checkpoint.write_bytes(replacement)

    predictions_2, truth_2 = run_oof(
        data_path=data_path,
        folds_path=folds_path,
        model_name="chemprop",
        model_id="chemprop_test",
        output_dir=tmp_path / "oof",
        chemprop_config=config,
        group_column="group_id",
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
    )

    new_calls = fake_chemprop[calls_before:]

    assert [cmd[1] for cmd in new_calls] == [
        "train", "predict",
        "train", "predict",
        "train", "predict",
    ]

    new_training_folds = [
        Path(cmd[cmd.index("--output-dir") + 1]).parent.parent.name
        for cmd in new_calls
        if cmd[1] == "train"
    ]
    assert new_training_folds == ["fold-0", "fold-1", "fold-2"]

    for fold in range(3):
        attempts_after = set(
            (output_dir / f"fold-{fold}").glob("attempt-*")
        )
        assert attempts_before[fold] < attempts_after
        assert len(attempts_after - attempts_before[fold]) == 1

    assert_frame_equal(truth_1, truth_2)
    keys = ["inchikey_full", "fold", "target"]
    assert_frame_equal(
        predictions_2.select(keys).sort(keys),
        truth_2.select(keys).sort(keys),
    )


def test_run_oof_chemprop_retries_failed_fold(
    tmp_path,
    loader_tables,
    chemprop_config,
    fake_chemprop,
    monkeypatch,
):
    successful_run = subprocess.run 
    failed_once = False

    def fake_run(cmd, **kwargs):
        nonlocal failed_once

        if cmd[1] == "train":
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])

            if "fold-1" in output_dir.parts and not failed_once:
                failed_once = True
                fake_chemprop.append(list(cmd))
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="",
                    stderr="Simulated training failure",
                )

        return successful_run(cmd, **kwargs)

    monkeypatch.setattr(
        "models.oof_chemprop.subprocess.run",
        fake_run,
    )
    
    data, folds = loader_tables

    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"

    data.write_csv(data_path)
    folds.write_csv(folds_path)

    with pytest.raises(RuntimeError, match="Chemprop training failed"):
        run_oof(
            data_path=data_path,
            folds_path=folds_path,
            model_name="chemprop",
            model_id="chemprop_test",
            output_dir=tmp_path / "oof",
            chemprop_config=chemprop_config,
            group_column="group_id",
            validation_fraction=0.3,
            validation_seed=42,
        )

    output_dir = tmp_path / "oof"

    assert (output_dir / "fold-0" / "complete.json").is_file()
    assert not (output_dir / "fold-1" / "complete.json").exists()

    attempts_before = set((output_dir / "fold-1").glob("attempt-*"))
    calls_before = len(fake_chemprop)

    predictions, truth = run_oof(
        data_path=data_path,
        folds_path=folds_path,
        model_name="chemprop",
        model_id="chemprop_test",
        output_dir=tmp_path / "oof",
        chemprop_config=chemprop_config,
        group_column="group_id",
        validation_fraction=0.3,
        validation_seed=42,
    )

    new_training_folds = [
        Path(cmd[cmd.index("--output-dir") + 1]).parent.parent.name
        for cmd in fake_chemprop[calls_before:]
        if cmd[1] == "train"
    ]
    assert new_training_folds == ["fold-1", "fold-2"]

    attempts_after = set((output_dir / "fold-1").glob("attempt-*"))
    assert attempts_before < attempts_after
    assert len(attempts_after - attempts_before) == 1


def test_run_oof_chemprop_timeout_changes_preserve_resume(
    tmp_path,
    loader_tables,
    chemprop_config,
    fake_chemprop,
):
    data, folds = loader_tables
    
    data_path = tmp_path / "data.csv"
    folds_path = tmp_path / "folds.csv"
    output_dir = tmp_path / "oof"


    data.write_csv(data_path)
    folds.write_csv(folds_path)


    overridden = {
        **chemprop_config,
        "training_timeout": 1200,
        "prediction_timeout": 600,
    }
    

    predictions_1, truth_1 = run_oof(
        data_path=data_path,
        folds_path=folds_path,
        model_name="chemprop",
        model_id="chemprop_test",
        output_dir=output_dir,
        chemprop_config=chemprop_config,
        group_column="group_id",
        validation_fraction=0.3,
        validation_seed=42,
    )

    calls_before = len(fake_chemprop)
    attempts_before = set(output_dir.glob("fold-*/attempt-*"))

    predictions_2, truth_2 = run_oof(
        data_path=data_path,
        folds_path=folds_path,
        model_name="chemprop",
        model_id="chemprop_test",
        output_dir=output_dir,
        chemprop_config=overridden,
        group_column="group_id",
        validation_fraction=0.3,
        validation_seed=42,
    )

    assert len(fake_chemprop) == calls_before
    assert set(output_dir.glob("fold-*/attempt-*")) == attempts_before

    assert_frame_equal(predictions_1, predictions_2)
    assert_frame_equal(truth_1, truth_2)

    assert calls_before == 6
