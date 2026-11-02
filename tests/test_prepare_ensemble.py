import polars as pl
import pytest
from polars.testing import assert_frame_equal

from data_tools.prepare_ensemble import prepare_ensemble_data, SCORED_TARGETS
from data_tools.standardize import connectivity_keys, stereo_keys
from data_tools.prepare_ensemble import make_oof_folds
from data_tools.split_column import _scaffold_groups, _butina_groups


@pytest.fixture
def ensemble_data():
    return pl.DataFrame({
        "SMILES": ["CCO", "CCC", "CCN"],
        "split": ["train", "val", "test"],
        "aux_label": [1.0, None, 3.0],
        **{target: [5.0, None, 7.0] for target in SCORED_TARGETS},
    })


def test_prepare_generates_keys_and_preserves_data(ensemble_data):
    result = prepare_ensemble_data(ensemble_data)

    assert_frame_equal(result.select(ensemble_data.columns), ensemble_data)
    assert result["inchikey_full"].n_unique() == ensemble_data.height
    assert result["inchikey_block"].to_list() == connectivity_keys(
        ensemble_data["SMILES"].to_list()
    )
    assert result["inchikey_full"].to_list() == stereo_keys(
        ensemble_data["SMILES"].to_list()
    )


def test_prepare_accepts_matching_existing_keys(ensemble_data):
    prepared = prepare_ensemble_data(ensemble_data)

    assert_frame_equal(prepare_ensemble_data(prepared), prepared)


def test_prepare_preserves_stereoisomers_in_same_split(ensemble_data):
    data = ensemble_data.head(2).with_columns(
        pl.Series("SMILES", ["C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"]),
        pl.lit("train").alias("split"),
    )

    result = prepare_ensemble_data(data)

    assert result.height == 2
    assert result["inchikey_full"].n_unique() == 2
    assert result["inchikey_block"].n_unique() == 1
    assert_frame_equal(result.select(data.columns), data)


@pytest.mark.parametrize(
    "case, match",
    [
        ("missing_smiles", "Missing columns"),
        ("missing_target", "Missing columns"),
        ("missing_split", "Missing columns"),
        ("empty", "Data is empty"),
        ("null_smiles", "non-null strings"),
        ("blank_smiles", "cannot be blank"),
        ("invalid_smiles", "Could not generate"),
        ("wrong_block", "Existing inchikey_block"),
        ("wrong_full", "Existing inchikey_full"),
        ("duplicate_full", "Duplicate full InChIKeys"),
        ("invalid_split", "split must contain"),
        ("null_split", "split must contain"),
        ("nonnumeric_label", "must be numeric"),
        ("infinite_label", "contains infinity"),
    ],
)
def test_prepare_rejects_invalid_inputs(ensemble_data, case, match):
    data = ensemble_data
    target = SCORED_TARGETS[0]

    if case == "missing_smiles":
        data = data.drop("SMILES")
    elif case == "missing_target":
        data = data.drop(target)
    elif case == "missing_split":
        data = data.drop("split")
    elif case == "empty":
        data = data.head(0)
    elif case in {"null_smiles", "blank_smiles", "invalid_smiles"}:
        value = {
            "null_smiles": None,
            "blank_smiles": " ",
            "invalid_smiles": "not_a_smiles",
        }[case]
        data = data.with_columns(
            pl.lit(value, dtype=pl.String).alias("SMILES")
        )
    elif case in {"wrong_block", "wrong_full"}:
        column = "inchikey_block" if case == "wrong_block" else "inchikey_full"
        data = data.with_columns(pl.lit("wrong").alias(column))
    elif case == "duplicate_full":
        data = pl.concat([data, data.head(1)])
    elif case in {"invalid_split", "null_split"}:
        value = "unknown" if case == "invalid_split" else None
        data = data.with_columns(
            pl.lit(value, dtype=pl.String).alias("split")
        )
    elif case == "nonnumeric_label":
        data = data.with_columns(pl.lit("bad").alias(target))
    elif case == "infinite_label":
        data = data.with_columns(pl.lit(float("inf")).alias(target))

    with pytest.raises(ValueError, match=match):
        prepare_ensemble_data(data)


def test_prepare_rejects_block_crossing_splits(ensemble_data):
    data = ensemble_data.head(2).with_columns(
        pl.Series("SMILES", ["C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O"]),
        pl.Series("split", ["train", "val"]),
    )

    with pytest.raises(ValueError, match="multiple splits"):
        prepare_ensemble_data(data)


@pytest.fixture
def fold_data():
    smiles = [
        "C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O",
        "c1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCOC1",
        "CCN", "CCCl",
    ]
    return prepare_ensemble_data(pl.DataFrame({
        "SMILES": smiles,
        "split": ["train"] * 6 + ["val", "test"],
        **{target: [5.0] * 8 for target in SCORED_TARGETS},
    }))


@pytest.mark.parametrize("strategy", ["scaffold", "butina", "random"])
def test_folds_preserve_rows_groups_and_seed(fold_data, strategy):
    original = fold_data.clone()
    train = fold_data.filter(pl.col("split") == "train")
    result = make_oof_folds(fold_data, strategy=strategy, n_folds=2)

    assert set(result.columns) == {"inchikey_full", "fold", "oof_group"}
    assert result.height == result["inchikey_full"].n_unique() == train.height
    assert set(result["inchikey_full"]) == set(train["inchikey_full"])
    assert set(result["fold"]) == {0, 1}
    assert_frame_equal(
        result,
        make_oof_folds(fold_data, strategy=strategy, n_folds=2),
    )
    assert_frame_equal(fold_data, original)

    joined = train.join(result, on="inchikey_full", validate="1:1")
    assert joined.group_by("inchikey_block").agg(
        pl.col("fold").n_unique()
    )["fold"].max() == 1

    if strategy != "random":
        smiles = train["SMILES"].to_list()
        groups = (
            _scaffold_groups(smiles) if strategy == "scaffold"
            else _butina_groups(smiles, 0.4)
        )
        lookup = dict(result.select("inchikey_full", "fold").iter_rows())
        ids = train["inchikey_full"].to_list()
        for rows in groups.values():
            assert len({lookup[ids[i]] for i in rows}) == 1


@pytest.mark.parametrize("kwargs, message", [
    ({"n_folds": 1}, "n_folds"),
    ({"n_folds": True}, "n_folds"),
    ({"n_folds": 2.5}, "n_folds"),
    ({"n_folds": 100}, "Not enough groups"),
    ({"seed": True}, "seed"),
    ({"strategy": "unknown"}, "Unknown strategy"),
])
def test_folds_reject_invalid_settings(fold_data, kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_oof_folds(fold_data, **kwargs)


def test_folds_reject_empty_train(fold_data):
    data = fold_data.filter(pl.col("split") != "train")
    with pytest.raises(ValueError, match="No pinned training"):
        make_oof_folds(data, n_folds=2)


@pytest.mark.parametrize("population", ["held-out", "training"])
def test_folds_require_target_coverage(fold_data, population):
    folds = make_oof_folds(fold_data, strategy="random", n_folds=2)
    # Remove labels from fold 0, or from its training complement.
    missing_fold = 0 if population == "held-out" else 1
    ids = folds.filter(pl.col("fold") == missing_fold)["inchikey_full"].to_list()
    target = SCORED_TARGETS[0]
    data = fold_data.with_columns(
        pl.when(pl.col("inchikey_full").is_in(ids))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col(target))
        .alias(target)
    )

    with pytest.raises(ValueError, match=f"Fold 0: no usable {population} labels"):
        make_oof_folds(data, strategy="random", n_folds=2)