"""Row alignment and NaN policy for the chemprop descriptor-file writer."""

import numpy as np
import polars as pl
import pytest

from data_tools.descriptors_out import write_descriptors


def _load(path):
    """Read back what chemprop would read: the arr_0 key of the npz."""
    return np.load(path)["arr_0"]


@pytest.fixture
def frame():
    return pl.DataFrame({"SMILES": ["c1ccncc1", "c1ccccc1", "c1c[nH]cn1"]})


def test_writes_one_row_per_input_row_in_order(tmp_path, frame):
    out, affected = write_descriptors(
        frame, ["cyp_heme_v2"], tmp_path / "d.npz",
        nan_policy="error", cache_dir=tmp_path / "cache",
    )
    arr = _load(out)
    assert arr.shape[0] == frame.height
    assert affected == 0


def test_array_is_stored_under_arr_0_as_float64(tmp_path, frame):
    """chemprop reads np.load(path)["arr_0"]; any other key is unreadable to it."""
    out, _ = write_descriptors(
        frame, ["cyp_heme_v2"], tmp_path / "d.npz",
        nan_policy="error", cache_dir=tmp_path / "cache",
    )
    with np.load(out) as z:
        assert "arr_0" in z
        assert z["arr_0"].dtype == np.float64


def test_returns_the_path_actually_written_when_suffix_is_missing(tmp_path, frame):
    """np.savez silently appends .npz, so the requested path may not be the real one."""
    out, _ = write_descriptors(
        frame, ["cyp_heme_v2"], tmp_path / "noext",
        nan_policy="error", cache_dir=tmp_path / "cache",
    )
    assert out.exists()
    assert out.suffix == ".npz"


def test_sidecar_records_the_column_order(tmp_path, frame):
    out, _ = write_descriptors(
        frame, ["cyp_heme_v2"], tmp_path / "d.npz",
        nan_policy="error", cache_dir=tmp_path / "cache",
    )
    from representations.cyp_pharmacophore import HemeCoordinationMotifs

    sidecar = out.with_name(out.name + ".columns.txt")
    names = sidecar.read_text(encoding="utf-8").split()
    assert names == HemeCoordinationMotifs().feature_names
    assert len(names) == _load(out).shape[1]


def test_stacked_inputs_concatenate_widths(tmp_path, frame):
    out, _ = write_descriptors(
        frame, ["cyp_heme_v2", "cyp_tdi_v2"], tmp_path / "d.npz",
        nan_policy="error", cache_dir=tmp_path / "cache",
    )
    assert _load(out).shape[1] == 13 + 6


def test_error_policy_refuses_to_write_when_a_row_is_nan(tmp_path):
    bad = pl.DataFrame({"SMILES": ["c1ccncc1", "not_a_smiles"]})
    target = tmp_path / "d.npz"
    with pytest.raises(ValueError, match="NaN"):
        write_descriptors(
            bad, ["cyp_heme_v2"], target,
            nan_policy="error", cache_dir=tmp_path / "cache",
        )
    assert not target.exists()


def test_fill_policy_replaces_nan_and_reports_the_count(tmp_path):
    bad = pl.DataFrame({"SMILES": ["c1ccncc1", "not_a_smiles"]})
    out, affected = write_descriptors(
        bad, ["cyp_heme_v2"], tmp_path / "d.npz",
        nan_policy="fill", fill_value=0.0, cache_dir=tmp_path / "cache",
    )
    arr = _load(out)
    assert arr.shape[0] == 2
    assert affected == 1
    assert not np.isnan(arr).any()


def test_drop_policy_reports_how_many_rows_it_removed(tmp_path):
    bad = pl.DataFrame({"SMILES": ["c1ccncc1", "not_a_smiles"]})
    out, dropped = write_descriptors(
        bad, ["cyp_heme_v2"], tmp_path / "d.npz",
        nan_policy="drop", cache_dir=tmp_path / "cache",
    )
    assert dropped == 1
    assert _load(out).shape[0] == 1


def test_drop_policy_can_emit_an_aligned_data_file(tmp_path):
    """Dropping rows desynchronizes the descriptor file from the original data CSV."""
    bad = pl.DataFrame({"SMILES": ["c1ccncc1", "not_a_smiles"]})
    data_out = tmp_path / "aligned.csv"
    out, dropped = write_descriptors(
        bad, ["cyp_heme_v2"], tmp_path / "d.npz",
        nan_policy="drop", cache_dir=tmp_path / "cache",
        also_write_data=data_out,
    )
    assert dropped == 1
    assert pl.read_csv(data_out).height == _load(out).shape[0] == 1


def test_unknown_input_name_raises_before_writing_anything(tmp_path, frame):
    target = tmp_path / "d.npz"
    with pytest.raises(ValueError):
        write_descriptors(
            frame, ["no_such_block"], target,
            nan_policy="error", cache_dir=tmp_path / "cache",
        )
    assert not target.exists()


def test_infinite_values_are_detected_not_silently_passed_through(tmp_path, monkeypatch):
    """np.isnan misses inf, and nan_to_num would turn it into ~1.8e308."""
    import data_tools.descriptors_out as do

    monkeypatch.setattr(
        do, "featurize", lambda df, names, cache: np.array([[1.0, 2.0], [np.inf, 3.0]])
    )
    frame = pl.DataFrame({"SMILES": ["c1ccncc1", "c1ccccc1"]})
    with pytest.raises(ValueError, match="NaN"):
        do.write_descriptors(
            frame, ["cyp_heme_v2"], tmp_path / "d.npz",
            nan_policy="error", cache_dir=tmp_path / "cache",
        )
    out, affected = do.write_descriptors(
        frame, ["cyp_heme_v2"], tmp_path / "f.npz",
        nan_policy="fill", fill_value=0.0, cache_dir=tmp_path / "cache",
    )
    arr = np.load(out)["arr_0"]
    assert affected == 1
    assert np.isfinite(arr).all()


def test_invalid_nan_policy_raises(tmp_path):
    frame = pl.DataFrame({"SMILES": ["c1ccncc1"]})
    with pytest.raises(ValueError, match="nan_policy"):
        write_descriptors(
            frame, ["cyp_heme_v2"], tmp_path / "d.npz",
            nan_policy="nonsense", cache_dir=tmp_path / "cache",
        )


def test_dropping_every_row_is_reported_rather_than_writing_an_empty_file(tmp_path):
    """chemprop would fail confusingly on a zero-row descriptor file."""
    bad = pl.DataFrame({"SMILES": ["not_a_smiles", "also_bad"]})
    with pytest.raises(ValueError, match="every row"):
        write_descriptors(
            bad, ["cyp_heme_v2"], tmp_path / "d.npz",
            nan_policy="drop", cache_dir=tmp_path / "cache",
        )
