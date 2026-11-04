# tests/test_chemeleon_loading.py
"""CheMeleon checkpoint location and loading. Never downloads."""

import importlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from representations import chemeleon


def _reload(monkeypatch, chemprop_home):
    if chemprop_home is None:
        monkeypatch.delenv("CHEMPROP_HOME", raising=False)
    else:
        monkeypatch.setenv("CHEMPROP_HOME", chemprop_home)
    return importlib.reload(chemeleon)


@pytest.fixture(autouse=True)
def _restore_module(monkeypatch):
    yield
    monkeypatch.undo()
    importlib.reload(chemeleon)


def test_default_checkpoint_dir_is_relative_chemeleon(monkeypatch):
    mod = _reload(monkeypatch, None)
    assert mod._CHECKPOINT_DIR == Path(".chemeleon")
    assert mod._CHECKPOINT_PATH == Path(".chemeleon/chemeleon_mp.pt")


def test_empty_chemprop_home_falls_back_to_default(monkeypatch):
    assert _reload(monkeypatch, "")._CHECKPOINT_DIR == Path(".chemeleon")


def test_chemprop_home_overrides_dir(monkeypatch, tmp_path):
    mod = _reload(monkeypatch, str(tmp_path))
    assert mod._CHECKPOINT_PATH == tmp_path / "chemeleon_mp.pt"


def test_truncated_checkpoint_is_not_redownloaded_and_fails_to_load(monkeypatch, tmp_path):
    """Reproduces the cluster failure: a partial file exists, so no download, torch.load fails."""
    pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    mod = _reload(monkeypatch, str(tmp_path))
    (tmp_path / "chemeleon_mp.pt").write_bytes(b"PK\x03\x04 truncated")

    def no_download(*args, **kwargs):
        raise AssertionError("download attempted for an existing file")

    monkeypatch.setattr(mod, "urlretrieve", no_download)
    with pytest.raises(RuntimeError, match="failed reading zip archive"):
        mod._load_model()


def test_real_checkpoint_loads_and_embeds():
    """Runs against the checkpoint at the configured path; skipped if it isn't there."""
    pytest.importorskip("torch")
    pytest.importorskip("chemprop")
    path = chemeleon._CHECKPOINT_PATH
    if not path.exists():
        pytest.skip(f"no checkpoint at {path.resolve()}")

    out = chemeleon.CheMeleon().transform(pl.Series(["CCO", "not_a_smiles"]))
    assert out.shape == (2, 2048)
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()
