"""Materialize any registered representation as a chemprop descriptor file.

`src/models/train_chemprop.py` forwards unknown flags to chemprop's own CLI
(`train_chemprop.py:86`), so `--descriptors-path` reaches chemprop -- but nothing
in this repo wrote such a file. This does.

The chemprop contract (verified against chemprop 2.2.3 source):

* Format is a NumPy ``.npz``, NOT a CSV. chemprop does
  ``np.load(args.descriptors_path)["arr_0"]`` (cli/train.py:1030-1031;
  cli/utils/MAB_parsing.py:47). The ``(n_rows, n_features)`` array must
  therefore be the first positional array, which ``np.savez(path, arr)``
  stores as ``arr_0``.
* Row alignment is positional: descriptor row i pairs with data row i. Use
  ``also_write_data`` to emit the matching ``--data-path`` CSV when rows were
  dropped.
* ``np.savez`` silently appends ``.npz`` to a filename lacking it, so the path
  actually written is returned, not the one requested.
* chemprop scales extra descriptors by default (it offers
  ``--no-descriptor-scaling``). This writer applies NO scaling of its own and
  nothing else in this repo scales features; pass chemprop's flag if unwanted.

An ``.npz`` cannot carry column names, so ``<out>.npz.columns.txt`` is written
alongside: one feature name per line, in column order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from data_tools.inputs import INPUT_REGISTRY, featurize, is_known_input


def _feature_names(input_names: list[str]) -> list[str]:
    """Column names in hstack order; empty for blocks whose names are unknown.

    CheMeleon is deliberately omitted: instantiating it triggers a model
    download, which must not happen inside a column-naming helper. Callers fall
    back to positional ``desc_N`` names when the result has the wrong width.
    """
    import representations.cyp_pharmacophore as cp
    from representations.cyp_mech import CYPMechanisticBlock
    from representations.cyp_xtb import XTBElectronic, XTBSolvation
    from representations.maccs import MACCS
    from representations.mordred import MordredDescriptors
    from representations.morgan import CountMorganFingerprint, MorganFingerprint
    from representations.rdkit_descriptors import RDKitDescriptors

    known = [
        RDKitDescriptors(), MorganFingerprint(), CountMorganFingerprint(),
        MACCS(), MordredDescriptors(), cp.HemeCoordinationMotifs(),
        cp.IsoformPreferenceDescriptors(), cp.MechanismBasedAlerts(),
        cp.ErGFingerprint(), XTBSolvation(), XTBElectronic(),
        CYPMechanisticBlock(),
    ]
    by_name = {b.name: b for b in known}
    names: list[str] = []
    for n in input_names:
        block = by_name.get(n)
        names.extend(block.feature_names if block is not None else [])
    return names


def write_descriptors(
    df: pl.DataFrame,
    input_names: list[str],
    out_path: str | Path,
    nan_policy: str = "error",
    fill_value: float = 0.0,
    cache_dir: str | Path = Path("data/features"),
    also_write_data: str | Path | None = None,
) -> tuple[Path, int]:
    """Write a chemprop descriptor .npz. Returns (path actually written, rows affected).

    nan_policy:
      "error" -- refuse to write if any row has a non-finite value (the safe default)
      "fill"  -- replace NaN/inf with fill_value; returns the number of rows touched
      "drop"  -- remove rows containing non-finite values; returns the number removed
    """
    if nan_policy not in {"error", "fill", "drop"}:
        raise ValueError(f"nan_policy must be error/fill/drop, got {nan_policy!r}")
    unknown = [n for n in input_names if not is_known_input(n)]
    if unknown:
        raise ValueError(
            f"Unknown input(s): {unknown}. Available: {list(INPUT_REGISTRY)}"
        )

    features = np.asarray(featurize(df, input_names, Path(cache_dir)), dtype=np.float64)
    nan_rows = (~np.isfinite(features)).any(axis=1)
    n_affected = int(nan_rows.sum())

    kept = df
    if n_affected:
        if nan_policy == "error":
            raise ValueError(
                f"{n_affected} of {len(features)} rows contain non-finite values "
                "(NaN or inf), from an unparseable SMILES or a descriptor that "
                "overflowed. Choose --nan-policy fill or drop, or clean "
                "the input. Refusing to write silently-imputed data."
            )
        if nan_policy == "fill":
            features = np.nan_to_num(
                features, nan=fill_value, posinf=fill_value, neginf=fill_value
            )
            print(f"[write-descriptors] filled non-finite values in {n_affected} row(s) with {fill_value}")
        else:
            if n_affected == len(features):
                raise ValueError(
                    f"nan_policy='drop' would remove every row ({n_affected} of "
                    f"{len(features)}) because all contain non-finite values; "
                    "refusing to write a zero-row descriptor file."
                )
            features = features[~nan_rows]
            kept = df.filter(~pl.Series(nan_rows))
            print(f"[write-descriptors] dropped {n_affected} row(s) containing non-finite values")

    names = _feature_names(input_names)
    if len(names) != features.shape[1]:
        names = [f"desc_{i}" for i in range(features.shape[1])]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix != ".npz":
        out = out.with_name(out.name + ".npz")  # mirror np.savez, so we return the real path
    np.savez(out, features)  # positional => stored under "arr_0", which chemprop reads
    out.with_name(out.name + ".columns.txt").write_text(
        "\n".join(names) + "\n", encoding="utf-8"
    )
    print(f"[write-descriptors] wrote {out} shape={features.shape}")

    if also_write_data is not None:
        data_out = Path(also_write_data)
        data_out.parent.mkdir(parents=True, exist_ok=True)
        kept.write_csv(data_out)
        print(f"[write-descriptors] wrote aligned data file {data_out} rows={kept.height}")

    return out, n_affected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a registered representation as a chemprop descriptor .npz."
    )
    parser.add_argument("--data-path", required=True, help="CSV with a SMILES column")
    parser.add_argument("--input", nargs="+", required=True,
                        help=f"Registered input name(s). Available: {list(INPUT_REGISTRY)}")
    parser.add_argument("--out", required=True, help="Descriptor .npz to write")
    parser.add_argument("--nan-policy", choices=["error", "fill", "drop"], default="error")
    parser.add_argument("--fill-value", type=float, default=0.0)
    parser.add_argument("--cache-dir", default="data/features")
    parser.add_argument("--also-write-data", default=None,
                        help="Write the row-aligned data CSV here (use with --nan-policy drop)")
    args = parser.parse_args()

    df = pl.read_csv(args.data_path)
    if "SMILES" not in df.columns:
        raise ValueError(f"{args.data_path} has no SMILES column; found {df.columns}")
    write_descriptors(
        df, args.input, args.out,
        nan_policy=args.nan_policy, fill_value=args.fill_value,
        cache_dir=args.cache_dir, also_write_data=args.also_write_data,
    )
