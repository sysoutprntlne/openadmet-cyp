"""Chemical similarity of candidate pretraining datasets to the challenge train and test sets.

Answers "which external dataset looks most like the molecules we are scored on",
so a pretraining set can be chosen on chemistry instead of on row count.

For every dataset D and each reference set R in {train, test}:

  D -> R      nearest-neighbour Tanimoto from each D molecule to R. High means D's
              molecules are the kind R contains.
  R -> D      nearest-neighbour Tanimoto from each R molecule to D. High means D
              *covers* R's chemistry: what pretraining on D could transfer from.
  shared      molecules of D whose InChIKey connectivity block is in R. Shared with
              test means they must be dropped before training (see build-union).

Rows for `train` and `test` give reference points: each compared against itself
excluding the molecule itself (leave-one-out), and against the other.

Morgan radius 2, 2048 bits, Tanimoto. Molecules are reduced to their largest
fragment and deduplicated on the connectivity block, as everywhere else.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from data_tools.standardize import _largest_fragment, connectivity_key

_THRESHOLDS = (0.4, 0.7)
_CHUNK = 2048


def load_molecules(path: Path, smiles_column: str = "SMILES") -> tuple[np.ndarray, list[str]]:
    """Return (uint8 fingerprint matrix, connectivity keys), one row per unique skeleton."""
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    df = pl.read_csv(path, infer_schema_length=None)
    if smiles_column not in df.columns:
        raise ValueError(f"{path} has no {smiles_column!r} column (columns: {df.columns})")

    seen: set[str] = set()
    fps, keys = [], []
    for smiles in df[smiles_column].drop_nulls().to_list():
        key = connectivity_key(smiles)
        if key is None or key in seen:
            continue
        mol = Chem.MolFromSmiles(_largest_fragment(smiles.strip()))
        if mol is None:
            continue
        seen.add(key)
        keys.append(key)
        fps.append(gen.GetFingerprintAsNumPy(mol))
    return np.vstack(fps).astype(np.uint8), keys


def nearest_neighbour(queries: np.ndarray, reference: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """Max Tanimoto from each query row to any reference row.

    With exclude_self the two matrices are the same molecules in the same
    order and each row's match with itself is ignored (leave-one-out).
    """
    ref = reference.astype(np.float32)
    ref_bits = ref.sum(axis=1)
    best = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), _CHUNK):
        q = queries[start : start + _CHUNK].astype(np.float32)
        inter = q @ ref.T
        union = q.sum(axis=1)[:, None] + ref_bits[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        if exclude_self:
            rows = np.arange(len(q))
            sim[rows, start + rows] = -1.0
        best[start : start + len(q)] = sim.max(axis=1)
    return best


def _stats(nn: np.ndarray) -> dict[str, float]:
    out = {"mean": float(nn.mean())}
    for t in _THRESHOLDS:
        out[f">={t}"] = float((nn >= t).mean())
    return out


def compare(
    name: str,
    fps: np.ndarray,
    keys: list[str],
    train: tuple[np.ndarray, list[str]],
    test: tuple[np.ndarray, list[str]],
) -> dict[str, float | int | str]:
    """One table row: how dataset `name` relates to train and test, in both directions."""
    row: dict[str, float | int | str] = {"dataset": name, "n": len(keys)}
    for ref_name, (ref_fps, ref_keys) in (("train", train), ("test", test)):
        same = name == ref_name
        forward = nearest_neighbour(fps, ref_fps, exclude_self=same)
        s = _stats(forward)
        row[f"->{ref_name} NN"] = s["mean"]
        row[f"->{ref_name} >=0.4"] = s[">=0.4"]
        row[f"->{ref_name} >=0.7"] = s[">=0.7"]
        row[f"shared {ref_name}"] = 0 if same else len(set(keys) & set(ref_keys))
        if not same:
            backward = nearest_neighbour(ref_fps, fps)
            row[f"{ref_name}-> NN"] = float(backward.mean())
            row[f"{ref_name}->  >=0.4"] = float((backward >= 0.4).mean())
        else:
            row[f"{ref_name}-> NN"] = float("nan")
            row[f"{ref_name}->  >=0.4"] = float("nan")
    return row


def to_markdown(table: pl.DataFrame) -> str:
    """Render the result table as GitHub markdown."""
    header = table.columns
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in table.iter_rows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("-" if np.isnan(value) else f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Entry point for the dataset-similarity CLI."""
    parser = argparse.ArgumentParser(
        description="Compare candidate datasets to the challenge train and test molecules "
        "(Morgan r2 Tanimoto, nearest neighbour) and print one table.",
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--metabolism-dir",
        type=Path,
        default=Path("data/metabolism"),
        help="Every *.csv in here is compared (written by download-data)",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="More datasets to compare, e.g. pubchem_aid1851=data/pubchem_aid1851.csv as a reference point",
    )
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument("--output-path", type=Path, default=Path("results/dataset_similarity.csv"))
    args = parser.parse_args()

    datasets: dict[str, Path] = {p.stem: p for p in sorted(args.metabolism_dir.glob("*.csv"))}
    for spec in args.extra:
        name, sep, path = spec.partition("=")
        if not sep:
            parser.error(f"--extra entries must be NAME=PATH, got {spec!r}")
        datasets[name] = Path(path)
    if not datasets:
        parser.error(f"no datasets found in {args.metabolism_dir}; run download-data first")

    print(f"Fingerprinting train and test ...", file=sys.stderr)
    train = load_molecules(args.train, args.smiles_column)
    test = load_molecules(args.test, args.smiles_column)
    print(f"  train {len(train[1])} skeletons, test {len(test[1])} skeletons", file=sys.stderr)

    rows = [compare("train", *train, train, test), compare("test", *test, train, test)]
    for name, path in datasets.items():
        print(f"{name} <- {path}", file=sys.stderr)
        fps, keys = load_molecules(path, args.smiles_column)
        rows.append(compare(name, fps, keys, train, test))

    table = pl.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.output_path)
    print(f"\nWrote {args.output_path}", file=sys.stderr)
    print(to_markdown(table))


if __name__ == "__main__":
    main()
