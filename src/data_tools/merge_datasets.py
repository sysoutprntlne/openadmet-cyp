"""Merge selected datasets into one wide multi-task table at training time.

`build-union` writes a fixed table to disk. This does the same join on the fly
for whichever sources a run asks for, so experiments can mix sources without
rebuilding anything. Every source x isoform x readout becomes its own column
(its own head): `CYP2D6-BindingDB`, `CYP3A4-TDI`, `CYP1A2-PubChem`, ... No
measurements are pooled across sources.

Two kinds of source, joined differently:

  in-house   challenge, tdi, emax, single_conc. Same OCNT molecules, keyed on
             Molecule_Name, then collapsed on the full InChIKey so
             stereoisomers stay distinct (as in build_union).
  external   pubchem, chembl, bindingdb. Joined on the InChIKey connectivity
             block so salts and stereo variants match.

The train/val/test split is computed on the in-house molecules alone, so it is
identical whichever external sources are selected and runs stay comparable.
Molecules with no scored label (including every external-only molecule) are
always train. Molecules whose skeleton appears in the blind test set are
dropped, as in build_union.
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl

from data_tools.overlap import add_keys, add_stereo_keys, collapse_duplicates, overlap_report
from data_tools.split_column import assign_splits

_KEY = "inchikey_block"
_STEREO_KEY = "inchikey_full"
_NAME = "Molecule_Name"
_SCORED_SOURCE = "challenge"


def _pivot_single_concentration(df: pl.DataFrame) -> pl.DataFrame:
    """Long (molecule, enzyme) rows to one `{enzyme}_log2fc` column per enzyme."""
    smiles = df.group_by(_NAME, maintain_order=True).agg(pl.col("SMILES").first())
    wide = df.pivot(on="enzyme", index=_NAME, values="log2fc_estimate", aggregate_function="mean")
    wide = wide.rename({c: f"{c}_log2fc" for c in wide.columns if c != _NAME})
    return smiles.join(wide, on=_NAME, how="left")


@dataclass(frozen=True)
class Source:
    """One selectable dataset: a file, how it joins, and how its columns become heads."""

    filename: str
    kind: str  # "inhouse" or "external"
    # (regex with one group capturing the isoform/gene, head-name template)
    heads: list[tuple[str, str]]
    prepare: Callable[[pl.DataFrame], pl.DataFrame] | None = None
    description: str = ""
    task: str = "regression"  # "classification" for 0/1 labels
    log10p1: bool = False  # label spans orders of magnitude: train on log10(x + 1)


SOURCES: dict[str, Source] = {
    "challenge": Source(
        "train.csv",
        "inhouse",
        [(r"^(.+)_pIC50_direct_inhibition$", "{}-Direct")],
        description="scored heads: direct-inhibition pIC50 (always included)",
    ),
    "tdi": Source(
        "tdi_train.csv",
        "inhouse",
        [(r"^(.+)_pIC50_TDI_condition$", "{}-TDI")],
        description="pIC50 under time-dependent-inhibition conditions",
    ),
    "emax": Source(
        "emax_train.csv",
        "inhouse",
        [
            (r"^(.+)_EmaxVsPosCtrl_TDI_condition$", "{}-Emax-TDI"),
            (r"^(.+)_EmaxVsPosCtrl_direct_inhibition$", "{}-Emax-Direct"),
        ],
        description="Emax vs positive control, TDI and direct conditions",
    ),
    "single_conc": Source(
        "single_concentration_train.csv",
        "inhouse",
        [(r"^(.+)_log2fc$", "{}-SingleConc")],
        prepare=_pivot_single_concentration,
        description="log2 fold change at one concentration (50 uM)",
    ),
    "pubchem": Source(
        "pubchem_aid1851.csv",
        "external",
        [
            (r"^(.+)_pIC50_aid1851$", "{}-PubChem"),
            (r"^(.+)_pctinhib_aid1851$", "{}-PubChem-pctinhib"),
        ],
        description="PubChem AID 1851 qHTS panel: pIC50 and percent inhibition",
    ),
    "expansionrx": Source(
        "metabolism/expansionrx.csv",
        "external",
        [(r"^(HLM|MLM) CLint$", "{}-ExpansionRx")],
        log10p1=True,
        description="ExpansionRx microsomal CLint (human, mouse), log10(x+1)",
    ),
    "tdc_hlm": Source(
        "metabolism/tdc_hlm.csv",
        "external",
        [(r"^(Y)$", "HLM-TDC")],
        task="classification",
        description="TDC human liver microsomal stability, binary",
    ),
    "tdc_rlm": Source(
        "metabolism/tdc_rlm.csv",
        "external",
        [(r"^(Y)$", "RLM-TDC")],
        task="classification",
        description="TDC rat liver microsomal stability, binary",
    ),
    "tdc_clearance_microsome_az": Source(
        "metabolism/tdc_clearance_microsome_az.csv",
        "external",
        [(r"^(Y)$", "MicrosomeClearance-AZ")],
        log10p1=True,
        description="TDC AstraZeneca microsomal clearance, log10(x+1)",
    ),
    "tdc_half_life_obach": Source(
        "metabolism/tdc_half_life_obach.csv",
        "external",
        [(r"^(Y)$", "HalfLife-Obach")],
        log10p1=True,
        description="TDC Obach half-life, log10(x+1)",
    ),
    "tdc_cyp2c9_substrate": Source(
        "metabolism/tdc_cyp2c9_substrate.csv",
        "external",
        [(r"^(Y)$", "CYP2C9-Substrate")],
        task="classification",
        description="TDC CYP2C9 substrate, binary",
    ),
    "tdc_cyp2d6_substrate": Source(
        "metabolism/tdc_cyp2d6_substrate.csv",
        "external",
        [(r"^(Y)$", "CYP2D6-Substrate")],
        task="classification",
        description="TDC CYP2D6 substrate, binary",
    ),
    "tdc_cyp3a4_substrate": Source(
        "metabolism/tdc_cyp3a4_substrate.csv",
        "external",
        [(r"^(Y)$", "CYP3A4-Substrate")],
        task="classification",
        description="TDC CYP3A4 substrate, binary",
    ),
    "chembl": Source(
        "family_heads.csv",
        "external",
        [(r"^(.+)_pIC50_chembl$", "{}-ChEMBL")],
        description="ChEMBL pIC50 across the P450 family",
    ),
    "bindingdb": Source(
        "family_heads.csv",
        "external",
        [(r"^(.+)_pIC50_bindingdb$", "{}-BindingDB")],
        description="BindingDB pIC50 across the P450 family",
    ),
}


def parse_dataset_specs(specs: list[str], with_challenge: bool = True) -> dict[str, list[str] | None]:
    """Parse `NAME` or `NAME:ISO,ISO` into {name: isoform filter or None}.

    `challenge` is present and first unless with_challenge is False: it
    carries the scored heads and the split.
    """
    parsed: dict[str, list[str] | None] = {_SCORED_SOURCE: None} if with_challenge else {}
    for spec in specs:
        name, _, isos = spec.partition(":")
        if name not in SOURCES:
            raise ValueError(f"unknown dataset {name!r}; choose from {sorted(SOURCES)}")
        wanted = [i for i in isos.split(",") if i]
        parsed[name] = wanted or None
    return parsed


def _select_heads(df: pl.DataFrame, source: Source, isoforms: list[str] | None) -> dict[str, str]:
    """Map existing column -> head name for every column the source defines."""
    mapping: dict[str, str] = {}
    for column in df.columns:
        for pattern, template in source.heads:
            match = re.match(pattern, column)
            if match and (isoforms is None or match.group(1) in isoforms):
                mapping[column] = template.format(match.group(1))
    return mapping


def _load(source: Source, data_dir: Path, isoforms: list[str] | None) -> tuple[pl.DataFrame, list[str]]:
    """Read a source and return (frame with SMILES [+ Molecule_Name] + renamed heads, head names)."""
    path = data_dir / source.filename
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (see the Data section of the README to build it)")
    # infer_schema_length=None: sparse label columns would otherwise be typed String.
    df = pl.read_csv(path, infer_schema_length=None)
    if source.prepare is not None:
        df = source.prepare(df)
    mapping = _select_heads(df, source, isoforms)
    if not mapping:
        raise ValueError(f"no heads from {source.filename} match isoforms {isoforms}")
    keep = [c for c in (_NAME, "SMILES") if c in df.columns]
    out = df.select(keep + list(mapping)).rename(mapping)
    if source.log10p1:
        out = out.with_columns([(pl.col(n).cast(pl.Float64) + 1).log10().alias(n) for n in mapping.values()])
    # Files shared by several sources (family_heads.csv) carry molecules that
    # have no label for the selected heads; they would only add empty rows.
    names = list(mapping.values())
    out = out.filter(pl.any_horizontal([pl.col(n).is_not_null() for n in names]))
    return out, names


def merge(
    datasets: dict[str, list[str] | None],
    data_dir: Path = Path("data"),
    test_path: Path | None = Path("data/test.csv"),
    split_strategy: str = "scaffold",
    split_seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[pl.DataFrame, list[str], list[str], dict[str, str]]:
    """Return (table with SMILES + heads + split, head names, scored head names, task per head).

    The task is "regression" or "classification" (0/1 labels).
    """
    heads: list[str] = []
    scored: list[str] = []
    tasks: dict[str, str] = {}
    inhouse: pl.DataFrame | None = None
    smiles_by_name: list[pl.DataFrame] = []

    for name, isoforms in datasets.items():
        source = SOURCES[name]
        if source.kind != "inhouse":
            continue
        frame, names = _load(source, data_dir, isoforms)
        print(f"  {name}: {len(frame)} rows, heads {names}", file=sys.stderr)
        smiles_by_name.append(frame.select([_NAME, "SMILES"]))
        frame = frame.drop("SMILES")
        inhouse = frame if inhouse is None else inhouse.join(frame, on=_NAME, how="full", coalesce=True)
        heads += names
        tasks.update({n: source.task for n in names})
        if name == _SCORED_SOURCE:
            scored = names

    smiles = pl.concat(smiles_by_name).unique(subset=[_NAME], keep="first", maintain_order=True)
    table = add_stereo_keys(add_keys(inhouse.join(smiles, on=_NAME, how="left")))
    # Stereoisomers are different molecules: collapse on the full InChIKey, not the skeleton.
    table = collapse_duplicates(table, heads, key=_STEREO_KEY).sort(_STEREO_KEY)

    splits = assign_splits(
        table,
        scored,
        strategy=split_strategy,  # type: ignore[arg-type]
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    table = table.with_columns(splits)

    for name, isoforms in datasets.items():
        source = SOURCES[name]
        if source.kind != "external":
            continue
        frame, names = _load(source, data_dir, isoforms)
        frame = collapse_duplicates(add_keys(frame), names)
        if source.task == "classification":
            # Averaging duplicate 0/1 labels can give 0.5; settle ties towards positive.
            frame = frame.with_columns([(pl.col(n) >= 0.5).cast(pl.Float64).alias(n) for n in names])
        overlap_report(table, frame, "in-house", name)
        print(f"  {name}: {len(frame)} unique skeletons, heads {names}", file=sys.stderr)
        table = table.join(frame.select([_KEY] + names), on=_KEY, how="full", coalesce=True)
        backfill = frame.select([_KEY, pl.col("SMILES").alias("_ext_smiles")])
        table = (
            table.join(backfill, on=_KEY, how="left", coalesce=True)
            .with_columns(pl.coalesce(["SMILES", "_ext_smiles"]).alias("SMILES"))
            .drop("_ext_smiles")
        )
        heads += names
        tasks.update({n: source.task for n in names})

    # Rows that exist only in external sources have no split: always train.
    table = table.with_columns(pl.col("split").fill_null("train"))

    if test_path is not None and test_path.exists():
        test_keys = set(add_keys(pl.read_csv(test_path, infer_schema_length=None))[_KEY].to_list())
        leaked = table.filter(pl.col(_KEY).is_in(test_keys))
        if len(leaked):
            print(f"  dropping {len(leaked)} row(s) whose skeleton is in the blind test set", file=sys.stderr)
            table = table.filter(~pl.col(_KEY).is_in(test_keys))

    table = table.filter(pl.col("SMILES").is_not_null())
    return table.select(["SMILES"] + heads + ["split"]), heads, scored, tasks


def describe() -> str:
    """One line per selectable dataset, for --list-datasets."""
    lines = [f"{name:26s} {src.kind:9s} {src.task:14s} {src.description}" for name, src in SOURCES.items()]
    lines.append("")
    lines.append("Restrict a dataset to isoforms with NAME:ISO,ISO  e.g. bindingdb:CYP2D6,CYP1A2")
    lines.append("Heads are named {isoform}-{source}, e.g. CYP2D6-BindingDB, CYP3A4-TDI.")
    return "\n".join(lines)
