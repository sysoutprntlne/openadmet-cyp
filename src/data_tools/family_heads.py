"""Reshape per-protein P450 family datasets into per-protein auxiliary heads.

`data fetch p450_family` writes one CSV per protein with a single pooled
`pAffinity` column (see data_tools.family_datasets). That shape cannot enter
the union table: every file would contribute a head named `pAffinity`, so two
files collide, and the pooling hides IC50/Ki/Kd/EC50 from two databases inside
one column where no task weighting can separate them.

This module pivots the manifest into one column per protein per source, named
`{gene}_pIC50_{source}`, so each lands in the union as its own head and
`head-search` discovers it automatically.

Note what the overlapping genes mean. A ChEMBL CYP3A4 IC50 head is not a
different isoform -- it is the *same endpoint* as the scored CYP3A4 column,
measured in other labs on other chemistry. Kept as a separate head it needs no
cross-assay correction, and it is a strong candidate for the most useful head
in the search, not a distractor.

By default only IC50 records are kept: it is the endpoint the challenge scores,
and mixing Ki/Kd/EC50 into one column reintroduces the pooling problem. Pass
--affinity-types to widen, or --split-by-affinity-type to give each its own head.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

_KEY = "inchikey_block"


def select_proteins(
    manifest_path: Path,
    organism: str | None,
    min_records: int,
    genes: list[str] | None,
    top_n: int | None,
) -> list[dict]:
    """Return manifest rows passing the filters, richest first.

    Filtering matters for the search, not just for runtime: most PF00067
    entries are bacterial or plant P450s whose ligand data says little about
    human CYP inhibition, and every extra candidate both costs runs and makes
    it likelier that the best-looking subset is noise.
    """
    with manifest_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    kept: list[dict] = []
    for row in rows:
        if organism and row.get("organism", "") != organism:
            continue
        if genes and row.get("gene", "") not in genes:
            continue
        try:
            n_total = int(row.get("n_total") or 0)
        except ValueError:
            n_total = 0
        if n_total < min_records:
            continue
        row["_n_total"] = n_total
        kept.append(row)

    kept.sort(key=lambda r: r["_n_total"], reverse=True)
    if top_n is not None:
        kept = kept[:top_n]

    print(f"{len(rows)} manifest rows -> {len(kept)} selected", file=sys.stderr)
    for row in kept:
        print(f"  {row['gene']:12} {row['accession']:8} {row['_n_total']:>7} records  {row['output_file']}", file=sys.stderr)
    return kept


def _head_name(gene: str, accession: str, source: str, affinity_type: str | None, gene_counts: dict[str, int]) -> str:
    """Build a column name, disambiguating genes that appear under two accessions."""
    base = gene if gene_counts.get(gene, 0) <= 1 else f"{gene}_{accession}"
    readout = affinity_type if affinity_type else "pIC50"
    return f"{base}_{readout}_{source}"


def build(
    manifest_path: Path,
    selected: list[dict],
    affinity_types: list[str],
    split_by_affinity_type: bool,
) -> tuple[pl.DataFrame, list[str]]:
    """Read every selected protein CSV and return (wide table, head columns)."""
    from data_tools.standardize import connectivity_key, standardize_smiles

    gene_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        gene_counts[row["gene"]] += 1

    wanted = {t.upper() for t in affinity_types}
    # key -> {"SMILES": str, head -> [values]}
    records: dict[str, dict] = {}
    head_names: list[str] = []

    for row in selected:
        path = manifest_path.parent / row["output_file"]
        if not path.exists():
            print(f"  warning: missing {path}, skipping", file=sys.stderr)
            continue

        n_rows = n_kept = 0
        with path.open(newline="") as f:
            for record in csv.DictReader(f):
                n_rows += 1
                affinity_type = (record.get("affinity_type") or "").strip().upper()
                if wanted and affinity_type not in wanted:
                    continue
                value_text = (record.get("pAffinity") or "").strip()
                smiles = (record.get("SMILES") or "").strip()
                source = (record.get("source") or "unknown").strip()
                if not smiles or not value_text:
                    continue
                try:
                    value = float(value_text)
                except ValueError:
                    continue

                key = connectivity_key(smiles)
                if key is None:
                    continue

                head = _head_name(
                    row["gene"],
                    row["accession"],
                    source,
                    affinity_type if split_by_affinity_type else None,
                    gene_counts,
                )
                if head not in head_names:
                    head_names.append(head)

                entry = records.setdefault(key, {"SMILES": standardize_smiles(smiles), _KEY: key})
                entry.setdefault(head, []).append(value)
                n_kept += 1

        print(f"  {row['gene']} ({row['accession']}): {n_rows} rows -> {n_kept} kept", file=sys.stderr)

    # Average repeated measurements of the same molecule for the same head.
    rows_out: list[dict] = []
    for entry in records.values():
        out = {"SMILES": entry["SMILES"], _KEY: entry[_KEY]}
        for head in head_names:
            values = entry.get(head)
            out[head] = sum(values) / len(values) if values else None
        rows_out.append(out)

    df = pl.DataFrame(rows_out, schema={"SMILES": pl.String, _KEY: pl.String, **{h: pl.Float64 for h in head_names}})
    print(f"\n{len(df)} unique molecules, {len(head_names)} heads", file=sys.stderr)
    for head in head_names:
        print(f"  {head}: {len(df) - df[head].null_count()} labels", file=sys.stderr)
    return df, head_names


def write_heads(
    manifest_path: Path,
    output_path: Path,
    organism: str | None = "Homo sapiens",
    min_records: int = 100,
    genes: list[str] | None = None,
    top_n: int = 20,
    affinity_types: tuple[str, ...] | list[str] = ("IC50",),
    split_by_affinity_type: bool = False,
) -> list[str]:
    """Select proteins from the manifest, pivot them into head columns, write the CSV. Returns the head names.

    Raises ValueError if no protein passes the filters or no usable record is found.
    """
    selected = select_proteins(manifest_path, organism, min_records, genes, top_n)
    if not selected:
        raise ValueError("No proteins passed the filters.")
    df, head_names = build(manifest_path, selected, list(affinity_types), split_by_affinity_type)
    if not head_names:
        raise ValueError("No usable records found.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    return head_names
