"""Download and reshape PubChem AID 1851 (Veith CYP qHTS panel) into model-ready heads.

Why this source, and why max_response rather than potency:

AID 1851 screens 17,143 substances in 15-point dose-response against five CYP
isoforms (1A2, 2C9, 2C19, 2D6, 3A4). Of the 85,715 compound x isoform rows,
only 40,684 have a fitted `Potency` -- but *all* of them have `Max_Response`.
The 42,460 rows the screen called Inactive have no potency at all, so a
potency-only ingest silently discards every negative the assay measured.
Those negatives are exactly what a challenge model trained on fitted pIC50
never sees: ChEMBL-style potency data is truncated from below (no curve, no
row), so the model learns what is potent but not what is inert.

Sign convention: `Max_Response` is a PERCENT change in luminescence and is
mostly negative, because inhibiting pro-luciferin conversion *decreases*
signal (median -26.6, 5th pct -105.5). We emit `-Max_Response` as
`..._pctinhib_aid1851` so that, like pIC50, larger means more inhibition.

These columns are auxiliary heads, never pooled into the scored pIC50
columns: the assay scale is percent inhibition at the top concentration, not
a fitted potency, so combining them would require a cross-assay correction.
As separate heads the shared encoder learns from them and each head keeps
its own scale.
"""

import csv
import sys
from pathlib import Path
from urllib.request import Request, urlopen

_PCGET_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi"
    "?query=download&record_type=datatable&actvty=all&response_type=save&aid=1851"
)

# Rows whose PUBCHEM_RESULT_TAG is one of these are the schema preamble, not data.
_PREAMBLE_TAGS = {"RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT", "RESULT_ATTR_CONC_MICROMOL"}

# PubChem panel name -> the isoform token used in our column names.
_PANEL_TO_ISOFORM = {
    "p450-cyp1a2": "CYP1A2",
    "p450-cyp2c9": "CYP2C9",
    "p450-cyp2c19": "CYP2C19",
    "p450-cyp2d6": "CYP2D6",
    "p450-cyp3a4": "CYP3A4",
}

_TIMEOUT_S = 600


def download_raw(dest: Path, force: bool = False) -> Path:
    """Download the full AID 1851 data table CSV (~24 MB) unless already present.

    Uses the bulk datatable endpoint: PUG REST refuses this assay outright
    ("Assay record retrieval is limited to 10000 SIDs") and the FTP archive
    only ships it inside a 1.3 GB multi-assay zip.
    """
    if dest.exists() and not force:
        print(f"Skipping download (already exists): {dest}", file=sys.stderr)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AID 1851 data table -> {dest}", file=sys.stderr)
    request = Request(_PCGET_URL, headers={"User-Agent": "openadmet-cyp/0.1"})
    with urlopen(request, timeout=_TIMEOUT_S) as response, dest.open("wb") as f:
        f.write(response.read())
    print(f"Saved {dest.stat().st_size / 1e6:.1f} MB", file=sys.stderr)
    return dest


def _parse_float(value: str) -> float | None:
    """Parse a PubChem numeric cell, treating blanks as missing."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _potency_to_pic50(potency_um: float | None) -> float | None:
    """Convert a PubChem `Potency` (micromolar AC50) to pIC50 = 6 - log10(uM)."""
    if potency_um is None or potency_um <= 0.0:
        return None
    import math

    return 6.0 - math.log10(potency_um)


def reshape(raw_path: Path, keep_inconclusive: bool = True) -> tuple[list[str], list[dict]]:
    """Pivot the long AID 1851 table into one row per molecule with per-isoform heads.

    Returns (fieldnames, rows). Each row carries the standardized SMILES, its
    InChIKey connectivity block, and for every isoform two auxiliary heads:
      {ISO}_pctinhib_aid1851 -- -Max_Response, percent inhibition at top conc
      {ISO}_pIC50_aid1851    -- 6 - log10(Potency uM), only where a curve was fitted
    """
    from data_tools.standardize import connectivity_key, standardize_smiles

    by_key: dict[str, dict] = {}
    n_rows = 0
    n_no_smiles = 0
    n_unparsable = 0
    n_skipped_outcome = 0

    with raw_path.open(newline="") as f:
        for record in csv.DictReader(f):
            tag = (record.get("PUBCHEM_RESULT_TAG") or "").strip()
            if tag in _PREAMBLE_TAGS or not tag.isdigit():
                continue
            n_rows += 1

            outcome = (record.get("PUBCHEM_ACTIVITY_OUTCOME") or "").strip()
            if outcome == "Inconclusive" and not keep_inconclusive:
                n_skipped_outcome += 1
                continue

            isoform = _PANEL_TO_ISOFORM.get((record.get("Panel Name") or "").strip().lower())
            if isoform is None:
                continue

            raw_smiles = (record.get("PUBCHEM_EXT_DATASOURCE_SMILES") or "").strip()
            if not raw_smiles:
                n_no_smiles += 1
                continue
            key = connectivity_key(raw_smiles)
            if key is None:
                n_unparsable += 1
                continue

            row = by_key.get(key)
            if row is None:
                row = {
                    "SMILES": standardize_smiles(raw_smiles),
                    "inchikey_block": key,
                    "PUBCHEM_CID": (record.get("PUBCHEM_CID") or "").strip(),
                }
                by_key[key] = row

            max_response = _parse_float(record.get("Max_Response", ""))
            if max_response is not None:
                row[f"{isoform}_pctinhib_aid1851"] = round(-max_response, 4)
            pic50 = _potency_to_pic50(_parse_float(record.get("Potency", "")))
            if pic50 is not None:
                row[f"{isoform}_pIC50_aid1851"] = round(pic50, 4)

    isoform_cols: list[str] = []
    for isoform in _PANEL_TO_ISOFORM.values():
        isoform_cols += [f"{isoform}_pctinhib_aid1851", f"{isoform}_pIC50_aid1851"]
    fieldnames = ["SMILES", "inchikey_block", "PUBCHEM_CID"] + isoform_cols

    rows = [{name: row.get(name, "") for name in fieldnames} for row in by_key.values()]

    print(
        f"Parsed {n_rows} assay rows -> {len(rows)} unique molecules "
        f"(skipped: {n_no_smiles} without SMILES, {n_unparsable} unparsable, "
        f"{n_skipped_outcome} inconclusive)",
        file=sys.stderr,
    )
    filled = {name: sum(1 for r in rows if r[name] != "") for name in isoform_cols}
    for name, count in filled.items():
        print(f"  {name}: {count} labels", file=sys.stderr)
    print(f"  total labels: {sum(filled.values())}", file=sys.stderr)
    return fieldnames, rows


def write_csv(fieldnames: list[str], rows: list[dict], dest: Path) -> None:
    """Write the reshaped table to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows x {len(fieldnames)} cols -> {dest}", file=sys.stderr)
