"""Build per-protein affinity datasets for the cytochrome P450 superfamily.

Pulls every reviewed UniProt entry in the P450 Pfam family (PF00067), fetches
known-ligand affinity data for each from ChEMBL and BindingDB, ranks proteins
by how much data is available, and writes a train_chemprop-ready CSV
(SMILES, pAffinity) for the top N — candidates for pretraining Chemprop
before fine-tuning/transferring to the CYP isoforms in this challenge.
"""

import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
_BINDINGDB_URL = "http://bindingdb.org/rest/getLigandsByUniprots"

_DEFAULT_PFAM_ID = "PF00067"
_DEFAULT_TOP_N = 200
_DEFAULT_MIN_RECORDS = 5
_DEFAULT_BINDINGDB_CUTOFF_NM = 10000.0
_DEFAULT_MAX_WORKERS = 8
_CHEMBL_ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]
_TIMEOUT_S = 30
_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+")


@dataclass
class Record:
    """One measured affinity: SMILES + pAffinity (-log10 molar), with provenance."""

    smiles: str
    p_affinity: float
    source: str
    affinity_type: str


@dataclass
class ProteinTarget:
    """A P450-family UniProt entry and the affinity records found for it."""

    accession: str
    gene: str
    organism: str
    protein_name: str
    chembl_target_id: str | None
    records: list[Record] = field(default_factory=list)


def _http_get_json(url: str, params: dict) -> dict:
    """GET url?params and parse the JSON response."""
    full_url = f"{url}?{urlencode(params)}"
    request = Request(full_url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=_TIMEOUT_S) as response:
        return json.loads(response.read())


def fetch_pfam_uniprot_entries(pfam_id: str = _DEFAULT_PFAM_ID) -> list[ProteinTarget]:
    """Return every reviewed UniProt entry in the given Pfam family, paginating via cursor."""
    fields = "accession,gene_primary,organism_name,protein_name,xref_chembl"
    query = f"xref:pfam-{pfam_id} AND reviewed:true"
    url = f"{_UNIPROT_SEARCH_URL}?{urlencode({'query': query, 'fields': fields, 'format': 'json', 'size': 500})}"

    targets: list[ProteinTarget] = []
    while url:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=_TIMEOUT_S) as response:
            payload = json.loads(response.read())
            link_header = response.headers.get("Link")

        for entry in payload.get("results", []):
            accession = entry["primaryAccession"]
            organism = entry.get("organism", {}).get("scientificName", "")
            protein_name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
            genes = entry.get("genes", [])
            gene = genes[0].get("geneName", {}).get("value", accession) if genes else accession
            chembl_ids = [x["id"] for x in entry.get("uniProtKBCrossReferences", []) if x.get("database") == "ChEMBL"]
            targets.append(
                ProteinTarget(
                    accession=accession,
                    gene=gene,
                    organism=organism,
                    protein_name=protein_name,
                    chembl_target_id=chembl_ids[0] if chembl_ids else None,
                )
            )

        url = None
        if link_header:
            match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link_header)
            if match:
                url = match.group(1)

    return targets


def fetch_chembl_records(target_chembl_id: str) -> list[Record]:
    """Return all IC50/Ki/Kd/EC50 records with a pChEMBL value for one ChEMBL target."""
    records: list[Record] = []
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type__in": ",".join(_CHEMBL_ACTIVITY_TYPES),
        "pchembl_value__isnull": "false",
        "limit": 1000,
        "offset": 0,
        "format": "json",
    }
    while True:
        payload = _http_get_json(_CHEMBL_ACTIVITY_URL, params)
        for activity in payload.get("activities", []):
            smiles = activity.get("canonical_smiles")
            pchembl = activity.get("pchembl_value")
            if smiles and pchembl is not None:
                records.append(Record(smiles, float(pchembl), "chembl", activity.get("standard_type", "")))
        next_page = payload.get("page_meta", {}).get("next")
        if not next_page:
            break
        params["offset"] += params["limit"]
    return records


def fetch_bindingdb_records(accession: str, cutoff_nm: float) -> list[Record]:
    """Return IC50/Ki/Kd/EC50 records (converted to pAffinity) for one UniProt accession."""
    payload = _http_get_json(_BINDINGDB_URL, {"uniprot": accession, "cutoff": cutoff_nm, "response": "application/json"})
    affinities = payload.get("getLindsByUniprotsResponse", {}).get("affinities", [])

    records: list[Record] = []
    for entry in affinities:
        smiles = entry.get("smile")
        raw_value = entry.get("affinity")
        match = _NUMERIC_RE.search(raw_value or "")
        if not smiles or not match:
            continue
        value_nm = float(match.group())
        if value_nm <= 0:
            continue
        p_affinity = 9.0 - math.log10(value_nm)
        records.append(Record(smiles, p_affinity, "bindingdb", entry.get("affinity_type", "")))
    return records


def _fetch_one_target(target: ProteinTarget, bindingdb_cutoff_nm: float) -> ProteinTarget:
    """Populate target.records from ChEMBL (if a target xref exists) and BindingDB."""
    records: list[Record] = []
    if target.chembl_target_id:
        try:
            records += fetch_chembl_records(target.chembl_target_id)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            pass
    try:
        records += fetch_bindingdb_records(target.accession, bindingdb_cutoff_nm)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        pass
    target.records = records
    return target


def collect_family_targets(
    pfam_id: str,
    bindingdb_cutoff_nm: float,
    max_workers: int,
) -> list[ProteinTarget]:
    """Fetch the P450-family UniProt list, then pull affinity data for every entry in parallel."""
    targets = fetch_pfam_uniprot_entries(pfam_id)
    print(f"Found {len(targets)} reviewed UniProt entries in {pfam_id}", file=sys.stderr)

    completed: list[ProteinTarget] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one_target, t, bindingdb_cutoff_nm): t for t in targets}
        for i, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if i % 100 == 0 or i == len(targets):
                print(f"  fetched {i}/{len(targets)} targets", file=sys.stderr)
    return completed


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "unknown"


def write_datasets(
    targets: list[ProteinTarget],
    output_dir: Path,
    top_n: int,
    min_records: int,
) -> list[ProteinTarget]:
    """Rank targets by record count, write a CSV per top-N target plus a manifest. Returns the selected targets."""
    ranked = sorted((t for t in targets if len(t.records) >= min_records), key=lambda t: len(t.records), reverse=True)
    selected = ranked[:top_n]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = ["accession,gene,organism,protein_name,chembl_target_id,n_chembl,n_bindingdb,n_total,output_file"]

    for target in selected:
        n_chembl = sum(1 for r in target.records if r.source == "chembl")
        n_bindingdb = sum(1 for r in target.records if r.source == "bindingdb")
        filename = f"{_sanitize(target.gene)}_{target.accession}.csv"
        out_path = output_dir / filename

        with out_path.open("w") as f:
            f.write("SMILES,pAffinity,source,affinity_type\n")
            for r in target.records:
                f.write(f'"{r.smiles}",{r.p_affinity:.4f},{r.source},{r.affinity_type}\n')

        manifest_rows.append(
            f'{target.accession},{target.gene},"{target.organism}","{target.protein_name}",'
            f"{target.chembl_target_id or ''},{n_chembl},{n_bindingdb},{len(target.records)},{filename}"
        )

    (output_dir / "manifest.csv").write_text("\n".join(manifest_rows) + "\n")
    return selected
