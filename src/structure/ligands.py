"""Ligand registry for the smoke test.

Known Type II CYP3A4 inhibitors that coordinate the heme iron through an
azole nitrogen. SMILES are fetched from PubChem PUG REST (isomeric, so the
stereocentres survive) and cached under data/structures/cache/.

The join key downstream is the InChIKey connectivity block
(data_tools.standardize.connectivity_key), so we record it here too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from structure.receptors import cache_dir

RDLogger.DisableLog("rdApp.*")

_PUG_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES,InChIKey/JSON"

# cid is kept for provenance; name is the PUG lookup key.
LIGANDS: dict[str, dict] = {
    "ketoconazole": {"cid": 3823},
    "itraconazole": {"cid": 3793},
}


@dataclass(frozen=True)
class Ligand:
    """A ligand ready for co-folding input."""

    name: str
    smiles: str
    inchikey: str
    connectivity_block: str  # first block of the InChIKey = T5/T6 join key


def fetch_ligand(name: str, root: Path | None = None) -> Ligand:
    """Return the cached-or-fetched isomeric SMILES + InChIKey for a ligand."""
    cache = cache_dir(root) / f"ligand_{name}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        resp = requests.get(_PUG_URL.format(name=name), timeout=60)
        resp.raise_for_status()
        props = resp.json()["PropertyTable"]["Properties"][0]
        # PubChem's name lookup returns the property under `SMILES` even when
        # IsomericSMILES was requested (both keys observed in the wild).
        smiles = props.get("IsomericSMILES") or props.get("SMILES")
        payload = {"smiles": smiles, "inchikey": props["InChIKey"]}
        cache.write_text(json.dumps(payload, indent=2))

    mol = Chem.MolFromSmiles(payload["smiles"])
    if mol is None:
        raise ValueError(f"{name}: PubChem SMILES does not parse: {payload['smiles']}")
    return Ligand(
        name=name,
        smiles=payload["smiles"],
        inchikey=payload["inchikey"],
        connectivity_block=payload["inchikey"].split("-")[0],
    )


def n_stereocenters(lig: Ligand) -> int:
    """Count defined stereocentres — sanity that stereochemistry survived."""
    mol = Chem.MolFromSmiles(lig.smiles)
    return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))


def mol_weight(lig: Ligand) -> float:
    mol = Chem.MolFromSmiles(lig.smiles)
    return Descriptors.MolWt(mol)
