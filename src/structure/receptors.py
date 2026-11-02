"""Isoform registry and receptor prep (T5 scope step 1).

For each scored CYP isoform we need three things to build a co-fold input:
  1. The canonical protein sequence (UniProt).
  2. The proximal heme-thiolate cysteine (the Fe axial ligand, type II
     anchoring). Documented residue numbers, verified at runtime against the
     fetched sequence and the conserved FGxGxRxCxG motif.
  3. A holo crystal structure with resolved heme, for reference geometry
     (heme-plane normal, Fe position) and later bake-off validation.

PDB choices per tasks/T5-structure-generation.md:
    CYP1A2 -> 2HI4, CYP2C9 -> 1OG5, CYP2D6 -> 4WNV, CYP3A4 -> 4NY4.

Receptor-prep choices (documented so downstream isn't silently invalidated):
  - Sequence is the UniProt canonical isoform, full length including signal
    peptide; co-folders were trained on full constructs.
  - We do NOT strip anything for co-folding: heme enters as CCD HEM, waters
    and crystallographic ligands are simply never included.
  - The Cys-Fe coordination is not expressed as a covalent bond for
    IntelliFold/AF3-style inputs (schema has no cross-entity bonds); HEM's CCD
    carries the Fe and the models saw CYP-HEM complexes in training. Protenix's
    explicit covalent_bonds block is the production follow-up (see T5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import requests

_UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
_RCSB_URL = "https://files.rcsbound.org"  # placeholder replaced below
_RCSB_CIF = "https://files.rcsb.org/download/{pdb}.cif"

# PROSITE PS00098 (P450_CYTOCHROME_CHOL), relaxed: [FW]-x-x-x-x-x-x-C.
# Scanned only in the C-terminal half, where the heme-binding domain lives;
# this avoids the upstream Cys clusters (e.g. CYP1A2's KKCC at 406).
_HEME_MOTIF = re.compile(r"[FW].{6}C")


@dataclass(frozen=True)
class Isoform:
    """One CYP isoform: identity, crystal reference, documented heme Cys."""

    gene: str
    uniprot: str
    pdb_holo: str
    heme_cys_1based: int  # UniProt numbering, proximal cysteine ligand to Fe
    chain: str = "A"


ISOFORMS: dict[str, Isoform] = {
    "CYP1A2": Isoform("CYP1A2", "P05177", "2HI4", 458),
    "CYP2C9": Isoform("CYP2C9", "P11712", "1OG5", 435),
    "CYP2D6": Isoform("CYP2D6", "P10635", "4WNV", 443),
    "CYP3A4": Isoform("CYP3A4", "P08684", "4NY4", 442),
}


def cache_dir(root: Path | None = None) -> Path:
    """Return (and create) data/structures/cache relative to the repo root."""
    d = (root or Path(__file__).resolve().parents[2]) / "data" / "structures" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_sequence(gene: str, root: Path | None = None) -> tuple[str, int]:
    """Return (sequence, heme_cys_1based) for an isoform gene.

    Verifies the documented cysteine against the fetched sequence and the
    conserved motif; raises on any disagreement so the error is loud, not
    silent.
    """
    iso = ISOFORMS[gene]
    cache = cache_dir(root) / f"{iso.uniprot}.fasta"
    if cache.exists():
        text = cache.read_text()
    else:
        resp = requests.get(_UNIPROT_URL.format(acc=iso.uniprot), timeout=60)
        resp.raise_for_status()
        text = resp.text
        cache.write_text(text)

    seq = "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))

    if seq[iso.heme_cys_1based - 1] != "C":
        raise ValueError(
            f"{gene}: documented heme Cys {iso.heme_cys_1based} is "
            f"'{seq[iso.heme_cys_1based - 1]}' in UniProt {iso.uniprot}; registry is stale"
        )

    motif_cys = [
        m.end() for m in _HEME_MOTIF.finditer(seq) if m.end() > len(seq) * 0.6
    ]  # position of the C in the motif, C-terminal half only
    if motif_cys and iso.heme_cys_1based not in motif_cys:
        raise ValueError(
            f"{gene}: motif Cys candidates {motif_cys} do not include documented "
            f"{iso.heme_cys_1based}; check registry"
        )

    return seq, iso.heme_cys_1based


def fetch_crystal(pdb_id: str, root: Path | None = None) -> Path:
    """Download (once) the mmCIF of a holo crystal structure; return its path."""
    cache = cache_dir(root) / f"{pdb_id}.cif"
    if not cache.exists():
        resp = requests.get(_RCSB_CIF.format(pdb=pdb_id), timeout=120)
        resp.raise_for_status()
        cache.write_text(resp.text)
    return cache
