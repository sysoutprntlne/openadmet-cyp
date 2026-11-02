# src/representations/cyp_mech.py
"""Mechanistic feature block: nitrogen/acid functionality, protonation state at
pH 7.4, and basic-nitrogen-to-aryl geometry (CYP2D6 salt-bridge pharmacophore).

Ported from the Lizard Wizard Gizzard team's open-code CYP Challenge submission:

    https://github.com/doctawho42/CYPChallenge
    (docs/method_report.md; code in src/feats.py `mech_block` / `dimorphite_charge`,
     and src/pka.py `most_basic_pka`, `frac_protonated`; read at commit bf018b8)
    MIT License, Copyright (c) 2026 the CYPChallenge contributors

The SMARTS tables, the pKa class table and its inductive-penalty weights, the
column definitions and their grouping are theirs. What changed here: adapted to
the `Representation` interface (polars Series in, float64 array out), invalid
SMILES give an all-NaN row instead of raising, the 30 columns are named and
grouped in module constants, and the dimorphite-dl call targets the 2.x API.

Their finding, for context (METHOD.md items 119, 230, 239): the block's value is
the geometry of the basic centre, not the pKa call and not the charge. Treat the
`state` group as redundant until measured here.

This module and `cyp_pharmacophore.py` overlap in intent but not in columns;
both are kept so either can be ablated.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from tqdm import tqdm

from representations.base import Representation

# Bump when any SMARTS, pKa class, penalty weight, or column changes; the
# registered name carries it so cached features cannot go stale.
TABLE_VERSION = "v1"

# --- pKa of the most basic centre (Lizard Wizard Gizzard, src/pka.py) -----------------
# (name, SMARTS with the basic atom first, intrinsic pKa of the unsubstituted parent).
# Order matters: the first class to claim an atom wins.
_PKA_CLASSES = [
    ("guanidine", "[NX3;!$(N[CX3]=[OX1])][CX3](=[NX2;!$(N[CX3]=[OX1])])[NX3]", 13.0),
    ("amidine", "[NX2;!$(N[CX3]=[OX1])]=[CX3][NX3]", 11.5),
    ("amine_2ary", "[NX3;H1;!$(N[a]);!$(N[CX3]=[OX1,SX1,NX2]);!$(N[SX4]);!$(N[PX4]);!$(N#*);!$(N=*);!$([N+])]([CX4])[CX4]", 10.9),
    ("amine_1ary", "[NX3;H2;!$(N[a]);!$(N[CX3]=[OX1,SX1,NX2]);!$(N[SX4]);!$(N[PX4])][CX4]", 10.6),
    ("amine_3ary", "[NX3;H0;!$(N[a]);!$(N[CX3]=[OX1,SX1,NX2]);!$(N[SX4]);!$(N[PX4]);!$(N#*);!$(N=*);!$([N+])]([CX4])([CX4])[CX4]", 9.8),
    ("imidazole", "[nX2;H0;$(n1cncc1),$(n1cnc[nH]1),$(n1cncn1)]", 7.0),
    ("pyridine", "[nX2;H0]", 5.2),
    ("aniline", "[NX3;!$(N[CX3]=[OX1]);!$(N[SX4]);!$(N#*);!$(N=*)][a]", 4.6),
]
_PKA_CLASSES = [(n, Chem.MolFromSmarts(s), p) for n, s, p in _PKA_CLASSES]

# Inductive withdrawal weight by topological distance from the basic nitrogen.
_INDUCTIVE_W = {2: 1.55, 3: 0.62, 4: 0.20}


def _parent_ring_atoms(mol: Chem.Mol, n_idx: int, cls: str) -> set[int]:
    """Atoms already inside the class's intrinsic pKa, so not penalised twice."""
    if cls not in ("pyridine", "imidazole", "aniline"):
        return set()
    out: set[int] = set()
    rings = mol.GetRingInfo().AtomRings()
    if cls in ("pyridine", "imidazole"):
        for ring in rings:
            if n_idx in ring:
                out.update(ring)
    else:
        for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors():
            if nb.GetIsAromatic():
                for ring in rings:
                    if nb.GetIdx() in ring:
                        out.update(ring)
    return out


def _ewg_penalty(mol: Chem.Mol, n_idx: int, dm: np.ndarray, skip: set[int]) -> float:
    pen = 0.0
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i == n_idx or i in skip:
            continue
        d = int(dm[n_idx][i])
        if d not in _INDUCTIVE_W:
            continue
        sym, arom = a.GetSymbol(), a.GetIsAromatic()
        s = 0.0
        if sym == "F":
            s = 1.0
        elif sym in ("Cl", "Br", "I"):
            s = 0.75
        elif sym == "O":
            s = 1.15
        elif sym == "N" and not arom:
            s = 0.55
        elif sym == "N" and arom:
            s = 0.5
        elif sym == "S":
            s = 0.5
        elif sym == "C" and arom:
            s = 0.45
        elif sym == "C" and any(b.GetBondTypeAsDouble() >= 2 for b in a.GetBonds()):
            s = 0.35
        pen += _INDUCTIVE_W[d] * s
    return pen


def most_basic_pka(mol: Chem.Mol) -> tuple[float, int, str]:
    """(pKa of the most basic centre, its atom index, class name); pKa -99 if none.

    A rule-based estimate, not a research-grade predictor.
    """
    dm = Chem.GetDistanceMatrix(mol)
    best = (-99.0, -1, "none")
    seen: set[int] = set()
    for name, patt, base in _PKA_CLASSES:
        for match in mol.GetSubstructMatches(patt):
            i = match[0]
            if i in seen:
                continue
            seen.add(i)
            pk = base - _ewg_penalty(mol, i, dm, _parent_ring_atoms(mol, i, name))
            if pk > best[0]:
                best = (pk, i, name)
    return best


def frac_protonated(pka: float, ph: float = 7.4) -> float:
    if pka < -50:
        return 0.0
    return 1.0 / (1.0 + 10 ** (ph - pka))


# --- functional-group counts and dimorphite sites (Lizard Wizard Gizzard, src/feats.py) --
_COUNT_SMARTS = {
    "basicN_ali": "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N[SX4](=O)=O);!$(N#*);!$([N+]);!$(N=*)]",
    "amine_prim": "[NX3;H2;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N[SX4](=O)=O)]",
    "amine_sec": "[NX3;H1;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N[SX4](=O)=O);!$(N=*)]",
    "amine_tert": "[NX3;H0;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N[SX4](=O)=O);!$(N=*);!$([N+])]",
    "amidine": "[NX3][CX3]=[NX2]",
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "pyridineN": "[nX2;$(n1ccccc1),$(n1ccncc1)]",
    "imidazoleN": "[nX2;$(n1ccnc1)]",
    "anilineN": "[NX3;!$(N[#6]=[O,N,S])][a]",
    "amideN": "[NX3][CX3]=[OX1]",
    "morpholine": "C1COCCN1",
    "piperazine": "C1CNCCN1",
    "piperidine": "[NX3;R]1[CX4][CX4][CX4][CX4][CX4]1",
    "acid": "[CX3](=O)[OX2H1]",
    "tetrazole": "c1nnn[nH]1",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
}
_COUNT_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in _COUNT_SMARTS.items()}

# Dimorphite sites whose protonated form is a cation (Bronsted bases); every other
# site in its table is an acid and must not enter a "most basic centre" statistic.
_BASIC_SITES = {
    "AmidineGuanidine1", "AmidineGuanidine2", "Anilines_primary", "Anilines_secondary",
    "Anilines_tertiary", "Aromatic_nitrogen_unprotonated", "*Aromatic_nitrogen_protonated",
    "Amines_primary_secondary_tertiary", "Primary_hydroxyl_amine",
}

GROUPS = [f"n_{k}" for k in _COUNT_SMARTS]
STATE = [
    "pka_max_basic", "frac_prot_74", "is_base_74", "n_basic_sites", "formal_charge",
    "ph74_net_charge", "ph74_n_cation", "ph74_n_anion",
]
GEOMETRY = [
    "topo_bN_to_arom_min", "topo_bN_to_arom_mean", "n_bN_geom", "topo_mbc_to_arom",
    "pharm_2d6", "pharm_2d6_x_prot",
]
FEATURES = GROUPS + STATE + GEOMETRY  # 16 + 8 + 6 = 30
_DIMORPHITE_COLS = ["ph74_net_charge", "ph74_n_cation", "ph74_n_anion"]


def _load_site_table() -> list[tuple[str, Chem.Mol, list[float]]]:
    """name -> (pattern, pKas) for every protonation site dimorphite-dl knows."""
    import dimorphite_dl

    path = Path(dimorphite_dl.__file__).parent / "smarts" / "site_substructures.smarts"
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        pkas = [float(f[i + 1]) for i in range(2, len(f), 3) if abs(float(f[i + 1])) < 900]
        patt = Chem.MolFromSmarts(f[1])
        if patt is not None and pkas:
            out.append((f[0], patt, pkas))
    return out


_SITES: list | None = None


def _n_basic_sites(mol: Chem.Mol) -> int:
    global _SITES
    if _SITES is None:
        _SITES = _load_site_table()
    return sum(
        len(mol.GetSubstructMatches(patt)) for name, patt, _ in _SITES if name in _BASIC_SITES
    )


def _topo_basic_to_aryl(mol: Chem.Mol) -> tuple[float, float, int]:
    """Bond distance from each candidate basic N to the nearest aromatic atom."""
    arom = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    bn = [i for (i,) in mol.GetSubstructMatches(_COUNT_PATTERNS["basicN_ali"])]
    bn += [m[0] for m in mol.GetSubstructMatches(_COUNT_PATTERNS["amidine"])]
    bn = sorted(set(bn))
    if not bn or not arom:
        return -1.0, -1.0, 0
    dm = Chem.GetDistanceMatrix(mol)
    ds = [min(dm[i][j] for j in arom) for i in bn]
    return float(min(ds)), float(np.mean(ds)), len(bn)


def _dimorphite_charges(smiles: str, ph: float = 7.4) -> tuple[float, float, float]:
    from dimorphite_dl import protonate_smiles

    try:
        out = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=0.0)
        mol = Chem.MolFromSmiles(out[0]) if out else None
    except Exception:
        mol = None
    if mol is None:
        return np.nan, np.nan, np.nan
    chs = [a.GetFormalCharge() for a in mol.GetAtoms()]
    return float(sum(chs)), float(sum(c > 0 for c in chs)), float(sum(c < 0 for c in chs))


def _compute_one(smiles: str | None) -> list[float]:
    nan_row = [np.nan] * len(FEATURES)
    if smiles is None or not smiles.strip():
        return nan_row
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return nan_row

    d: dict[str, float] = {f"n_{k}": float(len(mol.GetSubstructMatches(p))) for k, p in _COUNT_PATTERNS.items()}
    pka, idx, _ = most_basic_pka(mol)
    d["pka_max_basic"] = pka if pka > -50 else 0.0
    d["frac_prot_74"] = frac_protonated(pka)
    d["is_base_74"] = 1.0 if pka > 7.4 else 0.0
    d["n_basic_sites"] = float(_n_basic_sites(mol))
    d["formal_charge"] = float(Chem.GetFormalCharge(mol))
    d["topo_bN_to_arom_min"], d["topo_bN_to_arom_mean"], nb = _topo_basic_to_aryl(mol)
    d["n_bN_geom"] = float(nb)
    arom = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if idx >= 0:
        dm = Chem.GetDistanceMatrix(mol)
        d["topo_mbc_to_arom"] = float(min((dm[idx][j] for j in arom), default=-1.0))
    else:
        d["topo_mbc_to_arom"] = -1.0
    # CYP2D6 pharmacophore: a cation at pH 7.4 sitting 2-5 bonds (~5-7 A) from an aryl.
    d["pharm_2d6"] = 1.0 if 2 <= d["topo_mbc_to_arom"] <= 5 else 0.0
    d["pharm_2d6_x_prot"] = d["pharm_2d6"] * d["frac_prot_74"]
    d.update(zip(_DIMORPHITE_COLS, _dimorphite_charges(smiles.strip())))
    return [d[f] for f in FEATURES]


class CYPMechanisticBlock(Representation):
    """30 mechanistic columns: group counts, pH 7.4 protonation state, and
    basic-N-to-aryl geometry. Invalid SMILES give an all-NaN row."""

    name: ClassVar[str] = f"cyp_mech_{TABLE_VERSION}"

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURES)

    def transform(self, smiles: pl.Series) -> np.ndarray:
        out = np.full((len(smiles), len(FEATURES)), np.nan, dtype=np.float64)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc="mechanistic block", unit="mol", leave=True)):
            out[i] = _compute_one(smi)
        return out
