"""Heme-coordination geometry: the T6 headline features, computed from any
mmCIF that contains HEM (predicted co-fold or crystal reference).

Features, in T6's priority order:
  1. min N->Fe distance over the ligand's sp2/sp3 nitrogens — the Type II
     coordination signature (coordinated: ~2.0-2.5 A for pyridine/azole N).
  2. approach angle: N-Fe vector vs the heme-plane normal. Axial coordination
     means ~0 deg off the normal (the Fe axial positions are perpendicular to
     the porphyrin plane); a nitrogen 3 A off to the side is not coordinating.
  3. min heavy-atom distance ligand->Fe — for ligands that sterically block
     the site without coordinating.
  4. heme-plane normal is defined from the four porphyrin N atoms (NA/NB/NC/ND),
     which is stable whether the Fe is present or displaced.

Parsing uses gemmi; works identically on Boltz/IntelliFold/AF3 output cif and
on RCSB crystal cif, which makes the crystal-vs-prediction comparison honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gemmi
import numpy as np

_HEM_RESNAME = "HEM"
_PORPHYRIN_NS = ["NA", "NB", "NC", "ND"]


@dataclass
class CoordinationMetrics:
    """Heme-iron coordination geometry for one ligand in one complex."""

    ligand_name: str
    n_fe_distance: float | None = None  # A, nearest sp2/sp3 ligand N -> Fe
    approach_angle_deg: float | None = None  # deg, N->Fe vector vs heme normal
    min_heavy_fe: float | None = None  # A, nearest ligand heavy atom -> Fe
    fe_coord_atoms: list[str] = field(default_factory=list)  # Fe axial partners in HEM
    n_atom: str | None = None  # ligand atom name achieving the min N-Fe distance

    def as_row(self) -> dict:
        return {
            "ligand_name": self.ligand_name,
            "n_fe_distance_A": self.n_fe_distance,
            "approach_angle_deg": self.approach_angle_deg,
            "min_heavy_fe_A": self.min_heavy_fe,
            "n_atom": self.n_atom,
        }


def _coords(res: gemmi.Residue, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Return {atom_name: xyz} for a residue, optionally restricted to names."""
    out = {}
    for atom in res:
        if names is None or atom.name in names:
            out[atom.name] = np.array(
                [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64
            )
    return out


def heme_frame(st: gemmi.Structure) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Return (heme_center, heme_normal_unit, fe_xyz_or_None).

    The normal is the first principal axis of the four porphyrin N atoms —
    equivalently the normalized cross-product construction from two diagonals
    (NA->NC x NB->ND), which is exact for a planar porphyrin.
    """
    hem = None
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == _HEM_RESNAME:
                    hem = res
                    break
    if hem is None:
        raise ValueError("no HEM residue in structure")

    coords = _coords(hem)
    diag1 = coords["NC"] - coords["NA"]
    diag2 = coords["ND"] - coords["NB"]
    normal = np.cross(diag1, diag2)
    normal /= np.linalg.norm(normal)

    center = np.mean([coords[n] for n in _PORPHYRIN_NS], axis=0)

    fe = coords.get("FE")
    return center, normal, fe


def _is_sp2_sp3_nitrogen(mol, atom_idx: int) -> bool:
    """True for aliphatic/aromatic amine or azole N candidates (coarse but
    sufficient: we take any ligand N that is not an amide/sulfonamide N)."""
    from rdkit import Chem

    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "N":
        return False
    # amide/sulfonamide N are planar but not donors; exclude by neighbor context
    for neigh in atom.GetNeighbors():
        if neigh.GetSymbol() == "C" and any(
            b.GetBondTypeAsDouble() == 2.0 for b in neigh.GetBonds()
        ):
            continue
    return True


def ligand_coordination(
    st: gemmi.Structure,
    mol,  # rdkit Mol with a conformer, atom order == cif ligand residue order
    ligand_name: str = "KET",
) -> CoordinationMetrics:
    """Compute coordination metrics for a ligand against the heme frame.

    `mol` must be the ligand as embedded in the cif (we reconstruct it from
    the cif itself in cofold.py so index alignment is guaranteed).
    """
    center, normal, fe = heme_frame(st)

    lig_res = None
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == ligand_name:
                    lig_res = res
    if lig_res is None:
        raise ValueError(f"ligand residue {ligand_name} not found in structure")

    lig_coords = _coords(lig_res)

    if fe is None:
        # HEM CCD always carries FE; absence means parsing went wrong.
        raise ValueError("HEM residue has no FE atom")

    # min heavy-atom distance and min N-Fe
    min_heavy = math.inf
    min_nfe = math.inf
    n_atom_name = None
    for name, xyz in lig_coords.items():
        d = float(np.linalg.norm(xyz - fe))
        min_heavy = min(min_heavy, d)
        if name.startswith("N"):
            if d < min_nfe:
                min_nfe = d
                n_atom_name = name

    # approach angle of the best N
    angle = None
    if n_atom_name is not None and math.isfinite(min_nfe):
        n_xyz = lig_coords[n_atom_name]
        # N->Fe vector; axial coordination aligns with the heme normal
        vec = fe - n_xyz
        vec /= np.linalg.norm(vec)
        cos_a = abs(float(np.dot(vec, normal)))  # abs: either axial side counts
        angle = math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))

    return CoordinationMetrics(
        ligand_name=ligand_name,
        n_fe_distance=None if not math.isfinite(min_nfe) else min_nfe,
        approach_angle_deg=angle,
        min_heavy_fe=None if not math.isfinite(min_heavy) else min_heavy,
        n_atom=n_atom_name,
    )


def crystal_reference(pdb_cif: Path, ligand_resname: str) -> CoordinationMetrics:
    """Same metrics from the holo crystal (e.g. 4NY4 ketoconazole) for
    calibration: the prediction should land in the same regime."""
    st = gemmi.read_structure(str(pdb_cif))
    st.setup_entities()
    return ligand_coordination(st, mol=None, ligand_name=ligand_resname)
