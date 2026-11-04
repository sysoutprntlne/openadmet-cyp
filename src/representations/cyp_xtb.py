"""Semi-empirical (GFN2-xTB) descriptor blocks via ``tblite``.

Two blocks share one conformer-ensemble computation:

* ``XTBSolvation`` (``xtb_solv_v1``) -- solvation and ionization: ALPB solvation
  free energies in water and 1-octanol, the xTB logP derived from them, a
  heuristic fraction-neutral at pH 7.4, and the resulting logD estimate. This
  follows the permeability-feature idea of PXR rank 14 (bear), simplified:
  ionization comes from a fixed-pKa SMARTS table, not microstate enumeration.
* ``XTBElectronic`` (``xtb_elec_v1``) -- Boltzmann-averaged electronic, charge,
  surface, shape and flexibility terms, in the spirit of PXR rank 40 (briford's
  26 curated GFN2-xTB descriptors).

Geometry: ETKDGv3 conformers, MMFF94 (UFF fallback) optimized; xTB is used for
single points only (``tblite``'s Python API has no optimizer). Per-conformer
values are Boltzmann-averaged over the gas-phase xTB energies at 298.15 K.

Invalid SMILES, or molecules where embedding or xTB fails, give an all-NaN row.
Salts are reduced to their largest fragment. ``XTB_NJOBS`` (else ``NSLOTS``,
else the CPU count) sets the worker processes; each runs single-threaded.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA, rdMolDescriptors
from tqdm import tqdm

from representations.base import Representation

N_CONFS = 10
PH = 7.4
_T = 298.15
_KT_KCAL = 0.0019872041 * _T
_HARTREE_KCAL = 627.5094740631
_HARTREE_EV = 27.211386245988
_BOHR_ANG = 0.529177210903
_AU_DEBYE = 2.541746473

SOLV_FEATURES = [
    "dg_solv_water_kcal",
    "dg_solv_octanol_kcal",
    "logp_xtb",
    "log_fneutral_ph74",
    "logd_xtb",
    "dipole_gas_debye",
    "dipole_water_debye",
    "n_acid_groups",
    "n_base_groups",
]

ELEC_FEATURES = [
    "homo_ev",
    "lumo_ev",
    "gap_ev",
    "hardness_ev",
    "chem_potential_ev",
    "electrophilicity_ev",
    "charge_min",
    "charge_max",
    "charge_n_min",
    "charge_arom_n_min",
    "charge_pos_sum",
    "charge_neg_sum",
    "sasa_total",
    "sasa_polar",
    "sasa_polar_frac",
    "rg",
    "npr1",
    "npr2",
    "asphericity",
    "conf_e_range_kcal",
    "n_conf_within_3kcal",
    "n_conf_embedded",
    "n_heavy",
    "n_rot",
]

ALL_FEATURES = SOLV_FEATURES + ELEC_FEATURES

# (name, SMARTS, pKa, kind). Fixed-pKa approximation, order-of-magnitude only.
_IONIZABLE = [
    ("aliphatic_amine",
     "[NX3;!$(N-[#6,#16,#15]=[O,S,N]);!$(N-a);!$(N-C=[C,N]);!$(N-[N,O,S]);!$(N-C(F)(F))]",
     9.5, "base"),
    ("amidine_guanidine",
     "[NX3;!$(NC=O)][CX3;!$(C-[O,S])]=[NX2;!$(N-[O,N,S]);!a]", 11.5, "base"),
    ("imidazole", "[nX2;r5]:c:[nX3;r5]", 6.5, "base"),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]", 4.5, "acid"),
    ("tetrazole", "[$(c1nnn[nH]1),$(c1nn[nH]n1)]", 4.9, "acid"),
    ("acyl_sulfonamide", "C(=O)[NH]S(=O)(=O)", 4.5, "acid"),
]
_IONIZABLE_PATTERNS = [
    (Chem.MolFromSmarts(smarts), pka, kind) for _, smarts, pka, kind in _IONIZABLE
]


def _ionization(mol: Chem.Mol) -> tuple[float, int, int]:
    """Return (log10 fraction neutral at PH, n_acid, n_base) from the pKa table."""
    log_fn, n_acid, n_base, seen = 0.0, 0, 0, set()
    for patt, pka, kind in _IONIZABLE_PATTERNS:
        for match in mol.GetSubstructMatches(patt):
            key = (match[0], kind)
            if key in seen:
                continue
            seen.add(key)
            if kind == "base":
                n_base += 1
                log_fn -= math.log10(1.0 + 10.0 ** (pka - PH))
            else:
                n_acid += 1
                log_fn -= math.log10(1.0 + 10.0 ** (PH - pka))
    return log_fn, n_acid, n_base


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True)
    return max(frags, key=lambda f: f.GetNumHeavyAtoms()) if len(frags) > 1 else mol


def _xtb_single_point(numbers: np.ndarray, pos_bohr: np.ndarray, solvent: str | None) -> dict:
    from tblite.interface import Calculator

    calc = Calculator("GFN2-xTB", numbers, pos_bohr)
    calc.set("verbosity", 0)
    if solvent is not None:
        calc.add("alpb-solvation", solvent)
    res = calc.singlepoint()
    return {
        "energy": float(res.get("energy")),
        "dipole": float(np.linalg.norm(res.get("dipole"))) * _AU_DEBYE,
        "charges": np.asarray(res.get("charges")),
        "orbital_energies": np.asarray(res.get("orbital-energies")),
        "occupations": np.asarray(res.get("orbital-occupations")),
    }


def _shape(coords: np.ndarray, masses: np.ndarray) -> tuple[float, float, float, float]:
    """Radius of gyration, PMI ratios npr1/npr2, asphericity (mass-weighted)."""
    com = np.average(coords, axis=0, weights=masses)
    d = coords - com
    inertia = np.zeros((3, 3))
    for m, v in zip(masses, d):
        inertia += m * (np.dot(v, v) * np.eye(3) - np.outer(v, v))
    i1, i2, i3 = np.sort(np.linalg.eigvalsh(inertia))
    rg = math.sqrt(float(np.sum(masses * np.sum(d * d, axis=1)) / masses.sum()))
    gyr = np.sort(np.linalg.eigvalsh(np.einsum("i,ij,ik->jk", masses, d, d) / masses.sum()))
    tr = gyr.sum()
    asph = 0.0 if tr <= 0 else float(
        1.5 * np.sum((gyr - tr / 3.0) ** 2) / tr**2
    )
    if i3 <= 0:
        return rg, float("nan"), float("nan"), asph
    return rg, float(i1 / i3), float(i2 / i3), asph


def _compute_one(smiles: str | None, n_confs: int = N_CONFS) -> list[float]:
    """Return the ALL_FEATURES row (NaN row on any failure)."""
    nan_row = [float("nan")] * len(ALL_FEATURES)
    if smiles is None or not smiles.strip():
        return nan_row
    mol0 = Chem.MolFromSmiles(smiles.strip())
    if mol0 is None:
        return nan_row
    mol0 = _largest_fragment(mol0)
    log_fn, n_acid, n_base = _ionization(mol0)
    mol = Chem.AddHs(mol0)

    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    params.pruneRmsThresh = 0.5
    params.numThreads = 1
    cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:
        params.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mol, n_confs, params))
    if not cids:
        return nan_row
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, numThreads=1)
        else:
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=500, numThreads=1)
    except Exception:
        return nan_row

    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    is_n = numbers == 7
    arom_n = np.array([a.GetAtomicNum() == 7 and a.GetIsAromatic() for a in mol.GetAtoms()])
    polar = np.zeros(mol.GetNumAtoms(), dtype=bool)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() in (7, 8):
            polar[a.GetIdx()] = True
            for nb in a.GetNeighbors():
                if nb.GetAtomicNum() == 1:
                    polar[nb.GetIdx()] = True
    heavy = numbers > 1
    radii = rdFreeSASA.classifyAtoms(mol)

    per_conf: list[dict] = []
    try:
        for cid in cids:
            coords = mol.GetConformer(cid).GetPositions()
            pos_bohr = coords / _BOHR_ANG
            gas = _xtb_single_point(numbers, pos_bohr, None)
            wat = _xtb_single_point(numbers, pos_bohr, "water")
            octa = _xtb_single_point(numbers, pos_bohr, "octanol")
            n_occ = int(round(gas["occupations"].sum() / 2.0))
            homo = gas["orbital_energies"][n_occ - 1] * _HARTREE_EV
            lumo = gas["orbital_energies"][n_occ] * _HARTREE_EV
            q = gas["charges"]
            rdFreeSASA.CalcSASA(mol, radii, confIdx=cid)
            sasa = np.array([float(a.GetProp("SASA")) for a in mol.GetAtoms()])
            rg, npr1, npr2, asph = _shape(coords, masses)
            mu, hard = (homo + lumo) / 2.0, (lumo - homo) / 2.0
            per_conf.append({
                "energy": gas["energy"],
                "dg_w": (wat["energy"] - gas["energy"]) * _HARTREE_KCAL,
                "dg_o": (octa["energy"] - gas["energy"]) * _HARTREE_KCAL,
                "dip_gas": gas["dipole"],
                "dip_wat": wat["dipole"],
                "homo": homo, "lumo": lumo, "gap": lumo - homo,
                "hardness": hard, "mu": mu,
                "omega": mu * mu / (2.0 * hard) if hard > 0 else float("nan"),
                "q_min": q[heavy].min(), "q_max": q[heavy].max(),
                "q_n_min": q[is_n].min() if is_n.any() else 0.0,
                "q_arom_n_min": q[arom_n].min() if arom_n.any() else 0.0,
                "q_pos": q[heavy][q[heavy] > 0].sum(),
                "q_neg": q[heavy][q[heavy] < 0].sum(),
                "sasa": sasa.sum(), "sasa_polar": sasa[polar].sum(),
                "rg": rg, "npr1": npr1, "npr2": npr2, "asph": asph,
            })
    except Exception:
        return nan_row

    e = np.array([c["energy"] for c in per_conf]) * _HARTREE_KCAL
    e_rel = e - e.min()
    w = np.exp(-e_rel / _KT_KCAL)
    w /= w.sum()

    def avg(key: str) -> float:
        return float(np.dot(w, [c[key] for c in per_conf]))

    dg_w, dg_o = avg("dg_w"), avg("dg_o")
    logp = (dg_w - dg_o) / (_KT_KCAL * math.log(10.0))
    sasa_total = avg("sasa")
    row = {
        "dg_solv_water_kcal": dg_w,
        "dg_solv_octanol_kcal": dg_o,
        "logp_xtb": logp,
        "log_fneutral_ph74": log_fn,
        "logd_xtb": logp + log_fn,
        "dipole_gas_debye": avg("dip_gas"),
        "dipole_water_debye": avg("dip_wat"),
        "n_acid_groups": float(n_acid),
        "n_base_groups": float(n_base),
        "homo_ev": avg("homo"),
        "lumo_ev": avg("lumo"),
        "gap_ev": avg("gap"),
        "hardness_ev": avg("hardness"),
        "chem_potential_ev": avg("mu"),
        "electrophilicity_ev": avg("omega"),
        "charge_min": avg("q_min"),
        "charge_max": avg("q_max"),
        "charge_n_min": avg("q_n_min"),
        "charge_arom_n_min": avg("q_arom_n_min"),
        "charge_pos_sum": avg("q_pos"),
        "charge_neg_sum": avg("q_neg"),
        "sasa_total": sasa_total,
        "sasa_polar": avg("sasa_polar"),
        "sasa_polar_frac": avg("sasa_polar") / sasa_total if sasa_total > 0 else float("nan"),
        "rg": avg("rg"),
        "npr1": avg("npr1"),
        "npr2": avg("npr2"),
        "asphericity": avg("asph"),
        "conf_e_range_kcal": float(e_rel.max()),
        "n_conf_within_3kcal": float((e_rel <= 3.0).sum()),
        "n_conf_embedded": float(len(per_conf)),
        "n_heavy": float(mol0.GetNumHeavyAtoms()),
        "n_rot": float(rdMolDescriptors.CalcNumRotatableBonds(mol0)),
    }
    return [row[name] for name in ALL_FEATURES]


def _init_worker() -> None:
    # Set before tblite is imported so OpenMP starts single-threaded.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _n_jobs() -> int:
    return max(1, int(os.environ.get("XTB_NJOBS") or os.environ.get("NSLOTS") or os.cpu_count() or 1))


# In-process memo so a run that requests both blocks computes each molecule once.
_MEMO: dict[str, list[float]] = {}


def _compute_all(smiles: list[str | None]) -> np.ndarray:
    todo = list(dict.fromkeys(s for s in smiles if s is not None and s not in _MEMO))
    if todo:
        n_jobs = min(_n_jobs(), len(todo))
        if n_jobs == 1:
            _init_worker()
            rows = [_compute_one(s) for s in tqdm(todo, desc="xTB descriptors", unit="mol")]
        else:
            import multiprocessing as mp

            with ProcessPoolExecutor(
                max_workers=n_jobs, mp_context=mp.get_context("spawn"), initializer=_init_worker
            ) as pool:
                rows = list(tqdm(
                    pool.map(_compute_one, todo, chunksize=4),
                    total=len(todo), desc=f"xTB descriptors ({n_jobs} procs)", unit="mol",
                ))
        _MEMO.update(zip(todo, rows))
    nan_row = [float("nan")] * len(ALL_FEATURES)
    return np.array([_MEMO.get(s, nan_row) if s is not None else nan_row for s in smiles],
                    dtype=np.float64)


class _XTBBlock(Representation):
    _columns: ClassVar[list[str]]

    @property
    def feature_names(self) -> list[str]:
        return list(self._columns)

    def transform(self, smiles: pl.Series) -> np.ndarray:
        """Return a (n_molecules, n_features) float64 array; NaN rows on failure."""
        full = _compute_all(smiles.to_list())
        idx = [ALL_FEATURES.index(c) for c in self._columns]
        return full[:, idx]


class XTBSolvation(_XTBBlock):
    """ALPB water/octanol solvation, xTB logP, heuristic logD at pH 7.4."""

    name: ClassVar[str] = "xtb_solv_v1"
    _columns: ClassVar[list[str]] = SOLV_FEATURES


class XTBElectronic(_XTBBlock):
    """Boltzmann-averaged FMO, charge, SASA, shape and flexibility terms."""

    name: ClassVar[str] = "xtb_elec_v1"
    _columns: ClassVar[list[str]] = ELEC_FEATURES
