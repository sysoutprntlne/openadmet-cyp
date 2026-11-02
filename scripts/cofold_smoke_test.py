"""Smoke-test the repo-side structure pipeline without the engine.

Checks: sequence fetch + Cys verification, ligand fetch, crystal-reference
metrics from 4NY4 (ketoconazole holo crystal). Run:
    python scripts/cofold_smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure.ligands import fetch_ligand  # noqa: E402
from structure.receptors import ISOFORMS, fetch_crystal, fetch_sequence  # noqa: E402
from structure import metrics as M  # noqa: E402


def main() -> int:
    seq, cys = fetch_sequence("CYP3A4")
    print(f"CYP3A4 seq len={len(seq)} heme Cys={cys} context={seq[cys-6:cys+5]}")

    for gene in ("CYP1A2", "CYP2C9", "CYP2D6"):
        s, c = fetch_sequence(gene)
        assert s[c - 1] == "C", gene
        print(f"{gene}: len={len(s)} heme Cys={c} ok")

    lig = fetch_ligand("ketoconazole")
    print(f"ketoconazole: {lig.smiles[:70]}...")
    print(f"  InChIKey block: {lig.connectivity_block}")

    cif = fetch_crystal("4NY4")
    print(f"4NY4 cif: {cif.stat().st_size/1e6:.1f} MB")
    ref = M.crystal_reference(cif, "2QH")  # 4NY4's CCD code for ketoconazole
    print(f"crystal reference: {ref.as_row()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
