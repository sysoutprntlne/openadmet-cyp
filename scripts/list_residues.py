"""List non-polymer/hetero residues in a cif to find the ligand code."""

import sys
from pathlib import Path

import gemmi

cif = Path(sys.argv[1] if len(sys.argv) > 1 else "data/structures/cache/4NY4.cif")
st = gemmi.read_structure(str(cif))
st.setup_entities()

seen = {}
for model in st:
    for chain in model:
        for res in chain:
            kind = gemmi.find_tabulated_residue(res.name)
            kind_name = kind.kind if kind else "?"
            if kind_name in ("HOH",):
                continue
            if kind is None or kind.is_amino_acid() is False:
                seen.setdefault((res.name, kind_name), 0)
                seen[(res.name, kind_name)] += 1

for (name, kind), n in sorted(seen.items()):
    print(f"{name:8s} {kind:12s} x{n}")
