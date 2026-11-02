"""Compute heme-coordination metrics on the predicted co-fold model."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gemmi  # noqa: E402

from structure import metrics as M  # noqa: E402
from structure.visualize import render_html, render_pocket_png  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "structures" / "runs" / "cyp3a4_ketoconazole"
OUT = RUN / "output" / "cyp3a4_hem_lig"

cif = OUT / "cyp3a4_hem_lig_model.cif"
st = gemmi.read_structure(str(cif))
st.setup_entities()

m = M.ligand_coordination(st, mol=None, ligand_name="LIG_K")
row = m.as_row()
print("PREDICTION (IntelliFold-2, ranked-best):", json.dumps(row, indent=1))

summary = json.loads((OUT / "cyp3a4_hem_lig_summary_confidences.json").read_text())
print(f"iptm={summary['iptm']} ptm={summary['ptm']} ranking={summary['ranking_score']} "
      f"clash={summary['has_clash']} disordered={summary['fraction_disordered']}")

# crystal reference from earlier calibration
ref = M.crystal_reference(ROOT / "data" / "structures" / "cache" / "4NY4.cif", "2QH")
print("CRYSTAL (4NY4):", ref.as_row())

# visualization
html = render_html(cif, ROOT / "reports" / "structures" / "cyp3a4_ketoconazole.html", "LIG_K")
png = render_pocket_png(cif, ROOT / "reports" / "structures" / "cyp3a4_ketoconazole.png", "LIG_K", row)
print(f"wrote {html.name}, {png.name}")
