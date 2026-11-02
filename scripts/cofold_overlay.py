"""Render the prediction-vs-crystal overlay figure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure.visualize import render_overlay_png  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
pred = ROOT / "data/structures/runs/cyp3a4_ketoconazole/output/cyp3a4_hem_lig/cyp3a4_hem_lig_model.cif"
crys = ROOT / "data/structures/cache/4NY4.cif"
out = ROOT / "reports/structures/cyp3a4_ketoconazole_overlay.png"

print(render_overlay_png(pred, crys, out))
