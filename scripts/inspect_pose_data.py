"""Inspect the downloaded cyp-cofold-poses dataset: main CSV, scorecard, bake-off manifest."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "pose_data" / "hf"

df = pd.read_csv(ROOT / "p450_poses_scored.csv")
print(f"p450_poses_scored.csv: {len(df)} rows x {len(df.columns)} cols")
print("columns:", list(df.columns))
print("\nengine counts:\n", df["engine"].value_counts())
print("\ncoordinated counts:\n", df["coordinated"].value_counts())
print("\nper-engine coordination rate:")
print(df.groupby("engine")["coordinated"].mean().round(3))
print("\nlddt_pli / bisy_rmsd by engine:")
print(df.groupby("engine")[["lddt_pli", "bisy_rmsd"]].describe().T.round(3))
print("\npdb count:", df["pdb"].nunique(), "| uniprot count:", df["uniprot"].nunique())

score = json.loads((ROOT / "pool_scorecard.json").read_text())
print("\npool_scorecard.json: list of", len(score), "entries; first two:")
print(json.dumps(score[:2], indent=1)[:1200])

bake = ROOT / "t5_bakeoff"
for f in ("compounds.csv", "t5_manifest.csv"):
    if (bake / f).exists():
        b = pd.read_csv(bake / f)
        print(f"\nt5_bakeoff/{f}: {len(b)} rows x {len(b.columns)} cols")
        print("columns:", list(b.columns))
        print(b.head(3).to_string())
