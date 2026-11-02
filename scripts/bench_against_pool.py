"""Compare our IntelliFold-2 ketoconazole x CYP3A4 smoke-test result against
the cyp-cofold-poses pool distributions (same metric definitions where possible)."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "pose_data" / "hf"

df = pd.read_csv(HF / "p450_poses_scored.csv")
print("=== pool (p450_poses_scored.csv) ===")
print(f"rows={len(df)}  ligands(pairs)={df['pair'].nunique()}  pdbs={df['pdb'].nunique()}")
print(f"frac under 2A bisy_rmsd: {(df['bisy_rmsd'] < 2).mean():.3f}")
print(f"frac coordinated: {df['coordinated'].mean():.3f}")

score = json.loads((HF / "pool_scorecard.json").read_text())
print("\n=== scorecard tags ===")
for s in score:
    if not isinstance(s, dict) or "arms" not in s:
        print(f"(skipping non-scorecard entry: {str(s)[:100]})")
        continue
    print(f"{s.get('tag', '?')}: n_poses={s.get('n_poses')} ligands={s.get('n_ligands')}")
    for arm_name, arm in s["arms"].items():
        p = arm.get("fe_donor_dist_p", {})
        print(
            f"  {arm_name:>10}: coord={arm.get('frac_poses_coordinated'):.3f}"
            f"  <2A={arm.get('frac_poses_under_2A'):.3f}"
            f"  fe_donor p50={p.get('p50')} p25={p.get('p25')} p95={p.get('p95')}"
        )

# our smoke test: N-Fe 2.13 A, angle 7.7 deg, ipTM 0.77
OURS_NFE = 2.13
print("\n=== our smoke test (IntelliFold-2, ketoconazole x CYP3A4) ===")
print(f"N-Fe {OURS_NFE} A")
all_fe = [
    v
    for s in score
    if isinstance(s, dict) and "arms" in s
    for a in s["arms"].values()
    for v in a.get("fe_donor_dist_p", {}).values()
]
if all_fe:
    import numpy as np

    pct = np.mean([OURS_NFE >= v for v in all_fe])
    print(f"our N-Fe vs pool fe_donor percentiles: {pct:.0%} of reported percentiles are above ours")

# does the pool contain our test case?
hit = df[df["pair"].str.contains("KET", case=False, na=False)]
print(f"\npool rows mentioning KET ligand: {len(hit)}")
if len(hit):
    print(hit[["pair", "engine", "coordinated", "lddt_pli", "bisy_rmsd"]].to_string())
