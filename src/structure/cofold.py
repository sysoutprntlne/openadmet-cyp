"""CLI orchestrator: one ligand x CYP3A4 co-fold, end to end.

Usage (from repo root, with the ai_5090 env's python):
    python -m structure.cofold --ligand ketoconazole --engine auto

Steps:
  1. fetch CYP3A4 sequence (UniProt P08684) + ligand SMILES (PubChem)
  2. build engine input (IntelliFold AF3-JSON, or Boltz YAML fallback)
  3. run engine in WSL on the 5090, log wall time
  4. parse output -> common schema, write manifest row (T5 contract)
  5. compute heme-coordination metrics
  6. render HTML + PNG into reports/structures/

The manifest row is appended to data/structures/manifest.csv; failures are
recorded with completed=False rather than dropped (the T5 contract's
load-bearing flag).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from structure import engines
from structure import metrics as metrics_mod
from structure import visualize
from structure.ligands import fetch_ligand
from structure.receptors import ISOFORMS, fetch_crystal, fetch_sequence

MANIFEST_COLUMNS = [
    "inchikey_block",
    "isoform",
    "method",
    "structure_path",
    "confidence",
    "predicted_affinity",
    "completed",
]


def append_manifest(root: Path, row: dict) -> Path:
    """Append one row to data/structures/manifest.csv (create if needed)."""
    path = root / "data" / "structures" / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({c: row.get(c) for c in MANIFEST_COLUMNS})
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ligand", default="ketoconazole", choices=sorted(["ketoconazole", "itraconazole"]))
    ap.add_argument("--engine", default="auto", choices=["auto", "intellifold", "boltz2"])
    ap.add_argument("--gene", default="CYP3A4")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()

    root: Path = args.root
    run_id = f"{args.gene.lower()}_{args.ligand}"
    out_dir = root / "data" / "structures" / "runs" / run_id
    report_dir = root / "reports" / "structures"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] fetching inputs for {args.gene} + {args.ligand}")
    seq, heme_cys = fetch_sequence(args.gene, root)
    iso = ISOFORMS[args.gene]
    lig = fetch_ligand(args.ligand, root)
    print(f"    sequence len={len(seq)}  heme Cys={heme_cys}  ligand InChIKey block={lig.connectivity_block}")

    # ---- pick engine -------------------------------------------------------
    engine = args.engine
    if engine == "auto":
        engine = "intellifold" if engines._wsl(
            f"test -f {engines.INTELLIFOLD_DIR}/.venv/bin/intellifold && echo ok", timeout=30
        ).stdout.strip() == "ok" else "boltz2"
    print(f"[2/6] engine = {engine}")

    # ---- build input -------------------------------------------------------
    if engine == "intellifold":
        in_path = engines.build_intellifold_input(seq, lig.smiles, out_dir / "fold_input.json")
    else:
        in_path = engines.build_boltz_input(seq, lig.smiles, out_dir / "fold_input.yaml")
    print(f"    input: {in_path.name}")

    # ---- run ---------------------------------------------------------------
    print(f"[3/6] running {engine} in WSL on the 5090 (first run downloads weights)")
    t0 = time.time()
    if engine == "intellifold":
        ok, log, wall = engines.run_intellifold(in_path, out_dir)
        result = engines.parse_intellifold_output(out_dir, wall, log) if ok else None
    else:
        ok, log, wall = engines.run_boltz(in_path, out_dir)
        result = engines.parse_boltz_output(out_dir, wall, log) if ok else None
    wall_total = time.time() - t0

    if not ok or result is None or result.structure_path is None:
        print(f"    ENGINE FAILED after {wall_total:.0f}s; log tail:")
        print("\n".join(log.splitlines()[-25:]))
        (out_dir / "engine.log").write_text(log)
        append_manifest(
            root,
            {
                "inchikey_block": lig.connectivity_block,
                "isoform": args.gene,
                "method": engine,
                "completed": False,
            },
        )
        print("    manifest row written with completed=False")
        return 1

    print(f"    completed in {wall_total:.0f}s ({result.method})")
    (out_dir / "engine.log").write_text(log)

    # ---- metrics -----------------------------------------------------------
    print("[4/6] computing heme-coordination metrics")
    import gemmi

    st = gemmi.read_structure(str(result.structure_path))
    st.setup_entities()
    # The cif's ligand residue name: engines may rename SMILES ligands (UNL)
    # or keep the input id. Find any residue that is not the protein/HEM.
    lig_resname = None
    for model in st:
        for chain in model:
            for res in chain:
                if res.name not in {iso.gene, "HEM"} and not res.name.startswith(("A", "B")):
                    if res.name.isupper() and len(res.name) > 0 and res.name not in ("HOH",):
                        lig_resname = res.name
    if lig_resname is None:
        print("    WARNING: could not identify ligand residue in output cif")
        lig_resname = "UNK"

    m = metrics_mod.ligand_coordination(st, mol=None, ligand_name=lig_resname)
    row = m.as_row()
    print(f"    {row}")

    # crystal reference for calibration
    try:
        cif = fetch_crystal(iso.pdb_holo, root)
        # 4NY4 crystallizes ketoconazole under CCD code 2QH
        ref = metrics_mod.crystal_reference(cif, "2QH")
        print(f"    crystal reference ({iso.pdb_holo}): {ref.as_row()}")
    except Exception as exc:  # non-fatal
        print(f"    crystal reference unavailable: {exc}")

    # ---- visualize ---------------------------------------------------------
    print("[5/6] rendering visualization")
    html = visualize.render_html(result.structure_path, report_dir / f"{run_id}.html", lig_resname)
    png = visualize.render_pocket_png(
        result.structure_path, report_dir / f"{run_id}.png", lig_resname, row
    )
    print(f"    {html.name}, {png.name}")

    # ---- manifest ----------------------------------------------------------
    print("[6/6] writing manifest row")
    manifest_row = {
        "inchikey_block": lig.connectivity_block,
        "isoform": args.gene,
        "method": result.method,
        "structure_path": str(result.structure_path.relative_to(root)),
        "confidence": result.confidence,
        "predicted_affinity": result.predicted_affinity,
        "completed": result.completed,
    }
    append_manifest(root, manifest_row)
    print(f"    {manifest_row}")

    # sanity gate summary
    if m.n_fe_distance is not None and m.n_fe_distance < 3.0:
        print("SANITY GATE: PASS — azole N within coordination distance of Fe")
    else:
        print("SANITY GATE: REVIEW — azole N not within 3.0 A of Fe (pose may be wrong)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
