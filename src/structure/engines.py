"""Engine backends: IntelliFold-2 (primary) and Boltz-2 (fallback).

Engines run inside WSL2 Ubuntu (JAX/torch CUDA wheels are Linux-only; the
5090 needs CUDA >= 12.8). The repo lives on the Windows side and is visible
in WSL at /mnt/d/..., so inputs are written here and outputs parsed here;
only the engine invocation crosses the boundary.

IntelliFold-2 input: AlphaFold 3-style JSON (protein + ligand entities).
Boltz-2 input: Boltz YAML (protein + ligand entities, affinity property).

Both engines accept HEM as a CCD code — the CCD entry carries the Fe atom,
so the Cys442-Fe coordination is chemistry the models learned, not a bond we
have to specify. Protenix's explicit covalent_bonds block is the production
variant (see tasks/T5-structure-generation.md).

Output contract (tasks/T5-structure-generation.md):
    inchikey_block, isoform, method, structure_path, confidence,
    predicted_affinity, completed
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

WIN_REPO = Path(__file__).resolve().parents[2]
WSL_REPO = "/mnt/d/AI SOCIETY/Comp Bio/openadmet-cyp"
WSL_DISTRO = "Ubuntu"
INTELLIFOLD_DIR = "~/IntelliFold"


# --------------------------------------------------------------------------- #
# Input builders
# --------------------------------------------------------------------------- #

def build_intellifold_input(
    sequence: str,
    ligand_smiles: str,
    out_path: Path,
    name: str = "cyp3a4_hem_lig",
) -> Path:
    """Write an AF3-style JSON: CYP3A4 + HEM (CCD) + ligand (SMILES).

    No MSA/templates -> the caller must pass --norun_data_pipeline (the
    example ships with MSAs; ours deliberately skips the data pipeline to
    keep the smoke test inside its wall-clock budget).
    """
    payload = {
        "dialect": "alphafold3",
        "version": 1,
        "name": name,
        "modelSeeds": [0],
        "sequences": [
            # AF3 validation requires an explicit unpaired MSA per protein
            # chain even with --norun_data_pipeline; empty string = MSA-free
            # inference (no sequence databases needed on this machine).
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                }
            },
            {"ligand": {"id": "H", "ccdCodes": ["HEM"]}},
            {"ligand": {"id": "K", "smiles": ligand_smiles}},
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def build_boltz_input(
    sequence: str,
    ligand_smiles: str,
    out_path: Path,
    name: str = "cyp3a4_hem_lig",
) -> Path:
    """Write a Boltz-2 YAML: CYP3A4 + HEM (CCD) + ligand (SMILES) + affinity."""
    yaml_text = f"""version: 1
sequences:
  - protein:
      id: [A]
      sequence: {sequence}
  - ligand:
      id: [H]
      ccd: HEM
  - ligand:
      id: [K]
      smiles: "{ligand_smiles}"
properties:
  - affinity:
      binder: K
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_text)
    return out_path


# --------------------------------------------------------------------------- #
# Runners (WSL)
# --------------------------------------------------------------------------- #

def _wsl(cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a bash -lc command inside the WSL distro, streaming nothing."""
    return subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(WIN_REPO),
    )


def run_intellifold(
    input_path: Path, output_dir: Path, timeout: int = 3600
) -> tuple[bool, str, float]:
    """Run IntelliFold-2 on the JSON; return (ok, log, wall_seconds).

    Paths with spaces must be quoted — the repo lives under 'AI SOCIETY/...'.
    First run downloads the v2 weights from HuggingFace into model_v2/.
    """
    wsl_in = quote(to_wsl(input_path))
    wsl_out = quote(to_wsl(output_dir))
    cmd = (
        f"source {INTELLIFOLD_DIR}/.venv/bin/activate && "
        f"intellifold predict {wsl_in} --model-dir={INTELLIFOLD_DIR}/model_v2 "
        f"--output-dir {wsl_out} -- --norun_data_pipeline"
    )
    t0 = time.time()
    proc = _wsl(cmd, timeout=timeout)
    wall = time.time() - t0
    log = proc.stdout + proc.stderr
    ok = proc.returncode == 0
    return ok, log, wall


def run_boltz(
    input_path: Path, output_dir: Path, timeout: int = 3600
) -> tuple[bool, str, float]:
    """Run Boltz-2 on the YAML; return (ok, log, wall_seconds)."""
    wsl_in = to_wsl(input_path)
    wsl_out = to_wsl(output_dir)
    cmd = f"source ~/.venvs/boltz/bin/activate && boltz predict {wsl_in} --output_dir {wsl_out} --devices 1"
    t0 = time.time()
    proc = _wsl(cmd, timeout=timeout)
    wall = time.time() - t0
    log = proc.stdout + proc.stderr
    ok = proc.returncode == 0
    return ok, log, wall


def to_wsl(path: Path) -> str:
    """Map a Windows path to its /mnt/<drive> WSL equivalent."""
    p = str(path.resolve())
    drive = p[0].lower()
    rest = p[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


# --------------------------------------------------------------------------- #
# Output parsing -> common schema
# --------------------------------------------------------------------------- #

@dataclass
class CoFoldResult:
    """Normalized engine output, per the T5 manifest contract."""

    method: str
    structure_path: Path | None
    confidence: float | None  # ipTM-ish; engine-specific fallback
    iptm: float | None
    plddt: float | None
    predicted_affinity: float | None
    completed: bool
    wall_seconds: float
    log: str = ""


def parse_intellifold_output(output_dir: Path, wall: float, log: str) -> CoFoldResult:
    """Locate the top model's cif + confidence files under output_dir."""
    return _parse_af3_style_output(output_dir, method="intellifold2", wall=wall, log=log)


def parse_boltz_output(output_dir: Path, wall: float, log: str) -> CoFoldResult:
    """Locate Boltz-2's top model cif + confidence under output_dir."""
    return _parse_af3_style_output(output_dir, method="boltz2", wall=wall, log=log)


def _parse_af3_style_output(output_dir: Path, method: str, wall: float, log: str) -> CoFoldResult:
    """AF3/Boltz both emit model_*.cif + confidence JSON; find the best ranked one."""
    cifs = sorted(output_dir.rglob("*model*0.cif")) or sorted(output_dir.rglob("*.cif"))
    if not cifs:
        return CoFoldResult(method, None, None, None, None, None, False, wall, log)

    structure_path = cifs[0]
    conf_candidates = list(output_dir.rglob("*confidence*.json")) + list(
        output_dir.rglob("*scores*.json")
    )
    iptm = plddt = affinity = None
    if conf_candidates:
        scores = json.loads(conf_candidates[0].read_text())
        # AF3-style keys; Boltz uses slightly different names — try both.
        iptm = scores.get("iptm") or scores.get("confidence_iptm")
        plddt = scores.get("plddt") or scores.get("confidence_score")
        affinity = scores.get("affinity_probability_binary") or scores.get("affinity")

    confidence = iptm if iptm is not None else plddt
    return CoFoldResult(
        method=method,
        structure_path=structure_path,
        confidence=confidence,
        iptm=iptm,
        plddt=plddt,
        predicted_affinity=affinity,
        completed=True,
        wall_seconds=wall,
        log=log,
    )
