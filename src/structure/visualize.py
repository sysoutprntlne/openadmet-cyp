"""Visualization of a co-fold complex: interactive HTML + static PNG.

HTML: py3Dmol (3Dmol.js) — protein cartoon, heme sticks, ligand sticks, and
dashed lines for the Fe axial coordination. Opens in any browser.

PNG: matplotlib scatter/slice through the heme region — publication-grade
enough for reports/ without a Blender dependency. We plot the heme atoms,
the ligand heavy atoms, and annotate the N-Fe distance.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import py3Dmol


def render_html(
    cif_path: Path,
    out_html: Path,
    ligand_resname: str = "KET",
    focus_radius: float = 12.0,
) -> Path:
    """Render an interactive 3D view: protein cartoon + heme + ligand."""
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()

    # Find heme Fe position so we can center the camera on the pocket.
    fe_pos = None
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == "HEM":
                    for atom in res:
                        if atom.name == "FE":
                            fe_pos = [atom.pos.x, atom.pos.y, atom.pos.z]
    if fe_pos is None:
        raise ValueError("no FE in HEM; cannot center camera")

    data = Path(cif_path).read_text()
    view = py3Dmol.view(width=900, height=700)
    view.addModel(data, "cif")
    view.setStyle({"cartoon": {"color": "spectrum"}, "model": 0})
    view.addStyle({"resn": "HEM"}, {"stick": {"colorscheme": "brownCarbon", "radius": 0.2}})
    view.addStyle(
        {"resn": ligand_resname},
        {"stick": {"colorscheme": "cyanCarbon", "radius": 0.25}},
    )
    # Zoom to the pocket around the Fe.
    view.zoomTo({"position": {"x": fe_pos[0], "y": fe_pos[1], "z": fe_pos[2]}})
    view.zoom(2.2)
    view.spin(True)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(view._make_html())
    return out_html


def render_pocket_png(
    cif_path: Path,
    out_png: Path,
    ligand_resname: str = "KET",
    metrics_row: dict | None = None,
    cutoff: float = 14.0,
) -> Path:
    """Static slice: heme + ligand + Fe within `cutoff` A of the Fe.

    matplotlib 3D scatter; size encodes depth. Good enough to see whether
    the azole reaches the iron.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()

    hem_atoms, lig_atoms, fe = [], [], None
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == "HEM":
                    for atom in res:
                        p = [atom.pos.x, atom.pos.y, atom.pos.z]
                        hem_atoms.append((atom.name, p))
                        if atom.name == "FE":
                            fe = np.array(p)
                elif res.name == ligand_resname:
                    for atom in res:
                        lig_atoms.append((atom.name, [atom.pos.x, atom.pos.y, atom.pos.z]))

    if fe is None:
        raise ValueError("no FE found")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    h = np.array([p for _, p in hem_atoms])
    keep = np.linalg.norm(h - fe, axis=1) < cutoff
    h = h[keep]
    ax.scatter(h[:, 0], h[:, 1], h[:, 2], c="#8B4513", s=40, label="HEM", alpha=0.8)

    if lig_atoms:
        l = np.array([p for _, p in lig_atoms])
        keep = np.linalg.norm(l - fe, axis=1) < cutoff
        l = l[keep]
        ax.scatter(l[:, 0], l[:, 1], l[:, 2], c="#00BFFF", s=45, label=ligand_resname, alpha=0.9)

    ax.scatter(*fe, color="red", s=220, marker="*", label="Fe")

    title = f"CYP3A4 pocket: HEM + {ligand_resname}"
    if metrics_row:
        nfe = metrics_row.get("n_fe_distance_A")
        ang = metrics_row.get("approach_angle_deg")
        title += (
            f"\nN-Fe {nfe:.2f} A, angle {ang:.1f} deg"
            if nfe is not None and ang is not None
            else "\nno N within coordination distance"
        )
    ax.set_title(title)
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# Prediction vs crystal overlay
# --------------------------------------------------------------------------- #

def _hem_atoms(cif_path: Path) -> dict[str, np.ndarray]:
    """Return {atom_name: xyz} for the first HEM residue in a cif."""
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    for model in st:
        for chain in model:
            for res in chain:
                if res.name == "HEM":
                    return {
                        a.name: np.array([a.pos.x, a.pos.y, a.pos.z])
                        for a in res
                    }
    raise ValueError(f"no HEM in {cif_path}")


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) minimizing ||mobile @ R.T + t - target|| (mobile is Nx3)."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    P, Q = mobile - mc, target - tc
    V, S, Wt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    return R, tc - mc @ R.T


def render_overlay_png(
    pred_cif: Path,
    crystal_cif: Path,
    out_png: Path,
    pred_resname: str = "LIG_K",
    crystal_resname: str = "2QH",
    cutoff: float = 16.0,
) -> Path:
    """Heme-superposed prediction vs crystal overlay.

    Both complexes are aligned on their shared HEM atom names (Kabsch), so
    the two ligands and two Fe atoms can be compared directly: how far the
    predicted pose sits from the crystal pose in the pocket frame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = _hem_atoms(pred_cif)
    crys = _hem_atoms(crystal_cif)
    common = sorted(set(pred) & set(crys))
    if len(common) < 10:
        raise ValueError(f"only {len(common)} shared HEM atom names; alignment unsafe")

    R, t = _kabsch(
        np.stack([pred[n] for n in common]),
        np.stack([crys[n] for n in common]),
    )

    def _lig(cif: Path, resname: str, transform: bool) -> np.ndarray:
        st = gemmi.read_structure(str(cif))
        st.setup_entities()
        for model in st:
            for chain in model:
                for res in chain:
                    if res.name == resname:
                        pts = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in res])
                        return pts @ R.T + t if transform else pts
        raise ValueError(f"residue {resname} not in {cif}")

    lig_p = _lig(pred_cif, pred_resname, transform=True)
    lig_c = _lig(crystal_cif, crystal_resname, transform=False)

    fe_p = pred["FE"] @ R.T + t
    fe_c = crys["FE"]
    hem_ref = np.stack([crys[n] for n in common])
    center = hem_ref.mean(axis=0)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    keep = np.linalg.norm(hem_ref - center, axis=1) < cutoff
    ax.scatter(*hem_ref[keep].T, c="#8B4513", s=28, alpha=0.55, label="HEM (crystal frame)")

    for pts, color, name, marker in (
        (lig_c, "#00BFFF", f"ketoconazole crystal ({crystal_resname})", "o"),
        (lig_p, "#FF5533", "ketoconazole prediction (LIG_K)", "D"),
    ):
        k = np.linalg.norm(pts - center, axis=1) < cutoff
        ax.scatter(*pts[k].T, c=color, s=48, marker=marker, alpha=0.85, label=name)

    ax.scatter(*fe_c, c="black", s=260, marker="*", label="Fe crystal")
    ax.scatter(*fe_p, c="red", s=160, marker="P", label="Fe predicted")

    fe_dist = float(np.linalg.norm(fe_p - fe_c))
    # nearest N-Fe pair across the two poses
    d = np.linalg.norm(lig_p[:, None, :] - lig_c[None, :, :], axis=2)
    ax.set_title(
        f"IntelliFold-2 vs 4NY4 crystal, heme-superposed\n"
        f"Fe shift {fe_dist:.2f} A | ligand-ligand min heavy-atom {d.min():.2f} A"
    )
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    ax.legend(loc="upper left", fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png
