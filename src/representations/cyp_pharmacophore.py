# src/representations/cyp_pharmacophore.py
"""CYP-specific pharmacophore features: heme-coordination motifs, isoform
preference descriptors, and mechanism-based inactivation alerts.

The motif table below is the single source of truth. `src/data_tools/motif_report.py`
generates `reports/pharmacophore_features.md` from it, so the published report and
the computed features cannot drift, and a motif without a literature `source`
raises rather than emitting a blank cell.

Every SMARTS here was checked against named compounds before being committed;
see `tests/test_cyp_pharmacophore.py`. Sources are catalogued in
`docs/superpowers/specs/2026-10-08-cyp-pharmacophore-features-design.md` section 12.

DO NOT add a motif because it improved a validation score. A motif added that way
is a fitted parameter. Fix the list from the literature first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors, rdReducedGraphs
from tqdm import tqdm

from representations.base import Representation

# Bump when any motif's SMARTS, order, or membership changes. The registered
# name carries this version, so the content-addressed cache in
# `data_tools.inputs._featurize_one` cannot serve stale columns under a new table.
TABLE_VERSION = "v2"


@dataclass(frozen=True)
class Motif:
    """One SMARTS-counted substructure feature."""

    name: str
    smarts: str
    block: str  # "heme" | "tdi"
    source: str  # literature citation; empty blocks the report generator
    rollup: str = ""  # T6 donor label, for the 2D/3D comparison
    note: str = ""


# Donor nitrogen: aromatic, two-connected, no hydrogen. Covers pyridine,
# pyrimidine, imidazole N3, triazole/tetrazole N, thiazole N, oxazole N.
_DONOR = "[nX2H0]"
# Hindered = the donor has a substituted (or ring-fused) aromatic neighbour.
_DONOR_HINDERED = "[nX2H0;$(n:[cH0])]"
_DONOR_UNHINDERED = "[nX2H0;!$(n:[cH0])]"

_MONK_2020 = (
    "Monk et al. 2020, Biochim Biophys Acta Proteins Proteom — azoles carry an "
    "imidazole, triazole or tetrazole whose ring nitrogen coordinates as the "
    "sixth axial ligand to the heme iron."
)
_MOHAMED_2023 = (
    "Mohamed et al. 2023, J Inorg Biochem — imidazole- and pyridine-based ligands "
    "coordinate directly to ferric heme-thiolate as type II nitrogen donors."
)
_PERRIN_2009 = (
    "Perrin et al. 2009, Metallomics — inhibition potency depends on accessibility "
    "of the imidazolyl nitrogen lone pair, tuned by steric hindrance near the ring."
)
_PODGORSKI_2020 = (
    "Podgorski et al. 2020, Biochemistry — crystal structures: 4-(pyridin-3-yl)"
    "benzoic acid coordinates the heme iron; 4-(pyridin-2-yl)benzoic acid does not."
)
_PAINE_2003 = (
    "Paine et al. 2003, J Biol Chem — CYP2D6 substrates typically contain a basic "
    "nitrogen atom; Glu-216 and Asp-301 define that specificity."
)
_MIRZAEI_2021 = (
    "Mirzaei et al. 2021, Chem Res Toxicol — review of mechanism-based "
    "inactivation latent groups: thiophene, furan, alkylamines, terminal acetylene."
)
_MAO_2025 = (
    "Mao 2025, J Pharm Anal — furan, thiophene, acetylene and methylenedioxyphenyl "
    "drive CYP3A4 mechanism-based inactivation; MDP, alkylamine and hydrazine drive "
    "quasi-irreversible inactivation."
)
_DANG_2017 = (
    "Dang et al. 2017, Chem Res Toxicol — furan and thiophene alerts are highly "
    "conditional: each can form reactive metabolites through certain pathways but "
    "not always. Expect weak signal on direct-inhibition heads."
)

_AZOLE_EXTRAPOLATION = (
    "EXTRAPOLATION, not directly sourced: Monk et al. 2020 names imidazole, triazole "
    "and tetrazole as the azole heme-ligating pharmacophore. Thiazole and oxazole are "
    "five-membered azoles whose pyridine-type ring nitrogen is assumed to coordinate "
    "similarly, but no source in the spec's section 12 covers them. Treat these "
    "columns as unvalidated."
)
_DIAZINE_EXTRAPOLATION = (
    "EXTRAPOLATION, not directly sourced: Mohamed et al. 2023 covers imidazole- and "
    "pyridine-based ligands coordinating ferric heme-thiolate. Pyrimidine is a diazine "
    "bearing a pyridine-type donor nitrogen and is assumed to behave likewise; that "
    "extension is not in the paper."
)
_T6_AMINE_CATEGORY = (
    "tasks/round1/T6-structure-features.md:64-65 — T6's own coordinating-environment "
    "category list names 'aliphatic amine' alongside imidazole, triazole and pyridine, "
    "which is what this column mirrors. Note Paine et al. 2003 supports only that a "
    "basic nitrogen is characteristic of CYP2D6 ligands; it says nothing about "
    "aliphatic amines ligating heme iron."
)
_TPSA_EXTRAPOLATION = (
    "EXTRAPOLATION, not directly sourced: Kiani et al. 2019 names molecular weight, "
    "H-bond acceptors/donors and lipophilicity as the important descriptors of CYP "
    "inhibitors. TPSA is a polarity surrogate for that finding and is not named in "
    "the paper."
)
_CSP3_EXTRAPOLATION = (
    "EXTRAPOLATION, not directly sourced: Sansen et al. 2007 shows CYP1A2's cavity is "
    "adapted to large planar substrates. Fraction Csp3 is the three-dimensionality "
    "complement of that observation and is not a measure the paper uses."
)

_MOTIFS: tuple[Motif, ...] = (
    # --- ring systems bearing a coordinating nitrogen ---
    Motif("heme_imidazole", "[nX2H0]1c[nX3]cc1", "heme", _MONK_2020, "imidazole"),
    Motif("heme_triazole_124", "[nX2H0]1ncnc1", "heme", _MONK_2020, "triazole",
          "1,2,4-triazole ring; ring count, not donor count"),
    Motif("heme_triazole_123", "n1n[nX2H0]cc1", "heme", _MONK_2020, "triazole"),
    Motif("heme_tetrazole", "[nX2H0]1nnnc1", "heme", _MONK_2020, "triazole",
          "Tetrazole nitrogen also ligates heme (Monk 2020)."),
    Motif("heme_pyridine", "[nX2H0]1ccccc1", "heme", _MOHAMED_2023, "pyridine"),
    Motif("heme_pyrimidine", "[nX2H0]1cnccc1", "heme", _DIAZINE_EXTRAPOLATION, "pyridine"),
    Motif("heme_thiazole", "s1c[nX2H0]cc1", "heme", _AZOLE_EXTRAPOLATION, "imidazole",
          "Five-membered azole; donor N resembles imidazole N3 more than pyridine."),
    Motif("heme_oxazole", "o1c[nX2H0]cc1", "heme", _AZOLE_EXTRAPOLATION, "imidazole"),
    # --- donor accessibility ---
    Motif("heme_donor_total", _DONOR, "heme", _MOHAMED_2023, "",
          "All aromatic sp2 donor nitrogens, hindered or not."),
    Motif("heme_donor_unhindered", _DONOR_UNHINDERED, "heme", _PODGORSKI_2020, "",
          "Encodes the steric half of lone-pair accessibility only; Perrin 2009 "
          "shows intramolecular H-bonding also tunes it, and that is not captured."),
    Motif("heme_donor_hindered", _DONOR_HINDERED, "heme", _PERRIN_2009, ""),
    # --- aliphatic amines ---
    Motif("heme_amine_primary",
          "[NX3;H2;!$(NC=O);!$(NS(=O)=O);!$(N[a]);!$(N=*);!$(NC=[N,S,O]);!$(N[N,O])]",
          "heme", _T6_AMINE_CATEGORY, "aliphatic amine",
          "Excludes amides, sulfonamides and anilines, which are not donors."),
    Motif("heme_amine_basic",
          "[NX3;H0,H1;!$(NC=O);!$(NS(=O)=O);!$(N[a]);!$(N=*);!$(NC=[N,S,O]);!$(N[N,O]);!$(N#*)]",
          "heme", _T6_AMINE_CATEGORY, "aliphatic amine",
          "Secondary and tertiary basic aliphatic amines."),
    # --- mechanism-based inactivation (time-dependent, not direct, inhibition) ---
    Motif("tdi_mdp", "c1ccc2c(c1)OCO2", "tdi", _MAO_2025, "",
          "Methylenedioxyphenyl / benzodioxole."),
    Motif("tdi_terminal_alkyne", "[CH1X2]#[CX2]", "tdi", _MIRZAEI_2021, "",
          "Terminal only; internal alkynes are not alerts."),
    Motif("tdi_furan", "o1cccc1", "tdi", _DANG_2017, ""),
    Motif("tdi_thiophene", "s1cccc1", "tdi", _DANG_2017, ""),
    Motif("tdi_hydrazine", "[NX3;!$(N=*)][NX3;!$(N=*)]", "tdi", _MAO_2025, "",
          "Also matches hydrazides, which Mao 2025 includes."),
    Motif("tdi_alkylamine_tertiary",
          "[NX3;H0;!$(NC=O);!$(NS(=O)=O);!$(N[a]);!$(N=*);!$(NC=[N,S,O]);!$(N[N,O]);!$(N#*)]",
          "tdi", _MIRZAEI_2021, "",
          "Tertiary alkylamines. Mao 2025 lists alkylamine-containing compounds among "
          "the causes of quasi-irreversible CYP3A4 inactivation; the specific reactive "
          "intermediate is not asserted here."),
)


@dataclass(frozen=True)
class Descriptor:
    """One physicochemical feature with its literature justification."""

    name: str
    func: Callable[[Chem.Mol], float]
    source: str
    note: str = ""


_BASIC_AMINE_PAT = Chem.MolFromSmarts(
    "[NX3;H0,H1,H2;!$(NC=O);!$(NS(=O)=O);!$(N[a]);!$(N=*);!$(NC=[N,S,O]);!$(N[N,O]);!$(N#*)]"
)
_ACID_PAT = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")


def _count(pattern: Chem.Mol) -> Callable[[Chem.Mol], float]:
    def counter(mol: Chem.Mol) -> float:
        return float(len(mol.GetSubstructMatches(pattern)))

    return counter


def _aromatic_fraction(mol: Chem.Mol) -> float:
    """Aromatic heavy atoms / heavy atoms — a planarity proxy for CYP1A2."""
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0:
        return 0.0
    return sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / heavy


_SANSEN_2007 = (
    "Sansen et al. 2007, J Biol Chem — CYP1A2 has a compact, closed active-site "
    "cavity highly adapted for positioning relatively large, planar substrates."
)
_WESTER_2004 = (
    "Wester et al. 2004, J Biol Chem — CYP2C9 Arg-108 interacts directly with "
    "flurbiprofen's carboxylate, accounting for efficiency on anionic NSAIDs."
)
_YANO_2004 = (
    "Yano et al. 2004, J Biol Chem — CYP3A4 has a relatively large substrate-binding "
    "cavity consistent with oxidizing bulky substrates (cyclosporin, statins, taxanes)."
)
_KIANI_2019 = (
    "Kiani et al. 2019, Computation — across CYP1A2/2C9/2C19/2D6/3A4, molecular "
    "weight, H-bond acceptors/donors and lipophilicity are the important "
    "physicochemical descriptors of CYP inhibitors."
)
_STEYN_2020 = (
    "Steyn et al. 2020, Mol Pharm — over 397 clinical substrate-inhibitor pairs, "
    "basic and neutral compounds show a higher magnitude of P450 DDIs, with "
    "lipophilicity trends apparent."
)

_ISOFORM_FEATURES: tuple[Descriptor, ...] = (
    Descriptor("iso_basic_amine_count", _count(_BASIC_AMINE_PAT), _PAINE_2003,
               "Characteristic of CYP2D6 ligands. NOT a mechanism-strength proxy: "
               "Hutzler et al. 2003 found the Asp301 charge-pair may not be the "
               "primary binding interaction, so weak performance on the 2D6 head "
               "is a predicted result, not a bug."),
    Descriptor("iso_acidic_group_count", _count(_ACID_PAT), _WESTER_2004,
               "CYP2C9's anionic-ligand preference via Arg108."),
    Descriptor("iso_aromatic_rings", lambda m: float(Descriptors.NumAromaticRings(m)),
               _SANSEN_2007),
    Descriptor("iso_aromatic_frac", _aromatic_fraction, _SANSEN_2007,
               "Planarity proxy for CYP1A2's narrow planar cavity."),
    Descriptor("iso_fraction_csp3", lambda m: float(Descriptors.FractionCSP3(m)),
               _CSP3_EXTRAPOLATION, "Three-dimensionality; the complement of planarity."),
    Descriptor("iso_mw", lambda m: float(Descriptors.MolWt(m)), _YANO_2004,
               "Size, for CYP3A4's large cavity."),
    Descriptor("iso_clogp", lambda m: float(Descriptors.MolLogP(m)), _STEYN_2020),
    Descriptor("iso_tpsa", lambda m: float(Descriptors.TPSA(m)), _TPSA_EXTRAPOLATION),
    Descriptor("iso_hba", lambda m: float(Descriptors.NumHAcceptors(m)), _KIANI_2019),
    Descriptor("iso_hbd", lambda m: float(Descriptors.NumHDonors(m)), _KIANI_2019),
)


def _parse(smi: str | None) -> Chem.Mol | None:
    """Return a usable molecule, or None.

    Chem.MolFromSmiles("") returns a VALID zero-atom molecule rather than None,
    which would silently produce an all-zeros feature row indistinguishable from
    a real molecule with no matches. Both guards are required.
    """
    if smi is None or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


class _SmartsCountBlock(Representation):
    """Counts each motif in one block. Invalid SMILES give an all-NaN row;
    a valid molecule with no matches gives a zeros row."""

    block: ClassVar[str] = ""
    _desc: ClassVar[str] = "SMARTS motifs"

    @property
    def motifs(self) -> list[Motif]:
        return [m for m in _MOTIFS if m.block == self.block]

    @property
    def feature_names(self) -> list[str]:
        return [m.name for m in self.motifs]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        patterns = [Chem.MolFromSmarts(m.smarts) for m in self.motifs]
        out = np.full((len(smiles), len(patterns)), np.nan, dtype=np.float64)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc=self._desc, unit="mol", leave=True)):
            mol = _parse(smi)
            if mol is None:
                continue
            out[i] = [float(len(mol.GetSubstructMatches(p))) for p in patterns]
        return out


class HemeCoordinationMotifs(_SmartsCountBlock):
    """Substructure counts for nitrogens able to ligate the heme iron, plus a
    steric-accessibility split. The 2D control for round-1 T6."""

    name: ClassVar[str] = f"cyp_heme_{TABLE_VERSION}"
    block: ClassVar[str] = "heme"
    _desc: ClassVar[str] = "heme motifs"


class MechanismBasedAlerts(_SmartsCountBlock):
    """Structural alerts for time-dependent inactivation.

    Emitted as a separate registered block because these concern TDI rather than
    direct inhibition, so little signal is expected on the four scored heads.
    Round-1 T1 adds TDI heads and will want them.
    """

    name: ClassVar[str] = f"cyp_tdi_{TABLE_VERSION}"
    block: ClassVar[str] = "tdi"
    _desc: ClassVar[str] = "TDI alerts"


class IsoformPreferenceDescriptors(Representation):
    """Physicochemical descriptors chosen for the four scored isoforms.

    Each is justified from a cited medicinal-chemistry source in
    `_ISOFORM_FEATURES`; see the spec's section 12.2. Per RULES.md section 5,
    none is encoded from memory.
    """

    name: ClassVar[str] = f"cyp_isoform_{TABLE_VERSION}"

    @property
    def feature_names(self) -> list[str]:
        return [d.name for d in _ISOFORM_FEATURES]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        out = np.full((len(smiles), len(_ISOFORM_FEATURES)), np.nan, dtype=np.float64)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc="isoform descriptors", unit="mol", leave=True)):
            mol = _parse(smi)
            if mol is None:
                continue
            out[i] = [d.func(mol) for d in _ISOFORM_FEATURES]
        return out


# RDKit's ErG fingerprint length at default parameters. Verified, not assumed.
ERG_N_FEATURES = 315


class ErGFingerprint(Representation):
    """RDKit extended reduced graph (ErG) pharmacophore fingerprint.

    The off-the-shelf comparator for the hand-written CYP motifs. If the
    hand-written blocks do not beat this, they are not worth maintaining.

    Parameters are left at RDKit defaults deliberately: tuning them against
    validation scores would make them fitted parameters.
    """

    name: ClassVar[str] = "erg"

    @property
    def feature_names(self) -> list[str]:
        return [f"ErG_{i}" for i in range(ERG_N_FEATURES)]

    def transform(self, smiles: pl.Series) -> np.ndarray:
        out = np.full((len(smiles), ERG_N_FEATURES), np.nan, dtype=np.float64)
        for i, smi in enumerate(tqdm(smiles.to_list(), desc="ErG fingerprint", unit="mol", leave=True)):
            mol = _parse(smi)
            if mol is None:
                continue
            out[i] = np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float64)
        return out
