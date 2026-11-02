"""Locate the conserved heme-thiolate Cys in each isoform's UniProt sequence.

The canonical P450 Cys-heme ligand signature is [FW]-[TS]-x-[RK]-x-[FY]-x-[CTS]
-x-C-[AG]-x-[LIVMFA] (PROSITE PS00098: P450_CYTOCHROME_CHOL). We scan a relaxed
version and report context; CYP3A4's known answer (Cys442 in S-R-P-R-N-C-I-G)
is the calibration point.
"""

import re

import requests

# PROSITE PS00098, relaxed: F/W, x, x, x, x, x, x, C
PAT = re.compile(r"[FW].{6}C")

ISOFORMS = {"CYP1A2": "P05177", "CYP2C9": "P11712", "CYP2D6": "P10635", "CYP3A4": "P08684"}

for gene, acc in ISOFORMS.items():
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=60)
    seq = "".join(line.strip() for line in r.text.splitlines() if not line.startswith(">"))
    hits = [m.end() for m in PAT.finditer(seq)]
    # keep only cysteines in the C-terminal half where the motif lives
    hits = [h for h in hits if h > len(seq) * 0.6]
    ctx = [(h, seq[h - 10 : h + 4]) for h in hits]
    print(f"{gene} ({acc}): len={len(seq)} Cys candidates={ctx}")
