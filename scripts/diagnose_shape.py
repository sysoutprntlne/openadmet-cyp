
"""Diagnose failures in RDKit 3D shape generation."""

from collections import Counter

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem

df = pl.read_csv("data/train.csv")
smiles_list = df["SMILES"].head(100).to_list()

results = Counter()
examples = {}

for smi in smiles_list:
    try:
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            reason = "Invalid SMILES"
        else:
            mol = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.randomSeed = 42

            if AllChem.EmbedMolecule(mol, params) != 0:
                reason = "ETKDG embedding failed"

            elif not AllChem.MMFFHasAllMoleculeParams(mol):
                reason = "MMFF parameters unavailable"

            else:
                status = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)

                if status == 0:
                    reason = "Success"
                elif status == 1:
                    reason = "MMFF did not converge"
                else:
                    reason = f"MMFF status {status}"

        results[reason] += 1

        if reason != "Success":
            examples.setdefault(reason, smi)

    except Exception as e:
        reason = f"Exception: {type(e).__name__}"
        results[reason] += 1
        examples.setdefault(reason, smi)

print("\n--- Failure Breakdown ---")

for reason, count in results.items():
    print(f"{reason}: {count}")

print("\n--- Example Failed Molecules ---")

for reason, smi in examples.items():
    print(f"{reason}: {smi}")
