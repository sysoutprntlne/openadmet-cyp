"""Structure generation: co-folding and docking (T5).

Local smoke-test harness for protein-ligand co-folding. Production engines
(IntelliFold-2, Protenix) run on the SCC; this package lets us validate the
input contract, the output schema, and T6's heme-coordination metrics on a
single known complex (ketoconazole x CYP3A4) using whatever engine runs on
the local GPU.

Layout:
    receptors.py  -- isoform registry, UniProt sequence fetch, heme-Cys finder
    ligands.py    -- ligand registry (SMILES via PubChem PUG REST, cached)
    engines.py    -- engine backends (intellifold, boltz2): build input, run, parse
    cofold.py     -- CLI orchestrator: one ligand x one isoform, end to end
    metrics.py    -- heme-coordination geometry from a predicted (or crystal) complex
    visualize.py  -- interactive HTML + static PNG of the complex
"""
