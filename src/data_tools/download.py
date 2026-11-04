"""Download CYP challenge dataset splits, plus candidate metabolism datasets, from HuggingFace."""

from pathlib import Path

import polars as pl

_HF_BASE = "hf://datasets/openadmet/cyp-challenge-train-test/"
_HF_TRAIN = "cyp-challenge-TRAIN_inhibition.csv"
_HF_TEST = "cyp-challenge-TEST-BLINDED.csv"
_HF_TDI_TRAIN = "cyp-challenge-TRAIN_TDI.csv"
_HF_EMAX_TRAIN = "cyp-challenge-TRAIN_Emax.csv"
_HF_SINGLE_CONC_TRAIN = "cyp-challenge-single-concentration-TRAIN.csv"



# Metabolism datasets, candidates for pretraining. Written as raw CSVs to
# data/metabolism/{name}.csv. Compare their chemical overlap with the challenge
# before choosing one; never merge them into the scored columns.
#   name -> (hf path, what it is)
_METABOLISM_DIR = "metabolism"
_METABOLISM_DATASETS = {
    "expansionrx": (
        "hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train.csv",
        "ExpansionRx challenge train: HLM/MLM CLint plus LogD, KSOL, permeability, PPB (CC-BY-4.0)",
    ),
    "tdc_hlm": (
        "hf://datasets/scikit-fingerprints/TDC_hlm/tdc_hlm.csv",
        "TDC human liver microsomal stability, binary (6,013)",
    ),
    "tdc_rlm": (
        "hf://datasets/scikit-fingerprints/TDC_rlm/tdc_rlm.csv",
        "TDC rat liver microsomal stability, binary (5,590)",
    ),
    "tdc_clearance_microsome_az": (
        "hf://datasets/scikit-fingerprints/TDC_clearance_microsome_az/tdc_clearance_microsome_az.csv",
        "TDC AstraZeneca microsomal clearance, regression (1,102)",
    ),
    "tdc_half_life_obach": (
        "hf://datasets/scikit-fingerprints/TDC_half_life_obach/tdc_half_life_obach.csv",
        "TDC Obach drug half-life, regression (667)",
    ),
    "tdc_cyp2c9_substrate": (
        "hf://datasets/scikit-fingerprints/TDC_cyp2c9_substrate_carbonmangels/tdc_cyp2c9_substrate_carbonmangels.csv",
        "TDC CYP2C9 substrate, binary (about 670)",
    ),
    "tdc_cyp2d6_substrate": (
        "hf://datasets/scikit-fingerprints/TDC_cyp2d6_substrate_carbonmangels/tdc_cyp2d6_substrate_carbonmangels.csv",
        "TDC CYP2D6 substrate, binary (about 670)",
    ),
    "tdc_cyp3a4_substrate": (
        "hf://datasets/scikit-fingerprints/TDC_cyp3a4_substrate_carbonmangels/tdc_cyp3a4_substrate_carbonmangels.csv",
        "TDC CYP3A4 substrate, binary (670)",
    ),
}


def _download_split(hf_path: str, local_path: Path) -> None:
    if local_path.exists():
        print(f"Skipping (already exists): {local_path}")
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {hf_path} ...")
    df = pl.read_csv(hf_path)
    df.write_csv(local_path)
    print(f"Saved {len(df)} rows -> {local_path}")
