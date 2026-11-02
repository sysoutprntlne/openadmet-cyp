"""Download CYP challenge dataset splits, plus candidate metabolism datasets, from HuggingFace."""

import argparse
from pathlib import Path

import polars as pl

_HF_BASE = "hf://datasets/openadmet/cyp-challenge-train-test/"
_HF_TRAIN = "cyp-challenge-TRAIN_inhibition.csv"
_HF_TEST = "cyp-challenge-TEST-BLINDED.csv"
_HF_TDI_TRAIN = "cyp-challenge-TRAIN_TDI.csv"
_HF_EMAX_TRAIN = "cyp-challenge-TRAIN_Emax.csv"
_HF_SINGLE_CONC_TRAIN = "cyp-challenge-single-concentration-TRAIN.csv"

_TRAIN_FILE = "train.csv"
_TEST_FILE = "test.csv"
_TDI_TRAIN_FILE = "tdi_train.csv"
_EMAX_TRAIN_FILE = "emax_train.csv"
_SINGLE_CONC_TRAIN_FILE = "single_concentration_train.csv"

_DEFAULT_DATA_DIR = Path("data")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CYP challenge dataset splits.")
    parser.add_argument("--no-train", action="store_true", help="Skip the direct-inhibition train split.")
    parser.add_argument("--no-test", action="store_true", help="Skip the blinded test split.")
    parser.add_argument("--no-tdi", action="store_true", help="Skip the time-dependent inhibition (TDI) train split.")
    parser.add_argument("--no-emax", action="store_true", help="Skip the Emax train split.")
    parser.add_argument("--no-single-concentration", action="store_true", help="Skip the single-concentration train split.")
    parser.add_argument(
        "--train-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TRAIN_FILE,
        help="Destination path for the direct-inhibition train CSV.",
    )
    parser.add_argument(
        "--test-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TEST_FILE,
        help="Destination path for the blinded test CSV.",
    )
    parser.add_argument(
        "--tdi-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _TDI_TRAIN_FILE,
        help="Destination path for the TDI train CSV.",
    )
    parser.add_argument(
        "--emax-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _EMAX_TRAIN_FILE,
        help="Destination path for the Emax train CSV.",
    )
    parser.add_argument(
        "--single-concentration-location",
        type=Path,
        default=_DEFAULT_DATA_DIR / _SINGLE_CONC_TRAIN_FILE,
        help="Destination path for the single-concentration train CSV.",
    )
    parser.add_argument(
        "--metabolism",
        nargs="*",
        default=None,
        choices=sorted(_METABOLISM_DATASETS),
        metavar="NAME",
        help="Metabolism datasets to download (default: all). Pass names to pick some: "
        + ", ".join(_METABOLISM_DATASETS),
    )
    parser.add_argument("--no-metabolism", action="store_true", help="Skip the metabolism pretraining candidates.")
    parser.add_argument("--list-metabolism", action="store_true", help="List the metabolism datasets and exit.")
    parser.add_argument(
        "--metabolism-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR / _METABOLISM_DIR,
        help="Destination directory for the metabolism CSVs.",
    )
    args = parser.parse_args()

    if args.list_metabolism:
        for name, (_, description) in _METABOLISM_DATASETS.items():
            print(f"{name:26s} {description}")
        return

    if not args.no_train:
        _download_split(_HF_BASE + _HF_TRAIN, args.train_location)

    if not args.no_test:
        _download_split(_HF_BASE + _HF_TEST, args.test_location)

    if not args.no_tdi:
        _download_split(_HF_BASE + _HF_TDI_TRAIN, args.tdi_location)

    if not args.no_emax:
        _download_split(_HF_BASE + _HF_EMAX_TRAIN, args.emax_location)

    if not args.no_single_concentration:
        _download_split(_HF_BASE + _HF_SINGLE_CONC_TRAIN, args.single_concentration_location)

    if not args.no_metabolism:
        for name in args.metabolism or list(_METABOLISM_DATASETS):
            _download_split(_METABOLISM_DATASETS[name][0], args.metabolism_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
