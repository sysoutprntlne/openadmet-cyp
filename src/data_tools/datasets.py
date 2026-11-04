"""Registry of the datasets this project knows how to obtain.

Each Dataset is a recipe: `fetch` pulls the raw data from its source, `derive` (optional) builds the model-ready
table from what fetch wrote. Both take the data directory and write under it. To add a dataset, append one entry in
`_build()`; the data CLI picks it up with no other change.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

Step = Callable[[Path], None]


@dataclass(frozen=True)
class Dataset:
    name: str
    group: str
    description: str
    fetched: tuple[str, ...]  # files fetch writes, relative to the data directory
    fetch: Step
    derived: tuple[str, ...] = ()  # files derive writes, relative to the data directory
    derive: Step | None = None
    directories: tuple[str, ...] = ()  # extra directories fetch fills, removed whole by `data remove`

    @property
    def files(self) -> tuple[str, ...]:
        return self.fetched + self.derived

    @property
    def primary(self) -> str:
        """The file models consume: the last derived file, else the last fetched one."""
        return self.files[-1]


def _hf_csv(url: str, relative: str) -> Step:
    def fetch(data_dir: Path) -> None:
        from data_tools.download import _download_split

        _download_split(url, data_dir / relative)

    return fetch


def _fetch_pubchem(data_dir: Path) -> None:
    from data_tools.pubchem import download_raw

    download_raw(data_dir / "pubchem_aid1851_raw.csv")


def _derive_pubchem(data_dir: Path) -> None:
    from data_tools.pubchem import reshape, write_csv

    fieldnames, rows = reshape(data_dir / "pubchem_aid1851_raw.csv")
    write_csv(fieldnames, rows, data_dir / "pubchem_aid1851.csv")


def _fetch_p450_family(data_dir: Path) -> None:
    import data_tools.family_datasets as fd

    targets = fd.collect_family_targets(fd._DEFAULT_PFAM_ID, fd._DEFAULT_BINDINGDB_CUTOFF_NM, fd._DEFAULT_MAX_WORKERS)
    fd.write_datasets(targets, data_dir / "families", fd._DEFAULT_TOP_N, fd._DEFAULT_MIN_RECORDS)


def _derive_p450_family(data_dir: Path) -> None:
    from data_tools.family_heads import write_heads

    write_heads(data_dir / "families" / "manifest.csv", data_dir / "family_heads.csv")


def _build() -> dict[str, Dataset]:
    from data_tools import download as dl

    datasets = [
        Dataset("train", "challenge", "Direct-inhibition train split (pIC50 per isoform)",
                ("train.csv",), _hf_csv(dl._HF_BASE + dl._HF_TRAIN, "train.csv")),
        Dataset("test", "challenge", "Blinded test split",
                ("test.csv",), _hf_csv(dl._HF_BASE + dl._HF_TEST, "test.csv")),
        Dataset("tdi_train", "challenge", "Time-dependent inhibition (TDI) train split",
                ("tdi_train.csv",), _hf_csv(dl._HF_BASE + dl._HF_TDI_TRAIN, "tdi_train.csv")),
        Dataset("emax_train", "challenge", "Emax train split",
                ("emax_train.csv",), _hf_csv(dl._HF_BASE + dl._HF_EMAX_TRAIN, "emax_train.csv")),
        Dataset("single_concentration_train", "challenge", "Single-concentration train split",
                ("single_concentration_train.csv",),
                _hf_csv(dl._HF_BASE + dl._HF_SINGLE_CONC_TRAIN, "single_concentration_train.csv")),
        Dataset("pubchem_aid1851", "auxiliary", "PubChem AID 1851 (Veith CYP qHTS panel) reshaped into per-isoform heads",
                ("pubchem_aid1851_raw.csv",), _fetch_pubchem,
                ("pubchem_aid1851.csv",), _derive_pubchem),
        Dataset("p450_family", "auxiliary", "ChEMBL + BindingDB affinity across the P450 family, pivoted into per-protein heads",
                ("families/manifest.csv",), _fetch_p450_family,
                ("family_heads.csv",), _derive_p450_family, directories=("families",)),
    ]
    for name, (url, description) in dl._METABOLISM_DATASETS.items():
        relative = f"{dl._METABOLISM_DIR}/{name}.csv"
        datasets.append(Dataset(name, "metabolism", description, (relative,), _hf_csv(url, relative)))
    return {d.name: d for d in datasets}


def registry() -> dict[str, Dataset]:
    """Every known dataset by name."""
    return _build()
