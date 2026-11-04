"""Short aliases for dataset names, stored in data-alias.csv.

An alias stands in for a registered dataset name wherever the data CLI takes one, so `data fetch cyp_train` works
like `data fetch train`. Each alias names one dataset; a dataset may have several aliases.
"""

import csv
import re
from pathlib import Path

ALIAS_FILE = Path("data-alias.csv")
_VALID = re.compile(r"[\w.+-]+")


def load_aliases(file: Path = ALIAS_FILE) -> dict[str, str]:
    """Return {alias: dataset name}."""
    if not file.is_file():
        return {}
    with file.open(newline="") as f:
        return {row["alias"]: row["dataset"] for row in csv.DictReader(f)}


def _save(aliases: dict[str, str], file: Path) -> None:
    with file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "dataset"])
        writer.writerows(aliases.items())


def set_alias(dataset: str, alias: str, known: set[str], file: Path = ALIAS_FILE) -> None:
    """Point `alias` at `dataset`. Raises ValueError if the dataset is unknown or the alias is invalid or taken."""
    if dataset not in known:
        raise ValueError(f"unknown dataset {dataset!r}")
    if not _VALID.fullmatch(alias):
        raise ValueError(f"invalid alias {alias!r}: use letters, digits, and _ . + - only")
    if alias in known:
        raise ValueError(f"alias {alias!r} is the name of a dataset")
    aliases = load_aliases(file)
    if aliases.get(alias, dataset) != dataset:
        raise ValueError(f"alias {alias!r} is already used for {aliases[alias]}")
    aliases[alias] = dataset
    _save(aliases, file)


def remove_alias(alias: str, file: Path = ALIAS_FILE) -> None:
    """Delete `alias`. Raises ValueError if it does not exist."""
    aliases = load_aliases(file)
    if alias not in aliases:
        raise ValueError(f"no such alias {alias!r}")
    del aliases[alias]
    _save(aliases, file)


def resolve(name: str, known: set[str], file: Path = ALIAS_FILE) -> str:
    """The dataset a name or alias stands for. Raises ValueError listing the choices if it is neither."""
    if name in known:
        return name
    aliases = load_aliases(file)
    if name in aliases and aliases[name] in known:
        return aliases[name]
    raise ValueError(f"unknown dataset or alias {name!r}; choose from: {', '.join(sorted(known | set(aliases)))}")
