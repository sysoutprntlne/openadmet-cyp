"""Short aliases for foundation-model weights paths, stored in foundation-path-alias.csv.

An alias stands in for a weights path inside `chemprop[...]`, so `chemprop[grid_admet_chemeleon_seed0]` works
wherever `chemprop[/long/path/model_0/best.pt]` does. Paths are stored resolved; each path has one alias and each
alias names one path.
"""

import csv
import re
from pathlib import Path

ALIAS_FILE = Path("foundation-path-alias.csv")
_VALID = re.compile(r"[\w.+-]+")


def _key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def load_aliases(file: Path = ALIAS_FILE) -> dict[str, str]:
    """Return {alias: resolved weights path}."""
    if not file.is_file():
        return {}
    with file.open(newline="") as f:
        return {row["alias"]: row["path"] for row in csv.DictReader(f)}


def _save(aliases: dict[str, str], file: Path) -> None:
    with file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "path"])
        writer.writerows(aliases.items())


def resolve_alias(name: str, file: Path = ALIAS_FILE) -> str:
    """The path an alias stands for, or `name` unchanged if it isn't an alias."""
    return load_aliases(file).get(name, name)


def alias_for(path: str | Path, file: Path = ALIAS_FILE) -> str | None:
    key = _key(path)
    return next((alias for alias, p in load_aliases(file).items() if p == key), None)


def set_alias(path: str | Path, alias: str, file: Path = ALIAS_FILE) -> None:
    """Point `alias` at `path`, replacing any earlier alias of that path. Raises ValueError if the alias is invalid or taken."""
    if not _VALID.fullmatch(alias):
        raise ValueError(f"invalid alias {alias!r}: use letters, digits, and _ . + - only")
    key = _key(path)
    aliases = load_aliases(file)
    if aliases.get(alias, key) != key:
        raise ValueError(f"alias {alias!r} is already used for {aliases[alias]}")
    aliases = {a: p for a, p in aliases.items() if p != key}
    aliases[alias] = key
    _save(aliases, file)


def ensure_aliases(entries: list[tuple[Path, str]], file: Path = ALIAS_FILE) -> dict[str, str]:
    """Give every path without an alias one derived from its suggested name, never clashing with an existing alias.

    entries are (weights path, suggested name). Returns {resolved path: alias} for all of them.
    """
    aliases = load_aliases(file)
    by_path = {p: a for a, p in aliases.items()}
    changed = False
    for path, suggestion in entries:
        key = _key(path)
        if key in by_path:
            continue
        base = re.sub(r"[^\w.+-]+", "_", suggestion).strip("_") or "model"
        alias, n = base, 1
        while alias in aliases:
            n += 1
            alias = f"{base}_{n}"
        aliases[alias] = key
        by_path[key] = alias
        changed = True
    if changed:
        _save(aliases, file)
    return {_key(path): by_path[_key(path)] for path, _ in entries}
