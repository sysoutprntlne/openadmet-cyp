"""CLI for the project's datasets: list them, fetch them (deriving the model-ready table), and alias them."""

import argparse
import shutil
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from data_tools import dataset_alias
from data_tools.datasets import Dataset, registry

console = Console()


def _size(path: Path) -> str:
    n = float(path.stat().st_size)
    for unit in ("B", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return ""


def _rows(path: Path) -> str:
    newline = b"\n"
    with path.open("rb") as f:
        lines = sum(chunk.count(newline) for chunk in iter(lambda: f.read(1 << 20), b""))
    return f"{max(lines - 1, 0):,}"


def _status(dataset: Dataset, data_dir: Path) -> str:
    present = [(data_dir / f).exists() for f in dataset.files]
    if all(present):
        return "ready"
    return "partial" if any(present) else "missing"


def _list(args: argparse.Namespace, datasets: dict[str, Dataset]) -> None:
    aliases: dict[str, list[str]] = {}
    for alias, name in dataset_alias.load_aliases().items():
        aliases.setdefault(name, []).append(alias)

    table = Table(box=box.SIMPLE_HEAD)
    for column in ("dataset", "alias", "group", "status", "rows", "size"):
        table.add_column(column, justify="right" if column in ("rows", "size") else "left", no_wrap=True)
    table.add_column("description", overflow="ellipsis", no_wrap=True, max_width=45)
    for dataset in datasets.values():
        status = _status(dataset, args.data_dir)
        primary = args.data_dir / dataset.primary
        ready = primary.is_file()
        table.add_row(
            dataset.name,
            ", ".join(sorted(aliases.get(dataset.name, []))),
            dataset.group,
            f"[green]{status}[/]" if status == "ready" else f"[yellow]{status}[/]" if status == "partial" else f"[dim]{status}[/]",
            _rows(primary) if ready else "",
            _size(primary) if ready else "",
            dataset.description,
        )
    console.print(table)


def _fetch(args: argparse.Namespace, datasets: dict[str, Dataset]) -> None:
    known = set(datasets)
    if "all" in args.names:
        names = list(datasets)
    else:
        names = list(dict.fromkeys(dataset_alias.resolve(n, known) for n in args.names))
    for name in names:
        dataset = datasets[name]
        print(f"== {name}", file=sys.stderr)
        fetched = [args.data_dir / f for f in dataset.fetched]
        if args.force:
            for path in fetched:
                path.unlink(missing_ok=True)
        if all(path.exists() for path in fetched):
            print("Skipping fetch (already exists); pass --force to refetch", file=sys.stderr)
        else:
            dataset.fetch(args.data_dir)
        if dataset.derive is not None:
            dataset.derive(args.data_dir)


def _remove(args: argparse.Namespace, datasets: dict[str, Dataset]) -> None:
    known = set(datasets)
    names = list(dict.fromkeys(dataset_alias.resolve(n, known) for n in args.names))
    targets: list[Path] = []
    for name in names:
        dataset = datasets[name]
        for relative in (*dataset.files, *dataset.directories):
            path = args.data_dir / relative
            if path.exists() and path not in targets:
                targets.append(path)
    targets = [t for t in targets if not any(other in t.parents for other in targets)]
    if not targets:
        print("Nothing to remove.", file=sys.stderr)
        return
    for path in targets:
        print(f"{path}{'/' if path.is_dir() else ''}", file=sys.stderr)
    if not args.yes and input(f"Delete {len(targets)} path(s)? [y/N] ").strip().lower() != "y":
        print("Aborted.", file=sys.stderr)
        return
    for path in targets:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    print(f"Removed {len(targets)} path(s); `data fetch` brings them back.", file=sys.stderr)


def _alias(args: argparse.Namespace, datasets: dict[str, Dataset]) -> None:
    known = set(datasets)
    if args.remove:
        dataset_alias.remove_alias(args.remove)
        return
    if args.dataset is None or args.alias is None:
        raise ValueError("give a dataset and an alias, or --remove ALIAS")
    dataset_alias.set_alias(dataset_alias.resolve(args.dataset, known), args.alias, known)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the data CLI."""
    parser = argparse.ArgumentParser(description="Manage the project's datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory holding the datasets (default: data)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="List every known dataset with its status").set_defaults(run=_list)

    fetch = commands.add_parser("fetch", help="Fetch datasets by name or alias, then derive their model-ready tables")
    fetch.add_argument("names", nargs="+", metavar="NAME", help="Dataset names or aliases, or 'all'")
    fetch.add_argument("--force", action="store_true", help="Refetch even if the raw files exist")
    fetch.set_defaults(run=_fetch)

    remove = commands.add_parser("remove", help="Delete datasets' files (raw and derived); aliases are kept")
    remove.add_argument("names", nargs="+", metavar="NAME", help="Dataset names or aliases")
    remove.add_argument("--yes", "-y", action="store_true", help="Delete without asking")
    remove.set_defaults(run=_remove)

    alias = commands.add_parser("alias", help="Give a dataset a short alias, or remove one")
    alias.add_argument("dataset", nargs="?", help="Dataset name (or existing alias)")
    alias.add_argument("alias", nargs="?", help="New alias")
    alias.add_argument("--remove", metavar="ALIAS", help="Delete this alias")
    alias.set_defaults(run=_alias)

    args = parser.parse_args(argv)
    try:
        args.run(args, registry())
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


def download_all() -> None:
    """Entry point for download-data: `data fetch all`, passing any extra arguments (such as --force) through."""
    main(["fetch", "all", *sys.argv[1:]])


if __name__ == "__main__":
    main()
