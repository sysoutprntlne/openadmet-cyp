"""Alias for `train-model --model chemprop` (kept so existing commands and job scripts keep working)."""

from models.train_model import main as _main


def main() -> None:
    """Entry point for the train-chemprop CLI."""
    import sys

    _main(sys.argv[1:], model="chemprop")


if __name__ == "__main__":
    main()
