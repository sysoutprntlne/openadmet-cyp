"""Alias for `train-model --model unimol`."""

from models.train_model import main as _main


def main() -> None:
    """Entry point for the train-unimol CLI."""
    import sys

    _main(sys.argv[1:], model="unimol")


if __name__ == "__main__":
    main()
