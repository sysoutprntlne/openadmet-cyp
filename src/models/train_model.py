"""Train any supported model on the ADMET data plus auxiliary datasets.

  train-model --model chemprop  --datasets admet pubchem:CYP1A2 bindingdb:CYP2C8 --loss-function mae
  train-model --model chemeleon --datasets admet pubchem:CYP1A2
  train-model --model unimol    --datasets admet pubchem:CYP1A2 --learning-rate 2e-4

Only the datasets you list are trained on ('admet' is the ADMET labels). Repeat
--datasets for several configs, add --n-seeds N (or --seeds 0 1 2), and for
unimol give several --learning-rate values: every combination trains one model
(configs x learning rates x seeds):

  train-model --model unimol --n-seeds 3 --learning-rate 5e-5 2e-4 5e-4 \\
      --datasets admet \\
      --datasets admet pubchem:CYP1A2 bindingdb:CYP2C8 \\
      --datasets pubchem:CYP1A2 bindingdb:CYP2C8

Datasets, split and head selection are shared (models.training.common); each
backend adds its own flags. `--model chemeleon` is chemprop started from the
CheMeleon foundation weights. Flags a chemprop-based backend doesn't know are
passed through to `chemprop train`. See models.training to add a backend.
"""

import argparse
import copy
import sys

from data_tools.merge_datasets import describe
from models.training import BACKENDS
from models.training.common import add_common_arguments, slug

_ADMET = "admet"  # names the ADMET labels as training data; not a mergeable source


def main(argv: list[str] | None = None, model: str | None = None) -> None:
    """Entry point for the train-model CLI (`model` pins the backend for the legacy aliases)."""
    argv = sys.argv[1:] if argv is None else argv
    if model is None:
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--model", choices=sorted(BACKENDS))
        known, _ = pre.parse_known_args(argv)
        model = known.model
    if model is None:
        if "--list-datasets" in argv:
            print(describe())
            return
        print(f"usage: train-model --model {{{','.join(sorted(BACKENDS))}}} [...]  (add --help after --model)", file=sys.stderr)
        sys.exit(2)

    backend = BACKENDS[model]
    parser = argparse.ArgumentParser(
        prog=f"train-model --model {model}",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(BACKENDS), default=model, help="Which model to train")
    add_common_arguments(parser)
    backend.add_arguments(parser)
    parser.set_defaults(**backend.defaults)
    args, extra = parser.parse_known_args(argv)
    args.parser = parser

    if args.list_datasets:
        print(describe())
        return
    if extra and not backend.passthrough:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if backend.requires_datasets and not args.datasets:
        parser.error("pass --datasets (e.g. --datasets admet, or --datasets admet pubchem:CYP1A2); see --list-datasets")
    seeds = args.seeds or (list(range(args.seed, args.seed + args.n_seeds)) if args.n_seeds else [args.seed])
    lrs = args.learning_rate if hasattr(args, "learning_rate") else [None]  # only backends with the flag sweep it
    runs = [(group, lr, seed) for group in args.datasets or [None] for lr in lrs for seed in seeds]
    failed = []
    for index, (group, lr, seed) in enumerate(runs, 1):
        run = copy.copy(args)
        run.seed = seed
        if lr is not None:
            run.learning_rate = lr
        run.use_admet = True
        run.run_label = _ADMET
        if group is not None:
            run.use_admet = _ADMET in group
            run.datasets = [t for t in group if t != _ADMET]
            run.run_label = slug(group)
        if len(runs) > 1:
            base = args.output_dir or backend.results_dir
            run.output_dir = base / run.run_label / f"seed{seed}"
        backend.validate(run, parser)
        print(f"\n##### run {index}/{len(runs)}: {run.run_label}{f' lr {lr}' if lr is not None else ''} seed {seed}", file=sys.stderr)
        code = backend.run(run, extra)
        if code != 0:
            failed.append(f"{run.run_label} lr {lr} seed {seed}")
    if failed:
        print(f"\n{len(failed)}/{len(runs)} runs failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
