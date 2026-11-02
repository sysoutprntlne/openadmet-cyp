"""Uni-Mol backend: fine-tune jointly on the ADMET data plus any auxiliary datasets.

One model per (learning rate, seed) with every head at once. Missing labels are
masked in the loss (unimol_tools' multilabel_regression uses a NaN-masked MAE),
so molecules measured by only one source still train that head. Auxiliary-only
molecules always land in train, so no validation molecule is seen in training.

  train-model --model unimol --datasets pubchem:CYP1A2 bindingdb:CYP1A2,CYP2C8
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from data_tools.merge_datasets import parse_dataset_specs
from models.training.base import Backend
from models.training.common import prepare_joint_split, write_val_outputs


def set_seed(seed: int) -> None:
    """Seed the main random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_cuda() -> None:
    """Fail early, with the GPU and torch details, if this torch build has no kernels for the device."""
    try:
        torch.zeros(4, dtype=torch.long, device="cuda").eq(0).sum().item()  # forces a real kernel launch
    except RuntimeError as exc:
        major, minor = torch.cuda.get_device_capability()
        raise RuntimeError(
            f"{torch.cuda.get_device_name()} (sm_{major}{minor}) has no kernels in this torch build "
            f"(torch {torch.__version__}, CUDA {torch.version.cuda}, built for {torch.cuda.get_arch_list()}). "
            "Request a supported GPU (-l gpu_c=...), install a torch build that includes this architecture, "
            f"or pass --cpu.\n{exc}"
        ) from exc


class UnimolBackend(Backend):
    results_dir = Path("results/unimol")
    requires_datasets = True
    defaults = {"epochs": 30}

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--learning-rate", nargs="+", type=float, default=[2e-4], help="One or more learning rates; each is swept like a seed (default: 2e-4)"
        )
        parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    def run(self, args: argparse.Namespace, extra_args: list[str]) -> int:
        from unimol_tools import MolPredict, MolTrain  # heavy and optional: only needed for this backend

        try:
            specs = parse_dataset_specs(args.datasets or [], with_challenge=False)
        except ValueError as exc:
            args.parser.error(str(exc))
        set_seed(args.seed)

        train_df, val_df, target_columns, eval_map = prepare_joint_split(specs, args, args.parser)
        datasets = args.datasets or []
        use_cuda = torch.cuda.is_available() and not args.cpu
        if use_cuda:
            check_cuda()

        base_dir = args.output_dir or Path("results/unimol") / args.run_label
        output_dir = base_dir / f"lr_{args.learning_rate}_seed_{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Datasets: {', '.join(datasets) or 'ADMET only'}")
        print(f"Heads: {len(target_columns)} ({len(eval_map)} scored)")
        print(f"Training molecules: {len(train_df)}")
        print(f"Validation molecules: {len(val_df)}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Seed: {args.seed}")
        print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

        trainer = MolTrain(
            task="multilabel_regression",  # NaN-masked loss
            data_type="molecule",
            model_name="unimolv1",
            smiles_col="SMILES",
            target_cols=target_columns,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            save_path=str(output_dir),
            use_cuda=use_cuda,
            kfold=1,
            split_seed=args.seed,
            seed=args.seed,
            metrics="none",
        )
        # No official validation rows are passed to the trainer.
        trainer.fit(train_df[["SMILES", *target_columns]])

        predictor = MolPredict(load_model=str(output_dir))
        predictions = np.asarray(predictor.predict(val_df[["SMILES"]])).reshape(len(val_df), len(target_columns))
        if not np.isfinite(predictions).all():
            raise RuntimeError("Non-finite predictions returned by Uni-Mol.")

        write_val_outputs(
            output_dir,
            f"unimol_{args.run_label}_lr{args.learning_rate}_seed{args.seed}",
            val_df,
            train_df,
            dict(zip(target_columns, predictions.T)),
            eval_map,
            {"learning_rate": args.learning_rate, "seed": args.seed},
        )
        return 0
