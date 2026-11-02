"""`chemprop train` with extra training options chemprop's own CLI lacks.

Runs chemprop's CLI in-process after patching in:

  --loss-function huber   Huber loss (quadratic inside --huber-delta, linear outside)
  --huber-delta D         Huber threshold in label units (pIC50), default 1.0
  --label-noise S         train-time Gaussian jitter on regression targets, std S
  --label-shrink A        train-time pull of each target toward its batch mean per head:
                          y' = (1 - A) * y + A * mean(y)
  --lr-warmup-shape S     linear | cosine | exp | none     (chemprop default: linear)
  --lr-decay S            exp | cosine | linear | constant (chemprop default: exp)
  --freeze-epochs N       keep the message-passing encoder frozen for the first N epochs
  --mpn-dropout P         dropout in message passing only (default: --dropout)
  --ffn-dropout P         dropout in the prediction head only (default: --dropout)

Everything else is chemprop's own, e.g. --warmup-epochs, --init-lr, --max-lr,
--final-lr, --depth, --message-hidden-dim, --multi-hot-atom-featurizer-mode.
Noise and shrinkage touch only the training loss; validation and test scores
use the raw labels. Usage: python -m models.chemprop_ext train --data-path ...
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F
from chemprop.nn import LossFunctionRegistry
from chemprop.nn.metrics import MAE, MSE
from torch import nn

_CONFIG: dict[str, float | None] = {"huber_delta": 1.0, "label_noise": 0.0, "label_shrink": 0.0}


def _parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--label-noise", type=float, default=0.0)
    p.add_argument("--label-shrink", type=float, default=0.0)
    p.add_argument("--lr-warmup-shape", choices=["linear", "cosine", "exp", "none"], default=None)
    p.add_argument("--lr-decay", choices=["exp", "cosine", "linear", "constant"], default=None)
    p.add_argument("--freeze-epochs", type=int, default=0)
    p.add_argument("--mpn-dropout", type=float, default=None)
    p.add_argument("--ffn-dropout", type=float, default=None)
    return p.parse_known_args(argv)


def smooth_targets(targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Noise and shrink targets; only while gradients are on (i.e. in training, not validation)."""
    noise, shrink = _CONFIG["label_noise"], _CONFIG["label_shrink"]
    if not (noise or shrink) or not torch.is_grad_enabled():
        return targets
    if shrink:
        m = mask.to(targets.dtype)
        mean = (targets * m).sum(0, keepdim=True) / m.sum(0, keepdim=True).clamp(min=1)
        targets = (1 - shrink) * targets + shrink * mean
    if noise:
        targets = targets + noise * torch.randn_like(targets)
    return targets


@LossFunctionRegistry.register("huber")
class Huber(MSE):
    """Huber loss. Module-level so chemprop can pickle it into checkpoints (loading one needs models.chemprop_ext importable)."""

    def _calc_unreduced_loss(self, preds, targets, *args):
        return F.huber_loss(preds, targets, reduction="none", delta=float(_CONFIG["huber_delta"]))


def _patch_smoothing() -> None:
    for cls in (MSE, MAE, Huber):
        original = cls._calc_unreduced_loss

        def wrapped(self, preds, targets, *args, _orig=original):
            mask = args[0] if args else torch.ones_like(targets, dtype=torch.bool)
            return _orig(self, preds, smooth_targets(targets, mask), *args)

        cls._calc_unreduced_loss = wrapped


def lr_at(step: int, warm: int, total: int, init: float, peak: float, final: float, shape: str, decay: str) -> float:
    """Learning rate at an optimizer step: a warmup of `shape` up to peak, then `decay` down to final."""
    if step < warm:
        t = step / max(warm, 1)
        if shape == "cosine":
            return init + (peak - init) * (1 - math.cos(math.pi * t)) / 2
        if shape == "exp":
            return init * (peak / init) ** t
        return init + (peak - init) * t
    u = (step - warm) / max(total - warm, 1)
    u = min(max(u, 0.0), 1.0)
    if decay == "cosine":
        return final + (peak - final) * (1 + math.cos(math.pi * u)) / 2
    if decay == "linear":
        return peak + (final - peak) * u
    if decay == "constant":
        return peak
    return peak * (final / peak) ** u


def _patch_schedule(shape: str | None, decay: str | None) -> None:
    from chemprop.models.model import MPNN

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.max_lr)
        spe = self.trainer.num_training_batches
        warm = 0 if shape == "none" else self.warmup_epochs * spe
        total = self.trainer.max_epochs * spe
        init = self.max_lr if shape == "none" else self.init_lr
        peak, final = self.max_lr, self.final_lr

        def factor(step: int) -> float:
            return lr_at(step, warm, total, init, peak, final, shape or "linear", decay or "exp") / peak

        sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    MPNN.configure_optimizers = configure_optimizers


def _patch_freeze(n_epochs: int) -> None:
    from chemprop.models.model import MPNN

    def on_train_epoch_start(self):
        frozen = self.current_epoch < n_epochs
        for param in self.message_passing.parameters():
            param.requires_grad = not frozen

    MPNN.on_train_epoch_start = on_train_epoch_start


def _patch_dropout(mpn_p: float | None, ffn_p: float | None) -> None:
    import chemprop.cli.train as train_cli

    build_model = train_cli.build_model

    def patched(*args, **kwargs):
        model = build_model(*args, **kwargs)
        for part, p in ((model.message_passing, mpn_p), (model.predictor, ffn_p)):
            if p is not None:
                for module in part.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = p
        return model

    train_cli.build_model = patched


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    ext, rest = _parse(argv)
    _CONFIG.update(huber_delta=ext.huber_delta, label_noise=ext.label_noise, label_shrink=ext.label_shrink)
    if ext.label_noise or ext.label_shrink:
        _patch_smoothing()
    if ext.lr_warmup_shape or ext.lr_decay:
        _patch_schedule(ext.lr_warmup_shape, ext.lr_decay)
    if ext.freeze_epochs:
        _patch_freeze(ext.freeze_epochs)
    if ext.mpn_dropout is not None or ext.ffn_dropout is not None:
        _patch_dropout(ext.mpn_dropout, ext.ffn_dropout)

    from chemprop.cli.main import main as chemprop_main

    sys.argv = ["chemprop", *rest]
    chemprop_main()


if __name__ == "__main__":
    main()
