"""Custom chemprop losses and train-time label smoothing.

Lives in its own importable module, not in models.chemprop_ext, because that one runs as
`__main__` and chemprop would pickle the loss class into checkpoints as `__main__.Huber`.
Loading a checkpoint trained with --loss-function huber needs this module importable.
"""

import torch
import torch.nn.functional as F
from chemprop.nn import LossFunctionRegistry
from chemprop.nn.metrics import MAE, MSE

_CONFIG: dict[str, float | None] = {"huber_delta": 1.0, "label_noise": 0.0, "label_shrink": 0.0}


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
