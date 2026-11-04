"""GPU detection shared by the boosted-tree models."""

from functools import lru_cache


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True if torch is installed and sees a CUDA device; False on any failure."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device: str = "auto") -> str:
    """Map "auto" to "cuda" or "cpu"; any explicit value is returned unchanged."""
    if device == "auto":
        return "cuda" if cuda_available() else "cpu"
    return device
