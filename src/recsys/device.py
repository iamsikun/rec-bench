"""Device selection: GPUs first, CPU only when no accelerator is available."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """``"auto"`` picks the CUDA GPU with the most free memory, else Apple MPS, else CPU."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return torch.device("cuda", max(range(len(free)), key=free.__getitem__))
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
