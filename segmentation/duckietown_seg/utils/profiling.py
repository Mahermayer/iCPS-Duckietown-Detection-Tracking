from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), temp_path)
        return temp_path.stat().st_size / 1e6
    finally:
        temp_path.unlink(missing_ok=True)


def approximate_flops(model: nn.Module, input_size: tuple[int, int, int]) -> float | None:
    try:
        from ptflops import get_model_complexity_info  # type: ignore
    except Exception:
        return None
    macs, _ = get_model_complexity_info(
        model,
        input_size,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    return float(macs) * 2.0


@torch.inference_mode()
def inference_fps(
    model: nn.Module,
    device: torch.device,
    input_size: tuple[int, int, int],
    runs: int = 30,
    warmup: int = 5,
) -> float:
    model.eval()
    sample = torch.randn(1, *input_size, device=device)
    for _ in range(warmup):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return runs / max(elapsed, 1e-8)


def profile_model(
    model: nn.Module,
    device: torch.device,
    input_size: tuple[int, int, int],
    runs: int = 30,
    warmup: int = 5,
) -> dict[str, float | None]:
    return {
        "parameter_count": parameter_count(model),
        "model_size_mb": model_size_mb(model),
        "approximate_flops": approximate_flops(model, input_size),
        "inference_fps": inference_fps(model, device, input_size, runs=runs, warmup=warmup),
    }
