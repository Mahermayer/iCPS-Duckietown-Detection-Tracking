from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from duckietown_seg.losses.segmentation_losses import build_loss
from duckietown_seg.metrics.segmentation_metrics import average_metric_dicts, compute_segmentation_metrics


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
    num_classes: int,
    lane_class_ids: list[int],
    boundary_tolerance: int,
    loss_name: str,
    class_weights: torch.Tensor | None,
    focal_alpha: torch.Tensor | None,
    focal_gamma: float,
    dice_weight: float,
    ce_weight: float,
    focal_weight: float,
    ignore_index: int | None = None,
    progress_description: str = "Evaluating",
) -> dict[str, float | list[float]]:
    model.eval()
    metric_rows: list[dict[str, float | list[float]]] = []
    running_loss = 0.0
    batches = 0

    for images, masks in tqdm(dataloader, desc=progress_description, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss, _ = build_loss(
            loss_name,
            logits,
            masks,
            class_weights=class_weights,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            focal_weight=focal_weight,
            ignore_index=ignore_index,
        )
        running_loss += float(loss.detach().cpu())
        metric_rows.append(
            compute_segmentation_metrics(
                logits,
                masks,
                num_classes=num_classes,
                lane_class_ids=lane_class_ids,
                boundary_tolerance=boundary_tolerance,
                ignore_index=ignore_index,
            )
        )
        batches += 1

    metrics = average_metric_dicts(metric_rows)
    metrics["loss"] = running_loss / max(batches, 1)
    return metrics
