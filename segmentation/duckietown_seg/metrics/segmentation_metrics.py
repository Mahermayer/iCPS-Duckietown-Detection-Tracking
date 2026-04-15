from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from duckietown_seg.losses.segmentation_losses import one_hot_encode, soft_dice_score


def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=1)


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    valid = (targets >= 0) & (targets < num_classes)
    indices = num_classes * targets[valid] + predictions[valid]
    matrix = torch.bincount(indices, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes).to(torch.float64)


def per_class_stats(conf_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    true_positive = torch.diag(conf_matrix)
    false_positive = conf_matrix.sum(dim=0) - true_positive
    false_negative = conf_matrix.sum(dim=1) - true_positive
    return true_positive, false_positive, false_negative


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    lane_class_ids: Sequence[int],
    boundary_tolerance: int = 2,
    ignore_index: int | None = None,
) -> dict[str, float | list[float]]:
    probabilities = torch.softmax(logits, dim=1)
    predictions = logits_to_predictions(logits)
    conf_matrix = confusion_matrix(predictions, targets, num_classes)
    true_positive, false_positive, false_negative = per_class_stats(conf_matrix)
    epsilon = 1e-6

    iou = true_positive / (true_positive + false_positive + false_negative + epsilon)
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    f1 = 2.0 * precision * recall / (precision + recall + epsilon)

    valid_mask = (targets >= 0) & (targets < num_classes)
    if ignore_index is not None:
        valid_mask &= targets != ignore_index

    targets_one_hot = one_hot_encode(targets, num_classes, ignore_index=ignore_index)
    dice = soft_dice_score(probabilities, targets_one_hot, valid_mask=valid_mask).mean()

    lane_predictions = torch.zeros_like(predictions, dtype=torch.bool)
    lane_targets = torch.zeros_like(targets, dtype=torch.bool)
    for class_id in lane_class_ids:
        lane_predictions |= predictions == class_id
        lane_targets |= targets == class_id
    lane_predictions &= valid_mask
    lane_targets &= valid_mask
    lane_pixel_f1 = binary_f1(lane_predictions, lane_targets)
    boundary_f1_value = boundary_f1(lane_predictions, lane_targets, tolerance=boundary_tolerance)

    return {
        "miou": float(iou.mean().cpu()),
        "per_class_iou": [float(x.cpu()) for x in iou],
        "dice_score": float(dice.cpu()),
        "per_class_precision": [float(x.cpu()) for x in precision],
        "per_class_recall": [float(x.cpu()) for x in recall],
        "per_class_f1": [float(x.cpu()) for x in f1],
        "lane_pixel_f1": float(lane_pixel_f1.cpu()),
        "boundary_f1": float(boundary_f1_value.cpu()),
    }


def binary_f1(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    predictions = predictions.bool()
    targets = targets.bool()
    tp = (predictions & targets).sum().to(torch.float32)
    fp = (predictions & ~targets).sum().to(torch.float32)
    fn = (~predictions & targets).sum().to(torch.float32)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    return 2.0 * precision * recall / (precision + recall + epsilon)


def _extract_boundary(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float().unsqueeze(1)
    pooled = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    boundary = (pooled - eroded) > 0
    return boundary.squeeze(1)


def boundary_f1(predictions: torch.Tensor, targets: torch.Tensor, tolerance: int = 2) -> torch.Tensor:
    pred_boundary = _extract_boundary(predictions)
    target_boundary = _extract_boundary(targets)
    pred_float = pred_boundary.float().unsqueeze(1)
    target_float = target_boundary.float().unsqueeze(1)
    kernel_size = 2 * tolerance + 1
    pred_dilated = F.max_pool2d(pred_float, kernel_size=kernel_size, stride=1, padding=tolerance).squeeze(1) > 0
    target_dilated = F.max_pool2d(target_float, kernel_size=kernel_size, stride=1, padding=tolerance).squeeze(1) > 0
    pred_hits = (pred_boundary & target_dilated).sum().to(torch.float32)
    target_hits = (target_boundary & pred_dilated).sum().to(torch.float32)
    pred_total = pred_boundary.sum().to(torch.float32)
    target_total = target_boundary.sum().to(torch.float32)
    precision = pred_hits / (pred_total + 1e-6)
    recall = target_hits / (target_total + 1e-6)
    return 2.0 * precision * recall / (precision + recall + 1e-6)


def average_metric_dicts(metric_dicts: list[dict[str, float | list[float]]]) -> dict[str, float | list[float]]:
    if not metric_dicts:
        raise ValueError("No metric dictionaries provided.")
    aggregated: dict[str, float | list[float]] = {}
    keys = metric_dicts[0].keys()
    for key in keys:
        value = metric_dicts[0][key]
        if isinstance(value, list):
            length = len(value)
            sums = [0.0] * length
            for metrics in metric_dicts:
                for index, item in enumerate(metrics[key]):  # type: ignore[index]
                    sums[index] += float(item)
            aggregated[key] = [item / len(metric_dicts) for item in sums]
        else:
            aggregated[key] = sum(float(metrics[key]) for metrics in metric_dicts) / len(metric_dicts)
    return aggregated
