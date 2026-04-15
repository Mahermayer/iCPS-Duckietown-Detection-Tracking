from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from duckietown_seg.engine.evaluator import evaluate_model
from duckietown_seg.losses.segmentation_losses import build_loss, estimate_class_weights, estimate_focal_alpha
from duckietown_seg.metrics.segmentation_metrics import compute_segmentation_metrics
from duckietown_seg.utils.io import CsvMetricLogger, ensure_dir, save_checkpoint, save_json
from duckietown_seg.utils.logging import OptionalWandbLogger, configure_logging
from duckietown_seg.utils.profiling import profile_model


LOGGER = configure_logging()


@dataclass
class TrainerArtifacts:
    output_dir: Path
    best_checkpoint_path: Path
    final_checkpoint_path: Path
    metrics_csv_path: Path
    summary_json_path: Path


class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        output_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = ensure_dir(output_dir)
        self.device = device
        self.num_classes = int(config["model"]["num_classes"])
        self.train_cfg = config["train"]
        self.dataset_cfg = config["dataset"]
        self.metric_cfg = config["metrics"]
        self.ignore_index = self.dataset_cfg.get("ignore_index")
        self.use_amp = bool(self.train_cfg.get("amp", False)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.class_weights = (
            estimate_class_weights(train_loader, self.num_classes, device, ignore_index=self.ignore_index)
            if self.train_cfg.get("use_class_weights", True)
            else None
        )
        self.focal_alpha = estimate_focal_alpha(train_loader, self.num_classes, device, ignore_index=self.ignore_index)
        self.optimizer = self._build_optimizer()
        self.artifacts = TrainerArtifacts(
            output_dir=self.output_dir,
            best_checkpoint_path=self.output_dir / "best_checkpoint.pth",
            final_checkpoint_path=self.output_dir / "final_checkpoint.pth",
            metrics_csv_path=self.output_dir / "metrics.csv",
            summary_json_path=self.output_dir / "summary.json",
        )
        self.csv_logger = CsvMetricLogger(
            self.artifacts.metrics_csv_path,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_miou",
                "val_dice_score",
                "val_lane_pixel_f1",
                "val_boundary_f1",
            ],
        )
        logging_cfg = self.config.get("logging", {})
        model_group = self.config["model"].get("model_name", "model")
        variant_name = self.output_dir.parent.name if self.output_dir.parent != self.output_dir else self.output_dir.name
        config_name = self.output_dir.name
        default_group = variant_name or str(model_group)
        default_run_name = f"{variant_name}-{config_name}" if variant_name else str(config_name)
        configured_run_name = logging_cfg.get("run_name")
        configured_group = logging_cfg.get("group")
        self.wandb = OptionalWandbLogger(
            enabled=bool(logging_cfg.get("use_wandb", False)),
            project=str(logging_cfg.get("project", "iCPS Segmentation Duckietown")),
            entity=logging_cfg.get("entity"),
            config=config,
            name=str(configured_run_name) if configured_run_name else default_run_name,
            group=str(configured_group) if configured_group else default_group,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_name = str(self.train_cfg.get("optimizer", "adamw")).lower()
        learning_rate = float(self.train_cfg["learning_rate"])
        weight_decay = float(self.train_cfg.get("weight_decay", 0.0))
        if optimizer_name == "adamw":
            return AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name == "adam":
            return Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")

    def train(self) -> dict[str, Any]:
        best_miou = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        epochs = int(self.train_cfg["epochs"])
        patience = int(self.train_cfg.get("early_stopping_patience", 0))
        min_delta = float(self.train_cfg.get("early_stopping_min_delta", 0.0))

        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self._train_one_epoch(epoch)
            val_metrics = evaluate_model(
                self.model,
                self.val_loader,
                self.device,
                num_classes=self.num_classes,
                lane_class_ids=list(self.metric_cfg["lane_class_ids"]),
                boundary_tolerance=int(self.metric_cfg.get("boundary_tolerance", 2)),
                loss_name=str(self.train_cfg["loss_name"]),
                class_weights=self.class_weights,
                focal_alpha=self.focal_alpha,
                focal_gamma=float(self.train_cfg.get("focal_gamma", 2.0)),
                dice_weight=float(self.train_cfg.get("dice_weight", 0.5)),
                ce_weight=float(self.train_cfg.get("ce_weight", 0.5)),
                focal_weight=float(self.train_cfg.get("focal_weight", 0.5)),
                ignore_index=self.ignore_index,
                progress_description=f"Validation {epoch}",
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(val_metrics["loss"]),
                "val_miou": float(val_metrics["miou"]),
                "val_dice_score": float(val_metrics["dice_score"]),
                "val_lane_pixel_f1": float(val_metrics["lane_pixel_f1"]),
                "val_boundary_f1": float(val_metrics["boundary_f1"]),
            }
            self.csv_logger.log(row)
            self.wandb.log({"epoch": epoch, **row})

            LOGGER.info(
                "Epoch %03d | train_loss=%.4f | val_loss=%.4f | val_mIoU=%.4f | val_dice=%.4f | lane_F1=%.4f | boundary_F1=%.4f",
                epoch,
                train_loss,
                float(val_metrics["loss"]),
                float(val_metrics["miou"]),
                float(val_metrics["dice_score"]),
                float(val_metrics["lane_pixel_f1"]),
                float(val_metrics["boundary_f1"]),
            )

            if float(val_metrics["miou"]) > best_miou + min_delta:
                best_miou = float(val_metrics["miou"])
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    self.artifacts.best_checkpoint_path,
                    {
                        "epoch": epoch,
                        "config": self.config,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_metrics,
                    },
                )
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                LOGGER.info("Early stopping triggered at epoch %d.", epoch)
                break

        final_metrics = evaluate_model(
            self.model,
            self.val_loader,
            self.device,
            num_classes=self.num_classes,
            lane_class_ids=list(self.metric_cfg["lane_class_ids"]),
            boundary_tolerance=int(self.metric_cfg.get("boundary_tolerance", 2)),
            loss_name=str(self.train_cfg["loss_name"]),
            class_weights=self.class_weights,
            focal_alpha=self.focal_alpha,
            focal_gamma=float(self.train_cfg.get("focal_gamma", 2.0)),
            dice_weight=float(self.train_cfg.get("dice_weight", 0.5)),
            ce_weight=float(self.train_cfg.get("ce_weight", 0.5)),
            focal_weight=float(self.train_cfg.get("focal_weight", 0.5)),
            ignore_index=self.ignore_index,
            progress_description="Final evaluation",
        )

        save_checkpoint(
            self.artifacts.final_checkpoint_path,
            {
                "epoch": best_epoch,
                "config": self.config,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_metrics": final_metrics,
            },
        )

        summary: dict[str, Any] = {
            "best_epoch": best_epoch,
            "best_val_miou": best_miou,
            "final_val_metrics": final_metrics,
            "train_config_path": self.config.get("train_config_path"),
            "model_config_path": self.config.get("model_config_path"),
            "output_dir": str(self.output_dir),
        }

        if self.config.get("profiling", {}).get("enabled", False):
            profile = profile_model(
                self.model,
                self.device,
                (
                    int(self.config["model"]["in_channels"]),
                    int(self.config["input"]["image_size"][0]),
                    int(self.config["input"]["image_size"][1]),
                ),
                runs=int(self.config["profiling"].get("fps_runs", 30)),
                warmup=int(self.config["profiling"].get("fps_warmup", 5)),
            )
            summary["profiling"] = profile

        self.wandb.summary_update(summary)
        self.wandb.finish()
        save_json(self.artifacts.summary_json_path, summary)
        return summary

    def _train_one_epoch(self, epoch: int) -> tuple[float, dict[str, float | list[float]]]:
        self.model.train()
        running_loss = 0.0
        batch_metrics: list[dict[str, float | list[float]]] = []

        for images, masks in tqdm(self.train_loader, desc=f"Train {epoch}", leave=False):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss, _ = build_loss(
                    str(self.train_cfg["loss_name"]),
                    logits,
                    masks,
                    class_weights=self.class_weights,
                    focal_alpha=self.focal_alpha,
                    focal_gamma=float(self.train_cfg.get("focal_gamma", 2.0)),
                    dice_weight=float(self.train_cfg.get("dice_weight", 0.5)),
                    ce_weight=float(self.train_cfg.get("ce_weight", 0.5)),
                    focal_weight=float(self.train_cfg.get("focal_weight", 0.5)),
                    ignore_index=self.ignore_index,
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += float(loss.detach().cpu())
            batch_metrics.append(
                compute_segmentation_metrics(
                    logits.detach(),
                    masks.detach(),
                    num_classes=self.num_classes,
                    lane_class_ids=list(self.metric_cfg["lane_class_ids"]),
                    boundary_tolerance=int(self.metric_cfg.get("boundary_tolerance", 2)),
                    ignore_index=self.ignore_index,
                )
            )

        mean_loss = running_loss / max(len(self.train_loader), 1)
        mean_metrics = {
            "train_miou": sum(float(item["miou"]) for item in batch_metrics) / max(len(batch_metrics), 1),
            "train_dice_score": sum(float(item["dice_score"]) for item in batch_metrics) / max(len(batch_metrics), 1),
        }
        return mean_loss, mean_metrics
