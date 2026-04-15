#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from duckietown_seg.data.dataset import DuckietownLaneSegmentationDataset, build_train_val_datasets
from duckietown_seg.engine.evaluator import evaluate_model
from duckietown_seg.losses.segmentation_losses import estimate_class_weights, estimate_focal_alpha
from duckietown_seg.models import create_model
from duckietown_seg.utils.config import load_experiment_config
from duckietown_seg.utils.logging import configure_logging
from duckietown_seg.utils.seed import seed_everything


LOGGER = configure_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Duckietown segmentation checkpoint.")
    parser.add_argument("--model-config", required=True, help="Path to a model YAML config.")
    parser.add_argument("--train-config", required=True, help="Path to a training YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--dataset-root", default=None, help="Optional override for the dataset root directory.")
    parser.add_argument("--split", choices=["val", "test", "full"], default="val", help="Dataset split to evaluate.")
    return parser.parse_args()


def build_dataset_for_split(config: dict, split: str, dataset_root_override: str | None = None) -> DuckietownLaneSegmentationDataset:
    dataset_cfg = config["dataset"]
    dataset_root = Path(dataset_root_override) if dataset_root_override else PROJECT_ROOT / str(dataset_cfg["root"])
    if split == "test":
        image_dirname = dataset_cfg.get("test_image_dirname")
        mask_dirname = dataset_cfg.get("test_mask_dirname")
        if not image_dirname or not mask_dirname:
            raise ValueError("test split requested but test_image_dirname/test_mask_dirname are not defined in config.")
    else:
        image_dirname = dataset_cfg["image_dirname"]
        mask_dirname = dataset_cfg["mask_dirname"]
    return DuckietownLaneSegmentationDataset(
        dataset_root=dataset_root,
        image_dirname=str(image_dirname),
        mask_dirname=str(mask_dirname),
        image_size=config["input"]["image_size"],
        num_classes=int(config["model"]["num_classes"]),
    )


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.model_config, args.train_config)
    seed_everything(int(config.get("seed", 42)))

    dataset = build_dataset_for_split(config, args.split, args.dataset_root)
    dataset_cfg = config["dataset"]
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    if args.split == "val":
        _, eval_dataset = build_train_val_datasets(
            dataset,
            val_fraction=float(dataset_cfg["val_fraction"]),
            split_seed=int(dataset_cfg["split_seed"]),
        )
    else:
        eval_dataset = dataset

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(**config["model"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.split == "val":
        train_dataset, _ = build_train_val_datasets(
            dataset,
            val_fraction=float(dataset_cfg["val_fraction"]),
            split_seed=int(dataset_cfg["split_seed"]),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = eval_loader

    class_weights = estimate_class_weights(train_loader, int(config["model"]["num_classes"]), device)
    focal_alpha = estimate_focal_alpha(train_loader, int(config["model"]["num_classes"]), device)
    metrics = evaluate_model(
        model=model,
        dataloader=eval_loader,
        device=device,
        num_classes=int(config["model"]["num_classes"]),
        lane_class_ids=list(config["metrics"]["lane_class_ids"]),
        boundary_tolerance=int(config["metrics"].get("boundary_tolerance", 2)),
        loss_name=str(config["train"]["loss_name"]),
        class_weights=class_weights,
        focal_alpha=focal_alpha,
        focal_gamma=float(config["train"].get("focal_gamma", 2.0)),
        dice_weight=float(config["train"].get("dice_weight", 0.5)),
        ce_weight=float(config["train"].get("ce_weight", 0.5)),
        focal_weight=float(config["train"].get("focal_weight", 0.5)),
        progress_description=f"Evaluate {args.split}",
    )
    LOGGER.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()
