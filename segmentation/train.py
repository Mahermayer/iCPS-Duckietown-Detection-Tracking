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
from duckietown_seg.engine.trainer import SegmentationTrainer
from duckietown_seg.models import create_model
from duckietown_seg.utils.config import load_experiment_config
from duckietown_seg.utils.io import ensure_dir, save_json
from duckietown_seg.utils.logging import configure_logging
from duckietown_seg.utils.seed import seed_everything


LOGGER = configure_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Duckietown lane segmentation models.")
    parser.add_argument("--model-config", required=True, help="Path to a model YAML config.")
    parser.add_argument("--train-config", required=True, help="Path to a training YAML config.")
    parser.add_argument("--dataset-root", default=None, help="Optional override for the dataset root directory.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the run output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.model_config, args.train_config)
    seed_everything(int(config.get("seed", 42)))

    dataset_cfg = config["dataset"]
    dataset_root = Path(args.dataset_root) if args.dataset_root else PROJECT_ROOT / str(dataset_cfg["root"])
    dataset = DuckietownLaneSegmentationDataset(
        dataset_root=dataset_root,
        image_dirname=str(dataset_cfg["image_dirname"]),
        mask_dirname=str(dataset_cfg["mask_dirname"]),
        image_size=config["input"]["image_size"],
        num_classes=int(config["model"]["num_classes"]),
    )
    train_dataset, val_dataset = build_train_val_datasets(
        dataset,
        val_fraction=float(dataset_cfg["val_fraction"]),
        split_seed=int(dataset_cfg["split_seed"]),
    )

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", True)) and torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = create_model(**config["model"])
    output_root = PROJECT_ROOT / str(config.get("output_root", "outputs"))
    variant_name = Path(args.model_config).stem
    train_name = Path(args.train_config).stem
    output_dir = Path(args.output_dir) if args.output_dir else output_root / variant_name / train_name
    ensure_dir(output_dir)
    save_json(output_dir / "resolved_config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training on %s", device.type)
    trainer = SegmentationTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device,
    )
    summary = trainer.train()
    LOGGER.info("Training complete. Best validation mIoU: %.4f", float(summary["best_val_miou"]))


if __name__ == "__main__":
    main()
