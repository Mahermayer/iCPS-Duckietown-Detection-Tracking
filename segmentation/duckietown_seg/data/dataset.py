from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split

from duckietown_seg.data.transforms import SegmentationResize, image_to_tensor, mask_to_tensor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    mask_path: Path

    @property
    def sample_id(self) -> str:
        return self.image_path.stem


class DuckietownLaneSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for Duckietown lane segmentation with explicit pairing validation."""

    def __init__(
        self,
        dataset_root: str | Path,
        image_dirname: str,
        mask_dirname: str,
        image_size: Sequence[int],
        num_classes: int = 4,
        ignore_index: int | None = None,
        split_file: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / image_dirname
        self.mask_dir = self.dataset_root / mask_dirname
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.resize = SegmentationResize(image_size)
        self.samples = self._build_pairs(Path(split_file) if split_file else None)

    def _iter_files(self, directory: Path, allowed_extensions: Iterable[str]) -> dict[str, Path]:
        if not directory.exists():
            raise FileNotFoundError(f"Missing directory: {directory}")
        files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed_extensions]
        if not files:
            raise FileNotFoundError(f"No files found in {directory}")
        mapping: dict[str, Path] = {}
        for path in sorted(files):
            stem = path.stem
            if stem in mapping:
                raise ValueError(f"Duplicate sample id '{stem}' found in {directory}")
            mapping[stem] = path
        return mapping

    def _build_pairs(self, split_file: Path | None) -> list[SamplePair]:
        if split_file is not None:
            return self._build_pairs_from_manifest(split_file)

        images = self._iter_files(self.image_dir, IMAGE_EXTENSIONS)
        masks = self._iter_files(self.mask_dir, MASK_EXTENSIONS)

        image_ids = set(images)
        mask_ids = set(masks)
        if image_ids != mask_ids:
            missing_masks = sorted(image_ids - mask_ids)
            missing_images = sorted(mask_ids - image_ids)
            raise ValueError(
                "Image-mask pairing mismatch detected. "
                f"Missing masks for ids: {missing_masks[:10]}. "
                f"Missing images for ids: {missing_images[:10]}."
            )

        return [SamplePair(image_path=images[sample_id], mask_path=masks[sample_id]) for sample_id in sorted(image_ids)]

    def _build_pairs_from_manifest(self, split_file: Path) -> list[SamplePair]:
        manifest_path = split_file if split_file.is_absolute() else self.dataset_root / split_file
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing split manifest: {manifest_path}")

        samples: list[SamplePair] = []
        seen_ids: set[str] = set()
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid manifest line {line_number} in {manifest_path}: '{line}'")

                image_rel, mask_rel = parts
                image_path = self.dataset_root / image_rel
                mask_path = self.dataset_root / mask_rel
                if not image_path.exists():
                    raise FileNotFoundError(f"Missing image '{image_path}' referenced by {manifest_path}:{line_number}")
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing mask '{mask_path}' referenced by {manifest_path}:{line_number}")

                sample_id = Path(image_rel).stem
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate sample id '{sample_id}' found in {manifest_path}")
                seen_ids.add(sample_id)
                samples.append(SamplePair(image_path=image_path, mask_path=mask_path))

        if not samples:
            raise FileNotFoundError(f"No samples listed in {manifest_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")
        image, mask = self.resize(image, mask)
        image_tensor = image_to_tensor(image)
        mask_tensor = mask_to_tensor(mask)
        self._validate_mask(mask_tensor, sample.sample_id)
        return image_tensor, mask_tensor

    def _validate_mask(self, mask: torch.Tensor, sample_id: str) -> None:
        valid = (mask >= 0) & (mask < self.num_classes)
        if self.ignore_index is not None:
            valid |= mask == self.ignore_index
        if torch.all(valid):
            return

        invalid_values = sorted(torch.unique(mask[~valid]).tolist())
        if invalid_values:
            raise ValueError(
                f"Mask '{sample_id}' contains invalid class ids. "
                f"Expected [0, {self.num_classes - 1}]"
                + (f" or ignore_index={self.ignore_index}" if self.ignore_index is not None else "")
                + f", got invalid values {invalid_values[:10]}."
            )


def build_train_val_datasets(
    dataset: DuckietownLaneSegmentationDataset,
    val_fraction: float,
    split_seed: int,
) -> tuple[Subset[DuckietownLaneSegmentationDataset], Subset[DuckietownLaneSegmentationDataset]]:
    """Create deterministic train/val subsets."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for the dataset size.")
    generator = torch.Generator().manual_seed(split_seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def build_datasets_from_config(
    dataset_cfg: dict,
    dataset_root: str | Path,
    image_size: Sequence[int],
    num_classes: int,
) -> dict[str, Dataset[tuple[torch.Tensor, torch.Tensor]]]:
    common_kwargs = {
        "dataset_root": dataset_root,
        "image_dirname": str(dataset_cfg["image_dirname"]),
        "mask_dirname": str(dataset_cfg["mask_dirname"]),
        "image_size": image_size,
        "num_classes": num_classes,
        "ignore_index": dataset_cfg.get("ignore_index"),
    }

    train_manifest = dataset_cfg.get("train_split_file")
    val_manifest = dataset_cfg.get("val_split_file")
    test_manifest = dataset_cfg.get("test_split_file")
    all_manifest = dataset_cfg.get("all_split_file")

    if train_manifest or val_manifest or test_manifest or all_manifest:
        datasets: dict[str, Dataset[tuple[torch.Tensor, torch.Tensor]]] = {}
        if train_manifest:
            datasets["train"] = DuckietownLaneSegmentationDataset(
                **common_kwargs,
                split_file=str(train_manifest),
            )
        if val_manifest:
            datasets["val"] = DuckietownLaneSegmentationDataset(
                **common_kwargs,
                split_file=str(val_manifest),
            )
        if test_manifest:
            datasets["test"] = DuckietownLaneSegmentationDataset(
                **common_kwargs,
                split_file=str(test_manifest),
            )
        if all_manifest:
            datasets["full"] = DuckietownLaneSegmentationDataset(
                **common_kwargs,
                split_file=str(all_manifest),
            )
        return datasets

    dataset = DuckietownLaneSegmentationDataset(**common_kwargs)
    train_dataset, val_dataset = build_train_val_datasets(
        dataset,
        val_fraction=float(dataset_cfg["val_fraction"]),
        split_seed=int(dataset_cfg["split_seed"]),
    )
    datasets = {"full": dataset, "train": train_dataset, "val": val_dataset}

    test_image_dirname = dataset_cfg.get("test_image_dirname")
    test_mask_dirname = dataset_cfg.get("test_mask_dirname")
    if test_image_dirname and test_mask_dirname:
        datasets["test"] = DuckietownLaneSegmentationDataset(
            dataset_root=dataset_root,
            image_dirname=str(test_image_dirname),
            mask_dirname=str(test_mask_dirname),
            image_size=image_size,
            num_classes=num_classes,
            ignore_index=dataset_cfg.get("ignore_index"),
        )

    return datasets
