from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML at {path} must define a mapping at the top level.")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_experiment_config(model_config_path: str | Path, train_config_path: str | Path) -> dict[str, Any]:
    train_config = load_yaml(train_config_path)
    model_config = load_yaml(model_config_path)
    config = deep_merge(train_config, model_config)
    config["model_config_path"] = str(model_config_path)
    config["train_config_path"] = str(train_config_path)
    return config

