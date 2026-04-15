from __future__ import annotations

import logging
from typing import Any


LOGGER = logging.getLogger("duckietown_seg")


def configure_logging() -> logging.Logger:
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False
    return LOGGER


class OptionalWandbLogger:
    def __init__(
        self,
        enabled: bool,
        project: str,
        config: dict[str, Any],
        entity: str | None = None,
        name: str | None = None,
        group: str | None = None,
    ) -> None:
        self.enabled = False
        self.run = None
        if not enabled:
            return
        try:
            import wandb  # type: ignore
        except Exception:
            LOGGER.warning("wandb is not available. Continuing without wandb logging.")
            return
        self.run = wandb.init(project=project, entity=entity, config=config, name=name, group=group)
        self.enabled = True

    def log(self, metrics: dict[str, Any]) -> None:
        if self.enabled and self.run is not None:
            self.run.log(metrics)

    def summary_update(self, metrics: dict[str, Any]) -> None:
        if self.enabled and self.run is not None:
            for key, value in metrics.items():
                self.run.summary[key] = value

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()
