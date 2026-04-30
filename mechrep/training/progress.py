"""Small training utilities for progress bars and early stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, TypeVar


T = TypeVar("T")


def iter_progress(iterable: Iterable[T], *, enabled: bool = True, **kwargs) -> Iterable[T]:
    """Wrap an iterable in tqdm when available and enabled."""

    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def normalize_monitor_metric(metric_name: str, split: str = "valid") -> str:
    """Map config names such as ``valid_auprc`` to metric dict keys."""

    prefix = f"{split}_"
    if metric_name.startswith(prefix):
        return metric_name[len(prefix) :]
    return metric_name


def infer_metric_mode(metric_name: str) -> str:
    """Infer whether higher or lower values are better for a metric."""

    normalized = normalize_monitor_metric(metric_name)
    if "loss" in normalized or normalized in {"mr", "mse", "mae", "error"}:
        return "min"
    return "max"


@dataclass
class EarlyStopper:
    """Track validation improvement and decide when to stop."""

    metric_name: str | None
    patience: int | None
    min_delta: float = 0.0
    mode: str | None = None

    def __post_init__(self) -> None:
        if self.patience is not None:
            self.patience = int(self.patience)
        self.mode = self.mode or infer_metric_mode(self.metric_name or "")
        if self.mode not in {"max", "min"}:
            raise ValueError(f"early stopping mode must be 'max' or 'min', got {self.mode!r}")
        self.best_value = -math.inf if self.mode == "max" else math.inf
        self.best_epoch = 0
        self.bad_epochs = 0

    @property
    def enabled(self) -> bool:
        return bool(self.metric_name) and self.patience is not None and self.patience > 0

    def observe(self, value: float, *, epoch: int) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)`` after observing a metric value."""

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True, False

        self.bad_epochs += 1
        return False, self.enabled and self.bad_epochs >= int(self.patience)
