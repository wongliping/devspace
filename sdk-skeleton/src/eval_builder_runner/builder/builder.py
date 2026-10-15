"""Fluent builder that assembles and validates an `EvalSpec`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval_builder_runner.config import RunnerConfig
    from eval_builder_runner.models.spec import Dataset, EvalSpec, Task
    from eval_builder_runner.scorers.base import Scorer


class EvalBuilder:
    """Collects dataset, task, scorers, and config, then builds an `EvalSpec`."""

    def __init__(self, name: str) -> None:
        raise NotImplementedError

    def with_dataset(self, dataset: Dataset) -> EvalBuilder:
        """Set the dataset to evaluate on."""
        raise NotImplementedError

    def with_task(self, task: Task) -> EvalBuilder:
        """Set the system under test."""
        raise NotImplementedError

    def add_scorer(self, scorer: Scorer) -> EvalBuilder:
        """Add a scorer. May be called multiple times."""
        raise NotImplementedError

    def with_config(self, config: RunnerConfig) -> EvalBuilder:
        """Override default runner configuration."""
        raise NotImplementedError

    def build(self) -> EvalSpec:
        """Validate and return an immutable `EvalSpec`.

        Raises:
            SpecValidationError: if required parts are missing or inconsistent.
        """
        raise NotImplementedError
