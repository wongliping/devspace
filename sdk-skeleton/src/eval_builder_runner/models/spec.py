"""Input-side data models: cases, datasets, and the built evaluation spec."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from eval_builder_runner.config import RunnerConfig
from eval_builder_runner.scorers.base import Scorer


class Case(BaseModel):
    """A single evaluation case."""

    model_config = ConfigDict(frozen=True)

    id: str
    input: Any
    expected: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """A named collection of cases."""

    model_config = ConfigDict(frozen=True)

    name: str
    cases: list[Case]


# The system under test: takes a case, returns an output to be scored.
Task = Callable[[Case], Any]


@dataclass(frozen=True, slots=True)
class EvalSpec:
    """Immutable output of `EvalBuilder.build()` and input to the runners."""

    name: str
    dataset: Dataset
    task: Task
    scorers: Sequence[Scorer] = field(default_factory=tuple)
    config: RunnerConfig = field(default_factory=RunnerConfig)
