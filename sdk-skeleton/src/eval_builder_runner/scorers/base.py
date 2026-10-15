"""Scorer interface. Implement this protocol to add a new metric."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from eval_builder_runner.models.results import Score
    from eval_builder_runner.models.spec import Case


@runtime_checkable
class Scorer(Protocol):
    """Scores a task output against a case."""

    name: str

    def score(self, case: Case, output: Any) -> Score:
        """Return a `Score` for `output` on `case`."""
        ...
