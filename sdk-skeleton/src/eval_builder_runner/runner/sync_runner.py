"""Synchronous runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval_builder_runner.models.results import EvalReport
    from eval_builder_runner.models.spec import EvalSpec
    from eval_builder_runner.reporters.base import Reporter


class EvalRunner:
    """Executes an `EvalSpec` and returns an `EvalReport`."""

    def __init__(self, reporters: list[Reporter] | None = None) -> None:
        raise NotImplementedError

    def run(self, spec: EvalSpec) -> EvalReport:
        """Run every case in `spec`, score outputs, and aggregate a report."""
        raise NotImplementedError
