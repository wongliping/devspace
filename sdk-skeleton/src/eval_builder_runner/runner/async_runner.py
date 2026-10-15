"""Asynchronous runner (concurrency bounded by `RunnerConfig.max_concurrency`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval_builder_runner.models.results import EvalReport
    from eval_builder_runner.models.spec import EvalSpec
    from eval_builder_runner.reporters.base import Reporter


class AsyncEvalRunner:
    """Async counterpart of `EvalRunner`."""

    def __init__(self, reporters: list[Reporter] | None = None) -> None:
        raise NotImplementedError

    async def run(self, spec: EvalSpec) -> EvalReport:
        """Run every case in `spec` concurrently and aggregate a report."""
        raise NotImplementedError
