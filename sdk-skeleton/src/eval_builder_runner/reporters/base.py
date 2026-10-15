"""Reporter interface for emitting an `EvalReport` (console, JSON, HTML, ...)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from eval_builder_runner.models.results import EvalReport


class Reporter(Protocol):
    """Publishes a finished report somewhere."""

    def report(self, report: EvalReport) -> None:
        """Emit `report`."""
        ...
