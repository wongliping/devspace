"""Output-side data models: scores, per-case results, and the final report."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Score(BaseModel):
    """The outcome of one scorer on one case."""

    scorer: str
    value: float
    passed: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    """Everything recorded for one case."""

    case_id: str
    output: Any | None = None
    scores: list[Score] = Field(default_factory=list[Score])
    error: str | None = None
    duration_s: float | None = None


class EvalReport(BaseModel):
    """Aggregate result of a full run."""

    spec_name: str
    results: list[CaseResult] = Field(default_factory=list[CaseResult])
    summary: dict[str, float] = Field(default_factory=dict)
