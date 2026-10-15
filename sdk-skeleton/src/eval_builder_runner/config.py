"""Configuration objects. Immutable, explicit, and free of global state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Execution settings shared by the sync and async runners."""

    max_concurrency: int = 4
    timeout_s: float | None = None
    max_retries: int = 0
    fail_fast: bool = False
