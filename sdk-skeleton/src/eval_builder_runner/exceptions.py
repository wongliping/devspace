"""Exception hierarchy. Every error raised by the SDK derives from `EvalError`."""

from __future__ import annotations


class EvalError(Exception):
    """Base class for all errors raised by eval_builder_runner."""


class ConfigurationError(EvalError):
    """Invalid or missing configuration."""


class SpecValidationError(EvalError):
    """The evaluation spec produced by the builder is invalid or incomplete."""


class RunnerError(EvalError):
    """Failure in the runner itself (not in the task or scorer under evaluation)."""


class TaskExecutionError(RunnerError):
    """The task under evaluation raised while processing a case."""


class ScorerError(RunnerError):
    """A scorer raised while scoring a case."""


class EvalTimeoutError(RunnerError):
    """A case or run exceeded its configured timeout."""
