"""eval_builder_runner: build and run evaluations.

Only names in `__all__` are public API and covered by semantic versioning.
"""

from eval_builder_runner._version import __version__
from eval_builder_runner.builder import EvalBuilder
from eval_builder_runner.config import RunnerConfig
from eval_builder_runner.exceptions import (
    ConfigurationError,
    EvalError,
    EvalTimeoutError,
    RunnerError,
    ScorerError,
    SpecValidationError,
    TaskExecutionError,
)
from eval_builder_runner.models import (
    Case,
    CaseResult,
    Dataset,
    EvalReport,
    EvalSpec,
    Score,
    Task,
)
from eval_builder_runner.reporters import Reporter
from eval_builder_runner.runner import AsyncEvalRunner, EvalRunner
from eval_builder_runner.scorers import Scorer

__all__ = [
    "AsyncEvalRunner",
    "Case",
    "CaseResult",
    "ConfigurationError",
    "Dataset",
    "EvalBuilder",
    "EvalError",
    "EvalReport",
    "EvalRunner",
    "EvalSpec",
    "EvalTimeoutError",
    "Reporter",
    "RunnerConfig",
    "RunnerError",
    "Score",
    "Scorer",
    "ScorerError",
    "SpecValidationError",
    "Task",
    "TaskExecutionError",
    "__version__",
]
