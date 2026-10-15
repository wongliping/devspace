"""Placeholder tests: delete each one as the real implementation lands."""

import pytest

from eval_builder_runner import AsyncEvalRunner, EvalBuilder, EvalRunner


def test_builder_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        EvalBuilder("x")


def test_runner_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        EvalRunner()


def test_async_runner_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        AsyncEvalRunner()
