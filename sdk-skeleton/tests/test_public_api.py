"""The public surface must stay importable and consistent."""

import eval_builder_runner as ebr


def test_all_names_resolve() -> None:
    for name in ebr.__all__:
        assert hasattr(ebr, name), name


def test_version_is_string() -> None:
    assert isinstance(ebr.__version__, str)
