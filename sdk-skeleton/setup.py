"""Build shim so `python setup.py sdist bdist_wheel` works.

All metadata (name, version, dependencies, packages) lives in pyproject.toml;
setuptools reads it when `setup()` is called with no arguments.
Requires setuptools>=70.1 in the build environment (it bundles `bdist_wheel`).
"""

from setuptools import setup

setup()
