# Python SDK best practices (applied to eval_builder_runner)

## 1. Packaging
- **Single `pyproject.toml`** (PEP 621), no `setup.py`. Build backend: hatchling.
- **`src/` layout.** Prevents importing the working-directory copy instead of the installed one, catches packaging mistakes, and keeps editable installs honest. Cost: you must install (editable) to run code.
- **Ship `py.typed`** so users' type checkers use your annotations (PEP 561).
- **Single-source the version** (`_version.py`), read by the build backend.
- **Dev tools in `[dependency-groups]`** (PEP 735), not in runtime dependencies. Commit `uv.lock` if you want reproducible CI.
- Keep runtime dependencies minimal and bounded (`pydantic>=2.6,<3`); an SDK's deps become your users' deps.

## 2. Public API design
- **Explicit public surface:** re-export from `__init__.py` and list in `__all__`. Underscore-prefixed modules are private. Only the public surface is covered by SemVer.
- **One exception base class** (`EvalError`) with specific subclasses so callers can catch broadly or narrowly.
- **Immutable configuration** objects (frozen dataclasses); no module-level global state.
- **Protocols for extension points** (`Scorer`, `Reporter`) so users plug in their own without subclassing.
- **Typed everywhere**, strict type checking in CI; Pydantic models for data that crosses the SDK boundary and needs validation or serialization.
- **Sync and async variants** sharing the same spec and result types (`EvalRunner`, `AsyncEvalRunner`).
- **Fluent builder produces an immutable spec**; validation happens in `build()`, not scattered through the runner.

## 3. Quality tooling
- **ruff:** lint and format in one fast tool (replaces flake8, isort, black, pyupgrade, pydocstyle).
- **pyright (strict):** static type checker.
- **pytest** (+ pytest-asyncio, pytest-cov) for tests; test the public API, not internals.
- **pre-commit** hooks for fast local feedback.
- **A Makefile as the single entry point** so any CI (here: Jenkins) just runs `make ci`.

## 4. Release practice
- **SemVer**, a **CHANGELOG** (Keep a Changelog), and deprecation warnings before removals.
- Build with `uv build` (sdist + wheel); publish to your internal index from the pipeline.
- Version bump + changelog entry required for any public API change.

## 5. Not included in the skeleton (decide later)
- HTTP transport layer (retries, timeouts, auth), only needed if the SDK will call a remote evaluation service.
- Docs site (mkdocs / Sphinx), license, security policy, Jenkinsfile.

## Sources
- [src layout vs flat layout, Python Packaging User Guide](https://Packaging.Python.org/discussions/src-layout-vs-flat-layout/)
- [Modern Python toolchain, Trail of Bits](https://trailofbits.com/skills/modern-python/)
- [Modern Python tooling in 2026](https://levelup.gitconnected.com/modern-python-tooling-in-2026-uv-ruff-pyproject-toml-and-a-cleaner-workflow-4d1f7872f7be)
- [Python Project Setup 2026, KDnuggets](https://www.kdnuggets.com/python-project-setup-2026-uv-ruff-ty-polars)
- [Python SDK Design, Gravitee](https://gravitee.io/corpus/gen-1200/python-programming-language/python-sdk-design.html)
- PEP 561 (typed packages), PEP 621 (project metadata), PEP 735 (dependency groups)
