# Contributing

```bash
make sync        # create .venv and install dev dependencies (uv)
make fmt         # auto-format and auto-fix lint
make ci          # what the pipeline runs: lint, typecheck, test, build
```

Rules of thumb:
- Public API = names in `eval_builder_runner.__all__`. Anything under a `_`-prefixed module is private.
- Changing public API requires a CHANGELOG entry and a version bump per SemVer.
- Every new public function/class needs a docstring and type hints (enforced by ruff + pyright).
