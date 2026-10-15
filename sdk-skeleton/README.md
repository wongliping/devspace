# eval-builder-runner

SDK for **building** evaluations (dataset + task + scorers) and **running** them (sync or async).

> Status: skeleton only. All methods raise `NotImplementedError`.

## Layout

| Module | Responsibility |
|---|---|
| `builder` | `EvalBuilder`: fluent construction and validation of an `EvalSpec` |
| `models` | `Case`, `Dataset`, `EvalSpec` (inputs); `Score`, `CaseResult`, `EvalReport` (outputs) |
| `scorers` | `Scorer` protocol; built-in scorers go here |
| `runner` | `EvalRunner` (sync), `AsyncEvalRunner` (async) |
| `reporters` | `Reporter` protocol; console/JSON/etc. go here |
| `config` | `RunnerConfig` |
| `exceptions` | `EvalError` hierarchy |

## Intended usage

```python
from eval_builder_runner import EvalBuilder, EvalRunner

spec = (
    EvalBuilder("my-eval")
    .with_dataset(dataset)
    .with_task(my_task)
    .add_scorer(ExactMatch())
    .build()
)
report = EvalRunner().run(spec)
```

## Development

```bash
make sync     # create the virtualenv and install dev tools (uv)
make fmt      # auto-format and auto-fix lint
make ci       # lint, strict type check, tests, build
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Background: [docs/2026-10-14-python-sdk-best-practices.md](docs/2026-10-14-python-sdk-best-practices.md).

## Building the package

The pipeline builds with the plain setuptools command (metadata lives in `pyproject.toml`; `setup.py` is a thin shim):

```bash
python setup.py sdist bdist_wheel
```

Output: `dist/eval_builder_runner-<version>.tar.gz` and `dist/eval_builder_runner-<version>-py3-none-any.whl`.
The build environment needs `setuptools>=70.1` (it bundles `bdist_wheel`, so no separate `wheel` install is needed).

## Testing and validating the build artifact

Run these after every build, in this order. Layers 2 to 4 must run in a clean virtualenv, not the dev environment, so you test what users will actually install.

### 1. Inspect the contents

```bash
unzip -l dist/*.whl            # package modules and py.typed present; no tests, caches or .venv
tar tzf dist/*.tar.gz          # same check for the sdist
pip install twine check-wheel-contents
twine check dist/*             # metadata and README are valid
check-wheel-contents dist/*.whl
```

### 2. Install in a clean environment

Do this once from the wheel and once from the sdist. The sdist install proves the sdist is complete.

```bash
python -m venv /tmp/verify-wheel && /tmp/verify-wheel/bin/pip install dist/*.whl
python -m venv /tmp/verify-sdist && /tmp/verify-sdist/bin/pip install dist/*.tar.gz
/tmp/verify-wheel/bin/pip check
```

Import from a directory outside the project. With the `src/` layout this proves you import the installed copy and not the source tree:

```bash
cd /tmp && /tmp/verify-wheel/bin/python -c "import eval_builder_runner as e; print(e.__version__, len(e.__all__))"
```

### 3. Run the tests against the installed package

```bash
/tmp/verify-wheel/bin/pip install pytest pytest-asyncio
cd /tmp && /tmp/verify-wheel/bin/pytest -o addopts="" /path/to/project/tests
```

Running from outside the project keeps the source tree off the import path.

### 4. Check consumer-side typing

Confirms `py.typed` shipped, so users' type checkers can use the annotations.

```bash
/tmp/verify-wheel/bin/pip install pyright
cat > /tmp/consumer.py <<'EOF'
from eval_builder_runner import EvalBuilder, RunnerConfig

builder = EvalBuilder("demo")
config: RunnerConfig = RunnerConfig(max_concurrency=2)
EOF
/tmp/verify-wheel/bin/pyright --pythonpath /tmp/verify-wheel/bin/python /tmp/consumer.py
```

### Optional

- Repeat layers 2 and 3 on every supported Python version (3.10 to 3.13), for example as parallel Jenkins stages or with tox or nox.
- Run `pip-audit` in the clean environment to scan runtime dependencies.
- Check that the version in the filenames matches `src/eval_builder_runner/_version.py`.
