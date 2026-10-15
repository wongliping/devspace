# Single entry points so your Jenkins pipeline can just call `make ci`.
.PHONY: sync fmt lint typecheck test build ci

sync:
	uv sync

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run pyright

test:
	uv run pytest

build:
	uv run python setup.py sdist bdist_wheel

ci: sync lint typecheck test build
