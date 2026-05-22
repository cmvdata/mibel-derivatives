.PHONY: help install lint fmt typecheck test cov bundle clean

help:
	@echo "Targets:"
	@echo "  install   - uv sync (dev + notebooks extras)"
	@echo "  lint      - ruff check"
	@echo "  fmt       - ruff format"
	@echo "  typecheck - mypy on src/"
	@echo "  test      - pytest"
	@echo "  cov       - pytest with coverage"
	@echo "  bundle    - regenerate code_bundle.txt"
	@echo "  clean     - remove caches and build artifacts"

install:
	uv sync --extra dev --extra notebooks

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

typecheck:
	uv run mypy

test:
	uv run pytest

cov:
	uv run pytest --cov=src/mibel_derivatives --cov-report=term-missing

bundle:
	uv run python scripts/_dump_code_bundle.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
