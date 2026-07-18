.PHONY: install uv-sync test lint format typecheck all clean

install:
	pip install -e ".[dev]"

uv-sync:
	uv sync --extra dev

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	ty check src

all: lint typecheck test

clean:
	rm -rf data/ .pytest_cache/ .ruff_cache/ .ty_cache/ .coverage
