.PHONY: install uv-sync test lint format typecheck all clean clean-chroma clean-milvus

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

clean-chroma:
	rm -rf data/chroma

clean-milvus:
	rm -rf data/milvus.db

clean: clean-chroma clean-milvus
	rm -rf .pytest_cache/ .ruff_cache/ .ty_cache/ .coverage
