"""CLI shim: `python main.py` → launch the Gradio RAG compare app."""

from __future__ import annotations

from rag_learn.logging_config import setup_logging

# Configure logging before any application modules are imported so that
# module-level logger.info/debug/warning calls are handled consistently.
setup_logging()

from rag_learn.app import launch


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
