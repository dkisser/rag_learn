"""CLI shim: `python main.py` → launch the Gradio RAG compare app."""

from __future__ import annotations

from rag_learn.app import launch


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
