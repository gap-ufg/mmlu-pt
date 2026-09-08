#!/usr/bin/env python3
"""CLI do anotador LLM-as-a-Judge."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def main() -> int:
    """Carrega e executa o CLI mantido em src/."""
    sys.path.insert(0, str(SOURCE_ROOT))
    from mmlu_pt.llm_as_a_judge.judge_cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
