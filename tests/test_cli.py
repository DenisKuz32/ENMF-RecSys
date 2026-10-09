"""Командный интерфейс: точка входа `enmf` и разбор аргументов."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import tomllib

from enmf.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent


def test_console_script_points_to_cli_main() -> None:
    """`pip install -e .` создаёт команду `enmf`, ведущую в enmf.cli:main."""
    with open(ROOT / "pyproject.toml", "rb") as handle:
        scripts = tomllib.load(handle)["project"]["scripts"]
    module_name, function_name = scripts["enmf"].split(":")
    assert callable(getattr(importlib.import_module(module_name), function_name))


@pytest.mark.parametrize("command", ["train", "evaluate", "recommend", "compare-splits"])
def test_all_commands_are_registered(command: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as raised:
        parser.parse_args([command, "--help"])
    assert raised.value.code == 0
