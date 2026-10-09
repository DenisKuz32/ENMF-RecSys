"""Состав репозитория: в сдачу попадают исходники и итоговые артефакты.

Проверяется через `git check-ignore`, поэтому вне git-репозитория тест
пропускается.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _ignored(paths: list[str]) -> set[str]:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return set(result.stdout.split())


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="нужен git-репозиторий",
)


def test_sources_are_not_ignored() -> None:
    """Ни один исходный файл не должен попадать под .gitignore.

    Правило `data/` без слэша в начале задело бы и пакет src/enmf/data/.
    """
    sources = [str(path.relative_to(ROOT)) for path in (ROOT / "src").rglob("*.py")]
    assert sources
    assert _ignored(sources) == set()


def test_final_artifacts_are_kept() -> None:
    kept = [
        "runs/enmf_ml1m/config.yaml",
        "runs/enmf_ml1m/results.json",
        "runs/enmf_ml1m/split_stats.json",
        "runs/enmf_ml1m/best_model.pt",
        "runs/enmf_ml1m/tensorboard/events.out.tfevents.0",
        "runs/enmf_ddp1/results.json",
        "runs/enmf_ddp2/results.json",
        "runs/enmf_ddp4/results.json",
        "reports/experiments.md",
        "reports/experiments.json",
        "reports/tables.md",
    ]
    assert _ignored(kept) == set()


def test_datasets_and_intermediate_runs_are_ignored() -> None:
    dropped = [
        "data/ml-1m/ratings.dat",
        "data/ml-1m.zip",
        "runs/ml1m_neg_weight_neg_weight01/results.json",
        "runs/enmf_ddp2/best_model.pt",
    ]
    assert _ignored(dropped) == set(dropped)
