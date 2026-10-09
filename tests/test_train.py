"""Сквозная проверка цикла обучения на маленькой синтетике."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enmf.config import Config
from enmf.train import train


def _tiny_config(tmp_path: Path, **train_overrides) -> Config:
    config = Config.load(Path(__file__).resolve().parent.parent / "configs/synthetic.yaml")
    config.output.output_dir = str(tmp_path)
    config.output.run_name = "tiny"
    config.output.tensorboard = False
    for key, value in train_overrides.items():
        setattr(config.train, key, value)
    return config


def test_history_keeps_epoch_that_triggered_early_stopping(tmp_path: Path) -> None:
    """Эпоха, на которой сработала ранняя остановка, тоже попадает в историю.

    При lr=0 метрики не меняются, улучшения после первой эпохи нет, и при
    patience=1 обучение останавливается на второй эпохе.
    """
    config = _tiny_config(tmp_path, epochs=5, lr=0.0, early_stopping_patience=1)
    summary = train(config)

    assert [record["epoch"] for record in summary["history"]] == [1, 2]
    saved = json.loads((tmp_path / "tiny" / "results.json").read_text(encoding="utf-8"))
    assert [record["epoch"] for record in saved["history"]] == [1, 2]


def test_results_contain_both_baselines(tmp_path: Path) -> None:
    """В results.json -- метрики популярности и item-kNN и приросты к обоим."""
    summary = train(_tiny_config(tmp_path, epochs=1))
    for key in ("baseline_popularity", "baseline_item_knn"):
        assert 0.0 < summary[key]["ndcg@10"] <= 1.0
    assert set(summary["improvement_vs_item_knn"]) == set(summary["baseline_item_knn"])


def test_relative_improvement_is_json_safe() -> None:
    """Нулевой бейзлайн даёт None, а не NaN: results.json остаётся валидным JSON."""
    from enmf.train import relative_improvement

    improvement = relative_improvement(
        {"ndcg@10": 0.15, "recall@10": 0.2}, {"ndcg@10": 0.1, "recall@10": 0.0}
    )
    assert improvement["ndcg@10"] == pytest.approx(0.5)
    assert improvement["recall@10"] is None
    json.dumps(improvement, allow_nan=False)
