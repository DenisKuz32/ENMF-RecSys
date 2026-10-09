"""Модуль рекомендаций: тёплый пользователь -- ENMF, холодный -- популярность."""

from __future__ import annotations

from pathlib import Path

import pytest

from enmf.config import Config
from enmf.data.prepare import prepare_data
from enmf.recommend import recommend_for_user
from enmf.train import train


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Маленький запуск на синтетике с global-разбиением.

    На синтетике global оставляет многих пользователей без обучающей истории,
    поэтому в датасете гарантированно есть и тёплые, и холодные пользователи.
    """
    output = tmp_path_factory.mktemp("runs")
    config = Config.load(Path(__file__).resolve().parent.parent / "configs/synthetic.yaml")
    config.data.split_strategy = "global"
    config.train.epochs = 1
    config.output.output_dir = str(output)
    config.output.run_name = "tiny"
    config.output.tensorboard = False
    train(config)
    return output / "tiny"


def _users(run_dir: Path) -> tuple[int, int]:
    data = prepare_data(Config.load(run_dir / "config.yaml").data)
    warm = next(u for u in range(data.n_users) if u in data.train_by_user)
    cold = next(u for u in range(data.n_users) if u not in data.train_by_user)
    return warm, cold


def test_warm_user_gets_enmf_recommendations(run_dir: Path) -> None:
    warm, _ = _users(run_dir)
    result = recommend_for_user(run_dir, warm, k=5, raw_index=True)
    assert result["source"] == "enmf"
    assert result["history_size"] > 0
    seen = {entry["item_index"] for entry in result["recent_history"]}
    assert not seen & {entry["item_index"] for entry in result["recommendations"]}


def test_cold_user_falls_back_to_popularity(
    run_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Эмбеддинг холодного пользователя не обучен -- рекомендовать по нему нельзя."""
    _, cold = _users(run_dir)
    with caplog.at_level("WARNING", logger="enmf.recommend"):
        result = recommend_for_user(run_dir, cold, k=5, raw_index=True)

    assert result["source"] == "popularity"
    assert result["history_size"] == 0
    assert any("холодный" in record.getMessage() for record in caplog.records)

    data = prepare_data(Config.load(run_dir / "config.yaml").data)
    counts = data.train["item_id"].value_counts()
    top = [entry["item_index"] for entry in result["recommendations"]]
    assert [counts[item] for item in top] == sorted(counts.nlargest(5).tolist(), reverse=True)
