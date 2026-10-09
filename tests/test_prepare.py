"""Подготовка данных в режиме DDP: датасет скачивает только ранг 0."""

from __future__ import annotations

import pytest

from enmf.config import DataConfig
from enmf.data import prepare
from enmf.utils import DistributedContext


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    calls: dict[str, list] = {"download": [], "barrier": []}
    monkeypatch.setattr(
        prepare, "download_dataset", lambda name, data_dir: calls["download"].append(name)
    )
    monkeypatch.setattr(prepare, "barrier", lambda context: calls["barrier"].append(context.rank))
    return calls


def test_only_rank_zero_downloads(recorded: dict[str, list]) -> None:
    """Скачивает ранг 0, барьер проходят все ранги."""
    config = DataConfig(dataset="ml-1m")
    for rank in range(3):
        context = DistributedContext(rank=rank, local_rank=rank, world_size=3, enabled=True)
        prepare.ensure_dataset(config, context)

    assert recorded["download"] == ["ml-1m"]
    assert recorded["barrier"] == [0, 1, 2]


def test_synthetic_needs_no_download(recorded: dict[str, list]) -> None:
    prepare.ensure_dataset(DataConfig(dataset="synthetic"), DistributedContext())
    assert recorded == {"download": [], "barrier": []}


def test_user_based_condition_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """|U| > |I| -- user-based уместен; иначе в логе предупреждение."""
    with caplog.at_level("INFO", logger="enmf.data.prepare"):
        assert prepare.check_user_based(n_users=6040, n_items=3416) is True
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

        assert prepare.check_user_based(n_users=610, n_items=3650) is False
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "item-based" in warnings[0].getMessage()
