"""Проверка временного разбиения.

Ключевое требование задания: взаимодействия из будущего относительно
валидационной части не должны попадать в обучающую. Это и проверяется явно --
плюс корректность удаления холодных пользователей и объектов.
"""

from __future__ import annotations

import pandas as pd
import pytest

from enmf.data.dataset import make_synthetic_interactions
from enmf.data.split import compare_strategies, temporal_split


@pytest.fixture
def interactions() -> pd.DataFrame:
    return make_synthetic_interactions(n_users=60, n_items=40, seed=1)


def test_global_split_has_no_future_leakage(interactions: pd.DataFrame) -> None:
    """При глобальном разбиении train целиком раньше validation."""
    result = temporal_split(interactions, strategy="global", val_ratio=0.2)
    assert len(result.train) > 0
    assert result.train["timestamp"].max() < result.stats.cutoff_timestamp
    if len(result.validation):
        assert result.validation["timestamp"].min() >= result.stats.cutoff_timestamp


def test_per_user_split_has_no_future_leakage(interactions: pd.DataFrame) -> None:
    """При пользовательском разбиении для каждого юзера train раньше validation."""
    result = temporal_split(interactions, strategy="per_user", val_ratio=0.2)

    train_last = result.train.groupby("user_id")["timestamp"].max()
    val_first = result.validation.groupby("user_id")["timestamp"].min()
    shared = train_last.index.intersection(val_first.index)
    assert len(shared) > 0, "нет пользователей, попавших и в train, и в validation"
    assert (train_last[shared] <= val_first[shared]).all()


def test_per_user_split_keeps_train_history(interactions: pd.DataFrame) -> None:
    """У каждого валидационного пользователя остаётся непустая история обучения."""
    result = temporal_split(interactions, strategy="per_user", val_ratio=0.5)
    for user in result.validation["user_id"].unique():
        assert len(result.train_by_user.get(int(user), [])) > 0


def test_cold_users_and_items_removed_from_validation(interactions: pd.DataFrame) -> None:
    """В валидации не остаётся сущностей, отсутствующих в обучающей части."""
    result = temporal_split(interactions, strategy="global", val_ratio=0.3)
    train_users = set(result.train["user_id"])
    train_items = set(result.train["item_id"])
    assert set(result.validation["user_id"]) <= train_users
    assert set(result.validation["item_id"]) <= train_items


def test_train_and_validation_do_not_overlap(interactions: pd.DataFrame) -> None:
    """Одна и та же пара (пользователь, объект) не может быть в обеих частях."""
    result = temporal_split(interactions, strategy="per_user", val_ratio=0.2)
    train_pairs = set(map(tuple, result.train[["user_id", "item_id"]].to_numpy()))
    val_pairs = set(map(tuple, result.validation[["user_id", "item_id"]].to_numpy()))
    assert train_pairs.isdisjoint(val_pairs)


def test_grouped_histories_are_time_ordered(interactions: pd.DataFrame) -> None:
    """Истории сгруппированы по пользователю в хронологическом порядке."""
    result = temporal_split(interactions, strategy="per_user", val_ratio=0.2)
    ordered = interactions.sort_values(["user_id", "timestamp"], kind="stable")
    for user, items in list(result.train_by_user.items())[:5]:
        expected = ordered[ordered["user_id"] == user]["item_id"].to_numpy()
        assert list(items) == list(expected[: len(items)])


def test_compare_strategies_returns_both(interactions: pd.DataFrame) -> None:
    stats = compare_strategies(interactions, val_ratio=0.2)
    assert set(stats) == {"global", "per_user"}
    assert all(0.0 <= stat.val_user_coverage <= 1.0 for stat in stats.values())


def test_invalid_arguments_rejected(interactions: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        temporal_split(interactions, val_ratio=0.0)
    with pytest.raises(ValueError):
        temporal_split(interactions, strategy="random")


def test_grouped_histories_are_writable(interactions: pd.DataFrame) -> None:
    """Массивы историй -- собственные копии, а не read-only виды pandas.

    В pandas 3 `to_numpy()` отдаёт вид только для чтения, и torch при
    `as_tensor` на таком массиве предупреждает о неопределённом поведении.
    """
    result = temporal_split(interactions, strategy="per_user", val_ratio=0.2)
    for by_user in (result.train_by_user, result.val_by_user):
        assert all(items.flags.writeable for items in by_user.values())
