"""Разбиение взаимодействий на train и validation по времени.

Задание требует разбиения по времени с условием: взаимодействия из будущего
относительно валидационной части не должны попадать в обучающую. Реализованы
две стратегии, обе это условие удовлетворяют, но по-разному:

`global`
    Единая точка отсечения по времени для всего датасета: всё, что случилось
    позже момента T, уходит в валидацию. Условие задания выполняется буквально:
    ни одно обучающее взаимодействие не позже ни одного валидационного. Так же
    выглядит продакшен: модель обучена на прошлом и предсказывает будущее.
    Цена -- часть пользователей "будущего" не имеет истории в train, и
    user-based модель их оценить не может (нет обученного эмбеддинга). На
    MovieLens-1M при val_ratio=0.2 оценимы 1 143 пользователя из 1 783; на
    ml-latest-small -- только 25 из 122, там стратегия вырождается.

`per_user`
    Точка отсечения своя для каждого пользователя: последние взаимодействия
    каждого пользователя уходят в валидацию. Условие "будущее не попадает в
    прошлое" выполняется только на уровне пользователя: в train могут попасть
    взаимодействия пользователя A, сделанные позже валидационных
    взаимодействий пользователя B. Распространённый протокол в литературе,
    оценимы все пользователи.

Основная стратегия для MovieLens-1M -- `global` (configs/ml1m.yaml), `per_user`
используется как дополнительный эксперимент. Диагностика в `SplitStats`
показывает, какая доля валидационных пользователей вообще оценима; команда
`compare-splits` считает обе стратегии.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SplitStrategy = Literal["global", "per_user"]


@dataclass
class SplitStats:
    """Диагностика разбиения -- напрямую идёт в отчёт."""

    strategy: str
    n_train: int
    n_val_raw: int
    n_val_kept: int
    n_users_total: int
    n_users_in_train: int
    n_val_users_raw: int
    n_val_users_kept: int
    n_items_in_train: int
    cutoff_timestamp: int | None = None
    dropped_cold_users: int = 0
    dropped_cold_items: int = 0

    @property
    def val_user_coverage(self) -> float:
        """Доля валидационных пользователей, которых модель в принципе может оценить."""
        if self.n_val_users_raw == 0:
            return 0.0
        return self.n_val_users_kept / self.n_val_users_raw

    def describe(self) -> str:
        return (
            f"стратегия={self.strategy}  "
            f"train={self.n_train:,}  val={self.n_val_kept:,} (из {self.n_val_raw:,})  "
            f"оцениваемых пользователей={self.n_val_users_kept:,}/{self.n_val_users_raw:,} "
            f"({self.val_user_coverage:.1%})"
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["val_user_coverage"] = self.val_user_coverage
        return payload


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    stats: SplitStats
    train_by_user: dict[int, np.ndarray] = field(default_factory=dict)
    val_by_user: dict[int, np.ndarray] = field(default_factory=dict)


def _group_items(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    """user_id -> массив item_id, отсортированный по времени."""
    ordered = frame.sort_values(["user_id", "timestamp"], kind="stable")
    grouped = ordered.groupby("user_id", sort=True)["item_id"]
    # copy=True: в pandas 3 to_numpy() отдаёт вид только для чтения.
    return {int(user): items.to_numpy(dtype=np.int64, copy=True) for user, items in grouped}


def _global_time_split(
    frame: pd.DataFrame, val_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Одна точка отсечения по времени для всего датасета."""
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cut_index = int(len(ordered) * (1.0 - val_ratio))
    cutoff = int(ordered.loc[cut_index, "timestamp"])

    # Отсечение строго по значению времени, а не по позиции: взаимодействия с
    # одинаковой меткой времени не должны оказаться по разные стороны границы.
    train = ordered[ordered["timestamp"] < cutoff]
    validation = ordered[ordered["timestamp"] >= cutoff]
    return train.reset_index(drop=True), validation.reset_index(drop=True), cutoff


def _per_user_time_split(
    frame: pd.DataFrame, val_ratio: float, min_train_interactions: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Последняя доля взаимодействий каждого пользователя уходит в валидацию."""
    ordered = frame.sort_values(["user_id", "timestamp"], kind="stable").reset_index(
        drop=True
    )
    position = ordered.groupby("user_id", sort=False).cumcount()
    total = ordered.groupby("user_id", sort=False)["item_id"].transform("size")

    n_val = np.maximum(1, np.floor(total * val_ratio)).astype(np.int64)
    # Гарантируем, что в обучающей части у пользователя что-то остаётся.
    n_val = np.minimum(n_val, np.maximum(0, total - min_train_interactions))

    is_validation = position >= (total - n_val)
    train = ordered[~is_validation]
    validation = ordered[is_validation]
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def temporal_split(
    interactions: pd.DataFrame,
    strategy: SplitStrategy = "global",
    val_ratio: float = 0.2,
    min_train_interactions: int = 1,
) -> SplitResult:
    """Разбивает взаимодействия на train/validation по времени.

    После разбиения из валидации удаляются:
      * пользователи, у которых нет ни одного обучающего взаимодействия
        (у них нет обученного эмбеддинга -- оценить их user-based моделью
        невозможно);
      * объекты, не встречавшиеся в обучающей части (их эмбеддинги не обучены и
        остались в начальном состоянии; оставить их означало бы измерять шум).

    Оба удаления фиксируются в статистике, чтобы результат нельзя было
    случайно приукрасить.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio должен быть в (0, 1), получено {val_ratio}")

    cutoff: int | None = None
    if strategy == "global":
        train, validation, cutoff = _global_time_split(interactions, val_ratio)
    elif strategy == "per_user":
        train, validation = _per_user_time_split(
            interactions, val_ratio, min_train_interactions
        )
    else:
        raise ValueError(f"неизвестная стратегия разбиения: '{strategy}'")

    n_val_raw = len(validation)
    n_val_users_raw = validation["user_id"].nunique()

    train_users = set(train["user_id"].unique().tolist())
    train_items = set(train["item_id"].unique().tolist())

    cold_user_mask = ~validation["user_id"].isin(train_users)
    cold_item_mask = ~validation["item_id"].isin(train_items)
    dropped_cold_users = int(validation.loc[cold_user_mask, "user_id"].nunique())
    dropped_cold_items = int(validation.loc[cold_item_mask, "item_id"].nunique())

    validation = validation[~cold_user_mask & ~cold_item_mask].reset_index(drop=True)

    stats = SplitStats(
        strategy=strategy,
        n_train=len(train),
        n_val_raw=n_val_raw,
        n_val_kept=len(validation),
        n_users_total=int(interactions["user_id"].nunique()),
        n_users_in_train=len(train_users),
        n_val_users_raw=int(n_val_users_raw),
        n_val_users_kept=int(validation["user_id"].nunique()),
        n_items_in_train=len(train_items),
        cutoff_timestamp=cutoff,
        dropped_cold_users=dropped_cold_users,
        dropped_cold_items=dropped_cold_items,
    )
    logger.info("разбиение: %s", stats.describe())

    return SplitResult(
        train=train,
        validation=validation,
        stats=stats,
        train_by_user=_group_items(train),
        val_by_user=_group_items(validation),
    )


def compare_strategies(
    interactions: pd.DataFrame, val_ratio: float = 0.2
) -> dict[str, SplitStats]:
    """Считает обе стратегии и возвращает их статистику -- материал для отчёта."""
    return {
        strategy: temporal_split(
            interactions, strategy=strategy, val_ratio=val_ratio
        ).stats
        for strategy in ("global", "per_user")
    }
