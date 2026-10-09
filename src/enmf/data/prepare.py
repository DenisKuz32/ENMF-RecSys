"""Единая точка подготовки данных: загрузка -> разбиение -> датасет.

Один и тот же код используется обучением, оценкой и инференсом, поэтому
train/validation гарантированно строятся одинаково во всех сценариях.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..utils import DistributedContext, barrier
from .dataset import UserHistoryDataset, make_synthetic_interactions
from .movielens import InteractionData, download_dataset, load_interactions, reindex
from .split import SplitResult, SplitStats, temporal_split

logger = logging.getLogger(__name__)


@dataclass
class PreparedData:
    """Всё, что нужно для обучения и оценки."""

    n_users: int
    n_items: int
    train: pd.DataFrame
    validation: pd.DataFrame
    train_by_user: dict[int, np.ndarray]
    val_by_user: dict[int, np.ndarray]
    split_stats: SplitStats
    dataset: UserHistoryDataset
    item_map: dict[int, int]
    user_map: dict[int, int]

    @property
    def eval_users(self) -> list[int]:
        """Пользователи, у которых есть и обучающая история, и отложенные позитивы."""
        return sorted(
            user for user in self.val_by_user if len(self.train_by_user.get(user, ()))
        )

    def describe(self) -> str:
        return (
            f"пользователей={self.n_users:,} объектов={self.n_items:,}  "
            f"train={len(self.train):,} val={len(self.validation):,}  "
            f"оцениваем {len(self.eval_users):,} пользователей"
        )


def check_user_based(n_users: int, n_items: int) -> bool:
    """Проверяет условие задания: user-based вариант -- при |U| > |I|.

    Реализован только user-based ENMF. Если на датасете пользователей не
    больше, чем объектов, задание предписывает item-based вариант -- об этом
    пишется предупреждение, чтобы несоответствие было видно в логе запуска.
    """
    if n_users > n_items:
        logger.info(
            "user-based вариант соответствует условию: пользователей %d > объектов %d",
            n_users, n_items,
        )
        return True
    logger.warning(
        "пользователей (%d) не больше, чем объектов (%d): по условию задания здесь "
        "нужен item-based вариант ENMF, а реализован только user-based",
        n_users, n_items,
    )
    return False


def ensure_dataset(config: DataConfig, context: DistributedContext) -> None:
    """Скачивает датасет на ранге 0, остальные процессы DDP ждут на барьере.

    Без этого при запуске через torchrun каждый процесс сам начинает скачивать
    и распаковывать один и тот же архив в одну и ту же папку, и процессы
    читают недописанные файлы -- вплоть до того, что у разных рангов
    оказываются разные датасеты.
    """
    if config.dataset == "synthetic":
        return
    if context.is_main:
        download_dataset(config.dataset, config.data_dir)
    barrier(context)


def prepare_data(config: DataConfig) -> PreparedData:
    """Готовит данные согласно конфигурации.

    Специальное значение `dataset: synthetic` порождает синтетический датасет
    с латентной структурой -- он нужен для смоук-тестов и прогонов без доступа
    к сети, весь остальной конвейер при этом остаётся тем же самым.
    """
    if config.dataset == "synthetic":
        frame = make_synthetic_interactions(seed=0)
        data: InteractionData = reindex(frame)
        logger.info("синтетический датасет -- %s", data.describe())
    else:
        data = load_interactions(
            name=config.dataset,
            data_dir=config.data_dir,
            binarize_threshold=config.binarize_threshold,
            min_user_interactions=config.min_user_interactions,
            min_item_interactions=config.min_item_interactions,
        )

    check_user_based(data.n_users, data.n_items)

    split: SplitResult = temporal_split(
        data.interactions,
        strategy=config.split_strategy,
        val_ratio=config.val_ratio,
    )

    dataset = UserHistoryDataset(
        train_by_user=split.train_by_user,
        n_items=data.n_items,
        max_history=config.max_history,
    )

    prepared = PreparedData(
        n_users=data.n_users,
        n_items=data.n_items,
        train=split.train,
        validation=split.validation,
        train_by_user=split.train_by_user,
        val_by_user=split.val_by_user,
        split_stats=split.stats,
        dataset=dataset,
        item_map=data.item_map,
        user_map=data.user_map,
    )
    logger.info("данные готовы -- %s", prepared.describe())
    return prepared
