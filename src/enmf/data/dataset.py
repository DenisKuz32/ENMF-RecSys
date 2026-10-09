"""Датасет и коллатор для обучения ENMF-U.

Один элемент датасета -- это *один пользователь* вместе со всей его историей
взаимодействий, а не одна пара (пользователь, объект). Так устроен user-based
вариант ENMF: лосс для пользователя считается сразу по всем его позитивам плюс
по всему каталогу через матрицу Грама.

Истории имеют разную длину, поэтому коллатор дополняет их паддинговым индексом
`n_items`. Паддинг выравнивается по самой длинной истории *в батче*, а не по
всему датасету: на MovieLens-1M самый активный пользователь имеет более двух
тысяч оценок при медиане около сотни, и выравнивание по глобальному максимуму
раздуло бы батчи на порядок.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class UserHistoryDataset(Dataset):
    """Пользователи и их обучающие истории.

    Args:
        train_by_user: отображение user_id -> массив item_id.
        n_items: размер каталога; индекс `n_items` используется как паддинг.
        max_history: если задано, история усекается до последних N элементов.
    """

    def __init__(
        self,
        train_by_user: dict[int, np.ndarray],
        n_items: int,
        max_history: int | None = None,
    ) -> None:
        self.n_items = int(n_items)
        self.max_history = max_history

        self.users: list[int] = sorted(
            user for user, items in train_by_user.items() if len(items) > 0
        )
        self.histories: list[np.ndarray] = []
        for user in self.users:
            items = np.asarray(train_by_user[user], dtype=np.int64)
            if max_history is not None and len(items) > max_history:
                items = items[-max_history:]  # оставляем самые свежие
            self.histories.append(items)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray]:
        return self.users[index], self.histories[index]

    @property
    def max_history_length(self) -> int:
        return max((len(h) for h in self.histories), default=0)


def collate_histories(
    batch: list[tuple[int, np.ndarray]], n_items: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Собирает батч, дополняя истории паддинговым индексом до максимума в батче.

    Returns:
        users: LongTensor [B]
        histories: LongTensor [B, L], где L -- максимальная длина в батче.
    """
    users = torch.tensor([user for user, _ in batch], dtype=torch.long)
    lengths = [len(items) for _, items in batch]
    max_length = max(lengths) if lengths else 1

    histories = torch.full((len(batch), max_length), n_items, dtype=torch.long)
    for row, (_, items) in enumerate(batch):
        histories[row, : len(items)] = torch.from_numpy(items)
    return users, histories


def make_collate_fn(n_items: int):
    """Фабрика коллатора с зашитым размером каталога (для DataLoader)."""

    def _collate(batch):
        return collate_histories(batch, n_items)

    return _collate


def make_synthetic_interactions(
    n_users: int = 800,
    n_items: int = 200,
    n_latent: int = 8,
    interactions_per_user: tuple[int, int] = (20, 60),
    popularity_sigma: float = 0.5,
    seed: int = 0,
):
    """Синтетические взаимодействия с реальной латентной структурой.

    Нужны для тестов и для прогонов без доступа к сети. Важно, что данные здесь
    не случайные: каждому пользователю и объекту назначается латентный профиль,
    и вероятность взаимодействия зависит от их близости. Благодаря этому
    факторизационная модель обязана обыгрывать популярностный бейзлайн -- если
    этого не происходит, значит сломано обучение, а не данные.

    Returns:
        pandas.DataFrame с колонками user_id, item_id, timestamp.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)

    user_profiles = rng.dirichlet(np.ones(n_latent) * 0.4, size=n_users)
    item_profiles = rng.dirichlet(np.ones(n_latent) * 0.4, size=n_items)
    # Популярность объектов неравномерна, иначе бейзлайн по популярности
    # вырождается и сравнение с ним теряет смысл. Но разброс умеренный
    # (логнормальный, а не тяжелохвостый): при слишком сильном перекосе
    # популярность объясняет почти все взаимодействия, коллаборативного сигнала
    # в данных не остаётся, и датасет перестаёт годиться как проверка модели.
    popularity = rng.lognormal(0.0, popularity_sigma, size=n_items)

    affinity = user_profiles @ item_profiles.T * popularity[None, :]
    affinity = affinity / affinity.sum(axis=1, keepdims=True)

    rows = []
    timestamp = 1_000_000
    for user in range(n_users):
        count = int(rng.integers(interactions_per_user[0], interactions_per_user[1] + 1))
        count = min(count, n_items)
        chosen = rng.choice(n_items, size=count, replace=False, p=affinity[user])
        # Время растёт вместе с активностью: даёт осмысленный временной порядок.
        for item in chosen:
            timestamp += int(rng.integers(1, 60))
            rows.append((user, int(item), timestamp))

    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
