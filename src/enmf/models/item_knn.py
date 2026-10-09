"""Бейзлайн item-kNN: рекомендации по похожести фильмов на историю пользователя.

Классическая item-based коллаборативная фильтрация без обучения. Два фильма
считаются похожими, если они чаще обычного встречаются в историях одних и тех
же пользователей. Мера похожести -- корреляция Пирсона бинарных признаков
"фильм есть в истории пользователя" (для бинарных величин это коэффициент φ):

    φ(i, j) = (n·n_ij − n_i·n_j) / sqrt(n_i·(n − n_i) · n_j·(n − n_j))

где n -- число пользователей с обучающей историей, n_i -- у скольких из них
есть фильм i, n_ij -- у скольких есть оба фильма. Корреляция по малому числу
совместных наблюдений ненадёжна, поэтому она сглаживается:

    φ̃(i, j) = φ(i, j) · n_ij / (n_ij + α)

Отрицательные корреляции и корреляция фильма с самим собой обнуляются. Скор
кандидата i для пользователя u агрегирует его похожесть на фильмы из истории
H_u:

    sum       -- Σ_{j ∈ H_u} φ̃(i, j)            (классический item-kNN)
    topk_mean -- среднее k наибольших φ̃(i, j)

Все статистики считаются только по обучающей части, поэтому утечки из
валидации нет. Реализует тот же контракт `full_scores(users) -> [B, n_items]`,
что ENMF и популярность, и оценивается тем же кодом.
"""

from __future__ import annotations

import numpy as np
import torch


class ItemKNNRecommender:
    """Item-based коллаборативная фильтрация на сглаженной φ-корреляции."""

    def __init__(
        self,
        similarity: torch.Tensor,
        train_by_user: dict[int, np.ndarray],
        n_users: int,
        aggregation: str = "sum",
        top_k: int = 3,
    ) -> None:
        if aggregation not in ("sum", "topk_mean"):
            raise ValueError(f"неизвестная агрегация: '{aggregation}'")
        self.similarity = similarity
        self.n_items = int(similarity.size(0))
        self.aggregation = aggregation
        self.top_k = int(top_k)
        self.training = False

        # Матрица историй [n_users, n_items]: строка пользователя -- индикатор
        # фильмов из его обучающей истории.
        self.history = torch.zeros((n_users, self.n_items), dtype=torch.float32)
        for user, items in train_by_user.items():
            self.history[user, torch.as_tensor(items)] = 1.0

    @classmethod
    def fit(
        cls,
        train_by_user: dict[int, np.ndarray],
        n_users: int,
        n_items: int,
        shrinkage: float = 10.0,
        aggregation: str = "sum",
        top_k: int = 3,
    ) -> ItemKNNRecommender:
        """Считает матрицу похожестей по обучающим историям."""
        users = [user for user, items in train_by_user.items() if len(items)]
        matrix = np.zeros((len(users), n_items), dtype=np.float64)
        for row, user in enumerate(users):
            matrix[row, train_by_user[user]] = 1.0

        n = float(len(users))
        co_count = matrix.T @ matrix                      # n_ij, [n_items, n_items]
        item_count = np.diag(co_count).copy()             # n_i
        numerator = n * co_count - np.outer(item_count, item_count)
        spread = item_count * (n - item_count)
        denominator = np.sqrt(np.outer(spread, spread))
        phi = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        )
        # Пары без совместных появлений получают множитель 0 (и при α = 0
        # тоже, где иначе вышло бы 0/0).
        smoothed = co_count + shrinkage
        phi *= np.divide(co_count, smoothed, out=np.zeros_like(co_count), where=smoothed > 0)
        np.fill_diagonal(phi, 0.0)
        np.maximum(phi, 0.0, out=phi)

        similarity = torch.from_numpy(phi.astype(np.float32))
        return cls(similarity, train_by_user, n_users, aggregation, top_k)

    def to(self, device: torch.device | str) -> ItemKNNRecommender:
        self.similarity = self.similarity.to(device)
        self.history = self.history.to(device)
        return self

    def full_scores(self, users: torch.Tensor) -> torch.Tensor:
        """Скоры пользователей по всему каталогу, [B, n_items]."""
        history = self.history.to(users.device)[users]                  # [B, I]
        similarity = self.similarity.to(users.device)
        if self.aggregation == "sum":
            return history @ similarity                                  # Σ_j φ̃(i, j)

        scores = torch.zeros_like(history)
        for row in range(history.size(0)):
            seen = history[row].nonzero(as_tuple=True)[0]
            if seen.numel() == 0:
                continue
            k = min(self.top_k, seen.numel())
            # similarity[:, seen] -- похожесть каждого кандидата на фильмы истории.
            scores[row] = similarity[:, seen].topk(k, dim=1).values.mean(dim=1)
        return scores

    # Заглушки под интерфейс nn.Module, чтобы общий код оценки работал как есть.
    def eval(self) -> ItemKNNRecommender:
        return self

    def train(self, mode: bool = True) -> ItemKNNRecommender:
        return self
