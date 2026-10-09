"""Бейзлайн на основе популярности.

Рекомендует всем пользователям одни и те же самые популярные объекты каталога,
исключая уже просмотренные (исключение делает `enmf.metrics`, общее для всех
моделей). Это обязательная точка отсчёта: рекомендательная модель, не
обыгрывающая популярность, бесполезна, а на разреженных данных популярность --
на удивление сильный соперник.

Реализует тот же контракт `full_scores(users) -> [B, n_items]`, что и ENMF,
поэтому оценивается тем же самым кодом без каких-либо поблажек.
"""

from __future__ import annotations

import numpy as np
import torch


class PopularityRecommender:
    """Неперсонализированный бейзлайн: скор объекта = его частота в train."""

    def __init__(self, item_counts: torch.Tensor) -> None:
        self.item_counts = item_counts.to(torch.float32)
        self.n_items = int(item_counts.numel())
        self.training = False

    @classmethod
    def fit(
        cls, train_items: np.ndarray | torch.Tensor, n_items: int
    ) -> PopularityRecommender:
        """Считает частоты объектов по обучающей части."""
        items = torch.as_tensor(np.array(train_items, copy=True), dtype=torch.long)
        counts = torch.bincount(items, minlength=n_items).to(torch.float32)
        return cls(counts[:n_items])

    def to(self, device: torch.device | str) -> PopularityRecommender:
        self.item_counts = self.item_counts.to(device)
        return self

    def full_scores(self, users: torch.Tensor) -> torch.Tensor:
        """Один и тот же вектор популярности для каждого пользователя батча."""
        counts = self.item_counts.to(users.device)
        return counts.unsqueeze(0).expand(users.size(0), -1)

    # Заглушки под интерфейс nn.Module, чтобы общий код оценки работал как есть.
    def eval(self) -> PopularityRecommender:
        return self

    def train(self, mode: bool = True) -> PopularityRecommender:
        return self

    def top_k(self, k: int) -> torch.Tensor:
        return torch.topk(self.item_counts, k=min(k, self.n_items)).indices
