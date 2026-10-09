"""ENMF: Efficient Neural Matrix Factorization without Sampling.

Chen C., Zhang M., Zhang Y., Liu Y., Ma S.
"Efficient Neural Matrix Factorization without Sampling for Recommendation".
ACM TOIS 38(2), Article 14, 2020.  https://doi.org/10.1145/3373807

Архитектура предельно простая:

    y_hat(u, i) = h^T (p_u * q_i)

где p_u -- эмбеддинг пользователя, q_i -- эмбеддинг объекта, `*` --
поэлементное произведение, h -- обучаемый вектор весов предсказательного слоя.
Вся "нейронность" модели -- в этом одном слое; основной вклад статьи лежит не в
архитектуре, а в способе обучать её на всех данных без негативного сэмплинга
(см. `enmf.losses`).
"""

from __future__ import annotations

import torch
from torch import nn


class ENMF(nn.Module):
    """User/item факторизация с обучаемым предсказательным слоем h.

    Для объектов заводится один дополнительный "паддинговый" эмбеддинг с
    индексом `n_items`. Он инициализирован нулями и остаётся нулевым (PyTorch
    не пропускает градиент через `padding_idx`), поэтому предсказание для
    паддинговой позиции тождественно равно нулю. Это позволяет считать лосс по
    истории пользователя без явной маски -- см. комментарий в `losses.py`.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        dropout: float = 0.3,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_dim, padding_idx=self.n_items
        )
        self.h = nn.Parameter(torch.empty(self.embedding_dim))
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights(init_std)

    def _init_weights(self, init_std: float) -> None:
        nn.init.normal_(self.user_embedding.weight, std=init_std)
        nn.init.normal_(self.item_embedding.weight, std=init_std)
        with torch.no_grad():
            self.item_embedding.weight[self.item_embedding.padding_idx].zero_()
        nn.init.normal_(self.h, std=init_std)

    # ------------------------------------------------------------------ #
    # Базовые представления
    # ------------------------------------------------------------------ #

    @property
    def item_matrix(self) -> torch.Tensor:
        """Матрица эмбеддингов реальных объектов Q, shape [n_items, d].

        Паддинговая строка отрезается: она не является объектом каталога и не
        должна участвовать ни в ранжировании, ни в матрице Грама.
        """
        return self.item_embedding.weight[: self.n_items]

    def user_repr(self, users: torch.Tensor) -> torch.Tensor:
        """Представление пользователей с dropout (активен только в train-режиме).

        Args:
            users: LongTensor [B] -- индексы пользователей.
        Returns:
            FloatTensor [B, d].
        """
        return self.dropout(self.user_embedding(users))

    # ------------------------------------------------------------------ #
    # Скоринг
    # ------------------------------------------------------------------ #

    def score_items(self, user_repr: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Скоры для заданных пар (пользователь, объект).

        Args:
            user_repr: FloatTensor [B, d] -- уже полученные представления.
            items: LongTensor [B, L] -- индексы объектов (возможно с паддингом).
        Returns:
            FloatTensor [B, L]. Для паддинговых позиций ровно 0.0.
        """
        item_repr = self.item_embedding(items)            # [B, L, d]
        weighted = user_repr.unsqueeze(1) * item_repr     # [B, L, d]
        return weighted @ self.h                          # [B, L]

    def full_scores(self, users: torch.Tensor) -> torch.Tensor:
        """Скоры пользователей по всему каталогу объектов.

        Используется тождество h^T (p * q) = (p * h)^T q, что превращает
        скоринг в одно матричное умножение [B, d] @ [d, n_items].

        Args:
            users: LongTensor [B].
        Returns:
            FloatTensor [B, n_items].
        """
        user_repr = self.user_embedding(users)            # [B, d], без dropout
        return (user_repr * self.h) @ self.item_matrix.t()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return self.score_items(self.user_repr(users), items)

    # ------------------------------------------------------------------ #
    # Служебное
    # ------------------------------------------------------------------ #

    def l2_norm(self) -> torch.Tensor:
        """Сумма квадратов обучаемых параметров (для L2-регуляризации)."""
        return (
            self.user_embedding.weight.pow(2).sum()
            + self.item_matrix.pow(2).sum()
            + self.h.pow(2).sum()
        )

    def extra_repr(self) -> str:
        return (
            f"n_users={self.n_users}, n_items={self.n_items}, "
            f"embedding_dim={self.embedding_dim}"
        )
