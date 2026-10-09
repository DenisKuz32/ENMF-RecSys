"""Проверка корректности эффективного non-sampling лосса ENMF.

Ключевой тест всего проекта: эффективная формула (две матрицы Грама d x d)
должна с точностью до численной погрешности совпадать с прямым вычислением
взвешенного квадратичного лосса по всей матрице пользователь x объект.

Совпадения самих значений недостаточно -- обучается модель по градиентам,
поэтому отдельно сверяются и они.
"""

from __future__ import annotations

import pytest
import torch

from enmf.losses import enmf_loss, enmf_loss_naive
from enmf.models.enmf import ENMF


def _make_batch(n_users: int, n_items: int, max_len: int, seed: int = 0):
    """Синтетический батч историй с паддингом и разной длиной у пользователей."""
    generator = torch.Generator().manual_seed(seed)
    users = torch.arange(n_users)
    histories = torch.full((n_users, max_len), n_items, dtype=torch.long)
    for row in range(n_users):
        length = int(torch.randint(1, max_len + 1, (1,), generator=generator).item())
        items = torch.randperm(n_items, generator=generator)[:length]
        histories[row, :length] = items
    return users, histories


def _make_model(n_users: int, n_items: int, dim: int, seed: int = 0) -> ENMF:
    torch.manual_seed(seed)
    # dropout отключён: сравнение должно быть детерминированным.
    model = ENMF(n_users=n_users, n_items=n_items, embedding_dim=dim, dropout=0.0)
    # Крупная инициализация, чтобы скоры были заметно отличны от нуля и тест
    # реально проверял формулу, а не сравнивал два почти-нуля.
    with torch.no_grad():
        model.user_embedding.weight.normal_(std=0.5)
        model.item_embedding.weight.normal_(std=0.5)
        model.item_embedding.weight[model.n_items].zero_()
        model.h.normal_(std=0.5)
    model = model.double()
    model.eval()
    return model


@pytest.mark.parametrize("neg_weight", [0.0, 0.01, 0.1, 0.5, 1.0])
def test_efficient_matches_naive_value(neg_weight: float) -> None:
    n_users, n_items, dim = 17, 41, 12
    model = _make_model(n_users, n_items, dim)
    users, histories = _make_batch(n_users, n_items, max_len=9)

    fast = enmf_loss(model, users, histories, neg_weight=neg_weight)
    slow = enmf_loss_naive(model, users, histories, neg_weight=neg_weight)

    assert torch.allclose(fast, slow, atol=1e-9, rtol=1e-9), (
        f"neg_weight={neg_weight}: efficient={fast.item():.12f} "
        f"naive={slow.item():.12f}"
    )


@pytest.mark.parametrize("pos_weight", [0.5, 1.0, 2.0])
def test_efficient_matches_naive_pos_weight(pos_weight: float) -> None:
    n_users, n_items, dim = 13, 29, 8
    model = _make_model(n_users, n_items, dim, seed=3)
    users, histories = _make_batch(n_users, n_items, max_len=7, seed=3)

    fast = enmf_loss(model, users, histories, pos_weight=pos_weight, neg_weight=0.1)
    slow = enmf_loss_naive(
        model, users, histories, pos_weight=pos_weight, neg_weight=0.1
    )
    assert torch.allclose(fast, slow, atol=1e-9, rtol=1e-9)


def test_gradients_match() -> None:
    """Совпадение градиентов по всем обучаемым параметрам."""
    n_users, n_items, dim = 11, 23, 6
    users, histories = _make_batch(n_users, n_items, max_len=6, seed=7)

    grads = {}
    for name, loss_fn in (("fast", enmf_loss), ("slow", enmf_loss_naive)):
        model = _make_model(n_users, n_items, dim, seed=7)
        loss = loss_fn(model, users, histories, neg_weight=0.2)
        loss.backward()
        grads[name] = {
            "user": model.user_embedding.weight.grad.clone(),
            "item": model.item_embedding.weight.grad.clone(),
            "h": model.h.grad.clone(),
        }

    for key in ("user", "item", "h"):
        assert torch.allclose(grads["fast"][key], grads["slow"][key], atol=1e-9), (
            f"градиенты по '{key}' расходятся, "
            f"max|diff|={(grads['fast'][key] - grads['slow'][key]).abs().max():.3e}"
        )


def test_padding_contributes_nothing() -> None:
    """Добавление паддинга не меняет ни значение лосса, ни градиенты."""
    n_users, n_items, dim = 9, 19, 5
    model = _make_model(n_users, n_items, dim, seed=11)
    users, histories = _make_batch(n_users, n_items, max_len=5, seed=11)

    padded = torch.cat(
        [histories, torch.full((n_users, 4), n_items, dtype=torch.long)], dim=1
    )
    base = enmf_loss(model, users, histories, neg_weight=0.15)
    extra = enmf_loss(model, users, padded, neg_weight=0.15)
    assert torch.allclose(base, extra, atol=1e-12)


def test_padding_embedding_stays_zero_after_step() -> None:
    """Паддинговый эмбеддинг обязан оставаться нулевым после шага оптимизации.

    На этом держится отсутствие маски в лоссе, поэтому проверяется явно.
    """
    n_users, n_items, dim = 8, 15, 4
    model = _make_model(n_users, n_items, dim, seed=5)
    model.train()
    users, histories = _make_batch(n_users, n_items, max_len=5, seed=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for _ in range(5):
        optimizer.zero_grad()
        enmf_loss(model, users, histories, neg_weight=0.1).backward()
        optimizer.step()

    pad_row = model.item_embedding.weight[model.n_items]
    assert torch.allclose(pad_row, torch.zeros_like(pad_row), atol=1e-12)


def test_scoring_identity() -> None:
    """h^T (p * q) == (p * h)^T q -- тождество, на котором стоит full_scores."""
    n_users, n_items, dim = 6, 13, 7
    model = _make_model(n_users, n_items, dim, seed=13)
    users = torch.arange(n_users)
    all_items = torch.arange(n_items).unsqueeze(0).expand(n_users, -1)

    via_items = model.score_items(model.user_embedding(users), all_items)
    via_full = model.full_scores(users)
    assert torch.allclose(via_items, via_full, atol=1e-10)
