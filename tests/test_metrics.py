"""Проверка метрик ранжирования на примерах с известным ответом.

Метрики -- вторая после лосса точка, где ошибка не проявляется явно: код
отработает и выдаст правдоподобное число. Поэтому здесь всё считается вручную
и сверяется по значению.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from enmf.metrics import evaluate_model, ranking_metrics_batch
from enmf.models.popularity import PopularityRecommender


def test_perfect_ranking_gives_one() -> None:
    """Все релевантные объекты на первых позициях -> NDCG = 1."""
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    relevance = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
    result = ranking_metrics_batch(scores, relevance, ks=(2, 5))
    assert result["ndcg@2"].item() == pytest.approx(1.0)
    assert result["ndcg@5"].item() == pytest.approx(1.0)
    assert result["recall@2"].item() == pytest.approx(1.0)


def test_ndcg_hand_computed() -> None:
    """Релевантные объекты на позициях 2 и 4 при двух релевантных всего."""
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    relevance = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]])

    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)      # позиции 2 и 4
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)     # идеал: позиции 1 и 2
    expected = dcg / idcg

    result = ranking_metrics_batch(scores, relevance, ks=(5,))
    assert result["ndcg@5"].item() == pytest.approx(expected, rel=1e-12)


def test_idcg_uses_min_k_and_relevant() -> None:
    """При числе релевантных больше K идеальный DCG обрезается по K.

    Иначе пользователь с 10 отложенными объектами физически не смог бы
    получить NDCG@5 = 1, и метрика мерила бы активность, а не качество.
    """
    scores = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]])
    relevance = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    result = ranking_metrics_batch(scores, relevance, ks=(3,))
    assert result["ndcg@3"].item() == pytest.approx(1.0)


def test_no_relevant_items_yields_zero() -> None:
    scores = torch.tensor([[3.0, 2.0, 1.0]])
    relevance = torch.tensor([[0.0, 0.0, 0.0]])
    result = ranking_metrics_batch(scores, relevance, ks=(3,))
    assert result["ndcg@3"].item() == pytest.approx(0.0)
    assert result["recall@3"].item() == pytest.approx(0.0)
    assert result["hit_rate@3"].item() == pytest.approx(0.0)


def test_recall_and_hit_rate() -> None:
    """Один из трёх релевантных попал в топ-2."""
    scores = torch.tensor([[5.0, 1.0, 4.0, 3.0, 2.0]])
    relevance = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]])
    result = ranking_metrics_batch(scores, relevance, ks=(2,))
    assert result["recall@2"].item() == pytest.approx(1.0 / 3.0)
    assert result["hit_rate@2"].item() == pytest.approx(1.0)


def test_batch_sums_across_users() -> None:
    """Метрики возвращаются именно суммами по пользователям, а не средними."""
    scores = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    relevance = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = ranking_metrics_batch(scores, relevance, ks=(3,))
    assert result["ndcg@3"].item() == pytest.approx(2.0)


def test_seen_items_are_masked_out() -> None:
    """Объекты из обучающей истории не должны попадать в рекомендации."""
    train_by_user = {0: np.array([0, 1])}
    val_by_user = {0: np.array([2])}
    # Популярность максимальна у объектов 0 и 1, но они уже просмотрены.
    model = PopularityRecommender(torch.tensor([100.0, 50.0, 1.0, 0.0]))

    metrics = evaluate_model(
        model,
        users=[0],
        train_by_user=train_by_user,
        val_by_user=val_by_user,
        n_items=4,
        ks=(1,),
        device="cpu",
    )
    # Объект 2 -- лучший среди незамаскированных, значит он на первой позиции.
    assert metrics["ndcg@1"] == pytest.approx(1.0)
    assert metrics["hit_rate@1"] == pytest.approx(1.0)


def test_k_larger_than_catalogue() -> None:
    """K больше размера каталога не должен ронять расчёт."""
    scores = torch.tensor([[2.0, 1.0]])
    relevance = torch.tensor([[1.0, 0.0]])
    result = ranking_metrics_batch(scores, relevance, ks=(10,))
    assert result["ndcg@10"].item() == pytest.approx(1.0)
