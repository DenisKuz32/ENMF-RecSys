"""Бейзлайн item-kNN: похожести сверяются с расчётом вручную.

Истории четырёх пользователей: {0, 1}, {0, 1}, {2}, {0, 2}; n = 4.
Счётчики: n_0 = 3, n_1 = 2, n_2 = 2; совместные: n_01 = 2, n_02 = 1, n_12 = 0.

    φ(0, 1) = (4·2 − 3·2) / sqrt(3·1 · 2·2) = 2 / sqrt(12) ≈ 0.577
    φ(0, 2) = (4·1 − 3·2) / sqrt(12)        < 0  -> обнуляется
    φ(1, 2) = (4·0 − 2·2) / sqrt(2·2 · 2·2) = −1 -> обнуляется
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from enmf.metrics import evaluate_model
from enmf.models.item_knn import ItemKNNRecommender

HISTORIES = {
    0: np.array([0, 1]),
    1: np.array([0, 1]),
    2: np.array([2]),
    3: np.array([0, 2]),
}
PHI_01 = 2.0 / math.sqrt(12.0)


def test_similarity_matches_hand_computation() -> None:
    knn = ItemKNNRecommender.fit(HISTORIES, n_users=4, n_items=3, shrinkage=0.0)
    expected = torch.tensor(
        [[0.0, PHI_01, 0.0], [PHI_01, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert torch.allclose(knn.similarity, expected, atol=1e-6)


def test_shrinkage_damps_rare_pairs() -> None:
    """φ̃ = φ · n_ij / (n_ij + α): при n_01 = 2 и α = 2 корреляция делится пополам."""
    knn = ItemKNNRecommender.fit(HISTORIES, n_users=4, n_items=3, shrinkage=2.0)
    assert knn.similarity[0, 1].item() == pytest.approx(PHI_01 / 2, rel=1e-6)


def test_sum_and_topk_mean_aggregations() -> None:
    users = torch.tensor([0, 2, 3])
    total = ItemKNNRecommender.fit(HISTORIES, 4, 3, shrinkage=0.0, aggregation="sum")
    best = ItemKNNRecommender.fit(
        HISTORIES, 4, 3, shrinkage=0.0, aggregation="topk_mean", top_k=1
    )
    expected = torch.tensor(
        [
            [PHI_01, PHI_01, 0.0],  # история {0, 1}
            [0.0, 0.0, 0.0],        # история {2}: похожих фильмов нет
            [0.0, PHI_01, 0.0],     # история {0, 2}
        ]
    )
    assert torch.allclose(total.full_scores(users), expected, atol=1e-6)
    assert torch.allclose(best.full_scores(users), expected, atol=1e-6)


def test_unknown_aggregation_rejected() -> None:
    with pytest.raises(ValueError):
        ItemKNNRecommender.fit(HISTORIES, 4, 3, aggregation="median")


def test_evaluated_by_common_code_with_seen_items_masked() -> None:
    """У пользователя 3 в истории {0, 2}; лучший непросмотренный -- фильм 1."""
    knn = ItemKNNRecommender.fit(HISTORIES, n_users=4, n_items=3, shrinkage=0.0)
    metrics = evaluate_model(
        knn, users=[3], train_by_user=HISTORIES, val_by_user={3: np.array([1])},
        n_items=3, ks=(1,), device="cpu",
    )
    assert metrics["ndcg@1"] == pytest.approx(1.0)
