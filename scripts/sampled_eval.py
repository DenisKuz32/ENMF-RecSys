#!/usr/bin/env python3
"""Оценка по сэмплированному протоколу -- для сравнения с полным ранжированием.

    python scripts/sampled_eval.py --run runs/enmf_ml1m

Распространённый сэмплированный протокол: для каждого пользователя 3 случайных
отложенных объекта плюс 47 случайных объектов, которых он не видел ни в train,
ни в валидации; NDCG@10 по этим 50 кандидатам. Основная оценка проекта --
полное ранжирование каталога (enmf.metrics); скрипт нужен, чтобы показать,
насколько абсолютные числа и даже относительный выигрыш зависят от протокола.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enmf.data.prepare import prepare_data
from enmf.evaluate import build_model_from_checkpoint, load_run
from enmf.models.popularity import PopularityRecommender
from enmf.utils import setup_logging

N_POSITIVES, N_NEGATIVES, K = 3, 47, 10
DISCOUNTS = 1.0 / np.log2(np.arange(2, K + 2))


def ndcg_at_k(relevance_in_ranked_order: np.ndarray) -> float:
    top = relevance_in_ranked_order[:K]
    dcg = float((top * DISCOUNTS[: len(top)]).sum())
    idcg = float(DISCOUNTS[: min(K, int(relevance_in_ranked_order.sum()))].sum())
    return dcg / idcg


def main() -> int:
    parser = argparse.ArgumentParser(description="NDCG@10 по 3 + 47 кандидатам")
    parser.add_argument("--run", required=True, help="каталог запуска")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    config, checkpoint = load_run(args.run)
    model = build_model_from_checkpoint(config, checkpoint, torch.device("cpu"))
    data = prepare_data(config.data)
    popularity = PopularityRecommender.fit(data.train["item_id"].to_numpy(), data.n_items)
    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        enmf_scores = model.full_scores(torch.tensor(data.eval_users)).numpy()
    popularity_scores = popularity.item_counts.numpy()
    labels = np.r_[np.ones(N_POSITIVES), np.zeros(N_NEGATIVES)]

    results: dict[str, list[float]] = {"ENMF": [], "популярность": [], "случайно": []}
    for row, user in enumerate(data.eval_users):
        positives = data.val_by_user[user]
        if len(positives) < N_POSITIVES:
            continue
        seen = np.union1d(data.train_by_user[user], positives)
        pool = np.setdiff1d(np.arange(data.n_items), seen)
        candidates = np.concatenate([
            rng.choice(positives, N_POSITIVES, replace=False),
            rng.choice(pool, N_NEGATIVES, replace=False),
        ])
        for name, scores in (
            ("ENMF", enmf_scores[row, candidates]),
            ("популярность", popularity_scores[candidates]),
            ("случайно", rng.normal(size=len(candidates))),
        ):
            results[name].append(ndcg_at_k(labels[np.argsort(-scores, kind="stable")]))

    print(f"{args.run}: пользователей {len(results['ENMF'])}")
    for name, values in results.items():
        print(f"  {name:13s} NDCG@10 = {np.mean(values):.4f}")
    gain = np.mean(results["ENMF"]) / np.mean(results["популярность"]) - 1.0
    print(f"  ENMF к популярности: {gain:+.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
