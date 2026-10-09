#!/usr/bin/env python3
"""Диагностика обученной модели: не схлопнулись ли эмбеддинги и насколько
рекомендации ENMF разнообразнее популярности.

    python scripts/diagnose_run.py --run runs/enmf_ml1m [--write]

С флагом --write результат сохраняется в <run>/diagnostics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enmf.data.prepare import prepare_data
from enmf.evaluate import build_model_from_checkpoint, load_run
from enmf.models.popularity import PopularityRecommender
from enmf.utils import setup_logging


def effective_rank(matrix: torch.Tensor, threshold: float = 0.01) -> int:
    """Число сингулярных чисел больше threshold * максимального."""
    singular = torch.linalg.svdvals(matrix)
    return int((singular > threshold * singular[0]).sum())


def unique_top_items(model, users: list[int], train_by_user: dict, k: int) -> int:
    """Сколько разных объектов попадает в top-k хотя бы одного пользователя."""
    with torch.no_grad():
        scores = model.full_scores(torch.tensor(users)).clone()
    for row, user in enumerate(users):
        scores[row, torch.as_tensor(train_by_user[user])] = -float("inf")
    return int(scores.topk(k, dim=1).indices.unique().numel())


def diagnose(run_dir: Path, k: int = 10) -> dict:
    config, checkpoint = load_run(run_dir)
    model = build_model_from_checkpoint(config, checkpoint, torch.device("cpu"))
    data = prepare_data(config.data)
    users = data.eval_users
    popularity = PopularityRecommender.fit(data.train["item_id"].to_numpy(), data.n_items)

    with torch.no_grad():
        scores = model.full_scores(torch.tensor(users))
    item_counts = popularity.item_counts.double().numpy()
    mean_item_score = scores.mean(dim=0).double().numpy()

    return {
        "run": str(run_dir),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "embedding_dim": model.embedding_dim,
        "h_norm": float(model.h.norm()),
        "h_positive_components": int((model.h > 0).sum()),
        "user_embedding_median_norm": float(model.user_embedding.weight.norm(dim=1).median()),
        "item_embedding_median_norm": float(model.item_matrix.norm(dim=1).median()),
        "user_embedding_effective_rank": effective_rank(model.user_embedding.weight.detach()),
        "item_embedding_effective_rank": effective_rank(model.item_matrix.detach()),
        "score_std_over_catalogue_median": float(scores.std(dim=1).median()),
        "n_items": data.n_items,
        f"unique_items_in_top{k}_enmf": unique_top_items(model, users, data.train_by_user, k),
        f"unique_items_in_top{k}_popularity": unique_top_items(
            popularity, users, data.train_by_user, k
        ),
        "corr_mean_item_score_vs_popularity": float(
            np.corrcoef(mean_item_score, item_counts)[0, 1]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Диагностика обученной модели ENMF")
    parser.add_argument("--run", required=True, help="каталог запуска")
    parser.add_argument("--k", type=int, default=10, help="K для покрытия каталога")
    parser.add_argument("--write", action="store_true", help="сохранить в <run>/diagnostics.json")
    args = parser.parse_args()

    setup_logging()
    result = diagnose(Path(args.run), k=args.k)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.write:
        (Path(args.run) / "diagnostics.json").write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
