#!/usr/bin/env python3
"""Подбор настроек бейзлайна item-kNN по валидации.

    python scripts/tune_item_knn.py --config configs/ml1m.yaml
    python scripts/tune_item_knn.py --config configs/ml1m.yaml \\
        --set data.split_strategy=per_user --output reports/item_knn_grid_per_user

Бейзлайн подбирается так же, как ENMF: перебор по сетке, выбор по NDCG@10 на
той же валидации. Модель не обучается, поэтому один вариант считается за
секунды. Таблица пишется в reports/item_knn_grid.md и .json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enmf.config import Config
from enmf.data.prepare import prepare_data
from enmf.metrics import evaluate_model
from enmf.models.item_knn import ItemKNNRecommender
from enmf.utils import setup_logging

AGGREGATIONS = ["sum", "topk_mean"]
SHRINKAGES = [0.0, 10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Сетка настроек item-kNN")
    parser.add_argument("--config", required=True, help="конфиг эксперимента")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None)
    parser.add_argument("--output", default="reports/item_knn_grid", help="путь без расширения")
    args = parser.parse_args()

    setup_logging()
    config = Config.load(args.config).apply_overrides(args.overrides)
    data = prepare_data(config.data)
    ks = tuple(config.eval.ks)

    rows = []
    for aggregation in AGGREGATIONS:
        for shrinkage in SHRINKAGES:
            started = time.perf_counter()
            knn = ItemKNNRecommender.fit(
                data.train_by_user, data.n_users, data.n_items,
                shrinkage=shrinkage, aggregation=aggregation,
                top_k=config.eval.item_knn_top_k,
            )
            metrics = evaluate_model(
                knn, data.eval_users, data.train_by_user, data.val_by_user,
                data.n_items, ks=ks, batch_size=config.eval.batch_size,
            )
            row = {"aggregation": aggregation, "shrinkage": shrinkage}
            row.update({name: round(value, 4) for name, value in metrics.items()})
            row["seconds"] = round(time.perf_counter() - started, 1)
            rows.append(row)
            print(f"{aggregation:9s} α={shrinkage:7.0f}  ndcg@10={metrics['ndcg@10']:.4f}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = ["aggregation", "shrinkage", "ndcg@5", "ndcg@10", "ndcg@20", "recall@10",
               "hit_rate@10", "seconds"]
    lines = [
        f"# Сетка item-kNN ({config.data.dataset}, split={config.data.split_strategy})",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in sorted(rows, key=lambda item: -item["ndcg@10"]):
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = max(rows, key=lambda item: item["ndcg@10"])
    print(f"лучший: aggregation={best['aggregation']} shrinkage={best['shrinkage']} "
          f"ndcg@10={best['ndcg@10']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
