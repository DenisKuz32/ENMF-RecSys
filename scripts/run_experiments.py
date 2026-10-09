#!/usr/bin/env python3
"""Прогон серии экспериментов и сборка таблицы результатов.

    python scripts/run_experiments.py --config configs/ml1m.yaml --grid default
    python scripts/run_experiments.py --config configs/ml1m.yaml --grid neg_weight

Каждый запуск сохраняет свои артефакты в отдельный каталог, а сводная таблица
с гиперпараметрами и метриками пишется в reports/experiments.md и
reports/experiments.json -- это и есть требуемая "таблица запусков".
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enmf.config import Config
from enmf.train import train
from enmf.utils import setup_logging

logger = logging.getLogger(__name__)


# Сетки параметров. neg_weight вынесен в отдельную сетку, так как это главный
# гиперпараметр ENMF: его оптимум зависит от данных, поэтому он перебирается в
# широком диапазоне. Сетка доходит до 1.0 (равные веса наблюдённых и пропущенных пар); значение
# по умолчанию у авторов для ML-1M -- 0.5.
GRIDS: dict[str, dict[str, list]] = {
    "default": {
        "loss.neg_weight": [0.01, 0.05, 0.1, 0.2, 0.5],
        "model.embedding_dim": [32, 64],
        "model.dropout": [0.0, 0.3, 0.5],
    },
    "neg_weight": {
        "loss.neg_weight": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0],
    },
    "embedding_dim": {
        "model.embedding_dim": [16, 32, 64, 128, 256],
    },
    "dropout": {
        "model.dropout": [0.0, 0.1, 0.3, 0.5, 0.7],
    },
    "quick": {
        "loss.neg_weight": [0.05, 0.1, 0.2],
    },
}


def set_nested(config: Config, dotted_key: str, value) -> None:
    section_name, param_name = dotted_key.split(".", 1)
    setattr(getattr(config, section_name), param_name, value)


def slugify(values: dict[str, object]) -> str:
    parts = [f"{key.split('.')[-1]}{value}" for key, value in values.items()]
    return "_".join(parts).replace(".", "")


def run_grid(base_config_path: str, grid_name: str, output_path: Path) -> list[dict]:
    grid = GRIDS[grid_name]
    keys = list(grid)
    combinations = list(itertools.product(*(grid[key] for key in keys)))
    logger.info("сетка '%s': %d запусков", grid_name, len(combinations))

    rows: list[dict] = []
    for index, combination in enumerate(combinations, start=1):
        values = dict(zip(keys, combination))
        config = Config.load(base_config_path)
        for key, value in values.items():
            set_nested(config, key, value)
        config.output.run_name = f"{Path(base_config_path).stem}_{grid_name}_{slugify(values)}"
        config.output.tensorboard = True

        logger.info("[%d/%d] %s", index, len(combinations), values)
        started = time.perf_counter()
        summary = train(config)
        elapsed = time.perf_counter() - started

        row = {key.split(".")[-1]: value for key, value in values.items()}
        row["run_name"] = config.output.run_name
        row["best_epoch"] = summary["best_epoch"]
        row["seconds"] = round(elapsed, 1)
        row.update({f"enmf_{k}": round(v, 4) for k, v in summary["best_metrics"].items()})
        row.update(
            {f"pop_{k}": round(v, 4) for k, v in summary["baseline_popularity"].items()}
        )
        if summary.get("baseline_item_knn"):
            row["knn_ndcg@10"] = round(summary["baseline_item_knn"]["ndcg@10"], 4)
        rows.append(row)

        write_results(rows, output_path)  # пишем после каждого запуска
    return rows


def write_results(rows: list[dict], output_path: Path) -> None:
    """Сохраняет таблицу в markdown и json.

    Запись после каждого запуска, а не в конце: длинную серию может прервать
    что угодно, и терять уже посчитанное не хочется.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    if not rows:
        return
    columns = list(rows[0])
    lines = [
        "# Таблица экспериментов",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in sorted(rows, key=lambda item: -item.get("enmf_ndcg@10", 0)):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")

    with open(output_path.with_suffix(".md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Серия экспериментов ENMF")
    parser.add_argument("--config", required=True, help="базовый конфиг")
    parser.add_argument(
        "--grid", default="quick", choices=sorted(GRIDS), help="какая сетка параметров"
    )
    parser.add_argument(
        "--output", default="reports/experiments", help="путь к таблице без расширения"
    )
    args = parser.parse_args()

    setup_logging()
    rows = run_grid(args.config, args.grid, Path(args.output))

    best = max(rows, key=lambda row: row.get("enmf_ndcg@10", 0))
    logger.info("лучший запуск: %s (ndcg@10=%.4f)", best["run_name"], best["enmf_ndcg@10"])
    logger.info("таблица сохранена: %s.md / %s.json", args.output, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
