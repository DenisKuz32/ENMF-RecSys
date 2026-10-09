#!/usr/bin/env python3
"""Сборка таблиц для отчёта из артефактов запусков.

    python scripts/make_report_tables.py --run runs/enmf_ml1m

Печатает готовые markdown-таблицы (характеристики данных, сравнение с
бейзлайном, сравнение стратегий разбиения, масштабирование DDP), которые
используются в reports/report.md.

С флагом --write таблицы дополнительно сохраняются в reports/tables.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def table_dataset(split_stats: dict) -> str:
    lines = [
        "### Характеристики данных после предобработки",
        "",
        "| Характеристика | Значение |",
        "| --- | --- |",
        f"| Пользователей | {split_stats['n_users_total']:,} |",
        f"| Пользователей в train | {split_stats['n_users_in_train']:,} |",
        f"| Объектов в train | {split_stats['n_items_in_train']:,} |",
        f"| Взаимодействий в train | {split_stats['n_train']:,} |",
        f"| Взаимодействий в validation | {split_stats['n_val_kept']:,} |",
        (
            f"| Оцениваемых пользователей | {split_stats['n_val_users_kept']:,} "
            f"из {split_stats['n_val_users_raw']:,} "
            f"({split_stats['val_user_coverage']:.1%}) |"
        ),
    ]
    return "\n".join(lines)


def _gain(model: float, base: float) -> str:
    return f"{(model / base - 1.0) * 100:+.1f}%" if base > 0 else "—"


def table_comparison(results: dict) -> str:
    enmf = results["best_metrics"]
    popularity = results["baseline_popularity"]
    knn = results.get("baseline_item_knn")
    order = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@5", "recall@10", "recall@20",
             "hit_rate@5", "hit_rate@10", "hit_rate@20"]

    if knn:
        header = [
            "### Сравнение с бейзлайнами",
            "",
            (
                "| Метрика | Популярность | item-kNN | ENMF | Прирост к популярности "
                "| Прирост к item-kNN |"
            ),
            "| --- | --- | --- | --- | --- | --- |",
        ]
    else:
        header = [
            "### Сравнение с бейзлайном по популярности",
            "",
            "| Метрика | Популярность | ENMF | Прирост |",
            "| --- | --- | --- | --- |",
        ]
    lines = header
    for name in order:
        if name not in enmf:
            continue
        base, model = popularity.get(name, 0.0), enmf[name]
        if knn:
            other = knn.get(name, 0.0)
            lines.append(
                f"| {name.upper()} | {base:.4f} | {other:.4f} | {model:.4f} "
                f"| {_gain(model, base)} | {_gain(model, other)} |"
            )
        else:
            lines.append(f"| {name.upper()} | {base:.4f} | {model:.4f} | {_gain(model, base)} |")
    lines += [
        "",
        (
            f"Лучшая эпоха: {results['best_epoch']}. "
            f"Процессов: {results.get('world_size', 1)}. "
            f"Устройство: {results.get('device', 'cpu')}."
        ),
    ]
    return "\n".join(lines)


def table_splits(comparison: dict) -> str:
    lines = [
        "### Сравнение стратегий временного разбиения",
        "",
        "| Стратегия | Взаимодействий в train | Валидация | Оцениваемых пользователей |",
        "| --- | --- | --- | --- |",
    ]
    for name, stats in comparison.items():
        lines.append(
            f"| `{name}` | {stats['n_train']:,} | {stats['n_val_kept']:,} | "
            f"{stats['n_val_users_kept']:,} из {stats['n_val_users_raw']:,} "
            f"({stats['val_user_coverage']:.1%}) |"
        )
    return "\n".join(lines)


def table_scaling(run_dirs: list[Path]) -> str:
    """Таблица масштабирования DDP по нескольким каталогам запусков."""
    rows = []
    for run_dir in run_dirs:
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
        results = load_json(results_path)
        history = results.get("history", [])
        epoch_times = [record["train_seconds"] for record in history if "train_seconds" in record]
        if not epoch_times:
            continue
        median = sorted(epoch_times)[len(epoch_times) // 2]
        rows.append(
            {
                "world_size": results.get("world_size", 1),
                "seconds": median,
                "ndcg": results["best_metrics"].get("ndcg@10", float("nan")),
                "run": run_dir.name,
            }
        )
    if not rows:
        return "### Масштабирование DDP\n\n*Нет данных: запустите обучение на 1, 2 и 4 процессах.*"

    rows.sort(key=lambda row: row["world_size"])
    baseline = rows[0]["seconds"]
    lines = [
        "### Масштабирование DDP",
        "",
        "| Процессов | Секунд на эпоху (медиана) | Ускорение | NDCG@10 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        speedup = baseline / row["seconds"] if row["seconds"] > 0 else float("nan")
        lines.append(
            f"| {row['world_size']} | {row['seconds']:.2f} | {speedup:.2f}× | {row['ndcg']:.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Таблицы для отчёта")
    parser.add_argument("--run", required=True, help="основной каталог запуска")
    parser.add_argument(
        "--scaling-runs", nargs="*", default=None,
        help="каталоги запусков с разным числом процессов для таблицы масштабирования",
    )
    parser.add_argument(
        "--splits", default=None,
        help="json с выводом команды compare-splits",
    )
    parser.add_argument("--write", action="store_true", help="сохранить в reports/tables.md")
    args = parser.parse_args()

    run_dir = Path(args.run)
    results = load_json(run_dir / "results.json")

    blocks = [
        table_dataset(results["split_stats"]),
        table_comparison(results),
    ]
    if args.splits:
        blocks.append(table_splits(load_json(Path(args.splits))))
    if args.scaling_runs:
        blocks.append(table_scaling([Path(path) for path in args.scaling_runs]))

    output = "\n\n".join(blocks)
    print(output)

    if args.write:
        target = Path("reports/tables.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
        print(f"\n[сохранено в {target}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
