"""Командный интерфейс проекта.

    python -m enmf.cli train         --config configs/ml1m.yaml
    python -m enmf.cli evaluate      --run runs/enmf_ml1m
    python -m enmf.cli recommend     --run runs/enmf_ml1m --user 1000 --k 10
    python -m enmf.cli compare-splits --config configs/ml1m.yaml

Режим DDP включается запуском через torchrun, дополнительных флагов не требуется:

    torchrun --nproc_per_node=4 -m enmf.cli train --config configs/ml1m.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .utils import setup_logging


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True, help="путь к YAML-конфигу эксперимента"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        metavar="СЕКЦИЯ.ПАРАМЕТР=ЗНАЧЕНИЕ",
        help="переопределение параметров конфига, например loss.neg_weight=0.05",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enmf",
        description="ENMF: рекомендации на неявной обратной связи без сэмплирования",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="обучить модель")
    _add_config_arguments(train_parser)

    eval_parser = subparsers.add_parser(
        "evaluate", help="оценить сохранённую модель и бейзлайн"
    )
    eval_parser.add_argument("--run", required=True, help="каталог запуска")

    rec_parser = subparsers.add_parser(
        "recommend", help="получить Top-K рекомендаций для пользователя"
    )
    rec_parser.add_argument("--run", required=True, help="каталог запуска")
    rec_parser.add_argument("--user", type=int, required=True, help="исходный id пользователя")
    rec_parser.add_argument("--k", type=int, default=10, help="число рекомендаций")
    rec_parser.add_argument(
        "--raw-index", action="store_true",
        help="трактовать --user как внутренний индекс, а не исходный id",
    )

    split_parser = subparsers.add_parser(
        "compare-splits",
        help="сравнить стратегии временного разбиения (материал для отчёта)",
    )
    _add_config_arguments(split_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging()

    if args.command == "train":
        from .train import train

        config = Config.load(args.config).apply_overrides(args.overrides)
        train(config)
        return 0

    if args.command == "evaluate":
        from .evaluate import evaluate_run

        summary = evaluate_run(args.run)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "recommend":
        from .recommend import recommend_for_user

        result = recommend_for_user(
            run_dir=args.run, user=args.user, k=args.k, raw_index=args.raw_index
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "compare-splits":
        from .data.dataset import make_synthetic_interactions
        from .data.movielens import load_interactions, reindex
        from .data.split import compare_strategies

        config = Config.load(args.config).apply_overrides(args.overrides)
        if config.data.dataset == "synthetic":
            data = reindex(make_synthetic_interactions(seed=0))
        else:
            data = load_interactions(
                name=config.data.dataset,
                data_dir=config.data.data_dir,
                binarize_threshold=config.data.binarize_threshold,
                min_user_interactions=config.data.min_user_interactions,
                min_item_interactions=config.data.min_item_interactions,
            )
        stats = compare_strategies(data.interactions, val_ratio=config.data.val_ratio)
        print(
            json.dumps(
                {name: stat.to_dict() for name, stat in stats.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parser.error(f"неизвестная команда: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
