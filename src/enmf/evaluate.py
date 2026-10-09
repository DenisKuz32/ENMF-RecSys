"""Оценка сохранённой модели без переобучения.

Загружает конфиг и веса из каталога запуска, воспроизводит ровно то же
разбиение данных (оно детерминировано по времени, без случайности) и считает
метрики ENMF и бейзлайна по популярности.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .config import Config
from .data.prepare import prepare_data
from .metrics import evaluate_model
from .models.enmf import ENMF
from .models.item_knn import ItemKNNRecommender
from .models.popularity import PopularityRecommender
from .utils import format_metrics, resolve_device

logger = logging.getLogger(__name__)


def load_run(run_dir: str | Path) -> tuple[Config, dict]:
    """Читает конфиг и чекпоинт лучшей модели из каталога запуска."""
    run_path = Path(run_dir)
    config_path = run_path / "config.yaml"
    checkpoint_path = run_path / "best_model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"не найден конфиг запуска: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"не найден чекпоинт: {checkpoint_path}")

    config = Config.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return config, checkpoint


def build_model_from_checkpoint(
    config: Config, checkpoint: dict, device: torch.device
) -> ENMF:
    model = ENMF(
        n_users=checkpoint["n_users"],
        n_items=checkpoint["n_items"],
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
        init_std=config.model.init_std,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def evaluate_run(run_dir: str | Path, device: str = "auto") -> dict:
    """Считает метрики ENMF и популярностного бейзлайна для сохранённого запуска."""
    config, checkpoint = load_run(run_dir)
    resolved_device = resolve_device(device if device != "auto" else config.train.device)

    data = prepare_data(config.data)
    model = build_model_from_checkpoint(config, checkpoint, resolved_device)
    ks = tuple(config.eval.ks)

    model_metrics = evaluate_model(
        model, data.eval_users, data.train_by_user, data.val_by_user,
        data.n_items, ks=ks, batch_size=config.eval.batch_size, device=resolved_device,
    )
    baseline = PopularityRecommender.fit(
        data.train["item_id"].to_numpy(), n_items=data.n_items
    ).to(resolved_device)
    baseline_metrics = evaluate_model(
        baseline, data.eval_users, data.train_by_user, data.val_by_user,
        data.n_items, ks=ks, batch_size=config.eval.batch_size, device=resolved_device,
    )

    knn_metrics = None
    if config.eval.item_knn:
        knn = ItemKNNRecommender.fit(
            data.train_by_user, data.n_users, data.n_items,
            shrinkage=config.eval.item_knn_shrinkage,
            aggregation=config.eval.item_knn_aggregation,
            top_k=config.eval.item_knn_top_k,
        ).to(resolved_device)
        knn_metrics = evaluate_model(
            knn, data.eval_users, data.train_by_user, data.val_by_user,
            data.n_items, ks=ks, batch_size=config.eval.batch_size, device=resolved_device,
        )

    logger.info("ENMF:         %s", format_metrics(model_metrics))
    logger.info("популярность: %s", format_metrics(baseline_metrics))
    if knn_metrics is not None:
        logger.info("item-kNN:     %s", format_metrics(knn_metrics))

    summary = {
        "run_dir": str(run_dir),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "enmf": model_metrics,
        "popularity": baseline_metrics,
        "improvement_percent": {
            name: ((model_metrics[name] / value - 1.0) * 100.0) if value > 0 else None
            for name, value in baseline_metrics.items()
        },
        "item_knn": knn_metrics,
        "improvement_vs_item_knn_percent": (
            {
                name: ((model_metrics[name] / value - 1.0) * 100.0) if value > 0 else None
                for name, value in knn_metrics.items()
            }
            if knn_metrics is not None else None
        ),
        "split_stats": data.split_stats.to_dict(),
    }

    output_path = Path(run_dir) / "evaluation.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
