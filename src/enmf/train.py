"""Цикл обучения ENMF с поддержкой DDP и логированием в TensorBoard.

Запуск в один процесс:
    python -m enmf.cli train --config configs/ml1m.yaml

Запуск в режиме DDP на 4 процессах (CPU или GPU -- backend выбирается сам):
    torchrun --nproc_per_node=4 -m enmf.cli train --config configs/ml1m.yaml

Про корректность DDP для этого лосса
------------------------------------
Оба слагаемых non-sampling лосса пропорциональны числу пользователей в батче:
первое явно, второе -- через матрицу Грама пользователей. Лосс делится на размер
батча, поэтому DDP-усреднение градиентов между процессами эквивалентно обучению
одним процессом на объединённом батче. Матрица Грама объектов Q^T Q считается
целиком на каждом процессе -- параметры в DDP реплицированы, так что расхождения
это не создаёт, а стоит всего O(|I| * d^2) на шаг.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import Config
from .data.dataset import make_collate_fn
from .data.prepare import PreparedData, ensure_dataset, prepare_data
from .losses import ENMFTrainingModule
from .metrics import evaluate_model
from .models.enmf import ENMF
from .models.item_knn import ItemKNNRecommender
from .models.popularity import PopularityRecommender
from .utils import (
    DistributedContext,
    all_reduce_sum,
    cleanup_distributed,
    count_parameters,
    format_metrics,
    init_distributed,
    resolve_device,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Оценка
# ---------------------------------------------------------------------- #


def evaluate_distributed(
    model,
    data: PreparedData,
    ks: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    context: DistributedContext,
) -> dict[str, float]:
    """Оценка по пользователям валидации; в DDP пользователи делятся между процессами."""
    return evaluate_model(
        model, data.eval_users, data.train_by_user, data.val_by_user, data.n_items,
        ks=ks, batch_size=batch_size, device=device, context=context,
    )


def evaluate_baselines(
    config: Config,
    data: PreparedData,
    ks: tuple[int, ...],
    device: torch.device,
    context: DistributedContext,
) -> dict[str, dict[str, float]]:
    """Метрики бейзлайнов, посчитанные тем же кодом, что и для ENMF.

    Возвращает {"popularity": ..., "item_knn": ...}; item-kNN -- если он
    включён в конфиге.
    """
    batch_size = config.eval.batch_size
    baselines = {
        "popularity": PopularityRecommender.fit(
            data.train["item_id"].to_numpy(), n_items=data.n_items
        ).to(device)
    }
    if config.eval.item_knn:
        baselines["item_knn"] = ItemKNNRecommender.fit(
            data.train_by_user, data.n_users, data.n_items,
            shrinkage=config.eval.item_knn_shrinkage,
            aggregation=config.eval.item_knn_aggregation,
            top_k=config.eval.item_knn_top_k,
        ).to(device)
    return {
        name: evaluate_distributed(model, data, ks, batch_size, device, context)
        for name, model in baselines.items()
    }


def relative_improvement(
    metrics: dict[str, float], baseline: dict[str, float]
) -> dict[str, float | None]:
    """Относительный прирост к бейзлайну: 0.15 означает +15%.

    При нулевой метрике бейзлайна прирост не определён -- возвращается None
    (в JSON это null; NaN json.dump записал бы как невалидный JSON).
    """
    return {
        name: (metrics.get(name, 0.0) / value - 1.0) if value > 0 else None
        for name, value in baseline.items()
    }


def train_one_epoch(
    trainer: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    context: DistributedContext,
) -> float:
    """Одна эпоха обучения. Возвращает средний лосс на пользователя.

    `trainer` -- это `ENMFTrainingModule` (возможно, обёрнутый в DDP), и лосс
    считается именно вызовом `trainer(...)`, а не обращением к модели напрямую:
    только так DDP успевает подготовить синхронизацию градиентов.
    """
    trainer.train()
    total_loss = torch.zeros(1, dtype=torch.float64, device=device)
    total_users = torch.zeros(1, dtype=torch.float64, device=device)

    for users, histories in loader:
        users = users.to(device, non_blocking=True)
        histories = histories.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = trainer(users, histories)
        loss.backward()
        if config.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.train.grad_clip)
        optimizer.step()

        total_loss += loss.detach().to(torch.float64) * users.size(0)
        total_users += users.size(0)

    all_reduce_sum(total_loss, context)
    all_reduce_sum(total_users, context)
    return float(total_loss.item() / max(total_users.item(), 1.0))


def build_dataloader(
    data: PreparedData, config: Config, context: DistributedContext
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler: DistributedSampler | None = None
    if context.enabled:
        sampler = DistributedSampler(
            data.dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            drop_last=False,
        )
    loader = DataLoader(
        data.dataset,
        batch_size=config.train.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.train.num_workers,
        collate_fn=make_collate_fn(data.n_items),
        drop_last=False,
    )
    return loader, sampler


def train(config: Config) -> dict:
    """Полный сценарий обучения: данные, модель, цикл, оценка, сохранение."""
    device = resolve_device(config.train.device)
    context = init_distributed(device, config.train.ddp_backend)
    setup_logging(context.rank)
    set_seed(config.train.seed, context.rank)

    if context.is_main:
        logger.info(
            "устройство=%s, процессов=%d, DDP=%s (backend=%s)",
            device, context.world_size, "да" if context.enabled else "нет", context.backend,
        )

    ensure_dataset(config.data, context)
    data = prepare_data(config.data)

    # ---- выходные артефакты ---------------------------------------- #
    output_dir = Path(config.output.output_dir) / config.output.run_name
    writer = None
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        config.save(output_dir / "config.yaml")
        with open(output_dir / "split_stats.json", "w", encoding="utf-8") as handle:
            json.dump(config.data.__dict__ | data.split_stats.to_dict(), handle,
                      ensure_ascii=False, indent=2)
        if config.output.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    ks = tuple(config.eval.ks)

    # ---- бейзлайны: популярность и item-kNN -------------------------- #
    baselines = evaluate_baselines(config, data, ks, device, context)
    baseline_metrics = baselines["popularity"]
    knn_metrics = baselines.get("item_knn")
    if context.is_main:
        for name, metrics in baselines.items():
            logger.info("%-12s %s", name + ":", format_metrics(metrics))
            if writer is not None:
                for metric, value in metrics.items():
                    writer.add_scalar(f"baseline_{name}/{metric}", value, 0)

    # ---- модель ----------------------------------------------------- #
    model = ENMF(
        n_users=data.n_users,
        n_items=data.n_items,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
        init_std=config.model.init_std,
    ).to(device)
    if context.is_main:
        logger.info("обучаемых параметров: %s", f"{count_parameters(model):,}")

    trainer = ENMFTrainingModule(
        model,
        pos_weight=config.loss.pos_weight,
        neg_weight=config.loss.neg_weight,
        l2_reg=config.loss.l2_reg,
    ).to(device)

    if context.enabled:
        device_ids = [context.local_rank] if device.type == "cuda" else None
        trainer = DistributedDataParallel(trainer, device_ids=device_ids)

    optimizer = torch.optim.Adagrad(
        trainer.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    loader, sampler = build_dataloader(data, config, context)

    # ---- цикл ------------------------------------------------------- #
    best_score = -float("inf")
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    epochs_without_improvement = 0
    history: list[dict] = []
    target_metric = config.train.early_stopping_metric

    for epoch in range(1, config.train.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        started = time.perf_counter()
        train_loss = train_one_epoch(trainer, loader, optimizer, config, device, context)
        train_seconds = time.perf_counter() - started

        record = {"epoch": epoch, "train_loss": train_loss, "train_seconds": train_seconds}

        if epoch % config.train.eval_every == 0 or epoch == config.train.epochs:
            eval_started = time.perf_counter()
            # Оценка идёт по самой модели ENMF, а не по обучающей обёртке.
            metrics = evaluate_distributed(
                model, data, ks, config.eval.batch_size, device, context
            )
            record["eval_seconds"] = time.perf_counter() - eval_started
            record.update(metrics)

            score = metrics.get(target_metric, -float("inf"))
            improved = score > best_score
            if improved:
                best_score, best_epoch, best_metrics = score, epoch, dict(metrics)
                epochs_without_improvement = 0
                if context.is_main and config.output.save_checkpoint:
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "config": config.to_dict(),
                            "epoch": epoch,
                            "metrics": metrics,
                            "n_users": data.n_users,
                            "n_items": data.n_items,
                        },
                        output_dir / "best_model.pt",
                    )
            else:
                epochs_without_improvement += 1

            if context.is_main:
                logger.info(
                    "эпоха %3d | loss=%.4f | %s | %.1fs%s",
                    epoch, train_loss, format_metrics(metrics), train_seconds,
                    "  <- лучшая" if improved else "",
                )
                if writer is not None:
                    writer.add_scalar("train/loss", train_loss, epoch)
                    writer.add_scalar("train/epoch_seconds", train_seconds, epoch)
                    for name, value in metrics.items():
                        writer.add_scalar(f"validation/{name}", value, epoch)

        else:
            if context.is_main:
                logger.info("эпоха %3d | loss=%.4f | %.1fs", epoch, train_loss, train_seconds)
                if writer is not None:
                    writer.add_scalar("train/loss", train_loss, epoch)

        # Запись в историю -- до проверки ранней остановки, иначе эпоха, на
        # которой обучение остановилось, в историю не попадёт.
        history.append(record)

        if (
            config.train.early_stopping_patience > 0
            and epochs_without_improvement >= config.train.early_stopping_patience
        ):
            if context.is_main:
                logger.info(
                    "ранняя остановка на эпохе %d: %s не растёт %d эпох подряд",
                    epoch, target_metric, epochs_without_improvement,
                )
            break

    # ---- итог ------------------------------------------------------- #
    summary = {
        "run_name": config.output.run_name,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "baseline_popularity": baseline_metrics,
        "improvement": relative_improvement(best_metrics, baseline_metrics),
        "baseline_item_knn": knn_metrics,
        "improvement_vs_item_knn": (
            relative_improvement(best_metrics, knn_metrics) if knn_metrics else None
        ),
        "split_stats": data.split_stats.to_dict(),
        "config": config.to_dict(),
        "history": history,
        "world_size": context.world_size,
        "device": str(device),
    }

    if context.is_main:
        with open(output_dir / "results.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        if writer is not None:
            writer.add_hparams(
                config.flat_hparams(),
                {f"best/{name}": value for name, value in best_metrics.items()},
            )
            writer.close()

        logger.info("=" * 72)
        logger.info("лучшая эпоха: %d", best_epoch)
        logger.info("ENMF:         %s", format_metrics(best_metrics))
        for baseline_name, metrics in baselines.items():
            logger.info("%-12s %s", baseline_name + ":", format_metrics(metrics))
        for name in sorted(best_metrics):
            gains = [
                f"{baseline_name} {(best_metrics[name] / metrics[name] - 1.0) * 100:+.1f}%"
                for baseline_name, metrics in baselines.items()
                if metrics.get(name, 0.0) > 0
            ]
            logger.info("  %-12s прирост ENMF: %s", name, ", ".join(gains))
        logger.info("артефакты: %s", output_dir)

    cleanup_distributed(context)
    return summary
