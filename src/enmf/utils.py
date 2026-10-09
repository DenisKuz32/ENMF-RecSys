"""Вспомогательные утилиты: воспроизводимость, логирование, устройства, DDP."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist

# ---------------------------------------------------------------------- #
# Воспроизводимость и логирование
# ---------------------------------------------------------------------- #


def set_seed(seed: int, rank: int = 0) -> None:
    """Фиксирует генераторы случайных чисел.

    К зерну прибавляется ранг процесса: одинаковое зерно на всех процессах DDP
    сделало бы dropout-маски идентичными, что незаметно снижает эффективный
    уровень регуляризации.
    """
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def setup_logging(rank: int = 0, level: int = logging.INFO) -> None:
    """Настраивает логирование; на не-нулевых рангах остаются только warning."""
    logging.basicConfig(
        level=level if rank == 0 else logging.WARNING,
        format=f"[%(asctime)s][rank{rank}][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


# ---------------------------------------------------------------------- #
# Устройства
# ---------------------------------------------------------------------- #


def resolve_device(requested: str = "auto") -> torch.device:
    """Выбирает устройство для вычислений.

    Поддержан и Intel XPU (встроенная графика Arc) на случай, если окружение
    настроено; при `auto` и отсутствии ускорителя честно возвращается CPU.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def resolve_backend(requested: str, device: torch.device) -> str:
    """Выбирает backend для распределённого обучения.

    NCCL работает только с CUDA; во всех остальных случаях (CPU, XPU) берётся
    gloo, который поддерживает и CPU-тензоры.
    """
    if requested != "auto":
        return requested
    return "nccl" if device.type == "cuda" else "gloo"


# ---------------------------------------------------------------------- #
# DDP
# ---------------------------------------------------------------------- #


@dataclass
class DistributedContext:
    """Состояние распределённого окружения."""

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    enabled: bool = False
    backend: str = "gloo"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(device: torch.device, backend: str = "auto") -> DistributedContext:
    """Инициализирует process group, если процесс запущен через torchrun.

    Наличие переменных окружения RANK/WORLD_SIZE -- признак запуска под
    torchrun. При обычном запуске возвращается одиночный контекст, и весь
    остальной код работает без изменений.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistributedContext(backend=resolve_backend(backend, device))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world_size == 1:
        return DistributedContext(
            rank=0, local_rank=0, world_size=1, enabled=False,
            backend=resolve_backend(backend, device),
        )

    resolved_backend = resolve_backend(backend, device)
    dist.init_process_group(backend=resolved_backend, rank=rank, world_size=world_size)
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    # Сообщение об инициализации пишет train() после setup_logging(rank):
    # здесь логирование ещё настроено как для ранга 0.
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        enabled=True,
        backend=resolved_backend,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    """Ждёт, пока все процессы дойдут до этой точки (в одиночном режиме -- no-op)."""
    if context.enabled and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Суммирует тензор по всем процессам (на месте)."""
    if context.enabled and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics(metrics: dict[str, float]) -> str:
    return "  ".join(f"{name}={value:.4f}" for name, value in sorted(metrics.items()))
