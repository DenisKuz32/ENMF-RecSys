"""Метрики качества Top-K рекомендаций.

Протокол оценки
---------------
Для каждого пользователя ранжируется **весь каталог** объектов, за вычетом тех,
с которыми он уже взаимодействовал в обучающей части. Полное ранжирование, в
отличие от популярной схемы "один позитив против 100 случайных негативов", не
зависит от способа сэмплирования негативов и не завышает метрики; это же
рекомендуют и авторы ENMF.

NDCG@K при нескольких релевантных объектах
------------------------------------------
Временное разбиение оставляет пользователю не один, а несколько отложенных
взаимодействий, поэтому используется общая формула:

    DCG@K  = sum_{j=1..K} rel_j / log2(j + 1),  rel_j in {0, 1}
    IDCG@K = sum_{j=1..min(K, R)} 1 / log2(j + 1),  R -- число релевантных
    NDCG@K = DCG@K / IDCG@K

Нормировка на min(K, R), а не на R, обязательна: иначе у пользователя с числом
релевантных объектов больше K максимально достижимое значение метрики будет
меньше единицы, и средняя метрика начнёт зависеть от распределения активности
пользователей, а не от качества модели.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .utils import DistributedContext, all_reduce_sum


@dataclass
class MetricAccumulator:
    """Накопитель сумм метрик по пользователям (устойчив к батчам и к DDP)."""

    ks: tuple[int, ...]
    sums: dict[str, float] = field(default_factory=dict)
    n_users: int = 0

    def __post_init__(self) -> None:
        if not self.sums:
            self.sums = {
                f"{name}@{k}": 0.0
                for k in self.ks
                for name in ("ndcg", "recall", "hit_rate")
            }

    def update(self, batch_sums: dict[str, torch.Tensor], n_users: int) -> None:
        for key, value in batch_sums.items():
            self.sums[key] += float(value)
        self.n_users += int(n_users)

    def compute(self) -> dict[str, float]:
        if self.n_users == 0:
            return {key: 0.0 for key in self.sums}
        return {key: value / self.n_users for key, value in self.sums.items()}

    def as_tensor(self, device: torch.device) -> torch.Tensor:
        """Упаковка в один тензор для all_reduce в DDP."""
        ordered = [self.sums[key] for key in sorted(self.sums)]
        return torch.tensor(ordered + [float(self.n_users)], device=device, dtype=torch.float64)

    def load_tensor(self, tensor: torch.Tensor) -> None:
        keys = sorted(self.sums)
        values = tensor.tolist()
        for key, value in zip(keys, values[:-1]):
            self.sums[key] = value
        self.n_users = int(values[-1])


def _discounts(max_k: int, device: torch.device) -> torch.Tensor:
    """Вектор дисконтов 1 / log2(j + 1) для позиций j = 1..max_k."""
    positions = torch.arange(1, max_k + 1, device=device, dtype=torch.float64)
    return 1.0 / torch.log2(positions + 1.0)


def ranking_metrics_batch(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    ks: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Считает суммы метрик по батчу пользователей.

    Args:
        scores: FloatTensor [B, n_items] -- скоры с уже замаскированными
            обучающими взаимодействиями (маскированные позиции = -inf).
        relevance: FloatTensor [B, n_items] -- бинарная матрица релевантности
            по валидационной части.
        ks: значения K.

    Returns:
        Словарь "метрика@K" -> сумма по пользователям батча (скалярный тензор).
    """
    max_k = max(ks)
    max_k = min(max_k, scores.size(1))
    device = scores.device

    _, top_indices = torch.topk(scores, k=max_k, dim=1)
    top_relevance = relevance.gather(1, top_indices).to(torch.float64)  # [B, max_k]

    discounts = _discounts(max_k, device)                       # [max_k]
    gains = top_relevance * discounts.unsqueeze(0)              # [B, max_k]
    cumulative_gain = gains.cumsum(dim=1)                       # DCG@j для всех j
    ideal_cumulative = discounts.cumsum(dim=0)                  # IDCG@j для всех j

    n_relevant = relevance.sum(dim=1).to(torch.float64)         # [B]
    hits_cumulative = top_relevance.cumsum(dim=1)               # [B, max_k]

    results: dict[str, torch.Tensor] = {}
    for k in ks:
        effective_k = min(k, max_k)
        position = effective_k - 1

        dcg = cumulative_gain[:, position]
        ideal_count = torch.clamp(n_relevant, max=float(effective_k)).long()
        # ideal_count >= 1 всегда: пользователи без релевантных объектов в
        # валидацию не попадают, но clamp защищает от деления на ноль.
        idcg = ideal_cumulative[torch.clamp(ideal_count - 1, min=0)]
        ndcg = torch.where(n_relevant > 0, dcg / idcg, torch.zeros_like(dcg))

        hits = hits_cumulative[:, position]
        recall = torch.where(
            n_relevant > 0, hits / n_relevant, torch.zeros_like(hits)
        )
        hit_rate = (hits > 0).to(torch.float64)

        results[f"ndcg@{k}"] = ndcg.sum()
        results[f"recall@{k}"] = recall.sum()
        results[f"hit_rate@{k}"] = hit_rate.sum()
    return results


def build_mask_and_relevance(
    users: torch.Tensor,
    train_by_user: dict[int, np.ndarray],
    val_by_user: dict[int, np.ndarray],
    n_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Строит маску обучающих взаимодействий и матрицу релевантности.

    Returns:
        seen_mask: BoolTensor [B, n_items] -- True там, где объект уже был в train.
        relevance: FloatTensor [B, n_items] -- 1.0 для отложенных взаимодействий.
    """
    batch_size = users.size(0)
    seen_mask = torch.zeros((batch_size, n_items), dtype=torch.bool, device=device)
    relevance = torch.zeros((batch_size, n_items), dtype=torch.float64, device=device)

    for row, user in enumerate(users.tolist()):
        seen = train_by_user.get(user)
        if seen is not None and len(seen):
            seen_mask[row, torch.as_tensor(seen, device=device)] = True
        positives = val_by_user.get(user)
        if positives is not None and len(positives):
            relevance[row, torch.as_tensor(positives, device=device)] = 1.0
    return seen_mask, relevance


@torch.no_grad()
def evaluate_model(
    model,
    users: list[int],
    train_by_user: dict[int, np.ndarray],
    val_by_user: dict[int, np.ndarray],
    n_items: int,
    ks: tuple[int, ...] = (5, 10, 20),
    batch_size: int = 512,
    device: torch.device | str = "cpu",
    context: DistributedContext | None = None,
) -> dict[str, float]:
    """Полная оценка модели по списку пользователей.

    `model` должен предоставлять метод `full_scores(users) -> [B, n_items]`.
    Этому контракту удовлетворяют и ENMF, и популярностный бейзлайн, поэтому
    обе модели оцениваются одним и тем же кодом -- сравнение получается
    честным по построению.

    В режиме DDP (`context.enabled`) каждый процесс считает суммы метрик по
    своей доле пользователей, затем суммы и счётчик складываются через
    all_reduce. Такой порядок даёт результат, в точности совпадающий с одиночным
    процессом, -- усреднение средних по процессам такого свойства не имеет,
    если доли разного размера.
    """
    device = torch.device(device)
    distributed = context is not None and context.enabled
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    shard = users[context.rank :: context.world_size] if distributed else users
    accumulator = MetricAccumulator(ks=tuple(ks))
    for start in range(0, len(shard), batch_size):
        chunk = shard[start : start + batch_size]
        user_tensor = torch.as_tensor(chunk, dtype=torch.long, device=device)

        scores = model.full_scores(user_tensor).to(torch.float64)
        seen_mask, relevance = build_mask_and_relevance(
            user_tensor, train_by_user, val_by_user, n_items, device
        )
        scores = scores.masked_fill(seen_mask, float("-inf"))

        batch_sums = ranking_metrics_batch(scores, relevance, tuple(ks))
        accumulator.update(batch_sums, len(chunk))

    if distributed:
        packed = accumulator.as_tensor(device)
        all_reduce_sum(packed, context)
        accumulator.load_tensor(packed)

    if was_training and hasattr(model, "train"):
        model.train()
    return accumulator.compute()
