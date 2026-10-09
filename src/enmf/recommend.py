"""Получение Top-K рекомендаций для конкретного пользователя.

Небольшой модуль инференса поверх обученной модели: принимает исходный id
пользователя, возвращает список рекомендованных объектов с названиями (для
MovieLens) и скорами. Уже просмотренные объекты исключаются.

Холодный пользователь -- тот, у кого нет взаимодействий в обучающей части
(например, при global-разбиении он впервые появился после точки отсечения).
Его эмбеддинг ни разу не обновлялся и остался случайным с инициализации,
поэтому скоры ENMF для него бессмысленны. Для такого пользователя
рекомендации строятся по популярности, а в ответе это явно помечено полем
`source`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .data.movielens import load_item_titles
from .data.prepare import prepare_data
from .evaluate import build_model_from_checkpoint, load_run
from .models.popularity import PopularityRecommender
from .utils import resolve_device

logger = logging.getLogger(__name__)


def recommend_for_user(
    run_dir: str | Path,
    user: int,
    k: int = 10,
    raw_index: bool = False,
    device: str = "auto",
) -> dict:
    """Возвращает Top-K рекомендаций для пользователя.

    Args:
        run_dir: каталог запуска с конфигом и чекпоинтом.
        user: идентификатор пользователя. По умолчанию трактуется как исходный id
            из датасета; с `raw_index=True` -- как внутренний индекс модели.
        k: число рекомендаций.
        raw_index: см. выше.

    Returns:
        Словарь с историей пользователя и списком рекомендаций.
    """
    config, checkpoint = load_run(run_dir)
    resolved_device = resolve_device(device if device != "auto" else config.train.device)

    data = prepare_data(config.data)
    model = build_model_from_checkpoint(config, checkpoint, resolved_device)

    if raw_index:
        user_index = int(user)
    else:
        if user not in data.user_map:
            raise KeyError(
                f"пользователь с исходным id={user} отсутствует в датасете "
                f"(возможно, отфильтрован k-core)"
            )
        user_index = data.user_map[user]

    if not 0 <= user_index < data.n_users:
        raise IndexError(f"индекс пользователя вне диапазона: {user_index}")

    inverse_item_map = {int(index): int(raw) for raw, index in data.item_map.items()}
    titles = None
    if config.data.dataset != "synthetic":
        titles = load_item_titles(config.data.dataset, config.data.data_dir)

    seen = data.train_by_user.get(user_index)
    is_cold = seen is None or len(seen) == 0
    if is_cold:
        logger.warning(
            "пользователь %s холодный: нет обучающей истории, эмбеддинг не обучен -- "
            "рекомендации по популярности",
            user,
        )
        recommender = PopularityRecommender.fit(
            data.train["item_id"].to_numpy(), n_items=data.n_items
        ).to(resolved_device)
    else:
        recommender = model

    with torch.no_grad():
        user_tensor = torch.tensor([user_index], dtype=torch.long, device=resolved_device)
        scores = recommender.full_scores(user_tensor).squeeze(0).clone()

        if not is_cold:
            scores[torch.as_tensor(seen, device=resolved_device)] = float("-inf")

        top_scores, top_indices = torch.topk(scores, k=min(k, data.n_items))

    def describe(item_index: int) -> dict:
        raw_id = inverse_item_map.get(int(item_index))
        entry = {"item_index": int(item_index), "item_id": raw_id}
        if titles is not None and raw_id in titles:
            entry["title"] = titles[raw_id]
        return entry

    history = [describe(index) for index in (seen[-20:] if seen is not None else [])]
    recommendations = [
        describe(index) | {"score": float(score)}
        for index, score in zip(top_indices.tolist(), top_scores.tolist())
    ]

    return {
        "user_index": user_index,
        "user_id": None if raw_index else int(user),
        "source": "popularity" if is_cold else "enmf",
        "history_size": len(seen) if seen is not None else 0,
        "recent_history": history,
        "recommendations": recommendations,
    }
