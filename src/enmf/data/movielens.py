"""Загрузка и предобработка датасетов MovieLens в формат неявной обратной связи.

Поддерживаются ml-1m, ml-10m, ml-20m, ml-25m и ml-latest-small. Датасет
скачивается с files.grouplens.org при первом обращении и кэшируется.

Явные оценки приводятся к бинарным взаимодействиям: по умолчанию сам факт
оценки считается положительным сигналом (порог 0), что является стандартной
постановкой implicit feedback для MovieLens в литературе по рекомендациям,
включая исходную статью ENMF. Альтернативу (например, "положительно" = оценка
не ниже 4) можно включить параметром `binarize_threshold`.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

logger = logging.getLogger(__name__)

DATASET_URLS: dict[str, str] = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

# Внутри архивов разные версии используют разные имена и форматы файлов.
# Заголовок: 0 -- первая строка файла является заголовком и заменяется на наши
# имена колонок; None -- заголовка нет (формат .dat). Значение "infer" здесь
# использовать нельзя: вместе с `names` pandas трактует его как header=None, и
# строка заголовка CSV читается как строка данных.
_RATINGS_SPEC: dict[str, tuple[str, str, int | None]] = {
    # name -> (относительный путь, разделитель, строка заголовка)
    "ml-latest-small": ("ml-latest-small/ratings.csv", ",", 0),
    "ml-1m": ("ml-1m/ratings.dat", "::", None),
    "ml-10m": ("ml-10M100K/ratings.dat", "::", None),
    "ml-20m": ("ml-20m/ratings.csv", ",", 0),
    "ml-25m": ("ml-25m/ratings.csv", ",", 0),
}

COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def _encoding(separator: str) -> str:
    """CSV-версии MovieLens записаны в UTF-8, старые .dat -- в latin-1."""
    return "utf-8" if separator == "," else "latin-1"


@dataclass
class InteractionData:
    """Взаимодействия после предобработки и перенумерации индексов."""

    interactions: pd.DataFrame  # колонки: user_id, item_id, timestamp (индексы 0..n-1)
    n_users: int
    n_items: int
    user_map: dict[int, int]    # исходный id -> внутренний индекс
    item_map: dict[int, int]

    @property
    def n_interactions(self) -> int:
        return len(self.interactions)

    @property
    def density(self) -> float:
        return self.n_interactions / (self.n_users * self.n_items)

    def describe(self) -> str:
        return (
            f"пользователей: {self.n_users:,}  "
            f"объектов: {self.n_items:,}  "
            f"взаимодействий: {self.n_interactions:,}  "
            f"плотность: {self.density:.4%}"
        )


def download_dataset(name: str, data_dir: str | Path) -> Path:
    """Скачивает и распаковывает архив датасета, если его ещё нет локально."""
    if name not in DATASET_URLS:
        raise ValueError(
            f"неизвестный датасет '{name}'; доступны: {sorted(DATASET_URLS)}"
        )
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    relative_path, _, _ = _RATINGS_SPEC[name]
    ratings_path = data_dir / relative_path
    if ratings_path.exists():
        logger.info("датасет %s уже загружен: %s", name, ratings_path)
        return ratings_path

    archive_path = data_dir / f"{name}.zip"
    if not archive_path.exists():
        url = DATASET_URLS[name]
        logger.info("скачиваю %s -> %s", url, archive_path)
        # Сначала во временный файл, потом переименование: прерванное
        # скачивание не оставит битый архив под "правильным" именем.
        partial_path = archive_path.with_suffix(".zip.part")
        urlretrieve(url, partial_path)
        partial_path.replace(archive_path)

    logger.info("распаковываю %s", archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(data_dir)

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"после распаковки не найден файл с оценками: {ratings_path}"
        )
    return ratings_path


def read_ratings(name: str, data_dir: str | Path) -> pd.DataFrame:
    """Читает сырые оценки в DataFrame с колонками COLUMNS."""
    ratings_path = download_dataset(name, data_dir)
    _, separator, header = _RATINGS_SPEC[name]

    frame = pd.read_csv(
        ratings_path,
        sep=separator,
        header=header,
        names=COLUMNS,
        engine="python" if separator == "::" else "c",
        encoding=_encoding(separator),
    )
    # Явное приведение типов: если формат файла когда-нибудь снова разойдётся с
    # ожиданиями, лучше упасть здесь, чем молча сортировать время как строки.
    return frame[COLUMNS].astype(
        {"user_id": "int64", "item_id": "int64", "rating": "float64", "timestamp": "int64"}
    )


def binarize(frame: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Оставляет только взаимодействия с оценкой >= threshold и убирает оценку."""
    if threshold > 0:
        frame = frame[frame["rating"] >= threshold]
    return frame.drop(columns=["rating"]).reset_index(drop=True)


def apply_k_core(
    frame: pd.DataFrame, min_user_interactions: int = 5, min_item_interactions: int = 5
) -> pd.DataFrame:
    """Итеративная k-core фильтрация по пользователям и объектам.

    Одного прохода недостаточно: удаление редких объектов может сделать
    пользователя редким и наоборот, поэтому фильтрация повторяется до
    стабилизации.
    """
    if min_user_interactions <= 1 and min_item_interactions <= 1:
        return frame.reset_index(drop=True)

    while True:
        size_before = len(frame)
        if min_item_interactions > 1:
            item_counts = frame["item_id"].value_counts()
            keep_items = item_counts[item_counts >= min_item_interactions].index
            frame = frame[frame["item_id"].isin(keep_items)]
        if min_user_interactions > 1:
            user_counts = frame["user_id"].value_counts()
            keep_users = user_counts[user_counts >= min_user_interactions].index
            frame = frame[frame["user_id"].isin(keep_users)]
        if len(frame) == size_before:
            break
    return frame.reset_index(drop=True)


def reindex(frame: pd.DataFrame) -> InteractionData:
    """Перенумеровывает исходные id в плотные индексы 0..n-1."""
    unique_users = sorted(frame["user_id"].unique())
    unique_items = sorted(frame["item_id"].unique())
    user_map = {raw: idx for idx, raw in enumerate(unique_users)}
    item_map = {raw: idx for idx, raw in enumerate(unique_items)}

    frame = frame.copy()
    frame["user_id"] = frame["user_id"].map(user_map).astype("int64")
    frame["item_id"] = frame["item_id"].map(item_map).astype("int64")
    frame = frame.sort_values(["timestamp", "user_id"], kind="stable").reset_index(
        drop=True
    )

    return InteractionData(
        interactions=frame,
        n_users=len(user_map),
        n_items=len(item_map),
        user_map=user_map,
        item_map=item_map,
    )


def load_interactions(
    name: str,
    data_dir: str | Path,
    binarize_threshold: float = 0.0,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    deduplicate: bool = True,
) -> InteractionData:
    """Полный конвейер: загрузка -> бинаризация -> k-core -> перенумерация."""
    frame = read_ratings(name, data_dir)
    logger.info("прочитано сырых оценок: %d", len(frame))

    frame = binarize(frame, binarize_threshold)
    if deduplicate:
        # При дублях пара (u, i) оставляется с самым ранним временем: именно оно
        # определяет момент первого взаимодействия для временного разбиения.
        frame = (
            frame.sort_values("timestamp", kind="stable")
            .drop_duplicates(subset=["user_id", "item_id"], keep="first")
            .reset_index(drop=True)
        )
    frame = apply_k_core(frame, min_user_interactions, min_item_interactions)

    data = reindex(frame)
    logger.info("после предобработки -- %s", data.describe())
    return data


def load_item_titles(name: str, data_dir: str | Path) -> dict[int, str] | None:
    """Названия фильмов по исходному id (для человекочитаемых рекомендаций)."""
    paths = {
        "ml-latest-small": ("ml-latest-small/movies.csv", ",", 0),
        "ml-1m": ("ml-1m/movies.dat", "::", None),
        "ml-10m": ("ml-10M100K/movies.dat", "::", None),
        "ml-20m": ("ml-20m/movies.csv", ",", 0),
        "ml-25m": ("ml-25m/movies.csv", ",", 0),
    }
    if name not in paths:
        return None
    relative_path, separator, header = paths[name]
    full_path = Path(data_dir) / relative_path
    if not full_path.exists():
        return None

    frame = pd.read_csv(
        full_path,
        sep=separator,
        header=header,
        names=["item_id", "title", "genres"],
        engine="python" if separator == "::" else "c",
        encoding=_encoding(separator),
    )
    return dict(zip(frame["item_id"].astype(int), frame["title"].astype(str)))
