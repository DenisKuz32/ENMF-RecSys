"""Проверка чтения датасетов MovieLens.

Сетевого доступа тесты не требуют: файлы собираются во временной папке в том
же виде, в каком лежат внутри архивов, и `download_dataset` находит их на
месте.
"""

from __future__ import annotations

from pathlib import Path

from enmf.data.movielens import load_interactions, load_item_titles, read_ratings

# Метки времени подобраны так, что строковый порядок отличается от числового:
# '1000000000' < '964982703' < '99' как строки, но 99 < 964982703 < 1000000000.
RATINGS_CSV = (
    "userId,movieId,rating,timestamp\n"
    "1,10,4.0,964982703\n"
    "1,20,3.5,1000000000\n"
    "2,10,5.0,99\n"
)

MOVIES_CSV = (
    "movieId,title,genres\n"
    "10,GoldenEye (1995),Action|Adventure|Thriller\n"
    '20,"American President, The (1995)",Comedy|Drama|Romance\n'
)


def _write(directory: Path, relative: str, content: str) -> None:
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_csv_header_is_not_read_as_data(tmp_path: Path) -> None:
    """Строка заголовка CSV не должна попадать в данные, типы -- числовые."""
    _write(tmp_path, "ml-latest-small/ratings.csv", RATINGS_CSV)

    frame = read_ratings("ml-latest-small", tmp_path)

    assert len(frame) == 3
    assert frame["user_id"].dtype == "int64"
    assert frame["item_id"].dtype == "int64"
    assert frame["timestamp"].dtype == "int64"
    assert frame["rating"].dtype == "float64"
    assert frame.iloc[0].tolist() == [1, 10, 4.0, 964982703]


def test_csv_interactions_are_ordered_by_numeric_time(tmp_path: Path) -> None:
    """Порядок по времени числовой, а не лексикографический."""
    _write(tmp_path, "ml-latest-small/ratings.csv", RATINGS_CSV)

    data = load_interactions(
        "ml-latest-small", tmp_path, min_user_interactions=1, min_item_interactions=1
    )

    assert data.interactions["timestamp"].tolist() == [99, 964982703, 1000000000]
    assert (data.n_users, data.n_items) == (2, 2)


def test_csv_item_titles(tmp_path: Path) -> None:
    """Названия читаются без строки заголовка, запятая в кавычках не ломает разбор."""
    _write(tmp_path, "ml-latest-small/movies.csv", MOVIES_CSV)

    titles = load_item_titles("ml-latest-small", tmp_path)

    assert titles == {
        10: "GoldenEye (1995)",
        20: "American President, The (1995)",
    }
