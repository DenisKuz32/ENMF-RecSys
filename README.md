# ENMF: рекомендации на неявной обратной связи без негативного сэмплирования

Реализация на PyTorch алгоритма **ENMF** (Efficient Neural Matrix Factorization
without Sampling) для задачи Top-K рекомендаций на данных с неявной обратной
связью.

> Chen C., Zhang M., Zhang Y., Liu Y., Ma S. *Efficient Neural Matrix
> Factorization without Sampling for Recommendation.* ACM TOIS 38(2), 2020.
> <https://doi.org/10.1145/3373807>

Модель кодирует пользователей и объекты в векторы фиксированной размерности и
предсказывает релевантность как `y(u, i) = hᵀ (p_u ⊙ q_i)`. Ключевая
особенность подхода — обучение на **всей** матрице взаимодействий, включая все
пропущенные пары, без негативного сэмплирования: за счёт переформулировки лосса
через две матрицы Грама `d × d` сложность падает с `O(|U|·|I|·d)` до
`O(|R|·d + (|U| + |I|)·d²)`.

## Результаты

MovieLens-1M (6 040 пользователей, 3 416 фильмов, 999 611 взаимодействий),
разбиение по времени с единой точкой отсечения, полное ранжирование каталога,
1 143 оцениваемых пользователя:

| Модель | NDCG@5 | NDCG@10 | NDCG@20 | Recall@10 |
|---|---|---|---|---|
| Популярность | 0.2795 | 0.2675 | 0.2522 | 0.0370 |
| item-kNN | 0.2980 | 0.2765 | 0.2569 | 0.0391 |
| **ENMF** | **0.3038** | **0.2845** | **0.2629** | **0.0467** |

ENMF лучше популярности на 6.4% и item-kNN на 2.9% по NDCG@10; на трёх зёрнах
NDCG@10 = 0.2835 ± 0.0023. Полный отчёт по архитектуре, постановке и
результатам — [`reports/report.md`](reports/report.md); таблицы запусков с
гиперпараметрами — `reports/experiments*.md`, `reports/item_knn_grid*.md`.

---

## Установка

```bash
git clone <адрес репозитория>
cd enmf-recsys

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# CPU-сборка PyTorch: около 0.9 ГБ на диске вместо 3+ ГБ с CUDA-библиотеками
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .                 # ставит пакет enmf и команду `enmf`
```

Проверено на Python 3.12 (WSL2 Ubuntu). GPU не требуется — всё работает на CPU.

Первая команда нужна только без GPU. Без неё `pip install -r requirements.txt`
на Linux ставит сборку PyTorch с CUDA и тянет несколько гигабайт
библиотек NVIDIA; в WSL с ограниченной памятью такая установка может уронить
виртуальную машину. Если GPU есть, первую команду пропустите.

После `pip install -e .` все команды запускаются как `enmf <команда>`, а DDP —
как `torchrun ... -m enmf.cli ...`. Без установки пакета работает
альтернативный способ — указать путь к исходникам явно:

```bash
PYTHONPATH=src python -m enmf.cli train --config configs/synthetic.yaml
PYTHONPATH=src torchrun --nproc_per_node=2 -m enmf.cli train --config configs/ml1m.yaml
```

## Быстрый старт

Прогон на синтетических данных, без скачивания датасета (около 10 секунд):

```bash
enmf train --config configs/synthetic.yaml
```

Основной эксперимент на MovieLens-1M (датасет скачается автоматически при
первом запуске):

```bash
enmf train --config configs/ml1m.yaml
```

Разбиение train/validation — по времени с единой точкой отсечения
(`split_strategy: global`): ни одно обучающее взаимодействие не позже
валидационных. Дополнительный эксперимент с разбиением по времени внутри
каждого пользователя:

```bash
enmf train --config configs/ml1m.yaml \
    --set data.split_strategy=per_user output.run_name=enmf_ml1m_per_user
```

По завершении в `runs/enmf_ml1m/` появятся: `config.yaml` (точная конфигурация
запуска), `best_model.pt`, `results.json`, `split_stats.json` и логи
TensorBoard.

## Распределённое обучение (DDP)

Никаких дополнительных флагов не нужно — режим определяется по запуску через
`torchrun`. Backend выбирается автоматически: `nccl` при наличии CUDA, `gloo`
на CPU и Intel XPU.

```bash
# 4 процесса
torchrun --nproc_per_node=4 -m enmf.cli train --config configs/ml1m.yaml

# то же через скрипт-обёртку
./scripts/run_train_ddp.sh 4 configs/ml1m.yaml
```

Корректность распределённого режима проверяется тестом: градиенты DDP на двух
процессах с половинами батча совпадают с градиентами одиночного процесса на
полном батче (`tests/test_ddp.py`).

## TensorBoard

```bash
tensorboard --logdir runs/
```

Логируются: лосс и время эпохи, все метрики валидации по эпохам, метрики
бейзлайнов (популярность и item-kNN) и гиперпараметры запуска (`add_hparams`).

## Остальные команды

```bash
# оценка сохранённой модели без переобучения
enmf evaluate --run runs/enmf_ml1m

# Top-K рекомендаций для пользователя (по исходному id из датасета)
enmf recommend --run runs/enmf_ml1m --user 1000 --k 10
# пользователь без обучающей истории (при global-разбиении это, например,
# id 1-634) получит рекомендации по популярности: поле "source": "popularity"

# сравнение стратегий временного разбиения
enmf compare-splits --config configs/ml1m.yaml

# серия экспериментов с автоматической таблицей результатов
python scripts/run_experiments.py --config configs/ml1m.yaml --grid neg_weight

# готовые markdown-таблицы для отчёта из артефактов запусков
python scripts/make_report_tables.py --run runs/enmf_ml1m --write

# подбор настроек бейзлайна item-kNN по валидации
python scripts/tune_item_knn.py --config configs/ml1m.yaml

# диагностика модели: ранг эмбеддингов, разнообразие топ-10
python scripts/diagnose_run.py --run runs/enmf_ml1m
```

Любой параметр конфига переопределяется из командной строки:

```bash
enmf train --config configs/ml1m.yaml \
    --set loss.neg_weight=0.05 model.embedding_dim=128 output.run_name=dim128
```

## Воспроизведение результатов отчёта

Все прогоны выполняются на CPU; основной занимает около 1.5 минуты, весь набор —
около 40 минут. Зерно фиксировано, повторный запуск даёт те же метрики.

```bash
# 1. основной запуск и устойчивость к зерну (отчёт, 4.1–4.2)
enmf train --config configs/ml1m.yaml
enmf train --config configs/ml1m.yaml --set train.seed=1 output.run_name=enmf_ml1m_seed1
enmf train --config configs/ml1m.yaml --set train.seed=2 output.run_name=enmf_ml1m_seed2

# 2. дополнительный эксперимент: разбиение внутри пользователя (4.5)
enmf train --config configs/ml1m.yaml \
    --set data.split_strategy=per_user output.run_name=enmf_ml1m_per_user

# 3. масштабирование DDP при одинаковом числе потоков на процесс (4.6)
OMP_NUM_THREADS=2 enmf train --config configs/ml1m.yaml --set output.run_name=enmf_ddp1
OMP_NUM_THREADS=2 torchrun --nproc_per_node=2 -m enmf.cli train \
    --config configs/ml1m.yaml --set output.run_name=enmf_ddp2
OMP_NUM_THREADS=2 torchrun --nproc_per_node=4 --master_port=29501 -m enmf.cli train \
    --config configs/ml1m.yaml --set output.run_name=enmf_ddp4

# 4. сетки гиперпараметров и подбор бейзлайна item-kNN (4.3, 4.4, 4.7)
python scripts/run_experiments.py --config configs/ml1m.yaml --grid neg_weight
python scripts/run_experiments.py --config configs/ml1m.yaml --grid embedding_dim \
    --output reports/experiments_dim
python scripts/tune_item_knn.py --config configs/ml1m.yaml
python scripts/tune_item_knn.py --config configs/ml1m.yaml \
    --set data.split_strategy=per_user --output reports/item_knn_grid_per_user

# 5. таблицы и диагностика для отчёта
enmf compare-splits --config configs/ml1m.yaml > reports/splits.json
python scripts/make_report_tables.py --run runs/enmf_ml1m \
    --splits reports/splits.json \
    --scaling-runs runs/enmf_ddp1 runs/enmf_ddp2 runs/enmf_ddp4 --write
python scripts/diagnose_run.py --run runs/enmf_ml1m --write
```

Числа в `reports/report.md` взяты из `runs/*/results.json`, `reports/*.json` и
`runs/enmf_ml1m/diagnostics.json`.

В репозитории сохранены итоговые запуски, на которых построен отчёт:
`runs/enmf_ml1m` (с обученной моделью `best_model.pt`), `runs/enmf_ml1m_per_user`,
`runs/enmf_ddp1`, `runs/enmf_ddp2`, `runs/enmf_ddp4` — конфиг, метрики,
статистика разбиения и логи TensorBoard. Промежуточные прогоны сеток и зёрен в
репозиторий не входят, их итоги — в `reports/`.

## Тесты

```bash
python -m pytest              # все тесты
python -m pytest -m "not slow"  # без многопроцессных DDP-тестов
```

В первую очередь покрыты места, где ошибка не проявляется явно и обучение
внешне выглядит исправным:

| Что проверяется | Файл |
|---|---|
| Эффективный лосс совпадает с наивным по значению и по градиентам | `tests/test_loss.py` |
| NDCG/Recall/HitRate против посчитанных вручную значений | `tests/test_metrics.py` |
| Градиенты DDP совпадают с одиночным процессом | `tests/test_ddp.py` |
| Отсутствие утечки будущего во временном разбиении | `tests/test_split.py` |
| Чтение CSV-датасетов MovieLens (заголовок, типы, порядок по времени) | `tests/test_movielens.py` |
| В DDP датасет скачивает только ранг 0; проверка условия user-based | `tests/test_prepare.py` |
| Цикл обучения: история, ранняя остановка, валидный `results.json` | `tests/test_train.py` |
| Бейзлайн item-kNN: похожести против расчёта вручную | `tests/test_item_knn.py` |
| Рекомендации тёплому и холодному пользователю | `tests/test_recommend.py` |
| Точка входа `enmf` и команды CLI | `tests/test_cli.py` |
| Исходники и итоговые артефакты не попадают под `.gitignore` | `tests/test_repository.py` |

## Структура проекта

```
enmf-recsys/
├── configs/                 конфигурации экспериментов (YAML)
│   ├── ml1m.yaml            основной эксперимент
│   ├── ml_small.yaml        быстрый прогон на ml-latest-small
│   └── synthetic.yaml       синтетика, без обращения к сети
├── src/enmf/
│   ├── config.py            дата-классы конфигурации, загрузка и переопределения
│   ├── losses.py            non-sampling лосс (вывод формулы в докстринге),
│   │                        наивная версия для сверки, обёртка для DDP
│   ├── metrics.py           NDCG@K, Recall@K, HitRate@K, полное ранжирование
│   ├── train.py             цикл обучения, DDP, TensorBoard, ранняя остановка
│   ├── evaluate.py          оценка сохранённого запуска
│   ├── recommend.py         Top-K рекомендации для пользователя
│   ├── cli.py               командный интерфейс
│   ├── utils.py             сиды, логирование, устройства, помощники DDP
│   ├── data/
│   │   ├── movielens.py     загрузка, бинаризация, k-core, перенумерация
│   │   ├── split.py         временное разбиение и его диагностика
│   │   ├── dataset.py       датасет историй, паддинг, генератор синтетики
│   │   └── prepare.py       единый конвейер подготовки данных
│   └── models/
│       ├── enmf.py          модель
│       ├── popularity.py    бейзлайн по популярности
│       └── item_knn.py      бейзлайн item-kNN
├── scripts/                 серии экспериментов, подбор item-kNN, таблицы и
│                            диагностика для отчёта, пример расчёта лосса
├── tests/
├── reports/
│   ├── report.md            отчёт по архитектуре, постановке и результатам
│   ├── tables.md            таблицы, собранные из артефактов запусков
│   ├── experiments*.md      таблицы запусков с гиперпараметрами и метриками
│   └── item_knn_grid*.md    подбор настроек бейзлайна item-kNN
└── runs/                    итоговые запуски: конфиги, метрики, логи TensorBoard, модель
```

## Конфигурация

Основные параметры (`configs/ml1m.yaml`):

| Параметр | По умолчанию | Смысл |
|---|---|---|
| `loss.neg_weight` | 1.0 | вес пропущенных взаимодействий; главный гиперпараметр ENMF, подобран по сетке (отчёт, раздел 4.3) |
| `loss.pos_weight` | 1.0 | вес наблюдённых взаимодействий |
| `model.embedding_dim` | 64 | размерность векторных представлений |
| `model.dropout` | 0.3 | dropout на эмбеддинге пользователя |
| `data.split_strategy` | `global` | стратегия временного разбиения: `global` — единая точка отсечения по времени, `per_user` — своя у каждого пользователя; см. отчёт, раздел 3.2 |
| `data.val_ratio` | 0.2 | доля взаимодействий в валидации |
| `train.lr` | 0.05 | шаг Adagrad |
| `eval.item_knn*` | вкл., сумма, α = 20000 | второй бейзлайн item-kNN и его настройки (отчёт, раздел 4.7) |

## Поддерживаемые датасеты

`ml-latest-small`, `ml-1m`, `ml-10m`, `ml-20m`, `ml-25m` — скачиваются
автоматически; `synthetic` — генерируется локально.

Размерность выбирается под вычислительные ресурсы: `ml-latest-small` для
быстрой отладки, `ml-1m` для основного эксперимента.
