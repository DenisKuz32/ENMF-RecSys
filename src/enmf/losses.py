"""Функция потерь ENMF: обучение на всех данных без негативного сэмплинга.

ВЫВОД ФОРМУЛЫ
=============

Взвешенный квадратичный лосс по *всем* парам (пользователь, объект), где
наблюдённые взаимодействия -- это единицы с весом c+, а все пропущенные --
нули с весом c- (c- обычно много меньше c+):

    L = sum_u sum_{i in I} c_ui * (y_hat(u,i) - y(u,i))^2

Прямое вычисление стоит O(|U| * |I| * d) -- для реального каталога это
неподъёмно, поэтому обычно прибегают к негативному сэмплированию. ENMF вместо
этого переписывает сумму так, чтобы её можно было посчитать точно и дёшево.

Шаг 1. Разделяем наблюдённую и пропущенную части и добавляем/вычитаем
наблюдённую часть с отрицательным весом, чтобы вторая сумма шла по всему
каталогу:

    L = sum_u sum_{i in R_u} [ c+ (y_hat - 1)^2 - c- y_hat^2 ]
      + sum_u sum_{i in I}   c- y_hat^2

Первое слагаемое идёт только по наблюдённым взаимодействиям -- это O(|R| * d).

Шаг 2. Раскрываем второе слагаемое. Поскольку y_hat(u,i) = sum_k h_k p_uk q_ik,

    y_hat(u,i)^2 = sum_k sum_l h_k h_l p_uk p_ul q_ik q_il

и суммирование по u и i расщепляется:

    sum_u sum_i c- y_hat^2
        = c- * sum_{k,l} [ h_k h_l * (sum_u p_uk p_ul) * (sum_i q_ik q_il) ]
        = c- * sum( (h h^T) * (P^T P) * (Q^T Q) )        # * поэлементно

То есть вместо перебора |U| * |I| пар достаточно двух матриц Грама размера
d x d. По всем данным сразу это стоит O(|R| * d + (|U| + |I|) * d^2) вместо
O(|U| * |I| * d). Это и есть основной результат статьи. В мини-батчевом
обучении здесь шаг стоит O(|R_B| * d + (B + |I|) * d^2) вместо O(B * |I| * d):
Q^T Q пересчитывается на каждом шаге, потому что параметры меняются.

Шаг 3. Раскрываем квадрат в первом слагаемом:

    c+ (y_hat - 1)^2 - c- y_hat^2 = (c+ - c-) y_hat^2 - 2 c+ y_hat + c+

Константа c+ * |R| не зависит от параметров и на градиенты не влияет, поэтому в
реализации она отброшена. Побочный полезный эффект: для паддинговых позиций
y_hat тождественно равен нулю (паддинговый эмбеддинг объекта нулевой), и их
вклад в оставшееся выражение тоже равен нулю -- маска не нужна.

ВАЖНО ПРО СРАВНЕНИЕ С НАИВНОЙ ВЕРСИЕЙ
--------------------------------------
`enmf_loss_naive` возвращает лосс *с* этой константой, поэтому при сверке её
надо вычесть -- см. `tests/test_loss.py`.

РЕЖИМ DDP
---------
Оба слагаемых пропорциональны размеру батча пользователей: первое явно, второе
-- через матрицу Грама P_B^T P_B, которая суммируется по пользователям батча.
Поэтому деление на размер батча даёт корректный "средний лосс на пользователя",
и усреднение градиентов между процессами DDP эквивалентно обучению на общем
батче. Матрица Грама объектов Q^T Q при этом считается на каждом процессе
целиком -- это допустимо, так как DDP реплицирует параметры, и стоит всего
O(|I| * d^2) на шаг.
"""

from __future__ import annotations

import torch
from torch import nn

from .models.enmf import ENMF


def enmf_loss(
    model: ENMF,
    users: torch.Tensor,
    histories: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 0.1,
    l2_reg: float = 0.0,
) -> torch.Tensor:
    """Эффективный non-sampling лосс ENMF (user-based вариант, ENMF-U).

    Args:
        model: модель ENMF.
        users: LongTensor [B] -- индексы пользователей батча.
        histories: LongTensor [B, L] -- истории взаимодействий, дополненные
            паддинговым индексом `model.n_items`.
        pos_weight: вес наблюдённых взаимодействий c+.
        neg_weight: вес пропущенных взаимодействий c- (ключевой гиперпараметр).
        l2_reg: коэффициент L2-регуляризации.

    Returns:
        Скаляр -- средний лосс на одного пользователя батча (без аддитивной
        константы c+ * |R_batch|).
    """
    user_repr = model.user_repr(users)                          # [B, d]
    pos_scores = model.score_items(user_repr, histories)        # [B, L]

    # Слагаемое по наблюдённым взаимодействиям. Паддинг даёт ровно 0.
    observed = (
        (pos_weight - neg_weight) * pos_scores.pow(2) - 2.0 * pos_weight * pos_scores
    ).sum()

    # Слагаемое по всему каталогу через две матрицы Грама d x d.
    gram_users = user_repr.t() @ user_repr                      # [d, d]
    item_matrix = model.item_matrix                             # [n_items, d]
    gram_items = item_matrix.t() @ item_matrix                  # [d, d]
    gram_h = torch.outer(model.h, model.h)                      # [d, d]
    all_pairs = neg_weight * (gram_h * gram_users * gram_items).sum()

    loss = (observed + all_pairs) / users.size(0)
    if l2_reg > 0.0:
        loss = loss + l2_reg * model.l2_norm() / users.size(0)
    return loss


def enmf_loss_naive(
    model: ENMF,
    users: torch.Tensor,
    histories: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 0.1,
    l2_reg: float = 0.0,
    drop_constant: bool = True,
) -> torch.Tensor:
    """Прямое (наивное) вычисление того же лосса по полной матрице [B, n_items].

    Нужна исключительно для проверки корректности `enmf_loss`: она в точности
    реализует определение из шага 0 вывода, без каких-либо преобразований.
    Стоимость O(B * |I| * d), для обучения непригодна.

    Args:
        drop_constant: если True, вычитается аддитивная константа
            `pos_weight * |R_batch|`, чтобы результат совпадал с `enmf_loss`.
    """
    user_repr = model.user_repr(users)                          # [B, d]
    scores = (user_repr * model.h) @ model.item_matrix.t()      # [B, n_items]

    # Метки строим в матрице шириной n_items + 1: паддинговый индекс попадает в
    # лишний столбец, который затем отбрасывается. Так исключается риск затереть
    # настоящий позитив нулём при совпадении индексов.
    labels = scores.new_zeros(scores.size(0), model.n_items + 1)
    labels.scatter_(1, histories, 1.0)
    labels = labels[:, : model.n_items].contiguous()

    weights = torch.where(
        labels > 0,
        torch.full_like(scores, pos_weight),
        torch.full_like(scores, neg_weight),
    )
    loss = (weights * (scores - labels).pow(2)).sum()

    if drop_constant:
        loss = loss - pos_weight * labels.sum()

    loss = loss / users.size(0)
    if l2_reg > 0.0:
        loss = loss + l2_reg * model.l2_norm() / users.size(0)
    return loss


class ENMFTrainingModule(nn.Module):
    """Обёртка "модель + лосс", чей `forward` возвращает значение функции потерь.

    Нужна для корректной работы DistributedDataParallel. DDP навешивает хуки
    синхронизации градиентов внутри собственного `forward`: он помечает, какие
    параметры участвовали в шаге, и готовит редуктор к обратному проходу. Если
    вызывать методы модели напрямую в обход `forward` обёртки DDP, градиенты
    между процессами могут не синхронизироваться -- и обучение молча
    превратится в несколько независимых моделей, по одной на процесс.

    Поэтому в DDP заворачивается именно этот модуль, а сама `ENMF` остаётся
    доступной через атрибут `.model` для инференса и оценки.
    """

    def __init__(
        self,
        model: ENMF,
        pos_weight: float = 1.0,
        neg_weight: float = 0.1,
        l2_reg: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.l2_reg = float(l2_reg)

    def forward(self, users: torch.Tensor, histories: torch.Tensor) -> torch.Tensor:
        return enmf_loss(
            self.model,
            users,
            histories,
            pos_weight=self.pos_weight,
            neg_weight=self.neg_weight,
            l2_reg=self.l2_reg,
        )
