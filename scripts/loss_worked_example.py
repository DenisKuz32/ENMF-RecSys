#!/usr/bin/env python3
"""Разобранный пример лосса ENMF: 2 пользователя, 3 фильма, d = 2.

    python scripts/loss_worked_example.py

Лосс считается тремя способами, и все три обязаны совпасть:
  1) наивно -- по всем 6 парам (пользователь, фильм);
  2) эффективно -- через матрицы Грама d x d, как в статье ENMF;
  3) функциями проекта enmf_loss_naive и enmf_loss.
Все числа можно проверить на бумаге; вывод формулы -- reports/report.md,
раздел 2.2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from enmf.losses import enmf_loss, enmf_loss_naive
from enmf.models.enmf import ENMF

# ---- исходные данные --------------------------------------------------- #
P = np.array([[1.0, 0.5],     # пользователь 1
              [0.2, 1.0]])    # пользователь 2
Q = np.array([[1.0, 0.0],     # фильм 1
              [0.5, 0.5],     # фильм 2
              [0.0, 1.0]])    # фильм 3
h = np.array([1.0, 0.5])
R = np.array([[1, 0, 0],      # пользователь 1 смотрел фильм 1
              [0, 1, 1]])     # пользователь 2 смотрел фильмы 2 и 3
C_POS, C_NEG = 1.0, 0.2       # c+ и c-


def main() -> None:
    print("Скоры y(u,i) = h^T (p_u * q_i):")
    Y = (P * h) @ Q.T
    print(Y, "\n")

    # ---- 1. наивно: все пары ------------------------------------------- #
    weights = np.where(R == 1, C_POS, C_NEG)
    per_pair = weights * (Y - R) ** 2
    naive = per_pair.sum()
    print("1) Наивно, c_ui * (y_ui - r_ui)^2 по всем 6 парам:")
    print(per_pair.round(6))
    print(f"   сумма = {naive:.6f}")
    constant = C_POS * R.sum()
    print(f"   без константы c+ * |R| = {constant:.0f}:  {naive - constant:.6f}\n")

    # ---- 2. эффективно: матрицы Грама ---------------------------------- #
    observed_scores = Y[R == 1]
    observed = ((C_POS - C_NEG) * observed_scores**2 - 2 * C_POS * observed_scores).sum()
    gram_p, gram_q, gram_h = P.T @ P, Q.T @ Q, np.outer(h, h)
    all_pairs = C_NEG * (gram_h * gram_p * gram_q).sum()
    print("2) Через матрицы Грама:")
    print(f"   скоры позитивов: {observed_scores}")
    print(f"   слагаемое по позитивам  sum[(c+ - c-) y^2 - 2 c+ y] = {observed:.6f}")
    print(f"   P^T P =\n{gram_p}\n   Q^T Q =\n{gram_q}\n   h h^T =\n{gram_h}")
    print(f"   слагаемое по всем парам c- * sum(hh^T * P^TP * Q^TQ) = {all_pairs:.6f}")
    print(f"   (для сверки c- * sum(y^2) по всем парам = {C_NEG * (Y**2).sum():.6f})")
    efficient = observed + all_pairs
    print(f"   итого = {efficient:.6f}\n")

    # ---- 3. функции проекта -------------------------------------------- #
    model = ENMF(n_users=2, n_items=3, embedding_dim=2, dropout=0.0).double()
    with torch.no_grad():
        model.user_embedding.weight.copy_(torch.from_numpy(P))
        model.item_embedding.weight[:3] = torch.from_numpy(Q)
        model.h.copy_(torch.from_numpy(h))
    users = torch.tensor([0, 1])
    histories = torch.tensor([[0, 3],   # 3 = паддинг (n_items)
                              [1, 2]])
    kwargs = {"pos_weight": C_POS, "neg_weight": C_NEG}
    code_naive = enmf_loss_naive(model, users, histories, **kwargs).item()
    code_fast = enmf_loss(model, users, histories, **kwargs).item()
    print("3) Функции проекта (делят на размер батча B = 2):")
    print(f"   enmf_loss_naive = {code_naive:.6f}  (x2 = {2 * code_naive:.6f})")
    print(f"   enmf_loss       = {code_fast:.6f}  (x2 = {2 * code_fast:.6f})\n")

    assert np.isclose(naive - constant, efficient)
    assert np.isclose(2 * code_naive, efficient) and np.isclose(2 * code_fast, efficient)
    print("Все способы совпали.")


if __name__ == "__main__":
    main()
