"""Проверка корректности распределённого обучения.

Главный риск DDP в этом проекте не в том, что запуск упадёт, а в том, что он
тихо отработает *без* синхронизации градиентов: каждый процесс обучит свою
копию модели, обучение формально пройдёт, метрики окажутся хуже -- и причина
будет неочевидной. Поэтому проверяется именно эквивалентность:

    градиент DDP на двух процессах с половинами батча
        ==
    градиент одиночного процесса на полном батче

Дополнительно проверяется, что распределённая оценка (шардирование
пользователей + all_reduce) даёт ровно те же метрики, что и одиночная.

Тесты запускают настоящие процессы через torch.multiprocessing и backend gloo,
то есть проверяют реальный путь исполнения, а не его имитацию.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from enmf.losses import ENMFTrainingModule, enmf_loss
from enmf.metrics import MetricAccumulator, evaluate_model
from enmf.models.enmf import ENMF

N_USERS, N_ITEMS, DIM, HISTORY = 16, 24, 8, 6
NEG_WEIGHT = 0.2


def _fixture_batch():
    generator = torch.Generator().manual_seed(123)
    users = torch.arange(N_USERS)
    histories = torch.randint(
        0, N_ITEMS, (N_USERS, HISTORY), generator=generator, dtype=torch.long
    )
    return users, histories


def _fixture_model() -> ENMF:
    torch.manual_seed(321)
    model = ENMF(N_USERS, N_ITEMS, embedding_dim=DIM, dropout=0.0)
    with torch.no_grad():
        model.user_embedding.weight.normal_(std=0.3)
        model.item_embedding.weight.normal_(std=0.3)
        model.item_embedding.weight[model.n_items].zero_()
        model.h.normal_(std=0.3)
    return model.double()


def _ddp_gradient_worker(rank: int, world_size: int, result_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        users, histories = _fixture_batch()
        shard = slice(rank * (N_USERS // world_size), (rank + 1) * (N_USERS // world_size))

        model = _fixture_model()
        trainer = DistributedDataParallel(
            ENMFTrainingModule(model, neg_weight=NEG_WEIGHT)
        )
        loss = trainer(users[shard], histories[shard])
        loss.backward()

        if rank == 0:
            torch.save(
                {
                    "user": model.user_embedding.weight.grad.clone(),
                    "item": model.item_embedding.weight.grad.clone(),
                    "h": model.h.grad.clone(),
                },
                result_path,
            )
    finally:
        dist.destroy_process_group()


def _ddp_bypass_worker(rank: int, world_size: int, result_path: str) -> None:
    """Лосс считается в обход DDP.forward -- так делать нельзя (см. тест ниже)."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        users, histories = _fixture_batch()
        shard = slice(rank * (N_USERS // world_size), (rank + 1) * (N_USERS // world_size))

        model = _fixture_model()
        trainer = DistributedDataParallel(ENMFTrainingModule(model, neg_weight=NEG_WEIGHT))
        for _ in range(2):
            trainer.zero_grad()
            # Обход: модель вызывается напрямую, а не через trainer(...).
            enmf_loss(trainer.module.model, users[shard], histories[shard],
                      neg_weight=NEG_WEIGHT).backward()

        gathered = [torch.zeros_like(model.h.grad) for _ in range(world_size)]
        dist.all_gather(gathered, model.h.grad)
        if rank == 0:
            torch.save({"same_on_all_ranks": torch.allclose(gathered[0], gathered[1])},
                       result_path)
    finally:
        dist.destroy_process_group()


def _ddp_eval_worker(rank: int, world_size: int, result_path: str) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29518")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from enmf.data.dataset import UserHistoryDataset
        from enmf.data.prepare import PreparedData
        from enmf.train import evaluate_distributed
        from enmf.utils import DistributedContext

        model = _fixture_model().float()
        train_by_user = {u: np.arange(u % 3, u % 3 + 4) % N_ITEMS for u in range(N_USERS)}
        val_by_user = {u: np.array([(u * 5 + 7) % N_ITEMS]) for u in range(N_USERS)}

        import pandas as pd

        data = PreparedData(
            n_users=N_USERS,
            n_items=N_ITEMS,
            train=pd.DataFrame({"user_id": [0], "item_id": [0], "timestamp": [0]}),
            validation=pd.DataFrame({"user_id": [0], "item_id": [0], "timestamp": [0]}),
            train_by_user=train_by_user,
            val_by_user=val_by_user,
            split_stats=None,
            dataset=UserHistoryDataset(train_by_user, N_ITEMS),
            item_map={},
            user_map={},
        )
        context = DistributedContext(
            rank=rank, local_rank=rank, world_size=world_size, enabled=True, backend="gloo"
        )
        metrics = evaluate_distributed(
            model, data, ks=(5, 10), batch_size=4,
            device=torch.device("cpu"), context=context,
        )
        if rank == 0:
            torch.save(metrics, result_path)
    finally:
        dist.destroy_process_group()


def _run_workers(worker, world_size: int = 2, timeout: int = 300):
    """Запускает воркеры в отдельных процессах и забирает результат с ранга 0.

    Обмен идёт через файл, а не через multiprocessing.Queue: очередь держит
    живое соединение между процессами и в связке с `spawn` и завершением
    process group даёт нестабильные обрывы, не связанные с самим DDP.
    """
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as directory:
        result_path = os.path.join(directory, "result.pt")
        processes = [
            context.Process(target=worker, args=(rank, world_size, result_path))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=timeout)
            assert process.exitcode == 0, (
                f"процесс завершился с кодом {process.exitcode}"
            )
        assert os.path.exists(result_path), "ранг 0 не сохранил результат"
        return torch.load(result_path, map_location="cpu", weights_only=False)


@pytest.mark.slow
def test_ddp_gradients_match_single_process() -> None:
    """Градиенты DDP на 2 процессах совпадают с одиночным процессом."""
    ddp_grads = _run_workers(_ddp_gradient_worker, world_size=2)

    users, histories = _fixture_batch()
    model = _fixture_model()
    enmf_loss(model, users, histories, neg_weight=NEG_WEIGHT).backward()

    single = {
        "user": model.user_embedding.weight.grad,
        "item": model.item_embedding.weight.grad,
        "h": model.h.grad,
    }
    for key, expected in single.items():
        assert torch.allclose(ddp_grads[key], expected, atol=1e-8), (
            f"градиенты '{key}' расходятся между DDP и одиночным процессом; "
            f"max|diff|={(ddp_grads[key] - expected).abs().max():.3e}"
        )


@pytest.mark.slow
def test_bypassing_ddp_forward_silently_breaks_sync() -> None:
    """Ловушка, ради которой существует ENMFTrainingModule.

    Если вызвать модель в обход DDP.forward, градиенты между процессами не
    синхронизируются, и никакой ошибки при этом нет: каждый процесс молча
    обучает свою копию. Тест фиксирует это поведение PyTorch; парный тест
    test_ddp_gradients_match_single_process показывает, что через forward
    обёртки синхронизация есть.
    """
    result = _run_workers(_ddp_bypass_worker, world_size=2)
    assert result["same_on_all_ranks"] is False


@pytest.mark.slow
def test_ddp_evaluation_matches_single_process() -> None:
    """Распределённая оценка даёт те же метрики, что и одиночная."""
    ddp_metrics = _run_workers(_ddp_eval_worker, world_size=2)

    model = _fixture_model().float()
    train_by_user = {u: np.arange(u % 3, u % 3 + 4) % N_ITEMS for u in range(N_USERS)}
    val_by_user = {u: np.array([(u * 5 + 7) % N_ITEMS]) for u in range(N_USERS)}
    users = sorted(val_by_user)

    single_metrics = evaluate_model(
        model, users, train_by_user, val_by_user, N_ITEMS, ks=(5, 10), device="cpu"
    )
    for key in single_metrics:
        assert ddp_metrics[key] == pytest.approx(single_metrics[key], abs=1e-10), (
            f"метрика '{key}' расходится: DDP={ddp_metrics[key]} "
            f"одиночный={single_metrics[key]}"
        )


def test_metric_accumulator_roundtrip() -> None:
    """Упаковка/распаковка накопителя для all_reduce не теряет данные."""
    accumulator = MetricAccumulator(ks=(5, 10))
    accumulator.update(
        {key: torch.tensor(index + 1.0) for index, key in enumerate(sorted(accumulator.sums))},
        n_users=7,
    )
    packed = accumulator.as_tensor(torch.device("cpu"))

    restored = MetricAccumulator(ks=(5, 10))
    restored.load_tensor(packed)
    assert restored.n_users == 7
    assert restored.sums == accumulator.sums


def _averaged_gradient(shards: list[slice]) -> torch.Tensor:
    """Градиент так, как его получает DDP: среднее градиентов процессов,
    каждый из которых делит лосс на размер своей доли батча."""
    users, histories = _fixture_batch()
    gradients = []
    for shard in shards:
        model = _fixture_model()
        enmf_loss(model, users[shard], histories[shard], neg_weight=NEG_WEIGHT).backward()
        gradients.append(model.user_embedding.weight.grad.clone())
    return sum(gradients) / len(gradients)


def test_ddp_equivalence_requires_equal_shards() -> None:
    """Эквивалентность DDP и одного процесса точна только при равных долях.

    Каждый процесс делит лосс на свой размер батча, DDP затем усредняет
    градиенты; при долях 12/4 у пользователей меньшей доли вес втрое больше.
    """
    full = _averaged_gradient([slice(0, N_USERS)])
    equal = _averaged_gradient([slice(0, 8), slice(8, 16)])
    unequal = _averaged_gradient([slice(0, 12), slice(12, 16)])

    assert torch.allclose(equal, full, atol=1e-12)
    assert not torch.allclose(unequal, full, atol=1e-6)


@pytest.mark.parametrize(
    ("n_users", "world_size", "duplicates"),
    [(6040, 2, 0), (6040, 4, 0), (5400, 4, 0), (800, 3, 1), (6041, 4, 3)],
)
def test_distributed_sampler_equalizes_shards(
    n_users: int, world_size: int, duplicates: int
) -> None:
    """DistributedSampler выравнивает доли, дополняя данные повторами.

    Поэтому при одинаковом batch_size шаги на всех процессах равного размера
    и условие точной эквивалентности выполняется; цена -- до world_size - 1
    повторно обработанных пользователей за эпоху.
    """
    from torch.utils.data import DistributedSampler

    sizes = {
        len(DistributedSampler(range(n_users), num_replicas=world_size, rank=rank))
        for rank in range(world_size)
    }
    assert len(sizes) == 1
    assert sizes.pop() * world_size - n_users == duplicates
