"""Конфигурация экспериментов.

Все параметры собраны в дата-классы и загружаются из YAML. Любой параметр можно
переопределить из командной строки через `--set`, например:

    python -m enmf.cli train --config configs/ml1m.yaml --set model.embedding_dim=128

Такая схема нужна для воспроизводимости: конфиг эксперимента сохраняется рядом
с логами и чекпоинтом, поэтому по результату всегда можно восстановить, какими
именно параметрами он получен.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset: str = "ml-1m"
    data_dir: str = "data"
    binarize_threshold: float = 0.0
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    split_strategy: str = "global"       # "global" | "per_user"
    val_ratio: float = 0.2
    max_history: int | None = None


@dataclass
class ModelConfig:
    embedding_dim: int = 64
    dropout: float = 0.3
    init_std: float = 0.01


@dataclass
class LossConfig:
    pos_weight: float = 1.0
    neg_weight: float = 0.1              # ключевой гиперпараметр ENMF
    l2_reg: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256                # батч пользователей, не взаимодействий
    lr: float = 0.05
    weight_decay: float = 0.0
    num_workers: int = 0
    eval_every: int = 1
    early_stopping_patience: int = 10
    early_stopping_metric: str = "ndcg@10"
    seed: int = 42
    device: str = "auto"                 # "auto" | "cpu" | "cuda" | "xpu"
    ddp_backend: str = "auto"            # "auto" | "gloo" | "nccl"
    grad_clip: float = 0.0


@dataclass
class EvalConfig:
    ks: list[int] = field(default_factory=lambda: [5, 10, 20])
    batch_size: int = 512
    # Второй бейзлайн -- item-kNN (enmf.models.item_knn).
    item_knn: bool = True
    item_knn_shrinkage: float = 10.0     # α в сглаживании n_ij / (n_ij + α)
    item_knn_aggregation: str = "sum"    # "sum" | "topk_mean"
    item_knn_top_k: int = 3              # для topk_mean


@dataclass
class OutputConfig:
    run_name: str = "enmf"
    output_dir: str = "runs"
    tensorboard: bool = True
    save_checkpoint: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Config:
        sections = {
            "data": DataConfig,
            "model": ModelConfig,
            "loss": LossConfig,
            "train": TrainConfig,
            "eval": EvalConfig,
            "output": OutputConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, section_cls in sections.items():
            section_payload = payload.get(name, {}) or {}
            known = {f.name for f in dataclasses.fields(section_cls)}
            unknown = set(section_payload) - known
            if unknown:
                raise ValueError(
                    f"неизвестные параметры в секции '{name}': {sorted(unknown)}"
                )
            kwargs[name] = section_cls(**section_payload)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.to_dict(), handle, allow_unicode=True, sort_keys=False
            )

    def apply_overrides(self, overrides: list[str] | None) -> Config:
        """Применяет переопределения вида 'секция.параметр=значение'."""
        if not overrides:
            return self
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"переопределение должно иметь вид секция.параметр=значение, "
                    f"получено: '{override}'"
                )
            key, raw_value = override.split("=", 1)
            if "." not in key:
                raise ValueError(f"не указана секция в переопределении: '{override}'")
            section_name, param_name = key.split(".", 1)
            section = getattr(self, section_name, None)
            if section is None:
                raise ValueError(f"неизвестная секция конфига: '{section_name}'")
            if not hasattr(section, param_name):
                raise ValueError(
                    f"неизвестный параметр '{param_name}' в секции '{section_name}'"
                )
            # YAML-парсинг значения даёт правильные типы для int/float/bool/null.
            setattr(section, param_name, yaml.safe_load(raw_value))
        return self

    def flat_hparams(self) -> dict[str, Any]:
        """Плоский словарь параметров для логирования гиперпараметров."""
        flat: dict[str, Any] = {}
        for section_name, section in self.to_dict().items():
            for param_name, value in section.items():
                if isinstance(value, list):
                    value = ",".join(str(item) for item in value)
                if value is None:
                    value = "none"
                flat[f"{section_name}.{param_name}"] = value
        return flat
