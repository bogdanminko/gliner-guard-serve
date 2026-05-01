from __future__ import annotations

import os
from dataclasses import dataclass

import torch


_DTYPE_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
}

_NAME_TO_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass(frozen=True)
class TorchRuntimeConfig:
    name: str
    torch_dtype: torch.dtype

    @property
    def runtime_prefix(self) -> str:
        return f"pytorch-{self.name}"


def resolve_torch_dtype(raw_value: str | None = None) -> TorchRuntimeConfig:
    raw = (raw_value or os.environ.get("TORCH_DTYPE", "bf16")).strip().lower()
    name = _DTYPE_ALIASES.get(raw)
    if name is None:
        supported = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(
            f"Unsupported TORCH_DTYPE={raw!r}. Supported values: {supported}"
        )
    return TorchRuntimeConfig(name=name, torch_dtype=_NAME_TO_DTYPE[name])
