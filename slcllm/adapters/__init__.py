"""アダプタの登録と選択.

モデル ID（例: "Qwen/Qwen3-8B"）を ADAPTERS の上から順にパターンと照合し、
最初に当てはまったアダプタを使う. どれにも当てはまらなければ汎用アダプタを使う.

新しいモデル系列に対応させたいときは:
  1. このディレクトリに新しいモジュールを作り、BaseAdapter を継承したクラスを書く
  2. 下の ADAPTERS に追加する（より具体的なパターンほど上に置く）
"""

from __future__ import annotations

from typing import List, Optional, Type

from .base import BaseAdapter, GenericAdapter
from .gpt_oss import GptOssAdapter
from .llmjp import LLMjp3Adapter, LLMjp4Adapter
from .nvidia import NemotronAdapter
from .qwen import Qwen2Adapter, QwenAdapter
from .sarashina import SarashinaAdapter
from .swallow import GemmaSwallowAdapter, GptOssSwallowAdapter, LlamaSwallowAdapter, QwenSwallowAdapter

# 上から順に照合する（"Qwen3-Swallow" が "qwen3" に先にマッチしないよう、派生モデルを先に置く）
ADAPTERS: List[Type[BaseAdapter]] = [
    # Swallow（Llama / Gemma / Qwen3 / gpt-oss の派生）
    GptOssSwallowAdapter,
    QwenSwallowAdapter,
    GemmaSwallowAdapter,
    LlamaSwallowAdapter,
    # LLM-jp
    LLMjp4Adapter,
    LLMjp3Adapter,
    # NVIDIA
    NemotronAdapter,
    # OpenAI
    GptOssAdapter,
    # Qwen
    QwenAdapter,
    Qwen2Adapter,
    # SB Intuitions
    SarashinaAdapter,
]


def get_adapter_class(model_id: str, adapter_name: Optional[str] = None) -> Type[BaseAdapter]:
    """アダプタ名が指定されればそれを、なければモデル ID から自動選択する."""
    if adapter_name:
        for cls in ADAPTERS + [GenericAdapter]:
            if cls.name == adapter_name:
                return cls
        names = ", ".join(c.name for c in ADAPTERS + [GenericAdapter])
        raise ValueError(f"アダプタ '{adapter_name}' はありません（候補: {names}）")
    for cls in ADAPTERS:
        if cls.matches(model_id):
            return cls
    return GenericAdapter


def list_adapters() -> List[Type[BaseAdapter]]:
    return ADAPTERS + [GenericAdapter]


__all__ = ["ADAPTERS", "BaseAdapter", "GenericAdapter", "get_adapter_class", "list_adapters"]
