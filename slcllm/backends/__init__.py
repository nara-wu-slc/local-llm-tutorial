"""バックエンドの選択. 重いライブラリは使うときに初めて import する."""

from __future__ import annotations

from typing import Any

from .base import Backend

BACKENDS = {
    "hf": "Hugging Face Transformers（既定. ほぼ全モデルに対応）",
    "vllm": "vLLM（高速. まとめて大量に生成するとき向け）",
    "llamacpp": "llama.cpp / GGUF 量子化モデル（省メモリ）",
    "dryrun": "tokenizer だけ読み込んでプロンプトを確認（生成しない）",
}


def create_backend(name: str, model_id: str, adapter: Any, **kwargs: Any) -> Backend:
    if name in ("hf", "transformers"):
        from .hf import HFBackend

        return HFBackend(model_id, adapter, **kwargs)
    if name == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(model_id, adapter, **kwargs)
    if name in ("llamacpp", "llama.cpp", "gguf"):
        from .llamacpp import LlamaCppBackend

        return LlamaCppBackend(model_id, adapter, **kwargs)
    if name == "dryrun":
        from .hf import DryRunBackend

        return DryRunBackend(model_id, adapter, **kwargs)
    raise ValueError(f"未知のバックエンド: {name}（候補: {', '.join(BACKENDS)}）")


__all__ = ["BACKENDS", "Backend", "create_backend"]
