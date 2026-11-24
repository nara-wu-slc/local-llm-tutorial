"""OpenAI gpt-oss 用アダプタ.

* 出力は harmony 形式（analysis チャネル = 思考, final チャネル = 回答）
* 思考の深さは reasoning_effort（low / medium / high, 既定 medium）で指定する.
  思考を完全に無効化することはできないので、--no-think 指定時は low で代用する.
* 推奨サンプリング: temperature=1.0, top_p=1.0（OpenAI の推奨値）
* chat template では、こちらが渡す system メッセージは developer メッセージ
  （"# Instructions"）として埋め込まれる.
* Transformers で MXFP4 量子化のまま動かすには kernels / triton が必要.
  無い場合は自動的に bf16 に展開される（20b でも GPU メモリを 40GB 程度使う）.
"""

from __future__ import annotations

from typing import Optional

from ..types import GenerationParams
from .base import BaseAdapter


class GptOssAdapter(BaseAdapter):
    name = "gpt-oss"
    description = "OpenAI gpt-oss（harmony 形式, reasoning_effort 対応）"
    patterns = (r"gpt-oss",)

    reasoning_format = "harmony"
    default_thinking = True
    max_new_tokens = 4096
    max_new_tokens_thinking = 4096
    gguf_repo_map = {
        "openai/gpt-oss-20b": "ggml-org/gpt-oss-20b-GGUF",
        "openai/gpt-oss-120b": "ggml-org/gpt-oss-120b-GGUF",
    }

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        return GenerationParams(do_sample=True, temperature=1.0, top_p=1.0)
