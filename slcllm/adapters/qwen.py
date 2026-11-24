"""Alibaba Qwen 用アダプタ.

同じ Qwen でも世代によって振る舞いが違うので、モデル名から「系統（variant）」を判定して切り替える.

| 系統                 | 例                         | 思考モード                                   |
|----------------------|----------------------------|----------------------------------------------|
| Qwen2.5              | Qwen2.5-7B-Instruct        | なし                                         |
| Qwen3（ハイブリッド）| Qwen3-8B, Qwen3-30B-A3B    | 既定で思考. enable_thinking=False で無効化   |
| Qwen3 Instruct-2507  | Qwen3-4B-Instruct-2507     | 思考しない専用モデル                         |
| Qwen3 Thinking-2507  | Qwen3-4B-Thinking-2507     | 常に思考（<think> はテンプレート側で付与）   |
| Qwen3.5 / 3.8        | Qwen3.5-9B, Qwen3.8-27B    | enable_thinking で切替. 小型(≤9B)の 3.5 は既定で思考しない |
|                      |                            | 3.8 は reasoning_effort (low/medium/xhigh) にも対応 |

思考の切替そのものは基底クラスが chat template を見て自動で行うので、
ここでは「既定で思考するか」と「推奨サンプリングパラメータ」だけを定義している.
（Qwen3.5 / 3.8 は画像も扱えるモデルだが、ここではテキストのみで使う）
"""

from __future__ import annotations

import re
from typing import Optional

from ..types import GenerationParams
from .base import BaseAdapter


def _size_in_billions(model_id: str) -> Optional[float]:
    m = re.search(r"-(\d+(?:\.\d+)?)b(?:-|$)", model_id.lower())
    return float(m.group(1)) if m else None


class QwenAdapter(BaseAdapter):
    name = "qwen"
    description = "Qwen3 / Qwen3-2507 / Qwen3.5 / Qwen3.8 系（<think> 形式, enable_thinking 対応）"
    patterns = (r"qwen3", r"qwq")

    reasoning_format = "think"
    max_new_tokens = 2048
    max_new_tokens_thinking = 8192

    def __init__(self, model_id: str, think: Optional[bool] = None, reasoning_effort: Optional[str] = None):
        super().__init__(model_id, think, reasoning_effort)
        mid = model_id.lower()
        if re.search(r"qwen3\.[5-9]", mid):
            self.variant = "qwen3.8" if re.search(r"qwen3\.[6-9]", mid) else "qwen3.5"
            size = _size_in_billions(mid)
            # Qwen3.5 の小型モデル (0.8B/2B/4B/9B) は既定で思考しない
            self.default_thinking = not (self.variant == "qwen3.5" and size is not None and size <= 9)
        elif "instruct-2507" in mid:
            self.variant = "instruct-2507"
            self.default_thinking = False
        elif "thinking-2507" in mid or "qwq" in mid:
            self.variant = "thinking-2507"
            self.default_thinking = True
        else:
            self.variant = "qwen3"
            self.default_thinking = True

    def check_options(self) -> None:
        super().check_options()
        if self.variant == "instruct-2507" and self.think:
            self.note("Instruct-2507 は思考しない専用モデルです（思考させたい場合は Thinking-2507 を使ってください）")
        if self.variant == "thinking-2507" and self.think is False:
            self.note("Thinking-2507 は常に思考する専用モデルです（--no-think は効きません）")

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        if self.variant in ("qwen3.5", "qwen3.8"):
            if thinking:
                presence = 1.5 if self.variant == "qwen3.5" else 0.0
                return GenerationParams(do_sample=True, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                                        presence_penalty=presence)
            return GenerationParams(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                                    presence_penalty=1.5)
        if thinking:
            return GenerationParams(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
        return GenerationParams(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)


class Qwen2Adapter(BaseAdapter):
    name = "qwen2"
    description = "Qwen2 / Qwen2.5 系（思考なし）"
    patterns = (r"qwen2(\.5)?-",)

    reasoning_format = "none"

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        return GenerationParams(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)
