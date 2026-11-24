"""Sarashina（SB Intuitions）のモデル用アダプタ.

* Sarashina2.2-{0.5b,1b,3b}-instruct-v0.1 … 指示応答モデル（思考なし）
* Sarashina2 / 2.1 / 2.2 のベースモデル   … chat template を持たない事前学習モデル.
  簡易テンプレートで一応対話できるが、--no-chat-template で続きを書かせる使い方が本来の用途.
* Sarashina2 / 2.1 系の tokenizer は use_fast=False（sentencepiece）で読む必要がある
  （モデルカードの指示. 2.2 系は不要）.
"""

from __future__ import annotations

import re
from typing import Optional

from ..types import GenerationParams
from .base import BaseAdapter


class SarashinaAdapter(BaseAdapter):
    name = "sarashina"
    description = "Sarashina2 / 2.1 / 2.2 系（思考なし）"
    patterns = (r"sarashina",)

    reasoning_format = "none"

    def __init__(self, model_id: str, think: Optional[bool] = None, reasoning_effort: Optional[str] = None):
        super().__init__(model_id, think, reasoning_effort)
        if re.search(r"sarashina2(\.1)?-", model_id.lower()):
            self.tokenizer_kwargs = {"use_fast": False}

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        # モデルカードに推奨値の明記が無いため、一般的な値を設定
        return GenerationParams(do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.05)
