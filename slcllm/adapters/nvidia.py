"""NVIDIA Nemotron 系モデル用アダプタ.

同じ Nemotron でも世代ごとに「思考の切替方法」が違う.

| 系統                     | 例                                         | 思考の切替方法                              |
|--------------------------|--------------------------------------------|---------------------------------------------|
| Llama-Nemotron v1        | Llama-3.1-Nemotron-Nano-8B-v1, -4B-v1.1    | システムプロンプトを "detailed thinking on/off" にする |
| Llama-Nemotron v1.5      | Llama-3.3-Nemotron-Super-49B-v1.5          | システムプロンプトに /no_think              |
| Nemotron Nano v2         | NVIDIA-Nemotron-Nano-9B-v2(-Japanese)      | /think, /no_think または enable_thinking    |
| Nemotron 3               | NVIDIA-Nemotron-3-Nano-4B / 30B-A3B        | enable_thinking                             |

* Nano v2 / Nemotron 3 は Mamba-Transformer ハイブリッド（model_type = nemotron_h）.
  最近の transformers は nemotron_h を標準で扱えるので trust_remote_code は "auto"
  （標準実装で読めなければリポジトリのコードで読み直す）にしている.
* vLLM では精度のため mamba_ssm_cache_dtype=float32 を指定する（モデルカードの注意書き）.
* 推奨サンプリング: 思考オンは temperature=0.6, top_p=0.95（Nemotron 3 は temperature=1.0）,
  思考オフは greedy.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..templates import merge_system_into_user
from ..types import GenerationParams, Message
from .base import BaseAdapter


class NemotronAdapter(BaseAdapter):
    name = "nemotron"
    description = "NVIDIA Nemotron 系（Llama-Nemotron v1 / Nano v2 / Nemotron 3）"
    patterns = (r"nemotron",)

    reasoning_format = "think"
    default_thinking = True
    max_new_tokens = 1024
    max_new_tokens_thinking = 4096

    def __init__(self, model_id: str, think: Optional[bool] = None, reasoning_effort: Optional[str] = None):
        super().__init__(model_id, think, reasoning_effort)
        mid = model_id.lower()
        if re.search(r"llama-3\.\d-nemotron", mid) and "v1.5" not in mid:
            self.variant = "detailed-thinking"
        elif re.search(r"llama-3\.\d-nemotron", mid):
            self.variant = "system-flag"
        elif "nemotron-3-" in mid:
            self.variant = "nemotron-3"
        elif re.search(r"nemotron-nano-\d+b-v2", mid):
            self.variant = "nano-v2"
        else:
            self.variant = "other"
        if self.variant in ("nano-v2", "nemotron-3"):
            self.vllm_kwargs = {"mamba_ssm_cache_dtype": "float32"}

    def setup(self, chat_template, special_tokens=()):
        super().setup(chat_template, special_tokens)
        if self.variant == "system-flag":
            self.features.no_think_flag = True
            self.check_options()

    @property
    def can_toggle_thinking(self) -> bool:
        return self.variant == "detailed-thinking" or super().can_toggle_thinking

    def is_thinking(self, prompt: str) -> Optional[bool]:
        if self.variant == "detailed-thinking":
            return self.think is not False
        return super().is_thinking(prompt)

    def prepare_messages(self, messages: List[Message]) -> List[Message]:
        if self.variant != "detailed-thinking":
            return super().prepare_messages(messages)
        # Llama-Nemotron v1: システムプロンプトは切替用の文言だけにし、
        # それ以外の指示は user 発話に含める（モデルカードの指示）
        flag = "detailed thinking " + ("off" if self.think is False else "on")
        return [{"role": "system", "content": flag}] + merge_system_into_user(messages)

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        if thinking is False:
            return GenerationParams(do_sample=False)
        temperature = 1.0 if self.variant == "nemotron-3" else 0.6
        return GenerationParams(do_sample=True, temperature=temperature, top_p=0.95)
