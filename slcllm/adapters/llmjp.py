"""LLM-jp（国立情報学研究所）のモデル用アダプタ.

* LLM-jp-3 / 3.1 系 (llm-jp-3-*-instruct*, llm-jp-3.1-*-instruct4)
    - 通常の（思考しない）指示応答モデル
    - モデルカードの推奨: do_sample=True, top_p=0.95, temperature=0.7, repetition_penalty=1.05
    - システムプロンプトはモデルカードの例「以下は、タスクを説明する指示です。…」を既定にする
    - chat template が無い（ベースモデルなど）場合は旧チュートリアルの「### 指示 / ### 応答」形式を使う
* LLM-jp-4 / 4.1 系 (llm-jp-4*-thinking)
    - 常に思考する推論モデル. chat template は harmony 形式（gpt-oss 互換）
    - 思考の深さは reasoning_effort（low / medium / high）で指定する
    - 旧チュートリアル (llm-jp_tutorial) にならって trust_remote_code=True で読み込む
"""

from __future__ import annotations

from typing import Optional

from ..types import GenerationParams
from .base import BaseAdapter

# 旧チュートリアル (llm-jp_tutorial/sample_interactive.py) で使っていたテンプレート
LLMJP3_LEGACY_TEMPLATE = (
    "{{'以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n'}}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{'### 指示:\n' + message['content'] + '\n\n'}}{% endif %}"
    "{% if message['role'] == 'assistant' %}{{'### 応答:\n' + message['content'] + '\n\n'}}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '### 応答:\n' }}{% endif %}"
)


class LLMjp3Adapter(BaseAdapter):
    name = "llm-jp-3"
    description = "LLM-jp-3 / 3.1 系（指示応答モデル）"
    patterns = (r"llm-jp-3",)

    default_system_prompt = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"
    fallback_chat_template = LLMJP3_LEGACY_TEMPLATE
    reasoning_format = "none"
    max_new_tokens = 1024

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        return GenerationParams(do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.05)


class LLMjp4Adapter(BaseAdapter):
    name = "llm-jp-4"
    description = "LLM-jp-4 / 4.1 系（harmony 形式の推論モデル, reasoning_effort 対応）"
    patterns = (r"llm-jp-4",)

    trust_remote_code = True
    reasoning_format = "harmony"
    default_thinking = True
    max_new_tokens = 4096
    max_new_tokens_thinking = 4096

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        # モデルカードに推奨値の記載が無いため、旧チュートリアルの設定を踏襲
        return GenerationParams(do_sample=True, temperature=0.8, top_p=0.95, top_k=40, repetition_penalty=1.1)
