"""Swallow（東京科学大学 岡崎研・横田研 / 産総研）のモデル用アダプタ.

Swallow は既存モデルを日本語で継続学習したものなので、元モデルのアダプタを継承して
「Swallow 版で違うところ」だけを上書きしている（バージョン差の吸収の例）.

* Llama 3.1 / 3.3 Swallow     … Llama 3 系テンプレート. 推奨: temperature=0.6, top_p=0.9
* Gemma-2-Llama Swallow       … Gemma 2 系テンプレート（system ロール非対応）
* Qwen3-Swallow (2026/2)      … Qwen3 ベースの推論モデル. 思考は常にオン（切替不可）.
                                 推奨: temperature=0.6, top_p=0.95, top_k=20, min_p=0
* GPT-OSS-Swallow (2026/2)    … gpt-oss ベース. reasoning_effort は medium でのみ学習・評価.
                                 推奨: temperature=0.6, top_p=0.95, top_k=20, min_p=0
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import GenerationParams
from .base import BaseAdapter
from .gpt_oss import GptOssAdapter
from .qwen import QwenAdapter


class LlamaSwallowAdapter(BaseAdapter):
    name = "swallow-llama"
    description = "Llama 3.1 / 3.3 Swallow（Llama 3 系, 思考なし）"
    patterns = (r"llama-3(\.\d)?-swallow",)

    reasoning_format = "none"

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        return GenerationParams(do_sample=True, temperature=0.6, top_p=0.9)


class GemmaSwallowAdapter(LlamaSwallowAdapter):
    name = "swallow-gemma"
    description = "Gemma-2-Llama Swallow（Gemma 2 系, system ロール非対応）"
    patterns = (r"gemma-2-llama-swallow",)

    supports_system_role = False


class QwenSwallowAdapter(QwenAdapter):
    name = "swallow-qwen3"
    description = "Qwen3-Swallow（常に思考する推論モデル）"
    patterns = (r"qwen3-swallow",)

    def __init__(self, model_id: str, think: Optional[bool] = None, reasoning_effort: Optional[str] = None):
        super().__init__(model_id, think, reasoning_effort)
        self.variant = "qwen3-swallow"
        self.default_thinking = True

    @property
    def can_toggle_thinking(self) -> bool:
        return False

    def check_options(self) -> None:
        if self.reasoning_effort is not None:
            self.note("このモデルは reasoning_effort に対応していないため --reasoning-effort は無視されます")
        if self.think is False:
            self.note("Qwen3-Swallow は思考を無効化できません（--no-think は無視されます）")
        if "-cpt-" in self.model_id.lower():
            self.note("CPT 版は継続事前学習のみのモデルです. 対話には SFT 版か RL 版を使ってください")

    def chat_template_kwargs(self) -> Dict[str, Any]:
        # 元の Qwen3 テンプレートの enable_thinking を残していても、Swallow は常に思考する前提で
        # 学習されているので、思考を切る指定は渡さない.
        return {}

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        return GenerationParams(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)


class GptOssSwallowAdapter(GptOssAdapter):
    name = "swallow-gpt-oss"
    description = "GPT-OSS-Swallow（harmony 形式, reasoning_effort=medium 推奨）"
    patterns = (r"gpt-oss-swallow",)

    default_reasoning_effort = "medium"
    fallback_low_effort = "medium"  # medium 以外は学習していないので --no-think でも medium のまま

    def check_options(self) -> None:
        super().check_options()
        if self.reasoning_effort not in (None, "medium"):
            self.note("GPT-OSS-Swallow は reasoning_effort=medium でのみ学習・評価されています")
        if self.think is False:
            self.note("GPT-OSS-Swallow は思考を無効化できません")

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        return GenerationParams(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
