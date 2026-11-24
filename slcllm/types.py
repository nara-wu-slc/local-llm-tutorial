"""共通で使うデータ型の定義."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

# チャットの1メッセージ: {"role": "system" | "user" | "assistant", "content": "..."}
Message = Dict[str, str]

# 旧チュートリアルで使っていた共通のシステムプロンプト
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"


class _ModelDefault:
    """「モデル（アダプタ）側の既定値を使う」ことを表す目印."""

    def __repr__(self) -> str:  # pragma: no cover
        return "MODEL_DEFAULT"


MODEL_DEFAULT: Any = _ModelDefault()


@dataclass
class GenerationParams:
    """生成パラメータ. None の項目は「指定なし（下位の既定値を使う）」を意味する."""

    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "GenerationParams":
        if not d:
            return cls()
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"未知の生成パラメータ: {sorted(unknown)}")
        return cls(**d)

    def merged(self, *others: Optional["GenerationParams"]) -> "GenerationParams":
        """右側ほど優先して上書きした新しいオブジェクトを返す（None は上書きしない）."""
        result = GenerationParams(**asdict(self))
        for other in others:
            if other is None:
                continue
            for f in fields(other):
                v = getattr(other, f.name)
                if v is not None:
                    setattr(result, f.name, v)
        return result

    @property
    def greedy(self) -> bool:
        return self.do_sample is False or self.temperature == 0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RawGeneration:
    """バックエンドが返す生の生成結果（特殊トークンを含む文字列）."""

    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatResponse:
    """1回の応答. content が最終回答、reasoning が思考過程（あれば）."""

    content: str
    reasoning: str = ""
    raw: str = ""
    prompt: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    elapsed: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_per_sec(self) -> float:
        return self.output_tokens / self.elapsed if self.elapsed > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.content


@dataclass
class TemplateFeatures:
    """chat template の中身から自動判定した機能."""

    harmony: bool = False  # gpt-oss などの harmony 形式（<|channel|>analysis/final）
    think_tags: bool = False  # <think>...</think> 形式の思考
    enable_thinking: bool = False  # apply_chat_template(enable_thinking=...) に対応
    reasoning_effort: bool = False  # apply_chat_template(reasoning_effort=...) に対応
    no_think_flag: bool = False  # システムプロンプトの /think /no_think で切替
    system_role: bool = True  # system ロールを受け付けるか

    def summary(self) -> List[str]:
        return [k for k, v in asdict(self).items() if v and k != "system_role"] + (
            [] if self.system_role else ["no-system-role"]
        )
