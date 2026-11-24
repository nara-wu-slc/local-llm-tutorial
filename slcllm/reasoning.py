"""思考過程（reasoning）と最終回答を分離するための関数群.

モデルによって思考過程の出力形式が異なる:

* ``<think> ... </think>`` 形式: Qwen3 / Qwen3.5, Qwen3-Swallow, Nemotron など
* harmony 形式: gpt-oss, GPT-OSS-Swallow, LLM-jp-4 など
  ``<|channel|>analysis<|message|>（思考）<|end|><|start|>assistant<|channel|>final<|message|>（回答）<|return|>``

どちらも「生成途中の文字列」に対しても使えるように作ってある（ストリーミング表示用）.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .types import TemplateFeatures


@dataclass
class Parsed:
    reasoning: str
    content: str
    thinking_open: bool = False  # 思考がまだ閉じていない（生成途中 / 途中で打ち切られた）


# ---------------------------------------------------------------------------
# chat template の自動判定
# ---------------------------------------------------------------------------

_NO_SYSTEM_RE = re.compile(r"raise_exception\([^)]*[Ss]ystem")


def detect_template_features(template: Optional[str]) -> TemplateFeatures:
    """chat template（Jinja2 文字列）を眺めて、対応している機能を推定する."""
    t = template or ""
    return TemplateFeatures(
        harmony="<|channel|>" in t,
        think_tags="<think>" in t or "</think>" in t,
        enable_thinking="enable_thinking" in t,
        reasoning_effort="reasoning_effort" in t,
        no_think_flag="/no_think" in t,
        system_role=not bool(_NO_SYSTEM_RE.search(t)),
    )


# ---------------------------------------------------------------------------
# <think> ... </think> 形式
# ---------------------------------------------------------------------------


def prompt_opens_think(prompt: str, open_tag: str = "<think>", close_tag: str = "</think>") -> bool:
    """プロンプトの末尾で <think> が開いたままか（= 出力は思考から始まる）."""
    i = prompt.rfind(open_tag)
    if i == -1:
        return False
    return prompt.rfind(close_tag) < i


def parse_think(
    raw: str,
    starts_in_think: bool = False,
    open_tag: str = "<think>",
    close_tag: str = "</think>",
) -> Parsed:
    text = raw
    if starts_in_think and open_tag not in text:
        text = open_tag + text
    i = text.find(open_tag)
    if i == -1:
        j = text.find(close_tag)
        if j != -1:  # 開きタグがプロンプト側にあった場合
            return Parsed(text[:j], text[j + len(close_tag):], False)
        return Parsed("", text, False)
    before = text[:i]
    rest = text[i + len(open_tag):]
    j = rest.find(close_tag)
    if j == -1:
        return Parsed(rest, before, True)
    return Parsed(rest[:j], before + rest[j + len(close_tag):], False)


# ---------------------------------------------------------------------------
# harmony 形式
# ---------------------------------------------------------------------------

_HARMONY_END = r"<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|<\|channel\|>"
_HARMONY_MSG_RE = re.compile(
    r"<\|channel\|>\s*([A-Za-z_]+)[^<]*?<\|message\|>(.*?)(?=" + _HARMONY_END + r"|$)",
    re.S,
)


def parse_harmony(raw: str) -> Parsed:
    reasoning, content = [], []
    last_channel, last_closed = None, True
    for m in _HARMONY_MSG_RE.finditer(raw):
        channel, body = m.group(1), m.group(2)
        last_channel = channel
        last_closed = m.end() < len(raw)
        if channel == "analysis":
            reasoning.append(body)
        else:  # final / commentary
            content.append(body)
    if last_channel is None:
        # チャネル情報がない（思考なしで直接答えた等）
        if "<|message|>" in raw:
            return Parsed("", raw.rsplit("<|message|>", 1)[1], False)
        if raw.lstrip().startswith("<|channel|>"):
            return Parsed("", "", False)  # "<|channel|>analysis" まで来たところ（まだ何も表示しない）
        return Parsed("", raw, False)
    thinking_open = last_channel == "analysis" and not last_closed
    return Parsed("\n".join(reasoning), "\n".join(content), thinking_open)


# ---------------------------------------------------------------------------
# 後処理
# ---------------------------------------------------------------------------

# <|im_end|>, <|eot_id|>, <|return|> のような特殊トークン風の文字列
_SPECIAL_LIKE_RE = re.compile(r"<\|[A-Za-z0-9_\-\.]{1,40}\|>")


def remove_special_tokens(text: str, special_tokens: Iterable[str] = ()) -> str:
    for tok in special_tokens:
        if tok and len(tok) > 1:
            text = text.replace(tok, "")
    return _SPECIAL_LIKE_RE.sub("", text)


def holdback_partial_tag(text: str, window: int = 32) -> str:
    """ストリーミング表示で、書きかけのタグ（"<thi" など）を一時的に表示しないための処理."""
    i = text.rfind("<")
    if i != -1 and len(text) - i <= window and ">" not in text[i:]:
        return text[:i]
    return text
