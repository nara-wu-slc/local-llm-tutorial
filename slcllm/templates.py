"""chat template 関連の補助.

* モデルに chat template が含まれていない場合（ベースモデルなど）に使う簡易テンプレート
* GGUF（llama.cpp）に埋め込まれた chat template を Hugging Face と同じ流儀で描画する関数
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

# chat template を持たないモデル用の、ごく簡単な日本語テンプレート
SIMPLE_JA_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}{{ '### 指示:\n' + message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '### 応答:\n' + message['content'] + '\n\n' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '### 応答:\n' }}{% endif %}"
)


def merge_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """system ロールに対応していないモデル用: system の内容を最初の user 発話の先頭に移す."""
    system = [m["content"] for m in messages if m["role"] == "system" and m.get("content")]
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if system:
        for m in rest:
            if m["role"] == "user":
                m["content"] = "\n\n".join(system + [m["content"]])
                break
    return rest


def render_chat_template(
    template: str,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    special_tokens: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> str:
    """Hugging Face transformers とほぼ同じ環境で Jinja2 テンプレートを描画する（GGUF 用）."""
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    def tojson(x: Any, ensure_ascii: bool = False, indent: Any = None, separators: Any = None, sort_keys: bool = False) -> str:
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt: str) -> str:
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    variables = dict(special_tokens or {})
    variables.update(kwargs)
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=add_generation_prompt, **variables
    )
