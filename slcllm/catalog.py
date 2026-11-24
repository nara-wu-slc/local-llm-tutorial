"""よく使うモデルの一覧と、短い名前からモデル ID への変換.

ここに載っていないモデルも "組織名/モデル名" の形で指定すれば使える（アダプタは自動選択）.
旧チュートリアルと同じく "-m llm-jp-3-1.8b-instruct" のような短い名前でも指定できる.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelEntry:
    repo_id: str
    family: str
    note: str = ""


def _e(family: str, *ids: str, note: str = "") -> List[ModelEntry]:
    return [ModelEntry(i, family, note) for i in ids]


MODELS: List[ModelEntry] = [
    # --- LLM-jp ---
    *_e("LLM-jp", "llm-jp/llm-jp-3-1.8b-instruct", "llm-jp/llm-jp-3-1.8b-instruct3",
        "llm-jp/llm-jp-3-3.7b-instruct", "llm-jp/llm-jp-3-3.7b-instruct3",
        "llm-jp/llm-jp-3-7.2b-instruct3", "llm-jp/llm-jp-3-13b-instruct", "llm-jp/llm-jp-3-13b-instruct3",
        note="指示応答"),
    *_e("LLM-jp", "llm-jp/llm-jp-3.1-1.8b-instruct4", "llm-jp/llm-jp-3.1-13b-instruct4", note="指示応答"),
    *_e("LLM-jp", "llm-jp/llm-jp-3-1.8b", "llm-jp/llm-jp-3-3.7b", "llm-jp/llm-jp-3-13b", note="ベースモデル"),
    *_e("LLM-jp", "llm-jp/llm-jp-4.1-8b-thinking", "llm-jp/llm-jp-4.1-32b-a3b-thinking",
        "llm-jp/llm-jp-4.1-33b-thinking", "llm-jp/llm-jp-4-8b-thinking", "llm-jp/llm-jp-4-32b-a3b-thinking",
        "llm-jp/llm-jp-4-33b-thinking", note="推論（harmony）"),
    *_e("LLM-jp", "llm-jp/llm-jp-4.1-8b-thinking-gguf", "llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf",
        "llm-jp/llm-jp-4.1-33b-thinking-gguf", note="GGUF（--backend llamacpp）"),
    # --- OpenAI ---
    *_e("gpt-oss", "openai/gpt-oss-20b", "openai/gpt-oss-120b", note="推論（harmony）"),
    # --- Swallow ---
    *_e("Swallow", "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1", "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.2",
        "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3", "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
        "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3", "tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4",
        "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1", note="Llama ベース"),
    *_e("Swallow", "tokyotech-llm/Gemma-2-Llama-Swallow-2b-it-v0.1", "tokyotech-llm/Gemma-2-Llama-Swallow-9b-it-v0.1",
        "tokyotech-llm/Gemma-2-Llama-Swallow-27b-it-v0.1", note="Gemma 2 ベース"),
    *_e("Swallow", "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", "tokyotech-llm/Qwen3-Swallow-30B-A3B-RL-v0.2",
        "tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2", "tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2", note="推論（<think>）"),
    *_e("Swallow", "tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1", "tokyotech-llm/GPT-OSS-Swallow-120B-RL-v0.1",
        "tokyotech-llm/GPT-OSS-Swallow-20B-SFT-v0.1", note="推論（harmony）"),
    # --- Qwen ---
    *_e("Qwen", "Qwen/Qwen2.5-7B-Instruct", note="思考なし"),
    *_e("Qwen", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B", note="思考 on/off 切替"),
    *_e("Qwen", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507", note="思考なし専用"),
    *_e("Qwen", "Qwen/Qwen3-4B-Thinking-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507", note="思考専用"),
    *_e("Qwen", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B",
        note="思考 on/off 切替（既定 off）"),
    *_e("Qwen", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.8-27B", note="思考 on/off 切替（既定 on）"),
    *_e("Qwen", "Qwen/Qwen3-8B-GGUF", note="GGUF（--backend llamacpp）"),
    # --- Sarashina ---
    *_e("Sarashina", "sbintuitions/sarashina2.2-0.5b-instruct-v0.1", "sbintuitions/sarashina2.2-1b-instruct-v0.1",
        "sbintuitions/sarashina2.2-3b-instruct-v0.1", note="指示応答"),
    *_e("Sarashina", "sbintuitions/sarashina2.2-3b", "sbintuitions/sarashina2-7b", "sbintuitions/sarashina2-13b",
        note="ベースモデル"),
    # --- NVIDIA ---
    *_e("NVIDIA", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese", note="日本語強化. 思考 on/off 切替"),
    *_e("NVIDIA", "nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        note="思考 on/off 切替"),
    *_e("NVIDIA", "nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        note="detailed thinking on/off"),
]


# tools/smoke_test.py で動作確認する既定のモデル（各系列から小さめのもの）. download.py --smoke-test でも使う.
# (モデル ID, LocalLLM に渡す追加の引数)
SMOKE_TEST_CASES = [
    ("llm-jp/llm-jp-3.1-1.8b-instruct4", {}),
    ("llm-jp/llm-jp-4.1-8b-thinking", {"reasoning_effort": "low"}),
    ("openai/gpt-oss-20b", {"reasoning_effort": "low"}),
    ("tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5", {}),
    ("tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", {}),
    ("tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1", {}),
    ("Qwen/Qwen3-1.7B", {"think": True}),
    ("Qwen/Qwen3-1.7B", {"think": False}),
    ("Qwen/Qwen3.5-2B", {}),
    ("sbintuitions/sarashina2.2-3b-instruct-v0.1", {}),
    ("nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese", {"think": False}),
    ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", {"think": True}),
]


def _short(repo_id: str) -> str:
    return repo_id.split("/")[-1].lower()


def resolve_model_id(name: str) -> str:
    """短い名前（"Qwen3-8B" など）を "Qwen/Qwen3-8B" に変換する. それ以外はそのまま返す."""
    if "/" in name or os.path.exists(name):
        return name
    table: Dict[str, str] = {_short(m.repo_id): m.repo_id for m in MODELS}
    if name.lower() in table:
        return table[name.lower()]
    hint = difflib.get_close_matches(name.lower(), list(table), n=3, cutoff=0.5)
    msg = f"モデル '{name}' が一覧にありません. '組織名/モデル名' の形で指定してください"
    if hint:
        msg += "（もしかして: " + ", ".join(table[h] for h in hint) + "）"
    raise ValueError(msg)


def format_model_list() -> str:
    from .adapters import get_adapter_class

    lines, family = [], None
    for m in MODELS:
        if m.family != family:
            family = m.family
            lines.append(f"\n## {family}")
        adapter = get_adapter_class(m.repo_id).name
        lines.append(f"  {m.repo_id:<52} [{adapter}] {m.note}")
    return "\n".join(lines).lstrip("\n")
