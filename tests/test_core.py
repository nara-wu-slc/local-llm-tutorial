"""モデルを使わずに確認できる部分のテスト（アダプタ選択・出力解析・表示・テンプレート処理）.

    python3 -m pytest tests/
"""

import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from slcllm.adapters import get_adapter_class  # noqa: E402
from slcllm.catalog import MODELS, resolve_model_id  # noqa: E402
from slcllm.reasoning import parse_harmony, parse_think  # noqa: E402
from slcllm.templates import render_chat_template  # noqa: E402
from slcllm.types import ChatResponse  # noqa: E402
from slcllm.ui import StreamPrinter  # noqa: E402
import tiny_models as T  # noqa: E402


# ---------------------------------------------------------------------------
# アダプタの自動選択
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_id,expected", [
    ("llm-jp/llm-jp-3-1.8b-instruct", "llm-jp-3"),
    ("llm-jp/llm-jp-3.1-13b-instruct4", "llm-jp-3"),
    ("llm-jp/llm-jp-4.1-8b-thinking", "llm-jp-4"),
    ("llm-jp/llm-jp-4.1-32b-a3b-thinking-gguf", "llm-jp-4"),
    ("openai/gpt-oss-20b", "gpt-oss"),
    ("tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1", "swallow-gpt-oss"),
    ("tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", "swallow-qwen3"),
    ("tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5", "swallow-llama"),
    ("tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1", "swallow-llama"),
    ("tokyotech-llm/Gemma-2-Llama-Swallow-9b-it-v0.1", "swallow-gemma"),
    ("Qwen/Qwen3-8B", "qwen"),
    ("Qwen/Qwen3.5-9B", "qwen"),
    ("Qwen/Qwen3.8-27B", "qwen"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2"),
    ("sbintuitions/sarashina2.2-3b-instruct-v0.1", "sarashina"),
    ("nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese", "nemotron"),
    ("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", "nemotron"),
    ("google/gemma-3-4b-it", "generic"),
])
def test_adapter_routing(model_id, expected):
    assert get_adapter_class(model_id).name == expected


def test_catalog_short_names():
    assert resolve_model_id("llm-jp-3-1.8b-instruct") == "llm-jp/llm-jp-3-1.8b-instruct"
    assert resolve_model_id("Llama-3.1-Swallow-8B-Instruct-v0.1") == "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1"
    assert resolve_model_id("someone/new-model") == "someone/new-model"
    with pytest.raises(ValueError):
        resolve_model_id("Qwen3-8")
    assert len({m.repo_id for m in MODELS}) == len(MODELS)  # 重複なし


def test_qwen_variants():
    A = get_adapter_class("Qwen/Qwen3-8B")
    assert A("Qwen/Qwen3-8B").default_thinking is True
    assert A("Qwen/Qwen3-4B-Instruct-2507").default_thinking is False
    assert A("Qwen/Qwen3.5-9B").default_thinking is False
    assert A("Qwen/Qwen3.5-27B").default_thinking is True
    assert A("Qwen/Qwen3.8-27B").variant == "qwen3.8"


# ---------------------------------------------------------------------------
# 出力の解析
# ---------------------------------------------------------------------------
def test_parse_think():
    p = parse_think("<think>\n考え中\n</think>\n\n東京です。")
    assert (p.reasoning.strip(), p.content.strip(), p.thinking_open) == ("考え中", "東京です。", False)
    p = parse_think("考え中</think>東京", starts_in_think=True)
    assert (p.reasoning, p.content) == ("考え中", "東京")
    p = parse_think("<think>途中")
    assert p.thinking_open and p.reasoning == "途中" and p.content == ""
    assert parse_think("普通の回答").content == "普通の回答"


def test_parse_harmony():
    raw = ("<|channel|>analysis<|message|>ユーザは首都を聞いている。<|end|>"
           "<|start|>assistant<|channel|>final<|message|>東京です。<|return|>")
    p = parse_harmony(raw)
    assert (p.reasoning, p.content, p.thinking_open) == ("ユーザは首都を聞いている。", "東京です。", False)
    p = parse_harmony("<|channel|>analysis<|message|>考え")
    assert p.thinking_open and p.reasoning == "考え"
    assert parse_harmony("<|channel|>analysis").content == ""


# ---------------------------------------------------------------------------
# アダプタ単体（テンプレートを描画して確認）
# ---------------------------------------------------------------------------
def make_adapter(model_id, template, think=None, effort=None, special=("<|im_end|>",)):
    a = get_adapter_class(model_id)(model_id, think=think, reasoning_effort=effort)
    a.setup(template, list(special))
    return a


def render_fn(bos="", eos=""):
    def render(messages, template, **kwargs):
        return render_chat_template(template, messages, special_tokens={"bos_token": bos, "eos_token": eos}, **kwargs)
    return render


MSGS = [{"role": "system", "content": "あなたはアシスタントです。"}, {"role": "user", "content": "日本の首都は？"}]


def test_qwen_thinking_toggle():
    a = make_adapter("Qwen/Qwen3-8B", T.QWEN3_LIKE, think=False)
    prompt = a.build_prompt(render_fn(), MSGS)
    assert prompt.endswith("<think>\n\n</think>\n\n")
    assert a.is_thinking(prompt) is False
    assert a.generation_defaults(False).temperature == 0.7

    a = make_adapter("Qwen/Qwen3-8B", T.QWEN3_LIKE)
    prompt = a.build_prompt(render_fn(), MSGS)
    assert prompt.endswith("<|im_start|>assistant\n")
    assert a.is_thinking(prompt) is True
    reasoning, content = a.parse_output("<think>\nふむ\n</think>\n\n東京です。<|im_end|>", prompt)
    assert (reasoning, content) == ("ふむ", "東京です。")


def test_thinking_only_template():
    a = make_adapter("Qwen/Qwen3-4B-Thinking-2507", T.QWEN3_THINKING_LIKE, think=False)
    prompt = a.build_prompt(render_fn(), MSGS)
    assert prompt.endswith("<think>\n")
    assert any("Thinking-2507" in n for n in a.notes)
    assert a.parse_output("考え中\n</think>\n\n東京です。", prompt) == ("考え中", "東京です。")


def test_qwen3_swallow_never_disables_thinking():
    a = make_adapter("tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2", T.QWEN3_LIKE, think=False)
    prompt = a.build_prompt(render_fn(), MSGS)
    assert "</think>" not in prompt
    assert any("Qwen3-Swallow" in n for n in a.notes)
    assert a.generation_defaults(True).temperature == 0.6


def test_gpt_oss_reasoning_effort():
    a = make_adapter("openai/gpt-oss-20b", T.HARMONY_LIKE, think=False)
    prompt = a.build_prompt(render_fn(), MSGS)
    assert "Reasoning: low" in prompt and "# Instructions" in prompt
    assert any("reasoning_effort=low" in n for n in a.notes)
    raw = ("<|channel|>analysis<|message|>考える<|end|><|start|>assistant<|channel|>final<|message|>東京です。<|return|>")
    assert a.parse_output(raw, prompt) == ("考える", "東京です。")

    a = make_adapter("openai/gpt-oss-20b", T.HARMONY_LIKE, effort="high")
    assert "Reasoning: high" in a.build_prompt(render_fn(), MSGS)


def test_gpt_oss_swallow_keeps_medium():
    a = make_adapter("tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1", T.HARMONY_LIKE, think=False)
    assert "Reasoning: medium" in a.build_prompt(render_fn(), MSGS)
    a = make_adapter("tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1", T.HARMONY_LIKE, effort="high")
    a.build_prompt(render_fn(), MSGS)
    assert any("medium でのみ" in n for n in a.notes)


def test_nemotron_v2_no_think_flag():
    a = make_adapter("nvidia/NVIDIA-Nemotron-Nano-9B-v2", T.NEMOTRON_V2_LIKE, think=False)
    prompt = a.build_prompt(render_fn(), MSGS)
    assert prompt.endswith("<think></think>")
    assert "/no_think" not in prompt  # テンプレート側で除去される
    assert a.is_thinking(prompt) is False
    assert a.generation_defaults(False).do_sample is False
    assert a.vllm_kwargs == {"mamba_ssm_cache_dtype": "float32"}


def test_llama_nemotron_v1_detailed_thinking():
    a = make_adapter("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", T.LLAMA3_LIKE, think=False)
    prompt = a.build_prompt(render_fn(bos="<|begin_of_text|>"), MSGS)
    assert "system<|end_header_id|>\n\ndetailed thinking off<|eot_id|>" in prompt
    assert "あなたはアシスタントです。\n\n日本の首都は？" in prompt  # system の内容は user 側へ
    a = make_adapter("nvidia/Llama-3.1-Nemotron-Nano-8B-v1", T.LLAMA3_LIKE)
    assert "detailed thinking on" in a.build_prompt(render_fn(), MSGS)


def test_gemma_system_role_merge():
    a = make_adapter("tokyotech-llm/Gemma-2-Llama-Swallow-9b-it-v0.1", T.GEMMA2_LIKE)
    prompt = a.build_prompt(render_fn(bos="<bos>"), MSGS)
    assert prompt.startswith("<bos><start_of_turn>user\nあなたはアシスタントです。\n\n日本の首都は？")
    # 自動判定（汎用アダプタ）でも同じになる
    g = make_adapter("someone/gemma-like", T.GEMMA2_LIKE)
    assert g.build_prompt(render_fn(bos="<bos>"), MSGS) == prompt


def test_system_role_fallback_on_exception():
    # テンプレートの見た目から判定できなくても、例外が出たら system を user に移して再試行する
    tmpl = T.GEMMA2_LIKE.replace("raise_exception('System role not supported')", "raise_exception('no system messages')")
    tmpl = tmpl.replace("raise_exception(", "raise_exception (")  # 自動判定をすり抜ける形にする
    a = make_adapter("someone/odd-model", tmpl)
    assert a.features.system_role is True
    prompt = a.build_prompt(render_fn(bos="<bos>"), MSGS)
    assert "あなたはアシスタントです。" in prompt and a.features.system_role is False


def test_llmjp3_legacy_template_for_base_model():
    a = make_adapter("llm-jp/llm-jp-3-1.8b", None)
    assert a.using_fallback_template
    prompt = a.build_prompt(render_fn(), [{"role": "user", "content": "自然言語処理とは何か"}])
    assert prompt.endswith("### 指示:\n自然言語処理とは何か\n\n### 応答:\n")


def test_generic_detects_thinking_controls():
    a = make_adapter("someone/new-reasoner", T.QWEN3_LIKE, think=False)
    assert a.chat_template_kwargs() == {"enable_thinking": False}
    a = make_adapter("someone/plain", T.LLAMA3_LIKE, think=True)
    assert any("切替に対応していない" in n for n in a.notes)


# ---------------------------------------------------------------------------
# ストリーミング表示
# ---------------------------------------------------------------------------
def stream(adapter, prompt, raw, show_thinking=True, step=1):
    out = io.StringIO()
    p = StreamPrinter(adapter, prompt, show_thinking=show_thinking, out=out, color=False)
    for i in range(0, len(raw), step):
        p(raw[i:i + step])
    p.finish()
    return out.getvalue(), p.diverged


@pytest.mark.parametrize("step", [1, 3, 7])
def test_stream_printer_think(step):
    a = make_adapter("Qwen/Qwen3-8B", T.QWEN3_LIKE)
    text, diverged = stream(a, "<|im_start|>assistant\n", "<think>\n考え中です\n</think>\n\n東京です。<|im_end|>", step=step)
    assert not diverged
    assert text == "[思考]\n考え中です\n\n[回答]\n東京です。\n"


@pytest.mark.parametrize("step", [1, 5])
def test_stream_printer_harmony(step):
    a = make_adapter("openai/gpt-oss-20b", T.HARMONY_LIKE)
    raw = "<|channel|>analysis<|message|>考える<|end|><|start|>assistant<|channel|>final<|message|>東京です。<|return|>"
    text, diverged = stream(a, "", raw, step=step)
    assert not diverged
    assert text == "[思考]\n考える\n\n[回答]\n東京です。\n"
    text, _ = stream(a, "", raw, show_thinking=False, step=step)
    assert text == "(思考中...)\n\n[回答]\n東京です。\n"


def test_stream_printer_plain_and_empty_think():
    a = make_adapter("Qwen/Qwen3-8B", T.QWEN3_LIKE)
    text, d = stream(a, "", "<think>\n\n</think>\n\n東京です。")
    assert not d and text.endswith("[回答]\n東京です。\n")
    a = make_adapter("Qwen/Qwen2.5-7B-Instruct", T.QWEN3_LIKE)
    assert stream(a, "", "東京です。<|im_end|>")[0] == "東京です。\n"


def test_history_drops_reasoning():
    a = make_adapter("Qwen/Qwen3-8B", T.QWEN3_LIKE)
    r = ChatResponse(content="東京です。", reasoning="ふむ")
    assert a.history_message(r) == {"role": "assistant", "content": "東京です。"}
