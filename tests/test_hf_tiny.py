"""transformers バックエンドを極小のランダムモデルで通しで動かすテスト（GPU 不要）.

出力の中身はでたらめだが、読み込み → プロンプト作成 → 生成 → ストリーミング → 解析 → 履歴
の流れが例外なく動くことを確認する.
"""

import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import tiny_models as T  # noqa: E402
from slcllm import LocalLLM  # noqa: E402


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    root = os.environ.get("SLCLLM_TINY_MODELS") or str(tmp_path_factory.mktemp("tiny_models"))
    return T.build_all(root)


@pytest.mark.parametrize("name", list(T.TEMPLATES))
def test_conversation_runs(tiny, name):
    llm = LocalLLM(tiny[name], generation={"max_new_tokens": 8, "seed": 0}, verbose=False)
    conv = llm.conversation()
    chunks = []
    r1 = conv.ask("日本の首都は？", on_text=chunks.append)
    r2 = conv.ask("では大阪は？")
    assert r1.output_tokens <= 8 and r2.prompt_tokens > r1.prompt_tokens
    assert "".join(chunks) == r1.raw.split(T.EOS[name])[0] or r1.raw.startswith("".join(chunks))
    assert len(conv.history) == 4 and conv.history[1]["role"] == "assistant"


def test_thinking_params_follow_prompt(tiny):
    llm = LocalLLM(tiny["qwen3-like"], adapter="qwen", think=False, verbose=False)
    prompt = llm.build_prompt(llm.to_messages("こんにちは"))
    assert prompt.endswith("<think>\n\n</think>\n\n")
    p = llm.generation_params(prompt)
    assert (p.temperature, p.top_p, p.max_new_tokens) == (0.7, 0.8, 2048)
    # ユーザ指定が最優先
    p = llm.generation_params(prompt, {"temperature": 0.0})
    assert p.greedy


def test_batch_and_complete(tiny):
    llm = LocalLLM(tiny["llama3-like"], generation={"max_new_tokens": 5}, verbose=False)
    rs = llm.chat_batch(["一つ目", "二つ目", "三つ目"])
    assert len(rs) == 3 and all(r.output_tokens <= 5 for r in rs)
    r = llm.complete("昔々あるところに")
    assert r.prompt == "昔々あるところに"


def test_bos_not_duplicated(tiny):
    llm = LocalLLM(tiny["llama3-like"], backend="dryrun", verbose=False)
    tok = llm.backend.tokenizer
    from slcllm.backends.base import hf_encode

    prompt = llm.build_prompt(llm.to_messages("テスト"))
    ids = hf_encode(tok, prompt, None)
    assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id


def test_dryrun_backend(tiny):
    llm = LocalLLM(tiny["harmony-like"], backend="dryrun", adapter="gpt-oss", reasoning_effort="high", verbose=False)
    r = llm.chat("テスト")
    assert "Reasoning: high" in r.prompt and r.output_tokens == 0


def test_processor_chat_template_fallback(tiny, tmp_path):
    """tokenizer に chat template が無く chat_template.json だけある（画像対応モデル風）場合."""
    import json
    import shutil

    d = tmp_path / "vl-like"
    shutil.copytree(tiny["qwen3-like"], d)
    (d / "chat_template.jinja").unlink(missing_ok=True)
    cfg = json.loads((d / "tokenizer_config.json").read_text())
    cfg.pop("chat_template", None)
    (d / "tokenizer_config.json").write_text(json.dumps(cfg))
    (d / "chat_template.json").write_text(json.dumps({"chat_template": T.QWEN3_LIKE}))
    llm = LocalLLM(str(d), backend="dryrun", think=False, verbose=False)
    assert not llm.adapter.using_fallback_template
    assert llm.build_prompt(llm.to_messages("テスト")).endswith("<think>\n\n</think>\n\n")


def test_user_temperature_overrides_greedy_default(tiny):
    # Nemotron の思考オフは greedy 推奨だが、ユーザが temperature を指定したらサンプリングする
    llm = LocalLLM(tiny["nemotron-v2-like"], backend="dryrun", adapter="nemotron", think=False, verbose=False)
    prompt = llm.build_prompt(llm.to_messages("テスト"))
    assert llm.generation_params(prompt).greedy
    p = llm.generation_params(prompt, {"temperature": 0.8})
    assert not p.greedy and p.temperature == 0.8
