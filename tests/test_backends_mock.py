"""vLLM / llama.cpp バックエンドの配線を、ライブラリの代わりに偽物（モック）を入れて確認するテスト.

GPU や実ライブラリが無くても、引数の受け渡しやトークン列の扱いが正しいかを確かめられる.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import tiny_models as T  # noqa: E402

pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# vLLM のモック
# ---------------------------------------------------------------------------
class _FakeSamplingParams:
    def __init__(self, **kw):
        self.kw = kw


class _FakeVLLM:
    last_init = None
    last_prompts = None
    last_params = None
    script = "<think>\nふむ\n</think>\n\n東京です。<|im_end|>"

    def __init__(self, **kw):
        _FakeVLLM.last_init = kw

    def generate(self, prompts, params, use_tqdm=False):
        _FakeVLLM.last_prompts, _FakeVLLM.last_params = prompts, params
        out = types.SimpleNamespace(text=self.script, token_ids=list(range(7)))
        return [types.SimpleNamespace(outputs=[out]) for _ in prompts]


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = _FakeVLLM
    mod.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return _FakeVLLM


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = os.environ.get("SLCLLM_TINY_MODELS") or str(tmp_path_factory.mktemp("tiny_tok"))
    return T.build_all(root, with_model=False)


def test_vllm_backend(fake_vllm, tiny):
    from slcllm import LocalLLM

    llm = LocalLLM(tiny["qwen3-like"], backend="vllm", adapter="nemotron", think=True, verbose=False,
                   generation={"presence_penalty": 0.5}, max_model_len=4096)
    r = llm.chat("日本の首都は？")
    assert (r.reasoning, r.content) == ("ふむ", "東京です。")
    assert r.output_tokens == 7
    init = fake_vllm.last_init
    assert init["max_model_len"] == 4096 and init["trust_remote_code"] is False
    kw = fake_vllm.last_params.kw
    assert kw["skip_special_tokens"] is False and kw["presence_penalty"] == 0.5
    assert "prompt_token_ids" in fake_vllm.last_prompts[0]
    rs = llm.chat_batch(["a", "b"])
    assert len(fake_vllm.last_prompts) == 2 and len(rs) == 2


def test_vllm_greedy(fake_vllm, tiny):
    from slcllm import LocalLLM

    llm = LocalLLM(tiny["llama3-like"], backend="vllm", verbose=False, generation={"do_sample": False})
    llm.chat("テスト")
    assert fake_vllm.last_params.kw["temperature"] == 0.0
    ids = fake_vllm.last_prompts[0]["prompt_token_ids"]
    tok = llm.backend.tokenizer
    assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id  # BOS が二重にならない


# ---------------------------------------------------------------------------
# llama.cpp のモック（1文字 = 1トークンの簡単なもの）
# ---------------------------------------------------------------------------
SPECIALS = ["<s>", "</s>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


class _FakeLlama:
    script = "<think>考え中</think>東京です。<|im_end|>"
    instances = []

    def __init__(self, model_path, n_gpu_layers=-1, n_ctx=0, verbose=False):
        self.model_path = model_path
        self.metadata = {"tokenizer.chat_template": T.QWEN3_LIKE, "tokenizer.ggml.add_bos_token": "false"}
        self.vocab = list(SPECIALS)
        self.tokenized = None
        _FakeLlama.instances.append(self)

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab.append(piece)
        return self.vocab.index(piece)

    def _split(self, text):
        i, out = 0, []
        while i < len(text):
            for s in SPECIALS:
                if text.startswith(s, i):
                    out.append(s)
                    i += len(s)
                    break
            else:
                out.append(text[i])
                i += 1
        return out

    def token_eos(self):
        return 3  # <|im_end|>

    def token_bos(self):
        return 0

    def tokenize(self, b, add_bos=True, special=False):
        ids = [self._id(p) for p in self._split(b.decode("utf-8"))]
        self.tokenized = ([0] if add_bos else []) + ids
        return self.tokenized

    def detokenize(self, ids, prev_tokens=None, special=False):
        return "".join(self.vocab[i] for i in ids).encode("utf-8")

    def set_seed(self, seed):
        self.seed = seed

    def generate(self, ids, **kw):
        self.gen_kw = kw
        for p in self._split(self.script):
            yield self._id(p)


@pytest.fixture
def fake_llamacpp(monkeypatch, tmp_path):
    mod = types.ModuleType("llama_cpp")
    mod.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"")
    return str(gguf)


def test_llamacpp_backend(fake_llamacpp):
    from slcllm import LocalLLM

    llm = LocalLLM(fake_llamacpp, adapter="qwen", verbose=False, generation={"seed": 3, "top_k": 20})
    assert llm.backend.name == "llamacpp"  # .gguf を渡すと自動で llama.cpp になる
    chunks = []
    r = llm.chat("日本の首都は？", on_text=chunks.append)
    assert (r.reasoning, r.content) == ("考え中", "東京です。")
    assert "".join(chunks) == "<think>考え中</think>東京です。"  # 終了トークンは表示しない
    inst = _FakeLlama.instances[-1]
    assert inst.tokenized[0] != 0  # add_bos_token=false なので BOS を付けない
    assert inst.gen_kw["top_k"] == 20 and inst.seed == 3
    assert r.prompt.startswith("<|im_start|>system\n")


def test_resolve_gguf_repo_candidates(monkeypatch):
    """元モデル ID から GGUF リポジトリの候補を作る（取得そのものは tests/test_hub.py で確認）."""
    from slcllm import hub
    from slcllm.adapters import get_adapter_class
    from slcllm.backends.llamacpp import resolve_gguf

    calls = []
    monkeypatch.setattr(hub, "ensure_gguf", lambda repos, quant, f, mode: calls.append((list(repos), quant, mode)) or "x")
    for mid in ("openai/gpt-oss-20b", "llm-jp/llm-jp-4.1-8b-thinking", "llm-jp/llm-jp-4.1-8b-thinking-gguf"):
        resolve_gguf(mid, None, get_adapter_class(mid)(mid), mode="offline")
    assert calls[0][0][0] == "ggml-org/gpt-oss-20b-GGUF"
    assert "llm-jp/llm-jp-4.1-8b-thinking-gguf" in calls[1][0]
    assert calls[2][0] == ["llm-jp/llm-jp-4.1-8b-thinking-gguf"] and calls[2][2] == "offline"
