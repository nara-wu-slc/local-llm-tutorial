"""GPU メモリ見積もり（slcllm/memory.py, estimate_memory.py）のテスト.

実モデルと同じ寸法の config をその場で作って見積もる（重みもネットワークも不要）.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")

from slcllm import memory  # noqa: E402
from slcllm.memory import GPU, GiB  # noqa: E402


def _save(tmp_path, name, cfg, stored_gib=None):
    d = tmp_path / name
    cfg.save_pretrained(str(d))
    if stored_gib:  # 中身の無い（sparse な）ファイルで、量子化済みの重みファイルの大きさだけを再現
        with open(d / "model.safetensors", "wb") as f:
            f.truncate(int(stored_gib * GiB))
    return str(d)


@pytest.fixture
def qwen3_8b(tmp_path):
    cfg = transformers.Qwen3Config(hidden_size=4096, intermediate_size=12288, num_hidden_layers=36,
                                   num_attention_heads=32, num_key_value_heads=8, head_dim=128, vocab_size=151936,
                                   tie_word_embeddings=False, max_position_embeddings=40960)
    cfg.dtype = "bfloat16"
    return _save(tmp_path, "Qwen3-8B-like", cfg)


@pytest.fixture
def gpt_oss_20b(tmp_path):
    if not hasattr(transformers, "GptOssConfig"):
        pytest.skip("gpt-oss には transformers 4.55 以降が必要")
    cfg = transformers.GptOssConfig(num_hidden_layers=24, num_local_experts=32, hidden_size=2880,
                                    intermediate_size=2880, num_attention_heads=64, num_key_value_heads=8,
                                    head_dim=64, vocab_size=201088, max_position_embeddings=131072)
    cfg.quantization_config = {"quant_method": "mxfp4"}
    cfg.dtype = "bfloat16"
    return _save(tmp_path, "gpt-oss-20b-like", cfg, stored_gib=12.8)


def test_dense_model(qwen3_8b):
    est = memory.estimate_memory(qwen3_8b, context=8192, download="offline")
    assert abs(est.params.total / 1e9 - 8.19) < 0.01  # 公称 8.2B
    assert est.kv.bytes_full_per_token == 2 * 36 * 8 * 128 * 2  # = 144 KiB
    rows = {r.label: r for r in est.rows}
    bf16 = rows["transformers（bf16）"]
    assert abs(bf16.weights - est.params.total * 2) < 1
    assert rows["transformers --quantize 4bit"].weights < rows["transformers --quantize 8bit"].weights < bf16.weights
    assert rows["vLLM（max-model-len 40,960）"].kv == est.kv.bytes(40960)


def test_quantized_moe_with_sliding_window(gpt_oss_20b):
    est = memory.estimate_memory(gpt_oss_20b, context=8192, download="offline")
    assert abs(est.params.total / 1e9 - 20.9) < 0.1  # 公称 21B
    assert est.quant_method == "mxfp4"
    rows = {r.label: r for r in est.rows}
    assert abs(rows["transformers（mxfp4 のまま）"].weights / GiB - 12.8) < 0.01  # ファイルサイズを使う
    assert abs(rows["transformers（bf16 に展開）"].weights - est.params.total * 2) < 1
    kv = est.kv
    assert (kv.full_layers, kv.sliding_layers, kv.window) == (12, 12, 128)
    assert kv.bytes(1000) == kv.bytes_full_per_token * 1000 + kv.bytes_sliding_per_token * 128
    assert any("MoE" in n for n in est.notes)


def test_hybrid_layers_have_no_kv():
    c = types.SimpleNamespace(num_hidden_layers=6, num_attention_heads=8, num_key_value_heads=2, head_dim=64,
                              hidden_size=512, layer_types=None, layers_block_type=None,
                              hybrid_override_pattern="M-M*-M*", max_position_embeddings=4096)
    kv = memory.kv_cache_info(c)
    assert (kv.full_layers, kv.no_kv_layers) == (2, 3)
    assert kv.bytes_full_per_token == 2 * 2 * 2 * 64 * 2


def test_fit_messages():
    big = memory.Row("transformers（bf16）", 30 * GiB, 1 * GiB, 1 * GiB)
    one = [GPU(0, "A", 48 * GiB, 40 * GiB)]
    two = [GPU(0, "A", 24 * GiB, 23 * GiB), GPU(1, "B", 24 * GiB, 23 * GiB)]
    assert memory.fit_message(big, one).startswith("✅")
    assert "2枚に分けて" in memory.fit_message(big, two)
    v = memory.Row("vLLM（max-model-len 8,192）", 30 * GiB, 2 * GiB, 1.5 * GiB)
    assert "-tp 2" in memory.fit_message(v, two)
    g = memory.Row("llama.cpp（GGUF Q4_K_M）", 30 * GiB, 1 * GiB, 0.3 * GiB)
    assert "-ngl" in memory.fit_message(g, two)
    assert memory.fit_message(big, []) == ""


def test_gguf_size_from_hub_listing(monkeypatch):
    from slcllm import hub
    from slcllm.adapters import get_adapter_class

    files = [("x-Q4_K_M.gguf", 5 * GiB, None), ("x-Q8_0.gguf", 9 * GiB, None), ("mmproj-x.gguf", 1, None)]
    monkeypatch.setattr(hub, "fetch_repo_files", lambda repo, revision=None: ("sha", files)
                        if repo == "org/x-GGUF" else (_ for _ in ()).throw(OSError("404")))
    mid = "org/x"
    size, src = memory._gguf_bytes(mid, get_adapter_class(mid)(mid), "Q8_0", None)
    assert size == 9 * GiB and "実ファイル" in src
    p = memory.ParamCount(8_000_000_000, 0, "X")
    assert memory._gguf_bytes(mid, get_adapter_class(mid)(mid), "Q8_0", p, online=False)[1] == "パラメータ数からの推定"


def test_cli(qwen3_8b, monkeypatch, capsys):
    import estimate_memory

    monkeypatch.setattr(sys, "argv", ["estimate_memory.py", "-m", qwen3_8b, "--gpu-gb", "24", "--download", "offline"])
    assert estimate_memory.main() == 0
    out = capsys.readouterr().out
    assert "パラメータ数 8.19B" in out and "✅ GPU0 に載る" in out
