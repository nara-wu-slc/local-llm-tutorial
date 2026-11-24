"""テスト用の極小モデル（重みはランダム）と、実モデルを模した chat template を作る.

本物のモデルをダウンロードせずに、テンプレート処理・思考モード切替・出力解析・
transformers バックエンドの一連の流れを確認するためのもの.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>",
    "<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>", "<|startoftext|>",
    "<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>",
    "<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
]

# --- Qwen3 風（enable_thinking 対応, 履歴中の思考は削除） ---
QWEN3_LIKE = """{%- for message in messages %}
{%- if message.role == 'assistant' %}
{%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}
{{- '<|im_start|>assistant\\n' + content + '<|im_end|>\\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- if enable_thinking is defined and enable_thinking is false %}
{{- '<think>\\n\\n</think>\\n\\n' }}
{%- endif %}
{%- endif %}"""

# --- Qwen3-Thinking-2507 風（生成プロンプトに <think> を付ける） ---
QWEN3_THINKING_LIKE = QWEN3_LIKE.replace(
    "{%- if enable_thinking is defined and enable_thinking is false %}\n{{- '<think>\\n\\n</think>\\n\\n' }}\n{%- endif %}",
    "{{- '<think>\\n' }}",
)

# --- gpt-oss 風（harmony, reasoning_effort） ---
HARMONY_LIKE = """{{- '<|start|>system<|message|>You are ChatGPT.\\nReasoning: ' + (reasoning_effort if reasoning_effort is defined else 'medium') + '\\n\\n# Valid channels: analysis, commentary, final.<|end|>' }}
{%- for message in messages %}
{%- if message.role == 'system' %}
{{- '<|start|>developer<|message|># Instructions\\n\\n' + message.content + '<|end|>' }}
{%- elif message.role == 'user' %}
{{- '<|start|>user<|message|>' + message.content + '<|end|>' }}
{%- elif message.role == 'assistant' %}
{{- '<|start|>assistant<|channel|>final<|message|>' + message.content + '<|end|>' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|start|>assistant' }}{%- endif %}"""

# --- Gemma 2 風（system ロール非対応） ---
GEMMA2_LIKE = """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}
{%- for message in messages %}
{%- set role = 'model' if message['role'] == 'assistant' else message['role'] %}
{{- '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"""

# --- Nemotron Nano v2 風（/think, /no_think をシステムプロンプトで判定） ---
NEMOTRON_V2_LIKE = """{%- set ns = namespace(enable_thinking=true) %}
{%- for message in messages %}
{%- if message.role == 'system' and '/no_think' in message.content %}{%- set ns.enable_thinking = false %}{%- endif %}
{%- endfor %}
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\\n' + message.content.replace('/no_think', '').replace('/think', '').strip() + '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}{%- if ns.enable_thinking %}{{- '<think>\\n' }}{%- else %}{{- '<think></think>' }}{%- endif %}
{%- endif %}"""

# --- Llama 3 風（思考なし） ---
LLAMA3_LIKE = """{{- bos_token }}
{%- for message in messages %}
{{- '<|start_header_id|>' + message.role + '<|end_header_id|>\\n\\n' + message.content | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}"""

TEMPLATES: Dict[str, Optional[str]] = {
    "qwen3-like": QWEN3_LIKE,
    "qwen3-thinking-like": QWEN3_THINKING_LIKE,
    "harmony-like": HARMONY_LIKE,
    "gemma2-like": GEMMA2_LIKE,
    "nemotron-v2-like": NEMOTRON_V2_LIKE,
    "llama3-like": LLAMA3_LIKE,
    "base-no-template": None,
}

EOS = {
    "qwen3-like": "<|im_end|>", "qwen3-thinking-like": "<|im_end|>", "harmony-like": "<|return|>",
    "gemma2-like": "<end_of_turn>", "nemotron-v2-like": "<|im_end|>", "llama3-like": "<|eot_id|>",
    "base-no-template": "<|endoftext|>",
}
BOS = {"gemma2-like": "<bos>", "llama3-like": "<|begin_of_text|>", "base-no-template": None}

CORPUS = [
    "大規模言語モデルについて簡単に教えてください。", "日本の首都は東京です。", "こんにちは、今日はいい天気ですね。",
    "The quick brown fox jumps over the lazy dog.", "思考過程と最終回答を分離します。", "奈良女子大学",
] * 20


def build_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok)


def build_all(root: str, with_model: bool = True) -> Dict[str, str]:
    """root 以下に各テンプレートの極小モデルを作り、{名前: パス} を返す."""
    paths = {}
    base_tok = build_tokenizer()
    for name, template in TEMPLATES.items():
        path = os.path.join(root, name)
        paths[name] = path
        if os.path.exists(os.path.join(path, "config.json")):
            continue
        tok = build_tokenizer()
        tok.eos_token = EOS[name]
        tok.pad_token = "<|endoftext|>"
        if BOS.get(name):
            tok.bos_token = BOS[name]
        tok.chat_template = template
        tok.save_pretrained(path)
        if with_model:
            from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

            config = LlamaConfig(vocab_size=len(base_tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                                 eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
                                 pad_token_id=tok.pad_token_id)
            model = LlamaForCausalLM(config)
            model.save_pretrained(path)
            GenerationConfig(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                             bos_token_id=tok.bos_token_id).save_pretrained(path)
    return paths


if __name__ == "__main__":
    import sys

    print(build_all(sys.argv[1] if len(sys.argv) > 1 else "tiny_models"))
