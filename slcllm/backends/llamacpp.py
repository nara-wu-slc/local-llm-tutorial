"""llama.cpp (llama-cpp-python) バックエンド. GGUF 量子化モデルを省メモリで動かす.

モデルの指定方法:
  * 元モデルの ID（例: llm-jp/llm-jp-4.1-8b-thinking）+ --quant Q4_K_M
      → "<ID>-gguf" / "<ID>-GGUF" などのリポジトリから該当する量子化ファイルを探して取得
  * GGUF リポジトリの ID（例: Qwen/Qwen3-8B-GGUF）+ --quant Q4_K_M
  （どちらもキャッシュ済みならそれを使い、無ければ必要なファイルだけ取得する）
  * ローカルの .gguf ファイルのパス
chat template は GGUF に埋め込まれたものを使う（--tokenizer で元モデルの tokenizer を使うことも可能）.
"""

from __future__ import annotations

import codecs
import os
import re
from typing import Any, Dict, List, Optional

from ..templates import render_chat_template
from ..types import GenerationParams, RawGeneration
from .base import Backend, TextCallback, hf_render, load_hf_tokenizer, tokenizer_chat_template

# 終了判定の予備（llama_vocab_is_eog が使えない古い llama-cpp-python 用）.
# harmony 形式の <|end|> は「思考の終わり」で生成の終わりではないので含めない.
_COMMON_STOP_PIECES = {
    "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "<|return|>", "<|call|>",
    "</s>", "<end_of_turn>",
}


def resolve_gguf(model_id: str, quant: Optional[str], adapter: Any, gguf_file: Optional[str] = None,
                 mode: str = "auto") -> str:
    """GGUF ファイルのローカルパスを返す. キャッシュに無ければ Hub から取得する（slcllm/hub.py）."""
    if os.path.isfile(model_id) and model_id.endswith(".gguf"):
        return model_id
    if gguf_file and os.path.isfile(gguf_file):
        return gguf_file
    from ..hub import ensure_gguf

    repos = [model_id] if re.search(r"gguf", model_id, re.I) else adapter.gguf_repos(model_id)
    return ensure_gguf(repos, quant, gguf_file, mode=mode)


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(self, model_id: str, adapter: Any, *, quant: Optional[str] = None, gguf_file: Optional[str] = None,
                 n_gpu_layers: int = -1, ctx_size: int = 8192, tokenizer: Optional[str] = None,
                 download: str = "auto", debug: bool = False, **_: Any):
        from llama_cpp import Llama

        self.adapter = adapter
        path = resolve_gguf(model_id, quant, adapter, gguf_file, mode=download)
        self.llm = Llama(model_path=path, n_gpu_layers=n_gpu_layers, n_ctx=ctx_size, verbose=debug)
        meta: Dict[str, Any] = getattr(self.llm, "metadata", {}) or {}
        self._eos = self.llm.token_eos()
        self._bos = self.llm.token_bos()
        self._bos_text = self._piece(self._bos) if self._bos >= 0 else ""
        self._eos_text = self._piece(self._eos) if self._eos >= 0 else ""
        self._add_bos_default = meta.get("tokenizer.ggml.add_bos_token", "true") == "true"
        self.hf_tokenizer = None
        if tokenizer:  # 元モデルの tokenizer の chat template を使う
            from ..hub import ensure_model

            tok_path = ensure_model(tokenizer, mode=download, need_weights=False)
            self.hf_tokenizer, _ = load_hf_tokenizer(tok_path, adapter.trust_remote_code, adapter.tokenizer_kwargs)
            self.chat_template = tokenizer_chat_template(self.hf_tokenizer)
        else:
            self.chat_template = meta.get("tokenizer.chat_template")
        self.special_tokens = [t for t in (self._bos_text, self._eos_text) if t]
        self._stop_pieces = _COMMON_STOP_PIECES | {self._eos_text}

    def _piece(self, token: int) -> str:
        return self.llm.detokenize([token], special=True).decode("utf-8", errors="ignore")

    def _is_eog(self, token: int) -> bool:
        if token == self._eos:
            return True
        try:  # llama-cpp-python 0.3 系
            import llama_cpp

            return bool(llama_cpp.llama_vocab_is_eog(self.llm._model.vocab, token))
        except Exception:
            return self._piece(token) in self._stop_pieces

    def render(self, messages, chat_template, **kwargs) -> str:
        if self.hf_tokenizer is not None:
            return hf_render(self.hf_tokenizer, messages, chat_template, **kwargs)
        return render_chat_template(chat_template, messages, add_generation_prompt=True,
                                    special_tokens={"bos_token": self._bos_text, "eos_token": self._eos_text}, **kwargs)

    def generate(self, prompt: str, params: GenerationParams, on_text: Optional[TextCallback] = None) -> RawGeneration:
        add_bos = self.adapter.add_bos
        if add_bos is None:
            add_bos = self._add_bos_default
        add_bos = add_bos and not (self._bos_text and prompt.startswith(self._bos_text))
        ids = self.llm.tokenize(prompt.encode("utf-8"), add_bos=add_bos, special=True)
        if params.seed is not None:
            self.llm.set_seed(params.seed)
        kw: Dict[str, Any] = {"temp": 0.0 if params.greedy else (params.temperature if params.temperature is not None else 0.8)}
        if params.top_p is not None:
            kw["top_p"] = params.top_p
        if params.top_k is not None:
            kw["top_k"] = params.top_k
        kw["min_p"] = params.min_p if params.min_p is not None else 0.0
        if params.repetition_penalty is not None:
            kw["repeat_penalty"] = params.repetition_penalty
        if params.presence_penalty is not None:
            kw["presence_penalty"] = params.presence_penalty

        max_new = params.max_new_tokens or 1024
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        chunks: List[str] = []
        n_out = 0
        for token in self.llm.generate(ids, **kw):
            if self._is_eog(token):
                piece = self._piece(token)
                chunks.append(piece)  # 終了トークンも生の出力には残す（解析時に除去する）
                break
            n_out += 1
            text = decoder.decode(self.llm.detokenize([token], special=True), final=False)
            if text:
                chunks.append(text)
                if on_text:
                    on_text(text)
            if n_out >= max_new:
                break
        tail = decoder.decode(b"", final=True)
        if tail:
            chunks.append(tail)
            if on_text:
                on_text(tail)
        return RawGeneration(text="".join(chunks), prompt_tokens=len(ids), output_tokens=n_out)
