"""実行プログラムから使う共通インターフェース.

    from slcllm import LocalLLM
    llm = LocalLLM("Qwen/Qwen3-8B")
    print(llm.chat("日本の首都は？").content)

モデルが変わっても、この LocalLLM / Conversation の使い方は変わらない.
モデルごとの違いはアダプタ（slcllm/adapters/）が吸収する.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .adapters import BaseAdapter, get_adapter_class
from .backends import create_backend
from .backends.base import log
from .catalog import resolve_model_id
from .hub import ensure_model
from .types import MODEL_DEFAULT, ChatResponse, GenerationParams, Message

# ライブラリとしての最低限の既定値（モデル・アダプタ・ユーザ指定がなければこれを使う）
BASELINE_PARAMS = GenerationParams(do_sample=True, temperature=0.7, top_p=0.9)

ChatInput = Union[str, Sequence[Message]]


class LocalLLM:
    def __init__(
        self,
        model: str,
        *,
        backend: str = "hf",
        adapter: Optional[str] = None,
        think: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        system_prompt: Any = MODEL_DEFAULT,
        generation: Optional[Union[Dict[str, Any], GenerationParams]] = None,
        download: str = "auto",
        revision: Optional[str] = None,
        verbose: bool = True,
        **backend_kwargs: Any,
    ):
        self.model_id = resolve_model_id(model)
        if backend == "hf" and re.search(r"gguf", self.model_id, re.I):
            if verbose:
                log("GGUF モデルが指定されたので llama.cpp バックエンドを使います")
            backend = "llamacpp"
        adapter_cls = get_adapter_class(self.model_id, adapter)
        self.adapter: BaseAdapter = adapter_cls(self.model_id, think=think, reasoning_effort=reasoning_effort)
        if verbose:
            log(f"model={self.model_id} / backend={backend} / adapter={self.adapter.name}")
        # キャッシュ（HF_HOME）を確認し、無ければ必要なファイルだけ取得する（slcllm/hub.py）
        if backend in ("llamacpp", "llama.cpp", "gguf"):
            load_from = self.model_id
            backend_kwargs["download"] = download
        else:
            load_from = ensure_model(self.model_id, revision=revision, mode=download,
                                     need_weights=(backend != "dryrun"),
                                     extra_ignore=self.adapter.download_ignore_patterns, verbose=verbose)
        self.model_path = load_from
        self.backend = create_backend(backend, load_from, self.adapter, **backend_kwargs)
        self.adapter.setup(self.backend.chat_template, self.backend.special_tokens)
        self.system_prompt: Optional[str] = (
            self.adapter.default_system_prompt if system_prompt is MODEL_DEFAULT else system_prompt
        )
        if isinstance(generation, GenerationParams):
            self.user_params = generation
        else:
            self.user_params = GenerationParams.from_dict(generation)
        if verbose:
            log(self.adapter.describe())
            for n in self.adapter.notes:
                log("注意: " + n)

    # ------------------------------------------------------------------
    def set_thinking(self, think: Optional[bool] = None, reasoning_effort: Optional[str] = None) -> None:
        self.adapter.set_thinking(think, reasoning_effort)

    def to_messages(self, query: ChatInput, history: Sequence[Message] = (),
                    system_prompt: Any = MODEL_DEFAULT) -> List[Message]:
        if not isinstance(query, str):
            return [dict(m) for m in query]
        sp = self.system_prompt if system_prompt is MODEL_DEFAULT else system_prompt
        msgs: List[Message] = [{"role": "system", "content": sp}] if sp else []
        return msgs + [dict(m) for m in history] + [{"role": "user", "content": query}]

    def build_prompt(self, messages: Sequence[Message]) -> str:
        return self.adapter.build_prompt(self.backend.render, list(messages))

    def generation_params(self, prompt: str, overrides: Optional[Dict[str, Any]] = None) -> GenerationParams:
        """生成パラメータを「既定 < モデル同梱の設定 < アダプタ推奨 < ユーザ指定」の順で重ねて決める."""
        thinking = self.adapter.is_thinking(prompt)
        user = self.user_params.merged(GenerationParams.from_dict(overrides))
        if user.temperature and user.do_sample is None:  # temperature を指定したらサンプリングする
            user.do_sample = True
        params = BASELINE_PARAMS.merged(
            GenerationParams(max_new_tokens=self.adapter.default_max_new_tokens(thinking)),
            self.backend.model_generation_defaults(),
            self.adapter.generation_defaults(thinking),
            user,
        )
        if params.greedy:  # greedy のときはサンプリング用の値を消しておく（-d の表示を分かりやすく）
            params.do_sample = False
            params.temperature = params.top_p = params.top_k = params.min_p = None
        return params

    # ------------------------------------------------------------------
    def chat(self, query: ChatInput, *, history: Sequence[Message] = (), system_prompt: Any = MODEL_DEFAULT,
             on_text: Optional[Callable[[str], None]] = None, **gen_overrides: Any) -> ChatResponse:
        """1回分の応答を生成する. query は文字列（＋history）か、メッセージのリスト."""
        prompt = self.build_prompt(self.to_messages(query, history, system_prompt))
        return self._generate(prompt, on_text, gen_overrides)

    def complete(self, text: str, *, on_text: Optional[Callable[[str], None]] = None,
                 **gen_overrides: Any) -> ChatResponse:
        """chat template を使わず、text の続きを生成する（ベースモデル向け）."""
        return self._generate(text, on_text, gen_overrides, parse=False)

    def chat_batch(self, queries: Sequence[ChatInput], **gen_overrides: Any) -> List[ChatResponse]:
        """複数の入力をまとめて処理する（vLLM ではまとめて推論するので速い）."""
        prompts = [self.build_prompt(self.to_messages(q)) for q in queries]
        # 生成パラメータは最初のプロンプトで決める（思考の有無はどれも同じはず）
        params = self.generation_params(prompts[0], gen_overrides) if prompts else GenerationParams()
        t0 = time.perf_counter()
        raws = self.backend.generate_batch(prompts, params)
        elapsed = time.perf_counter() - t0
        return [self._to_response(p, r, params, elapsed / max(len(prompts), 1)) for p, r in zip(prompts, raws)]

    def conversation(self, system_prompt: Any = MODEL_DEFAULT) -> "Conversation":
        return Conversation(self, system_prompt)

    # ------------------------------------------------------------------
    def _generate(self, prompt: str, on_text, overrides: Dict[str, Any], parse: bool = True) -> ChatResponse:
        params = self.generation_params(prompt, overrides)
        t0 = time.perf_counter()
        raw = self.backend.generate(prompt, params, on_text)
        return self._to_response(prompt, raw, params, time.perf_counter() - t0, parse)

    def _to_response(self, prompt, raw, params, elapsed, parse: bool = True) -> ChatResponse:
        if parse:
            reasoning, content = self.adapter.parse_output(raw.text, prompt)
        else:
            reasoning, content = "", raw.text
        return ChatResponse(content=content, reasoning=reasoning, raw=raw.text, prompt=prompt,
                            prompt_tokens=raw.prompt_tokens, output_tokens=raw.output_tokens,
                            elapsed=elapsed, params=params.to_dict())


class Conversation:
    """対話履歴を保持して、続けて質問できるようにするクラス."""

    def __init__(self, llm: LocalLLM, system_prompt: Any = MODEL_DEFAULT):
        self.llm = llm
        self.system_prompt = llm.system_prompt if system_prompt is MODEL_DEFAULT else system_prompt
        self.history: List[Message] = []

    def ask(self, text: str, *, on_text: Optional[Callable[[str], None]] = None, **gen_overrides: Any) -> ChatResponse:
        resp = self.llm.chat(text, history=self.history, system_prompt=self.system_prompt,
                             on_text=on_text, **gen_overrides)
        self.history.append({"role": "user", "content": text})
        self.history.append(self.llm.adapter.history_message(resp))
        return resp

    def reset(self) -> None:
        self.history = []
