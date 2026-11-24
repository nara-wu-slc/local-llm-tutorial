"""vLLM バックエンド（高速. 大量のプロンプトをまとめて処理するときに向く）.

* プロンプトは transformers の tokenizer で作り、トークン ID 列として vLLM に渡す
  （文字列で渡すと BOS が二重に付くことがあるため）.
* オフライン推論 (vllm.LLM) はストリーミング表示に対応していないので、生成が終わってから表示する.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from ..types import GenerationParams, RawGeneration
from .base import (
    Backend,
    TextCallback,
    eos_token_ids,
    find_chat_template,
    from_pretrained_with_remote_code,
    generation_config_defaults,
    hf_encode,
    hf_render,
    load_hf_tokenizer,
    log,
    tokenizer_special_tokens,
)


class VLLMBackend(Backend):
    name = "vllm"
    supports_streaming = False

    def __init__(self, model_id: str, adapter: Any, *, dtype: str = "auto", trust_remote_code: Any = None,
                 tensor_parallel_size: int = 1, max_model_len: Optional[int] = None,
                 gpu_memory_utilization: float = 0.9, quantize: Optional[str] = None,
                 revision: Optional[str] = None, debug: bool = False, **_: Any):
        if not debug:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        from vllm import LLM

        self.adapter = adapter
        trc = adapter.trust_remote_code if trust_remote_code is None else trust_remote_code
        self.tokenizer, trc_tok = load_hf_tokenizer(model_id, trc, adapter.tokenizer_kwargs, revision)
        if trc == "auto":
            from transformers import AutoConfig

            _, trc_cfg = from_pretrained_with_remote_code(
                AutoConfig.from_pretrained, model_id, "auto", "config", **({"revision": revision} if revision else {}))
            trc = trc_tok or trc_cfg
        kwargs: Dict[str, Any] = dict(
            model=model_id, trust_remote_code=bool(trc), tensor_parallel_size=tensor_parallel_size,
            dtype=dtype, gpu_memory_utilization=gpu_memory_utilization,
        )
        if revision:
            kwargs["revision"] = revision
        if max_model_len:
            kwargs["max_model_len"] = max_model_len
        if quantize:
            log("vLLM では --quantize は使えません（量子化済みモデル（AWQ 等）を指定してください）")
        kwargs.update(adapter.vllm_kwargs)
        if debug:
            log(f"vllm.LLM({kwargs})")
        self.llm = LLM(**kwargs)
        self.chat_template = find_chat_template(self.tokenizer, model_id, revision)
        self.special_tokens = tokenizer_special_tokens(self.tokenizer)
        try:
            from transformers import GenerationConfig

            self._gen_config = GenerationConfig.from_pretrained(model_id, **({"revision": revision} if revision else {}))
        except Exception:
            self._gen_config = None

    def render(self, messages, chat_template, **kwargs) -> str:
        return hf_render(self.tokenizer, messages, chat_template, **kwargs)

    def model_generation_defaults(self) -> GenerationParams:
        return generation_config_defaults(self._gen_config)

    def _sampling_params(self, p: GenerationParams):
        from vllm import SamplingParams

        kw: Dict[str, Any] = dict(
            max_tokens=p.max_new_tokens or 1024,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            # generation_config.json の終了トークンも確実に止まるように指定する
            stop_token_ids=eos_token_ids(self.tokenizer, self._gen_config),
        )
        if p.greedy:
            kw["temperature"] = 0.0
        else:
            for name in ("temperature", "top_p", "top_k", "min_p"):
                v = getattr(p, name)
                if v is not None:
                    kw[name] = v
        if p.repetition_penalty is not None:
            kw["repetition_penalty"] = p.repetition_penalty
        if p.presence_penalty is not None:
            kw["presence_penalty"] = p.presence_penalty
        if p.seed is not None:
            kw["seed"] = p.seed
        return SamplingParams(**kw)

    def generate_batch(self, prompts: Sequence[str], params: GenerationParams) -> List[RawGeneration]:
        token_prompts = [{"prompt_token_ids": hf_encode(self.tokenizer, p, self.adapter.add_bos)} for p in prompts]
        outputs = self.llm.generate(token_prompts, self._sampling_params(params), use_tqdm=len(prompts) > 1)
        results = []
        for tp, out in zip(token_prompts, outputs):
            o = out.outputs[0]
            results.append(RawGeneration(text=o.text, prompt_tokens=len(tp["prompt_token_ids"]),
                                         output_tokens=len(o.token_ids)))
        return results

    def generate(self, prompt: str, params: GenerationParams, on_text: Optional[TextCallback] = None) -> RawGeneration:
        result = self.generate_batch([prompt], params)[0]
        if on_text:
            on_text(result.text)
        return result
