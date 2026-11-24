"""Hugging Face Transformers バックエンド（既定）."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from ..types import GenerationParams, RawGeneration
from .base import (
    Backend,
    TextCallback,
    find_chat_template,
    from_pretrained_with_remote_code,
    generation_config_defaults,
    hf_encode,
    hf_render,
    load_hf_tokenizer,
    log,
    tokenizer_special_tokens,
)


def _make_streamer(tokenizer: Any, on_text: TextCallback):
    """生成されたトークンを少しずつ文字列にして on_text に渡すストリーマ."""
    from transformers.generation.streamers import BaseStreamer

    class CallbackStreamer(BaseStreamer):
        def __init__(self) -> None:
            self.prompt_skipped = False
            self.cache: List[int] = []
            self.printed = 0

        def put(self, value) -> None:
            if not self.prompt_skipped:  # 最初に渡されるのはプロンプト
                self.prompt_skipped = True
                return
            self.cache.extend(int(x) for x in value.reshape(-1).tolist())
            text = tokenizer.decode(self.cache, skip_special_tokens=False)
            if text.endswith("�"):  # マルチバイト文字の途中
                return
            if len(text) > self.printed:
                on_text(text[self.printed:])
            self.printed = len(text)
            if text.endswith("\n"):  # 改行ごとにキャッシュを捨てて再デコードのコストを抑える
                self.cache, self.printed = [], 0

        def end(self) -> None:
            if self.cache:
                text = tokenizer.decode(self.cache, skip_special_tokens=False)
                if len(text) > self.printed:
                    on_text(text[self.printed:])
            self.cache, self.printed = [], 0

    return CallbackStreamer()


def _dtype_kwarg_name() -> str:
    """transformers 4.56 以降は torch_dtype ではなく dtype."""
    import transformers
    from packaging import version

    return "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"


def _choose_auto_class(config: Any, override: Optional[str]):
    import transformers

    if override:
        return getattr(transformers, override)
    # 画像も扱えるモデル（Qwen3.5 など）は ImageTextToText 側で登録されていることがある
    candidates = ["AutoModelForCausalLM", "AutoModelForImageTextToText"]
    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        mapping = getattr(cls, "_model_mapping", None)
        try:
            if mapping is not None and type(config) in mapping:
                return cls
        except Exception:
            pass
    return transformers.AutoModelForCausalLM


class DryRunBackend(Backend):
    """tokenizer だけを読み込み、生成はしないバックエンド.

    モデル本体を落とさずに「実際にモデルへ渡されるプロンプト」を確認するのに使う:
        python3 generate.py -m Qwen/Qwen3-8B -b dryrun -d --no-think
    """

    name = "dryrun"

    def __init__(self, model_id: str, adapter: Any, *, trust_remote_code: Any = None, revision: Optional[str] = None,
                 **_: Any):
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.adapter = adapter
        trc = adapter.trust_remote_code if trust_remote_code is None else trust_remote_code
        self.tokenizer, _ = load_hf_tokenizer(model_id, trc, adapter.tokenizer_kwargs, revision)
        self.chat_template = find_chat_template(self.tokenizer, model_id, revision)
        self.special_tokens = tokenizer_special_tokens(self.tokenizer)

    def render(self, messages, chat_template, **kwargs) -> str:
        return hf_render(self.tokenizer, messages, chat_template, **kwargs)

    def generate(self, prompt: str, params: GenerationParams, on_text: Optional[TextCallback] = None) -> RawGeneration:
        ids = hf_encode(self.tokenizer, prompt, self.adapter.add_bos)
        log(f"(dryrun) prompt tokens = {len(ids)}, params = {params.to_dict()}")
        return RawGeneration(text="", prompt_tokens=len(ids), output_tokens=0)


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_id: str, adapter: Any, *, dtype: str = "auto", device_map: Any = "auto",
                 quantize: Optional[str] = None, trust_remote_code: Any = None, attn_implementation: Optional[str] = None,
                 model_class: Optional[str] = None, revision: Optional[str] = None, debug: bool = False, **_: Any):
        import torch
        from transformers import AutoConfig
        from transformers.utils import logging as hf_logging

        if not debug:
            hf_logging.set_verbosity_error()
        self.debug = debug
        self.adapter = adapter
        trc = adapter.trust_remote_code if trust_remote_code is None else trust_remote_code

        self.tokenizer, trc_tok = load_hf_tokenizer(model_id, trc, adapter.tokenizer_kwargs, revision)
        rev = {"revision": revision} if revision else {}
        config, trc_model = from_pretrained_with_remote_code(AutoConfig.from_pretrained, model_id, trc, "config", **rev)

        kwargs: Dict[str, Any] = {"device_map": device_map, _dtype_kwarg_name(): dtype, **rev}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if quantize in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type="nf4")
                if quantize == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
            )
        kwargs.update(adapter.model_kwargs)
        auto_cls = _choose_auto_class(config, model_class)
        if debug:
            log(f"{auto_cls.__name__}.from_pretrained({model_id!r}, trust_remote_code={trc_model}, {kwargs})")
        self.model = auto_cls.from_pretrained(model_id, config=config, trust_remote_code=trc_model, **kwargs)
        self.model.eval()

        self.chat_template = find_chat_template(self.tokenizer, model_id, revision)
        self.special_tokens = tokenizer_special_tokens(self.tokenizer)
        self._warned_presence = False

    # ------------------------------------------------------------------
    def render(self, messages, chat_template, **kwargs) -> str:
        return hf_render(self.tokenizer, messages, chat_template, **kwargs)

    def model_generation_defaults(self) -> GenerationParams:
        return generation_config_defaults(getattr(self.model, "generation_config", None))

    def _generate_kwargs(self, p: GenerationParams) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"max_new_tokens": p.max_new_tokens or 1024}
        if p.greedy:
            kw["do_sample"] = False
        else:
            kw["do_sample"] = True
            for name in ("temperature", "top_p", "top_k", "min_p"):
                v = getattr(p, name)
                if v is not None:
                    kw[name] = v
        if p.repetition_penalty is not None:
            kw["repetition_penalty"] = p.repetition_penalty
        if p.presence_penalty and not self._warned_presence:
            log("transformers は presence_penalty に対応していないため無視します（vLLM / llama.cpp では有効）")
            self._warned_presence = True
        return kw

    def generate(self, prompt: str, params: GenerationParams, on_text: Optional[TextCallback] = None) -> RawGeneration:
        import torch

        ids = hf_encode(self.tokenizer, prompt, self.adapter.add_bos)
        input_ids = torch.tensor([ids], device=self.model.device)
        attention_mask = torch.ones_like(input_ids)
        if params.seed is not None:
            from transformers import set_seed

            set_seed(params.seed)
        kw = self._generate_kwargs(params)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
            pad = eos[0] if isinstance(eos, (list, tuple)) else eos
        streamer = _make_streamer(self.tokenizer, on_text) if on_text else None
        with torch.inference_mode(), warnings.catch_warnings():
            if not self.debug:
                warnings.simplefilter("ignore")
            out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, streamer=streamer,
                                      pad_token_id=pad, **kw)
        gen_ids = out[0][len(ids):].tolist()
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        return RawGeneration(text=text, prompt_tokens=len(ids), output_tokens=len(gen_ids))
