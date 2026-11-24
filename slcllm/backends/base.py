"""推論バックエンドの共通部分.

バックエンドの役割は「文字列のプロンプトを受け取って、続きを生成する」ことだけ.
モデル固有の処理（テンプレートへの引数・出力の解析など）はアダプタ側が担当する.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..types import GenerationParams, RawGeneration

TextCallback = Callable[[str], None]


def log(message: str) -> None:
    print(f"[slcllm] {message}", file=sys.stderr, flush=True)


class Backend:
    name = "base"
    supports_streaming = True

    # 読み込み後に設定されるもの
    chat_template: Optional[str] = None
    special_tokens: List[str] = []

    def render(self, messages: List[Dict[str, str]], chat_template: Optional[str], **kwargs: Any) -> str:
        """chat template を使ってメッセージ列をプロンプト文字列にする."""
        raise NotImplementedError

    def generate(self, prompt: str, params: GenerationParams, on_text: Optional[TextCallback] = None) -> RawGeneration:
        raise NotImplementedError

    def generate_batch(self, prompts: Sequence[str], params: GenerationParams) -> List[RawGeneration]:
        """まとめて生成（既定では1件ずつ）. vLLM では一括処理する."""
        return [self.generate(p, params) for p in prompts]

    def model_generation_defaults(self) -> GenerationParams:
        """モデル同梱の generation_config.json に書かれた推奨値."""
        return GenerationParams()


# ---------------------------------------------------------------------------
# Hugging Face 系（transformers / vLLM）で共通に使う関数
# ---------------------------------------------------------------------------


def from_pretrained_with_remote_code(loader: Callable[..., Any], model_id: str, trust_remote_code: Any,
                                     what: str, **kwargs: Any) -> Tuple[Any, bool]:
    """trust_remote_code="auto" の場合、まず標準実装で読み、必要ならリポジトリのコードで読み直す."""
    if trust_remote_code in (True, False):
        return loader(model_id, trust_remote_code=trust_remote_code, **kwargs), trust_remote_code
    try:
        return loader(model_id, trust_remote_code=False, **kwargs), False
    except ValueError as e:
        if "trust_remote_code" not in str(e):
            raise
        log(f"{what} の読み込みにリポジトリ内のコードが必要なため trust_remote_code=True で読み直します")
        return loader(model_id, trust_remote_code=True, **kwargs), True


def load_hf_tokenizer(model_id: str, trust_remote_code: Any, tokenizer_kwargs: Dict[str, Any], revision: Optional[str] = None):
    from transformers import AutoTokenizer

    kwargs = dict(tokenizer_kwargs)
    if revision:
        kwargs["revision"] = revision
    try:
        return from_pretrained_with_remote_code(AutoTokenizer.from_pretrained, model_id, trust_remote_code, "tokenizer", **kwargs)
    except (TypeError, ValueError, OSError) as e:
        if not tokenizer_kwargs:
            raise
        # transformers のバージョンによっては use_fast などの引数が使えない
        log(f"tokenizer 引数 {tokenizer_kwargs} で読めなかったため引数なしで再試行します ({type(e).__name__})")
        kwargs = {"revision": revision} if revision else {}
        return from_pretrained_with_remote_code(AutoTokenizer.from_pretrained, model_id, trust_remote_code, "tokenizer", **kwargs)


def tokenizer_chat_template(tokenizer: Any) -> Optional[str]:
    t = getattr(tokenizer, "chat_template", None)
    if isinstance(t, dict):  # 複数テンプレートを持つ場合
        return t.get("default") or next(iter(t.values()), None)
    return t


def find_chat_template(tokenizer: Any, model_id: str, revision: Optional[str] = None) -> Optional[str]:
    """tokenizer の chat template を返す. 無ければ chat_template.jinja / .json（画像対応モデルの
    processor 用に置かれていることがある）を探す."""
    template = tokenizer_chat_template(tokenizer)
    if template:
        return template
    import json
    import os

    for fname in ("chat_template.jinja", "chat_template.json"):
        try:
            if os.path.isdir(model_id):
                path = os.path.join(model_id, fname)
            else:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(model_id, fname, revision=revision)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if fname.endswith(".json"):
                text = json.loads(text).get("chat_template")
            if text:
                log(f"tokenizer に chat template が無いため {fname} を使います")
                return text
        except Exception:
            continue
    return None


def eos_token_ids(tokenizer: Any, gen_config: Any = None) -> List[int]:
    ids = []
    eos = getattr(gen_config, "eos_token_id", None) if gen_config is not None else None
    for x in (eos if isinstance(eos, (list, tuple)) else [eos]) + [tokenizer.eos_token_id]:
        if isinstance(x, int) and x not in ids:
            ids.append(x)
    return ids


def tokenizer_special_tokens(tokenizer: Any) -> List[str]:
    toks = set()
    try:
        toks.update(tokenizer.all_special_tokens)
    except Exception:
        pass
    try:
        for t in tokenizer.added_tokens_decoder.values():
            if getattr(t, "special", False):
                toks.add(str(t))
    except Exception:
        pass
    # 思考タグは解析に使うので消さない
    return sorted(t for t in toks if t not in ("<think>", "</think>"))


def hf_render(tokenizer: Any, messages: List[Dict[str, str]], chat_template: Optional[str], **kwargs: Any) -> str:
    return tokenizer.apply_chat_template(
        messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True, **kwargs
    )


def hf_encode(tokenizer: Any, prompt: str, add_bos: Optional[bool]) -> List[int]:
    """プロンプトをトークン ID 列にする.

    chat template が BOS を含んでいることが多いので、ここでは特殊トークンを自動付与せずにトークン化し、
    「tokenizer が通常 BOS を付ける設定」かつ「プロンプトが BOS で始まっていない」ときだけ BOS を足す.
    """
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    if add_bos is None:
        try:
            probe = tokenizer("a")["input_ids"]
            add_bos = bos is not None and len(probe) > 0 and probe[0] == bos
        except Exception:
            add_bos = False
    if add_bos and bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    return ids


def generation_config_defaults(gen_config: Any) -> GenerationParams:
    """generation_config のうち、transformers の既定値から変更されている項目だけを取り出す."""
    if gen_config is None:
        return GenerationParams()
    try:
        diff = gen_config.to_diff_dict()
    except Exception:
        return GenerationParams()
    p = GenerationParams()
    for name in ("do_sample", "temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
        if diff.get(name) is not None:
            setattr(p, name, diff[name])
    return p
