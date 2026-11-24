"""アダプタの基底クラス（＝汎用アダプタ）.

アダプタは「モデルごとの違い」をすべて引き受ける部品で、実行プログラム側
（chat.py / generate.py）はどのモデルでも同じ呼び出し方で済むようにする.

モデルごとの違いとして、このクラスで扱っているもの:

1. 読み込み時のオプション     … trust_remote_code, tokenizer の引数, vLLM の引数 など
2. システムプロンプト         … 既定のシステムプロンプト、system ロール非対応モデルへの対処
3. chat template への追加引数 … enable_thinking, reasoning_effort など（思考モードの切替）
4. 生成パラメータの推奨値     … temperature, top_p など（思考モードの有無で変わることも）
5. 出力の解析                 … 思考過程と最終回答の分離（<think> 形式 / harmony 形式）
6. 対話履歴への積み方         … 思考過程は履歴に残さない、など

新しいモデルに対応させるときは、このクラスを継承して必要な部分だけ上書きする.
何も上書きしなくても、chat template の中身から思考モードの切替方法などを自動判定するので、
多くのモデルはこの汎用アダプタのままで動く.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ..reasoning import (
    Parsed,
    detect_template_features,
    parse_harmony,
    parse_think,
    prompt_opens_think,
    remove_special_tokens,
)
from ..templates import SIMPLE_JA_TEMPLATE, merge_system_into_user
from ..types import DEFAULT_SYSTEM_PROMPT, ChatResponse, GenerationParams, Message, TemplateFeatures

# backend.render(messages, chat_template, **kwargs) -> str
RenderFn = Callable[..., str]


class BaseAdapter:
    # ------------------------------------------------------------------
    # クラス属性（サブクラスで上書きする）
    # ------------------------------------------------------------------
    name = "generic"
    description = "chat template をそのまま使う汎用アダプタ（思考モード等は自動判定）"
    # モデル ID（"組織名/モデル名"）に対する正規表現. 大文字小文字は区別しない.
    patterns: Sequence[str] = ()

    # --- 1. 読み込み時のオプション ---
    # True / False / "auto"（まず False で読み、カスタムコードが必要なら True で読み直す）
    trust_remote_code: Union[bool, str] = "auto"
    tokenizer_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    vllm_kwargs: Dict[str, Any] = {}

    # --- 2. プロンプト ---
    default_system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    # tokenizer に chat template が無いときに使うテンプレート（ベースモデル用）
    fallback_chat_template: Optional[str] = SIMPLE_JA_TEMPLATE
    # system ロールに対応しているか. None なら chat template から自動判定.
    supports_system_role: Optional[bool] = None
    # プロンプト先頭に BOS を付けるか. None なら tokenizer の設定に従って自動判定.
    add_bos: Optional[bool] = None

    # --- 3. 思考（推論）モード ---
    # 出力形式: "auto"（template から判定）/ "think"（<think>タグ）/ "harmony" / "none"
    reasoning_format = "auto"
    # 何も指定しないときにモデルが思考するか（生成パラメータの選択に使う. None=不明）
    default_thinking: Optional[bool] = None
    # reasoning_effort の既定値（None なら chat template 側の既定値に任せる）
    default_reasoning_effort: Optional[str] = None
    # 思考を無効化できないモデルで think=False が指定されたときに使う reasoning_effort
    fallback_low_effort: Optional[str] = "low"

    # --- 4. 生成パラメータ ---
    max_new_tokens = 1024
    max_new_tokens_thinking = 4096

    # --- モデルの取得 ---
    # 既定（slcllm/hub.py の DEFAULT_IGNORE_PATTERNS）に加えて取得しないファイル
    download_ignore_patterns: Sequence[str] = ()
    # GGUF (llama.cpp) の取得元. {元モデル ID: GGUF リポジトリ ID}
    gguf_repo_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def __init__(self, model_id: str, think: Optional[bool] = None, reasoning_effort: Optional[str] = None):
        self.model_id = model_id
        self.think = think
        self.reasoning_effort = reasoning_effort
        self.notes: List[str] = []
        self.chat_template: Optional[str] = None
        self.using_fallback_template = False
        self.special_tokens: List[str] = []
        self.features = TemplateFeatures()
        self._format = self.reasoning_format

    @classmethod
    def matches(cls, model_id: str) -> bool:
        return any(re.search(p, model_id, re.IGNORECASE) for p in cls.patterns)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    # ------------------------------------------------------------------
    # バックエンドがモデルを読み込んだ後に呼ばれる
    # ------------------------------------------------------------------
    def setup(self, chat_template: Optional[str], special_tokens: Sequence[str] = ()) -> None:
        template = self.override_chat_template(chat_template)
        if template is None and self.fallback_chat_template:
            template = self.fallback_chat_template
            self.using_fallback_template = True
            self.note("chat template が見つからないため簡易テンプレートを使います（ベースモデル？）")
        self.chat_template = template
        self.special_tokens = [t for t in special_tokens if t]
        self.features = detect_template_features(template)
        if self.supports_system_role is not None:
            self.features.system_role = self.supports_system_role
        if self._format == "auto":
            self._format = "harmony" if self.features.harmony else "think"
        self.check_options()

    def override_chat_template(self, chat_template: Optional[str]) -> Optional[str]:
        """chat template を差し替えたいときに上書きする（既定ではそのまま）."""
        return chat_template

    def set_thinking(self, think: Optional[bool] = None, reasoning_effort: Optional[str] = None) -> None:
        self.think = think
        self.reasoning_effort = reasoning_effort
        self.notes = [n for n in self.notes if "思考" not in n and "reasoning" not in n]
        self.check_options()

    # ------------------------------------------------------------------
    # 思考モードの扱い
    # ------------------------------------------------------------------
    @property
    def can_toggle_thinking(self) -> bool:
        f = self.features
        return f.enable_thinking or f.no_think_flag

    def check_options(self) -> None:
        """指定されたオプションがモデルで使えるか確認し、注意事項を notes に残す."""
        f = self.features
        if self.think is not None and not self.can_toggle_thinking:
            if self._format == "harmony" and f.reasoning_effort:
                if self.think is False and self.reasoning_effort is None:
                    self.note(f"このモデルは思考を完全には無効化できないため reasoning_effort={self.fallback_low_effort} で代用します")
            else:
                self.note("このモデルは思考モードの切替に対応していないため --think/--no-think は無視されます")
        if self.reasoning_effort is not None and not f.reasoning_effort:
            self.note("このモデルは reasoning_effort に対応していないため --reasoning-effort は無視されます")

    def effective_reasoning_effort(self) -> Optional[str]:
        if self.reasoning_effort is not None:
            return self.reasoning_effort
        if self.think is False and not self.can_toggle_thinking and self.features.reasoning_effort:
            return self.fallback_low_effort
        return self.default_reasoning_effort

    def is_thinking(self, prompt: str) -> Optional[bool]:
        """このプロンプトでモデルが思考するかを推定する（生成パラメータの選択用）."""
        if self._format == "harmony":
            return True
        if self.features.think_tags:
            if prompt_opens_think(prompt):
                return True
            if re.search(r"<think>\s*</think>\s*$", prompt):
                return False
        if self.think is not None and self.can_toggle_thinking:
            return self.think
        return self.default_thinking

    # ------------------------------------------------------------------
    # 1) メッセージの前処理
    # ------------------------------------------------------------------
    def prepare_messages(self, messages: List[Message]) -> List[Message]:
        msgs = [dict(m) for m in messages]
        # システムプロンプトの /think, /no_think で思考を切り替えるタイプ（Nemotron Nano v2 など）
        if self.think is not None and self.features.no_think_flag:
            flag = "/think" if self.think else "/no_think"
            if msgs and msgs[0]["role"] == "system":
                msgs[0]["content"] = (msgs[0]["content"] + "\n" + flag).strip()
            else:
                msgs.insert(0, {"role": "system", "content": flag})
        if not self.features.system_role:
            msgs = merge_system_into_user(msgs)
        return msgs

    # ------------------------------------------------------------------
    # 2) chat template に渡す追加の引数
    # ------------------------------------------------------------------
    def chat_template_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {}
        if self.think is not None and self.features.enable_thinking:
            kw["enable_thinking"] = self.think
        effort = self.effective_reasoning_effort()
        if effort is not None and self.features.reasoning_effort:
            kw["reasoning_effort"] = effort
        return kw

    def build_prompt(self, render: RenderFn, messages: List[Message]) -> str:
        msgs = self.prepare_messages(messages)
        kwargs = self.chat_template_kwargs()
        try:
            return render(msgs, self.chat_template, **kwargs)
        except Exception as e:  # system ロール非対応のテンプレートは例外を投げることがある
            if any(m["role"] == "system" for m in msgs) and "system" in str(e).lower():
                self.features.system_role = False
                self.note("このモデルは system ロールに対応していないため、システムプロンプトを最初の発話に含めます")
                return render(merge_system_into_user(msgs), self.chat_template, **kwargs)
            raise

    # ------------------------------------------------------------------
    # 3) 生成パラメータの推奨値
    # ------------------------------------------------------------------
    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        """アダプタとしての推奨値. 汎用アダプタはモデルの generation_config.json に任せる."""
        return GenerationParams()

    def default_max_new_tokens(self, thinking: Optional[bool]) -> int:
        return self.max_new_tokens_thinking if thinking else self.max_new_tokens

    # ------------------------------------------------------------------
    # 4) 出力の解析
    # ------------------------------------------------------------------
    def parse_stream(self, raw: str, prompt: str = "") -> Parsed:
        """生成途中の文字列にも使える解析（前後の空白は削らない）."""
        if self._format == "harmony":
            p = parse_harmony(raw)
        elif self._format == "none":
            p = Parsed("", raw, False)
        else:
            p = parse_think(raw, starts_in_think=prompt_opens_think(prompt))
        return Parsed(
            remove_special_tokens(p.reasoning, self.special_tokens),
            remove_special_tokens(p.content, self.special_tokens),
            p.thinking_open,
        )

    def parse_output(self, raw: str, prompt: str = "") -> Tuple[str, str]:
        """(思考過程, 最終回答) を返す."""
        p = self.parse_stream(raw, prompt)
        return p.reasoning.strip(), p.content.strip()

    # ------------------------------------------------------------------
    # 5) 対話履歴に積むメッセージ
    # ------------------------------------------------------------------
    def history_message(self, response: ChatResponse) -> Message:
        # 多くの思考モデルは「過去の思考過程は履歴に含めない」ことを推奨している
        return {"role": "assistant", "content": response.content}

    # ------------------------------------------------------------------
    # GGUF (llama.cpp) の取得元
    # ------------------------------------------------------------------
    def gguf_repos(self, model_id: str) -> List[str]:
        repos = []
        if model_id in self.gguf_repo_map:
            repos.append(self.gguf_repo_map[model_id])
        repos += [model_id + "-GGUF", model_id + "-gguf"]
        return repos

    # ------------------------------------------------------------------
    def describe(self) -> str:
        feats = ", ".join(self.features.summary()) or "-"
        return f"adapter={self.name} / format={self._format} / template features: {feats}"


class GenericAdapter(BaseAdapter):
    """どのパターンにも当てはまらないモデル用（BaseAdapter そのもの）."""
