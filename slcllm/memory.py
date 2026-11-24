"""実行前に GPU メモリの必要量を見積もる.

モデルの重みはダウンロードせず、config.json（数 KB）だけを使う:

1. パラメータ数 … config から「中身の無い（meta デバイス上の）モデル」を組み立てて数える.
   アーキテクチャごとの計算式を書かなくても正確に数えられる（accelerate estimate-memory と同じ方法）.
2. 重みのメモリ … パラメータ数 × 1パラメータあたりのバイト数（bf16 なら 2）.
   量子化済みのリポジトリ（MXFP4, AWQ, FP8 など）は Hub 上のファイルサイズを使う.
3. KV キャッシュ … 2(K,V) × 注意機構の層数 × KV ヘッド数 × ヘッド次元 × バイト数 × トークン数.
   スライディングウィンドウ層（gpt-oss など）は窓の長さで頭打ち、Mamba / 線形注意の層（Nemotron-H,
   Qwen3.5 など）は KV キャッシュを持たない（小さな固定サイズの状態のみ）.
4. その他 … CUDA の初期化や計算途中の値など（目安として 0.8 GiB + 重みの 5%）.

あくまで目安であり、実際の使用量はライブラリの版や入力の長さで数 GiB 程度前後する.
"""

from __future__ import annotations

import glob
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Optional

GiB = 1024 ** 3

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "float8_e4m3fn": 1, "float8_e5m2": 1}
# GGUF の量子化ごとの 1 パラメータあたりのビット数（llama.cpp の実測値に基づく目安）
GGUF_BPW = {"Q2_K": 2.96, "Q3_K_M": 3.91, "Q4_0": 4.55, "Q4_K_M": 4.85, "Q4_K_S": 4.58, "Q5_K_M": 5.69,
            "Q5_0": 5.54, "Q6_K": 6.56, "Q8_0": 8.5, "BF16": 16.0, "F16": 16.0, "MXFP4": 4.25}
CUDA_OVERHEAD = 0.8 * GiB
VLLM_OVERHEAD = 1.5 * GiB  # CUDA graph など（vLLM のログ: "CUDA graphs can take additional 1~3 GiB"）


def gib(n: float) -> str:
    return f"{n / GiB:.1f} GiB"


# ---------------------------------------------------------------------------
# GPU の情報
# ---------------------------------------------------------------------------
@dataclass
class GPU:
    index: int
    name: str
    total: float  # bytes
    free: float


def gpu_info() -> List[GPU]:
    """nvidia-smi で GPU の容量と空きを調べる（CUDA_VISIBLE_DEVICES で絞られていればそれに従う）."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, free = [x.strip() for x in line.split(",")]
        gpus.append(GPU(int(idx), name, float(total) * 1024 ** 2, float(free) * 1024 ** 2))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and all(v.strip().isdigit() for v in visible.split(",")):
        order = [int(v) for v in visible.split(",")]
        gpus = [g for i in order for g in gpus if g.index == i]
    return gpus


# ---------------------------------------------------------------------------
# パラメータ数・KV キャッシュ
# ---------------------------------------------------------------------------
@dataclass
class ParamCount:
    total: int
    linear: int  # bitsandbytes で量子化される部分（lm_head 以外の nn.Linear）
    model_class: str


def count_parameters(config: Any, trust_remote_code: bool = False) -> ParamCount:
    """meta デバイス上にモデルを組み立ててパラメータ数を数える（メモリはほぼ使わない）."""
    import torch
    import transformers

    errors = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        for cfg in (config, config.get_text_config() if hasattr(config, "get_text_config") else None):
            if cfg is None:
                continue
            try:
                with torch.device("meta"):
                    model = cls.from_config(cfg, trust_remote_code=trust_remote_code)
            except Exception as e:  # このクラスでは組み立てられない → 次を試す
                errors.append(f"{cls_name}: {type(e).__name__}")
                continue
            total = sum(p.numel() for p in model.parameters())
            linear = sum(m.weight.numel() for n, m in model.named_modules()
                         if isinstance(m, torch.nn.Linear) and not n.endswith("lm_head"))
            return ParamCount(total, linear, type(model).__name__)
    raise RuntimeError("パラメータ数を数えられませんでした: " + ", ".join(errors))


@dataclass
class KVInfo:
    full_layers: int
    sliding_layers: int
    window: Optional[int]
    no_kv_layers: int  # Mamba / 線形注意など、KV キャッシュを持たない層
    bytes_full_per_token: int
    bytes_sliding_per_token: int
    max_context: Optional[int]

    def bytes(self, tokens: int) -> int:
        sliding_tokens = min(tokens, self.window) if self.window else tokens
        return self.bytes_full_per_token * tokens + self.bytes_sliding_per_token * sliding_tokens


def kv_cache_info(config: Any, dtype_bytes: int = 2) -> KVInfo:
    c = config.get_text_config() if hasattr(config, "get_text_config") else config
    n_layers = getattr(c, "num_hidden_layers", 0) or 0
    n_heads = getattr(c, "num_attention_heads", None) or 1
    kv_heads = getattr(c, "num_key_value_heads", None) or n_heads
    head_dim = getattr(c, "head_dim", None) or (getattr(c, "hidden_size", 0) // n_heads)

    full = sliding = none = 0
    types = getattr(c, "layer_types", None) or getattr(c, "layers_block_type", None)
    pattern = getattr(c, "hybrid_override_pattern", None)
    if types:
        for t in types:
            if t in ("full_attention", "attention", "global"):
                full += 1
            elif t in ("sliding_attention", "local", "chunked_attention"):
                sliding += 1
            elif t in ("linear_attention", "mamba", "mamba2"):
                none += 1
    elif pattern:  # 古い Nemotron-H: "M-M*-..."（* が注意機構, M が Mamba）
        full, none = pattern.count("*"), pattern.count("M")
    else:
        full = n_layers
    window = getattr(c, "sliding_window", None) if sliding else None
    per_layer = 2 * kv_heads * head_dim * dtype_bytes
    if getattr(c, "kv_lora_rank", None):  # DeepSeek 系の MLA（圧縮された KV）
        per_layer = (c.kv_lora_rank + getattr(c, "qk_rope_head_dim", 0)) * dtype_bytes
    return KVInfo(full, sliding, window, none, full * per_layer, sliding * per_layer,
                  getattr(c, "max_position_embeddings", None))


def config_dtype_bytes(config: Any) -> int:
    d = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if d is None and hasattr(config, "get_text_config"):
        tc = config.get_text_config()
        d = getattr(tc, "dtype", None) or getattr(tc, "torch_dtype", None)
    return DTYPE_BYTES.get(str(d).replace("torch.", ""), 2)


# ---------------------------------------------------------------------------
# 見積もり
# ---------------------------------------------------------------------------
@dataclass
class Row:
    label: str
    weights: float
    kv: float
    overhead: float
    note: str = ""

    @property
    def total(self) -> float:
        return self.weights + self.kv + self.overhead


@dataclass
class Estimate:
    model_id: str
    params: Optional[ParamCount]
    stored_bytes: Optional[float]
    dtype_bytes: int
    quant_method: Optional[str]
    kv: KVInfo
    context: int
    rows: List[Row] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _stored_weight_bytes(model_id: str, local_path: Optional[str], online: bool = True) -> Optional[float]:
    """safetensors ファイルの合計サイズ（取得対象のものだけ）. キャッシュ → Hub の順に調べる."""
    from .hub import check_local, download_patterns, fetch_repo_files, select_files

    snap = local_path if local_path and os.path.isdir(local_path) else None
    if snap is None:
        st = check_local(model_id)
        snap = st.snapshot if st.complete else None
    if snap:
        files = [os.path.relpath(p, snap) for p in glob.glob(os.path.join(snap, "**", "*.safetensors"), recursive=True)]
        allow, ignore = download_patterns(files)
        sizes = [os.path.getsize(os.path.join(snap, f)) for f in select_files(files, allow, ignore)]
        if sizes:
            return float(sum(sizes))
    if os.path.exists(model_id) or not online:
        return None
    try:
        _, files = fetch_repo_files(model_id)
    except Exception:
        return None
    allow, ignore = download_patterns([n for n, _, _ in files])
    chosen = set(select_files([n for n, _, _ in files], allow, ignore))
    total = sum(s for n, s, _ in files if n in chosen and n.endswith(".safetensors"))
    return float(total) or None


def _gguf_bytes(model_id: str, adapter: Any, quant: str, params: Optional[ParamCount],
                online: bool = True) -> Optional[tuple]:
    """GGUF の実ファイルサイズ（Hub から）. 分からなければパラメータ数からの推定値."""
    from .hub import _pick_gguf, fetch_repo_files

    repos = [model_id] if "gguf" in model_id.lower() else adapter.gguf_repos(model_id)
    for repo in repos if online else []:
        try:
            _, files = fetch_repo_files(repo)
        except Exception:
            continue
        parts = set(_pick_gguf([n for n, _, _ in files], quant, None))
        if parts:
            return float(sum(s for n, s, _ in files if n in parts)), f"{repo} の実ファイル"
    if params and quant.upper() in GGUF_BPW:
        return params.total * GGUF_BPW[quant.upper()] / 8, "パラメータ数からの推定"
    return None


def estimate_memory(model_id: str, *, context: int = 8192, quant: str = "Q4_K_M", download: str = "auto",
                    max_model_len: Optional[int] = None) -> Estimate:
    """model_id を各バックエンドで動かすときの GPU メモリの目安を返す."""
    from transformers import AutoConfig

    from .adapters import get_adapter_class
    from .backends.base import from_pretrained_with_remote_code
    from .catalog import resolve_model_id
    from .hub import ensure_model, offline_env

    model_id = resolve_model_id(model_id)
    online = download != "offline" and not offline_env()
    adapter = get_adapter_class(model_id)(model_id)
    notes: List[str] = []

    is_gguf_repo = "gguf" in model_id.lower()
    path = ensure_model(model_id, mode=download, need_weights=False, verbose=False) if not is_gguf_repo else None
    params = None
    config = None
    if path:
        config, trc = from_pretrained_with_remote_code(AutoConfig.from_pretrained, path, adapter.trust_remote_code, "config")
        try:
            params = count_parameters(config, trust_remote_code=trc)
        except Exception as e:
            notes.append(f"パラメータ数を数えられませんでした（{e}）. ファイルサイズから見積もります")
    dtype_bytes = config_dtype_bytes(config) if config is not None else 2
    kv = kv_cache_info(config, 4 if dtype_bytes == 4 else 2) if config is not None else \
        KVInfo(0, 0, None, 0, 0, 0, None)
    qcfg = getattr(config, "quantization_config", None) if config is not None else None
    quant_method = (qcfg.get("quant_method") if isinstance(qcfg, dict) else getattr(qcfg, "quant_method", None))
    stored = _stored_weight_bytes(model_id, path, online) if path else None

    est = Estimate(model_id, params, stored, dtype_bytes, str(quant_method) if quant_method else None, kv, context,
                   notes=notes)
    kv_ctx = kv.bytes(context)

    def overhead(w: float) -> float:
        return CUDA_OVERHEAD + 0.05 * w

    # --- transformers ---
    if path:
        if quant_method:
            if stored:
                est.rows.append(Row(f"transformers（{quant_method} のまま）", stored, kv_ctx, overhead(stored)))
            if str(quant_method).lower() == "mxfp4" and params:
                w = params.total * 2
                est.rows.append(Row("transformers（bf16 に展開）", w, kv_ctx, overhead(w),
                                    "kernels / triton が無い・GPU が非対応のとき"))
        else:
            w = params.total * dtype_bytes if params else stored
            if w:
                label = {4: "float32", 2: "bf16"}.get(dtype_bytes, f"{dtype_bytes}byte")
                est.rows.append(Row(f"transformers（{label}）", w, kv_ctx, overhead(w)))
                if dtype_bytes == 4:
                    notes.append("このモデルは float32 で保存されています. --dtype bfloat16 を付けると重みは半分になります")
        if params and not quant_method:
            other = params.total - params.linear
            for bits, label in ((4, "--quantize 4bit"), (8, "--quantize 8bit")):
                w = params.linear * (0.53 if bits == 4 else 1.02) + other * 2
                est.rows.append(Row(f"transformers {label}", w, kv_ctx, overhead(w)))
            if params.linear < 0.5 * params.total:
                notes.append("重みの大半が nn.Linear 以外（MoE の専門家など）のため、--quantize ではあまり小さくなりません")

        # --- vLLM ---
        w = stored if (quant_method and stored) else (params.total * dtype_bytes if params else stored)
        mml = max_model_len or kv.max_context or context
        if w:
            est.rows.append(Row(f"vLLM（max-model-len {mml:,}）", w, kv.bytes(mml), VLLM_OVERHEAD))
            notes.append("vLLM の行は最低限必要な量です. 実際の vLLM は GPU 容量の 90%（--gpu-memory-utilization）を"
                         "確保し、余った分を KV キャッシュに使います")
            if max_model_len is None and kv.max_context and kv.max_context > context:
                notes.append(f"vLLM は既定でモデルの最大長（{kv.max_context:,} トークン）分の KV キャッシュが必要です. "
                             f"載らないときは --max-model-len {context} のように短くしてください")

    # --- llama.cpp ---
    g = _gguf_bytes(model_id, adapter, quant, params, online)
    if g:
        size, src = g
        est.rows.append(Row(f"llama.cpp（GGUF {quant}）", size, kv.bytes(context), 0.3 * GiB, src))

    if kv.no_kv_layers:
        notes.append(f"{kv.no_kv_layers} 層は Mamba / 線形注意で、KV キャッシュの代わりに小さな固定サイズの状態を持ちます")
    if kv.sliding_layers and kv.window:
        notes.append(f"{kv.sliding_layers} 層はスライディングウィンドウ（{kv.window:,} トークン）なので KV キャッシュが頭打ちになります")
    if config is not None:
        tc = config.get_text_config() if hasattr(config, "get_text_config") else config
        n_exp = max([getattr(tc, k, 0) or 0 for k in ("num_experts", "num_local_experts", "n_routed_experts")] + [0])
        if n_exp > 1:
            notes.append(f"MoE モデル（専門家 {n_exp} 個）です. 1トークンの計算に使うのは一部でも、"
                         "すべての専門家の重みを GPU に載せる必要があります")
    else:
        notes.append("GGUF リポジトリを指定したため KV キャッシュは計算していません（元モデルの ID を指定すると計算します）")
    return est


# ---------------------------------------------------------------------------
# GPU に載るかの判定
# ---------------------------------------------------------------------------
def fit_message(row: Row, gpus: List[GPU]) -> str:
    if not gpus:
        return ""
    need = row.total
    best = max(gpus, key=lambda g: g.free)
    if row.label.startswith("vLLM"):
        per_gpu = min(g.total for g in gpus) * 0.9
        for tp in (1, 2, 4, 8):
            if tp > len(gpus):
                break
            if (row.weights + row.kv) / tp + row.overhead <= per_gpu:
                return "✅ 1枚で動く" if tp == 1 else f"⚠️ -tp {tp}（{tp}枚に分割）なら動く"
        return "❌ 載らない（--max-model-len を小さく / 量子化モデルを使う）"
    if need <= best.free:
        return f"✅ GPU{best.index} に載る"
    if row.label.startswith("transformers") and need <= sum(g.free for g in gpus) * 0.95 and len(gpus) > 1:
        return f"⚠️ {len(gpus)}枚に分けて載る（device_map=auto）"
    if row.label.startswith("llama.cpp"):
        frac = max(0.0, (best.free - row.overhead - row.kv) / row.weights)
        return f"⚠️ 約 {int(frac * 100)}% の層だけ GPU に（-ngl で調整）" if frac > 0.05 else "❌ 載らない"
    return "❌ 載らない"


def _width(text: str) -> int:
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    space = " " * max(0, width - _width(text))
    return space + text if right else text + space


def format_estimate(est: Estimate, gpus: List[GPU]) -> str:
    lines = [f"■ {est.model_id}"]
    if est.params:
        lines.append(f"  パラメータ数 {est.params.total / 1e9:.2f}B（{est.params.model_class}）"
                     + (f" / 量子化済み: {est.quant_method}" if est.quant_method else ""))
    kv = est.kv
    if kv.bytes_full_per_token or kv.bytes_sliding_per_token:
        lens = [4096, 32768] + ([kv.max_context] if kv.max_context and kv.max_context > 32768 else [])
        extra = f"（＋スライディング層 {kv.window:,} トークン分）" if kv.bytes_sliding_per_token and kv.window else ""
        lines.append(f"  KV キャッシュ {kv.bytes_full_per_token / 1024:.0f} KiB/トークン{extra}: "
                     + " / ".join(f"{n // 1024}K→{gib(kv.bytes(n))}" for n in lens))
    lines.append("")
    w = max([_width(r.label) for r in est.rows] + [10])
    cols = [("重み", 10), (f"KV({est.context:,})", 12), ("その他", 9), ("合計", 10)]
    lines.append("  " + _pad("", w) + "".join(_pad(h, n, right=True) for h, n in cols))
    for r in est.rows:
        vals = [gib(r.weights), gib(r.kv), gib(r.overhead), gib(r.total)]
        line = "  " + _pad(r.label, w) + "".join(_pad(v, n, right=True) for v, (_, n) in zip(vals, cols))
        fit = fit_message(r, gpus)
        lines.append(line + (f"  {fit}" if fit else "") + (f"  ※{r.note}" if r.note else ""))
    for n in est.notes:
        lines.append("  注意: " + n)
    return "\n".join(lines)
