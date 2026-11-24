"""Hugging Face のキャッシュ（HF_HOME）にモデルが取得済みかを確認し、無ければ取得する.

キャッシュの場所は次の順で決まる（huggingface_hub の仕様）:
  HF_HUB_CACHE が設定されていればそこ → なければ $HF_HOME/hub → なければ ~/.cache/huggingface/hub

取得モード（chat.py / generate.py / download.py の --download、LocalLLM(download=...)）:
  auto    … キャッシュに必要なファイルがそろっていればネットワークに接続せずにそれを使う.
             足りなければ不足分だけ取得する（既定）
  offline … キャッシュにあるものだけを使う. 無ければエラー（ネットワークに出られない計算ノード用）
  update  … Hub に問い合わせて、新しい版があれば取得する

取得するファイルは必要なものに絞る（gpt-oss の original/ や metal/ にある重複した重み、
GGUF・ONNX・TensorFlow 用ファイル、safetensors があるときの *.bin などは取得しない）.
"""

from __future__ import annotations

import fnmatch
import glob
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .backends.base import log

MODES = ("auto", "offline", "update")

# 推論に使わないので取得しないファイル
DEFAULT_IGNORE_PATTERNS = [
    "*.gguf", "original/*", "metal/*", "onnx/*", "*.onnx", "*.onnx_data", "openvino/*", "coreml/*",
    "*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*", "*.ot", "*.mlmodel",
]
# tokenizer だけが必要なとき（dryrun や llama.cpp の --tokenizer）に取得するファイル
TOKENIZER_ALLOW_PATTERNS = [
    "*.json", "*.jinja", "*.model", "*.tiktoken", "*.txt", "*.py", "tokenizer*", "vocab*", "merges*", "*.vocab",
]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json", "tiktoken.model",
                   "spiece.model", "sentencepiece.bpe.model"]


class DownloadError(RuntimeError):
    """モデルの確認・取得に関するエラー（chat.py などではトレースバックなしで表示する）."""


class ModelNotCachedError(DownloadError):
    pass


# ---------------------------------------------------------------------------
# キャッシュの場所
# ---------------------------------------------------------------------------
def hub_cache_dir() -> str:
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE


def cache_env_message() -> Optional[str]:
    """HF_HOME / HF_HUB_CACHE が未設定なら注意文を返す."""
    if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
        return None
    return ("HF_HOME が設定されていないため、モデルはホームディレクトリ（~/.cache/huggingface）に保存されます. "
            "研究室のマシンでは先に `source /slc/share/dot.zshrc.slc` を実行してください")


def offline_env() -> bool:
    try:
        from huggingface_hub import constants

        return bool(constants.HF_HUB_OFFLINE)
    except Exception:
        return os.environ.get("HF_HUB_OFFLINE", "0") not in ("", "0", "false", "False")


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# ファイルの選別
# ---------------------------------------------------------------------------
def _normalize(patterns: Optional[Iterable[str]]) -> List[str]:
    # huggingface_hub と同じく "dir/" は "dir/*" とみなす
    return [p + "*" if p.endswith("/") else p for p in (patterns or [])]


def select_files(files: Sequence[str], allow: Optional[Sequence[str]], ignore: Optional[Sequence[str]]) -> List[str]:
    allow, ignore = _normalize(allow), _normalize(ignore)
    out = []
    for f in files:
        if allow and not any(fnmatch.fnmatch(f, p) for p in allow):
            continue
        if any(fnmatch.fnmatch(f, p) for p in ignore):
            continue
        out.append(f)
    return out


def download_patterns(files: Sequence[str], need_weights: bool = True,
                      extra_ignore: Sequence[str] = ()) -> Tuple[Optional[List[str]], List[str]]:
    """リポジトリのファイル一覧から (allow_patterns, ignore_patterns) を決める."""
    if not need_weights:
        return list(TOKENIZER_ALLOW_PATTERNS), list(DEFAULT_IGNORE_PATTERNS) + ["*.safetensors", "*.bin", "*.pt", "*.pth"]
    ignore = list(DEFAULT_IGNORE_PATTERNS) + list(extra_ignore)
    root_files = [f for f in files if "/" not in f]
    if any(f.endswith(".safetensors") for f in root_files):
        ignore += ["*.bin", "*.pt", "*.pth"]  # safetensors があれば PyTorch 形式の重みは不要
        if any(f.startswith("model") and f.endswith(".safetensors") for f in root_files):
            ignore += ["consolidated*"]  # Mistral 形式の重複した重み
    return None, ignore


# ---------------------------------------------------------------------------
# ローカル（キャッシュ）の確認. ネットワークには接続しない.
# ---------------------------------------------------------------------------
@dataclass
class LocalStatus:
    repo_id: str
    snapshot: Optional[str] = None
    missing: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.snapshot is not None and not self.missing


def local_snapshot(repo_id: str, revision: Optional[str] = None) -> Optional[str]:
    """キャッシュにあるスナップショット（ファイルが並んだディレクトリ）のパス. 無ければ None."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id, revision=revision, local_files_only=True)
    except Exception:
        return None


def _auto_map_modules(snapshot: str) -> List[str]:
    """config.json / tokenizer_config.json の auto_map が参照する .py ファイル（カスタムコード）."""
    files = set()
    for name in ("config.json", "tokenizer_config.json"):
        path = os.path.join(snapshot, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                auto_map = json.load(f).get("auto_map") or {}
        except Exception:
            continue
        for v in auto_map.values():
            for ref in (v if isinstance(v, list) else [v]):
                if isinstance(ref, str) and "--" not in ref and "." in ref:
                    files.add(ref.rsplit(".", 1)[0] + ".py")
    return sorted(files)


def check_local(repo_id: str, revision: Optional[str] = None, need_weights: bool = True) -> LocalStatus:
    """キャッシュだけを見て、モデルを動かすのに必要なファイルがそろっているかを調べる."""
    snap = local_snapshot(repo_id, revision)
    if snap is None:
        return LocalStatus(repo_id, None, ["（未取得）"])

    def has(name: str) -> bool:
        return os.path.exists(os.path.join(snap, name))  # 実体（blob）が無いリンクは False になる

    missing = []
    if not has("config.json"):
        missing.append("config.json")
    if not any(has(f) for f in TOKENIZER_FILES):
        missing.append("tokenizer のファイル")
    if need_weights:
        index = next((f for f in ("model.safetensors.index.json", "pytorch_model.bin.index.json") if has(f)), None)
        if index:
            with open(os.path.join(snap, index), encoding="utf-8") as f:
                shards = sorted(set(json.load(f).get("weight_map", {}).values()))
            missing += [s for s in shards if not has(s)]
        elif not (glob.glob(os.path.join(snap, "*.safetensors")) or has("pytorch_model.bin")):
            missing.append("モデルの重み（*.safetensors）")
    missing += [m for m in _auto_map_modules(snap) if not has(m)]
    return LocalStatus(repo_id, snap, missing)


# ---------------------------------------------------------------------------
# Hub への問い合わせ
# ---------------------------------------------------------------------------
@dataclass
class RemotePlan:
    repo_id: str
    commit: str
    files: List[Tuple[str, int, Optional[str]]]  # (ファイル名, サイズ, blob 名)
    to_download: List[Tuple[str, int]] = field(default_factory=list)

    @property
    def download_bytes(self) -> int:
        return sum(s for _, s in self.to_download)

    @property
    def total_bytes(self) -> int:
        return sum(s for _, s, _ in self.files)


def _repo_cache_folder(repo_id: str) -> str:
    return os.path.join(hub_cache_dir(), "models--" + repo_id.replace("/", "--"))


def fetch_repo_files(repo_id: str, revision: Optional[str] = None) -> Tuple[str, List[Tuple[str, int, Optional[str]]]]:
    """Hub に問い合わせて (コミット ID, [(ファイル名, サイズ, blob 名)]) を返す."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    files = []
    for s in info.siblings or []:
        lfs = getattr(s, "lfs", None)
        blob = (getattr(lfs, "sha256", None) if lfs is not None else None) or getattr(s, "blob_id", None)
        size = getattr(s, "size", None) or (getattr(lfs, "size", 0) if lfs is not None else 0) or 0
        files.append((s.rfilename, size, blob))
    return info.sha, files


def make_plan(repo_id: str, commit: str, all_files: Sequence[Tuple[str, int, Optional[str]]],
              allow: Optional[Sequence[str]], ignore: Optional[Sequence[str]]) -> RemotePlan:
    """取得対象のファイルのうち、まだキャッシュに実体（blob）が無いものを調べる."""
    names = set(select_files([n for n, _, _ in all_files], allow, ignore))
    files = [f for f in all_files if f[0] in names]
    blobs_dir = os.path.join(_repo_cache_folder(repo_id), "blobs")
    todo = [(n, s) for n, s, b in files if not (b and os.path.exists(os.path.join(blobs_dir, b)))]
    return RemotePlan(repo_id, commit, files, todo)


def explain_hub_error(repo_id: str, e: Exception) -> str:
    name = type(e).__name__
    url = f"https://huggingface.co/{repo_id}"
    if name == "GatedRepoError":
        return (f"{repo_id} は利用規約への同意が必要なモデルです. {url} で同意したうえで、"
                "`hf auth login`（古い版では `huggingface-cli login`）でログインしてください")
    if name == "RepositoryNotFoundError":
        return f"{repo_id} が見つかりません（名前の誤り、または非公開・要ログインのリポジトリ）: {url}"
    if name == "RevisionNotFoundError":
        return f"{repo_id} に指定された revision がありません"
    if (name in ("OfflineModeIsEnabled", "ConnectionError", "ConnectTimeout", "ReadTimeout", "ProxyError")
            or "connect" in str(e).lower() or "proxy" in str(e).lower()):
        return ("Hugging Face Hub に接続できません（ネットワーク・プロキシ、または HF_HUB_OFFLINE=1 の設定を"
                f"確認してください）[{name}]")
    return f"{name}: {e}"


def _check_disk_space(nbytes: int) -> None:
    path = hub_cache_dir()
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    free = shutil.disk_usage(path).free
    if nbytes > free * 0.98:
        raise DownloadError(f"キャッシュの置き場所 {path} の空き容量が足りません（必要 {human_size(nbytes)} / 空き {human_size(free)}）")


# ---------------------------------------------------------------------------
# まとめ: 確認して、必要なら取得する
# ---------------------------------------------------------------------------
def ensure_model(repo_id: str, *, revision: Optional[str] = None, mode: str = "auto", need_weights: bool = True,
                 extra_ignore: Sequence[str] = (), verbose: bool = True) -> str:
    """モデルのローカルパス（キャッシュ内のスナップショット）を返す. 必要なら取得する."""
    if mode not in MODES:
        raise ValueError(f"download モードは {MODES} のいずれか: {mode}")
    if os.path.exists(repo_id):  # ローカルのディレクトリが指定された
        return repo_id
    if mode != "offline" and offline_env():
        mode = "offline"

    status = check_local(repo_id, revision, need_weights)
    if status.complete and mode in ("auto", "offline"):
        if verbose:
            log(f"キャッシュ済みのモデルを使います: {status.snapshot}")
        return status.snapshot  # type: ignore[return-value]
    if mode == "offline":
        detail = f"（不足: {', '.join(status.missing)}）" if status.snapshot else ""
        raise ModelNotCachedError(
            f"{repo_id} はキャッシュ（{hub_cache_dir()}）にありません{detail}. "
            f"ネットワークに出られるマシンで `python3 download.py -m {repo_id}` を実行してください")

    if verbose:
        msg = cache_env_message()
        if msg:
            log("注意: " + msg)
    try:
        commit, all_files = fetch_repo_files(repo_id, revision)
    except Exception as e:
        if status.complete:  # update モードで Hub に繋がらない場合は手元のものを使う
            log(f"更新の確認ができませんでした（{explain_hub_error(repo_id, e)}）. キャッシュ済みのものを使います")
            return status.snapshot  # type: ignore[return-value]
        raise ModelNotCachedError(explain_hub_error(repo_id, e)) from e
    allow, ignore = download_patterns([n for n, _, _ in all_files], need_weights, extra_ignore)
    plan = make_plan(repo_id, commit, all_files, allow, ignore)

    if plan.to_download:
        _check_disk_space(plan.download_bytes)
        if verbose:
            log(f"{repo_id} を取得します: {len(plan.to_download)} ファイル / {human_size(plan.download_bytes)}"
                f"（モデル全体 {human_size(plan.total_bytes)}）→ {hub_cache_dir()}")
    elif verbose:
        log(f"{repo_id} は最新の版がキャッシュ済みです")

    from huggingface_hub import snapshot_download

    # revision はブランチ名（既定 main）のまま渡す. コミット ID で取得すると refs/main が更新されず、
    # 次回 local_snapshot() がキャッシュを見つけられなくなる.
    path = snapshot_download(repo_id, revision=revision, allow_patterns=allow, ignore_patterns=ignore)
    after = check_local(repo_id, revision, need_weights)
    if after.missing and verbose:
        log(f"注意: 取得後も見つからないファイルがあります: {', '.join(after.missing)}")
    return path


# ---------------------------------------------------------------------------
# GGUF（llama.cpp）用
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"(.*)-(\d{5})-of-(\d{5})\.gguf$")
DEFAULT_QUANT = "Q4_K_M"


def _pick_gguf(names: Sequence[str], quant: Optional[str], gguf_file: Optional[str]) -> List[str]:
    """候補ファイル名から使うもの（分割ファイルなら全パート）を選ぶ. 見つからなければ空リスト.

    quant を指定しなかった場合は Q4_K_M を探し、無ければリポジトリ内の唯一の GGUF を使う
    （gpt-oss の mxfp4 版のように量子化の種類が1つしかないリポジトリ用）.
    """
    ggufs = sorted(n for n in names if n.endswith(".gguf") and "mmproj" not in n.lower())
    if gguf_file:
        cands = [n for n in ggufs if os.path.basename(n) == os.path.basename(gguf_file)]
    else:
        q = (quant or DEFAULT_QUANT).lower()
        cands = [n for n in ggufs if q in os.path.basename(n).lower()]
        if not cands and quant is None and len(ggufs) == 1:
            cands = ggufs
    if not cands:
        return []
    m = _SPLIT_RE.match(cands[0])
    if not m:
        return [cands[0]]
    return [n for n in cands if n.startswith(m.group(1) + "-") and _SPLIT_RE.match(n)]  # 分割ファイルの全パート


def ensure_gguf(repos: Sequence[str], quant: Optional[str] = None, gguf_file: Optional[str] = None, *,
                mode: str = "auto",
                verbose: bool = True) -> str:
    """GGUF ファイルのローカルパスを返す（分割ファイルなら1つ目）. 必要なら取得する."""
    if mode != "offline" and offline_env():
        mode = "offline"
    if mode in ("auto", "offline"):
        for repo in repos:
            snap = local_snapshot(repo)
            if not snap:
                continue
            names = [os.path.relpath(p, snap) for p in glob.glob(os.path.join(snap, "**", "*.gguf"), recursive=True)]
            parts = _pick_gguf(names, quant, gguf_file)
            m = _SPLIT_RE.match(parts[0]) if parts else None
            if parts and (not m or len(parts) == int(m.group(3))):
                if all(os.path.exists(os.path.join(snap, p)) for p in parts):
                    if verbose:
                        log(f"キャッシュ済みの GGUF を使います: {os.path.join(snap, parts[0])}")
                    return os.path.join(snap, parts[0])
        if mode == "offline":
            raise ModelNotCachedError(f"{gguf_file or quant or DEFAULT_QUANT} の GGUF がキャッシュにありません（{', '.join(repos)}）")

    from huggingface_hub import hf_hub_download

    errors = []
    for repo in repos:
        try:
            commit, all_files = fetch_repo_files(repo)
        except Exception as e:
            errors.append(f"{repo}: {explain_hub_error(repo, e)}")
            continue
        plan = make_plan(repo, commit, all_files, ["*.gguf"], ["*mmproj*"])
        parts = _pick_gguf([n for n, _, _ in plan.files], quant, gguf_file)
        if not parts:
            raise ModelNotCachedError(f"{repo} に {gguf_file or quant or DEFAULT_QUANT} の GGUF がありません. "
                                      f"候補: {[n for n, _, _ in plan.files]}")
        need = [(n, s) for n, s in plan.to_download if n in parts]
        if need:
            nbytes = sum(s for _, s in need)
            _check_disk_space(nbytes)
            if verbose:
                log(f"GGUF を取得します: {repo} / {', '.join(parts)}（{human_size(nbytes)}）→ {hub_cache_dir()}")
                msg = cache_env_message()
                if msg:
                    log("注意: " + msg)
        paths = [hf_hub_download(repo, p) for p in parts]
        return paths[0]
    raise ModelNotCachedError("GGUF リポジトリが見つかりません: " + " / ".join(errors))


# ---------------------------------------------------------------------------
# キャッシュ済みモデルの一覧
# ---------------------------------------------------------------------------
def list_cached_models() -> List[Any]:
    from huggingface_hub import scan_cache_dir

    cache = hub_cache_dir()
    if not os.path.isdir(cache):
        return []
    info = scan_cache_dir(cache)
    return sorted((r for r in info.repos if r.repo_type == "model"), key=lambda r: r.repo_id.lower())
