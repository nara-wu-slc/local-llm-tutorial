#!/usr/bin/env python3
"""モデルが HF_HOME のキャッシュに取得済みかを確認し、未取得なら取得する.

    # 確認して、無ければ取得（複数指定可. 短い名前も使える）
    python3 download.py -m Qwen/Qwen3-8B llm-jp-3-1.8b-instruct

    # 確認だけ（取得しない）. すべてそろっていれば終了コード 0
    python3 download.py -m openai/gpt-oss-20b --check

    # GGUF（llama.cpp 用）の量子化ファイルを取得
    python3 download.py -m llm-jp/llm-jp-4.1-8b-thinking -b llamacpp -q Q4_K_M

    # キャッシュ済みのモデル一覧
    python3 download.py --list

    # tools/smoke_test.py で使うモデル一式を取得
    python3 download.py --smoke-test

chat.py / generate.py もモデルを読み込む前に同じ確認を行うので、このスクリプトは
「先にまとめて取得しておきたい」「どれが取得済みか知りたい」ときに使う.
"""

import argparse
import os
import sys
from datetime import datetime

from slcllm.adapters import get_adapter_class
from slcllm.catalog import SMOKE_TEST_CASES, resolve_model_id
from slcllm.hub import (
    ModelNotCachedError,
    cache_env_message,
    check_local,
    download_patterns,
    ensure_gguf,
    ensure_model,
    explain_hub_error,
    fetch_repo_files,
    human_size,
    hub_cache_dir,
    list_cached_models,
    make_plan,
)


def print_header() -> None:
    print(f"キャッシュの場所: {hub_cache_dir()}")
    print(f"  HF_HOME={os.environ.get('HF_HOME', '(未設定)')}  HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '(未設定)')}")
    msg = cache_env_message()
    if msg:
        print("  注意: " + msg)
    print()


def cmd_list() -> int:
    repos = list_cached_models()
    if not repos:
        print("キャッシュ済みのモデルはありません")
        return 0
    total = 0
    print(f"{'モデル':<56} {'サイズ':>10} {'ファイル数':>8}  最終利用")
    for r in repos:
        last = datetime.fromtimestamp(r.last_accessed).strftime("%Y-%m-%d") if r.last_accessed else "-"
        print(f"{r.repo_id:<56} {human_size(r.size_on_disk):>10} {r.nb_files:>8}  {last}")
        total += r.size_on_disk
    print(f"\n合計 {len(repos)} モデル / {human_size(total)}")
    print("不要になったモデルは `hf cache delete`（古い版では `huggingface-cli delete-cache`）で削除できます")
    return 0


def check_one(model_id: str, args) -> bool:
    """取得済みかを表示する（取得はしない）. そろっていれば True."""
    if args.backend == "llamacpp":
        adapter = get_adapter_class(model_id)(model_id)
        repos = [model_id] if "gguf" in model_id.lower() else adapter.gguf_repos(model_id)
        try:
            path = ensure_gguf(repos, args.quant, args.gguf_file, mode="offline", verbose=False)
            print(f"✅ {model_id} [{args.quant or 'Q4_K_M'}]  キャッシュ済み: {path}")
            return True
        except ModelNotCachedError:
            print(f"❌ {model_id} [{args.quant or 'Q4_K_M'}]  未取得")
            return False

    need_weights = args.backend != "dryrun"
    status = check_local(model_id, args.revision, need_weights)
    if status.complete:
        print(f"✅ {model_id}  キャッシュ済み: {status.snapshot}")
        return True
    if status.snapshot:
        print(f"⚠️ {model_id}  一部不足: " + ", ".join(status.missing[:5]))
    else:
        print(f"❌ {model_id}  未取得")
    try:  # ネットワークに繋がれば、取得に必要な容量も表示する
        commit, files = fetch_repo_files(model_id, args.revision)
        allow, ignore = download_patterns([n for n, _, _ in files], need_weights,
                                          get_adapter_class(model_id).download_ignore_patterns)
        plan = make_plan(model_id, commit, files, allow, ignore)
        print(f"     取得が必要: {len(plan.to_download)} ファイル / {human_size(plan.download_bytes)}"
              f"（モデル全体 {human_size(plan.total_bytes)}）")
    except Exception as e:
        print(f"     （Hub に問い合わせできませんでした: {explain_hub_error(model_id, e)}）")
    return False


def fetch_one(model_id: str, args) -> bool:
    mode = "update" if args.update else "auto"
    try:
        if args.backend == "llamacpp":
            adapter = get_adapter_class(model_id)(model_id)
            repos = [model_id] if "gguf" in model_id.lower() else adapter.gguf_repos(model_id)
            path = ensure_gguf(repos, args.quant, args.gguf_file, mode=mode)
        else:
            path = ensure_model(model_id, revision=args.revision, mode=mode, need_weights=args.backend != "dryrun",
                                extra_ignore=get_adapter_class(model_id).download_ignore_patterns)
        print(f"✅ {model_id}: {path}")
        return True
    except Exception as e:
        print(f"❌ {model_id}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="モデルがキャッシュ（HF_HOME）に取得済みかを確認し、未取得なら取得する")
    ap.add_argument("-m", "--model", nargs="*", default=[], help="モデル ID または短い名前（複数可）")
    ap.add_argument("-f", "--file", help="1行に1モデルを書いたテキストファイル")
    ap.add_argument("--smoke-test", action="store_true", help="tools/smoke_test.py で使うモデル一式")
    ap.add_argument("--check", action="store_true", help="確認だけして取得しない")
    ap.add_argument("--update", action="store_true", help="キャッシュ済みでも Hub に新しい版があれば取得する")
    ap.add_argument("--list", action="store_true", help="キャッシュ済みのモデル一覧を表示")
    ap.add_argument("-b", "--backend", default="hf", choices=["hf", "vllm", "llamacpp", "dryrun"],
                    help="使う予定のバックエンド（llamacpp なら GGUF を、dryrun なら tokenizer だけを取得）")
    ap.add_argument("-q", "--quant", help="GGUF の量子化の種類（-b llamacpp のとき. 既定: Q4_K_M）")
    ap.add_argument("--gguf-file", help="GGUF のファイル名（-b llamacpp のとき）")
    ap.add_argument("--revision", help="モデルの版（ブランチ名・タグ・コミット ID）")
    args = ap.parse_args()

    print_header()
    if args.list:
        return cmd_list()

    names = list(args.model)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            names += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if args.smoke_test:
        names += [m for m, _ in SMOKE_TEST_CASES]
    names = list(dict.fromkeys(names))  # 重複を除く（順序は保つ）
    if not names:
        ap.print_help()
        return 2

    ok = True
    for name in names:
        try:
            model_id = resolve_model_id(name)
        except ValueError as e:
            print(f"❌ {e}")
            ok = False
            continue
        ok &= check_one(model_id, args) if args.check else fetch_one(model_id, args)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n中断しました（途中まで取得したファイルは、次回の実行時に続きから取得されます）")
        sys.exit(1)
