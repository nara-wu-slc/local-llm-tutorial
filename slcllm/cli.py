"""chat.py / generate.py で共通に使うコマンドライン引数."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from .types import MODEL_DEFAULT


def add_common_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("モデル")
    g.add_argument("-m", "--model", help="モデル ID（例: Qwen/Qwen3-8B）または短い名前（例: llm-jp-3-1.8b-instruct）")
    g.add_argument("-b", "--backend", default="hf", choices=["hf", "vllm", "llamacpp", "dryrun"],
                   help="推論バックエンド（既定: hf = transformers. dryrun はプロンプト確認用）")
    g.add_argument("--adapter", help="アダプタを手動で指定（通常はモデル名から自動選択）")
    g.add_argument("--download", default="auto", choices=["auto", "offline", "update"],
                   help="モデルの取得方法. auto: キャッシュになければ取得（既定）/ offline: キャッシュのみ使う / "
                        "update: 新しい版があれば取得")
    g.add_argument("--revision", help="モデルの版（ブランチ名・タグ・コミット ID）を固定する")
    g.add_argument("--list-models", action="store_true", help="登録済みモデルの一覧を表示して終了")
    g.add_argument("--list-adapters", action="store_true", help="アダプタの一覧を表示して終了")

    g = p.add_argument_group("プロンプト・思考モード")
    g.add_argument("--system", help="システムプロンプト（既定: モデルごとの既定値）")
    g.add_argument("--no-system-prompt", action="store_true", help="システムプロンプトを使わない")
    t = g.add_mutually_exclusive_group()
    t.add_argument("--think", dest="think", action="store_const", const=True, default=None, help="思考モードを有効にする")
    t.add_argument("--no-think", dest="think", action="store_const", const=False, help="思考モードを無効にする")
    g.add_argument("--reasoning-effort", help="思考の深さ（gpt-oss / LLM-jp-4 / Qwen3.8 など. 例: low, medium, high）")
    g.add_argument("--hide-thinking", action="store_true", help="思考過程を表示しない")
    g.add_argument("--no-chat-template", action="store_true", help="chat template を使わず続きを生成（ベースモデル向け）")

    g = p.add_argument_group("生成パラメータ（指定しなければモデルごとの推奨値）")
    g.add_argument("--max-tokens", "--max-new-tokens", dest="max_new_tokens", type=int)
    g.add_argument("--temperature", type=float)
    g.add_argument("--top-p", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--min-p", type=float)
    g.add_argument("--repetition-penalty", type=float)
    g.add_argument("--presence-penalty", type=float)
    g.add_argument("--greedy", action="store_true", help="サンプリングせず greedy decoding にする")
    g.add_argument("--seed", type=int)

    g = p.add_argument_group("transformers (hf) バックエンド")
    g.add_argument("--dtype", default="auto", help="auto / bfloat16 / float16 など")
    g.add_argument("--quantize", choices=["4bit", "8bit"], help="bitsandbytes で量子化して読み込む")
    g.add_argument("--trust-remote-code", choices=["auto", "yes", "no"],
                   help="リポジトリ内のコードを実行するか（既定: アダプタの設定）")

    g = p.add_argument_group("vLLM バックエンド")
    g.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    g.add_argument("--max-model-len", type=int)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.9)

    g = p.add_argument_group("llama.cpp バックエンド（GGUF）")
    g.add_argument("-q", "--quant", help="量子化の種類（既定: Q4_K_M. 無ければリポジトリ内の唯一の GGUF）")
    g.add_argument("--gguf-file", help="GGUF ファイル名またはローカルパス")
    g.add_argument("-ngl", "--n-gpu-layers", type=int, default=-1, help="GPU に載せる層数（-1 = 全部）")
    g.add_argument("-c", "--ctx-size", type=int, default=8192)
    g.add_argument("--tokenizer", help="chat template を元モデルの tokenizer から取る場合のモデル ID")

    p.add_argument("-d", "--debug", action="store_true", help="プロンプトや速度などを表示")


def handle_list_options(args: argparse.Namespace) -> bool:
    if args.list_models:
        from .catalog import format_model_list

        print(format_model_list())
        return True
    if args.list_adapters:
        from .adapters import list_adapters

        for cls in list_adapters():
            pats = ", ".join(cls.patterns) or "(該当なしのとき)"
            print(f"{cls.name:<16} {cls.description}\n{'':<16} patterns: {pats}")
        return True
    if not args.model:
        print("エラー: -m/--model でモデルを指定してください（--list-models で一覧）", file=sys.stderr)
        sys.exit(2)
    return False


def generation_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    d = {k: getattr(args, k) for k in ("max_new_tokens", "temperature", "top_p", "top_k", "min_p",
                                         "repetition_penalty", "presence_penalty", "seed")}
    if args.greedy:
        d["do_sample"] = False
    return {k: v for k, v in d.items() if v is not None}


def build_llm(args: argparse.Namespace):
    from .llm import LocalLLM

    if args.no_system_prompt:
        system_prompt: Any = None
    elif args.system is not None:
        system_prompt = args.system
    else:
        system_prompt = MODEL_DEFAULT
    trc = {"auto": "auto", "yes": True, "no": False, None: None}[args.trust_remote_code]
    backend_kwargs: Dict[str, Any] = dict(debug=args.debug, dtype=args.dtype, quantize=args.quantize)
    if trc is not None:
        backend_kwargs["trust_remote_code"] = trc
    if args.backend == "vllm":
        backend_kwargs.update(tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len,
                              gpu_memory_utilization=args.gpu_memory_utilization)
    if args.backend == "llamacpp" or "gguf" in (args.model or "").lower():
        backend_kwargs.update(quant=args.quant, gguf_file=args.gguf_file, n_gpu_layers=args.n_gpu_layers,
                              ctx_size=args.ctx_size, tokenizer=args.tokenizer)
    return LocalLLM(args.model, backend=args.backend, adapter=args.adapter, think=args.think,
                    reasoning_effort=args.reasoning_effort, system_prompt=system_prompt,
                    generation=generation_overrides(args), download=args.download, revision=args.revision,
                    **backend_kwargs)
