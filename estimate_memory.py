#!/usr/bin/env python3
"""モデルを動かすのに必要な GPU メモリを、実行前に見積もる.

重みはダウンロードせず、config.json など数十 KB のファイルだけを使う.
GPU があれば（nvidia-smi で）空き容量と比べて、載るかどうかも表示する.

    python3 estimate_memory.py -m Qwen/Qwen3-8B
    python3 estimate_memory.py -m openai/gpt-oss-20b llm-jp-4.1-8b-thinking --context 32768
    python3 estimate_memory.py --smoke-test             # smoke test で使うモデル一式
    python3 estimate_memory.py -m Qwen/Qwen3-32B --gpu-gb 24 --gpu-gb 24   # 別のマシン（24GB×2）を想定
"""

import argparse
import sys

from slcllm.catalog import SMOKE_TEST_CASES
from slcllm.hub import DownloadError
from slcllm.memory import GPU, GiB, estimate_memory, format_estimate, gpu_info


def main() -> int:
    ap = argparse.ArgumentParser(description="実行前に GPU メモリの必要量を見積もる")
    ap.add_argument("-m", "--model", nargs="*", default=[], help="モデル ID または短い名前（複数可）")
    ap.add_argument("--smoke-test", action="store_true", help="tools/smoke_test.py で使うモデル一式")
    ap.add_argument("--context", type=int, default=8192,
                    help="想定する会話の長さ（入力＋出力のトークン数. 既定 8192）")
    ap.add_argument("-q", "--quant", default="Q4_K_M", help="llama.cpp の行で想定する GGUF の量子化（既定 Q4_K_M）")
    ap.add_argument("--max-model-len", type=int, help="vLLM の行で想定する --max-model-len（既定: モデルの最大長）")
    ap.add_argument("--gpu-gb", type=float, action="append",
                    help="GPU の容量（GiB）を手で指定（複数回指定で複数枚）. nvidia-smi の無いマシンや別マシンの想定に")
    ap.add_argument("--download", default="auto", choices=["auto", "offline"],
                    help="config.json などの小さなファイルを取得してよいか（offline: キャッシュのみ）")
    args = ap.parse_args()

    names = list(args.model) + ([m for m, _ in SMOKE_TEST_CASES] if args.smoke_test else [])
    names = list(dict.fromkeys(names))
    if not names:
        ap.print_help()
        return 2

    if args.gpu_gb:
        gpus = [GPU(i, "(指定)", g * GiB, g * GiB) for i, g in enumerate(args.gpu_gb)]
    else:
        gpus = gpu_info()
    if gpus:
        print("GPU: " + " / ".join(f"[{g.index}] {g.name} {g.total / GiB:.0f} GiB（空き {g.free / GiB:.1f} GiB）"
                                   for g in gpus))
    else:
        print("GPU: 見つかりません（nvidia-smi が使えない環境では --gpu-gb 24 のように容量を指定できます）")
    print("※ 見積もりは目安です（ライブラリの版・入力の長さ・バッチの大きさで数 GiB 程度前後します）\n")

    ok = True
    for name in names:
        try:
            est = estimate_memory(name, context=args.context, quant=args.quant, download=args.download,
                                  max_model_len=args.max_model_len)
            print(format_estimate(est, gpus) + "\n")
        except (DownloadError, ValueError, RuntimeError, OSError) as e:
            print(f"■ {name}\n  見積もれませんでした: {e}\n")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
