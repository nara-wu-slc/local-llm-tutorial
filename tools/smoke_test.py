#!/usr/bin/env python3
"""研究室の GPU マシンで、各系列のモデルが一通り動くかをまとめて確認するスクリプト.

モデルごとに別プロセスで起動する（GPU メモリを確実に解放するため）. 結果は Markdown の表で出力する.

    python3 tools/smoke_test.py                      # 既定の小さめのモデル一式
    python3 tools/smoke_test.py -m Qwen/Qwen3-1.7B openai/gpt-oss-20b
    python3 tools/smoke_test.py -b vllm -o smoke_vllm.md

モデルは事前に `python3 download.py --smoke-test` で取得しておくと、ダウンロード時間が結果に混ざらない.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from slcllm.catalog import SMOKE_TEST_CASES as DEFAULT_CASES  # noqa: E402

QUESTION = "日本で一番高い山は何ですか？一文で答えてください。"


def run_child(model: str, opts: dict, backend: str, max_new_tokens: int) -> None:
    from slcllm import LocalLLM

    t0 = time.perf_counter()
    llm = LocalLLM(model, backend=backend, generation={"max_new_tokens": max_new_tokens, "seed": 0}, **opts)
    load = time.perf_counter() - t0
    r = llm.chat(QUESTION)
    print("@@RESULT@@" + json.dumps({
        "adapter": llm.adapter.name, "load_sec": load, "gen_sec": r.elapsed, "tps": r.tokens_per_sec,
        "output_tokens": r.output_tokens, "reasoning_chars": len(r.reasoning), "content": r.content,
        "notes": llm.adapter.notes,
    }, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--models", nargs="*", help="確認するモデル（省略時は既定の一式）")
    ap.add_argument("-b", "--backend", default="hf")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=1800, help="1モデルあたりの制限時間（秒）")
    ap.add_argument("-o", "--output", default="smoke_test_result.md")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--child-opts", default="{}", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        run_child(args.child, json.loads(args.child_opts), args.backend, args.max_new_tokens)
        return 0

    cases = [(m, {}) for m in args.models] if args.models else DEFAULT_CASES
    rows = ["| モデル | オプション | 結果 | adapter | 読込[s] | 生成[tok/s] | 思考 | 回答（先頭） |",
            "|---|---|---|---|---|---|---|---|"]
    for model, opts in cases:
        print(f"=== {model} {opts}", file=sys.stderr, flush=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--child", model, "--child-opts", json.dumps(opts),
               "-b", args.backend, "--max-new-tokens", str(args.max_new_tokens)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, cwd=ROOT)
            line = next((ln for ln in p.stdout.splitlines() if ln.startswith("@@RESULT@@")), None)
            if line is None:
                err = (p.stderr.strip().splitlines() or ["(no output)"])[-1]
                rows.append(f"| {model} | {opts or ''} | ❌ | | | | | {err[:120]} |")
                print(p.stderr[-3000:], file=sys.stderr)
                continue
            r = json.loads(line[len("@@RESULT@@"):])
            content = r["content"].replace("\n", " ").replace("|", "／")[:60]
            rows.append(f"| {model} | {opts or ''} | ✅ | {r['adapter']} | {r['load_sec']:.0f} | {r['tps']:.1f} | "
                        f"{'あり' if r['reasoning_chars'] else 'なし'} | {content} |")
            for n in r["notes"]:
                print("  注意: " + n, file=sys.stderr)
        except subprocess.TimeoutExpired:
            rows.append(f"| {model} | {opts or ''} | ⏱ タイムアウト | | | | | |")
    table = "\n".join(rows)
    print(table)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# smoke test ({args.backend})\n\n質問: {QUESTION}\n\n{table}\n")
    print(f"\n結果を {args.output} に保存しました", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
