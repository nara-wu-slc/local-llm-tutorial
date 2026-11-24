#!/usr/bin/env python3
"""一問一答・バッチ生成のサンプル（旧チュートリアルの gpt-oss sample.py に相当）.

    # 1問だけ
    python3 generate.py -m openai/gpt-oss-20b "大規模言語モデルについて簡単に教えてください。"

    # ファイルの各行をプロンプトとしてまとめて処理し、JSON Lines で保存
    python3 generate.py -m Qwen/Qwen3-8B -b vllm -i prompts.txt -o outputs.jsonl
"""

import argparse
import json
import sys

from slcllm.cli import add_common_args, build_llm, handle_list_options
from slcllm.ui import StreamPrinter, print_stats


def main() -> int:
    parser = argparse.ArgumentParser(description="ローカル LLM で文章を生成する")
    parser.add_argument("prompt", nargs="?", default="大規模言語モデルについて簡単に教えてください。",
                        help="入力文（省略時はサンプルの質問）")
    parser.add_argument("-i", "--input", help="1行1プロンプトのテキストファイル（指定するとバッチ処理）")
    parser.add_argument("-o", "--output", help="バッチ処理の結果を書き出す JSON Lines ファイル（省略時は標準出力）")
    parser.add_argument("--batch-size", type=int, default=0, help="まとめて処理する件数（0 = 全件. vLLM 向け）")
    add_common_args(parser)
    args = parser.parse_args()
    if handle_list_options(args):
        return 0

    llm = build_llm(args)

    # ---- 1問だけ ----
    if not args.input:
        if args.no_chat_template:
            printer = StreamPrinter(llm.adapter, args.prompt, parse=False)
            resp = llm.complete(args.prompt, on_text=printer)
        else:
            prompt = llm.build_prompt(llm.to_messages(args.prompt))
            if args.debug:
                print("--- prompt\n" + prompt + "\n--- output", file=sys.stderr)
            printer = StreamPrinter(llm.adapter, prompt, show_thinking=not args.hide_thinking)
            resp = llm.chat(args.prompt, on_text=printer)
        printer.finish()
        if args.debug:
            print_stats(resp)
        return 0

    # ---- バッチ処理 ----
    with open(args.input, encoding="utf-8") as f:
        prompts = [line.rstrip("\n") for line in f if line.strip()]
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    size = args.batch_size or len(prompts)
    for start in range(0, len(prompts), size):
        chunk = prompts[start:start + size]
        if args.no_chat_template:
            results = [llm.complete(p) for p in chunk]
        else:
            results = llm.chat_batch(chunk)
        for p, r in zip(chunk, results):
            record = {"model": llm.model_id, "prompt": p, "response": r.content, "reasoning": r.reasoning,
                      "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[slcllm] {min(start + size, len(prompts))}/{len(prompts)} 件完了", file=sys.stderr)
    if out is not sys.stdout:
        out.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
