#!/usr/bin/env python3
"""対話デモ（旧チュートリアルの sample_interactive.py に相当）.

    python3 chat.py -m Qwen/Qwen3-1.7B
    python3 chat.py -m llm-jp-3-1.8b-instruct
    python3 chat.py -m openai/gpt-oss-20b --reasoning-effort low

input> というプロンプトが出たら文字列を入力し、空行のまま Enter で送信.
何も入力せずに空行のまま Enter するとプログラムを終了する.
入力欄では次のコマンドも使える:
    /reset              … 対話履歴を消す
    /think on|off|auto  … 思考モードを切り替える
    /effort low|medium|high … reasoning_effort を切り替える
"""

import argparse
import sys

try:
    import readline  # noqa: F401  （入力行の編集・履歴を有効にする）
except ImportError:
    pass

from slcllm.cli import add_common_args, build_llm, handle_list_options
from slcllm.ui import StreamPrinter, print_stats


def read_query() -> str:
    lines = []
    while True:
        value = input("input> ")
        if not value:
            break
        lines.append(value)
    return "\n".join(lines)


def handle_command(text: str, llm, conv) -> bool:
    """/ で始まるコマンドを処理したら True."""
    if not text.startswith("/"):
        return False
    cmd, *rest = text.split()
    arg = rest[0] if rest else ""
    if cmd == "/reset":
        conv.reset()
        print("(対話履歴を消去しました)")
    elif cmd == "/think":
        think = {"on": True, "off": False}.get(arg)
        llm.set_thinking(think, llm.adapter.reasoning_effort)
        print(f"(think={think})", *llm.adapter.notes, sep="\n")
    elif cmd == "/effort":
        llm.set_thinking(llm.adapter.think, arg or None)
        print(f"(reasoning_effort={arg or None})", *llm.adapter.notes, sep="\n")
    else:
        print("コマンド: /reset, /think on|off|auto, /effort low|medium|high")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="ローカル LLM と対話するデモ")
    add_common_args(parser)
    args = parser.parse_args()
    if handle_list_options(args):
        return 0

    llm = build_llm(args)
    conv = llm.conversation()
    raw_history = ""  # --no-chat-template 用

    while True:
        text = read_query()
        if text == "":
            break
        if handle_command(text, llm, conv):
            continue

        if args.no_chat_template:
            prompt = raw_history + text
            printer = StreamPrinter(llm.adapter, prompt, parse=False)
            resp = llm.complete(prompt, on_text=printer)
            raw_history = prompt + resp.content
        else:
            messages = llm.to_messages(text, conv.history, conv.system_prompt)
            prompt = llm.build_prompt(messages)
            if args.debug:
                print("--- prompt\n" + prompt + "\n--- output", file=sys.stderr)
            printer = StreamPrinter(llm.adapter, prompt, show_thinking=not args.hide_thinking)
            resp = conv.ask(text, on_text=printer)
        printer.finish()
        if args.debug:
            print_stats(resp)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        sys.exit(1)
