"""端末表示まわり（思考過程と回答を見分けやすく表示する）."""

from __future__ import annotations

import sys
from typing import Optional, TextIO

from .adapters import BaseAdapter
from .reasoning import Parsed, holdback_partial_tag
from .types import ChatResponse

THINK_LABEL = "[思考]"
ANSWER_LABEL = "[回答]"


class StreamPrinter:
    """生成途中の文字列を受け取り、思考過程と回答を分けて少しずつ表示する.

    受け取った文字列全体をアダプタで毎回解析し直し、表示内容の「増えた分」だけを出力する.
    """

    def __init__(self, adapter: BaseAdapter, prompt: str = "", show_thinking: bool = True,
                 out: TextIO = sys.stdout, color: Optional[bool] = None, parse: bool = True):
        self.adapter = adapter
        self.prompt = prompt
        self.show_thinking = show_thinking
        self.out = out
        self.color = out.isatty() if color is None else color
        self.parse = parse
        self.raw = ""
        self.printed = ""
        self.diverged = False
        self._reasoning_started = False

    def _style(self, code: str) -> str:
        return code if self.color else ""

    def _format(self, p: Parsed) -> str:
        s = ""
        # 一度「思考」欄を表示し始めたら、思考が空で終わっても欄を残す（表示を巻き戻せないため）
        has_reasoning = bool(p.reasoning.strip()) or p.thinking_open or self._reasoning_started
        self._reasoning_started = has_reasoning
        if has_reasoning:
            if self.show_thinking:
                reasoning = p.reasoning.lstrip("\n")
                s += self._style("\033[2m") + THINK_LABEL + "\n" + reasoning
                if not p.thinking_open:
                    s += self._style("\033[0m") + ("\n" if reasoning.endswith("\n") else "\n\n")
            else:
                s += "(思考中...)"
                if not p.thinking_open:
                    s += "\n\n"
            if not p.thinking_open:
                s += ANSWER_LABEL + "\n"
        if not p.thinking_open:
            s += p.content.lstrip() if has_reasoning else p.content.lstrip("\n")
        return s

    def __call__(self, delta: str) -> None:
        self.raw += delta
        self._render(final=False)

    def _render(self, final: bool) -> None:
        p = self.adapter.parse_stream(self.raw, self.prompt) if self.parse else Parsed("", self.raw)
        text = self._format(p)
        if not final:
            text = holdback_partial_tag(text)
        elif p.thinking_open:
            text += self._style("\033[0m") + "\n(思考の途中で max_new_tokens に達しました)"
        if text.startswith(self.printed):
            self.out.write(text[len(self.printed):])
            self.printed = text
            self.out.flush()
        else:
            # 解析結果が途中で変わり、表示済みの内容と食い違った（まれ）. 最後にまとめて表示し直す.
            self.diverged = True
            if final:
                self.out.write(self._style("\033[0m") + "\n----\n" + text)
                self.printed = text

    def finish(self) -> None:
        self._render(final=True)
        self.out.write(self._style("\033[0m") + "\n")
        self.out.flush()


def print_stats(resp: ChatResponse, out: TextIO = sys.stderr) -> None:
    print(f"prompt tokens = {resp.prompt_tokens}", file=out)
    print(f"output tokens = {resp.output_tokens} ({resp.tokens_per_sec:.2f} tokens/s)", file=out)
    print(f"   total time = {resp.elapsed:.2f} s", file=out)
    print(f"       params = {resp.params}", file=out)
