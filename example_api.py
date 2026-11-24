#!/usr/bin/env python3
"""Python プログラムの中から使う例.

    python3 example_api.py Qwen/Qwen3-1.7B

モデル名を変えるだけで、下のコードは一切変更せずにどのモデルでも動く.
"""

import sys

from slcllm import LocalLLM

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"

# 1. モデルを読み込む（アダプタは自動選択. think / reasoning_effort はモデルが対応していれば反映される）
llm = LocalLLM(model)

# 2. 一問一答
res = llm.chat("大規模言語モデルについて一文で説明してください。")
print("思考:", res.reasoning or "(なし)")
print("回答:", res.content)
print(f"({res.output_tokens} tokens, {res.tokens_per_sec:.1f} tokens/s)")

# 3. 対話（履歴を保持して続けて質問する）
conv = llm.conversation()
print(conv.ask("奈良県の県庁所在地はどこですか？").content)
print(conv.ask("そこの有名な観光地を3つ挙げてください。").content)

# 4. 生成パラメータをその場で上書きする
print(llm.chat("好きな果物を一つ挙げてください。", temperature=1.0, max_new_tokens=64).content)

# 5. まとめて処理する（vLLM バックエンドなら一括で推論されて速い）
for r in llm.chat_batch(["1+1は？", "2+2は？", "3+3は？"]):
    print(r.content)
