"""slcllm: Hugging Face のローカル LLM を同じインターフェースで動かすための小さなライブラリ.

    from slcllm import LocalLLM
    llm = LocalLLM("llm-jp/llm-jp-4.1-8b-thinking", reasoning_effort="low")
    res = llm.chat("大規模言語モデルについて簡単に教えてください。")
    print(res.reasoning)  # 思考過程（あれば）
    print(res.content)    # 最終回答
"""

from .llm import Conversation, LocalLLM
from .types import DEFAULT_SYSTEM_PROMPT, MODEL_DEFAULT, ChatResponse, GenerationParams

__all__ = ["LocalLLM", "Conversation", "ChatResponse", "GenerationParams", "DEFAULT_SYSTEM_PROMPT", "MODEL_DEFAULT"]
__version__ = "0.1.0"
