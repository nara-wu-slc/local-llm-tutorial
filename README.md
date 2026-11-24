# Lab4SLC ローカル LLM チュートリアル（統合版）

Hugging Face で公開されているさまざまな LLM を、**同じプログラム・同じ使い方**で研究室のマシン上で動かすためのチュートリアルです。
これまで提供元ごとに分かれていた次のリポジトリを 1 つにまとめたものです。

* [llm-jp_tutorial](https://github.com/nara-wu-slc/llm-jp_tutorial)（LLM-jp）
* [gpt-oss-tutorial](https://github.com/nara-wu-slc/gpt-oss-tutorial)（gpt-oss）
* [Swallow_tutorial](https://github.com/nara-wu-slc/Swallow_tutorial)（Swallow）

上の 3 系列に加えて、**Qwen**（Qwen3 / Qwen3.5 / Qwen3.8）、**Sarashina**（SB Intuitions）、**NVIDIA Nemotron** にも対応しています。
一覧にないモデルも、`組織名/モデル名` を指定すればそのまま動くことが多いです（後述の「汎用アダプタ」）。

```bash
python3 chat.py -m llm-jp/llm-jp-4.1-8b-thinking
python3 chat.py -m openai/gpt-oss-20b --reasoning-effort low
python3 chat.py -m Qwen/Qwen3-8B --no-think
python3 chat.py -m nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese
```

---

## 0. しくみ

モデルごとに異なる部分（chat template への引数、思考モードの切替方法、推奨パラメータ、出力の形式など）を
**アダプタ**（`slcllm/adapters/` 以下のモジュール）に閉じ込め、実行プログラムからは共通のインターフェース
`LocalLLM` だけを使うようにしています。モデル名から適切なアダプタが自動的に選ばれます。

```
chat.py / generate.py / 自分のプログラム
        │  LocalLLM("モデル名").chat("質問")   ← どのモデルでも同じ
        ▼
   slcllm.LocalLLM
     ├─ アダプタ（モデルごとの違いを吸収）  slcllm/adapters/
     │    llmjp.py  gpt_oss.py  swallow.py  qwen.py  sarashina.py  nvidia.py  base.py(汎用)
     ├─ バックエンド（推論エンジン）        slcllm/backends/
     │    hf（transformers, 既定） / vllm / llamacpp（GGUF） / dryrun（プロンプト確認用）
     ├─ モデルの取得（キャッシュ確認・不足分の取得）  slcllm/hub.py, download.py
     └─ GPU メモリの見積もり                        slcllm/memory.py, estimate_memory.py
```

アダプタが担当していること:

| 項目 | 例 |
|---|---|
| 読み込みオプション | LLM-jp-4 は `trust_remote_code=True`、Sarashina2 は `use_fast=False`、Nemotron は vLLM で `mamba_ssm_cache_dtype=float32` |
| システムプロンプト | LLM-jp-3 はモデルカードの「以下は、タスクを説明する指示です。…」を既定に、Gemma 系は system ロール非対応なので最初の発話に移す |
| 思考モードの切替 | Qwen3 は `enable_thinking`、gpt-oss / LLM-jp-4 は `reasoning_effort`、Nemotron Nano v2 は `/no_think`、Llama-Nemotron v1 は `detailed thinking on/off` |
| 推奨生成パラメータ | Qwen3 の思考オン: temperature=0.6, top_p=0.95, top_k=20 ／ 思考オフ: 0.7, 0.8, 20 など |
| 出力の解析 | `<think>…</think>` 形式と harmony 形式（`<|channel|>analysis` / `final`）から、思考過程と最終回答を分離 |
| 対話履歴 | 過去の思考過程は履歴に残さない（各モデルの推奨） |

---

## 1. リポジトリの取得と仮想環境の作成

```bash
# 作業ディレクトリに移動 (/path/to/dir は好きなところで良い、相対パスでもOK)
cd /path/to/dir

# 本チュートリアルのリポジトリをクローンして移動
git clone https://github.com/nara-wu-slc/local-llm-tutorial.git
cd local-llm-tutorial

# 本チュートリアル用のPython仮想環境を作成して有効化
python3 -m venv .venv
source .venv/bin/activate
```

Python 3.10 以上が必要です。
`uv` を使う場合は `uv venv --python 3.12 && source .venv/bin/activate` とし、以下の `pip3 install` を `uv pip install` に読み替えてください。

## 2. ライブラリのインストール

### (A) Transformers（基本. まずはこれだけで OK）

```bash
# PyTorch をインストール（研究室の CUDA に合わせて cu124 / cu126 / cu128 などを選ぶ）
pip3 install torch --index-url https://download.pytorch.org/whl/cu124

# Transformers などをインストール
pip3 install -r requirements.txt
```

* **Qwen3.5 / Qwen3.8** は新しい transformers（5 系）が必要です: `pip3 install -U transformers`
* **gpt-oss** を MXFP4 量子化のまま動かす場合は `pip3 install -U kernels triton` も入れてください
  （入れない場合は bf16 に展開されるため、20b でも GPU メモリを 40GB 程度使います）
* bitsandbytes で 4bit / 8bit 量子化して読み込む（`--quantize 4bit`）場合は `pip3 install bitsandbytes`

### (B) vLLM（任意. 大量のプロンプトをまとめて処理したいとき）

```bash
pip3 install vllm   # PyTorch もまとめて入る
```

### (C) llama.cpp / GGUF（任意. GPU メモリを節約したいとき）

```bash
pip3 install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
# wheel が見つからない場合はソースから CUDA 有効でビルド
CMAKE_ARGS="-DGGML_CUDA=on" pip3 install llama-cpp-python --no-cache-dir
```

## 3. キャッシュディレクトリの設定

```bash
# モデル保存用のキャッシュディレクトリの設定を読み込む
# これをしないと自分のホームディレクトリ配下にバカデカいファイルを読み込んでしまって無駄が多い
source /slc/share/dot.zshrc.slc
```

### モデルの取得状況の確認と事前取得（`download.py`）

`chat.py` / `generate.py` は、モデルを読み込む前に**キャッシュ（`HF_HOME`）に必要なファイルがそろっているかを自動で確認**し、
そろっていればネットワークに接続せずにそれを使い、足りなければ**不足分だけ**を取得します。
大きなモデルを先にまとめて取得しておきたいときや、どれが取得済みかを知りたいときは `download.py` を使います。

```bash
# 取得済みか確認だけする（取得はしない. 未取得なら必要な容量も表示）
python3 download.py -m Qwen/Qwen3-8B openai/gpt-oss-20b --check

# 確認して、無ければ取得する（複数指定可. 短い名前も使える）
python3 download.py -m Qwen/Qwen3-8B llm-jp-3-1.8b-instruct

# GGUF（llama.cpp 用）の量子化ファイルを取得する
python3 download.py -m llm-jp/llm-jp-4.1-8b-thinking -b llamacpp -q Q4_K_M

# キャッシュ済みのモデル一覧（サイズ・最終利用日）
python3 download.py --list

# tools/smoke_test.py で使うモデル一式を取得する
python3 download.py --smoke-test
```

```text
キャッシュの場所: /path/to/cache/hub
  HF_HOME=/path/to/cache  HF_HUB_CACHE=(未設定)

✅ Qwen/Qwen3-8B  キャッシュ済み: /path/to/cache/hub/models--Qwen--Qwen3-8B/snapshots/...
❌ openai/gpt-oss-20b  未取得
     取得が必要: 12 ファイル / 12.8 GB（モデル全体 12.8 GB）
```

（上の容量などの数値は表示のイメージです）

* **取得するファイルは推論に必要なものだけ**です。gpt-oss の `original/`・`metal/` にある重複した重み、safetensors があるときの `*.bin`、
  GGUF・ONNX・TensorFlow 用のファイルなどは取得しません（GGUF は `-b llamacpp` のときだけ、指定した量子化のファイルを取得します）。
* 取得前に空き容量を確認し、足りなければ取得せずに止まります。`HF_HOME` が未設定のときは、ホームディレクトリに保存される旨の注意が出ます。
* 取得の仕方は `--download` で変えられます（`chat.py` / `generate.py` / `LocalLLM(download=...)` 共通）。

| `--download` | 動作 |
|---|---|
| `auto`（既定） | キャッシュに必要なファイルがそろっていればそれを使う（ネットワークに接続しない）。足りなければ不足分を取得 |
| `offline` | キャッシュにあるものだけを使う。無ければエラー（ネットワークに出られない計算ノード用。`HF_HUB_OFFLINE=1` でも同じ） |
| `update` | Hub に新しい版があれば取得する（`download.py` では `--update`） |

* 利用規約への同意が必要なモデル（gated model）は、モデルのページで同意したうえで `hf auth login`（古い版では `huggingface-cli login`）でログインしてください。
* 研究で使うモデルの版を固定したいときは `--revision`（コミット ID やタグ）を指定します。
* 不要になったモデルは `hf cache delete`（古い版では `huggingface-cli delete-cache`）で削除できます。

### 必要な GPU メモリの見積もり（`estimate_memory.py`）

モデルを動かす前に、必要な GPU メモリの目安を確認できます。重みはダウンロードせず、`config.json` など数十 KB のファイルだけを使います。
GPU があれば（`nvidia-smi` で調べた）空き容量と比べて、載るかどうかも表示します。

```bash
python3 estimate_memory.py -m Qwen/Qwen3-8B openai/gpt-oss-20b
python3 estimate_memory.py -m Qwen/Qwen3-8B --context 32768        # 長い会話を想定
python3 estimate_memory.py -m Qwen/Qwen3-32B --gpu-gb 24 --gpu-gb 24  # 別のマシン（24GB×2枚）を想定
python3 estimate_memory.py --smoke-test                             # smoke test で使うモデル一式
```

出力例（Qwen3-8B と同じ寸法の設定で、24 GiB の GPU 2 枚を想定）:

```text
■ Qwen/Qwen3-8B
  パラメータ数 8.19B（Qwen3ForCausalLM）
  KV キャッシュ 144 KiB/トークン: 4K→0.6 GiB / 32K→4.5 GiB / 40K→5.6 GiB

                                    重み  KV(32,768)   その他      合計
  transformers（bf16）          15.3 GiB     4.5 GiB  1.6 GiB  21.3 GiB  ✅ GPU0 に載る
  transformers --quantize 4bit   5.7 GiB     4.5 GiB  1.1 GiB  11.3 GiB  ✅ GPU0 に載る
  transformers --quantize 8bit   8.9 GiB     4.5 GiB  1.2 GiB  14.7 GiB  ✅ GPU0 に載る
  vLLM（max-model-len 40,960）  15.3 GiB     5.6 GiB  1.5 GiB  22.4 GiB  ⚠️ -tp 2（2枚に分割）なら動く
  llama.cpp（GGUF Q4_K_M）       4.6 GiB     4.5 GiB  0.3 GiB   9.4 GiB  ✅ GPU0 に載る  ※パラメータ数からの推定
```

見積もりの方法:

| 項目 | 求め方 |
|---|---|
| 重み | config から中身の無い（meta デバイス上の）モデルを組み立ててパラメータ数を数え、× 2 バイト（bf16）。量子化済みのリポジトリ（MXFP4・AWQ・FP8 など）は Hub 上のファイルサイズ |
| `--quantize 4bit/8bit` | `nn.Linear` の重みだけが量子化される（bitsandbytes の仕様）として計算 |
| KV キャッシュ | 2(K,V) × 注意機構の層数 × KV ヘッド数 × ヘッド次元 × 2 バイト × トークン数。スライディングウィンドウ層は窓の長さで頭打ち、Mamba / 線形注意の層は KV キャッシュなし |
| GGUF | Hub 上の実ファイルサイズ（取得できなければパラメータ数 × 量子化のビット数からの推定） |
| その他 | CUDA の初期化や計算途中の値など。目安として 0.8 GiB + 重みの 5%（vLLM は 1.5 GiB） |

* **あくまで目安**です。ライブラリの版、入力の長さ、バッチの大きさで数 GiB 程度前後します。ぎりぎりのときは余裕を見てください。
* **MoE モデル**（Qwen3-30B-A3B、gpt-oss など）は、1トークンの計算に使うのが一部の専門家でも、**すべての専門家の重みを GPU に載せる**必要があります。
  また専門家の重みは `nn.Linear` ではないことが多く、`--quantize 4bit` ではほとんど小さくなりません（GGUF 版を使うのが有効です）。
* **gpt-oss** は MXFP4 のまま載れば 20b で約 13 GiB ですが、`kernels` / `triton` が無い・GPU が対応していない場合は bf16 に展開されて約 40 GiB になります。両方の行を表示します。
* **vLLM** は既定でモデルの最大長（Qwen3.5 なら 262,144 トークンなど）分の KV キャッシュを確保しようとするため、載らないときは `--max-model-len` で短くしてください。

---

## 4. 対話デモ（`chat.py`）

プログラムの立ち上げが終わると `input>` というプロンプトが出てくるので文字列を入力します。
空行のまま Enter すると入力終了となり応答が生成されます（何も入力せずに空行のまま Enter するとプログラムを閉じます）。

```bash
# CUDA_VISIBLE_DEVICES=0 は1枚目のGPUだけを使う、というおまじない
env CUDA_VISIBLE_DEVICES=0 python3 ./chat.py -m Qwen/Qwen3-1.7B
```

思考するモデルでは、思考過程が `[思考]`（薄い色）、最終回答が `[回答]` として表示されます（以下は表示のイメージ）。

```text
input> 日本で一番高い山は？
input>
[思考]
ユーザは日本で一番高い山を聞いている。富士山で、標高は3776m。…

[回答]
日本で一番高い山は富士山（標高3,776m）です。
input>
```

入力欄では次のコマンドも使えます。

| コマンド | 内容 |
|---|---|
| `/reset` | 対話履歴を消す |
| `/think on` / `/think off` / `/think auto` | 思考モードを切り替える（対応モデルのみ） |
| `/effort low` / `/effort high` | reasoning_effort を切り替える（gpt-oss, LLM-jp-4 など） |

### モデル別の実行例

```bash
# LLM-jp（旧 llm-jp_tutorial と同じく短い名前でも指定できる）
python3 chat.py -m llm-jp-3-1.8b-instruct
python3 chat.py -m llm-jp/llm-jp-4.1-8b-thinking --reasoning-effort low

# gpt-oss（旧 gpt-oss-tutorial）
python3 chat.py -m openai/gpt-oss-20b

# Swallow（旧 Swallow_tutorial. vLLM で動かす場合は -b vllm）
python3 chat.py -m Llama-3.1-Swallow-8B-Instruct-v0.5 -b vllm
python3 chat.py -m tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2
python3 chat.py -m tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1

# Qwen
python3 chat.py -m Qwen/Qwen3-8B --no-think        # 思考なしで即答させる
python3 chat.py -m Qwen/Qwen3.5-9B --think         # Qwen3.5 の小型モデルは既定で思考しない

# Sarashina
python3 chat.py -m sbintuitions/sarashina2.2-3b-instruct-v0.1

# NVIDIA
python3 chat.py -m nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese --no-think
python3 chat.py -m nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

# GGUF 量子化モデル（旧 sample_interactive_gguf.py）
python3 chat.py -m llm-jp/llm-jp-4.1-8b-thinking -b llamacpp -q Q4_K_M
python3 chat.py -m llm-jp-4.1-32b-a3b-thinking-gguf -ngl 20     # 一部の層だけ GPU に載せる
```

登録済みのモデルは `python3 chat.py --list-models`、アダプタは `python3 chat.py --list-adapters` で確認できます。

## 5. 一問一答・バッチ処理（`generate.py`）

```bash
# 1問だけ（旧 gpt-oss-tutorial の sample.py に相当）
python3 generate.py -m openai/gpt-oss-20b "大規模言語モデルについて簡単に教えてください。"

# prompts.txt の各行をプロンプトとして処理し、結果を JSON Lines で保存
# （vLLM バックエンドではまとめて推論するので速い）
python3 generate.py -m Qwen/Qwen3-8B -b vllm --no-think -i prompts.txt -o outputs.jsonl
```

出力ファイルの各行は `{"model", "prompt", "response", "reasoning", "prompt_tokens", "output_tokens"}` です。

## 6. Python プログラムから使う（`example_api.py`）

```python
from slcllm import LocalLLM

llm = LocalLLM("llm-jp/llm-jp-4.1-8b-thinking", reasoning_effort="low")

res = llm.chat("大規模言語モデルについて簡単に教えてください。")
print(res.reasoning)   # 思考過程（思考しないモデルでは空文字列）
print(res.content)     # 最終回答

conv = llm.conversation()                  # 履歴を保持した対話
conv.ask("奈良県の県庁所在地はどこですか？")
print(conv.ask("そこの有名な観光地は？").content)

llm.chat("好きな果物は？", temperature=1.0, max_new_tokens=64)   # パラメータをその場で変更
llm.chat_batch(["1+1は？", "2+2は？"])                            # まとめて処理
```

モデル名を変えるだけで、コードは一切変えずに別のモデルに切り替えられます。

---

## 7. 対応モデルとアダプタ

| 提供元 | 主なモデル | アダプタ | 思考モード | 備考 |
|---|---|---|---|---|
| LLM-jp | llm-jp-3-*-instruct*, llm-jp-3.1-*-instruct4 | `llm-jp-3` | なし | chat template の無いベースモデルは旧チュートリアルの「### 指示 / ### 応答」形式 |
| LLM-jp | llm-jp-4 / 4.1-*-thinking（GGUF 版も） | `llm-jp-4` | 常にオン（`reasoning_effort` で深さ指定） | harmony 形式 |
| OpenAI | gpt-oss-20b / 120b | `gpt-oss` | 常にオン（`reasoning_effort`） | harmony 形式 |
| Swallow | Llama-3 / 3.1 / 3.3-Swallow-*-Instruct | `swallow-llama` | なし | |
| Swallow | Gemma-2-Llama-Swallow-*-it | `swallow-gemma` | なし | system ロール非対応（自動で対処） |
| Swallow | Qwen3-Swallow-*-RL / SFT | `swallow-qwen3` | 常にオン（切替不可） | |
| Swallow | GPT-OSS-Swallow-*-RL / SFT | `swallow-gpt-oss` | 常にオン（medium 推奨） | |
| Qwen | Qwen3（ハイブリッド） | `qwen` | 既定オン. `--no-think` で無効 | |
| Qwen | Qwen3-*-Instruct-2507 / Thinking-2507 | `qwen` | オフ専用 / オン専用 | |
| Qwen | Qwen3.5, Qwen3.8 | `qwen` | `--think` / `--no-think`（3.5 の 9B 以下は既定オフ） | 3.8 は `reasoning_effort`（low/medium/xhigh）にも対応 |
| Qwen | Qwen2.5-*-Instruct | `qwen2` | なし | |
| SB Intuitions | sarashina2.2-*-instruct-v0.1 | `sarashina` | なし | ベースモデル（sarashina2 等）は `--no-chat-template` 推奨 |
| NVIDIA | NVIDIA-Nemotron-Nano-9B-v2(-Japanese), 12B-v2 | `nemotron` | 既定オン. `--no-think` で無効 | Mamba ハイブリッド |
| NVIDIA | NVIDIA-Nemotron-3-Nano-4B / 30B-A3B | `nemotron` | 既定オン. `--no-think` で無効 | Mamba ハイブリッド MoE |
| NVIDIA | Llama-3.1-Nemotron-Nano-8B-v1, 4B-v1.1 | `nemotron` | `detailed thinking on/off` を自動設定 | |
| その他 | 上記以外 | `generic` | chat template から自動判定 | |

**汎用アダプタ（generic）**: どのアダプタにも当てはまらないモデルでは、chat template の中身を見て
`enable_thinking` / `reasoning_effort` / `/no_think` / harmony 形式 / `<think>` タグ / system ロール非対応 などを自動判定します。
生成パラメータはモデル同梱の `generation_config.json` の値を使います。
そのため Qwen や gpt-oss を元にした派生モデル（例: LLM-jp-4 ベースの ELYZA-Thinking など）は、多くの場合そのまま動きます。

### 生成パラメータの決まり方

後ろほど優先されます。どの値が使われたかは `-d` で表示されます。

1. ライブラリの既定値（temperature=0.7, top_p=0.9, max_new_tokens=1024 / 思考時は 4096 など）
2. モデル同梱の `generation_config.json` の値
3. アダプタの推奨値（モデルカードの記載。思考オン/オフで切り替わるものもある）
4. コマンドラインの指定（`--temperature` など）や `llm.chat(..., temperature=...)` の引数

## 8. 主なオプション

| オプション | 内容 |
|---|---|
| `-m`, `--model` | モデル ID（`Qwen/Qwen3-8B`）、短い名前（`llm-jp-3-1.8b-instruct`）、ローカルのパス |
| `-b`, `--backend` | `hf`（既定）/ `vllm` / `llamacpp` / `dryrun` |
| `--think`, `--no-think` | 思考モードの有効・無効（指定しなければモデルの既定） |
| `--reasoning-effort` | 思考の深さ（`low` / `medium` / `high` など） |
| `--hide-thinking` | 思考過程を表示しない |
| `--system "…"`, `--no-system-prompt` | システムプロンプトの変更・無効化 |
| `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--min-p`, `--repetition-penalty`, `--presence-penalty`, `--greedy`, `--seed` | 生成パラメータ |
| `--no-chat-template` | chat template を使わずに続きを生成（ベースモデル用） |
| `--quantize 4bit` | bitsandbytes で量子化して読み込む（hf） |
| `-tp`, `--max-model-len`, `--gpu-memory-utilization` | vLLM 用 |
| `-q`, `--gguf-file`, `-ngl`, `-c`, `--tokenizer` | llama.cpp 用（量子化の種類、ファイル、GPU に載せる層数、コンテキスト長） |
| `--download auto\|offline\|update` | モデルの取得方法（「モデルの取得状況の確認と事前取得」参照） |
| `--revision` | モデルの版（ブランチ名・タグ・コミット ID）を固定する |
| `--adapter` | アダプタを手動で指定 |
| `-d`, `--debug` | 実際のプロンプト、トークン数、速度、使われた生成パラメータを表示 |

**プロンプトだけ確認したいとき**は `dryrun` バックエンドが便利です（tokenizer だけを読み込み、モデル本体は読み込みません）。

```bash
python3 generate.py -m Qwen/Qwen3-8B -b dryrun -d --no-think "こんにちは"
```

---

## 9. 新しいモデルに対応させるには

まずは何もせずに `-m 組織名/モデル名` で試してください（汎用アダプタで動くことが多いです）。
うまく動かない・推奨パラメータを設定したい、という場合はアダプタを追加します。

1. `slcllm/adapters/` に新しいファイルを作り、`BaseAdapter`（または近いモデルのアダプタ）を継承したクラスを書く
2. `slcllm/adapters/__init__.py` の `ADAPTERS` に追加する（派生モデルほど上に置く）
3. 必要なら `slcllm/catalog.py` の `MODELS` に追加する（`--list-models` と短い名前指定のため）

```python
# slcllm/adapters/mymodel.py
from typing import Optional
from ..types import GenerationParams
from .base import BaseAdapter

class MyModelAdapter(BaseAdapter):
    name = "mymodel"
    description = "MyModel 系"
    patterns = (r"my-org/mymodel",)          # モデル ID に対する正規表現

    default_system_prompt = "あなたは親切なアシスタントです。"
    trust_remote_code = True                  # 必要なら

    def generation_defaults(self, thinking: Optional[bool]) -> GenerationParams:
        if thinking:
            return GenerationParams(temperature=0.6, top_p=0.95)
        return GenerationParams(temperature=0.7, top_p=0.8)
```

上書きできる主なメソッド: `prepare_messages`（メッセージの前処理）、`chat_template_kwargs`（テンプレートへの引数）、
`override_chat_template`（テンプレートの差し替え）、`generation_defaults`（推奨パラメータ）、
`parse_stream` / `parse_output`（出力の解析）、`history_message`（履歴への積み方）。
同じ系列でバージョンによって振る舞いが違う場合は、`qwen.py` や `nvidia.py` のように `__init__` でモデル名から
系統（variant）を判定して切り替える方法と、`swallow.py` のように元モデルのアダプタを継承して違う部分だけ上書きする方法があります。

---

## 10. 旧チュートリアルからの移行

| 旧 | 新 |
|---|---|
| `llm-jp_tutorial/sample_interactive.py -m llm-jp-3-1.8b-instruct` | `chat.py -m llm-jp-3-1.8b-instruct` |
| `llm-jp_tutorial/sample_interactive_gguf.py -m llm-jp-4.1-8b-thinking-gguf -ngl 20` | `chat.py -m llm-jp-4.1-8b-thinking-gguf -ngl 20` |
| `--no-instruct` | `--no-chat-template` |
| `gpt-oss-tutorial/sample.py` | `generate.py -m openai/gpt-oss-20b "大規模言語モデルについて簡単に教えてください。"` |
| `Swallow_tutorial/sample_interactive.py -m Llama-3.1-Swallow-8B-Instruct-v0.1` | `chat.py -m Llama-3.1-Swallow-8B-Instruct-v0.1 -b vllm` |

## 11. トラブルシューティング

* **`Unrecognized configuration class` / `model type ... not recognized`**: transformers が古い可能性があります（`pip3 install -U transformers`）。
* **カスタムコードの実行について**: 標準の transformers で読めないモデルは、自動的に `trust_remote_code=True`（リポジトリ内の Python コードを実行）で読み直します。その際はメッセージが表示されます。止めたい場合は `--trust-remote-code no`。
* **GPU メモリが足りない**: まず `estimate_memory.py` で必要量を確認し、`--quantize 4bit`、GGUF 版（`-b llamacpp -q Q4_K_M -ngl 20`）、より小さいモデル、の順に試してください。
* **思考が長すぎて回答が出ない**: `--max-tokens 8192` のように増やすか、`--no-think` / `--reasoning-effort low` を使ってください。思考の途中で上限に達した場合はその旨が表示されます。
* **Nemotron Nano v2 / Nemotron 3 を vLLM で使う**: 精度のため `mamba_ssm_cache_dtype=float32` を自動で指定しています。
* **Qwen3.5 / 3.8** は画像も扱えるモデルですが、本チュートリアルではテキストのみで使います。

---

## 開発者向け

```bash
pip3 install pytest
python3 -m pytest tests/              # 極小のダミーモデルで一連の処理を確認（GPU 不要, 数秒）
python3 download.py --smoke-test      # smoke test で使う実モデルを先に取得しておく
python3 tools/smoke_test.py           # 実モデルを各系列1つずつ動かして結果を表にまとめる（GPU 必要）
```

* `tests/test_hub.py` は、本物と同じ構造の偽キャッシュを一時ディレクトリに作り、Hub への問い合わせとダウンロードだけを
  偽物に差し替えて、キャッシュ確認・不足分の取得・GGUF の選択などを確認します（ネットワーク不要）。
* `tests/test_memory.py` は、Qwen3-8B・gpt-oss-20b と同じ寸法の config をその場で作り、パラメータ数（公称値との一致）と
  KV キャッシュの計算を確認します。

* `tests/` はランダムな重みの極小モデルと、各モデルの chat template を模したテンプレートを使って、
  アダプタの選択・思考モード切替・出力解析・ストリーミング表示・各バックエンドの受け渡しを確認します。
  transformers 4.51（huggingface_hub 0.36）と transformers 5.24（huggingface_hub 2.4）で確認しています。
* 新しいモデルやライブラリの版で動作を確認したら、`tools/smoke_test.py` の結果（`smoke_test_result.md`）を残しておくと便利です。
