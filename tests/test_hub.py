"""キャッシュ確認・取得（slcllm/hub.py, download.py）のテスト.

本物と同じ構造（blobs/ refs/ snapshots/）の偽キャッシュを一時ディレクトリに作り、
Hub への問い合わせとダウンロードだけを偽物に差し替えて確認する（ネットワーク不要）.
"""

import hashlib
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

huggingface_hub = pytest.importorskip("huggingface_hub")

from slcllm import hub  # noqa: E402

# ---------------------------------------------------------------------------
# 偽の Hub
# ---------------------------------------------------------------------------
INDEX = {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
REPOS = {
    "test-org/sharded": {
        "config.json": b'{"model_type": "llama"}',
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "generation_config.json": b"{}",
        "model.safetensors.index.json": json.dumps(INDEX).encode(),
        "model-00001-of-00002.safetensors": b"W1" * 1000,
        "model-00002-of-00002.safetensors": b"W2" * 1000,
        "original/model.safetensors": b"ORIG" * 5000,  # gpt-oss のような重複した重み → 取得しない
        "pytorch_model.bin": b"BIN" * 3000,  # safetensors があるので取得しない
        "README.md": b"# readme",
    },
    "test-org/custom-code": {
        "config.json": json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_x.XForCausalLM",
                                                 "AutoConfig": "other-org/repo--configuration_y.YConfig"}}).encode(),
        "modeling_x.py": b"# custom",
        "tokenizer.json": b"{}",
        "model.safetensors": b"W" * 100,
    },
    "test-org/tiny-GGUF": {
        "tiny-Q4_K_M.gguf": b"G4" * 100,
        "tiny-Q8_0.gguf": b"G8" * 100,
        "mmproj-tiny-f16.gguf": b"MM" * 100,
        "README.md": b"x",
    },
    "test-org/big-GGUF": {
        "big-Q8_0-00001-of-00002.gguf": b"P1" * 100,
        "big-Q8_0-00002-of-00002.gguf": b"P2" * 100,
        "big-Q4_K_M.gguf": b"Q4" * 50,
    },
}
SHA = "0123456789abcdef0123456789abcdef01234567"


def _blob(name, data):
    return hashlib.sha256(data).hexdigest() if name.endswith((".safetensors", ".gguf", ".bin")) else \
        hashlib.sha1(b"blob " + data).hexdigest()


def materialize(cache, repo_id, files):
    folder = os.path.join(cache, "models--" + repo_id.replace("/", "--"))
    os.makedirs(os.path.join(folder, "blobs"), exist_ok=True)
    os.makedirs(os.path.join(folder, "refs"), exist_ok=True)
    with open(os.path.join(folder, "refs", "main"), "w") as f:
        f.write(SHA)
    snap = os.path.join(folder, "snapshots", SHA)
    for name, data in files.items():
        blob = os.path.join(folder, "blobs", _blob(name, data))
        if not os.path.exists(blob):
            with open(blob, "wb") as f:
                f.write(data)
        link = os.path.join(snap, name)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(blob, os.path.dirname(link)), link)
    return snap


class FakeHub:
    def __init__(self, cache, repos=REPOS):
        self.cache, self.repos = cache, repos
        self.info_calls, self.downloaded, self.offline = 0, [], False

    def model_info(self, api_self, repo_id, revision=None, files_metadata=False, **kw):
        self.info_calls += 1
        if self.offline:
            raise ConnectionError("cannot connect")
        if repo_id not in self.repos:
            raise type("RepositoryNotFoundError", (Exception,), {})("404")
        sib = []
        for name, data in self.repos[repo_id].items():
            lfs = types.SimpleNamespace(sha256=_blob(name, data), size=len(data)) \
                if name.endswith((".safetensors", ".gguf", ".bin")) else None
            sib.append(types.SimpleNamespace(rfilename=name, size=len(data), lfs=lfs,
                                             blob_id=None if lfs else _blob(name, data)))
        return types.SimpleNamespace(sha=SHA, siblings=sib)

    def snapshot_download(self, repo_id, revision=None, local_files_only=False, allow_patterns=None,
                          ignore_patterns=None, **kw):
        if local_files_only:
            return self.real_snapshot_download(repo_id, revision=revision, local_files_only=True, **kw)
        from huggingface_hub.utils import filter_repo_objects  # 本物の huggingface_hub と同じ規則で選別

        files = self.repos[repo_id]
        names = list(filter_repo_objects(list(files), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns))
        self.downloaded += names
        return materialize(self.cache, repo_id, {n: files[n] for n in names})

    def hf_hub_download(self, repo_id, filename, **kw):
        self.downloaded.append(filename)
        return os.path.join(materialize(self.cache, repo_id, {filename: self.repos[repo_id][filename]}), filename)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    from huggingface_hub import constants

    cache = str(tmp_path / "hub")
    os.makedirs(cache)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", cache)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    f = FakeHub(cache)
    f.real_snapshot_download = huggingface_hub.snapshot_download
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda s, *a, **k: f.model_info(s, *a, **k))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", f.snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", f.hf_hub_download)
    return f


# ---------------------------------------------------------------------------
def test_pattern_rules_match_huggingface_hub():
    from huggingface_hub.utils import filter_repo_objects

    files = list(REPOS["test-org/sharded"])
    allow, ignore = hub.download_patterns(files)
    assert hub.select_files(files, allow, ignore) == list(filter_repo_objects(files, allow_patterns=allow,
                                                                            ignore_patterns=ignore))
    selected = hub.select_files(files, allow, ignore)
    assert "original/model.safetensors" not in selected and "pytorch_model.bin" not in selected
    assert "model-00002-of-00002.safetensors" in selected
    allow, ignore = hub.download_patterns(files, need_weights=False)
    assert not any(f.endswith(".safetensors") for f in hub.select_files(files, allow, ignore))


def test_offline_mode_without_cache(fake):
    assert not hub.check_local("test-org/sharded").complete
    with pytest.raises(hub.ModelNotCachedError, match="download.py -m test-org/sharded"):
        hub.ensure_model("test-org/sharded", mode="offline")
    assert fake.info_calls == 0  # offline ではネットワークに接続しない


def test_auto_downloads_only_needed_files_then_uses_cache(fake, capsys):
    path = hub.ensure_model("test-org/sharded")
    assert sorted(fake.downloaded) == sorted(["config.json", "tokenizer.json", "tokenizer_config.json",
                                              "generation_config.json", "model.safetensors.index.json",
                                              "model-00001-of-00002.safetensors",
                                              "model-00002-of-00002.safetensors", "README.md"])
    assert os.path.exists(os.path.join(path, "model-00002-of-00002.safetensors"))
    assert "8 ファイル" in capsys.readouterr().err
    # 2回目はキャッシュを使い、Hub には問い合わせない
    calls = fake.info_calls
    assert hub.ensure_model("test-org/sharded") == path
    assert fake.info_calls == calls
    assert hub.ensure_model("test-org/sharded", mode="offline") == path


def test_partial_cache_fetches_only_missing(fake):
    files = dict(REPOS["test-org/sharded"])
    files.pop("model-00002-of-00002.safetensors")
    files.pop("original/model.safetensors")
    files.pop("pytorch_model.bin")
    materialize(fake.cache, "test-org/sharded", files)
    st = hub.check_local("test-org/sharded")
    assert st.snapshot and st.missing == ["model-00002-of-00002.safetensors"]
    commit, all_files = hub.fetch_repo_files("test-org/sharded")
    allow, ignore = hub.download_patterns([n for n, _, _ in all_files])
    plan = hub.make_plan("test-org/sharded", commit, all_files, allow, ignore)
    assert [n for n, _ in plan.to_download] == ["model-00002-of-00002.safetensors"]
    hub.ensure_model("test-org/sharded")
    assert hub.check_local("test-org/sharded").complete


def test_tokenizer_only(fake):
    hub.ensure_model("test-org/sharded", need_weights=False)
    assert not any(n.endswith(".safetensors") for n in fake.downloaded)
    assert hub.check_local("test-org/sharded", need_weights=False).complete
    assert not hub.check_local("test-org/sharded").complete  # 重みはまだ無い


def test_custom_code_files_are_checked(fake):
    files = {k: v for k, v in REPOS["test-org/custom-code"].items() if k != "modeling_x.py"}
    materialize(fake.cache, "test-org/custom-code", files)
    assert hub.check_local("test-org/custom-code").missing == ["modeling_x.py"]  # 他リポジトリ参照は対象外
    hub.ensure_model("test-org/custom-code")
    assert hub.check_local("test-org/custom-code").complete


def test_hub_errors_are_explained(fake):
    with pytest.raises(hub.ModelNotCachedError, match="見つかりません"):
        hub.ensure_model("test-org/no-such-model")
    e = type("GatedRepoError", (Exception,), {})("403")
    assert "利用規約" in hub.explain_hub_error("meta/x", e)


def test_update_mode_falls_back_to_cache_when_offline(fake, capsys):
    path = hub.ensure_model("test-org/sharded")
    fake.offline = True
    assert hub.ensure_model("test-org/sharded", mode="update") == path
    assert "キャッシュ済みのものを使います" in capsys.readouterr().err


def test_disk_space_check(fake, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda p: types.SimpleNamespace(free=10, total=10, used=0))
    with pytest.raises(hub.DownloadError, match="空き容量"):
        hub.ensure_model("test-org/sharded")


def test_hf_home_warning(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert "dot.zshrc.slc" in hub.cache_env_message()
    monkeypatch.setenv("HF_HOME", "/slc/cache")
    assert hub.cache_env_message() is None


def test_gguf(fake):
    p = hub.ensure_gguf(["test-org/tiny-GGUF"], "Q4_K_M")
    assert p.endswith("tiny-Q4_K_M.gguf") and fake.downloaded == ["tiny-Q4_K_M.gguf"]
    fake.downloaded.clear()
    assert hub.ensure_gguf(["test-org/tiny-GGUF"], "Q4_K_M", mode="offline") == p
    with pytest.raises(hub.ModelNotCachedError):
        hub.ensure_gguf(["test-org/tiny-GGUF"], "Q8_0", mode="offline")
    # 分割ファイルは全パートを取得. 存在しないリポジトリ候補は飛ばす
    p = hub.ensure_gguf(["test-org/big-gguf-missing", "test-org/big-GGUF"], "Q8_0")
    assert p.endswith("00001-of-00002.gguf") and len(fake.downloaded) == 2
    assert hub.ensure_gguf(["test-org/big-GGUF"], "Q8_0", mode="offline") == p


def test_list_cached_models(fake):
    hub.ensure_model("test-org/sharded")
    repos = hub.list_cached_models()
    assert [r.repo_id for r in repos] == ["test-org/sharded"] and repos[0].size_on_disk > 0


# ---------------------------------------------------------------------------
# LocalLLM / download.py との結合
# ---------------------------------------------------------------------------
def test_local_llm_loads_from_cache(fake, monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    import tiny_models as T

    root = os.environ.get("SLCLLM_TINY_MODELS") or str(tmp_path / "tiny")
    src = T.build_all(root, with_model=True)["qwen3-like"]
    files = {}
    for name in os.listdir(src):
        with open(os.path.join(src, name), "rb") as f:
            files[name] = f.read()
    fake.repos = dict(REPOS, **{"test-org/qwen3-like": files})

    from slcllm import LocalLLM

    with pytest.raises(hub.ModelNotCachedError):
        LocalLLM("test-org/qwen3-like", download="offline", verbose=False)
    llm = LocalLLM("test-org/qwen3-like", generation={"max_new_tokens": 4}, verbose=False)  # 取得してから読み込む
    assert llm.adapter.name == "qwen"  # アダプタはモデル ID で選ばれる
    assert llm.model_path.startswith(fake.cache)
    calls = fake.info_calls
    llm2 = LocalLLM("test-org/qwen3-like", download="offline", verbose=False)
    assert llm2.chat("テスト").output_tokens >= 0 and fake.info_calls == calls


def test_download_script(fake, monkeypatch, capsys):
    import download

    monkeypatch.setattr(sys, "argv", ["download.py", "-m", "test-org/sharded", "--check"])
    assert download.main() == 1
    out = capsys.readouterr().out
    assert "❌ test-org/sharded" in out and "取得が必要: 8 ファイル" in out

    monkeypatch.setattr(sys, "argv", ["download.py", "-m", "test-org/sharded"])
    assert download.main() == 0
    monkeypatch.setattr(sys, "argv", ["download.py", "-m", "test-org/sharded", "--check"])
    assert download.main() == 0

    monkeypatch.setattr(sys, "argv", ["download.py", "-m", "test-org/tiny-GGUF", "-b", "llamacpp", "-q", "Q8_0"])
    assert download.main() == 0
    monkeypatch.setattr(sys, "argv", ["download.py", "--list"])
    assert download.main() == 0
    assert "test-org/sharded" in capsys.readouterr().out
