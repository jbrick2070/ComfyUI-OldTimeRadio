"""GGUF row registry unit tests (GPU-independent, no live model load).

Covers the 2026-07-16 GGUF-row-registry CODE chunk: the canonical GGUF_ROWS
registry, the immutable per-slot load_config, gemma resolution-identity, the
whitelisted env precedence, catalog projection (per-row on_disk, VRAM no-halve
+ per-row KV), structured-JSON marker, sampling + per-call seed, the
qwen3 think-strip, stop merge, readiness validation, and the loader
resident-reuse identity. Plan of record: docs/2026-07-16-gguf-row-registry.md.
"""
from __future__ import annotations

import pathlib

import pytest

from nodes import _otr_gguf_backend as GGF
from nodes import _otr_model_catalog as cat
from nodes import _otr_structured_call as SC
from nodes._otr_shared.llm_policy import BASELINE_POLICY, LLMRuntimePolicy

GEMMA = GGF.ROW_ID  # unsloth/gemma-4-12b-it-GGUF
QWEN = "unsloth/Qwen3-8B-GGUF"


def _qwen_policy(**over):
    base = dict(gguf_quant="Q4_K_M", gguf_n_ctx=8192)
    base.update(over)
    return LLMRuntimePolicy(**base)


class FakeLlama:
    def __init__(self, reply="the answer"):
        self.calls = []
        self.reply = reply
        self.closed = False

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.reply},
                             "finish_reason": "stop"}]}

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Registry construction + fail-loud validation
# ---------------------------------------------------------------------------


def test_registry_has_two_rows_gemma_and_qwen():
    ids = {r.repo_id for r in GGF.GGUF_ROWS}
    assert GEMMA in ids and QWEN in ids
    assert len(GGF.GGUF_ROWS) == 2
    assert GGF.GGUF_TOP_K == 40


def test_gemma_row_shape():
    row = GGF.gguf_row_for_repo(GEMMA)
    assert row.context_window == 4096
    assert row.kv_gb_per_1k == 0.7
    assert row.vram_fit_tier == "PASS"
    assert row.think_policy == "none"
    assert row.requires_auth is False
    assert "Q8_0" in row.artifacts


def test_qwen_row_is_pinned_and_pass():
    # PROMOTED 2026-07-16 after the live bake-off (3x RESULT SUCCESS + obs asset,
    # ctx=8192 CUDA, peak ~11.8 GB, no fallback): UNKNOWN/unpinned -> PASS/pinned.
    row = GGF.gguf_row_for_repo(QWEN)
    assert row.context_window == 8192
    assert row.kv_gb_per_1k == 0.70            # measured 5.60 GB @ n_ctx=8192
    assert row.vram_fit_tier == "PASS"
    assert row.think_policy == "qwen3_no_think"
    fn, size, sha = row.artifacts["Q4_K_M"]
    assert fn == "Qwen3-8B-Q4_K_M.gguf"
    assert size == 5027784512
    assert sha == (
        "120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4"
    )
    assert row.approx_artifact_gb() > 0.0       # pinned bytes -> a real estimate


def test_gemma_approx_derived_from_pinned_bytes():
    row = GGF.gguf_row_for_repo(GEMMA)
    assert row.approx_artifact_gb() == round(
        GGF.EXPECTED_Q8_0_SIZE_BYTES / (1024 ** 3), 1
    )


def test_registry_error_fails_loud_on_malformed_row():
    # Explicit raise (not assert -- must survive -O). Bad think_policy.
    with pytest.raises(GGF.GGUFRegistryError):
        GGF.GGUFRow(
            repo_id="x/y", subdir="y",
            artifacts={"Q4_K_M": ("y.gguf", None, None)},
            context_window=4096, kv_gb_per_1k=None, vram_fit_tier="PASS",
            license="apache_2_0", license_audit_status="mit_equivalent",
            requires_auth=False, stop_tokens=(), think_policy="BOGUS",
        )


def test_registry_error_on_unsafe_subdir_and_bad_context():
    with pytest.raises(GGF.GGUFRegistryError):
        GGF.GGUFRow(
            repo_id="x/y", subdir="../escape",
            artifacts={"Q4_K_M": ("y.gguf", None, None)},
            context_window=4096, kv_gb_per_1k=None, vram_fit_tier="PASS",
            license="apache_2_0", license_audit_status="mit_equivalent",
            requires_auth=False, stop_tokens=(), think_policy="none",
        )
    with pytest.raises(GGF.GGUFRegistryError):
        GGF.GGUFRow(
            repo_id="x/y", subdir="y",
            artifacts={"Q4_K_M": ("y.gguf", None, None)},
            context_window=0, kv_gb_per_1k=None, vram_fit_tier="PASS",
            license="apache_2_0", license_audit_status="mit_equivalent",
            requires_auth=False, stop_tokens=(), think_policy="none",
        )


# ---------------------------------------------------------------------------
# load_config resolution: gemma identity, n_ctx validate-not-clamp, precedence
# ---------------------------------------------------------------------------


def test_gemma_resolution_identity(monkeypatch):
    for k in ("OTR_GGUF_N_CTX", "OTR_GGUF_N_BATCH", "OTR_GGUF_N_GPU_LAYERS",
              "GEMMA4_12B_N_CTX", "GEMMA4_12B_GGUF_PATH"):
        monkeypatch.delenv(k, raising=False)
    lc = GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY, seed=7)
    # Identity is path / quant / effective-n_ctx (NOT token output).
    assert lc.quant == "Q8_0" == BASELINE_POLICY.gguf_quant
    assert lc.n_ctx == 4096 == BASELINE_POLICY.gguf_n_ctx
    assert lc.model_path == str(GGF.resolve_gguf_path("Q8_0"))
    assert lc.n_batch == GGF.DEFAULT_N_BATCH
    assert lc.n_gpu_layers == -1  # cuda baseline
    assert lc.think_policy == "none"


def test_n_ctx_validate_not_clamp(monkeypatch):
    monkeypatch.delenv("GEMMA4_12B_N_CTX", raising=False)
    # In range -> accepted verbatim.
    monkeypatch.setenv("OTR_GGUF_N_CTX", "2048")
    lc = GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY, seed=1)
    assert lc.n_ctx == 2048
    # Over the gemma row's 4096 window -> RAISE, never clamp to 4096.
    monkeypatch.setenv("OTR_GGUF_N_CTX", "8192")
    with pytest.raises(GGF.GGUFNativeConfigError):
        GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY, seed=1)
    # Below the 512 floor -> RAISE.
    monkeypatch.setenv("OTR_GGUF_N_CTX", "128")
    with pytest.raises(GGF.GGUFNativeConfigError):
        GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY, seed=1)


def test_qwen_accepts_8192_n_ctx(monkeypatch):
    for k in ("OTR_GGUF_N_CTX", "GEMMA4_12B_N_CTX"):
        monkeypatch.delenv(k, raising=False)
    lc = GGF.build_gguf_load_config(repo_id=QWEN, policy=_qwen_policy(), seed=1)
    assert lc.n_ctx == 8192  # equals the qwen row window; validated, not clamped


def test_env_precedence_otr_over_gemma(monkeypatch):
    monkeypatch.setenv("OTR_GGUF_N_CTX", "1536")
    monkeypatch.setenv("GEMMA4_12B_N_CTX", "1024")
    lc = GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY, seed=1)
    assert lc.n_ctx == 1536  # OTR_GGUF_* wins over the legacy gemma var


def test_malformed_override_raises(monkeypatch):
    monkeypatch.setenv("OTR_GGUF_N_CTX", "not-an-int")
    with pytest.raises(GGF.GGUFNativeConfigError):
        GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY, seed=1)


def test_gemma_var_is_gemma_only_qwen_ignores_it(monkeypatch):
    monkeypatch.delenv("OTR_GGUF_N_CTX", raising=False)
    monkeypatch.setenv("GEMMA4_12B_N_CTX", "1024")  # gemma-only lever
    lc = GGF.build_gguf_load_config(repo_id=QWEN, policy=_qwen_policy(), seed=1)
    assert lc.n_ctx == 8192  # qwen ignores GEMMA4_12B_N_CTX -> policy value


# ---------------------------------------------------------------------------
# Path binding: GEMMA4_12B_GGUF_PATH is gemma-only; qwen never leaks gemma
# ---------------------------------------------------------------------------


def test_gemma_path_env_binds_gemma_only(monkeypatch, tmp_path):
    gpath = tmp_path / "my-gemma.gguf"
    monkeypatch.setenv("GEMMA4_12B_GGUF_PATH", str(gpath))
    lc_g = GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY, seed=1)
    assert lc_g.model_path == str(gpath)  # gemma honors the whole-path hatch
    # Qwen must NOT resolve to the gemma override path.
    lc_q = GGF.build_gguf_load_config(repo_id=QWEN, policy=_qwen_policy(), seed=1)
    assert lc_q.model_path != str(gpath)
    assert "Qwen3-8B-Q4_K_M.gguf" in lc_q.model_path
    assert "Qwen3-8B" in lc_q.model_path


def test_qwen_negative_no_gemma_leak(monkeypatch):
    monkeypatch.delenv("GEMMA4_12B_GGUF_PATH", raising=False)
    lc = GGF.build_gguf_load_config(repo_id=QWEN, policy=_qwen_policy(), seed=1)
    low = lc.model_path.lower()
    assert "gemma" not in low
    assert lc.think_policy == "qwen3_no_think"
    assert lc.repo_id == QWEN


# ---------------------------------------------------------------------------
# Quant-in-artifacts fail-loud + mixed slot
# ---------------------------------------------------------------------------


def test_quant_mismatch_raises_with_slot(monkeypatch):
    # BASELINE picks Q8_0, which the qwen row does not carry.
    with pytest.raises(GGF.GGUFNativeConfigError) as ei:
        GGF.build_gguf_load_config(
            repo_id=QWEN, policy=BASELINE_POLICY, slot="technical", seed=1,
        )
    msg = str(ei.value)
    assert "technical" in msg and "Q8_0" in msg and "Q4_K_M" in msg


# ---------------------------------------------------------------------------
# OTR_GGUF_SEED: OPTIONAL uint32 env (2026-07-20 operator: the hard "required"
# guard was removed as unneeded -- unset draws fresh OS entropy; a pinned value is
# byte-reproducible; a malformed pin still fails loud).
# ---------------------------------------------------------------------------


def test_seed_env_optional_defaults_to_entropy(monkeypatch):
    monkeypatch.delenv("OTR_GGUF_SEED", raising=False)
    lc = GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY)
    assert isinstance(lc.seed, int) and 0 <= lc.seed <= 0xFFFFFFFF
    monkeypatch.setenv("OTR_GGUF_SEED", "4294967295")
    lc = GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY)
    assert lc.seed == 4294967295
    monkeypatch.setenv("OTR_GGUF_SEED", "-1")  # out of uint32 range
    with pytest.raises(GGF.GGUFNativeConfigError):
        GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY)
    monkeypatch.setenv("OTR_GGUF_SEED", "abc")  # malformed
    with pytest.raises(GGF.GGUFNativeConfigError):
        GGF.build_gguf_load_config(repo_id=GEMMA, policy=BASELINE_POLICY)


# ---------------------------------------------------------------------------
# Catalog projection: per-row on_disk, VRAM no-halve + per-row KV
# ---------------------------------------------------------------------------


def test_per_row_nonzero_on_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path))
    monkeypatch.delenv("GEMMA4_12B_GGUF_PATH", raising=False)
    assert GGF.gguf_native_row_on_disk(GEMMA) is False  # nothing materialized
    art = tmp_path / "LLM" / "converted" / "gemma-4-12b-it" / "gemma-4-12b-it-Q8_0.gguf"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"")           # zero-byte -> NOT on disk
    assert GGF.gguf_native_row_on_disk(GEMMA) is False
    art.write_bytes(b"gguf-bytes")  # regular non-zero -> present
    assert GGF.gguf_native_row_on_disk(GEMMA) is True


def test_vram_estimate_no_halve_plus_kv():
    # gemma gguf: real on-disk artifact size (no /2) + per-row KV at ctx window.
    est = cat._estimate_resident_gb(GEMMA)
    weights = round(GGF.EXPECTED_Q8_0_SIZE_BYTES / (1024 ** 3), 1)
    kv = (4096 / 1024.0) * 0.7
    assert est == pytest.approx(weights + kv, abs=1e-6)
    # A curated transformers row is still halved (BF16 download heuristic).
    mistral = cat._estimate_resident_gb("mistralai/Mistral-Nemo-Instruct-2407")
    assert mistral == pytest.approx(24.0 / 2.0, abs=1e-6)


def test_vram_estimate_pinned_qwen():
    # PROMOTED 2026-07-16: pinned bytes (no /2) + per-row measured KV @ ctx window.
    est = cat._estimate_resident_gb(QWEN)
    weights = round(5027784512 / (1024 ** 3), 1)
    kv = (8192 / 1024.0) * 0.70
    assert est == pytest.approx(weights + kv, abs=1e-6)


def test_qwen_row_visible_and_ordered_after_gemma(tmp_path):
    ids = [e.repo_id for e in cat.build_dropdown_choices(hub_root=tmp_path)]
    assert QWEN in ids
    # Both gguf rows sit beside the native Gemma rows; gemma gguf first.
    assert ids.index(GEMMA) < ids.index(QWEN)


# ---------------------------------------------------------------------------
# Sampling + per-call seed + pinned top_k reach create_chat_completion
# ---------------------------------------------------------------------------


def test_sampling_and_per_call_seed_reach_llama():
    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake}
    fn = GGF.make_gguf_generate_fn(
        ce,
        sampling={"top_p": 0.91, "min_p": 0.04, "repeat_penalty": 1.07},
        seed=1000,
    )
    fn([{"role": "user", "content": "a"}], temperature=0.3, max_new_tokens=16)
    fn([{"role": "user", "content": "b"}], temperature=0.3, max_new_tokens=16)
    c0, c1 = fake.calls[0], fake.calls[1]
    assert c0["top_p"] == 0.91 and c0["min_p"] == 0.04
    assert c0["repeat_penalty"] == 1.07
    assert c0["top_k"] == GGF.GGUF_TOP_K == 40
    assert c0["seed"] == 1000
    assert c1["seed"] == 1001  # deterministic base + ordinal


def test_seed_base_from_cache_entry_when_not_passed():
    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake, "gguf_seed": 500}
    fn = GGF.make_gguf_generate_fn(ce)
    fn([{"role": "user", "content": "a"}], max_new_tokens=8)
    assert fake.calls[0]["seed"] == 500


def test_no_seed_no_sampling_leaves_them_unset():
    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake}
    fn = GGF.make_gguf_generate_fn(ce)  # no sampling, no seed base
    fn([{"role": "user", "content": "a"}], temperature=0.2, max_new_tokens=8)
    call = fake.calls[0]
    assert "seed" not in call and "top_p" not in call
    assert call["top_k"] == 40  # top_k still pinned (== llama default)


# ---------------------------------------------------------------------------
# think-strip: envelope-only, literal preserved, skipped under response_format
# ---------------------------------------------------------------------------


def test_think_strip_leading_envelope_only():
    fake = FakeLlama(reply="<think>reason here</think>\nANSWER BODY")
    ce = {"provider": "gguf_native", "model": fake,
          "think_policy": "qwen3_no_think"}
    out = GGF.GGUFNativeBackend().generate(
        ce, [{"role": "user", "content": "x"}], max_new_tokens=8,
    )
    assert out == "ANSWER BODY"


def test_think_strip_preserves_later_literal():
    fake = FakeLlama(reply="<think>a</think>BODY then <think>keep me</think>")
    ce = {"provider": "gguf_native", "model": fake,
          "think_policy": "qwen3_no_think"}
    out = GGF.GGUFNativeBackend().generate(
        ce, [{"role": "user", "content": "x"}], max_new_tokens=8,
    )
    assert out == "BODY then <think>keep me</think>"


def test_think_strip_skipped_under_response_format():
    fake = FakeLlama(reply="<think>x</think>Y")
    ce = {"provider": "gguf_native", "model": fake,
          "think_policy": "qwen3_no_think"}
    out = GGF.GGUFNativeBackend().generate(
        ce, [{"role": "user", "content": "x"}], max_new_tokens=8,
        response_format={"type": "json_object"},
    )
    assert out == "<think>x</think>Y"  # rf present -> no strip


def test_think_strip_noop_for_none_policy():
    fake = FakeLlama(reply="<think>x</think>Y")
    ce = {"provider": "gguf_native", "model": fake, "think_policy": "none"}
    out = GGF.GGUFNativeBackend().generate(
        ce, [{"role": "user", "content": "x"}], max_new_tokens=8,
    )
    assert out == "<think>x</think>Y"


# ---------------------------------------------------------------------------
# stop merge: row stop_tokens + caller stop, de-duplicated, row first
# ---------------------------------------------------------------------------


def test_stop_merge_row_and_caller():
    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake,
          "stop_tokens": ("<|end|>", "###")}
    GGF.GGUFNativeBackend().generate(
        ce, [{"role": "user", "content": "x"}], max_new_tokens=8,
        stop=["###", "STOPWORD"],
    )
    assert fake.calls[0]["stop"] == ["<|end|>", "###", "STOPWORD"]


def test_stop_absent_when_empty():
    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake}
    GGF.GGUFNativeBackend().generate(
        ce, [{"role": "user", "content": "x"}], max_new_tokens=8,
    )
    assert "stop" not in fake.calls[0]


# ---------------------------------------------------------------------------
# structured JSON: for_slot("technical") wrapper AND direct factory
# ---------------------------------------------------------------------------


def test_json_object_forced_via_for_slot_wrapper(monkeypatch):
    from nodes.OTR_LedgerScriptWriter import _SlotScheduler
    from nodes import _otr_model_loader as loader

    fake = FakeLlama()
    monkeypatch.setattr(
        loader, "request_slot",
        lambda slot, model_id, policy=None, load_config=None: {
            "provider": "gguf_native", "model": fake,
        },
    )
    sched = _SlotScheduler(
        creative_id="mistralai/Mistral-Nemo-Instruct-2407",
        technical_id=GEMMA, top_p=0.92, min_p=0.0, repetition_penalty=1.0,
    )
    fn = sched.for_slot("technical")
    assert getattr(fn, "_otr_supports_json_object", False) is True
    SC.invoke_structured_slot(
        fn, [{"role": "user", "content": "x"}], temperature=0.2,
        max_new_tokens=16,
    )
    assert fake.calls[-1]["response_format"] == {"type": "json_object"}


def test_json_object_forced_via_direct_factory():
    fake = FakeLlama()
    ce = {"provider": "gguf_native", "model": fake}
    fn = GGF.make_gguf_generate_fn(ce)
    assert getattr(fn, "_otr_supports_json_object", False) is True
    SC.invoke_structured_slot(
        fn, [{"role": "user", "content": "x"}], temperature=0.2,
        max_new_tokens=16,
    )
    assert fake.calls[-1]["response_format"] == {"type": "json_object"}


def test_json_object_not_forced_for_local_transformers_wrapper(monkeypatch):
    from nodes.OTR_LedgerScriptWriter import _SlotScheduler
    from nodes import _otr_model_loader as loader

    captured = {}

    def fake_local_fn(messages, *, temperature, max_new_tokens, stop=None,
                      response_format=None):
        captured["response_format"] = response_format
        return "ok"

    # A local (transformers) slot has no json_object mode -> marker False.
    monkeypatch.setattr(
        loader, "request_slot",
        lambda slot, model_id, policy=None, load_config=None: {
            "model": object(), "tokenizer": object(),
        },
    )
    monkeypatch.setattr(
        "nodes.OTR_LedgerScriptWriter._build_truncating_generate_fn",
        lambda cache_entry, **_kw: fake_local_fn,
    )
    sched = _SlotScheduler(
        creative_id="mistralai/Mistral-Nemo-Instruct-2407",
        technical_id="mistralai/Mistral-Nemo-Instruct-2407",
        top_p=0.92, min_p=0.0, repetition_penalty=1.0,
    )
    fn = sched.for_slot("technical")
    assert getattr(fn, "_otr_supports_json_object", False) is False
    SC.invoke_structured_slot(
        fn, [{"role": "user", "content": "x"}], temperature=0.2,
        max_new_tokens=16,
    )
    assert captured["response_format"] is None  # never forced on the local lane


# ---------------------------------------------------------------------------
# validate_gguf_ready: pinned size/sha, permission wrap, memo invalidation
# ---------------------------------------------------------------------------


def _lc_for(path, quant="Q4_K_M", repo=QWEN):
    return GGF.GGUFLoadConfig(
        repo_id=repo, model_path=str(path), quant=quant, n_ctx=8192,
        n_batch=512, n_gpu_layers=-1, kv_gb_per_1k=None, seed=1,
        stop_tokens=(), think_policy="qwen3_no_think",
    )


def test_validate_ready_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.gguf"
    with pytest.raises(GGF.GGUFNativeConfigError):
        GGF.validate_gguf_ready(QWEN, "Q4_K_M", _lc_for(missing))


def test_validate_ready_existence_only_when_unpinned(tmp_path):
    # An UNPINNED artifact (size/sha None) -> existence + non-zero is enough.
    # gemma Q6_K stays unpinned; qwen Q4_K_M is now PINNED (promoted 2026-07-16),
    # so it validates size+sha and is covered by the pinned tests instead.
    art = tmp_path / "gemma-4-12b-it-Q6_K.gguf"
    art.write_bytes(b"gguf")
    GGF.validate_gguf_ready(
        GEMMA, "Q6_K", _lc_for(art, quant="Q6_K", repo=GEMMA)
    )  # no raise


def test_validate_ready_sha_mismatch_and_memo_invalidation(tmp_path, monkeypatch):
    import hashlib as _h

    art = tmp_path / "gemma-4-12b-it-Q8_0.gguf"
    art.write_bytes(b"AAAA")
    good = _h.sha256(b"AAAA").hexdigest()
    # Register a temporary gemma-like row carrying a pinned sha for Q8_0.
    row = GGF.GGUFRow(
        repo_id="test/sha-row", subdir="sha-row",
        artifacts={"Q8_0": (art.name, 4, good)},
        context_window=4096, kv_gb_per_1k=0.7, vram_fit_tier="PASS",
        license="apache_2_0", license_audit_status="mit_equivalent",
        requires_auth=False, stop_tokens=(), think_policy="none",
    )
    monkeypatch.setitem(GGF.GGUF_ROWS_BY_REPO, "test/sha-row", row)
    GGF._SHA_MEMO.clear()
    lc = _lc_for(art, quant="Q8_0", repo="test/sha-row")
    GGF.validate_gguf_ready("test/sha-row", "Q8_0", lc)  # matches -> ok
    # Replace the file content (new size + mtime) -> memo must re-hash and the
    # now-wrong pinned sha must FAIL loud.
    art.write_bytes(b"BBBBB")
    with pytest.raises(GGF.GGUFNativeConfigError):
        GGF.validate_gguf_ready("test/sha-row", "Q8_0", lc)


# ---------------------------------------------------------------------------
# Loader resident-reuse identity: reload on path/n_ctx change; close order
# ---------------------------------------------------------------------------


class _FakeBackend:
    def __init__(self):
        self.loads = []

    def load(self, repo_id, row, policy=None, load_config=None):
        self.loads.append((repo_id, getattr(load_config, "n_ctx", None)))
        return {"provider": "gguf_native", "model_id": repo_id,
                "model": FakeLlama()}


@pytest.fixture
def clean_cache():
    from nodes import _otr_model_loader as loader
    saved = dict(loader.LLM_CACHE)
    loader.LLM_CACHE.update(
        {"model_id": None, "slot": None, "cache_entry": None,
         "policy_key": None, "gguf_load_key": None}
    )
    yield loader
    loader.LLM_CACHE.clear()
    loader.LLM_CACHE.update(saved)


def _lc(repo, n_ctx, quant="Q8_0"):
    return GGF.GGUFLoadConfig(
        repo_id=repo, model_path=f"/models/{repo}/{quant}.gguf", quant=quant,
        n_ctx=n_ctx, n_batch=512, n_gpu_layers=-1, kv_gb_per_1k=None, seed=1,
        stop_tokens=(), think_policy="none",
    )


def test_reuse_hit_and_reload_on_n_ctx_change(monkeypatch, clean_cache):
    loader = clean_cache
    backend = _FakeBackend()
    unloads = {"n": 0}
    monkeypatch.setattr(
        "nodes._otr_model_runtime.get_backend_for_row", lambda row: backend,
    )
    monkeypatch.setattr(loader, "unload_llm",
                        lambda: unloads.__setitem__("n", unloads["n"] + 1))
    lc1 = _lc(GEMMA, 4096)
    loader.request_slot("technical", GEMMA, policy=BASELINE_POLICY, load_config=lc1)
    # Same load identity -> cache HIT, no second load.
    loader.request_slot("technical", GEMMA, policy=BASELINE_POLICY, load_config=lc1)
    assert len(backend.loads) == 1
    # Different n_ctx -> reuse_key changes -> teardown + reload.
    lc2 = _lc(GEMMA, 2048)
    loader.request_slot("technical", GEMMA, policy=BASELINE_POLICY, load_config=lc2)
    assert len(backend.loads) == 2
    assert unloads["n"] >= 1


def test_gemma_qwen_gemma_close_order(monkeypatch, clean_cache):
    loader = clean_cache
    backend = _FakeBackend()
    order = []
    monkeypatch.setattr(
        "nodes._otr_model_runtime.get_backend_for_row", lambda row: backend,
    )
    monkeypatch.setattr(loader, "unload_llm", lambda: order.append("unload"))
    loader.request_slot("technical", GEMMA, policy=BASELINE_POLICY,
                        load_config=_lc(GEMMA, 4096))
    loader.request_slot("technical", QWEN, policy=_qwen_policy(),
                        load_config=_lc(QWEN, 8192, quant="Q4_K_M"))
    loader.request_slot("technical", GEMMA, policy=BASELINE_POLICY,
                        load_config=_lc(GEMMA, 4096))
    # Each model id switch tears the prior singleton down before the new load.
    assert [r for r, _ in backend.loads] == [GEMMA, QWEN, GEMMA]
    assert order.count("unload") >= 2


def test_source_is_ascii_no_em_dash():
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    src.encode("ascii")
    assert chr(0x2014) not in src  # no em-dash (repo ASCII convention)
