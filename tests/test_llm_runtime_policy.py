"""S1 LLM runtime policy threading (platform-portability spec section 3).

The frozen LLMRuntimePolicy is built from the writer's inputs in
``_resolve_inputs``, rides ``_SlotScheduler -> request_slot -> every
backend .load(..., policy)``, and replaces the deleted FA2 auto-probe +
tag-based auto-quant with EXPLICIT fields whose defaults equal the nv50
16 GB baseline. request_slot enforces the lane allowlist at runtime
(backstop) and keys the resident-model cache on the artifact-shaping
policy fields.
"""
from __future__ import annotations

import inspect
import types

import pytest

from nodes import _otr_model_loader as ml
from nodes._otr_shared import llm_policy as lp


# --------------------------------------------------------------------------
# The policy object itself
# --------------------------------------------------------------------------

def test_baseline_equals_spec_section4_defaults():
    p = lp.BASELINE_POLICY
    assert (p.device, p.attn_impl, p.quant_policy) == ("cuda", "sdpa",
                                                       "bnb_nf4")
    assert p.vram_ceiling_gb == 14.5
    assert p.lane_allowlist == lp.ALL_LANES


@pytest.mark.parametrize("kw", [
    {"device": "tpu"},
    {"attn_impl": "flash_attention_3"},
    {"quant_policy": "gptq"},
    {"vram_ceiling_gb": -1.0},
    {"lane_allowlist": ("warp_drive",)},
    {"lane_allowlist": ()},
])
def test_policy_validation_fails_loud(kw):
    with pytest.raises(lp.LLMPolicyError):
        lp.LLMRuntimePolicy(**kw)


def test_cache_key_covers_artifact_fields_only():
    """vram_ceiling_gb (pre-load gate) + lane_allowlist (admission) must
    NOT force a reload; device/attn/quant fields must."""
    base = lp.LLMRuntimePolicy()
    same = lp.LLMRuntimePolicy(vram_ceiling_gb=0,
                               lane_allowlist=("transformers",))
    assert base.cache_key() == same.cache_key()
    assert base.cache_key() != lp.LLMRuntimePolicy(
        attn_impl="eager").cache_key()
    assert base.cache_key() != lp.LLMRuntimePolicy(
        quant_policy="none").cache_key()
    assert base.cache_key() != lp.LLMRuntimePolicy(
        device="cpu").cache_key()


def test_lane_for_row_mapping():
    assert lp.lane_for_row(None) == lp.LANE_TRANSFORMERS
    assert lp.lane_for_row(
        types.SimpleNamespace(loader_backend="openrouter_http")
    ) == lp.LANE_OPENROUTER
    with pytest.raises(lp.LLMPolicyError, match="no lane mapping"):
        lp.lane_for_row(types.SimpleNamespace(loader_backend="carrier_pigeon"))


# --------------------------------------------------------------------------
# request_slot: lane backstop + policy-keyed cache
# --------------------------------------------------------------------------

@pytest.fixture
def clean_llm_cache(monkeypatch):
    saved = dict(ml.LLM_CACHE)
    ml.LLM_CACHE.clear()
    try:
        yield ml.LLM_CACHE
    finally:
        ml.LLM_CACHE.clear()
        ml.LLM_CACHE.update(saved)


def test_request_slot_lane_backstop(monkeypatch, clean_llm_cache):
    from nodes import _otr_model_catalog as cat

    monkeypatch.setattr(cat, "validate_model_id", lambda mid, **kwargs: mid)
    row = types.SimpleNamespace(loader_backend="openrouter_http")
    monkeypatch.setattr(cat, "_by_repo_id", lambda: {"fake-remote": row})

    pol = lp.LLMRuntimePolicy(lane_allowlist=("transformers",))
    with pytest.raises(ml.ModelLoaderError, match="not admitted"):
        ml.request_slot("creative", "fake-remote", policy=pol)


def test_request_slot_cache_is_policy_keyed(monkeypatch, clean_llm_cache):
    """Same model + same policy = reuse; changed artifact-shaping field =
    teardown + reload. Silent stale-policy reuse is the bug class the
    campaign kills."""
    from nodes import _otr_model_catalog as cat

    monkeypatch.setattr(cat, "validate_model_id", lambda mid, **kwargs: mid)
    monkeypatch.setattr(cat, "_by_repo_id", lambda: {})
    monkeypatch.setattr(
        cat, "resolve_context_cap",
        lambda mid, **kwargs: cat.ContextCapVerdict("UNKNOWN", 8192, "fixture estimate", explicit_pin=kwargs.get("context_pin")))
    monkeypatch.setattr(
        cat, "check_vram_fit",
        lambda mid, cap, **kw: types.SimpleNamespace(
            tier="PASS", estimated_gb=1.0, ceiling_gb=14.5, reason="stub"))
    monkeypatch.setattr(
        cat, "auto_download_if_missing", lambda mid, **_kwargs: None,
    )

    loads = {"n": 0}

    def _fake_load_llm(mid, **kw):
        loads["n"] += 1
        return {"model": object(), "tokenizer": object(), "model_id": mid,
                "device": "cuda", "quantized": True, "context_cap": 8192}

    unloads = {"n": 0}
    # request_slot's self-triggered transitions route through the
    # ownership-checked _self_unload (PBUG-20260825-04 r3), not the public
    # unconditional unload_llm() -- monkeypatch the real call target, and
    # forward to the real implementation so its epoch bookkeeping (which
    # the subsequent Step-9 store depends on) stays correct.
    _original_self_unload = ml._self_unload

    def _counting_self_unload(my_epoch, *, slot):
        unloads["n"] += 1
        return _original_self_unload(my_epoch, slot=slot)

    monkeypatch.setattr(ml, "load_llm", _fake_load_llm)
    monkeypatch.setattr(ml, "_self_unload", _counting_self_unload)

    base = lp.LLMRuntimePolicy()
    ml.request_slot("creative", "stub/model", policy=base)
    assert loads["n"] == 1
    # Same policy -> cache hit, no reload.
    ml.request_slot("technical", "stub/model", policy=base)
    assert loads["n"] == 1 and unloads["n"] == 0
    # Admission-only change -> still a hit (cache_key excludes it).
    ml.request_slot("creative", "stub/model",
                    policy=lp.LLMRuntimePolicy(vram_ceiling_gb=0))
    assert loads["n"] == 1 and unloads["n"] == 0
    # Artifact-shaping change -> teardown + reload.
    ml.request_slot("creative", "stub/model",
                    policy=lp.LLMRuntimePolicy(attn_impl="eager"))
    assert loads["n"] == 2 and unloads["n"] == 1


# --------------------------------------------------------------------------
# Backends: every .load accepts policy; remote lanes assert admission
# --------------------------------------------------------------------------


def test_native_pin_is_captured_once_and_shapes_reuse(monkeypatch, clean_llm_cache, tmp_path):
    from nodes import _otr_model_catalog as cat
    from nodes import _otr_hf_env as hf
    calls, roots, loaded = [], [], []
    monkeypatch.setattr(hf, "ensure_hf_home", lambda: (calls.append("root") or str(tmp_path)))
    original_validate = cat.validate_model_id

    def validate(model_id, **kwargs):
        assert calls[-1] == "root"
        roots.append(kwargs["hub_root"])
        return original_validate(model_id, **kwargs)

    monkeypatch.setattr(cat, "validate_model_id", validate)
    monkeypatch.setattr(cat, "_read_config_context", lambda mid, **kwargs: 262144)
    monkeypatch.setattr(cat, "auto_download_if_missing", lambda mid, **kwargs: roots.append(kwargs["hub_root"]))
    monkeypatch.setattr(ml, "_require_transformers_model_support", lambda *args: None)

    def load(model_id, **kwargs):
        assert "context_cap" not in kwargs  # estimate must not become a pin
        verdict = kwargs["context_verdict"]
        loaded.append(verdict)
        roots.append(kwargs["hub_root"])
        monkeypatch.setenv("OTR_HARD_VRAM_CONTEXT_LIMIT", "999")  # changes while load is running
        return {"model_id": model_id, "context_cap": verdict.value}

    monkeypatch.setattr(ml, "load_llm", load)
    policy = lp.LLMRuntimePolicy(vram_ceiling_gb=0)
    monkeypatch.setenv("OTR_HARD_VRAM_CONTEXT_LIMIT", " 000128 ")
    first = ml.request_slot("creative", "Qwen/Qwen3.5-4B:nf4", policy=policy)
    assert first["context_cap"] == 128
    assert ml.LLM_CACHE["policy_key"] == (policy.cache_key(), 128)
    monkeypatch.setenv("OTR_HARD_VRAM_CONTEXT_LIMIT", "128")
    assert ml.request_slot("technical", "Qwen/Qwen3.5-4B:nf4", policy=policy) is first
    assert len(loaded) == 1
    monkeypatch.delenv("OTR_HARD_VRAM_CONTEXT_LIMIT", raising=False)
    assert ml.request_slot("creative", "Qwen/Qwen3.5-4B:nf4", policy=policy)["context_cap"] == 262144
    assert loaded[-1].explicit_pin is None
    assert ml.LLM_CACHE["policy_key"] == (policy.cache_key(), None)
    monkeypatch.setenv("OTR_HARD_VRAM_CONTEXT_LIMIT", "invalid")
    ml.request_slot("technical", "Qwen/Qwen3.5-4B:nf4", policy=policy)
    assert len(loaded) == 2
    monkeypatch.setenv("OTR_HARD_VRAM_CONTEXT_LIMIT", "256")
    ml.request_slot("creative", "Qwen/Qwen3.5-4B:nf4", policy=policy)
    assert len(loaded) == 3 and loaded[-1].explicit_pin == 256
    assert all(path == tmp_path / "hub" for path in roots)


@pytest.mark.parametrize(
    "backend", ["openrouter_http", "comfy_credits_http", "google_api_http"])
def test_virtual_routes_do_not_resolve_or_scan_hf(backend, monkeypatch, clean_llm_cache):
    from nodes import _otr_model_catalog as cat
    from nodes import _otr_hf_env as hf
    row = types.SimpleNamespace(loader_backend=backend, context_window=8192)
    monkeypatch.setattr(cat, "_by_repo_id", lambda: {"fixture/virtual": row})

    def forbidden(*args, **kwargs):
        raise AssertionError("virtual route must not touch HF discovery")

    monkeypatch.setattr(hf, "ensure_hf_home", forbidden)
    monkeypatch.setattr(cat, "scan_local_llm_cache", forbidden)
    monkeypatch.setattr(cat, "resolve_context_cap", forbidden)
    remote = types.SimpleNamespace(load=lambda *args, **kwargs: {"provider": backend})
    monkeypatch.setattr("nodes._otr_model_runtime.get_backend_for_row", lambda r: remote)
    entry = ml.request_slot("creative", "fixture/virtual", policy=lp.LLMRuntimePolicy(vram_ceiling_gb=0))
    assert entry["provider"] == backend

def _backend_classes():
    from nodes import _otr_comfy_backend as cb
    from nodes import _otr_model_runtime as rt
    from nodes import _otr_openrouter_backend as ob
    from nodes._otr_google_api import llm as gl

    return [
        rt.TransformersSafetensorsBackend,
        rt.TransformersMultimodalTextOnlyBackend,
        rt.TransformersGPTQInt4Backend,
        ob.OpenRouterBackend,
        cb.ComfyCreditsBackend,
        gl.GoogleAPIBackend,
    ]


@pytest.mark.parametrize("cls", _backend_classes(),
                         ids=lambda c: c.__name__)
def test_every_backend_load_accepts_policy(cls):
    sig = inspect.signature(cls.load)
    assert "policy" in sig.parameters, (
        f"{cls.__name__}.load must accept the S1 policy kwarg")


def test_remote_backends_assert_lane_admission():
    from nodes import _otr_comfy_backend as cb
    from nodes import _otr_openrouter_backend as ob
    from nodes._otr_google_api import llm as gl
    from nodes._otr_google_api.client import GoogleAPIError

    local_only = lp.LLMRuntimePolicy(lane_allowlist=("transformers",))
    row = types.SimpleNamespace(context_window=8192)

    with pytest.raises(ob.OpenRouterConfigError, match="not admitted"):
        ob.OpenRouterBackend().load("openrouter:slot-a", row,
                                    policy=local_only)
    with pytest.raises(cb.ComfyCreditsConfigError, match="not admitted"):
        cb.ComfyCreditsBackend().load("comfy:slot-a", row,
                                      policy=local_only)
    with pytest.raises(GoogleAPIError, match="not admitted"):
        gl.GoogleAPIBackend().load("google:slot-a", row, policy=local_only)


# --------------------------------------------------------------------------
# Loader: deleted auto machinery
# --------------------------------------------------------------------------

def test_loader_auto_probes_are_gone():
    """The FA2 auto-probe and the tag-based auto-quant predicate are
    deleted -- attention + quantization come from the policy only.
    (Code patterns, not names: explanatory comments may cite history.)"""
    src = inspect.getsource(ml)
    assert 'distribution("flash-attn")' not in src, "FA2 probe is back"
    assert "import flash_attn" not in src, "FA2 import probe is back"
    assert "vram_safe_tags = (" not in src, "tag-based auto-quant is back"
    assert "WING DING" not in src


# --------------------------------------------------------------------------
# Writer: _resolve_inputs builds the policy
# --------------------------------------------------------------------------

def test_resolve_inputs_builds_baseline_policy_by_default():
    from nodes.OTR_LedgerScriptWriter import _resolve_inputs

    resolved = _resolve_inputs(custom_premise="test premise")
    pol = resolved["llm_policy"]
    assert isinstance(pol, lp.LLMRuntimePolicy)
    # One Qwen identity: default device is cuda, which bakes NF4. That is
    # also the LLMRuntimePolicy() baseline, so a fresh node matches it.
    assert pol == lp.LLMRuntimePolicy()


def test_resolve_inputs_threads_explicit_policy_fields():
    from nodes.OTR_LedgerScriptWriter import _resolve_inputs

    resolved = _resolve_inputs(
        custom_premise="test premise",
        llm_device="cpu", llm_quant_policy="none",
        llm_vram_ceiling_gb=0,
    )
    pol = resolved["llm_policy"]
    assert (pol.device, pol.quant_policy) == ("cpu", "none")
    assert pol.vram_ceiling_gb == 0


def test_resolve_inputs_rejects_bad_policy_enum():
    from nodes.OTR_LedgerScriptWriter import _resolve_inputs

    with pytest.raises(lp.LLMPolicyError):
        _resolve_inputs(custom_premise="test premise", llm_device="tpu")


# --------------------------------------------------------------------------
# Post-ship audit (2026-07-10): the ledger policy stamp + downstream readers
# --------------------------------------------------------------------------

def test_policy_from_meta_roundtrip_and_failure_modes():
    pol = lp.LLMRuntimePolicy(device="cpu", quant_policy="none",
                              vram_ceiling_gb=0)
    stamp = {"device": pol.device, "attn_impl": pol.attn_impl,
             "quant_policy": pol.quant_policy,
             "vram_ceiling_gb": pol.vram_ceiling_gb,
             "lane_allowlist": list(pol.lane_allowlist)}
    assert lp.policy_from_meta({"llm_policy": stamp}) == pol
    # Absent stamp -> None (pre-stamp ledgers keep the BASELINE backstop).
    assert lp.policy_from_meta({}) is None
    assert lp.policy_from_meta(None) is None
    # Present-but-malformed stamp fails LOUD.
    with pytest.raises(lp.LLMPolicyError):
        lp.policy_from_meta({"llm_policy": "not-a-dict"})
    with pytest.raises(lp.LLMPolicyError):
        lp.policy_from_meta({"llm_policy": {"device": "cuda"}})


def test_writer_stamps_llm_policy_into_meta_source():
    """The writer must stamp meta['llm_policy'] (the freeze cascade +
    shot-lock derivation read it back). Source-level pin."""
    from tests.fixtures.writer_family import family_source

    src = family_source()
    assert 'meta["llm_policy"]' in src


def test_shot_lock_threads_the_ledger_policy():
    """Shot-lock's request_slot call carries policy=policy_from_meta(...)
    -- the post-ship audit's MUST-FIX for the live generating consumer.
    The freeze cascade no longer requests a slot at all; its
    no-acquisition proof lives in tests/test_freeze_policy_readonly.py."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "nodes"
    sl = (root / "otr_shot_lock.py").read_text(encoding="utf-8")
    assert "policy=policy_from_meta(meta)" in sl


def test_stable_audio_music_has_no_device_waterfall():
    """eng_stable_audio was the adapter the S4 sweep missed: it must read
    requested_device (theme-node stamp), never probe cuda."""
    import inspect

    from nodes._otr_audio_engines import eng_stable_audio as sa

    src = inspect.getsource(sa)
    assert 'getattr(self, "requested_device"' in src
    assert 'dev = "cuda" if torch.cuda.is_available() else "cpu"' not in src


# --------------------------------------------------------------------------
# A1 (2026-07-27): the ceiling is an ADMISSION field, so it is evaluated on
# every REQUEST -- ahead of cache reuse and ahead of loading. Before this it
# could only gate a fresh transformers load, because the cache hit returned
# above it, and the reuse key does not carry the ceiling (correctly --
# admission does not shape the artifact, so it cannot be closed by widening
# a key).
# --------------------------------------------------------------------------

_EIGHT_GB_TIER = 6.8  # config/profiles/otr_8gb_ltx.json -> llm.vram_ceiling_gb
_GEMMA_HF = "google/gemma-4-12b-it"


class _CountingBackend:
    """Records what actually reached a backend. Nothing is loaded."""

    def __init__(self):
        self.loads = []

    def load(self, model_id, row, policy=None):
        self.loads.append(model_id)
        return {"model_id": model_id, "model": object(), "backend": "stub"}


def test_remote_lane_is_exempt_from_the_ceiling(monkeypatch, clean_llm_cache):
    """A remote row uses ZERO local VRAM, so no tier ceiling may refuse it.
    It is exempt BY PLACEMENT (the remote dispatch returns above the gate);
    pinned so a later hoist cannot sweep the remote lanes in."""
    from nodes import _otr_model_catalog as cat

    backend = _CountingBackend()
    row = types.SimpleNamespace(loader_backend="openrouter_http")
    monkeypatch.setattr(cat, "validate_model_id", lambda mid, **kwargs: mid)
    monkeypatch.setattr(cat, "_by_repo_id", lambda: {"remote/model": row})
    monkeypatch.setattr(
        "nodes._otr_model_runtime.get_backend_for_row", lambda r: backend)

    ml.request_slot("creative", "remote/model",
                    policy=lp.LLMRuntimePolicy(vram_ceiling_gb=0.5))
    assert backend.loads == ["remote/model"]


def test_transformers_cache_hit_cannot_inherit_a_permissive_ceiling(
    monkeypatch, clean_llm_cache,
):
    """Ceiling is telemetry, not a reuse killer. A model already resident
    under a 16 GB policy may serve an 8 GB-ceiling request of the same
    load identity -- the estimator must not evict a working runtime."""
    from nodes import _otr_model_catalog as cat

    loads = []
    monkeypatch.setattr(cat, "auto_download_if_missing",
                        lambda mid, **kwargs: None)
    monkeypatch.setattr(ml, "_require_transformers_model_support",
                        lambda mid: None)

    def _fake_load_llm(mid, **kwargs):
        loads.append(mid)
        return {"model_id": mid, "model": object(), "context_cap": 4096}

    monkeypatch.setattr(ml, "load_llm", _fake_load_llm)

    ml.request_slot("creative", _GEMMA_HF, policy=lp.LLMRuntimePolicy())
    assert loads == [_GEMMA_HF]

    reused = ml.request_slot(
        "creative", _GEMMA_HF,
        policy=lp.LLMRuntimePolicy(vram_ceiling_gb=_EIGHT_GB_TIER),
    )
    assert reused["model_id"] == _GEMMA_HF
    assert loads == [_GEMMA_HF]


def test_admission_runs_before_every_cache_read_in_source():
    """Structural pin: the admission call must sit ABOVE both cache reads.
    A future edit that moves a cache read up, or the gate down, silently
    restores the defect -- the behavioural tests above only catch it for the
    two lanes they drive."""
    import inspect

    src = inspect.getsource(ml.request_slot)
    gate = src.index("_assert_policy_admits_vram(")
    assert gate < src.index("_try_cache_hit_locked(")
    assert gate < src.index('LLM_CACHE.get("policy_key")')


