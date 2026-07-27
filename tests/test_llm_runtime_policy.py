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

from nodes import _otr_gguf_backend as gguf
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
    assert (p.gguf_n_ctx, p.gguf_quant) == (4096, "Q8_0")
    assert p.lane_allowlist == lp.ALL_LANES


@pytest.mark.parametrize("kw", [
    {"device": "tpu"},
    {"attn_impl": "flash_attention_3"},
    {"quant_policy": "gptq"},
    {"vram_ceiling_gb": -1.0},
    {"gguf_n_ctx": 128},
    {"gguf_quant": "Q5_K"},
    {"lane_allowlist": ("warp_drive",)},
    {"lane_allowlist": ()},
])
def test_policy_validation_fails_loud(kw):
    with pytest.raises(lp.LLMPolicyError):
        lp.LLMRuntimePolicy(**kw)


def test_cache_key_covers_artifact_fields_only():
    """vram_ceiling_gb (pre-load gate) + lane_allowlist (admission) must
    NOT force a reload; device/attn/quant/gguf fields must."""
    base = lp.LLMRuntimePolicy()
    same = lp.LLMRuntimePolicy(vram_ceiling_gb=0,
                               lane_allowlist=("transformers",))
    assert base.cache_key() == same.cache_key()
    assert base.cache_key() != lp.LLMRuntimePolicy(
        attn_impl="eager").cache_key()
    assert base.cache_key() != lp.LLMRuntimePolicy(
        quant_policy="none").cache_key()
    assert base.cache_key() != lp.LLMRuntimePolicy(
        gguf_n_ctx=2048).cache_key()


def test_lane_for_row_mapping():
    assert lp.lane_for_row(None) == lp.LANE_TRANSFORMERS
    row = types.SimpleNamespace(loader_backend="gguf_native")
    assert lp.lane_for_row(row) == lp.LANE_GGUF
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

    monkeypatch.setattr(cat, "validate_model_id", lambda mid: mid)
    row = types.SimpleNamespace(loader_backend="openrouter_http")
    monkeypatch.setattr(cat, "_by_repo_id", lambda: {"fake-remote": row})

    pol = lp.LLMRuntimePolicy(lane_allowlist=("transformers", "gguf"))
    with pytest.raises(ml.ModelLoaderError, match="not admitted"):
        ml.request_slot("creative", "fake-remote", policy=pol)


def test_request_slot_cache_is_policy_keyed(monkeypatch, clean_llm_cache):
    """Same model + same policy = reuse; changed artifact-shaping field =
    teardown + reload. Silent stale-policy reuse is the bug class the
    campaign kills."""
    from nodes import _otr_model_catalog as cat

    monkeypatch.setattr(cat, "validate_model_id", lambda mid: mid)
    monkeypatch.setattr(cat, "_by_repo_id", lambda: {})
    monkeypatch.setattr(
        cat, "resolve_context_cap",
        lambda mid: types.SimpleNamespace(tier="PASS", value=8192))
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

    def _fake_unload():
        unloads["n"] += 1
        ml.LLM_CACHE.clear()

    monkeypatch.setattr(ml, "load_llm", _fake_load_llm)
    monkeypatch.setattr(ml, "unload_llm", _fake_unload)

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

def _backend_classes():
    from nodes import _otr_comfy_backend as cb
    from nodes import _otr_model_runtime as rt
    from nodes import _otr_openrouter_backend as ob
    from nodes._otr_google_api import llm as gl

    return [
        rt.TransformersSafetensorsBackend,
        rt.TransformersMultimodalTextOnlyBackend,
        rt.TransformersGPTQInt4Backend,
        gguf.GGUFNativeBackend,
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
# GGUF: artifact table + deleted silent tolerances
# --------------------------------------------------------------------------

def test_gguf_artifact_table_known_and_unknown():
    name, size = gguf.gguf_artifact_for_quant("Q8_0")
    assert name == "gemma-4-12b-it-Q8_0.gguf"
    assert size == gguf.EXPECTED_Q8_0_SIZE_BYTES
    for quant in ("Q6_K", "Q4_K_M"):
        fname, fsize = gguf.gguf_artifact_for_quant(quant)
        assert quant in fname and fname.endswith(".gguf")
    with pytest.raises(gguf.GGUFNativeConfigError, match="artifact-table"):
        gguf.gguf_artifact_for_quant("Q5_K")


def test_gguf_load_missing_quant_file_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path))
    monkeypatch.delenv("GEMMA4_12B_GGUF_PATH", raising=False)
    pol = lp.LLMRuntimePolicy(gguf_quant="Q6_K")
    with pytest.raises(gguf.GGUFNativeConfigError,
                       match="gemma-4-12b-it-Q6_K.gguf"):
        gguf.GGUFNativeBackend().load(
            "unsloth/gemma-4-12b-it-GGUF",
            types.SimpleNamespace(context_window=4096), policy=pol)


def test_gguf_load_size_mismatch_fails_loud(monkeypatch, tmp_path):
    bad = tmp_path / "gemma-4-12b-it-Q8_0.gguf"
    bad.write_bytes(b"placeholder-not-a-real-gguf")
    monkeypatch.setenv("GEMMA4_12B_GGUF_PATH", str(bad))
    with pytest.raises(gguf.GGUFNativeConfigError, match="Incomplete"):
        gguf.GGUFNativeBackend().load(
            "unsloth/gemma-4-12b-it-GGUF",
            types.SimpleNamespace(context_window=4096),
            policy=lp.BASELINE_POLICY)


def test_gguf_silent_tolerances_are_gone():
    """The 4096->2048 downgrade (truncated original_concept JSON, see
    d526c8b7) and the 'proceeding anyway' preflight tolerance must not
    exist in the module source anymore."""
    src = inspect.getsource(gguf)
    assert "Dynamically downgrading" not in src
    assert "n_ctx = 2048" not in src, "the silent context downgrade is back"
    assert 'log.warning("[GGUFNative] VRAM preflight check failed' not in src, (
        "the proceed-anyway preflight tolerance is back")


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
    assert pol == lp.BASELINE_POLICY


def test_resolve_inputs_threads_explicit_policy_fields():
    from nodes.OTR_LedgerScriptWriter import _resolve_inputs

    resolved = _resolve_inputs(
        custom_premise="test premise",
        llm_device="cpu", llm_quant_policy="none",
        llm_vram_ceiling_gb=0, gguf_n_ctx=2048, gguf_quant="Q4_K_M",
    )
    pol = resolved["llm_policy"]
    assert (pol.device, pol.quant_policy) == ("cpu", "none")
    assert pol.vram_ceiling_gb == 0
    assert (pol.gguf_n_ctx, pol.gguf_quant) == (2048, "Q4_K_M")


def test_resolve_inputs_rejects_bad_policy_enum():
    from nodes.OTR_LedgerScriptWriter import _resolve_inputs

    with pytest.raises(lp.LLMPolicyError):
        _resolve_inputs(custom_premise="test premise", llm_device="tpu")


# --------------------------------------------------------------------------
# Post-ship audit (2026-07-10): the ledger policy stamp + downstream readers
# --------------------------------------------------------------------------

def test_policy_from_meta_roundtrip_and_failure_modes():
    pol = lp.LLMRuntimePolicy(device="cpu", quant_policy="none",
                              vram_ceiling_gb=0, gguf_quant="Q4_K_M")
    stamp = {"device": pol.device, "attn_impl": pol.attn_impl,
             "quant_policy": pol.quant_policy,
             "vram_ceiling_gb": pol.vram_ceiling_gb,
             "gguf_n_ctx": pol.gguf_n_ctx, "gguf_quant": pol.gguf_quant,
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
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / "nodes"
           / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    assert 'meta["llm_policy"]' in src


def test_downstream_llm_consumers_thread_the_ledger_policy():
    """FreezeCascade + shot-lock request_slot calls carry
    policy=policy_from_meta(...) -- the post-ship audit's MUST-FIX."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "nodes"
    lfc = (root / "OTR_LedgerFreezeCascade.py").read_text(encoding="utf-8")
    assert "policy_from_meta" in lfc
    assert "policy=lfc_policy" in lfc
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
# every REQUEST -- ahead of cache reuse and ahead of loading, on BOTH local
# lanes. Before this it could only gate a fresh transformers load: the GGUF
# dispatch and the transformers cache hit both returned above it, and neither
# reuse key carries the ceiling (correctly -- admission does not shape the
# artifact, so it cannot be closed by widening a key).
# --------------------------------------------------------------------------

_EIGHT_GB_TIER = 6.8  # config/profiles/otr_8gb_ltx.json -> llm.vram_ceiling_gb
_GEMMA_GGUF = "unsloth/gemma-4-12b-it-GGUF"
_GEMMA_HF = "google/gemma-4-12b-it"


class _CountingBackend:
    """Records what actually reached a backend. Nothing is loaded."""

    def __init__(self):
        self.loads = []

    def load(self, model_id, row, policy=None, load_config=None):
        self.loads.append(model_id)
        return {"model_id": model_id, "model": object(), "backend": "stub"}


def _gguf_load_config(repo, n_ctx=4096, quant="Q8_0"):
    return gguf.GGUFLoadConfig(
        repo_id=repo, model_path=f"/models/{repo}/{quant}.gguf", quant=quant,
        n_ctx=n_ctx, n_batch=512, n_gpu_layers=-1, kv_gb_per_1k=None, seed=1,
        stop_tokens=(), think_policy="none",
    )


def test_shipped_registry_admits_the_default_and_refuses_the_8gb_tier():
    """Real registry arithmetic, no fixture and no stub. The ceiling this
    repo ships (14.5) must keep admitting the GGUF writer it ships, and the
    8 GB tier's 6.8 must refuse it. If the first flips, the default workflow
    is broken; if the second flips, the tier ceiling is decorative again."""
    from nodes import _otr_model_catalog as cat

    admitted = cat.check_vram_fit(_GEMMA_GGUF, 4096, ceiling_gb=14.5)
    assert admitted.tier != "FAIL", admitted.reason

    refused = cat.check_vram_fit(_GEMMA_GGUF, 2048, ceiling_gb=_EIGHT_GB_TIER)
    assert refused.tier == "FAIL"
    assert refused.estimated_gb > _EIGHT_GB_TIER
    assert refused.ceiling_gb == _EIGHT_GB_TIER


def test_gguf_cache_hit_cannot_inherit_a_permissive_ceiling(
    monkeypatch, clean_llm_cache,
):
    """THE A1 case. Load under the 16 GB ceiling, then ask for the same
    model under the 8 GB tier's ceiling at the SAME load identity. Because
    GGUFLoadConfig.reuse_key() excludes the ceiling, the second request used
    to be served from the resident singleton and the stricter policy never
    applied to anything."""
    from nodes._otr_model_inputs import VRAMFitFailedError

    backend = _CountingBackend()
    monkeypatch.setattr(
        "nodes._otr_model_runtime.get_backend_for_row", lambda row: backend)

    load_config = _gguf_load_config(_GEMMA_GGUF)
    ml.request_slot("technical", _GEMMA_GGUF, policy=lp.LLMRuntimePolicy(),
                    load_config=load_config)
    assert backend.loads == [_GEMMA_GGUF]
    assert ml.LLM_CACHE.get("model_id") == _GEMMA_GGUF  # genuinely resident

    strict = lp.LLMRuntimePolicy(vram_ceiling_gb=_EIGHT_GB_TIER)
    with pytest.raises(VRAMFitFailedError) as excinfo:
        ml.request_slot("technical", _GEMMA_GGUF, policy=strict,
                        load_config=load_config)
    message = str(excinfo.value)
    assert _GEMMA_GGUF in message
    assert "ceiling" in message
    assert "6.8 GB ceiling" in message
    # Refused, not quietly served from the resident entry.
    assert backend.loads == [_GEMMA_GGUF]


def test_gguf_fresh_load_consults_the_policy_ceiling(
    monkeypatch, clean_llm_cache,
):
    """Not only the reuse path. With nothing resident, the GGUF lane reached
    its backend without the POLICY ceiling ever being consulted: the lane's
    own preflight measures PHYSICAL free VRAM, which a 16 GB card passes for
    an 8 GB tier's model. A physical probe cannot enforce a tier."""
    from nodes._otr_model_inputs import VRAMFitFailedError

    backend = _CountingBackend()
    monkeypatch.setattr(
        "nodes._otr_model_runtime.get_backend_for_row", lambda row: backend)

    with pytest.raises(VRAMFitFailedError, match="ceiling"):
        ml.request_slot(
            "technical", _GEMMA_GGUF,
            policy=lp.LLMRuntimePolicy(vram_ceiling_gb=_EIGHT_GB_TIER),
            load_config=_gguf_load_config(_GEMMA_GGUF),
        )
    assert backend.loads == []  # refused BEFORE the backend was touched


def test_gguf_ceiling_zero_disables_the_gate(monkeypatch, clean_llm_cache):
    """vram_ceiling_gb == 0 is the cpu tier: there is no VRAM to fit, so
    admission must be DISABLED there, not maximally strict."""
    backend = _CountingBackend()
    monkeypatch.setattr(
        "nodes._otr_model_runtime.get_backend_for_row", lambda row: backend)

    ml.request_slot(
        "technical", _GEMMA_GGUF,
        policy=lp.LLMRuntimePolicy(device="cpu", vram_ceiling_gb=0),
        load_config=_gguf_load_config(_GEMMA_GGUF),
    )
    assert backend.loads == [_GEMMA_GGUF]


def test_remote_lane_is_exempt_from_the_ceiling(monkeypatch, clean_llm_cache):
    """A remote row uses ZERO local VRAM, so no tier ceiling may refuse it.
    It is exempt BY PLACEMENT (the remote dispatch returns above the gate);
    pinned so a later hoist cannot sweep the remote lanes in."""
    from nodes import _otr_model_catalog as cat

    backend = _CountingBackend()
    row = types.SimpleNamespace(loader_backend="openrouter_http")
    monkeypatch.setattr(cat, "validate_model_id", lambda mid: mid)
    monkeypatch.setattr(cat, "_by_repo_id", lambda: {"remote/model": row})
    monkeypatch.setattr(
        "nodes._otr_model_runtime.get_backend_for_row", lambda r: backend)

    ml.request_slot("creative", "remote/model",
                    policy=lp.LLMRuntimePolicy(vram_ceiling_gb=0.5))
    assert backend.loads == ["remote/model"]


def test_transformers_cache_hit_cannot_inherit_a_permissive_ceiling(
    monkeypatch, clean_llm_cache,
):
    """The same defect on the other local lane: the transformers cache hit
    also returned above the gate, and LLMRuntimePolicy.cache_key() excludes
    the ceiling, so a stricter request reused a permissively-admitted
    model."""
    from nodes import _otr_model_catalog as cat
    from nodes._otr_model_inputs import VRAMFitFailedError

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

    with pytest.raises(VRAMFitFailedError, match="ceiling"):
        ml.request_slot(
            "creative", _GEMMA_HF,
            policy=lp.LLMRuntimePolicy(vram_ceiling_gb=_EIGHT_GB_TIER),
        )
    assert loads == [_GEMMA_HF]


def test_admission_runs_before_every_cache_read_in_source():
    """Structural pin: the admission call must sit ABOVE both cache reads.
    A future edit that moves a cache read up, or the gate down, silently
    restores the defect -- the behavioural tests above only catch it for the
    two lanes they drive."""
    import inspect

    src = inspect.getsource(ml.request_slot)
    gate = src.index("_assert_policy_admits_vram(")
    assert gate < src.index("in _GGUF_DISPATCH_BACKENDS")
    assert gate < src.index('LLM_CACHE.get("gguf_load_key")')
    assert gate < src.index('LLM_CACHE.get("policy_key")')
