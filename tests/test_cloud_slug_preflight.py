"""Queue-time cloud slug preflight -- pure checker, no socket, no Comfy tree."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nodes._otr_shared import cloud_slug_preflight as csp
from nodes._otr_shared.cloud_media_invoke import _declared_input_names, partner_rows

REPO = Path(__file__).resolve().parents[1]


class _Opt:
    def __init__(self, key):
        self.key = key


class _Inp:
    def __init__(self, id, options):
        self.id = id
        self.options = options


class _Schema:
    def __init__(self, inputs):
        self.inputs = inputs


class _FakePixverse:
    @classmethod
    def define_schema(cls):
        return _Schema([
            _Inp("quality", ["360p", "540p", "720p", "1080p"]),
            _Inp("duration_seconds", [5, 8]),
            _Inp("motion_mode", ["normal", "fast"]),
        ])


class _FakeEleven:
    @classmethod
    def define_schema(cls):
        return _Schema([
            _Inp("model", [_Opt("eleven_multilingual_v2"), _Opt("eleven_v3")]),
            _Inp("apply_text_normalization", ["auto", "on", "off"]),
        ])


class _FakeClassic:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": (["photon-flash-1", "photon-1"],)}}


class _FakeEngine:
    def __init__(self, name, selectors, catalog=None, node_key=""):
        self.name = name
        self.node_key = node_key or name
        self.cloud_catalog = catalog
        self._selectors = selectors

    def cloud_selectors(self):
        return self._selectors


def test_options_from_class_combo_and_dynamiccombo():
    pix = csp.options_from_class(_FakePixverse)
    assert pix["quality"] == ("360p", "540p", "720p", "1080p")
    assert "5" in pix["duration_seconds"] and "8" in pix["duration_seconds"]
    elev = csp.options_from_class(_FakeEleven)
    assert elev["model"] == ("eleven_multilingual_v2", "eleven_v3")
    classic = csp.options_from_class(_FakeClassic)
    assert classic["model"] == ("photon-flash-1", "photon-1")


def test_missing_and_renamed_values_are_findings():
    schema = csp.options_from_class(_FakePixverse)

    def schema_fn(node_key):
        assert node_key == "cloud_pixverse_i2v"
        return schema

    eng = _FakeEngine("cloud_probe_i2v", {
        "cloud_pixverse_i2v": {"quality": ("bogus",), "motion_mode": ("normal",)},
    }, node_key="cloud_pixverse_i2v")
    hits = csp.check_engine(
        eng, eng.cloud_selectors(),
        schema_options_fn=schema_fn,
        catalog_fn=lambda _a: csp.CatalogResult(frozenset(), "", ""))
    assert len(hits) == 1
    assert hits[0].value == "bogus"
    assert hits[0].severity == "refuse"


def test_class_import_failure_is_a_finding():
    eng = _FakeEngine("cloud_probe_i2v", {"cloud_pixverse_i2v": {"quality": ("720p",)}})

    def boom(_key):
        raise RuntimeError("cannot resolve pinned partner class")

    hits = csp.check_engine(
        eng, eng.cloud_selectors(),
        schema_options_fn=boom,
        catalog_fn=lambda _a: csp.CatalogResult(frozenset(), "", ""))
    assert hits and "could not be imported" in hits[0].reason


def test_t2_catalog_outage_warns_on_any_host():
    """An unreachable catalog is a failed probe, and no probe may fail a
    workflow (operator rule, 2026-10-14) -- a warning whether or not the paid
    call shares the host. The note says which case it is."""
    same = csp.catalog_outage_hint(csp.GOOGLE_HOST, csp.GOOGLE_HOST)
    other = csp.catalog_outage_hint(csp.OPENROUTER_HOST, csp.COMFY_PAID_HOST)
    assert "fails loudly" in same
    assert "check is skipped" in other

    def dead(_auth):
        return csp.CatalogResult(None, "timeout", csp.GOOGLE_HOST)

    eng = _FakeEngine("google_omni_video", {
        "google_omni_video": {"model": ("gemini-omni-1.1-flash",)},
    }, catalog="google")
    hits = csp.check_engine(
        eng, eng.cloud_selectors(),
        schema_options_fn=lambda _k: {},
        catalog_fn=dead)
    assert [h.severity for h in hits] == ["warn"]
    assert "fails loudly" in hits[0].fix_hint


def test_t2_google_models_prefix_matches_bare_catalog_id():
    """Validator congruity: posted models/gemini-... must match catalog gemini-..."""
    eng = _FakeEngine("google_omni_video", {
        "google_omni_video": {"model": ("models/gemini-omni-1.1-flash",)},
    }, catalog="google")
    hits = csp.check_engine(
        eng, eng.cloud_selectors(),
        schema_options_fn=lambda _k: {},
        catalog_fn=lambda _a: csp.CatalogResult(
            frozenset({"gemini-omni-1.1-flash"}), "", csp.GOOGLE_HOST))
    assert hits == []


def test_t2_missing_catalog_id_refuses():
    eng = _FakeEngine("google_image", {
        "google_image": {"model": ("not-a-real-gemini",)},
    }, catalog="google")
    hits = csp.check_engine(
        eng, eng.cloud_selectors(),
        schema_options_fn=lambda _k: {},
        catalog_fn=lambda _a: csp.CatalogResult(
            frozenset({"gemini-3.1-flash-image"}), "", csp.GOOGLE_HOST))
    assert hits[0].value == "not-a-real-gemini"
    assert hits[0].severity == "refuse"


def test_all_findings_in_one_raise(monkeypatch):
    prompt = {"63": {"class_type": "OTR_WorkflowValidator", "inputs": {}}}
    findings = [
        csp.Finding("a", "a", "model", "x", "bad x", "fix x", "refuse"),
        csp.Finding("b", "b", "model", "y", "bad y", "fix y", "refuse"),
        csp.Finding("c", "c", "catalog", "", "timeout", "warn only", "warn"),
    ]
    monkeypatch.setattr(csp, "collect_findings", lambda *_a, **_k: findings)
    with pytest.raises(ValueError) as exc:
        csp.ensure_prompt_cloud_slugs(prompt, "u1")
    text = str(exc.value)
    assert "2 issue(s)" in text
    assert "bad x" in text and "bad y" in text


# ---------------------------------------------------------------------------
# Every queue is checked afresh (plan item 1, 2026-10-14). The production call
# passes NO keyword overrides -- the validator's own path -- so these tests mock
# the module's catalog and engine lookups instead of injecting them, and never
# touch private state between calls.
# ---------------------------------------------------------------------------


@pytest.fixture
def production_path(monkeypatch):
    """Module-level catalog/engine stand-ins plus a call counter."""
    state = {"catalog_calls": 0, "ids": {"openrouter": {"google/gemini-3.5-flash"}}}

    def fake_catalog_ids(authority, get_json=None):
        state["catalog_calls"] += 1
        auth = str(authority)
        host = csp.GOOGLE_HOST if auth == "google" else csp.OPENROUTER_HOST
        ids = state["ids"].get(auth)
        if ids is None:
            return csp.CatalogResult(None, "timeout", host)
        return csp.CatalogResult(frozenset(ids), "", host)

    monkeypatch.setattr(csp, "catalog_ids", fake_catalog_ids)
    monkeypatch.setattr(csp, "_default_resolve_engine", lambda _eid: None)
    monkeypatch.setattr(csp, "schema_options", lambda _k: {})
    for name in ("OTR_OPENROUTER_SLOT_A_DEFAULT", "OTR_OPENROUTER_SLOT_B_DEFAULT",
                 "OPENROUTER_MODEL_A", "OPENROUTER_MODEL_B"):
        monkeypatch.delenv(name, raising=False)
    return state


def _or_writer(slug, validator="63", model="openrouter:slot-a"):
    return {
        validator: {"class_type": "OTR_WorkflowValidator", "inputs": {}},
        "70": {"class_type": "OTR_LedgerScriptWriter", "inputs": {
            "gate_in": [validator, 0],
            "creative_writing_model": model,
            "technical_model": model,
            "openrouter_slot_a_model": slug,
        }},
    }


def test_a_refusal_does_not_outlive_the_repaired_pick(production_path):
    with pytest.raises(ValueError):
        csp.ensure_prompt_cloud_slugs(_or_writer("dead/not-a-model"), "63")
    findings = csp.ensure_prompt_cloud_slugs(
        _or_writer("google/gemini-3.5-flash"), "63")
    assert not [f for f in findings if f.severity == "refuse"]
    # ...and back again: a good result does not shelter the next bad pick.
    with pytest.raises(ValueError):
        csp.ensure_prompt_cloud_slugs(_or_writer("dead/not-a-model"), "63")


def test_a_local_first_queue_does_not_wave_through_a_later_cloud_pick(production_path):
    local = _or_writer("", model="Qwen/Qwen3.5-4B")
    assert csp.ensure_prompt_cloud_slugs(local, "63") == []
    assert production_path["catalog_calls"] == 0
    with pytest.raises(ValueError) as exc:
        csp.ensure_prompt_cloud_slugs(_or_writer("dead/not-a-model"), "63")
    assert "dead/not-a-model" in str(exc.value)


def test_every_queue_and_every_validator_is_checked(production_path):
    good = "google/gemini-3.5-flash"
    csp.ensure_prompt_cloud_slugs(_or_writer(good), "63")
    csp.ensure_prompt_cloud_slugs(_or_writer(good), "63")
    csp.ensure_prompt_cloud_slugs(_or_writer(good, validator="99"), "99")
    assert production_path["catalog_calls"] == 3


def test_a_catalog_outage_warns_on_every_queue_and_never_refuses(production_path):
    production_path["ids"] = {}
    for _ in range(2):
        findings = csp.ensure_prompt_cloud_slugs(
            _or_writer("google/gemini-3.5-flash"), "63")
        assert [f.severity for f in findings] == ["warn"]
    assert production_path["catalog_calls"] == 2
    # The outage clears: the same queue is verified, and a dead slug refuses.
    production_path["ids"] = {"openrouter": {"google/gemini-3.5-flash"}}
    with pytest.raises(ValueError):
        csp.ensure_prompt_cloud_slugs(_or_writer("dead/not-a-model"), "63")


def test_replay_skips_writer_and_still_checks_video():
    seen = []

    class _Video:
        name = "cloud_wan_i2v"
        node_key = "cloud_wan_i2v"
        cloud_catalog = None

        def cloud_selectors(self):
            seen.append("video")
            return {"cloud_wan_i2v": {"quality": ("720p",)}}

    def resolve(eid):
        return _Video() if eid == "cloud_wan_i2v" else None

    def schema_fn(_key):
        return {"quality": ("360p", "540p", "720p", "1080p")}

    def catalog_fn(_auth):
        raise AssertionError("writer catalog must not run on replay")

    prompt = {
        "63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
        "70": {
            "class_type": "OTR_LedgerScriptWriter",
            "inputs": {
                "gate_in": ["63", 0],
                "replay_from": "bundle.json",
                "creative_writing_model": "comfy:slot-a",
                "technical_model": "comfy:slot-a",
                "comfy_slot_a_model": "google/gemini-3.5-flash",
            },
        },
        "80": {
            "class_type": "OTR_VideoDirector",
            "inputs": {
                "gate_in": ["63", 0],
                "announcer_video_model": "cloud_wan_i2v",
                "music_video_model": "cloud_wan_i2v",
                "character_video_model": "cloud_wan_i2v",
                "announcer_image_model": "z_image_turbo",
                "music_image_model": "z_image_turbo",
                "character_image_model": "z_image_turbo",
            },
        },
    }
    findings = csp.collect_findings(
        prompt, "63",
        resolve_engine=resolve,
        schema_options_fn=schema_fn,
        catalog_fn=catalog_fn,
    )
    assert seen == ["video"]
    assert not [f for f in findings if f.severity == "refuse" and f.engine.startswith("comfy:")]


# ---------------------------------------------------------------------------
# Writer lanes: the checked slug is the one the backend will POST.
# ---------------------------------------------------------------------------

_LIVE_OR = "google/gemini-3.5-flash"
_LIVE_GOOGLE = "gemini-3.5-flash"
_DEAD = "dead/not-a-model"


def _writer_prompt(handle, slot_widget, slot_value, extra_inputs=None):
    """Queued-prompt shape from test_replay_skips_writer_and_still_checks_video."""
    inputs = {
        "gate_in": ["63", 0],
        "creative_writing_model": handle,
        "technical_model": handle,
        slot_widget: slot_value,
    }
    inputs.update(extra_inputs or {})
    return {
        "63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
        "70": {"class_type": "OTR_LedgerScriptWriter", "inputs": inputs},
    }


def _catalog(ids_by_authority):
    def catalog_fn(authority):
        auth = str(authority)
        host = csp.GOOGLE_HOST if auth == "google" else csp.OPENROUTER_HOST
        ids = ids_by_authority.get(auth)
        if ids is None:
            return csp.CatalogResult(None, "timeout", host)
        return csp.CatalogResult(frozenset(ids), "", host)
    return catalog_fn


_LIVE_CATALOG = _catalog({"openrouter": {_LIVE_OR}, "google": {_LIVE_GOOGLE}})


def _run(prompt, catalog_fn, **kwargs):
    return csp.ensure_prompt_cloud_slugs(
        prompt, "63",
        resolve_engine=lambda _eid: None,
        schema_options_fn=lambda _k: {},
        catalog_fn=catalog_fn,
        **kwargs)


@pytest.fixture
def clean_slot_env(monkeypatch):
    for name in ("OTR_OPENROUTER_SLOT_A_DEFAULT", "OTR_OPENROUTER_SLOT_B_DEFAULT",
                 "OPENROUTER_MODEL_A", "OPENROUTER_MODEL_B",
                 "OTR_COMFY_SLOT_A_DEFAULT", "OTR_COMFY_SLOT_B_DEFAULT"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize("handle,widget,label", [
    ("openrouter:slot-a", "openrouter_slot_a_model", "openrouter"),
    ("google_api:slot-a", "google_api_slot_a_model", "google"),
    ("comfy:slot-a", "comfy_slot_a_model", "openrouter"),
])
def test_dead_writer_slug_refuses_on_every_lane(clean_slot_env, handle, widget, label):
    prompt = _writer_prompt(handle, widget, _DEAD)
    with pytest.raises(ValueError) as exc:
        _run(prompt, _LIVE_CATALOG)
    text = str(exc.value)
    assert handle in text
    assert _DEAD in text
    assert "%s catalog" % label in text


@pytest.mark.parametrize("handle,widget,live", [
    ("openrouter:slot-a", "openrouter_slot_a_model", _LIVE_OR),
    ("google_api:slot-a", "google_api_slot_a_model", _LIVE_GOOGLE),
    ("comfy:slot-a", "comfy_slot_a_model", _LIVE_OR),
])
def test_live_writer_slug_passes_on_every_lane(clean_slot_env, handle, widget, live):
    prompt = _writer_prompt(handle, widget, live)
    findings = _run(prompt, _LIVE_CATALOG)
    assert not [f for f in findings if f.severity == "refuse"]


def test_google_models_prefix_matches_bare_catalog_id(clean_slot_env):
    prompt = _writer_prompt("google_api:slot-a", "google_api_slot_a_model",
                            "models/" + _LIVE_GOOGLE)
    findings = _run(prompt, _LIVE_CATALOG)
    assert not [f for f in findings if f.severity == "refuse"]


def test_openrouter_placeholder_checks_the_env_fallback_slug(clean_slot_env):
    clean_slot_env.setenv("OTR_OPENROUTER_SLOT_A_DEFAULT", "dead/or-env")
    prompt = _writer_prompt("openrouter:slot-a", "openrouter_slot_a_model",
                            "(enable OpenRouter)")
    with pytest.raises(ValueError) as exc:
        _run(prompt, _LIVE_CATALOG)
    text = str(exc.value)
    assert "dead/or-env" in text
    assert "openrouter:slot-a" in text


def test_openrouter_placeholder_passes_when_env_fallback_is_live(clean_slot_env):
    clean_slot_env.setenv("OTR_OPENROUTER_SLOT_A_DEFAULT", _LIVE_OR)
    prompt = _writer_prompt("openrouter:slot-a", "openrouter_slot_a_model",
                            "(enable OpenRouter)")
    findings = _run(prompt, _LIVE_CATALOG)
    assert not [f for f in findings if f.severity == "refuse"]


def test_comfy_placeholder_checks_the_recommended_fallback_slug(clean_slot_env):
    from nodes import _otr_comfy_backend as occ
    fallback = occ.recommended_slug_for_slot("A")
    assert fallback
    prompt = _writer_prompt("comfy:slot-a", "comfy_slot_a_model",
                            "(enable Comfy Credits)")
    findings = _run(prompt, _catalog({"openrouter": {fallback}}))
    assert not [f for f in findings if f.severity == "refuse"]
    with pytest.raises(ValueError) as exc:
        _run(prompt, _catalog({"openrouter": {_LIVE_OR}}))
    assert fallback in str(exc.value)


def test_google_unbound_refuses_without_consulting_catalog(clean_slot_env):
    def no_catalog(_auth):
        raise AssertionError("an unbound Google slot must refuse before any list ping")

    for placeholder in ("", "(enable Google API)"):
        prompt = _writer_prompt("google_api:slot-a", "google_api_slot_a_model",
                                placeholder)
        with pytest.raises(ValueError) as exc:
            _run(prompt, no_catalog)
        text = str(exc.value)
        assert "google_api:slot-a" in text
        assert "No request was sent" in text


def test_openrouter_catalog_down_warns(clean_slot_env):
    """Same host as the paid call: still a warning -- an outage is a failed
    probe, and the paid call itself fails loudly if the host is really down."""
    prompt = _writer_prompt("openrouter:slot-a", "openrouter_slot_a_model", _LIVE_OR)
    findings = _run(prompt, _catalog({}))
    assert [f.severity for f in findings] == ["warn"]
    assert "catalog unavailable" in findings[0].reason
    assert "fails loudly" in findings[0].fix_hint


def test_comfy_writer_openrouter_catalog_down_warns(clean_slot_env):
    prompt = _writer_prompt("comfy:slot-a", "comfy_slot_a_model", _LIVE_OR)
    findings = _run(prompt, _catalog({}))
    warns = [f for f in findings if f.severity == "warn"]
    assert warns and warns[0].engine == "comfy:slot-a"
    assert csp.COMFY_PAID_HOST in warns[0].fix_hint
    assert not [f for f in findings if f.severity == "refuse"]


def test_google_catalog_down_warns(clean_slot_env):
    prompt = _writer_prompt("google_api:slot-a", "google_api_slot_a_model", _LIVE_GOOGLE)
    findings = _run(prompt, _catalog({}))
    assert [f.severity for f in findings] == ["warn"]
    assert "catalog unavailable" in findings[0].reason
    assert csp.GOOGLE_HOST in findings[0].fix_hint


def test_writer_check_restores_process_global_bindings(clean_slot_env):
    from nodes import _otr_openrouter_backend as orb
    from nodes._otr_google_api import models as gai
    orb.set_slot_bindings(slot_a="before/or", slot_b=None)
    gai.set_slot_bindings(slot_a="before-google", slot_b=None)
    try:
        _run(_writer_prompt("openrouter:slot-a", "openrouter_slot_a_model", _LIVE_OR),
             _LIVE_CATALOG)
        _run(_writer_prompt("google_api:slot-a", "google_api_slot_a_model", _LIVE_GOOGLE),
             _LIVE_CATALOG)
        assert orb._slot_bindings == {"A": "before/or", "B": None}
        assert gai._slot_bindings == {"A": "before-google", "B": None}
    finally:
        orb.clear_slot_bindings()
        gai.clear_slot_bindings()


def test_director_google_engine_dead_model_refuses():
    class _Omni:
        name = "google_omni_video"
        node_key = "google_omni_video"
        cloud_catalog = "google"

        def cloud_selectors(self):
            return {"google_omni_video": {"model": ("gemini-omni-flash-preview",)}}

    class _Image:
        name = "google_image"
        node_key = "google_image"
        cloud_catalog = "google"

        def cloud_selectors(self):
            return {"google_image": {"model": ("gemini-dead-image",)}}

    engines = {"google_omni_video": _Omni(), "google_image": _Image()}
    prompt = {
        "63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
        "80": {
            "class_type": "OTR_VideoDirector",
            "inputs": {
                "gate_in": ["63", 0],
                "announcer_video_model": "google_omni_video",
                "music_video_model": "google_omni_video",
                "character_video_model": "google_omni_video",
                "announcer_image_model": "google_image",
                "music_image_model": "google_image",
                "character_image_model": "google_image",
            },
        },
    }
    with pytest.raises(ValueError) as exc:
        csp.ensure_prompt_cloud_slugs(
            prompt, "63",
            resolve_engine=lambda eid: engines.get(eid),
            schema_options_fn=lambda _k: {},
            catalog_fn=_catalog({"google": {"gemini-omni-1.1-flash",
                                            "gemini-3.1-flash-image"}}))
    text = str(exc.value)
    assert "gemini-omni-flash-preview" in text
    assert "google catalog" in text


def test_a_retired_engine_is_named_retired_not_unregistered():
    """A saved graph naming a retired lane meets this gate before the director
    runs. It must hear the tombstone's diagnosis, not the generic "not
    registered" one that reads like a broken install."""
    prompt = {
        "63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
        "80": {
            "class_type": "OTR_VideoDirector",
            "inputs": {
                "gate_in": ["63", 0],
                "announcer_video_model": "google_veo_video",
                "music_video_model": "google_veo_video",
                "character_video_model": "google_veo_video",
                "announcer_image_model": "z_image_turbo",
                "music_image_model": "z_image_turbo",
                "character_image_model": "z_image_turbo",
            },
        },
    }
    with pytest.raises(ValueError) as exc:
        csp.ensure_prompt_cloud_slugs(
            prompt, "63",
            resolve_engine=lambda eid: None,
            schema_options_fn=lambda _k: {},
            catalog_fn=_catalog({}))
    text = str(exc.value)
    assert "video engine 'google_veo_video' is retired and is no longer selectable" in text
    assert "is not registered" not in text


# ---------------------------------------------------------------------------
# Voice engines behind the validator (plan item 2, 2026-10-14). The shipped
# graphs leave CastLock's gate_in unlinked; the cast it locks is consumed by
# the voice nodes behind the validator, so it is in scope without a JSON edit.
# ---------------------------------------------------------------------------


def _shipped_prompt(name):
    import json
    from nodes._otr_workflow_apply import build_offline_schemas, workflow_to_api_prompt
    workflow = json.loads((REPO / "workflows" / name).read_text(encoding="utf-8"))
    prompt = workflow_to_api_prompt(workflow, build_offline_schemas())
    validator = next(nid for nid, n in prompt.items()
                     if n.get("class_type") == "OTR_WorkflowValidator")
    castlock = next(nid for nid, n in prompt.items()
                    if n.get("class_type") == "OTR_CastLock")
    return prompt, validator, castlock


def _looked_up(prompt, validator):
    seen = []

    def resolve(eid):
        seen.append(eid)
        return None

    csp.collect_findings(prompt, validator, resolve_engine=resolve,
                         schema_options_fn=lambda _k: {},
                         catalog_fn=lambda _a: csp.CatalogResult(frozenset(), "", ""),
                         skip_writer=True)
    return seen


@pytest.mark.parametrize("name,voice", [
    ("otr_cloud_low.json", "cloud_elevenlabs"),
    ("otr_cloud_deluxe_3act.json", "cloud_elevenlabs"),
    ("otr_google_still.json", "google_tts"),
])
def test_a_shipped_cloud_voice_reaches_the_slug_check_and_the_estimate(name, voice):
    from nodes._otr_shared import cloud_balance_preflight as cbp
    prompt, validator, castlock = _shipped_prompt(name)
    assert (prompt[castlock]["inputs"].get("gate_in") is None), \
        "the premise: the shipped CastLock is not gated"
    assert prompt[castlock] in csp._collect_scoped(prompt, validator)
    assert voice in _looked_up(prompt, validator)
    lines = cbp.estimate_lines(prompt, validator, resolve_engine=lambda _e: object(),
                               unit_usd_fn=lambda _e, _g: 1.0,
                               writer_usd_fn=lambda _h, _s: (0.0, "x"),
                               skip_writer=True)
    voice_lines = [ln for ln in lines if ln.label == voice]
    assert len(voice_lines) == 1, voice_lines   # char + announcer: one line


def test_a_local_cast_adds_no_voice_check_or_cost():
    from nodes._otr_shared import cloud_balance_preflight as cbp
    prompt, validator, _castlock = _shipped_prompt("otr_canonical.json")
    assert not [e for e in _looked_up(prompt, validator)
                if e in ("cloud_elevenlabs", "google_tts")]
    lines = cbp.estimate_lines(prompt, validator, resolve_engine=lambda _e: object(),
                               unit_usd_fn=lambda _e, _g: 1.0,
                               writer_usd_fn=lambda _h, _s: (0.0, "x"),
                               skip_writer=True)
    assert not [ln for ln in lines if ln.label in ("cloud_elevenlabs", "google_tts")]


def test_a_rolled_cast_checks_and_charges_no_saved_voice():
    from nodes._otr_shared import cloud_balance_preflight as cbp
    prompt, validator, castlock = _shipped_prompt("otr_cloud_low.json")
    prompt[castlock]["inputs"]["roll_audio_engines"] = True
    prompt[castlock]["inputs"]["cast_voice_policy"] = "auto_registry"
    assert "cloud_elevenlabs" not in _looked_up(prompt, validator)
    lines = cbp.estimate_lines(prompt, validator, resolve_engine=lambda _e: object(),
                               unit_usd_fn=lambda _e, _g: 1.0,
                               writer_usd_fn=lambda _h, _s: (0.0, "x"),
                               skip_writer=True)
    assert not [ln for ln in lines if ln.label == "cloud_elevenlabs"]
    # The switch alone, without auto_registry, still casts the saved pick.
    prompt[castlock]["inputs"]["cast_voice_policy"] = "preserve_ledger"
    assert "cloud_elevenlabs" in _looked_up(prompt, validator)


def test_a_wired_voice_slot_warns_instead_of_raising():
    prompt, validator, castlock = _shipped_prompt("otr_cloud_low.json")
    prompt[castlock]["inputs"]["char_voice_engine"] = ["999", 0]
    findings = csp.collect_findings(prompt, validator, resolve_engine=lambda _e: None,
                                    schema_options_fn=lambda _k: {},
                                    catalog_fn=lambda _a: csp.CatalogResult(frozenset(), "", ""),
                                    skip_writer=True)
    wired = [f for f in findings if f.input_name == "char_voice_engine"]
    assert [f.severity for f in wired] == ["warn"]


def test_a_castlock_stays_with_its_own_validator():
    def node(cls, **inputs):
        return {"class_type": cls, "inputs": inputs}

    prompt = {
        "63": node("OTR_WorkflowValidator"),
        "64": node("OTR_WorkflowValidator"),
        # Behind 63: a voice node consuming the ungated CastLock 80.
        "81": node("OTR_BatchKokoroGenerator", gate_in=["63", 0], ledger_json=["80", 0]),
        "80": node("OTR_CastLock", char_voice_engine="cloud_elevenlabs",
                   announcer_voice_engine="kokoro"),
        # Behind 64 only: its own ungated CastLock 90.
        "91": node("OTR_BatchKokoroGenerator", gate_in=["64", 0], ledger_json=["90", 0]),
        "90": node("OTR_CastLock", char_voice_engine="google_tts",
                   announcer_voice_engine="kokoro"),
        # A CastLock gated on 64 that a 63 node consumes stays 64's.
        "82": node("OTR_BatchKokoroGenerator", gate_in=["63", 0], ledger_json=["95", 0]),
        "95": node("OTR_CastLock", gate_in=["64", 0],
                   char_voice_engine="cloud_elevenlabs", announcer_voice_engine="kokoro"),
    }
    scope63 = csp._collect_scoped(prompt, "63")
    scope64 = csp._collect_scoped(prompt, "64")
    assert prompt["80"] in scope63 and prompt["90"] not in scope63
    assert prompt["95"] not in scope63
    assert prompt["90"] in scope64 and prompt["80"] not in scope64
    assert prompt["95"] in scope64
    assert sum(1 for n in scope63 if n is prompt["80"]) == 1


def test_a_rejected_key_refuses_but_an_outage_or_empty_catalog_warns():
    import urllib.error
    from nodes._otr_google_api.client import GoogleAPIKeyMissingError

    def or_401(_url):
        raise urllib.error.HTTPError(_url, 401, "Unauthorized", {}, None)

    assert csp.catalog_ids("openrouter", get_json=or_401).key_rejected
    assert not csp.catalog_ids("openrouter", get_json=lambda _u: {"data": []}).key_rejected
    assert csp.catalog_ids("openrouter", get_json=lambda _u: {"data": []}).ids is None

    def google_no_key(_path):
        raise GoogleAPIKeyMissingError("no key")

    class _Forbidden(RuntimeError):
        http_status = 403
        failure_kind = "auth"
        retryable = False

    def google_403(_path):
        raise _Forbidden("API key not valid")

    def google_down(_path):
        raise OSError("connection refused")

    assert csp.catalog_ids("google", get_json=google_no_key).key_rejected
    assert csp.catalog_ids("google", get_json=google_403).key_rejected
    assert not csp.catalog_ids("google", get_json=google_down).key_rejected
    empty = csp.catalog_ids("google", get_json=lambda _p: {"models": []})
    assert empty.ids is None and not empty.key_rejected

    eng = _FakeEngine("google_omni_video", {
        "google_omni_video": {"model": ("gemini-omni-1.1-flash",)},
    }, catalog="google")

    def severity(catalog):
        hits = csp.check_engine(eng, eng.cloud_selectors(),
                                schema_options_fn=lambda _k: {},
                                catalog_fn=lambda _a: catalog)
        return [h.severity for h in hits]

    assert severity(csp.catalog_ids("google", get_json=google_403)) == ["refuse"]
    assert severity(csp.catalog_ids("google", get_json=google_down)) == ["warn"]
    assert severity(empty) == ["warn"]


def test_the_live_catalog_pings_use_the_short_probe_timeout(monkeypatch):
    from nodes._otr_google_api import client
    seen = {}

    def fake_get_json(path, **kwargs):
        seen.update(kwargs)
        return {"models": [{"name": "models/gemini-3.5-flash"}]}

    monkeypatch.setattr(client, "get_json", fake_get_json)
    csp._live_google_get_json("/v1beta/models?pageSize=1000")
    assert seen.get("timeout_s") == csp.CATALOG_TIMEOUT_S
    src = (REPO / "nodes" / "_otr_shared" / "cloud_slug_preflight.py").read_text(
        encoding="utf-8")
    assert "timeout=CATALOG_TIMEOUT_S" in src


def test_catalog_ids_openrouter_and_google_with_injected_get_json():
    def or_get(url):
        assert url == csp.OPENROUTER_MODELS_URL
        return {"data": [{"id": "google/gemini-3.5-flash"}, {"id": "x/y"}, {"nope": 1}]}

    got = csp.catalog_ids("openrouter", get_json=or_get)
    assert got.ids == frozenset({"google/gemini-3.5-flash", "x/y"})
    assert got.host == csp.OPENROUTER_HOST and got.error == ""

    def google_get(path):
        assert path.startswith("/v1beta/models?pageSize=")
        return {"models": [{"name": "models/gemini-3.5-flash"},
                           {"name": "models/gemini-omni-1.1-flash"}]}

    got = csp.catalog_ids("google", get_json=google_get)
    assert got.ids == frozenset({"gemini-3.5-flash", "gemini-omni-1.1-flash"})
    assert got.host == csp.GOOGLE_HOST and got.error == ""

    def down(_path):
        raise OSError("connection refused")

    assert csp.catalog_ids("openrouter", get_json=down).ids is None
    assert csp.catalog_ids("google", get_json=down).ids is None
    with pytest.raises(ValueError):
        csp.catalog_ids("bogus", get_json=or_get)


def test_validator_calls_slug_check_before_visual_assets():
    src = (REPO / "nodes" / "_otr_workflow_validator.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    helper = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_queue_time_readiness_gates":
            helper = node
            break
    assert helper is not None
    names = []
    for node in ast.walk(helper):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.append(func.id)
        elif isinstance(func, ast.Attribute):
            names.append(func.attr)
    assert "ensure_prompt_cloud_slugs" in names
    assert "ensure_prompt_cloud_balance" in names
    assert "ensure_prompt_visual_assets" in names
    assert names.index("ensure_prompt_cloud_slugs") < names.index(
        "ensure_prompt_cloud_balance")
    assert names.index("ensure_prompt_cloud_balance") < names.index(
        "ensure_prompt_visual_assets")
    # The third argument is the queue's Comfy key -- since plan 0k the one
    # OTR_ComfyCredential bound for this prompt, read from the stash.
    calls = [line.strip() for line in src.splitlines()
             if line.strip() == "_queue_time_readiness_gates(prompt, unique_id, _queue_api_key())"]
    assert len(calls) == 2


def test_every_workflow_json_has_enabled_validator():
    workflows = sorted((REPO / "workflows").rglob("*.json"))
    assert workflows
    missing = []
    for path in workflows:
        import json
        doc = json.loads(path.read_text(encoding="utf-8"))
        hits = [n for n in (doc.get("nodes") or [])
                if n.get("type") == "OTR_WorkflowValidator"]
        if len(hits) != 1:
            missing.append("%s count=%d" % (path.name, len(hits)))
    assert not missing, missing


def _provider_side_evidence(engine_id, eng):
    reasons = []
    if str(engine_id or "").startswith("cloud_"):
        reasons.append("id")
    if getattr(eng, "provider_side", False):
        reasons.append("provider_side")
    if getattr(eng, "native", True) is False:
        reasons.append("native")
    if str(getattr(eng, "node_key", "") or "").startswith("cloud_"):
        reasons.append("node_key")
    if str(engine_id or "").startswith("google_"):
        reasons.append("google")
    if engine_id in {"sonilo", "ideo"}:
        reasons.append("paid-id")
    return reasons


def test_every_paid_engine_implements_cloud_selectors():
    from nodes import _otr_audio_engines  # noqa: F401 -- register
    from nodes import _otr_image_engines  # noqa: F401
    from nodes import _otr_video_engines  # noqa: F401
    from nodes._otr_audio_engines import registry as areg
    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_video_engines import registry as vreg

    missing = []
    for registry in (vreg, ireg, areg):
        for engine_id in sorted(getattr(registry, "_registry", {}) or {}):
            eng = registry.get_engine(engine_id)
            if not _provider_side_evidence(engine_id, eng):
                continue
            if not callable(getattr(eng, "cloud_selectors", None)):
                missing.append(engine_id)
    assert missing == []


def test_selector_keys_are_declared_partner_inputs():
    from nodes import _otr_audio_engines  # noqa: F401
    from nodes import _otr_image_engines  # noqa: F401
    from nodes import _otr_video_engines  # noqa: F401
    from nodes._otr_audio_engines import registry as areg
    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_video_engines import registry as vreg

    rows = partner_rows()
    extras = []
    for registry in (vreg, ireg, areg):
        for engine_id in sorted(getattr(registry, "_registry", {}) or {}):
            eng = registry.get_engine(engine_id)
            if not callable(getattr(eng, "cloud_selectors", None)):
                continue
            if getattr(eng, "cloud_catalog", None):
                continue
            selectors = eng.cloud_selectors()
            for node_key, fields in selectors.items():
                row = rows.get(node_key)
                if not isinstance(row, dict):
                    extras.append("%s unknown row %s" % (engine_id, node_key))
                    continue
                required, optional, hidden = _declared_input_names(row)
                declared = required | optional | hidden
                unknown = sorted(set(fields) - declared)
                if unknown:
                    extras.append("%s %s extras=%s" % (engine_id, node_key, unknown))
    assert extras == []


def test_default_partner_selectors_v3_and_product():
    v3 = csp.default_partner_selectors("cloud_seedance_2")
    assert "model" in v3["cloud_seedance_2"]
    empty = csp.default_partner_selectors("cloud_flux_pro")
    assert empty == {"cloud_flux_pro": {}}
