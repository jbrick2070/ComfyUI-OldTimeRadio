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


def test_t2_refuse_same_host_warn_other_host():
    assert csp.transport_severity(csp.GOOGLE_HOST, csp.GOOGLE_HOST) == "refuse"
    assert csp.transport_severity(csp.OPENROUTER_HOST, csp.COMFY_PAID_HOST) == "warn"
    assert csp.transport_severity(csp.OPENROUTER_HOST, csp.OPENROUTER_HOST) == "refuse"

    def dead(_auth):
        return csp.CatalogResult(None, "timeout", csp.GOOGLE_HOST)

    eng = _FakeEngine("google_veo_video", {
        "google_veo_video": {"model": ("veo-3.1-lite-generate-preview",)},
    }, catalog="google")
    hits = csp.check_engine(
        eng, eng.cloud_selectors(),
        schema_options_fn=lambda _k: {},
        catalog_fn=dead)
    assert hits[0].severity == "refuse"


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


def test_all_findings_in_one_raise():
    csp._CACHE.clear()
    prompt = {"63": {"class_type": "OTR_WorkflowValidator", "inputs": {}}}
    findings = [
        csp.Finding("a", "a", "model", "x", "bad x", "fix x", "refuse"),
        csp.Finding("b", "b", "model", "y", "bad y", "fix y", "refuse"),
        csp.Finding("c", "c", "catalog", "", "timeout", "warn only", "warn"),
    ]
    csp._CACHE["u1"] = findings
    with pytest.raises(ValueError) as exc:
        csp.ensure_prompt_cloud_slugs(prompt, "u1")
    text = str(exc.value)
    assert "2 issue(s)" in text
    assert "bad x" in text and "bad y" in text
    csp._CACHE.clear()


def test_per_prompt_cache_does_not_recompute():
    csp._CACHE.clear()
    calls = {"n": 0}

    def walk(_prompt, _uid):
        calls["n"] += 1
        return []

    prompt = {"63": {"class_type": "OTR_WorkflowValidator", "inputs": {}}}
    csp.ensure_prompt_cloud_slugs(prompt, "cache-me", walk_fn=walk)
    csp.ensure_prompt_cloud_slugs(prompt, "cache-me")
    assert calls["n"] == 1
    csp.ensure_prompt_cloud_slugs(prompt, "other", walk_fn=walk)
    assert calls["n"] == 2
    csp._CACHE.clear()


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
    assert "ensure_prompt_visual_assets" in names
    assert names.index("ensure_prompt_cloud_slugs") < names.index(
        "ensure_prompt_visual_assets")
    calls = [line.strip() for line in src.splitlines()
             if line.strip() == "_queue_time_readiness_gates(prompt, unique_id)"]
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
