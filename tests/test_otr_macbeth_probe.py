"""Hermetic coverage for the Macbeth II.ii cloud-safety probe harness.

100% offline. The 4-cell live run IS this harness's integration test, so
nothing here spends a cent, and there is deliberately NO ``live`` pytest
marker -- an unused marker invites a future session to register a paid leg
against a contract nobody wrote.

The load-bearing tests are the ones that would turn RED if a future fixture
edit dropped a ledger field. A synthetic ledger silently inverts any gate that
reads ``meta.<field>`` and treats absent as permissive: drop ``source_bank``
and the banana route rewrites Macbeth's daggers to bananas, the providers
cheerfully return four green cells, and the probe reports a safety filter it
never exercised. That failure is invisible at runtime, which is exactly why it
is pinned here.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import wave

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "otr_macbeth_probe.py")


def _load():
    spec = importlib.util.spec_from_file_location("otr_macbeth_probe", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load()


@pytest.fixture(autouse=True)
def _no_sys_path_leak():
    """Restore ``sys.path`` around every test in this file.

    The harness legitimately reorders ``sys.path`` (the ComfyUI core ships
    ``nodes.py`` while OTR ships a ``nodes/`` package, so order decides which
    one wins the name). Inside a shared pytest process that mutation would
    outlive the test and change how later FILES resolve their imports -- it
    briefly did, breaking four import-behaviour tests that ran after this
    module alphabetically. The harness now keeps the core out of the
    OTR-only paths; this fixture makes the isolation unconditional.
    """
    import sys as _sys

    saved = list(_sys.path)
    try:
        yield
    finally:
        _sys.path[:] = saved


# --------------------------------------------------------------------------- #
# Import hygiene
# --------------------------------------------------------------------------- #


def test_module_import_is_side_effect_free():
    """Loading the harness must not bootstrap paths, resolve keys, or import
    the comfy core -- the hermetic tests have to run anywhere. Asserted by
    OBSERVING sys.path/sys.modules across the import, not by inspection."""
    import sys as _sys

    path_before = list(_sys.path)
    comfy_before = {m for m in _sys.modules if m.startswith("comfy_api_nodes")}

    reloaded = _load()

    assert _sys.path == path_before, "import mutated sys.path"
    comfy_after = {m for m in _sys.modules if m.startswith("comfy_api_nodes")}
    assert comfy_after == comfy_before, "import pulled in the comfy core"
    assert reloaded.PROBE_ID == "macbeth_probe"
    assert reloaded.REPORT_SCHEMA_VERSION == "1.0.0"


def test_otr_only_helpers_never_add_the_comfy_core_to_sys_path():
    """The core ships ``nodes.py``; OTR ships a ``nodes/`` package. Dropping
    the core into a shared interpreter changes how every LATER import in that
    process resolves, which leaks across test files. Only the live/pre-flight
    paths may add it."""
    import inspect
    import sys as _sys

    before = list(_sys.path)
    probe.assert_ledger_fields(probe.build_probe_meta())
    probe.build_frozen_inputs()
    probe._run_dir_for("20260808T000000Z")
    added = [p for p in _sys.path if p not in before]
    assert not any("comfy_api_nodes" in p or "ComfyUI-Installs" in p
                   for p in added), added

    for fn in (probe.assert_ledger_fields, probe.build_visual_prompt,
               probe.build_frozen_inputs, probe._run_dir_for):
        src = inspect.getsource(fn)
        assert "bootstrap_sys_path()" not in src, fn.__name__
        assert "ensure_repo_on_path()" in src, fn.__name__


def test_outcome_taxonomy_is_exactly_seven_states():
    assert len(probe.OUTCOMES) == 7
    assert set(probe.OUTCOMES) == {
        "PASS", "SAFETY_REFUSAL", "INFRA_BLOCKED", "PROVIDER_ERROR",
        "INVALID_ARTIFACT", "DEPENDENCY_BLOCKED", "UNKNOWN",
    }
    # ORPHANED is an UNKNOWN sub-state, never a peer outcome.
    assert probe.ORPHANED not in probe.OUTCOMES


def test_no_live_pytest_marker_is_registered():
    """r4 CUT-4: do not register an unused ``live`` marker.

    An unused marker is an invitation for a later session to hang a paid leg
    off a contract nobody wrote. The 4-cell live run is the integration test;
    a marker gets added WITH a real opt-in test or not at all.
    """
    pyproject = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
                 ).read_text(encoding="utf-8")
    assert '"live' not in pyproject
    assert "'live" not in pyproject


# --------------------------------------------------------------------------- #
# Violence inspection -- what decides discharge eligibility
# --------------------------------------------------------------------------- #


def test_inspect_violence_finds_explicit_terms():
    out = probe.inspect_violence("bloody daggers and crimson hands")
    assert out["retains_explicit_violence"] is True
    assert "daggers" in out["violence_terms"]
    assert "bloody" in out["violence_terms"]
    assert out["banana_route_fired"] is False


def test_inspect_violence_rejects_mood_words_as_violence():
    """The closed beat/arc clause vocabularies carry no violence at all.
    A probe that counted 'tension' or 'peak intensity' as violent would
    discharge the gate on an abstraction that no filter would ever refuse."""
    clauses = ("gathering tension, the story's peak intensity, "
               "slow cinematic camera drift, no on-screen text")
    out = probe.inspect_violence(clauses)
    assert out["retains_explicit_violence"] is False
    assert out["violence_terms"] == []


def test_inspect_violence_matches_whole_words_only():
    assert probe.inspect_violence("bloodhound")["retains_explicit_violence"] is False
    assert probe.inspect_violence("blood")["retains_explicit_violence"] is True


def test_inspect_violence_detects_banana_substitution():
    """The banana route is the single transform that most completely inverts
    this probe: it rewrites the INSTRUMENT and leaves the gore words alone,
    so 'bloody daggers' becomes 'bloody bananas' -- still scores as violent
    on the gore term, but the weapon the filter was supposed to judge is
    gone. Detecting the marker is what stops that reading as a real PASS."""
    out = probe.inspect_violence("bloody bananas and crimson hands")
    assert out["banana_route_fired"] is True
    assert out["banana_markers"] == ["bananas"]


# --------------------------------------------------------------------------- #
# Discharge eligibility -- the harness refuses, it does not decide
# --------------------------------------------------------------------------- #


def _frozen(cells):
    return {
        "cells": cells,
        "all_cells_retain_violence": all(
            c["inspection"]["retains_explicit_violence"] for c in cells.values()),
        "banana_route_fired_anywhere": any(
            c["inspection"]["banana_route_fired"] for c in cells.values()),
    }


def _cell(text):
    return {"inspection": probe.inspect_violence(text)}


def test_discharge_eligible_when_all_inputs_retain_violence():
    frozen = _frozen({cid: _cell("bloody daggers") for cid in "ABCD"})
    verdict = probe.discharge_eligibility(frozen)
    assert verdict["eligible"] is True
    assert verdict["reason"] == "explicit_violence_retained"


def test_discharge_refused_when_an_input_is_an_abstraction():
    frozen = _frozen({
        "A1": _cell("bloody daggers"),
        "A2": _cell("gathering tension, rising stakes"),
    })
    verdict = probe.discharge_eligibility(frozen)
    assert verdict["eligible"] is False
    assert verdict["reason"] == "abstraction_control"
    assert verdict["non_retaining_cells"] == ["A2"]
    assert "ESCALATE" in verdict["detail"]


def test_discharge_refused_when_banana_route_fired():
    frozen = _frozen({"A1": _cell("bloody bananas")})
    verdict = probe.discharge_eligibility(frozen)
    assert verdict["eligible"] is False
    assert verdict["reason"] == "banana_route_fired"


# --------------------------------------------------------------------------- #
# The five fail-closed ledger-field assertions
# --------------------------------------------------------------------------- #


def test_probe_meta_stamps_every_field_a_gate_reads():
    meta = probe.build_probe_meta()
    assert meta["source_bank"] == "shakespeare"
    # A DIFFERENT field from source_bank, keyed by _otr_style_catalog.
    assert meta["style_pool_class"] == "adaptation"


def test_ledger_assertions_hold_for_the_probe_meta(monkeypatch):
    monkeypatch.delenv("OTR_BANANA_INCLUDE_FIDELITY_BANKS", raising=False)
    receipt = probe.assert_ledger_fields(probe.build_probe_meta())
    assert receipt["banana_stills_gate"] is False
    assert receipt["banana_video_gate"] is False
    assert receipt["include_fidelity_banks"] is False
    assert receipt["lemmy_excluded"] is True
    assert receipt["banana_receipt"]["banana_route"] == "off"


def test_missing_source_bank_fails_closed(monkeypatch):
    """THE defect this whole pre-flight exists for. Absent field -> the
    banana gate returns True -> daggers become bananas -> four green cells
    that prove nothing. It must raise, not warn."""
    monkeypatch.delenv("OTR_BANANA_INCLUDE_FIDELITY_BANKS", raising=False)
    meta = probe.build_probe_meta()
    del meta["source_bank"]
    with pytest.raises(probe.PreflightError) as exc:
        probe.assert_ledger_fields(meta)
    assert "source_bank" in str(exc.value)
    assert "banana" in str(exc.value).lower()


def test_missing_style_pool_class_fails_closed(monkeypatch):
    """Stamping source_bank does NOT close this one: the style pool is keyed
    on style_pool_class and defaults to 'generic'."""
    monkeypatch.delenv("OTR_BANANA_INCLUDE_FIDELITY_BANKS", raising=False)
    meta = probe.build_probe_meta()
    meta["style_pool_class"] = "generic"
    with pytest.raises(probe.PreflightError) as exc:
        probe.assert_ledger_fields(meta)
    assert "style_pool_class" in str(exc.value)


def test_fidelity_bank_override_fails_closed(monkeypatch):
    """OTR_BANANA_INCLUDE_FIDELITY_BANKS force-enables the banana route even
    on shakespeare, so an operator shell that happens to export it would
    silently invalidate a live run."""
    monkeypatch.setenv("OTR_BANANA_INCLUDE_FIDELITY_BANKS", "1")
    with pytest.raises(probe.PreflightError) as exc:
        probe.assert_ledger_fields(probe.build_probe_meta())
    assert "include_fidelity_banks" in str(exc.value)


def test_env_pin_mismatch_fails_closed(monkeypatch):
    monkeypatch.setenv("OTR_GOOGLE_VEO_RESOLUTION", "1080p")
    with pytest.raises(probe.PreflightError) as exc:
        probe.assert_env_pins()
    assert "OTR_GOOGLE_VEO_RESOLUTION" in str(exc.value)


def test_env_pins_accept_unset_and_report_effective(monkeypatch):
    for name in probe.ENV_PINS:
        monkeypatch.delenv(name, raising=False)
    resolved = probe.assert_env_pins()
    assert resolved["OTR_GOOGLE_TTS_ENABLE_MODEL_RETRY"]["effective"] == "0"
    assert resolved["OTR_GOOGLE_TTS_ENABLE_MODEL_RETRY"]["env_value"] is None


def test_tts_model_retry_pin_is_off():
    """The engine walks a MODEL LADDER when retry is enabled, and every rung
    is another paid POST of the same line."""
    assert probe.ENV_PINS["OTR_GOOGLE_TTS_ENABLE_MODEL_RETRY"] == "0"


# --------------------------------------------------------------------------- #
# Frozen inputs -- built through the REAL production seam
# --------------------------------------------------------------------------- #


def test_frozen_inputs_cover_all_four_cells():
    frozen = probe.build_frozen_inputs()
    assert sorted(frozen["cells"]) == ["A1", "A2", "A3", "A4"]
    assert frozen["cells"]["A4"]["arm"] == "google_tts"
    assert frozen["banana_route_fired_anywhere"] is False


def test_tts_input_has_stage_direction_and_speaker_label_stripped():
    """``[Stabbing him]`` is outside Google TTS's stage-tag allowlist
    ({sighs, whispers, gasps, laughs}) and ``_otr_script_prep._BRACKET``
    removes any bracket tag up to 40 chars. The probe must report the string
    the provider ACTUALLY receives, not the authored line."""
    frozen = probe.build_frozen_inputs()
    spoken = frozen["cells"]["A4"]["provider_bound_input"]
    assert "[Stabbing him]" not in spoken
    assert "Stabbing" not in spoken
    assert not spoken.startswith("MACBETH:")
    assert "I have done the deed" in spoken


def test_visual_prompt_carries_the_brief_core_not_just_clauses():
    """Violence can only survive into a video engine through the <=90-char
    brief core; the beat/arc clause vocabularies are closed and non-violent."""
    frozen = probe.build_frozen_inputs()
    prompt = frozen["cells"]["A1"]["provider_bound_input"]
    assert "daggers" in prompt
    assert probe.inspect_violence(prompt)["retains_explicit_violence"] is True


def test_all_four_frozen_inputs_retain_violence_today():
    """Pins the VERIFY-AT-BUILD finding. If a future composition change
    abstracts the prompt, this turns RED and discharge eligibility must be
    re-decided by the operator rather than silently lost."""
    frozen = probe.build_frozen_inputs()
    for cid, cell in frozen["cells"].items():
        assert cell["inspection"]["retains_explicit_violence"] is True, cid
    assert probe.discharge_eligibility(frozen)["eligible"] is True


# --------------------------------------------------------------------------- #
# Refusal classification -- structured fields only, never exception text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", sorted(probe.REFUSAL_CODES))
def test_structured_refusal_codes_are_recognized(code):
    out = probe.classify_refusal({"finish_reason": code})
    assert out["is_refusal"] is True
    assert out["source"] == "structured_field"


def test_prompt_feedback_block_reason_is_recognized():
    out = probe.classify_refusal({"promptFeedback": {"blockReason": "safety"}})
    assert out["is_refusal"] is True


def test_refusal_never_inferred_from_message_text():
    """An infrastructure error whose message merely contains the word
    'safety' must not be reported as a content refusal -- that is precisely
    the false signal this ratify gate exists to prevent."""
    out = probe.classify_refusal({"error": {"message": "safety system offline"}})
    assert out["is_refusal"] is False


def test_completed_response_without_sample_is_not_a_refusal():
    assert probe.classify_refusal({"done": True})["is_refusal"] is False


@pytest.mark.parametrize("where,payload", [
    ("response", {"response": {"raiMediaFilteredReasons": ["violence"]}}),
    ("response.generateVideoResponse",
     {"response": {"generateVideoResponse": {"raiMediaFilteredReasons": ["x"]}}}),
])
def test_veo_rai_probed_at_both_documented_locations(where, payload):
    """Comfy's Vertex model puts the RAI fields directly on ``response``;
    OTR reads ``response.generateVideoResponse``. The Developer-endpoint
    location is UNCONFIRMED, so both are probed."""
    out = probe.classify_veo_refusal(payload)
    assert out["is_refusal"] is True
    assert out["source"] == where


def test_veo_done_without_sample_stays_unknown():
    """Reclassifying this as SAFETY_REFUSAL requires a sanitized fixture and
    a test first -- guessing here manufactures the exact false signal the
    gate is meant to catch."""
    out = probe.classify_veo_refusal({"done": True, "response": {}})
    assert out["is_refusal"] is False


def test_empty_200_response_carries_its_body_so_a_refusal_is_not_lost():
    """Gemini surfaces a CONTENT BLOCK on this endpoint shape as a 200 OK with
    no media and a ``promptFeedback.blockReason``. Both extractors used to
    raise bare, so a genuine refusal on A1/A4 -- the two arms most likely to
    receive one this way -- arrived as an evidence-free error and was
    classified UNKNOWN. That is the mirror of a false green: a LOST
    SAFETY_REFUSAL."""
    from nodes._otr_audio_engines import eng_google_tts as tts
    from nodes._otr_image_engines import eng_google_image as img

    blocked = {"promptFeedback": {"blockReason": "SAFETY"}}

    with pytest.raises(tts.GoogleTTSError) as a:
        tts._extract_audio_data(blocked)
    assert a.value.response_json == blocked

    with pytest.raises(Exception) as b:
        img._extract_image_data(blocked)
    assert b.value.response_json == blocked

    # ...and the harness now classifies both as a real refusal.
    for exc in (a.value, b.value):
        rep = _Rep()
        attempt = rep.start_attempt("A1")
        assert probe._record_exception(rep, "A1", attempt, exc) == \
            probe.SAFETY_REFUSAL


def test_empty_200_without_a_refusal_code_stays_unknown():
    """http_status is deliberately left None on that path: absent a structured
    code, completed-but-empty is UNKNOWN, never an inferred refusal."""
    from nodes._otr_audio_engines import eng_google_tts as tts

    with pytest.raises(tts.GoogleTTSError) as exc:
        tts._extract_audio_data({"steps": []})
    rep = _Rep()
    attempt = rep.start_attempt("A4")
    assert probe._record_exception(rep, "A4", attempt, exc.value) == probe.UNKNOWN


def test_veo_filtered_count_zero_is_not_a_refusal():
    out = probe.classify_veo_refusal(
        {"response": {"raiMediaFilteredCount": 0}})
    assert out["is_refusal"] is False


# --------------------------------------------------------------------------- #
# Google evidence plumbing -- round trip
# --------------------------------------------------------------------------- #


def _client():
    from nodes._otr_google_api import client
    return client


def test_non_json_error_body_preserves_http_status():
    """The regression this plumbing fixes: ``_safe_json`` raised on a
    non-JSON error body BEFORE the status could be classified, so callers
    could not tell auth from quota from a content refusal."""
    client = _client()
    parsed, raw = client._best_effort_json(b"<html>502 Bad Gateway</html>")
    assert parsed is None
    assert "502" in raw


def test_best_effort_json_parses_real_json():
    client = _client()
    parsed, raw = client._best_effort_json(b'{"error": {"message": "nope"}}')
    assert parsed == {"error": {"message": "nope"}}
    assert raw is None


@pytest.mark.parametrize("status,kind,retryable", [
    (401, "auth", False),
    (403, "auth", False),
    (429, "quota", True),
    (400, "request_shape", False),
    (503, "server", True),
    (None, "transport", True),
])
def test_evidence_classifies_failure_kind(status, kind, retryable):
    client = _client()
    exc = client._attach_evidence(client.GoogleAPIError("x"), status=status)
    assert exc.failure_kind == kind
    assert exc.retryable is retryable
    assert exc.http_status == status


def test_retry_after_is_bounded():
    client = _client()

    class _H:
        def __init__(self, v):
            self._v = v

        def get(self, _k):
            return self._v

    assert client._retry_after_seconds(_H("30")) == 30.0
    assert client._retry_after_seconds(_H("99999")) == 300.0   # capped
    assert client._retry_after_seconds(_H("-5")) is None
    assert client._retry_after_seconds(_H("soon")) is None
    assert client._retry_after_seconds(None) is None


def test_tts_wrapper_carries_evidence_across_redaction():
    """Wrapping a provider error in a redacted adapter error is correct;
    losing the HTTP status while doing it is not."""
    from nodes._otr_audio_engines import eng_google_tts as tts

    client = _client()
    source = client._attach_evidence(
        client.GoogleAPIBillingOrQuotaError("boom"), status=403)
    target = tts._carry_evidence(tts.GoogleTTSError("redacted"), source)
    assert target.http_status == 403
    assert target.failure_kind == "auth"


def test_tts_posts_through_the_shared_client(monkeypatch):
    """r4 SF-1/CUT-3: the adapter must not hand-roll a second HTTP parser."""
    from nodes._otr_audio_engines import eng_google_tts as tts

    seen = {}

    def _fake_post_json(path, payload, *, timeout_s=None, _api_key=None, **_kw):
        seen["path"] = path
        seen["api_key"] = _api_key
        return {"output_audio": {"data": "", "sample_rate": 24000}}

    monkeypatch.setattr("nodes._otr_google_api.client.post_json", _fake_post_json)
    tts._post_interaction("KEY", {"model": "m", "input": "hi"})
    assert seen["path"] == "/v1beta/interactions"
    assert seen["api_key"] == "KEY"


# --------------------------------------------------------------------------- #
# Artifact validation at the pinned thresholds
# --------------------------------------------------------------------------- #


def _png(tmp_path, pixels):
    from PIL import Image

    path = tmp_path / "x.png"
    img = Image.new("L", (len(pixels), 1))
    img.putdata(pixels)
    img.save(path)
    return path


def test_image_flat_frame_is_invalid(tmp_path):
    assert probe.validate_image(_png(tmp_path, [7, 7, 7, 7]))["valid"] is False


def test_dark_but_detailed_image_is_valid(tmp_path):
    """A brightness floor would reject an intentionally dark Macbeth scene;
    the pinned contract is spatial luma RANGE, not absolute brightness."""
    result = probe.validate_image(_png(tmp_path, [0, 1, 0, 2]))
    assert result["valid"] is True
    assert result["luma_max"] <= 2


def _wav(tmp_path, *, rate=24000, channels=1, frames=b"\x01\x00"):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)
    return path


def test_audio_pins_rate_channels_and_nonzero(tmp_path):
    assert probe.validate_audio(_wav(tmp_path))["valid"] is True


def test_audio_all_zero_samples_is_invalid(tmp_path):
    out = probe.validate_audio(_wav(tmp_path, frames=b"\x00\x00" * 8))
    assert out["valid"] is False
    assert "zero" in out["reason"]


def test_audio_wrong_sample_rate_is_invalid(tmp_path):
    out = probe.validate_audio(_wav(tmp_path, rate=16000))
    assert out["valid"] is False
    assert "sample_rate" in out["reason"]


def test_audio_rate_is_pinned_not_taken_from_the_provider(tmp_path):
    """Validating the response's rate against itself is tautological -- the
    assertion could never fail, which is the same as not having one.

    Asserted BEHAVIOURALLY: the default really is the pin, and a file at a
    different rate really is rejected when the caller passes no expectation.
    """
    import inspect

    assert probe.PINNED_AUDIO_RATE_HZ == 24000
    assert probe.PINNED_AUDIO_CHANNELS == 1

    sig = inspect.signature(probe.validate_audio)
    assert sig.parameters["expect_rate"].default == probe.PINNED_AUDIO_RATE_HZ
    assert sig.parameters["expect_channels"].default == probe.PINNED_AUDIO_CHANNELS

    # The whole point: a 48k file fails even though nobody told validate_audio
    # what to expect -- the pin, not the provider, decides.
    assert probe.validate_audio(_wav(tmp_path, rate=48000))["valid"] is False


def test_audio_stereo_is_invalid(tmp_path):
    out = probe.validate_audio(
        _wav(tmp_path, channels=2, frames=b"\x01\x00\x01\x00"))
    assert out["valid"] is False
    assert "channels" in out["reason"]


# --------------------------------------------------------------------------- #
# Report: created before spending, atomic replacement, attempt state machine
# --------------------------------------------------------------------------- #


def _report(tmp_path):
    return probe.ProbeReport(tmp_path / "report.json", {
        "schema_version": probe.REPORT_SCHEMA_VERSION,
        "run_id": "test",
        "cells": {"A1": {"cell_id": "A1", "attempts": []}},
    })


def test_report_write_is_atomic_and_leaves_no_temp(tmp_path):
    rep = _report(tmp_path)
    rep.write()
    assert rep.path.is_file()
    assert json.loads(rep.path.read_text(encoding="utf-8"))["run_id"] == "test"
    assert not list(tmp_path.glob("*.tmp-*"))


def test_report_write_stamps_updated_at(tmp_path):
    rep = _report(tmp_path)
    rep.write()
    assert rep.data["updated_at"].endswith("Z")


def test_attempt_is_persisted_before_the_call(tmp_path):
    """A receipt written after the POST is a receipt that can be lost."""
    rep = _report(tmp_path)
    rep.write()
    attempt = rep.start_attempt("A1")
    on_disk = json.loads(rep.path.read_text(encoding="utf-8"))
    assert on_disk["cells"]["A1"]["attempts"][0]["status"] == "RUNNING"
    # Starts PREPARING; the runner moves it to SUBMITTING immediately before
    # the call actually goes out.
    assert attempt["state"] == probe.ST_PREPARING
    rep.transition("A1", attempt, probe.ST_SUBMITTING)
    on_disk = json.loads(rep.path.read_text(encoding="utf-8"))
    assert on_disk["cells"]["A1"]["attempts"][0]["state"] == probe.ST_SUBMITTING


def test_operation_name_is_persisted_at_acceptance(tmp_path):
    rep = _report(tmp_path)
    rep.write()
    attempt = rep.start_attempt("A1")
    rep.transition("A1", attempt, probe.ST_ACCEPTED, operation_name="ops/123")
    on_disk = json.loads(rep.path.read_text(encoding="utf-8"))
    assert on_disk["cells"]["A1"]["attempts"][0]["operation_name"] == "ops/123"
    assert on_disk["cells"]["A1"]["attempts"][0]["state"] == probe.ST_ACCEPTED


# --------------------------------------------------------------------------- #
# Crash points -- no ambiguous state may ever auto-resubmit
# --------------------------------------------------------------------------- #


class _Rep(probe.ProbeReport):
    """A report that records outcomes without touching disk."""

    def __init__(self):
        super().__init__(pathlib.Path("unused.json"), {
            "run_id": "t",
            "cells": {cid: {"cell_id": cid, "attempts": []}
                      for cid in ("A1", "A2", "A3", "A4")},
        })
        self.writes = 0

    def write(self, *, retries=5):
        self.writes += 1


def test_a_failure_before_transmission_is_not_called_orphaned():
    """PREPARING covers imports, payload building and the A2 tensor check --
    nothing has been sent. Reporting those as "the request was transmitted"
    sends the operator hunting for a charge that never happened."""
    assert probe.ST_PREPARING not in probe.AMBIGUOUS_STATES
    rep = _Rep()
    attempt = rep.start_attempt("A3")            # defaults to PREPARING
    assert attempt["state"] == probe.ST_PREPARING
    outcome = probe._record_exception(rep, "A3", attempt, ImportError("boom"))
    assert outcome == probe.UNKNOWN
    assert rep.cell("A3")["outcome_substate"] != probe.ORPHANED
    assert "orphan_note" not in rep.cell("A3")["error"]


def test_a2_cloud_error_code_is_preserved_in_evidence():
    """The installed Comfy client raises a bare Exception; OTR's own
    CloudMediaError code is the only structure an A2 failure has."""
    rep = _Rep()
    attempt = rep.start_attempt("A2")
    exc = RuntimeError("partner failed")
    exc.code = "PROVIDER_REJECTED"
    probe._record_exception(rep, "A2", attempt, exc)
    assert rep.cell("A2")["error"]["cloud_error_code"] == "PROVIDER_REJECTED"


@pytest.mark.parametrize("state", sorted(probe.AMBIGUOUS_STATES))
def test_crash_in_an_ambiguous_state_is_orphaned_not_retried(state):
    """SUBMITTING / ACCEPTED / POLLING all mean the request was transmitted.
    Without a verified provider idempotency key a crash there is TERMINAL
    UNKNOWN -- resubmitting would double-charge a paid call."""
    rep = _Rep()
    attempt = rep.start_attempt("A3", state)
    outcome = probe._record_exception(rep, "A3", attempt, RuntimeError("reset"))
    assert outcome == probe.UNKNOWN
    assert rep.cell("A3")["outcome_substate"] == probe.ORPHANED
    assert "NEVER auto-resubmitted" in rep.cell("A3")["error"]["orphan_note"]


def test_a_charged_job_records_its_receipt_before_the_name_is_parsed():
    """`_extract_operation_name` raises on a malformed envelope. If that ran
    BEFORE the receipt was persisted, an already-CHARGED job would be
    stranded with nothing on disk to poll or reconcile it by."""
    import inspect

    src = inspect.getsource(probe.run_cell_a3)
    post = src.index("predictLongRunning")
    persisted = src.index("accepted_envelope_keys")
    parsed = src.index("_extract_operation_name(operation)")
    assert post < persisted < parsed, "receipt must be persisted before parsing"


def test_structured_refusal_survives_the_exception_path():
    rep = _Rep()
    attempt = rep.start_attempt("A1")
    exc = RuntimeError("provider said no")
    exc.http_status = 400
    exc.response_json = {"finish_reason": "prohibited_content"}
    outcome = probe._record_exception(rep, "A1", attempt, exc)
    assert outcome == probe.SAFETY_REFUSAL


@pytest.mark.parametrize("status,expected", [
    (401, probe.INFRA_BLOCKED),
    (403, probe.INFRA_BLOCKED),
    (500, probe.PROVIDER_ERROR),
])
def test_http_status_maps_to_outcome(status, expected):
    rep = _Rep()
    attempt = rep.start_attempt("A1")
    exc = RuntimeError("boom")
    exc.http_status = status
    assert probe._record_exception(rep, "A1", attempt, exc) == expected


def test_unstructured_failure_is_unknown_never_refusal():
    """A2's installed Comfy client raises a BARE Exception with no .status
    and no .body, so an A2 failure is structurally unclassifiable."""
    rep = _Rep()
    attempt = rep.start_attempt("A1")
    outcome = probe._record_exception(
        rep, "A1", attempt, Exception("Comfy partner call failed"))
    assert outcome == probe.UNKNOWN
    assert rep.cell("A1")["outcome"] != probe.SAFETY_REFUSAL


def test_evidence_hash_is_recorded():
    rep = _Rep()
    attempt = rep.start_attempt("A1")
    probe._record_exception(rep, "A1", attempt, RuntimeError("x"))
    assert len(rep.cell("A1")["error"]["evidence_sha256"]) == 64


def test_evidence_is_an_allowlist_not_a_raw_body():
    """A provider body can echo the REQUEST back, and the request carries the
    API key -- so evidence is projected onto an allowlist rather than dumped
    and scrubbed."""
    out = probe.sanitize_response({
        "finish_reason": "safety",
        "error": {"code": 400, "status": "INVALID", "message": "key=SECRET123"},
        "candidates": [{"content": "a whole generation"}],
        "echoed_request": {"x-goog-api-key": "SECRET123"},
    })
    assert out["finish_reason"] == "safety"
    assert out["error"] == {"code": 400, "status": "INVALID"}
    assert "message" not in out["error"]
    assert "candidates" not in out
    assert "echoed_request" not in out
    assert "SECRET123" not in json.dumps(out)


def test_recorded_evidence_never_contains_a_raw_response_body():
    rep = _Rep()
    attempt = rep.start_attempt("A1")
    exc = RuntimeError("boom")
    exc.http_status = 400
    exc.response_json = {"finish_reason": "safety",
                         "echoed_request": {"key": "SECRET123"}}
    probe._record_exception(rep, "A1", attempt, exc)
    serialized = json.dumps(rep.cell("A1")["error"])
    assert "SECRET123" not in serialized
    assert "response_json" not in rep.cell("A1")["error"]
    assert rep.cell("A1")["error"]["response_fields"]["finish_reason"] == "safety"


# --------------------------------------------------------------------------- #
# Bounded polling -- retry polling, never the paid submit
# --------------------------------------------------------------------------- #


def test_poll_caps_each_request_and_never_exceeds_the_deadline(monkeypatch):
    """The engine's own ``_poll_operation`` hands the FULL campaign timeout to
    every GET, so a single wedged request can eat the whole budget and the
    loop never reaches its deadline check again."""
    seen = []

    def _fake_get_json(path, *, timeout_s=None, _api_key=None, **_kw):
        seen.append(timeout_s)
        return {"done": True, "name": path}

    monkeypatch.setattr("nodes._otr_google_api.client.get_json", _fake_get_json)
    out = probe.poll_operation_bounded("models/x/operations/1", api_key="K")
    assert out["done"] is True
    assert seen and all(t <= probe.POLL_REQUEST_CAP_S for t in seen)


def test_poll_raises_rather_than_resubmitting_when_it_runs_out_of_time(monkeypatch):
    monkeypatch.setattr(
        "nodes._otr_google_api.client.get_json",
        lambda path, **_kw: {"done": False})
    with pytest.raises(probe.ProbeError) as exc:
        probe.poll_operation_bounded("ops/1", api_key="K", deadline_s=1)
    assert "NOT resubmitted" in str(exc.value) or "did not finish" in str(exc.value)


def test_poll_survives_a_transient_error_without_resubmitting(monkeypatch):
    """Retrying a GET against a STORED operation_name cannot double-charge --
    the paid submit is not replayed. Aborting on a 429 blip would burn the
    whole $0.20 attempt and label it ORPHANED when nothing is ambiguous."""
    from nodes._otr_google_api import client

    calls = {"n": 0}

    def _flaky(path, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise client._attach_evidence(
                client.GoogleAPIBillingOrQuotaError("rate limited"), status=429)
        return {"done": True}

    monkeypatch.setattr("nodes._otr_google_api.client.get_json", _flaky)
    monkeypatch.setattr(probe, "POLL_INTERVAL_S", 0)
    out = probe.poll_operation_bounded("ops/1", api_key="K", deadline_s=60)
    assert out["done"] is True
    assert calls["n"] == 2


def test_poll_propagates_a_non_retryable_error(monkeypatch):
    """A 403 is not a blip; retrying it just burns the deadline."""
    from nodes._otr_google_api import client

    def _denied(path, **_kw):
        raise client._attach_evidence(
            client.GoogleAPIBillingOrQuotaError("denied"), status=403)

    monkeypatch.setattr("nodes._otr_google_api.client.get_json", _denied)
    with pytest.raises(client.GoogleAPIError):
        probe.poll_operation_bounded("ops/1", api_key="K", deadline_s=60)


def test_poll_uses_the_engine_path_builder(monkeypatch):
    """A bare name, an absolute path, and a full URL are all valid operation
    names; reimplementing that mapping is how one of the three breaks."""
    seen = {}

    def _fake_get_json(path, **_kw):
        seen["path"] = path
        return {"done": True}

    monkeypatch.setattr("nodes._otr_google_api.client.get_json", _fake_get_json)
    probe.poll_operation_bounded("models/veo/operations/abc", api_key="K")
    assert seen["path"] == "/v1beta/models/veo/operations/abc"
    probe.poll_operation_bounded("https://example/op", api_key="K")
    assert seen["path"] == "https://example/op"


# --------------------------------------------------------------------------- #
# Dependency propagation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("upstream", [
    probe.SAFETY_REFUSAL, probe.INFRA_BLOCKED, probe.PROVIDER_ERROR,
    probe.INVALID_ARTIFACT, probe.UNKNOWN,
])
def test_a2_is_dependency_blocked_on_every_non_pass_a1(upstream, tmp_path):
    """A2's init image is loaded FROM DISK by the engine; there is no
    substitute, so any non-PASS A1 blocks A2 rather than failing it."""
    rep = _Rep()
    frozen = {"cells": {"A2": {"provider_bound_input": "x"}}}
    outcome = probe.run_cell_a2(rep, frozen, tmp_path,
                                init_image=None, upstream=upstream)
    assert outcome == probe.DEPENDENCY_BLOCKED
    assert rep.cell("A2")["blocked_by_cell"] == "A1"
    assert rep.cell("A2")["upstream_outcome"] == upstream


def test_a2_drives_the_shipped_engine_and_satisfies_the_partner_row(tmp_path):
    """The cloud_wan_i2v row REQUIRES first_frame/model/prompt_extend/seed/
    watermark, with the text prompt NESTED inside `model` -- hand-building
    those inputs (image/prompt/duration) is rejected by
    _validate_declared_inputs and would have crashed the first live run.
    Driving the shipped engine is also what makes A2 measure production."""
    import inspect

    from PIL import Image

    from nodes._otr_video_engines.eng_cloud_video import CloudWanI2VEngine

    src = inspect.getsource(probe.run_cell_a2)
    assert "invoke_partner_node" not in src
    assert "CloudWanI2VEngine" in src

    still = tmp_path / "init.png"
    Image.new("RGB", (64, 36), (10, 10, 10)).save(still)
    engine = CloudWanI2VEngine()
    # THE SHARED builder the runner uses -- not a hand-copied duplicate that
    # would keep passing after a key rename and only break on the paid run.
    request = probe.a2_request("bloody daggers", still)
    inputs = engine._partner_inputs(request)
    assert set(inputs) >= {"first_frame", "model", "prompt_extend", "seed",
                           "watermark"}
    assert inputs["model"]["prompt"]
    assert inputs["seed"] == 20260808          # seed_bundle.request_seed, not .seed


def test_a2_blocked_when_init_image_absent_from_disk(tmp_path):
    rep = _Rep()
    frozen = {"cells": {"A2": {"provider_bound_input": "x"}}}
    outcome = probe.run_cell_a2(rep, frozen, tmp_path,
                                init_image=tmp_path / "nope.png",
                                upstream=probe.PASS)
    assert outcome == probe.DEPENDENCY_BLOCKED


# --------------------------------------------------------------------------- #
# Billing basis
# --------------------------------------------------------------------------- #


def test_billing_distinguishes_reserved_from_actual():
    basis = probe._billing("A3")
    assert basis["reserved_usd"] == probe.RESERVED_USD["A3"]
    assert basis["billed_estimate_usd"] is None
    assert basis["unit"] == "usd"


def test_a2_provider_cost_is_null_no_invented_conversion():
    """Comfy bills in credits; inventing a credit->USD rate would put a
    fabricated number in a report the operator reads as evidence."""
    assert probe._billing("A2", provider_cost=None)["provider_cost"] is None


def test_image_request_uses_the_top_level_width_the_engine_reads():
    """``_canvas_wh`` reads TOP-LEVEL width/height and never looks inside a
    ``canvas`` dict, so a canvas-only request silently yields the 832x1216
    PORTRAIT default -- a portrait still handed to a 16:9 video arm."""
    from nodes._otr_image_engines.eng_google_image import (
        _aspect_for_request, _canvas_wh)

    req = probe.image_request("x")
    assert _canvas_wh(req) == (probe.PINNED_CANVAS_W, probe.PINNED_CANVAS_H)
    assert _aspect_for_request(req) == "16:9"


def test_video_request_pins_the_billed_duration_without_an_env_var(monkeypatch):
    """Veo derives its billed duration from target_frame_count / fps and
    falls back to EIGHT seconds when it cannot -- double the reserved spend.
    The request must carry the frames itself."""
    from nodes._otr_video_engines.eng_google_veo_video import _duration_s

    for name in ("OTR_GOOGLE_VEO_DURATION_S", "OTR_GOOGLE_VIDEO_DURATION_S"):
        monkeypatch.delenv(name, raising=False)
    req = probe.video_request("x", shot_id="s")
    assert _duration_s(req, resolution="720p") == probe.PINNED_VIDEO_DURATION_S


def test_duration_preflight_fails_closed_when_the_engine_would_bill_more(monkeypatch):
    monkeypatch.setenv("OTR_GOOGLE_VEO_DURATION_S", "8")
    with pytest.raises(probe.PreflightError) as exc:
        probe.assert_resolved_video_duration()
    assert "reserved" in str(exc.value)


def test_reserved_total_matches_the_locked_scope():
    assert sorted(probe.RESERVED_USD) == ["A1", "A2", "A3", "A4"]
    assert round(sum(probe.RESERVED_USD.values()), 3) == 0.725


# --------------------------------------------------------------------------- #
# CLI contract
# --------------------------------------------------------------------------- #


def test_live_and_dry_run_are_mutually_exclusive():
    parser = probe.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--live", "--dry-run"])


def test_a_mode_is_required_there_is_no_live_default():
    parser = probe.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_only_arm_a2_flag_was_cut():
    """r4 CUT-2: a diagnostic runs A1+A2 together for negligible cost and
    preserves the real dependency, so the external-init-image subprotocol
    never needed to exist."""
    parser = probe.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dry-run", "--only-arm", "A2"])


def test_only_cell_is_accepted():
    args = probe.build_arg_parser().parse_args(["--live", "--only-cell", "A1"])
    assert args.only_cell == ["A1"]


# --------------------------------------------------------------------------- #
# Output path contract
# --------------------------------------------------------------------------- #


def test_run_dir_is_not_a_reserved_underscore_entry():
    """r4 MF-6: ``otr/episodes/_probe_macbeth`` collides with the output-tree
    contract -- ``_otr_paths`` reserves every underscore-prefixed episode
    entry and the production write helpers raise on them."""
    from nodes._otr_paths import is_reserved_episode_entry

    run_dir = probe._run_dir_for("20260808T120000Z")
    assert run_dir.name == "macbeth_probe_20260808T120000Z"
    assert is_reserved_episode_entry(run_dir.name) is False
