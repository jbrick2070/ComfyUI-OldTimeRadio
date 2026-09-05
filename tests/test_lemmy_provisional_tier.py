"""The PROVISIONAL voice-route tier -- a second, deliberately weaker claim.

WHAT IS BEING PROTECTED HERE, and it is not "the tier works". The tier is a
convenience: it stops Lemmy being a different stranger every episode on the five
engines where no audition has ever run. What these tests protect is the set of
properties that make the convenience safe to ship while the operator's ears are
deferred:

  1. A PROVISIONAL ROW CAN NEVER PASS AS QUALIFIED. `is_qualified_route` is False
     for every provisional row and there is no field one could gain to flip it.
  2. A PROVISIONAL CLAIM CAN NEVER STAMP `voice_route`. That field means "a
     qualified route was proved"; the voice node raises on any non-empty one
     whose status is not `qualified`, so a provisional claim carrying one would
     kill EVERY render on these engines. The claim type has no such attribute,
     and this file pins that absence.
  3. IT DEGRADES, IT NEVER RAISES. A missing bank row, a moved reference, bytes
     that no longer match -- all of them return a closed reason code and the row
     takes the ordinary draw. A render must not die over an unauditioned
     convenience row.
  4. NO SURROGATE VERDICT. `operator_verdict` and `decided_by` may not appear in
     a provisional receipt at all -- not blank, ABSENT -- so nobody can promote a
     route by filling in a field.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from config import cast_pools as POOLS
from nodes import _otr_voice_route as ROUTE
from nodes._otr_voice_bank import load_voice_bank
from nodes.cast_lock import CastLock

# Real bank rows, so the resolver's bank checks have something true to find.
CLONE_REF = "cb_peter_yearsley"          # chatterbox, a mirrored local wav
KOKORO_REF = "bm_george"                 # kokoro, a `.pt` voice tensor
PROVIDER_REF = "el_daniel"               # elevenlabs, a provider voice id
PROVIDER_VOICE_ID = "onwK4e9ZLuTAKqWW03F9"

CAST = [
    {"char_id": "c01", "name": "MONTY", "gender": "male",
     "voice_preset": "v2/en_speaker_1"},
    {"char_id": "c02", "name": "LEMMY", "gender": "male",
     "voice_preset": "v2/en_speaker_8"},
    {"char_id": "a1", "name": "ANNOUNCER", "gender": "male",
     "voice_preset": "v2/en_speaker_6"},
]


def _ledger(cast=None, meta=None):
    return json.dumps({
        "meta": meta or {"episode_seed": 42},
        "cast": json.loads(json.dumps(cast if cast is not None else CAST)),
        "lines": [],
    })


def _provisional(engine, voice_ref_id, identity_kind, *, state="rendered_pending_listen",
                 route_id=None, **receipt_extra):
    receipt = {
        "engine": engine,
        "identity_kind": identity_kind,
        "identity_id": voice_ref_id,
        "state": state,
    }
    if state == "rendered_pending_listen":
        receipt.update({
            "audition_manifest_path": "otr/episodes/lemmy_cross_engine/MANIFEST.json",
            "audition_manifest_sha256": "a" * 64,
            "neutral_clip_path": "otr/episodes/lemmy_cross_engine/%s_neutral.wav" % engine,
            "neutral_clip_sha256": "b" * 64,
            "emotional_clip_path": "otr/episodes/lemmy_cross_engine/%s_emotional.wav" % engine,
            "emotional_clip_sha256": "c" * 64,
            "rendered_utc": "2026-08-16T21:00:00Z",
        })
    if identity_kind == "provider_voice":
        receipt.setdefault("provider", engine)
        receipt.setdefault("provider_voice_id", PROVIDER_VOICE_ID)
    receipt.update(receipt_extra)
    return {
        "route_id": route_id or "lemmy-%s-provisional-v1" % engine,
        "route_contract_version": 1,
        "engine": engine,
        "voice_ref_id": voice_ref_id,
        "provisional_receipt": receipt,
    }


def _policy(provisional=None, *, qualified=None, character_key="lemmy"):
    return {
        "policy_version": "lemmy-cockney-v1-test",
        "character_key": character_key,
        "approved_native_routes": qualified or {},
        "provisional_native_routes": provisional or {},
    }


@pytest.fixture()
def pin_provisional(monkeypatch):
    """Install a policy carrying only provisional routes."""
    def install(routes, **kw):
        monkeypatch.setattr("nodes.cast_lock._lemmy_voice_policy",
                            lambda: _policy(routes, **kw))
    return install


# ---------------------------------------------------------------------------
# 1. THE SAFETY PROPERTY: the two tiers are mutually exclusive.
# ---------------------------------------------------------------------------


def test_every_shipped_provisional_row_is_well_formed():
    for engine, record in POOLS.LEMMY_VOICE_POLICY["provisional_native_routes"].items():
        problems = POOLS.provisional_route_problems(record, engine=engine)
        assert problems == [], "%s: %s" % (engine, problems)


def test_the_two_cloud_rows_are_configured_and_NOT_rendered():
    """They were never rendered and the sprint report may not say they were:
    both engines need an API key and spend money, and the root scope rule is
    100% local, no API keys, no paid services."""
    rows = POOLS.LEMMY_VOICE_POLICY["provisional_native_routes"]
    for engine in ("google_tts", "elevenlabs"):
        receipt = rows[engine]["provisional_receipt"]
        assert receipt["state"] == "configured_unrendered"
        assert receipt["identity_kind"] == "provider_voice"
        for field in POOLS._PROVISIONAL_RECEIPT_RENDER_ARTIFACTS:
            assert field not in receipt, "%s claims %s without rendering" % (engine, field)


def test_bark_deliberately_has_no_provisional_row():
    """Not an omission. Bark has ZERO bank rows, its adapter reads `voice_preset`,
    and `lemmy_row()` already pins him at writer time -- there is no defect to fix
    and the route machinery cannot express a preset."""
    assert "bark" not in POOLS.LEMMY_VOICE_POLICY["provisional_native_routes"]


def test_the_contract_version_cannot_drift_from_the_validator():
    """The tier duplicates the supported-version tuple because config may not
    import a node. Duplication is fine; SILENT drift is not."""
    assert (POOLS.PROVISIONAL_ROUTE_CONTRACT_VERSIONS
            == ROUTE.SUPPORTED_ROUTE_CONTRACT_VERSIONS)


# ---------------------------------------------------------------------------
# 2. NO SURROGATE VERDICT.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["operator_verdict", "decided_by", "approved_by",
                                   "verdict"])
def test_a_receipt_carrying_a_judgement_field_is_refused(field):
    record = _provisional("chatterbox", CLONE_REF, "local_wav", **{field: "sounds fine"})
    problems = POOLS.provisional_route_problems(record, engine="chatterbox")
    assert any("banned field" in p for p in problems), problems


def test_no_shipped_receipt_carries_a_judgement_field():
    for engine, record in POOLS.LEMMY_VOICE_POLICY["provisional_native_routes"].items():
        receipt = record["provisional_receipt"]
        assert not (POOLS.PROVISIONAL_RECEIPT_BANNED_FIELDS & set(receipt)), engine


@pytest.mark.parametrize("key", ["qualification_receipt", "qualification_record"])
def test_a_provisional_route_may_not_wear_qualified_machinery(key):
    record = _provisional("chatterbox", CLONE_REF, "local_wav")
    record[key] = {"anything": "at all"}
    problems = POOLS.provisional_route_problems(record, engine="chatterbox")
    assert any("qualified-tier key" in p for p in problems), problems


# ---------------------------------------------------------------------------
# 3. THE STATE-DEPENDENT RECEIPT. A flat field set cannot express this matrix.
# ---------------------------------------------------------------------------
def test_a_rendered_route_must_carry_BOTH_clips():
    """The frozen audition is two lines -- a neutral one and an emotional one --
    and the emotional one is the whole reason the pair exists. A receipt with one
    clip has not proved the thing most likely to break."""
    record = _provisional("chatterbox", CLONE_REF, "local_wav")
    del record["provisional_receipt"]["emotional_clip_sha256"]
    problems = POOLS.provisional_route_problems(record, engine="chatterbox")
    assert any("emotional_clip_sha256" in p for p in problems), problems


def test_an_unrendered_route_may_not_claim_render_artifacts():
    record = _provisional("google_tts", "gt_algenib", "provider_voice",
                          state="configured_unrendered",
                          neutral_clip_path="somewhere/nice.wav")
    problems = POOLS.provisional_route_problems(record, engine="google_tts")
    assert any("neutral_clip_path" in p and "did not render" in p
               for p in problems), problems


def test_an_undefined_state_is_refused():
    record = _provisional("chatterbox", CLONE_REF, "local_wav", state="probably_fine")
    problems = POOLS.provisional_route_problems(record, engine="chatterbox")
    assert any("not a defined state" in p for p in problems), problems


# ---------------------------------------------------------------------------
# 4. THE LOCAL-CLONE ALLOWLIST. Prose is not enforcement.
# ---------------------------------------------------------------------------
def test_a_provider_engine_may_not_take_the_reference_wav():
    """`rights.scope` is only ever read as a non-blank string, so a route whose
    scope said "cloud engines only" would validate exactly like one that said the
    opposite. The enforcement is a list."""
    record = _provisional("elevenlabs", PROVIDER_REF, "local_wav")
    problems = POOLS.provisional_route_problems(record, engine="elevenlabs")
    assert any("local clone engines only" in p for p in problems), problems


def test_no_provider_engine_is_on_the_clone_allowlist():
    assert not ({"elevenlabs", "google_tts"}
                & POOLS.PROVISIONAL_LOCAL_CLONE_ENGINES)


def test_a_provider_route_needs_its_provider_fields():
    record = _provisional("elevenlabs", PROVIDER_REF, "provider_voice",
                          state="configured_unrendered")
    del record["provisional_receipt"]["provider_voice_id"]
    problems = POOLS.provisional_route_problems(record, engine="elevenlabs")
    assert any("provider_voice_id is missing" in p for p in problems), problems


def test_an_engine_filed_under_the_wrong_key_is_refused():
    record = _provisional("chatterbox", CLONE_REF, "local_wav")
    problems = POOLS.provisional_route_problems(record, engine="dia")
    assert any("ENGINE DISAGREEMENT" in p for p in problems), problems


# ---------------------------------------------------------------------------
# 5. THE SIBLING RESOLVER.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bank():
    return load_voice_bank()[0]


def _resolve(policy, engine, bank_entries):
    from nodes._otr_voice_node_common import _resolve_ref_to_disk
    return ROUTE.resolve_provisional_route_claim(
        policy, engine, bank_entries=bank_entries, repo_root=REPO_ROOT,
        path_resolver=_resolve_ref_to_disk)


def test_a_resolved_claim_HAS_NO_voice_route_ATTRIBUTE(bank):
    """THE structural pin. `cast_lock` does `dict(policy_claim.voice_route)`
    unconditionally in both stamp branches, so a claim type carrying a None route
    is a server crash one branch away. This type cannot reach those lines because
    it has no such attribute -- the absence IS the safety property."""
    claim = _resolve(_policy({"chatterbox": _provisional(
        "chatterbox", CLONE_REF, "local_wav")}), "chatterbox", bank)
    assert isinstance(claim, ROUTE.ProvisionalPolicyClaim)
    assert not hasattr(claim, "voice_route")
    with pytest.raises(AttributeError):
        claim.voice_route


@pytest.mark.parametrize("engine,ref,kind", [
    ("chatterbox", CLONE_REF, "local_wav"),
    ("kokoro", KOKORO_REF, "bank_voice_id"),
    ("elevenlabs", PROVIDER_REF, "provider_voice"),
])
def test_all_three_identity_kinds_resolve(bank, engine, ref, kind):
    """Three kinds, three code paths: a WAV path, a `.pt` named by bank id, and a
    cloud voice id with no local bytes at all."""
    state = ("configured_unrendered" if kind == "provider_voice"
             else "rendered_pending_listen")
    claim = _resolve(_policy({engine: _provisional(engine, ref, kind, state=state)}),
                     engine, bank)
    assert isinstance(claim, ROUTE.ProvisionalPolicyClaim), claim
    assert claim.engine == engine
    assert claim.voice_ref_id == ref
    assert claim.identity_kind == kind
    assert claim.tier == ROUTE.ROUTE_TIER_PROVISIONAL


def test_a_qualified_engine_never_reaches_the_lower_tier(bank):
    """Belt and braces on top of the caller's ordering: an engine present in BOTH
    dicts resolves qualified, and the provisional selector refuses it by name."""
    policy = _policy({"indextts2": _provisional("indextts2", "vz_peter_yearsley",
                                                "local_wav")},
                     qualified={"indextts2": {"route_id": "whatever"}})
    result = ROUTE.select_provisional_route(policy, "indextts2")
    assert isinstance(result, ROUTE.ProvisionalRouteDegradation)
    assert result.reason_code == "no_route_for_engine"
    assert "QUALIFIED" in result.detail


@pytest.mark.parametrize("policy,engine,expected", [
    (None, "kokoro", "no_policy"),
    (_policy({}), "kokoro", "no_provisional_routes"),
    (_policy({"kokoro": _provisional("kokoro", KOKORO_REF, "bank_voice_id")}),
     None, "engine_unresolved"),
    (_policy({"kokoro": _provisional("kokoro", KOKORO_REF, "bank_voice_id")}),
     "dia", "no_route_for_engine"),
])
def test_selection_degrades_with_a_closed_reason_code(policy, engine, expected):
    result = ROUTE.select_provisional_route(policy, engine)
    assert isinstance(result, ROUTE.ProvisionalRouteDegradation)
    assert result.reason_code == expected
    assert result.reason_code in ROUTE.PROVISIONAL_REASON_CODES


def test_a_missing_bank_row_degrades_rather_than_raising(bank):
    result = _resolve(_policy({"chatterbox": _provisional(
        "chatterbox", "cb_nobody_by_that_name", "local_wav")}), "chatterbox", bank)
    assert isinstance(result, ROUTE.ProvisionalRouteDegradation)
    assert result.reason_code == "bank_row_missing"


def test_an_ambiguous_bank_row_degrades(bank):
    """Two rows sharing one id on one engine means the bank cannot say which
    bytes this route means. That is a REFUSAL with a cause, not a coin flip."""
    twin = next(e for e in bank if e.voice_ref_id == CLONE_REF)
    result = _resolve(_policy({"chatterbox": _provisional(
        "chatterbox", CLONE_REF, "local_wav")}), "chatterbox",
        list(bank) + [twin])
    assert isinstance(result, ROUTE.ProvisionalRouteDegradation)
    assert result.reason_code == "bank_row_ambiguous"


def test_a_reference_that_is_not_on_disk_degrades(bank, tmp_path):
    result = ROUTE.resolve_provisional_route_claim(
        _policy({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")}),
        "chatterbox", bank_entries=bank, repo_root=str(tmp_path),
        path_resolver=lambda p: str(tmp_path / "gone.wav"))
    assert isinstance(result, ROUTE.ProvisionalRouteDegradation)
    assert result.reason_code == "reference_file_missing"


def test_reference_bytes_that_do_not_match_the_bank_degrade(bank, tmp_path):
    """A file that exists and is the WRONG file is worse than a missing one, and
    it is the shape that survives a re-render under the same name."""
    impostor = tmp_path / "impostor.wav"
    impostor.write_bytes(b"not the audition reference at all")
    result = ROUTE.resolve_provisional_route_claim(
        _policy({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")}),
        "chatterbox", bank_entries=bank, repo_root=str(tmp_path),
        path_resolver=lambda p: str(impostor))
    assert isinstance(result, ROUTE.ProvisionalRouteDegradation)
    assert result.reason_code == "reference_bytes_mismatch"


def test_a_provider_id_that_disagrees_with_the_bank_degrades(bank):
    record = _provisional("elevenlabs", PROVIDER_REF, "provider_voice",
                          state="configured_unrendered",
                          provider_voice_id="some-other-voice-id")
    result = _resolve(_policy({"elevenlabs": record}), "elevenlabs", bank)
    assert isinstance(result, ROUTE.ProvisionalRouteDegradation)
    assert result.reason_code == "provider_identity_mismatch"


def test_a_malformed_record_degrades_and_names_the_field(bank):
    record = _provisional("chatterbox", CLONE_REF, "local_wav")
    record["route_contract_version"] = 99
    result = _resolve(_policy({"chatterbox": record}), "chatterbox", bank)
    assert isinstance(result, ROUTE.ProvisionalRouteDegradation)
    assert result.reason_code == "record_malformed"
    assert "route_contract_version" in result.detail


@pytest.mark.parametrize("record", [
    {}, {"route_id": ""}, {"route_id": "x", "provisional_receipt": "a string"},
    {"route_id": "x", "route_contract_version": 1, "engine": "chatterbox",
     "provisional_receipt": {"state": "rendered_pending_listen"}},
])
def test_the_resolver_never_raises_on_a_broken_record(bank, record):
    """A render must not die over an unauditioned convenience row. Whatever
    arrives, the answer is a reason code."""
    result = _resolve(_policy({"chatterbox": record}), "chatterbox", bank)
    assert isinstance(result, ROUTE.ProvisionalRouteDegradation)
    assert result.reason_code in ROUTE.PROVISIONAL_REASON_CODES


# ---------------------------------------------------------------------------
# 6. CASTLOCK -- auto_registry.
# ---------------------------------------------------------------------------
def _rows(ledger_json):
    return {e["char_id"]: e for e in json.loads(ledger_json)["cast"]}


def test_the_claimed_row_takes_the_provisional_identity(pin_provisional):
    pin_provisional({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")})
    out = CastLock().lock(script_json=_ledger(), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy="auto_registry")[0]
    lemmy = _rows(out)["c02"]
    assert lemmy["voice_ref_id"] == CLONE_REF
    assert lemmy["voice_engine"] == "chatterbox"
    assert lemmy["voice_cast_fallback"] == "provisional_route"
    assert lemmy[ROUTE.CAST_ROW_TIER_FIELD] == ROUTE.ROUTE_TIER_PROVISIONAL
    assert lemmy[ROUTE.CAST_ROW_ROUTE_ID_FIELD] == "lemmy-chatterbox-provisional-v1"


def test_a_provisional_stamp_NEVER_writes_voice_route(pin_provisional):
    """The render-killer, pinned. Any non-empty `voice_route` whose status is not
    exactly `qualified` raises `VoiceRouteError` at dispatch -- so this one
    assertion is the difference between a working engine and a dead render."""
    pin_provisional({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")})
    out = CastLock().lock(script_json=_ledger(), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy="auto_registry")[0]
    assert "voice_route" not in _rows(out)["c02"]


def test_only_the_claimed_row_is_touched(pin_provisional):
    baseline = _rows(CastLock().lock(
        script_json=_ledger(), voice_bank="default_clean",
        char_voice_engine="chatterbox", cast_voice_policy="auto_registry")[0])
    pin_provisional({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")})
    after = _rows(CastLock().lock(
        script_json=_ledger(), voice_bank="default_clean",
        char_voice_engine="chatterbox", cast_voice_policy="auto_registry")[0])
    for cid in ("c01", "a1"):
        assert after[cid]["voice_ref_id"] == baseline[cid]["voice_ref_id"], cid
        assert ROUTE.CAST_ROW_TIER_FIELD not in after[cid], cid


def test_a_row_nothing_claims_is_stamped_unrouted_with_its_reason(pin_provisional):
    """`unrouted` is the honest name for the ordinary seeded draw. The field is
    written whatever happens, so a reader never has to tell it apart from a field
    that was never written."""
    pin_provisional({"dia": _provisional("dia", "dia_peter_yearsley", "local_wav")})
    out = CastLock().lock(script_json=_ledger(), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy="auto_registry")[0]
    lemmy = _rows(out)["c02"]
    assert lemmy[ROUTE.CAST_ROW_TIER_FIELD] == ROUTE.ROUTE_TIER_UNROUTED
    assert lemmy[ROUTE.CAST_ROW_REASON_FIELD] == "no_route_for_engine"
    assert lemmy["voice_ref_id"], "an unrouted row still gets an ordinary draw"


def test_a_dormant_policy_stamps_no_tier_field_at_all(monkeypatch):
    """The pre-tier behaviour, byte for byte. Both dicts empty means no match, no
    stamp, and no new field on any row."""
    monkeypatch.setattr("nodes.cast_lock._lemmy_voice_policy",
                        lambda: _policy({}, qualified={}))
    out = CastLock().lock(script_json=_ledger(), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy="auto_registry")[0]
    for row in json.loads(out)["cast"]:
        assert ROUTE.CAST_ROW_TIER_FIELD not in row


def test_the_provisional_tier_survives_an_empty_qualified_dict(pin_provisional):
    """THE COUPLING TEST. The dormancy gate used to key only off
    `approved_native_routes`, which silently made this tier's reachability depend
    on an unrelated dict: demote the one qualified route and every provisional
    route goes dormant with no error."""
    pin_provisional({"kokoro": _provisional("kokoro", KOKORO_REF, "bank_voice_id")},
                    qualified={})
    out = CastLock().lock(script_json=_ledger(), voice_bank="kokoro_builtin",
                          char_voice_engine="kokoro",
                          cast_voice_policy="auto_registry")[0]
    lemmy = _rows(out)["c02"]
    assert lemmy[ROUTE.CAST_ROW_TIER_FIELD] == ROUTE.ROUTE_TIER_PROVISIONAL
    assert lemmy["voice_ref_id"] == KOKORO_REF


# ---------------------------------------------------------------------------
# 7. THE TIER TRANSITION -- stale identity surviving a switch.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy_mode", ["auto_registry", "preserve_ledger"])
def test_a_stale_qualified_route_is_cleared_on_a_switch(pin_provisional, policy_mode):
    """A row locked once under the IndexTTS2 route and re-locked on another engine
    keeps `voice_route`, and the voice node then raises ENGINE DISAGREEMENT -- a
    render killed by a leftover."""
    cast = json.loads(json.dumps(CAST))
    cast[1]["voice_route"] = {"route_id": "lemmy-indextts2-algenib-cockney-v1",
                              "engine": "indextts2", "status": "qualified"}
    pin_provisional({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")})
    out = CastLock().lock(script_json=_ledger(cast=cast), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy=policy_mode)[0]
    assert "voice_route" not in _rows(out)["c02"]


@pytest.mark.parametrize("policy_mode", ["auto_registry", "preserve_ledger"])
@pytest.mark.parametrize("stale_field,stale_value", [
    ("provider_voice_id", "onwK4e9ZLuTAKqWW03F9"),
    ("voice_ref_path", "models/TTS/refs/indextts2/somebody_else.wav"),
    ("ref_path", "models/TTS/refs/indextts2/somebody_else.wav"),
])
def test_stale_engine_specific_identity_is_cleared(pin_provisional, stale_field,
                                                   stale_value, policy_mode):
    """Clearing `voice_route` alone is NOT enough. The dispatch PREFERS an
    existing engine-specific field over resolving from the id stamped a moment
    ago, so a survivor is not stale metadata -- it is the file or the cloud voice
    that actually renders.

    PARAMETRIZED OVER BOTH MODES because an earlier cut of this test was not, and
    the gap was an implementation gap too: preserve_ledger cleared before a
    provisional stamp and not before a qualified one. A test pinned to one mode
    says nothing about the other, and the canonical graph runs only one of them.
    """
    cast = json.loads(json.dumps(CAST))
    cast[1][stale_field] = stale_value
    pin_provisional({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")})
    out = CastLock().lock(script_json=_ledger(cast=cast), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy=policy_mode)[0]
    assert stale_field not in _rows(out)["c02"]


@pytest.mark.parametrize("policy_mode", ["auto_registry", "preserve_ledger"])
def test_a_provisional_row_re_locked_QUALIFIED_drops_its_old_identity(
        policy_mode, monkeypatch):
    """THE OTHER DIRECTION, which the sprint contract names explicitly and an
    earlier cut proved in only one branch: provisional -> qualified.

    Uses the REAL shipped record, RE-QUALIFIED onto the current runtime
    (voice identity 2026-08-18, [QA-7]). The shipped record is deliberately not
    selectable any more: this fix changed IndexTTS2's seed handling and its
    emotion blend, so `select_policy_route` demotes a route whose stored
    adapter/worker fingerprint no longer matches the code that would render it,
    and Lemmy stays unqualified until somebody re-auditions him.

    THE TRANSITION ITSELF STILL HAS TO WORK, and that is what this proves --
    a re-qualified route must promote a provisional row and strip its old
    identity. Copying the real record rather than inventing one keeps the
    original intent: everything except the runtime line is the shipped
    evidence, so a rotted receipt still fails here.

    RE-QUALIFIED 2026-08-18 (`prod-audition-2026-08-18`), so the shipped record
    now matches the live fingerprint and the assignment below writes the value
    that is already there. That is deliberate: this test proves the PROMOTION,
    and it must not start passing or failing for the unrelated reason that the
    shipped route happens to be qualified this week. The gate itself is
    asserted in `tests/test_voice_identity_fix.py` --
    `test_the_shipped_lemmy_route_is_selected_again` for the qualified case and
    `test_a_stale_record_is_not_selected` for the demotion.
    """
    from nodes import _otr_voice_route as ROUTE

    ROUTE._LIVE_FINGERPRINT_CACHE.clear()
    requalified = json.loads(json.dumps(
        POOLS.LEMMY_VOICE_POLICY["approved_native_routes"]["indextts2"]))
    requalified["qualification_record"]["runtime"]["engine_impl_version"] = (
        ROUTE.live_engine_impl_version("indextts2"))
    monkeypatch.setattr(
        "nodes.cast_lock._lemmy_voice_policy",
        lambda: dict(POOLS.LEMMY_VOICE_POLICY,
                     approved_native_routes={"indextts2": requalified}))

    cast = json.loads(json.dumps(CAST))
    cast[1].update({
        "voice_ref_id": PROVIDER_REF,
        "voice_engine": "elevenlabs",
        "provider_voice_id": PROVIDER_VOICE_ID,
        ROUTE.CAST_ROW_TIER_FIELD: ROUTE.ROUTE_TIER_PROVISIONAL,
        ROUTE.CAST_ROW_ROUTE_ID_FIELD: "lemmy-elevenlabs-daniel-provisional-v1",
    })
    out = CastLock().lock(script_json=_ledger(cast=cast), voice_bank="default",
                          char_voice_engine="indextts2",
                          cast_voice_policy=policy_mode)[0]
    lemmy = _rows(out)["c02"]
    assert lemmy[ROUTE.CAST_ROW_TIER_FIELD] == ROUTE.ROUTE_TIER_QUALIFIED
    assert lemmy["voice_engine"] == "indextts2"
    assert lemmy["voice_route"]["status"] == "qualified"
    assert "provider_voice_id" not in lemmy, (
        "an ElevenLabs voice id survived onto a row now claiming indextts2")


def test_a_row_the_caster_never_reaches_is_left_EXACTLY_as_it_arrived(pin_provisional):
    """CLEARING AND STAMPING ARE ONE OPERATION, and this is the case that proves
    why. `bark_legacy` is an ordinary dropdown choice that resolves NO character
    engine, so the caster reaches no row. An earlier cut cleared the claimed row
    up front and then fell through that door, leaving Lemmy carrying a stale
    ElevenLabs identity with `lemmy_route_tier="unrouted"` on top -- a field that
    says the ordinary draw chose this voice when nothing was drawn at all. The
    credits roll prefers `voice_engine`/`voice_ref_id` over `voice_preset`, so a
    Bark episode would have credited him to a voice nobody heard.
    """
    cast = json.loads(json.dumps(CAST))
    cast[1].update({
        "voice_ref_id": PROVIDER_REF,
        "voice_engine": "elevenlabs",
        "provider_voice_id": PROVIDER_VOICE_ID,
        ROUTE.CAST_ROW_TIER_FIELD: ROUTE.ROUTE_TIER_PROVISIONAL,
        ROUTE.CAST_ROW_ROUTE_ID_FIELD: "lemmy-elevenlabs-daniel-provisional-v1",
    })
    before = json.loads(json.dumps(cast[1]))
    pin_provisional({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")})
    out = CastLock().lock(script_json=_ledger(cast=cast), voice_bank="bark_legacy",
                          cast_voice_policy="auto_registry")[0]
    after = _rows(out)["c02"]
    for field, value in before.items():
        assert after.get(field) == value, (
            "%s changed on a row the caster never re-cast" % field)


def test_the_writer_stage_bark_preset_SURVIVES_the_normalizer(pin_provisional):
    """`voice_preset` is bark's identity, written far upstream by `lemmy_row()`,
    and this module does not own it. The 2026-08-16 acceptance leg proved the
    two-stage behaviour that depends on it surviving."""
    pin_provisional({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")})
    out = CastLock().lock(script_json=_ledger(), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy="auto_registry")[0]
    assert _rows(out)["c02"]["voice_preset"] == "v2/en_speaker_8"


# ---------------------------------------------------------------------------
# 8. CASTLOCK -- preserve_ledger. The canonical graph runs auto_registry, so a
#    stamp wired only there is invisible in half of production.
# ---------------------------------------------------------------------------
def test_preserve_ledger_reaches_the_provisional_tier(pin_provisional):
    pin_provisional({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")})
    out = CastLock().lock(script_json=_ledger(), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy="preserve_ledger")[0]
    lemmy = _rows(out)["c02"]
    assert lemmy["voice_ref_id"] == CLONE_REF
    assert lemmy[ROUTE.CAST_ROW_TIER_FIELD] == ROUTE.ROUTE_TIER_PROVISIONAL
    assert "voice_route" not in lemmy


def test_preserve_ledger_leaves_an_unclaimed_row_completely_alone(pin_provisional):
    """This mode's whole contract is that a preserved row keeps the bytes it
    arrived with -- so a row nothing claims is not touched, not even to write a
    tier field saying so. auto_registry differs deliberately: it draws, so it
    reports."""
    pin_provisional({"dia": _provisional("dia", "dia_peter_yearsley", "local_wav")})
    out = CastLock().lock(script_json=_ledger(), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy="preserve_ledger")[0]
    lemmy = _rows(out)["c02"]
    assert ROUTE.CAST_ROW_TIER_FIELD not in lemmy
    assert "voice_ref_id" not in lemmy


# ---------------------------------------------------------------------------
# 9. CACHE INVALIDATION. A provisional row has no `voice_route`, so the
#    route-shaped fingerprint cannot see it.
# ---------------------------------------------------------------------------
def _voice_node():
    from nodes.batch_character_voices import BatchCharacterVoices
    return BatchCharacterVoices


def _ledger_with_tier(voice_ref_id, engine, tier=ROUTE.ROUTE_TIER_PROVISIONAL):
    cast = json.loads(json.dumps(CAST))
    cast[1].update({
        "voice_ref_id": voice_ref_id,
        "voice_engine": engine,
        ROUTE.CAST_ROW_TIER_FIELD: tier,
        ROUTE.CAST_ROW_ROUTE_ID_FIELD: "lemmy-%s-provisional-v1" % engine,
        ROUTE.CAST_ROW_REASON_FIELD: "",
    })
    return _ledger(cast=cast)


def test_a_ledger_with_no_routes_still_fingerprints_to_static():
    """Every shipping local render keeps its exact in-graph caching behaviour --
    not "something stable", the same literal string as before."""
    assert _voice_node().IS_CHANGED(
        script_json=_ledger(), engine="chatterbox") == "static"


def test_a_provisional_row_is_NOT_static():
    """Swap the reference WAV under an unchanged id and "static" would serve the
    previous render's audio while the ledger named the new identity."""
    answer = _voice_node().IS_CHANGED(
        script_json=_ledger_with_tier(CLONE_REF, "chatterbox"), engine="chatterbox")
    assert answer != "static"
    assert isinstance(answer, str) and len(answer) == 64


def test_kokoros_pt_voice_is_fingerprinted_too():
    """The `bank_voice_id` kind is the one identity neither the qualified route
    nor the clone rows cover, and hashing only WAVs would leave it cacheable."""
    answer = _voice_node().IS_CHANGED(
        script_json=_ledger_with_tier(KOKORO_REF, "kokoro"), engine="kokoro")
    assert answer != "static"


def test_a_different_identity_gives_a_different_fingerprint():
    a = _voice_node().IS_CHANGED(
        script_json=_ledger_with_tier(CLONE_REF, "chatterbox"), engine="chatterbox")
    b = _voice_node().IS_CHANGED(
        script_json=_ledger_with_tier("cb_bill_boerst", "chatterbox"),
        engine="chatterbox")
    assert a != b


def test_an_unreadable_local_identity_fails_OPEN():
    """The Bug Bible unavailable-input rule: rerun and let the render path fail
    loudly, rather than quietly reusing audio."""
    answer = _voice_node().IS_CHANGED(
        script_json=_ledger_with_tier("cb_nobody_by_that_name", "chatterbox"),
        engine="chatterbox")
    assert isinstance(answer, float) and math.isnan(answer)


def test_a_cloud_identity_never_touches_the_network():
    """A provider voice contributes its id and nothing else. No fetch, and no
    cloud URI treated as a file to hash."""
    from nodes._otr_voice_node_common import _provisional_identity_fingerprint
    answer = _provisional_identity_fingerprint("elevenlabs", PROVIDER_REF)
    assert answer == "provider:%s" % PROVIDER_VOICE_ID


# ---------------------------------------------------------------------------
# 9B. THE RECEIPTS MUST AGREE WITH THE ARTIFACTS, not merely look plausible.
# ---------------------------------------------------------------------------
def _output_root(relative_probe):
    """The output root that actually CONTAINS ``relative_probe``, or None.

    Receipt paths are written `otr/episodes/...`, relative to the output root --
    the convention the qualified route already uses. There is more than one
    plausible root on a box with a separate ComfyUI install, and picking the
    first one that merely EXISTS is how this test came to skip silently while the
    artifacts sat on disk a directory away. So it probes for the file, not for
    the folder: a test that cannot find what it is checking must say nothing
    rather than pass."""
    candidates = []
    try:
        import folder_paths                              # ComfyUI runtime
        candidates.append(folder_paths.get_output_directory())
    except Exception:                                    # noqa: BLE001
        pass
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(REPO_ROOT))), "ComfyUI", "output"))
    for base in candidates:
        if base and os.path.isfile(os.path.join(base, relative_probe)):
            return base
    return None


def _rendered_routes():
    return [(engine, record)
            for engine, record in
            POOLS.LEMMY_VOICE_POLICY["provisional_native_routes"].items()
            if record["provisional_receipt"]["state"] == "rendered_pending_listen"]


def test_a_rendered_receipt_names_artifacts_that_exist_and_still_match():
    """TRANSCRIBING MACHINE FACTS BY HAND IS HOW A RECEIPT STARTS LYING. The
    three local rows were written from a manifest by a human reading numbers off
    a terminal; this checks the numbers against the bytes.

    SKIPS on a tree with no renders -- a fresh clone has no artifacts and that is
    not a failure. FAILS the moment they exist and disagree, which is the only
    state anyone could be misled by.
    """
    rendered = _rendered_routes()
    assert rendered, "no rendered provisional route to check"
    probe = rendered[0][1]["provisional_receipt"]["audition_manifest_path"]
    root = _output_root(probe)
    if root is None:
        pytest.skip("the audition artifacts are not on this box")
    checked = 0
    for engine, record in rendered:
        receipt = record["provisional_receipt"]
        manifest_path = os.path.join(root, receipt["audition_manifest_path"])
        with open(manifest_path, "rb") as fh:
            blob = fh.read()
        assert hashlib.sha256(blob).hexdigest() == \
            receipt["audition_manifest_sha256"], \
            "%s cites a manifest hash that no longer matches the file" % engine

        manifest = json.loads(blob.decode("utf-8"))
        row = (manifest.get("engines") or {}).get(engine) or {}
        assert row.get("identity_id") == receipt["identity_id"], engine
        for kind in ("neutral", "emotional"):
            path = os.path.join(root, receipt["%s_clip_path" % kind])
            assert os.path.isfile(path), "%s: %s is missing" % (engine, path)
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            assert digest == receipt["%s_clip_sha256" % kind], (
                "%s %s clip has been re-rendered or replaced since its receipt "
                "was written" % (engine, kind))
            assert digest == (row.get("clips") or {}).get(kind, {}).get("sha256"), (
                "%s %s: the receipt and the manifest disagree" % (engine, kind))
            checked += 1
    assert checked == 6, "expected two clips for each of three rendered routes"


def test_every_rendered_route_cites_the_SAME_manifest():
    """One audition, one manifest. Two manifests would mean two batches
    presented as one comparison, and the whole point of the frozen lines is that
    the arms differ only in the engine."""
    digests = {record["provisional_receipt"]["audition_manifest_sha256"]
               for _engine, record in _rendered_routes()}
    assert len(digests) == 1, digests


# ---------------------------------------------------------------------------
# 10. ROUTE -> CASTLOCK -> DISPATCH, for all three identity families.
#
# WHY THESE EXIST WHEN THERE IS ALSO A LIVE LEG. One canonical leg pins ONE
# engine, because the graph pins one character engine -- and the three identity
# kinds reach the adapter through three different fields. A leg on kokoro proves
# nothing about whether a provider route delivers its voice id or a clone route
# resolves the right WAV. These cover the interfaces the leg cannot; the leg
# covers the graph these cannot.
#
# They resolve the identity the SAME WAY the dispatch does, by importing the real
# helpers rather than restating the logic -- a test that reimplements the code it
# is testing agrees with itself and nothing else.
# ---------------------------------------------------------------------------
def _dispatch_identity(engine, cast_row, episode_seed=42):
    """What the adapter would actually be handed for this row.

    Mirrors `_render_per_line`: the ref FIELD comes from the adapter, a wav-path
    engine falls back to the shared clone resolver when the row carries no path,
    and nothing else is invented.
    """
    from nodes._otr_audio_engines import get_engine
    from nodes._otr_voice_node_common import (
        _engine_requires_voice_ref, _resolve_clone_ref_path,
    )

    adapter = get_engine(engine)
    ref_field = getattr(adapter, "voice_ref_field", "voice_ref_path")
    if ref_field == "voice_ref_path":
        voice_ref = cast_row.get("voice_ref_path") or cast_row.get("ref_path")
    else:
        voice_ref = cast_row.get(ref_field)
    if _engine_requires_voice_ref(adapter) and not voice_ref:
        voice_ref = _resolve_clone_ref_path(engine, cast_row, episode_seed)
    return ref_field, voice_ref


def _locked_lemmy(pin_provisional, engine, ref, kind, voice_bank):
    state = ("configured_unrendered" if kind == "provider_voice"
             else "rendered_pending_listen")
    pin_provisional({engine: _provisional(engine, ref, kind, state=state)})
    out = CastLock().lock(script_json=_ledger(), voice_bank=voice_bank,
                          char_voice_engine=engine,
                          cast_voice_policy="auto_registry")[0]
    return _rows(out)["c02"]


def test_family_clone_wav_reaches_the_selected_bank_wav(pin_provisional):
    row = _locked_lemmy(pin_provisional, "chatterbox", CLONE_REF, "local_wav",
                        "default_clean")
    ref_field, voice_ref = _dispatch_identity("chatterbox", row)
    assert ref_field == "voice_ref_path"
    assert voice_ref and os.path.isfile(voice_ref), voice_ref
    # ...and it is the file the ROUTE names, not some other male reference the
    # gender fallback would have been happy with.
    entry = next(e for e in load_voice_bank()[0]
                 if e.voice_ref_id == CLONE_REF and e.engine == "chatterbox")
    assert os.path.basename(voice_ref) == os.path.basename(entry.ref_path)


def test_family_bank_voice_id_reaches_voice_ref_id(pin_provisional):
    row = _locked_lemmy(pin_provisional, "kokoro", KOKORO_REF, "bank_voice_id",
                        "kokoro_builtin")
    ref_field, voice_ref = _dispatch_identity("kokoro", row)
    assert ref_field == "voice_ref_id"
    assert voice_ref == KOKORO_REF


def test_family_provider_voice_reaches_provider_voice_id(pin_provisional):
    row = _locked_lemmy(pin_provisional, "elevenlabs", PROVIDER_REF,
                        "provider_voice", "elevenlabs_cloud")
    ref_field, voice_ref = _dispatch_identity("elevenlabs", row)
    assert ref_field == "provider_voice_id"
    assert voice_ref == PROVIDER_VOICE_ID
    assert row["provider_voice_id"] == PROVIDER_VOICE_ID


@pytest.mark.parametrize("engine,ref,kind,voice_bank", [
    ("chatterbox", CLONE_REF, "local_wav", "default_clean"),
    ("kokoro", KOKORO_REF, "bank_voice_id", "kokoro_builtin"),
    ("elevenlabs", PROVIDER_REF, "provider_voice", "elevenlabs_cloud"),
])
def test_no_family_stamps_a_route_and_none_of_them_RAISES(
        pin_provisional, engine, ref, kind, voice_bank):
    """THE RENDER-KILLER, proved at the real call site rather than by inspection.

    `resolve_and_verify_reference` is what the voice node calls for every cast
    row before it builds a request, and it RAISES on any non-empty voice_route
    whose status is not exactly `qualified`. Running it here on each family's
    locked row is the difference between believing the tier is safe and knowing
    it: a provisional row must resolve to LEGACY_REFERENCE and must not raise.
    """
    row = _locked_lemmy(pin_provisional, engine, ref, kind, voice_bank)
    assert "voice_route" not in row
    resolved = ROUTE.resolve_and_verify_reference(row, engine)
    assert resolved is ROUTE.LEGACY_REFERENCE
    assert resolved.is_policy_route is False


# ---------------------------------------------------------------------------
# 11. LEMMY STAYS IN HIS BOX (operator 2026-08-17). Measured on a published
#     episode: MOE GORDON -- not Lemmy -- drew `idx_lemmy_algenib_cockney_v1`,
#     the reference a blinded audition qualified as LEMMY'S voice.
# ---------------------------------------------------------------------------
def test_lemmys_own_clone_is_never_drawn_for_another_character(bank):
    """A voice can only mean "this is Lemmy" if it is the only place it is heard.

    Reserved rows are pre-filtered out of the pool exactly like audited rejects
    -- a deterministic filter, never a score reweight.
    """
    from nodes._otr_voice_bank import assign_voice_for_slot, reserved_voice_ref_ids

    reserved = reserved_voice_ref_ids()
    assert reserved, "nothing is reserved -- the pin is vacuous"
    for engine in ("indextts2", "chatterbox", "dia"):
        for seed in range(40):
            ref = assign_voice_for_slot(
                role="char_voice", engine=engine, char_id="c03", gender="male",
                timbre=(), age_band="adult", episode_seed=seed,
                casting_policy_version="v1", allow_voice_reuse=True,
                used_voice_ref_ids=set(), bank=bank)
            assert ref.voice_ref_id not in reserved, (engine, seed, ref.voice_ref_id)


def test_only_his_OWN_recording_is_reserved_not_the_voices_he_borrows():
    """THE OVER-BROAD VERSION OF THIS FIX WOULD HAVE CAUSED ITS OWN REGRESSION.

    A first cut reserved every voice any Lemmy route named, which swept in
    `bm_george` -- a shared kokoro catalogue voice tagged `preferred_announcer`
    -- plus the two shared cloud voices. He is provisionally POINTED AT those; he
    does not own them, and pulling the preferred announcer from the pool to
    protect a cameo trades one defect for another. Only `local_wav` routes, which
    are clones of the operator-approved recording of Lemmy himself, are his.
    """
    from nodes._otr_voice_bank import reserved_voice_ref_ids

    reserved = reserved_voice_ref_ids()
    assert "idx_lemmy_algenib_cockney_v1" in reserved
    assert CLONE_REF not in reserved, "a generic mirrored row must stay castable"
    for borrowed in ("bm_george", "el_daniel", "gt_algenib"):
        assert borrowed not in reserved, (
            "%s is a shared catalogue voice, not Lemmy's identity" % borrowed)


def test_reserving_his_voice_cannot_starve_lemmy_of_it(pin_provisional, bank):
    """The policy path stamps its bank entry DIRECTLY and never comes through the
    selector, so the reservation removes his voice from everyone else's pool
    without removing it from his own."""
    pin_provisional({"chatterbox": _provisional(
        "chatterbox", "cb_lemmy_algenib_cockney_v1", "local_wav")})
    out = CastLock().lock(script_json=_ledger(), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy="auto_registry")[0]
    lemmy = _rows(out)["c02"]
    assert lemmy["voice_ref_id"] == "cb_lemmy_algenib_cockney_v1"
    assert lemmy[ROUTE.CAST_ROW_TIER_FIELD] == ROUTE.ROUTE_TIER_PROVISIONAL
