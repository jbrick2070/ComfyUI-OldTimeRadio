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
def test_no_shipped_provisional_row_can_pass_as_qualified():
    """THE test. If this ever fails, an unauditioned voice has acquired the
    authority of an audition and the tier has become the bug it prevents."""
    rows = POOLS.LEMMY_VOICE_POLICY["provisional_native_routes"]
    assert rows, "the shipped provisional tier is empty -- did a row get lost?"
    for engine, record in rows.items():
        assert POOLS.is_qualified_route(record) is False, engine


def test_the_qualified_row_is_not_a_provisional_row_either():
    """Exclusion runs both ways: a qualified record carries the two keys the
    provisional shape bans outright, so it can never be read as the weaker tier."""
    qualified = POOLS.LEMMY_VOICE_POLICY["approved_native_routes"]["indextts2"]
    assert POOLS.is_provisional_route(qualified, engine="indextts2") is False


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


@pytest.mark.parametrize("stale_field,stale_value", [
    ("provider_voice_id", "onwK4e9ZLuTAKqWW03F9"),
    ("voice_ref_path", "models/TTS/refs/indextts2/somebody_else.wav"),
    ("ref_path", "models/TTS/refs/indextts2/somebody_else.wav"),
])
def test_stale_engine_specific_identity_is_cleared(pin_provisional, stale_field,
                                                   stale_value):
    """Clearing `voice_route` alone is NOT enough. The dispatch PREFERS an
    existing engine-specific field over resolving from the id stamped a moment
    ago, so a survivor is not stale metadata -- it is the file or the cloud voice
    that actually renders."""
    cast = json.loads(json.dumps(CAST))
    cast[1][stale_field] = stale_value
    pin_provisional({"chatterbox": _provisional("chatterbox", CLONE_REF, "local_wav")})
    out = CastLock().lock(script_json=_ledger(cast=cast), voice_bank="default_clean",
                          char_voice_engine="chatterbox",
                          cast_voice_policy="auto_registry")[0]
    assert stale_field not in _rows(out)["c02"]


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
