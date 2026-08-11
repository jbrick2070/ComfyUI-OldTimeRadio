"""Plan 5.2 -- the explicit, qualified CastLock re-pin.

WHAT IS BEING PROTECTED HERE. Lemmy was never redrawn per episode: 33 of the 35
reference-carrying LEMMY ledger rows name the SAME reference, because every one
of them had ``meta.episode_seed=None`` and the generic selector therefore derived
an identical seed. He was ACCIDENTALLY PINNED. So the repair is a narrow,
provable claim on ONE named row -- and the most important assertions in this file
are the NEGATIVE ones: that the generic selector, the seed, and every unclaimed
row behave exactly as they did before this path existed.

The shipped policy has an EMPTY ``approved_native_routes``, so the mechanism is
inert in production until an operator audition fills it in. These tests inject a
policy to exercise the machinery that day will switch on.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from nodes import _otr_voice_route as ROUTE
from nodes.cast_lock import CastLock

# A real indextts2 entry, so the bank-agreement check has something true to find.
PINNED_REF = "vz_peter_yearsley"
OTHER_REF = "vz_bill_boerst"

CAST = [
    {"char_id": "c01", "name": "MONTY", "gender": "male",
     "voice_preset": "v2/en_speaker_1"},
    # Lemmy sits on a POSITIONAL char_id, which is the whole reason the policy
    # matches on name as well. A matcher keyed to `char_id == "lemmy"` claims
    # nobody here -- and would have silently done nothing forever.
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


def _receipt(tmp_path, *, engine="indextts2", voice_ref_id=PINNED_REF,
             status="qualified", technical_verdict="pass",
             rights_status="approved"):
    """A COMPLETE, honest route record -- real bytes, real hash, real bank id."""
    wav = tmp_path / "lemmy_algenib_cockney_v1.wav"
    wav.write_bytes(b"RIFF....WAVEfmt not-really-audio-but-really-bytes")
    digest = hashlib.sha256(wav.read_bytes()).hexdigest()
    return {
        "route_id": "lemmy-indextts2-algenib-v1",
        "route_contract_version": 1,
        "qualification_record": {
            "record_id": "qr-lemmy-0001",
            "status": status,
            "technical_verdict": technical_verdict,
            "engine": engine,
            "voice_ref_id": voice_ref_id,
            "rights": {
                "status": rights_status,
                "source": "self-generated Google TTS (Algenib)",
                "terms_snapshot_ref": "docs/2026-08-10-G0-RIGHTS-DECISION-CARD-lemmy.md",
                "terms_snapshot_date": "2026-08-10",
                "scope": "clone reference for local engines",
                "decided_at": "2026-08-10T20:37:17Z",
                "revoked_at": None,
                "expires_at": None,
            },
            "runtime": {
                "model_id": "IndexTTS-2",
                "engine_impl_version": "1.0.0",
                "weight_revision": "abc123",
            },
            "reference": {
                "kind": "local_wav",
                "absolute_path": str(wav),
                "source_ref_sha256": digest,
            },
            "audition_manifest": {
                "path": "otr/auditions/lemmy-g1/manifest.json",
                "sha256": "b" * 64,
            },
        },
    }


def _policy(record, *, engine="indextts2", character_key="lemmy"):
    return {
        "policy_version": "lemmy-cockney-v1-test",
        "character_key": character_key,
        "approved_native_routes": {engine: record} if record else {},
    }


@pytest.fixture()
def pin(monkeypatch, tmp_path):
    """Install a proving policy and hand the test its record for mutation."""
    def install(record=..., **kw):
        rec = _receipt(tmp_path) if record is ... else record
        monkeypatch.setattr("nodes.cast_lock._lemmy_voice_policy",
                            lambda: _policy(rec, **kw))
        return rec
    return install


# ---------------------------------------------------------------------------
# The shipped state: DORMANT. Nothing below may change any current render.
# ---------------------------------------------------------------------------
def test_the_shipped_policy_now_carries_a_REAL_qualified_route():
    """G1 Test A PASSED on 2026-08-10, so this dict is no longer empty.

    It REPLACES `test_the_shipped_policy_approves_nothing_so_the_repin_is_inert`,
    which pinned the pre-audition state and was correct for exactly as long as
    nothing had been auditioned. What must never happen is a route appearing
    WITHOUT the evidence, so the assertions below are about the evidence, not
    about the route's existence.
    """
    from config.cast_pools import LEMMY_VOICE_POLICY as P

    assert P["character_key"] == "lemmy"
    route = P["approved_native_routes"]["indextts2"]
    qual = route["qualification_record"]
    assert qual["status"] == "qualified"
    assert qual["technical_verdict"] == "pass"
    assert qual["rights"]["status"] == "approved"
    assert qual["voice_ref_id"] == "idx_lemmy_algenib_cockney_v1"
    # A human said yes, and said what they heard. Nothing else can supply this.
    verdict = route["qualification_receipt"]["operator_verdict"]
    assert verdict.startswith("PASS"), verdict
    assert "blinded" in verdict.lower()
    # The runtime identity is derived from real artifacts, never a made-up
    # version string -- both are 16 hex chars from a sha256.
    for field in ("engine_impl_version", "weight_revision"):
        value = qual["runtime"][field]
        assert re.fullmatch(r"[0-9a-f]{16}", value), (field, value)


def test_the_live_route_actually_validates_against_the_real_bank_and_disk():
    """The end-to-end proof: the shipped receipt survives the real validator,
    against the real bank, with the reference bytes re-hashed off disk."""
    from datetime import datetime, timezone
    from config.cast_pools import LEMMY_VOICE_POLICY as P
    from nodes._otr_voice_bank import load_voice_bank
    from nodes._otr_voice_node_common import _resolve_ref_to_disk

    claim = ROUTE.resolve_policy_route_claim(
        P, "indextts2", datetime.now(timezone.utc),
        bank_entries=load_voice_bank()[0],
        path_resolver=_resolve_ref_to_disk)
    assert claim is not None
    assert claim.voice_ref_id == "idx_lemmy_algenib_cockney_v1"
    assert claim.voice_route["route_id"] == "lemmy-indextts2-algenib-cockney-v1"


def test_a_bank_with_no_character_engine_is_reported_not_raised():
    """`bark_legacy` is a PRESET bank with no reference entries, so no engine
    resolves and the indextts2 route simply does not apply. Raising here would
    break a legitimate bank choice because an unrelated engine is qualified --
    which it did, the first time the route went live."""
    out = CastLock().lock(script_json=_ledger(), voice_bank="bark_legacy",
                          cast_voice_policy="auto_registry")
    led = json.loads(out[0])
    for entry in led["cast"]:
        assert "voice_route" not in entry


def test_a_bank_that_COULD_serve_the_route_but_resolves_no_engine_fails_closed(
        monkeypatch, pin):
    """The other half of the bark_legacy carve-out, and the reason it is not a
    hole. `bark_legacy` is fine because it carries no indextts2 entries at all.
    A bank that DOES carry them, yet resolves no engine, is the resolver
    declining for some other reason -- and a qualified route silently not
    applying is the exact failure this path exists to end.

    Two independent reviews flagged that the first cut could not tell these
    apart. Neither could construct a live trigger, so this pins the distinction
    rather than fixing a reachable bug."""
    pin()
    monkeypatch.setattr(CastLock, "_resolve_char_engine",
                        staticmethod(lambda *a, **k: None))
    with pytest.raises(ROUTE.VoiceRouteError, match="no character voice engine"):
        CastLock().lock(script_json=_ledger(), voice_bank="default",
                        cast_voice_policy="auto_registry")


def test_a_dormant_policy_still_costs_nothing(monkeypatch):
    """The empty-routes path must not pay for a bank load, and must not raise
    when the active engine is unresolved. Still reachable: any policy whose
    approved_native_routes is empty."""
    def explode(*a, **k):                       # pragma: no cover -- must not run
        raise AssertionError("dormant policy loaded the voice bank")

    monkeypatch.setattr("nodes._otr_voice_bank.load_voice_bank", explode)
    monkeypatch.setattr("nodes.cast_lock._lemmy_voice_policy",
                        lambda: _policy(None))
    assert CastLock()._resolve_policy_claim("default", None) is None


def test_auto_registry_is_deterministic_with_the_live_route():
    a = CastLock().lock(script_json=_ledger(), cast_voice_policy="auto_registry")[0]
    b = CastLock().lock(script_json=_ledger(), cast_voice_policy="auto_registry")[0]
    assert a == b
    # Lemmy is on the route; nobody else is.
    rows = {e["char_id"]: e for e in json.loads(a)["cast"]}
    assert rows["c02"]["voice_ref_id"] == "idx_lemmy_algenib_cockney_v1"
    assert "voice_route" not in rows["c01"]
    assert "voice_route" not in rows["a1"]


# ---------------------------------------------------------------------------
# Row matching (plan step 3) -- name OR char_id, never positional assumption.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("entry,expected", [
    ({"char_id": "c02", "name": "LEMMY"}, True),
    ({"char_id": "c02", "name": "  lemmy  "}, True),
    ({"char_id": "LEMMY", "name": "Cockney Bloke"}, True),
    ({"char_id": "c02", "name": "MONTY"}, False),
    ({"char_id": "c02"}, False),
    ({}, False),
    ("not a dict", False),
])
def test_cast_row_matches_policy(entry, expected):
    assert ROUTE.cast_row_matches_policy(entry, "lemmy") is expected


def test_an_empty_character_key_claims_nobody():
    assert ROUTE.cast_row_matches_policy({"name": "LEMMY"}, "") is False


# ---------------------------------------------------------------------------
# Selection (plan step 4) -- the three outcomes, and the difference matters.
# ---------------------------------------------------------------------------
def test_routes_for_another_engine_select_nothing_and_do_not_raise(tmp_path):
    """Qualifying Lemmy on IndexTTS2 must never break a bark render."""
    policy = _policy(_receipt(tmp_path), engine="indextts2")
    assert ROUTE.select_policy_route(policy, "bark") is None


def test_an_unresolved_active_engine_with_real_routes_is_LOUD(tmp_path):
    """Silently skipping a qualified route is the floor-evidence failure this
    whole module exists to end -- so it raises rather than shrugs."""
    policy = _policy(_receipt(tmp_path))
    with pytest.raises(ROUTE.VoiceRouteError, match="could not be resolved"):
        ROUTE.select_policy_route(policy, None)


def test_a_dormant_policy_with_no_engine_is_silent():
    assert ROUTE.select_policy_route(_policy(None), None) is None


# ---------------------------------------------------------------------------
# Proving the claim (plan step 4) -- and refusing to fall back (step 6).
# ---------------------------------------------------------------------------
def _claim(record, bank=None):
    from datetime import datetime, timezone
    from nodes._otr_voice_bank import load_voice_bank

    entries = bank if bank is not None else load_voice_bank()[0]
    return ROUTE.resolve_policy_route_claim(
        _policy(record), "indextts2", datetime.now(timezone.utc),
        bank_entries=entries)


def test_a_complete_receipt_proves_and_yields_the_bank_entry(tmp_path):
    claim = _claim(_receipt(tmp_path))
    assert claim.character_key == "lemmy"
    assert claim.engine == "indextts2"
    assert claim.voice_ref_id == PINNED_REF
    assert claim.bank_entry.voice_ref_id == PINNED_REF
    assert claim.bank_entry.engine == "indextts2"
    assert claim.voice_route["route_id"] == "lemmy-indextts2-algenib-v1"
    assert claim.voice_route["reference_kind"] == "local_wav"
    assert claim.voice_route["qualification_record_id"] == "qr-lemmy-0001"
    assert claim.voice_route["runtime"]["model_id"] == "IndexTTS-2"


@pytest.mark.parametrize("kw,fragment", [
    ({"status": "candidate"}, "status"),
    ({"technical_verdict": "not_run"}, "technical_verdict"),
    ({"rights_status": "pending"}, "rights.status"),
    ({"voice_ref_id": "no_such_reference_anywhere"}, "not present in the voice bank"),
])
def test_an_unproved_route_raises_and_never_falls_back(tmp_path, kw, fragment):
    with pytest.raises(ROUTE.VoiceRouteError) as exc:
        _claim(_receipt(tmp_path, **kw))
    assert fragment in str(exc.value)
    assert "no fallback" in str(exc.value)


def test_swapped_reference_bytes_are_caught(tmp_path):
    record = _receipt(tmp_path)
    ref = record["qualification_record"]["reference"]
    # Someone replaced the WAV under the verdict.
    with open(ref["absolute_path"], "wb") as fh:
        fh.write(b"a completely different recording")
    with pytest.raises(ROUTE.VoiceRouteError, match="BYTES DO NOT MATCH"):
        _claim(record)


def test_a_missing_reference_file_is_caught(tmp_path):
    record = _receipt(tmp_path)
    ref = record["qualification_record"]["reference"]
    import os
    os.remove(ref["absolute_path"])
    with pytest.raises(ROUTE.VoiceRouteError, match="does not exist"):
        _claim(record)


def test_engine_disagreement_between_route_and_active_scalar(tmp_path):
    from datetime import datetime, timezone
    from nodes._otr_voice_bank import load_voice_bank

    # The policy approves chatterbox; chatterbox is what is selected; but the
    # receipt inside it claims indextts2. A policy must never approve one engine
    # while another renders.
    record = _receipt(tmp_path, engine="indextts2")
    policy = _policy(record, engine="chatterbox")
    with pytest.raises(ROUTE.VoiceRouteError, match="ENGINE DISAGREEMENT"):
        ROUTE.resolve_policy_route_claim(
            policy, "chatterbox", datetime.now(timezone.utc),
            bank_entries=load_voice_bank()[0])


def test_an_ambiguous_bank_id_is_a_reject_not_a_coin_flip(tmp_path):
    from nodes._otr_voice_bank import load_voice_bank

    bank = load_voice_bank()[0]
    dupe = next(e for e in bank if e.voice_ref_id == PINNED_REF
                and e.engine == "indextts2")
    with pytest.raises(ROUTE.VoiceRouteError, match="needs exactly one"):
        _claim(_receipt(tmp_path), bank=tuple(bank) + (dupe,))


def test_a_policy_that_names_no_character_cannot_be_applied(tmp_path):
    from datetime import datetime, timezone
    from nodes._otr_voice_bank import load_voice_bank

    policy = _policy(_receipt(tmp_path), character_key="")
    with pytest.raises(ROUTE.VoiceRouteError, match="claims nobody"):
        ROUTE.resolve_policy_route_claim(
            policy, "indextts2", datetime.now(timezone.utc),
            bank_entries=load_voice_bank()[0])


# ---------------------------------------------------------------------------
# Through the real lock(), both modes, both reuse settings (plan step 5).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("allow_voice_reuse", [True, False])
def test_auto_registry_pins_the_claimed_row_only(pin, allow_voice_reuse):
    """The canonical workflow ships allow_voice_reuse=true, so both are pinned."""
    baseline = json.loads(CastLock().lock(
        script_json=_ledger(), cast_voice_policy="auto_registry",
        allow_voice_reuse=allow_voice_reuse)[0])

    pin()
    led = json.loads(CastLock().lock(
        script_json=_ledger(), cast_voice_policy="auto_registry",
        allow_voice_reuse=allow_voice_reuse)[0])

    rows = {e["char_id"]: e for e in led["cast"]}
    base = {e["char_id"]: e for e in baseline["cast"]}

    # The claimed row is re-pinned, and carries its proof.
    assert rows["c02"]["voice_ref_id"] == PINNED_REF
    assert rows["c02"]["voice_engine"] == "indextts2"
    assert rows["c02"]["voice_cast_fallback"] == "policy_route"
    assert rows["c02"]["voice_route"]["route_id"] == "lemmy-indextts2-algenib-v1"

    # Every other row is untouched, INCLUDING the announcer, and none of them
    # gains a route. This is the assertion that says "the generic selector was
    # not rewritten".
    for cid in ("c01", "a1"):
        assert rows[cid]["voice_ref_id"] == base[cid]["voice_ref_id"], cid
        assert "voice_route" not in rows[cid], cid


def test_preserve_ledger_changes_only_the_claimed_row(pin):
    pin()
    led = json.loads(CastLock().lock(
        script_json=_ledger(), cast_voice_policy="preserve_ledger")[0])
    rows = {e["char_id"]: e for e in led["cast"]}

    assert rows["c02"]["voice_ref_id"] == PINNED_REF
    assert rows["c02"]["voice_route"]["route_id"] == "lemmy-indextts2-algenib-v1"
    # preserve_ledger's contract is byte-safety for everyone else: these rows
    # arrived with no voice_ref_id and must still have none.
    for cid in ("c01", "a1"):
        assert "voice_ref_id" not in rows[cid], cid
        assert "voice_route" not in rows[cid], cid


def test_the_pin_beats_the_hybrid_llm_voice_fit(pin):
    """A qualified route outranks a writer proposal -- step 4 puts it first."""
    pin()
    meta = {
        "episode_seed": 42,
        "voice_cast_decision": {
            "c02": {"accepted_id": OTHER_REF, "engine": "indextts2"},
        },
    }
    led = json.loads(CastLock().lock(
        script_json=_ledger(meta=meta), cast_voice_policy="auto_registry")[0])
    rows = {e["char_id"]: e for e in led["cast"]}
    assert rows["c02"]["voice_ref_id"] == PINNED_REF
    assert "hybrid" not in rows["c02"]["voice_cast_fallback"]


def test_a_genderless_claimed_row_is_still_pinned(pin):
    """The gender gate exists to feed the SCORER. A pinned route did not need
    scoring, so a row with no gender must not fall through unpinned."""
    pin()
    cast = [{"char_id": "c02", "name": "LEMMY", "voice_preset": "v2/en_speaker_8"}]
    led = json.loads(CastLock().lock(
        script_json=_ledger(cast), cast_voice_policy="auto_registry")[0])
    assert led["cast"][0]["voice_ref_id"] == PINNED_REF


def test_a_proved_route_with_no_matching_row_is_reported_not_raised(pin):
    """The route proved itself; that this episode did not cast Lemmy is an
    ordinary fact about the episode."""
    pin()
    cast = [{"char_id": "c01", "name": "MONTY", "gender": "male",
             "voice_preset": "v2/en_speaker_1"}]
    out = CastLock().lock(script_json=_ledger(cast),
                          cast_voice_policy="preserve_ledger")
    assert "no cast row matches" in out[2]
    assert "voice_route" not in out[0]


def test_a_failed_route_stops_the_lock(pin, tmp_path):
    pin(_receipt(tmp_path, status="rejected"))
    with pytest.raises(ROUTE.VoiceRouteError):
        CastLock().lock(script_json=_ledger(), cast_voice_policy="auto_registry")


# ---------------------------------------------------------------------------
# Ordering (plan steps 1 and 6).
# ---------------------------------------------------------------------------
def test_the_revision_is_stamped_before_any_route_is_resolved(monkeypatch, pin):
    """Step 1 in its load-bearing form: by the time route resolution runs, the
    revision that attempted it is already on the ledger meta. A route failure
    with no revision on it is a failure nobody can locate afterwards.

    The spy reads the NODE's own meta dict -- reached through `led`, which the
    node parsed from script_json -- not the test's copy.
    """
    pin()
    seen = {}
    original = CastLock._stamp_voice_engine_selection

    def stamp_spy(self, led, *a, **kw):
        seen["revision"] = (led.get("meta") or {}).get("cast_lock_revision")
        return original(self, led, *a, **kw)

    monkeypatch.setattr(CastLock, "_stamp_voice_engine_selection", stamp_spy)
    out = CastLock().lock(
        script_json=_ledger(meta={"episode_seed": 1, "cast_lock_revision": 7}),
        cast_voice_policy="auto_registry")

    assert seen["revision"] == 8, "engine/route resolution ran before the stamp"
    assert out[1] == 8
    assert json.loads(out[0])["meta"]["cast_lock_revision"] == 8
