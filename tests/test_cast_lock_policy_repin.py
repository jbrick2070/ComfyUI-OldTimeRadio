"""Plan 5.2 -- the explicit, qualified CastLock re-pin.

WHAT IS BEING PROTECTED HERE. Lemmy was never redrawn per episode: 33 of the 35
reference-carrying LEMMY ledger rows name the SAME reference, because every one
of them had ``meta.episode_seed=None`` and the generic selector therefore derived
an identical seed. He was ACCIDENTALLY PINNED. So the repair is a narrow,
provable claim on ONE named row -- and the most important assertions in this file
are the NEGATIVE ones: that the generic selector, the seed, and every unclaimed
row behave exactly as they did before this path existed.

THAT DAY CAME. This file was written while ``approved_native_routes`` was EMPTY
and the mechanism was inert, and most tests below still INJECT a policy so the
machinery can be exercised in isolation. But G1 Test A passed on 2026-08-10, so
the shipped policy now carries a real qualified IndexTTS2 route, and the tests
that matter most are the ones using it UNINJECTED -- they fail if the receipt in
`config/cast_pools.py` stops validating, stops reaching the durable record, or
starts applying to an episode that never cast him.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def test_the_route_survives_the_DURABLE_stamp(monkeypatch):
    """Plan section 7 step 3's last clause: cast data is preserved on durable
    reload. It was the one acceptance item with no test.

    Not a hypothetical: `stamp_durable` replaces the `cast` section WHOLESALE on
    the singleton, so a route stamped only on the local wire dict would reach the
    voice node and then vanish from the persisted ledger -- the receipt would name
    a route the durable record never mentions.

    Production already proved this works (the 2026-08-11 sweep's on-disk ledger
    carries `voice_route` on Lemmy's row), but a live episode is an expensive
    regression detector. This is the cheap one.

    Deliberately does NOT use the `pin` fixture: it exercises the REAL shipped
    policy and the REAL qualified route, so it fails if the receipt in
    `cast_pools.py` stops reaching the durable record.
    """
    captured = {}

    import nodes.production_ledger as pl

    def spy(*, sections=None, meta_updates=None, source=""):
        captured["sections"] = sections
        captured["source"] = source
        return "test-mode"

    monkeypatch.setattr(pl, "stamp_durable", spy)
    monkeypatch.setattr("nodes.cast_lock.stamp_durable", spy, raising=False)

    CastLock().lock(script_json=_ledger(), cast_voice_policy="auto_registry")

    cast = (captured.get("sections") or {}).get("cast")
    assert cast, "the durable stamp received no cast section at all"
    lemmy = [c for c in cast
             if str(c.get("name") or "").upper() == "LEMMY"]
    assert lemmy, "LEMMY is missing from the durably-stamped cast"
    route = lemmy[0].get("voice_route") or {}
    assert route.get("route_id") == "lemmy-indextts2-algenib-cockney-v1", route
    assert route.get("source_ref_sha256"), (
        "the durable route lost its reference hash -- the receipt would name a "
        "route nobody could re-verify")


def test_no_claimed_row_means_no_claim_at_all(pin, monkeypatch):
    """OPERATOR CONTRACT 2026-08-10: a route guard may only speak about a cast
    this episode actually has.

    THIS IS THE FIDELITY RULE, not an optimisation. Lemmy is a recurring-cameo
    character and the source-faithful banks EXCLUDE him outright --
    `_source_bank_excludes_lemmy` in `_otr_casting.py` covers `shakespeare` and
    `public_domain` plus their bake-off variants, and it overrides BOTH the
    entropy roll and the operator's `always include` setting. A Shakespeare
    episode legitimately has no Lemmy row, so nothing about his route may fire
    there -- including the fail-closed raise below, which would otherwise abort
    an episode that never cast him.
    """
    pin()
    # Make the engine unresolvable: without the no-claimed-row check this is
    # exactly the state that raises.
    monkeypatch.setattr(CastLock, "_resolve_char_engine",
                        staticmethod(lambda *a, **k: None))
    cast = [{"char_id": "c01", "name": "MACBETH", "gender": "male",
             "voice_preset": "v2/en_speaker_1"}]
    out = CastLock().lock(script_json=_ledger(cast),
                          cast_voice_policy="auto_registry")
    led = json.loads(out[0])
    assert all("voice_route" not in e for e in led["cast"])


def test_the_exclusion_the_contract_rests_on_is_real_and_covers_both_banks():
    """If this ever narrows, the test above is guarding a rule that no longer
    exists. `always include` must never override source fidelity."""
    from nodes._otr_casting import _source_bank_excludes_lemmy

    for excluded in ("shakespeare", "public_domain",
                     "shakespeare_v2", "public_domain_v3"):
        assert _source_bank_excludes_lemmy(excluded), excluded
    for allowed in ("original", "archive", None, ""):
        assert not _source_bank_excludes_lemmy(allowed), allowed


#: Every SHIPPED source bank, and whether the Lemmy cameo may appear in it.
#: An EXHAUSTIVE map whose keys are asserted equal to the shipped registry --
#: the same shape `ENGINE_COVERAGE` uses in tests/test_slug_provenance.py, and
#: for the same reason.
#:
#: CORRECTED 2026-08-11 FROM A LIVE SWEEP, and the correction is the lesson.
#: The first version of this map was written from the bank list without
#: measuring anything, and it asserted `scifi_news: cameo_allowed`. A six-bank
#: render sweep with the cameo FORCED on every leg proved otherwise: that lane
#: runs the `scifi_news_circuit` pipeline, which never calls `lock_cast()`, so
#: it writes an EMPTY `cast_contract` -- no cast_seed, no lemmy_hit, no
#: lemmy_policy -- and ignores both `lemmy_cameo` and `num_characters` (asked
#: for 2 characters, produced 3).
#:
#: A map that claims a rule nobody measured is the exact defect this file's
#: neighbours exist to catch, so the values below now say what was OBSERVED.
BANK_CAMEO_POLICY = {
    # PROVEN by live render, 2026-08-10/11: forced cameo produced a LEMMY row
    # on the qualified IndexTTS2 route.
    "original": "cameo_allowed",
    "media_archive": "cameo_allowed",
    # PROVEN by live render: forced cameo REFUSED, with
    # lemmy_policy="source_fidelity_exclusion" recorded on the ledger.
    "public_domain": "source_fidelity_excluded",
    "shakespeare": "source_fidelity_excluded",
    # MEASURED, NOT INTENDED-BY-ANYONE-I-CAN-CITE. The circuit lane owns its own
    # cast and never consults the cameo. Whether that is by design or an
    # oversight is an OPERATOR question, recorded rather than guessed at --
    # see docs/2026-08-11-FINDING-lane-cast-contract-divergence.md.
    "scifi_news": "lane_owns_its_cast",
    # NOT MEASURED. Its leg failed in the WRITER before casting ran
    # ("[scifi_fable2] pass 'script' failed after 4 attempt(s): markup ladder
    # exhausted"), so nothing about its cameo behaviour was observed. Marked
    # unmeasured rather than assumed from its sibling -- that assumption is the
    # one that made this map wrong the first time.
    "scifi_news_pro": "unmeasured",
    # Operator-supplied; the shipped map cannot see a bank this repo has never
    # heard of. See the docstring on the completeness test below.
    "custom_source_bank": "cameo_allowed",
}

#: The policies whose claim is "Lemmy may appear here" -- the only ones the
#: `_source_bank_excludes_lemmy` cross-check can speak about.
_CAMEO_ALLOWED_POLICIES = ("cameo_allowed", "lane_owns_its_cast", "unmeasured")


def _shipped_bank_ids():
    import json
    import os

    path = os.path.join(REPO_ROOT, "nodes", "story_packs", "banks.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    banks = data if isinstance(data, list) else data.get("banks", data)
    if isinstance(banks, dict):
        banks = list(banks.values())
    return {str(b.get("source_bank_id") or b.get("id"))
            for b in banks if isinstance(b, dict)}


def test_every_shipped_bank_has_a_DECIDED_cameo_policy():
    """THE MAINTENANCE HAZARD, converted into a build failure.

    The exclusion is a hardcoded frozenset of two names
    (`_LEMMY_EXCLUDED_SOURCE_BANK_IDS`), so a NEW source-faithful bank would
    escape it silently -- nothing would fail, Lemmy would simply start appearing
    in an adaptation. That is not hypothetical bookkeeping: the operator's rule
    is that `always include` must NEVER override source fidelity, and today that
    rule rests entirely on someone remembering to edit a set.

    Asserting the map is EQUAL to the shipped registry means a new bank cannot be
    added without someone deciding, in writing, which side it is on.

    WHAT THIS CANNOT COVER, said plainly: `custom_source_bank` and any user bank
    dropped under `user_packs/source_banks/<id>/` carry ids this repo has never
    seen. No static map can know whether an operator's own bank is a faithful
    adaptation. If a user ships a fidelity bank, its exclusion is their call.
    """
    shipped = _shipped_bank_ids()
    undecided = sorted(shipped - set(BANK_CAMEO_POLICY))
    stale = sorted(set(BANK_CAMEO_POLICY) - shipped)
    assert not undecided, (
        "shipped bank(s) with no cameo decision: %s -- add each to "
        "BANK_CAMEO_POLICY as cameo_allowed or source_fidelity_excluded, and if "
        "it is a faithful adaptation add it to _LEMMY_EXCLUDED_SOURCE_BANK_IDS "
        "too" % undecided)
    assert not stale, "BANK_CAMEO_POLICY names bank(s) no longer shipped: %s" % stale


def test_the_decided_policy_matches_what_the_code_actually_does():
    """The map above is a claim; this is the check that it is true.

    Scoped to what `_source_bank_excludes_lemmy` can actually answer: it decides
    SOURCE FIDELITY, and nothing else. `lane_owns_its_cast` and `unmeasured`
    banks are not excluded BY IT -- they simply never reach the cameo path -- so
    asserting exclusion for them would be testing the wrong function.
    """
    from nodes._otr_casting import _source_bank_excludes_lemmy

    for bank, policy in sorted(BANK_CAMEO_POLICY.items()):
        expected = (policy == "source_fidelity_excluded")
        assert _source_bank_excludes_lemmy(bank) is expected, (bank, policy)
        if policy in _CAMEO_ALLOWED_POLICIES:
            assert not _source_bank_excludes_lemmy(bank), bank


def test_every_policy_value_is_one_we_defined():
    known = set(_CAMEO_ALLOWED_POLICIES) | {"source_fidelity_excluded"}
    for bank, policy in BANK_CAMEO_POLICY.items():
        assert policy in known, (bank, policy)


def test_public_domain_story_is_a_CORPUS_DIRECTORY_not_a_bank_id():
    """Pinned because a review mistook one for the other and filed it as a
    fidelity breach.

    `config/source_banks/public_domain_story/` holds fetched source TEXTS
    (`scripts/otr_fetch_public_domain.py --dest-dir ...`), and every occurrence
    of the string as a `source_bank_id` VALUE lives in
    `docs/multimodal-story-schema/schema-examples/`. The shipped runtime id is
    `public_domain`, which IS excluded. If that ever stops being true -- if a
    bank really is named `public_domain_story` -- this test fails and the
    exclusion set genuinely does need it.
    """
    assert "public_domain_story" not in _shipped_bank_ids()
    assert "public_domain" in _shipped_bank_ids()


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


def test_an_episode_without_the_claimed_row_is_left_entirely_alone(pin):
    """SUPERSEDED BEHAVIOUR, deliberately. This test used to assert that the
    route resolved and THEN reported "no cast row matches".

    Under the operator contract of 2026-08-10 the claim is not resolved at all
    when no row matches, so there is nothing to report. That is the better
    answer: on a source-faithful bank Lemmy is excluded BY DESIGN, and emitting
    "a qualified route did not apply" on every Shakespeare episode would be
    noise dressed as diligence -- it would train the reader to ignore the one
    line that should mean something.
    """
    pin()
    cast = [{"char_id": "c01", "name": "MONTY", "gender": "male",
             "voice_preset": "v2/en_speaker_1"}]
    out = CastLock().lock(script_json=_ledger(cast),
                          cast_voice_policy="preserve_ledger")
    assert "voice_route" not in out[0]
    assert "no cast row matches" not in out[2]
    # preserve_ledger's own contract still holds: nothing was re-cast.
    assert "preserved" in out[2]


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
