"""Machine enforcement for `docs/TTS_VOICE_PREFLIGHT.md`.

The house rule for a preflight sibling (`VIDEO_LANE_PREFLIGHT.md`, "The
family") is that the enforcement code exists BEFORE the doc, never an empty
paper checklist. This file is that code; the doc narrates these gates and is
never a substitute for running them.

Every gate below exists because something real bit during the 2026-08-16
cross-engine Lemmy work. Each test names its gate id.
"""
import json
import os

import pytest

from config import cast_pools as POOLS
from nodes._otr_audio_engines import registry as REGISTRY
from nodes._otr_engine_profiles import legacy_first_engines
from nodes._otr_voice_bank import load_voice_bank

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BANK_JSON = os.path.join(_REPO, "config", "voice_reference_bank.json")

#: Engines whose identity is an engine-native PRESET, not a reference clip.
#: They legitimately carry zero voice-bank rows.
_PRESET_ONLY_ENGINES = {"bark"}


def _bank_rows():
    with open(_BANK_JSON, "r", encoding="utf-8") as handle:
        return json.load(handle)["voices"]


# --- P1: the dispatch surface enumerates every engine, on every path -------


def _tts_voice_nodes():
    from nodes.announcer_voice import AnnouncerVoice
    from nodes.batch_character_voices import BatchCharacterVoices

    return [BatchCharacterVoices, AnnouncerVoice]


@pytest.mark.parametrize("node", _tts_voice_nodes(),
                         ids=lambda n: n.ROLE)
def test_p1_the_degraded_fallback_matches_the_profiles_table(node):
    """P1.1 -- A fallback that is not equal to the thing it stands in for is a
    second, worse answer.

    `build_engine_combo` prefers `legacy_first_engines(role)` and drops to the
    node's `LEGACY_FIRST_FALLBACK` only when the profiles import RAISES -- i.e.
    exactly when nobody is watching. BOTH voice nodes had drifted two engines
    short of the table (elevenlabs and google_tts were appended to the profiles
    and the hardcoded stand-ins were never updated), so a degraded boot offered
    a dropdown that could not even represent a saved graph using either, and
    nothing logged it.

    Order matters as much as membership: index 0 is the byte-identical default
    for the role, so a reordered fallback silently changes what a degraded boot
    renders with.
    """
    assert list(node.LEGACY_FIRST_FALLBACK) == legacy_first_engines(node.ROLE)


def test_p1_2_every_char_voice_engine_is_registered_and_serves_the_role():
    """P1.2 -- A name in the dropdown that no adapter answers to is a queue-time
    crash wearing a menu entry."""
    for name in legacy_first_engines("char_voice"):
        assert REGISTRY.is_registered(name), name
        assert "char_voice" in REGISTRY.get_engine(name).roles, name


# --- P2: bank truth -------------------------------------------------------


def test_p2_1_every_routed_voice_ref_resolves_to_exactly_one_bank_row():
    """P2.1 -- The route resolver demands EXACTLY ONE bank entry for
    (voice_ref_id, engine) and has no fallback; zero or two is a hard failure at
    cast time, not a degraded render."""
    bank, _ = load_voice_bank()
    routes = POOLS.LEMMY_VOICE_POLICY.get("approved_native_routes") or {}

    for engine, route in routes.items():
        ref_id = route["qualification_record"]["voice_ref_id"]
        matches = [e for e in bank
                   if e.voice_ref_id == ref_id and e.engine == engine]
        assert len(matches) == 1, (
            "route %r names %r on %r and the bank holds %d matching rows"
            % (route.get("route_id"), ref_id, engine, len(matches))
        )


def test_p2_2_a_preset_engine_may_carry_zero_bank_rows():
    """P2.2 -- Do NOT read "no bank rows" as "engine is broken". Bark is
    preset-driven by design, so a missing row is the correct state and a route
    for it is inexpressible (a route requires a bank-resident voice_ref_id)."""
    rows = _bank_rows()

    for engine in _PRESET_ONLY_ENGINES:
        assert not [r for r in rows if r["engine"] == engine], (
            "%s gained voice-bank rows; if that is intended, the preflight's "
            "preset-engine gate needs revisiting" % engine
        )
        assert REGISTRY.get_engine(engine).voice_ref_field == "voice_preset"


def test_p2_3_the_sha_sentinels_say_which_kind_of_row_this_is():
    """P2.3 -- `ref_sha256` is a hash for clone rows and a SENTINEL otherwise
    ("cloud", "pending"). Feeding a sentinel to a byte check is how `gt_algenib`
    once looked usable as a local IndexTTS2 reference."""
    for row in _bank_rows():
        digest = str(row.get("ref_sha256") or "")
        if str(row.get("ref_path") or "").startswith("cloud:"):
            assert digest == "cloud", row["voice_ref_id"]
            assert row.get("provider_voice_id"), row["voice_ref_id"]
        else:
            assert digest == "pending" or len(digest) == 64, row["voice_ref_id"]


# --- P3: generated data must not be destroyed by its own generator ---------


def _mirror_module():
    import importlib.util

    script = os.path.join(_REPO, "scripts", "_otr_mirror_clone_refs.py")
    spec = importlib.util.spec_from_file_location("_otr_mirror_clone_refs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p3_1_the_mirror_generator_never_deletes_a_row_it_cannot_recreate():
    """P3.1 -- THE ONE THAT NEARLY COST THREE ROWS.

    The generator used to drop EVERY chatterbox/dia row and rebuild from the
    indextts2 rows, so anything it did not itself produce was destroyed. The
    bank had since gained `cb_announcer_female`, `dia_announcer_male` and
    `dia_announcer_female` -- pinned by nine assertions across
    tests/test_voice_bank.py and tests/test_tts_engine_sidecars.py -- and a
    re-run invited by its own "idempotent" docstring would have deleted all
    three, turning the suite red for reasons nobody would connect to
    "I refreshed the mirror".

    A generator may only own the keys it can actually recreate.
    """
    before = _bank_rows()
    plan = _mirror_module().plan_rows(before)

    surviving = {row["voice_ref_id"] for row in plan["voices"]}
    for row in before:
        assert row["voice_ref_id"] in surviving, (
            "the generator would DELETE %r, which it does not regenerate"
            % row["voice_ref_id"]
        )


def test_p3_2_a_second_run_is_a_no_op():
    """P3.1 (cont.) -- Prove a re-run reproduces the bank before trusting the
    word 'idempotent'. Running the planner over its own output must be a fixed
    point, or the bank churns every time somebody refreshes it."""
    module = _mirror_module()

    once = module.plan_rows(_bank_rows())["voices"]
    twice = module.plan_rows(once)["voices"]

    assert once == twice


def test_p3_3_generated_ids_do_not_carry_the_source_prefix():
    """P3.2 -- The id helper stripped only `vz_`, so the first non-`vz_` source
    row would have minted `cb_idx_lemmy_...`. An id is a contract; a doubled
    prefix in it is a rename waiting to happen."""
    module = _mirror_module()

    assert module._new_id("cb_", "idx_lemmy_algenib_cockney_v1") == \
        "cb_lemmy_algenib_cockney_v1"
    assert module._new_id("dia_", "vz_bill_boerst") == "dia_bill_boerst"


def test_p3_4_the_generator_never_drops_a_FIELD_from_a_row_it_owns():
    """P3.4 -- THE HALF THE ROW-LEVEL FIX MISSED, found by running it for real.

    Ownership was fixed at the ROW level and the receipt said so:
    `mirrored=83 added=2 preserved-unmanaged=3`. That line prints identically
    whether or not a FIELD is lost, because it counts rows -- so the first real
    run against the real bank quietly stripped `speaker_id` from all eight
    mirrored rows that carried one (it was not on a seven-field allow-list
    written before the field existed) and reverted a hand-improved
    `cb_announcer_male` to a hardcoded literal.

    A row that survives with fewer fields than it had has not survived.
    """
    before = _bank_rows()
    after = {(row["engine"], row["voice_ref_id"]): row
             for row in _mirror_module().plan_rows(before)["voices"]}

    for row in before:
        key = (row["engine"], row["voice_ref_id"])
        missing = sorted(set(row) - set(after[key]))
        assert not missing, (
            "the generator would DROP %s from %r -- a generator owns the fields "
            "it derives, never the ones it does not know about" % (missing, key))


def test_p3_5_a_mirror_carries_its_sources_speaker_id():
    """P3.5 -- and this is WHY the field-level rule matters, not just that it is
    tidy. `speaker_id` records the real human behind a reference, because a
    ref_path collision cannot catch two recordings of ONE person: LibriVox's
    Mark F. Smith has a plain and a grandfatherly take in two different files.
    Strip it from the clone engines and one narrator can be cast as two
    characters in the same episode -- on chatterbox and dia only, which is a
    casting defect nobody would trace back to a bank generator."""
    planned = _mirror_module().plan_rows(_bank_rows())["voices"]
    by_id = {(row["engine"], row["voice_ref_id"]): row for row in planned}

    sources = [row for row in planned
               if row["engine"] == "indextts2" and row.get("speaker_id")]
    assert sources, "no indextts2 row carries a speaker_id -- the pin is vacuous"

    for src in sources:
        if "char_voice" not in src.get("roles", []):
            continue
        for engine, prefix in (("chatterbox", "cb_"), ("dia", "dia_")):
            mirror = by_id.get(
                (engine, _mirror_module()._new_id(prefix, src["voice_ref_id"])))
            assert mirror is not None
            assert mirror.get("speaker_id") == src["speaker_id"], (
                "%s mirror of %r lost its speaker_id"
                % (engine, src["voice_ref_id"]))


# --- P4: the route contract -----------------------------------------------


def test_p4_1_qualification_still_requires_a_human():
    """P4.1 -- `operator_verdict` is the field no automated pass may supply. A
    driver-signed verdict is the evidence-shaped-but-not-evidence pattern that
    produced BUG-12.86."""
    assert "operator_verdict" in POOLS.QUALIFICATION_RECEIPT_REQUIRED_FIELDS


@pytest.mark.parametrize("receipt", [
    None,
    "canonical_bark_preset_v1",          # the historical bare-string lie
    {},
    {"operator_verdict": ""},            # present-but-empty is the same lie
])
def test_p4_2_an_unproven_route_is_never_qualified(receipt):
    """P4.2 -- Fail-closed: a route that cannot prove it was auditioned has not
    been auditioned."""
    assert POOLS.is_qualified_route({"qualification_receipt": receipt}) is False


def test_p4_3_the_canonical_route_stays_honestly_unqualified():
    """P4.3 -- Bark's default route is a ROUTING fact, not an audition, and its
    null receipt is what says so."""
    assert POOLS.LEMMY_VOICE_POLICY["canonical_route"]["qualification_receipt"] is None


# --- P5: engine runtime declarations --------------------------------------


def test_p5_1_every_engine_declares_how_its_identity_arrives():
    """P5.1 -- A cross-engine harness switches on ADAPTER METADATA, never a
    hand-written engine table that goes stale the first time an engine lands.

    The two declarations are NOT interchangeable, and the difference is the
    whole gate:

    * ``voice_ref_kind`` is how the reference is PASSED to the adapter. The
      three clone engines say ``wav_path`` and take it as a call argument, so
      they carry no row field at all.
    * ``voice_ref_field`` is which CAST-ROW FIELD holds the identity. Bark reads
      ``voice_preset`` and kokoro reads ``voice_ref_id``; neither takes a clip.

    An engine that declares NEITHER cannot be driven by a generic harness, which
    is exactly the failure this gate exists to catch before a batch is rendered.
    """
    known_kinds = {"wav_path", "voice_ref_id", "voice_preset", "provider_voice_id"}

    for name in legacy_first_engines("char_voice"):
        engine = REGISTRY.get_engine(name)
        kind = getattr(engine, "voice_ref_kind", None)
        field = getattr(engine, "voice_ref_field", None)

        assert kind or field, (
            "%s declares neither voice_ref_kind nor voice_ref_field -- nothing "
            "can tell how to hand it a voice" % name
        )
        if kind is not None:
            assert kind in known_kinds, (name, kind)
        if getattr(engine, "requires_voice_ref", False):
            assert kind == "wav_path", (
                "%s requires a reference, so it owes a wav_path kind" % name
            )


def test_p5_2_every_char_voice_engine_declares_a_positive_sample_rate():
    """P5.2 -- `pack_audio_batch` raises on any clip whose rate differs from the
    profile rate, so a wrong or absent declaration fails at pack time, well
    after the render is paid for."""
    for name in legacy_first_engines("char_voice"):
        rate = getattr(REGISTRY.get_engine(name), "sample_rate", 0)
        assert isinstance(rate, int) and rate > 0, (name, rate)
