# -*- coding: utf-8 -*-
"""The banana route -- contract tests (docs/2026-08-06-BUILD-SPEC-banana-route.md).

Covers the pure module (table closure ENUMERATED over every episode variety
combination, idempotence as a property, the quote state machine that shields
still_word's quoted script text, case rules, the phrase-safe cap), the gate
(guarded env reads, the fidelity-bank idiom + its drift-parity with casting),
the anti-rot property (no table source term may match OTR's own composition
vocabulary), and the video funnel integration (pinned ordering, receipt in
observability, OFF-path byte-identity).
"""
from __future__ import annotations

import logging
import re

import pytest

from nodes import _otr_banana_route as b


# --------------------------------------------------------------------------- #
# table shape
# --------------------------------------------------------------------------- #

def test_table_is_longest_source_first():
    lengths = [len(src) for src, _ in b.BANANA_TABLE]
    assert lengths == sorted(lengths, reverse=True)


def test_closure_enumerated_over_every_episode_table():
    """No replacement string, in ANY episode variety combination, may contain
    any source term -- one pass is final, and the RESERVED trap (cannon ->
    banana cannon) is caught here, not in a render."""
    combos = b._all_episode_pick_combos()
    assert len(combos) == 4  # SIDEARM x LONG pools; enumeration, not a magic count
    sources = [src for src, _ in b._ROWS]
    for picks in combos:
        for _src, rep in b._table_for(picks):
            for source in sources:
                assert not re.search(
                    r"\b" + re.escape(source) + r"\b", rep, re.IGNORECASE), (
                    picks, rep, source)


def test_idempotence_property_over_every_episode_table():
    """EVERY variety combo, reached through a REAL key (Sonnet QA catch: an
    earlier draft hard-coded variety_key='' and ran the v1 table four times)."""
    corpus = ("the gunman drew his revolver, tommy guns rattled, a hand "
              "grenade rolled by the straight razor and the ray gun")
    wanted = {tuple(sorted(p.items())) for p in b._all_episode_pick_combos()}
    seen = set()
    for i in range(512):
        key = "probe-%d" % i
        combo = tuple(sorted(b.select_varieties(key).items()))
        if combo in seen:
            continue
        seen.add(combo)
        r1 = b.apply(corpus, variety_key=key)
        r2 = b.apply(r1.text, variety_key=key)
        assert r2.text == r1.text and r2.substitutions == 0, key
        if seen == wanted:
            break
    assert seen == wanted, "keys never reached combos %s" % (wanted - seen)


def test_camera_corpus_is_untouched():
    for text in ("cinematic establishing shot",
                 "in-character cinematic medium shot, head and shoulders",
                 "centered and fully visible, wide shot",
                 "a medium/wide shot of the room"):
        r = b.apply(text)
        assert r.text == text and r.substitutions == 0


def test_excluded_terms_stay_excluded():
    for text in ("the water tank hisses", "he chops wood with an axe",
                 "a game of blackjack", "gunfire echoes", "a single gunshot",
                 "held at gunpoint", "the night club glows"):
        r = b.apply(text)
        assert r.text == text and r.substitutions == 0, text


def test_stakes_pass_untouched_the_accepted_residual():
    for text in ("two shots rang out", "Macbeth stands over Duncan's body",
                 "blood on the floor"):
        r = b.apply(text)
        assert r.text == text and r.substitutions == 0


# --------------------------------------------------------------------------- #
# substitution semantics
# --------------------------------------------------------------------------- #

def test_bare_noun_articles_and_possessives_survive():
    assert b.apply("his revolver").text == "his banana"
    assert b.apply("a rifle").text == "a long banana"
    assert b.apply("the brass knuckles").text == "the banana peels"


def test_case_preservation():
    assert b.apply("revolver").text == "banana"
    assert b.apply("Revolver").text == "Banana"
    assert b.apply("REVOLVER").text == "BANANA"
    assert b.apply("Gunman").text == "Man wielding a banana"
    assert b.apply("GUNMEN").text == "MEN WIELDING BANANAS"


def test_gunman_wrapper_wields_never_holds():
    r = b.apply("the gunman waits")
    assert r.text == "the man wielding a banana waits"
    assert "holding" not in r.text
    assert b.apply("three gunmen wait").text == "three men wielding bananas wait"


def test_longest_source_wins_over_substrings():
    assert b.apply("a machine gun").text == "a bunch of bananas"
    assert b.apply("a ray gun").text == "a banana beam"
    # Indefinite-article agreement is deliberately NOT repaired: both sources
    # are vowel-initial, so "an assault rifle" -> "an long banana" and
    # "an ice pick" -> "an banana" are the real output. Cosmetic only -- no
    # hash, receipt or shielding consequence -- and an article pass would need
    # a vowel-sound table, a new subsystem to fix two rows.
    assert b.apply("an assault rifle").text == "an long banana"
    assert b.apply("a hand grenade").text == "a banana"


def test_punctuation_and_whitespace_preserved():
    assert b.apply("He dropped the knife.").text == "He dropped the banana."
    assert b.apply("knives, daggers; swords!").text == "bananas, bananas; bananas!"


# --------------------------------------------------------------------------- #
# the quote state machine (script integrity for still_word cards)
# --------------------------------------------------------------------------- #

def test_still_word_card_quoted_script_is_byte_identical():
    text = 'a title card displaying the words "he drew his revolver" in bold serif'
    r = b.apply(text)
    assert '"he drew his revolver"' in r.text
    assert r.substitutions == 0


def test_unquoted_text_around_a_quoted_span_still_transforms():
    r = b.apply('the gunman reads "his revolver gleamed" from a card')
    assert r.text.startswith("the man wielding a banana reads ")
    assert '"his revolver gleamed"' in r.text


def test_multiple_spans_shield_independently():
    r = b.apply('"a revolver" and a revolver and "a rifle"')
    assert r.text == '"a revolver" and a banana and "a rifle"'


def test_curly_quotes_pair_and_cross_style_does_not():
    curly = "\u201chis revolver\u201d and his revolver"
    r = b.apply(curly)
    assert r.text == "\u201chis revolver\u201d and his banana"
    # cross-style: straight opener + curly closer never pair -- both literal,
    # so the span between them transforms
    cross = '"his revolver\u201d stays open'
    assert b.apply(cross).substitutions == 1


def test_unmatched_opener_does_not_shield_the_remainder():
    r = b.apply('he said " and drew his revolver')
    assert r.text == 'he said " and drew his banana'


def test_escaped_quotes_do_not_open_spans():
    # odd backslash run = escaped delimiter = literal, not an opener
    r = b.apply('label \\" his revolver')
    assert "banana" in r.text
    # even run = the delimiter is a REAL opener (unmatched -> literal anyway)
    r2 = b.apply('label \\\\" his revolver')
    assert "banana" in r2.text


# --------------------------------------------------------------------------- #
# variety
# --------------------------------------------------------------------------- #

def test_empty_key_is_index_zero_and_byte_identical_to_v1():
    picks = b.select_varieties("")
    assert all(v == 0 for v in picks.values())
    assert b.varieties_receipt(picks) == "SIDEARM=banana,LONG=long banana"


def test_variety_is_deterministic_per_key_and_rotates():
    a1 = b.select_varieties("freeze_A")
    a2 = b.select_varieties("freeze_A")
    assert a1 == a2
    seen = {tuple(sorted(b.select_varieties("freeze_%d" % i).items()))
            for i in range(32)}
    assert len(seen) > 1, "32 keys never rotated any pool"


def test_variety_reaches_the_text_and_the_receipt():
    for i in range(64):
        key = "k%d" % i
        if b.select_varieties(key)["LONG"] == 1:
            r = b.apply("his rifle", variety_key=key)
            assert r.text == "his plantain"
            assert "LONG=plantain" in r.varieties
            break
    else:
        pytest.fail("no key selected the plantain pool in 64 tries")


# --------------------------------------------------------------------------- #
# the phrase-safe cap
# --------------------------------------------------------------------------- #

def test_cap_under_budget_is_unchanged():
    assert b.cap_phrase_safe("short prompt", 188) == "short prompt"


def test_cap_preserves_trailing_clause_and_word_boundary():
    text = ("word " * 50).strip() + ", slow cinematic camera drift, no on-screen text"
    capped = b.cap_phrase_safe(text, 100)
    assert len(capped) <= 100
    assert capped.endswith("no on-screen text")
    assert not capped.split(",")[0].endswith("wor")  # no mid-word cut


def test_cap_never_splits_a_replacement_phrase():
    body = ("x" * 150) + ", man wielding a red banana, no on-screen text"
    capped = b.cap_phrase_safe(body, 170)
    assert "man wielding a red ban" not in capped or \
        "man wielding a red banana" in capped
    assert capped.endswith("no on-screen text")


def test_cap_empty_text_returns_empty():
    assert b.cap_phrase_safe("", 188) == ""


# --------------------------------------------------------------------------- #
# the cap's PHRASE-INTEGRITY property (QA r2/r4)
#
# The old single-retreat loop truncated INSIDE a replacement phrase in 68 of the
# 3,641 cases this file's property test builds, by two mechanisms: the phrase
# inventory OVERLAPS in ordinary text ("...red banana peels off his mask"
# contains both `man wielding a red banana` and `banana peels`, so retreating
# out of the second lands inside the first), and the separator strip walked the
# cut back INTO a phrase because a space is a legal interior character of every
# multi-word entry. The retreat is now a fixed point.
# --------------------------------------------------------------------------- #

def _phrase_occurrences(text):
    """[(start, end)] of every replacement-phrase occurrence, case-folded."""
    low = text.lower()
    spans = []
    for phrase in b._PHRASES:
        i = low.find(phrase)
        while i >= 0:
            spans.append((i, i + len(phrase)))
            i = low.find(phrase, i + 1)
    return spans


def _ends_on_a_partial_phrase(source, capped):
    """True when the VISIBLE text ends part-way through a replacement phrase.

    A cut that is interior to one occurrence but exactly the END of another is
    NOT a split -- in "red banana peels", cutting at 10 yields the complete
    phrase "red banana" even though 10 is interior to "banana peels". The
    reader sees a whole phrase, which is the property that matters."""
    cut = 0
    low_c, low_s = capped.lower(), source.lower()
    while cut < len(capped) and cut < len(source) and low_c[cut] == low_s[cut]:
        cut += 1
    if cut == 0 or cut >= len(source):
        return False
    spans = _phrase_occurrences(source)
    covered = any(start < cut < end for start, end in spans)
    ends_whole = any(end == cut for _start, end in spans)
    return covered and not ends_whole


def _a_whole_phrase_fits(source, budget):
    """Could ANY complete phrase occurrence have fit inside the budget? When
    none can, the documented impossible-budget fallback runs and a partial
    phrase is the only thing left to return."""
    return any(end <= budget for _start, end in _phrase_occurrences(source))


def test_cap_never_ends_on_a_partial_phrase_property():
    """Fixtures DERIVED from the live _PHRASES inventory, so a new pool entry
    or _WRAP_TEMPLATES variant is covered automatically instead of silently
    escaping a hand-written fixture."""
    tails = (" peels off his mask", " beam tail words", ", no on-screen text",
             " and more words here", " peels red bananas here")
    checked = 0
    for phrase in b._PHRASES:
        for tail in tails:
            source = ("x" * 60) + " " + phrase + tail
            for max_chars in range(40, len(source)):
                checked += 1
                capped = b.cap_phrase_safe(source, max_chars)
                if not _a_whole_phrase_fits(source, max_chars):
                    continue        # the documented impossible-budget fallback
                assert not _ends_on_a_partial_phrase(source, capped), (
                    "cut inside a phrase: phrase=%r max_chars=%d -> %r"
                    % (phrase, max_chars, capped[-40:]))
    assert checked > 3000, checked


@pytest.mark.parametrize("source,max_chars", [
    # the four regressions the single-break loop actually failed on
    (("x" * 100) + " man wielding a red banana peels off his mask", 128),
    (("x" * 100) + " man wielding a banana peels off his mask", 124),
    (("x" * 40) + " long banana beam tail words here", 52),
    (("x" * 40) + " long banana beam, no on-screen text", 71),
])
def test_cap_reproduced_phrase_split_regressions(source, max_chars):
    capped = b.cap_phrase_safe(source, max_chars)
    assert not _ends_on_a_partial_phrase(source, capped), capped[-40:]
    assert len(capped) <= max_chars


def test_cap_retreat_that_consumes_the_body_keeps_a_whole_phrase():
    """When the retreat eats the entire body the answer is the longest COMPLETE
    phrase that fits -- never a re-split one, and never blank."""
    source = "red banana peels off his mask"
    capped = b.cap_phrase_safe(source, 12)
    assert capped                                   # never blank
    assert len(capped) <= 12
    assert not _ends_on_a_partial_phrase(source, capped)
    assert capped == "red banana"


def test_cap_postcondition_holds_for_direct_callers():
    """cap_phrase_safe is exported and called directly by tests, so the
    ``<= max_chars`` postcondition must not depend on the funnel's
    boundary-crossing precondition."""
    clause = "no on-screen text"
    corpus = [
        "a gunman, " + clause + ", " + ("tail content " * 20),   # clause mid-prompt
        clause + ", " + ("a gunman walks, " * 12),               # clause at index 0
        ("x" * 300),
        ", , , , , ,",
        "man wielding a red banana",
    ]
    for source in corpus:
        for max_chars in range(0, 200, 7):
            capped = b.cap_phrase_safe(source, max_chars)
            assert len(capped) <= max_chars, (source[:30], max_chars, capped)


# --------------------------------------------------------------------------- #
# the cap's BRANCH-PUBLISHED protected clause (QA r2)
# --------------------------------------------------------------------------- #

_IA2V_CLAUSE = ("is talking to the viewer, lips opening and closing naturally "
                "in sync with the speech, subtle head and hand gestures. "
                "Static camera.")
_DRIFT_CLAUSE = "slow cinematic camera drift"


def test_cap_honours_a_MIXED_CASE_protected_clause():
    """The regression guard for a silent no-op: the old carve lowered only the
    haystack, so a clause carrying a capital letter -- `Static camera.` -- was
    never found, and the fix would have quietly reproduced the bug it fixes."""
    body = "A weathered Lead Astronomer with a gunman at his shoulder"
    text = b.apply("%s %s" % (body, _IA2V_CLAUSE), variety_key="").text
    capped = b.cap_phrase_safe(text, 190, _IA2V_CLAUSE)
    assert capped.endswith("Static camera.")
    assert len(capped) <= 190


def test_cap_protected_clause_none_and_empty_match_the_default():
    text = ("word " * 50).strip() + ", " + _DRIFT_CLAUSE + ", no on-screen text"
    default = b.cap_phrase_safe(text, 100)
    assert b.cap_phrase_safe(text, 100, None) == default
    assert b.cap_phrase_safe(text, 100, "") == default
    assert default.endswith("no on-screen text")


def test_cap_branch_clause_is_honoured_mid_prompt_but_the_fallback_is_not():
    """A BRANCH clause is matched anywhere -- its promise covers everything from
    the clause to the end, which is how the brief+beat motion clause keeps the
    era tail spliced in after it. The module fallback is TRAILING-only, because
    a mid-prompt `no on-screen text` is content, not a render constraint."""
    text = ("A dockside gunman levels his revolver at a night-shift dispatcher "
            "in a rain-drowned harbor town, gathering tension, rising stakes, "
            + _DRIFT_CLAUSE + ", salt, no on-screen text")
    post = b.apply(text, variety_key="").text
    with_clause = b.cap_phrase_safe(post, 188, _DRIFT_CLAUSE)
    assert _DRIFT_CLAUSE in with_clause
    assert with_clause.endswith("no on-screen text")
    # fallback only: the drift clause is expendable body
    assert _DRIFT_CLAUSE not in b.cap_phrase_safe(post, 188)
    # a mid-prompt fallback clause is CONTENT -- it does not become a suffix
    mid = "a gunman, no on-screen text, " + ("tail words " * 20)
    assert len(b.cap_phrase_safe(mid, 60)) <= 60


def test_cap_falls_back_when_the_branch_clause_was_already_trimmed_away():
    """A branch whose composer already ate its clause must not come out WORSE
    than before -- it falls back to the module clause, never to no protection."""
    text = ("A dockside gunman levels his revolver at a night-shift dispatcher "
            "in a rain-drowned harbor town, gathering tension, slow, "
            "no on-screen text")
    post = b.apply(text, variety_key="").text
    capped = b.cap_phrase_safe(post, 150, _DRIFT_CLAUSE)
    assert capped.endswith("no on-screen text")


def test_protected_clauses_are_transform_invariant():
    """The cap can only anchor on a clause the transform cannot move or grow."""
    for clause in (b._TRAILING_CLAUSE, _DRIFT_CLAUSE, _IA2V_CLAUSE):
        for key in ("", "k1", "k2", "k3"):
            result = b.apply(clause, variety_key=key)
            assert result.substitutions == 0, clause
            assert result.text == clause, clause


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #

def test_guarded_env_reader_never_raises(monkeypatch):
    monkeypatch.setenv("OTR_BANANA_STILLS", "sideways")
    assert b.banana_stills_enabled() is True     # default, with a warning
    monkeypatch.setenv("OTR_BANANA_STILLS", "0")
    assert b.banana_stills_enabled() is False
    monkeypatch.setenv("OTR_BANANA_STILLS", "off")
    assert b.banana_stills_enabled() is False
    monkeypatch.delenv("OTR_BANANA_STILLS", raising=False)
    assert b.banana_stills_enabled() is True


def test_present_but_EMPTY_knob_is_malformed_not_true(monkeypatch):
    """`""` used to sit in _TRUE_TOKENS, so a present-but-empty
    OTR_BANANA_INCLUDE_FIDELITY_BANKS read as TRUE and silently bananafied the
    fidelity lanes -- the one exclusion an operator ruling protects. `raw is
    None` already covers "unset", so empty is a MALFORMED knob: default, plus
    one warning (BUILD-SPEC section 1)."""
    for raw in ("", " ", "\t"):
        monkeypatch.setenv("OTR_BANANA_INCLUDE_FIDELITY_BANKS", raw)
        assert b.include_fidelity_banks() is False, repr(raw)
        assert b.banana_gate({"source_bank": "shakespeare"},
                             lane="stills") is False, repr(raw)
        assert b.banana_gate({"source_bank": "public_domain"},
                             lane="video") is False, repr(raw)


def test_malformed_knob_warns_ONCE_per_distinct_value(monkeypatch, caplog):
    """The gate runs per still, per video request AND per beat at ShotLock
    cast-time preflight, so an undeduped warning is hundreds of lines on one
    episode against BUILD-SPEC section 1's ONE warning naming the key. A
    CHANGED bad value is a new mistake and warns again."""
    monkeypatch.setenv("OTR_BANANA_STILLS", "sideways")
    b._WARNED_MALFORMED_ENV.discard(("OTR_BANANA_STILLS", "sideways"))
    b._WARNED_MALFORMED_ENV.discard(("OTR_BANANA_STILLS", "diagonally"))
    with caplog.at_level(logging.WARNING, logger="OTR"):
        for _ in range(6):
            assert b.banana_stills_enabled() is True
        first = [r for r in caplog.records if "sideways" in r.getMessage()]
        assert len(first) == 1, [r.getMessage() for r in caplog.records]
        monkeypatch.setenv("OTR_BANANA_STILLS", "diagonally")
        assert b.banana_stills_enabled() is True
        second = [r for r in caplog.records if "diagonally" in r.getMessage()]
        assert len(second) == 1


def test_fidelity_banks_default_off_and_variants_inherit(monkeypatch):
    monkeypatch.delenv("OTR_BANANA_INCLUDE_FIDELITY_BANKS", raising=False)
    for bank in ("shakespeare", "public_domain", "shakespeare_v2",
                 "public_domain_v3", "  Shakespeare  "):
        assert b.source_bank_excludes_banana(bank), bank
        assert b.banana_gate({"source_bank": bank}, lane="stills") is False
    for bank in ("original", "media_archive", "scifi_news", ""):
        assert not b.source_bank_excludes_banana(bank), bank


def test_override_forces_fidelity_lanes_on(monkeypatch):
    monkeypatch.setenv("OTR_BANANA_INCLUDE_FIDELITY_BANKS", "1")
    assert b.banana_gate({"source_bank": "shakespeare"}, lane="video") is True


def test_gate_lane_switches_are_independent(monkeypatch):
    monkeypatch.setenv("OTR_BANANA_STILLS", "0")
    monkeypatch.delenv("OTR_BANANA_VIDEO", raising=False)
    meta = {"source_bank": "original"}
    assert b.banana_gate(meta, lane="stills") is False
    assert b.banana_gate(meta, lane="video") is True
    with pytest.raises(ValueError):
        b.banana_gate(meta, lane="audio")


def test_exclusion_frozenset_parity_with_casting():
    """ONE answer in the tree to 'what is a fidelity lane' -- the copy is the
    operator's ruling; this equality is what stops it drifting."""
    from nodes import _otr_casting as casting
    assert (b._BANANA_EXCLUDED_SOURCE_BANK_IDS
            == casting._LEMMY_EXCLUDED_SOURCE_BANK_IDS)


# --------------------------------------------------------------------------- #
# anti-rot: the table may never collide with OTR's own composition vocabulary
# --------------------------------------------------------------------------- #

def test_no_source_term_matches_otr_composition_constants():
    from nodes._otr_video_engines import render_driver as rd
    import nodes._otr_video_engines  # noqa: F401 -- populate the registry
    from nodes._otr_video_engines import registry as vreg

    constants = ["cinematic establishing shot",
                 rd._CHAR_FACE_FALLBACK_PROMPT]
    constants.extend(rd._LTX_MOTION_PROMPT_BY_ROLE.values())
    constants.extend(rd._INTENT_CLAUSES.values())
    constants.extend(rd._ARC_CLAUSES.values())
    for name in vreg.all_engine_names():
        plan = getattr(vreg.get_engine(name), "still_plan", ()) or ()
        for row_ in plan:
            geometry = getattr(row_, "framing_geometry", "") or ""
            constants.append(str(geometry))
    # ALL NINE style packs, not just the sci_fi_radio fixture (Sonnet QA):
    # motion registers / grade tails / era tails are freely authored per pack,
    # and a future pack edit must not silently collide with the table.
    import json
    import pathlib
    packs_dir = (pathlib.Path(__file__).resolve().parent.parent
                 / "nodes" / "visual_styles")

    def _strings(node):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from _strings(v)
        elif isinstance(node, list):
            for v in node:
                yield from _strings(v)

    for pack in sorted(packs_dir.glob("*.json")):
        constants.extend(_strings(json.loads(pack.read_text(encoding="utf-8"))))
    offenders = []
    for source, _spec in b._ROWS:
        pattern = re.compile(r"\b" + re.escape(source) + r"\b", re.IGNORECASE)
        for text in constants:
            if pattern.search(text):
                offenders.append((source, text[:60]))
    assert offenders == [], offenders


# --------------------------------------------------------------------------- #
# receipts
# --------------------------------------------------------------------------- #

_RECEIPT_KEYS = ("banana_route", "banana_table_version",
                 "banana_substitutions", "banana_sha256_before",
                 "banana_sha256_after", "banana_varieties")


def test_on_receipt_shape():
    r = b.apply("his revolver")
    keys = b.receipt_keys(r)
    assert tuple(sorted(keys)) == tuple(sorted(_RECEIPT_KEYS))
    assert keys["banana_route"] == "on"
    assert keys["banana_substitutions"] == 1
    assert keys["banana_sha256_before"] != keys["banana_sha256_after"]


def test_off_receipt_shape_and_forensics():
    keys = b.off_receipt("his revolver", variety_key="freeze_X")
    assert tuple(sorted(keys)) == tuple(sorted(_RECEIPT_KEYS))
    assert keys["banana_route"] == "off"
    assert keys["banana_substitutions"] == 0
    assert keys["banana_sha256_before"] == keys["banana_sha256_after"]
    # the OFF receipt still names the fruits that WOULD have applied
    assert keys["banana_varieties"] == b.varieties_receipt(
        b.select_varieties("freeze_X"))


def test_zero_match_prompt_receipt_is_honest():
    r = b.apply("a quiet reading room, warm lamplight")
    assert r.substitutions == 0
    assert r.sha256_before == r.sha256_after


# --------------------------------------------------------------------------- #
# the video funnel (integration: pinned ordering, ambient gate, OFF identity)
# --------------------------------------------------------------------------- #

def _music_open_ledger(bank="original"):
    from nodes._otr_video_engines import render_driver as rd
    return {
        "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA,
                  "ledger_frozen": True},
        "meta": {
            "visual_style": "video_art",
            "style": "archive signal mystery",
            "source_bank": bank,
            "freeze_timestamp": "freeze_banana_probe",
            "story_brief_terms": {"setting": ["tape archive"],
                                  "atmosphere": ["electrical unease"]},
        },
        "lines": [
            {"line_id": "b001", "char_id": "c01",
             "speaker_role": "character", "start_s": 0.0, "dur_s": 4.0},
        ],
        "images": {"images": [
            {"object_id": "still_b001", "kind": "scene_character",
             "beat_id": "b001", "char_id": "c01", "path": "X:/img/scene.png"},
        ]},
        "video": {"video_revision": 1, "fps": 25, "shots": [
            {"shot_id": "shot_b001",
             "source_line_ids": ["b001"],
             "role": "character_video",
             "engine_id": "ltx_video",
             "family": "text_to_video",
             "group_id": "grp_char",
             "target_frame_count": 100,
             "degradation_trail": [],
             "render_request_hash": "hash_banana_probe",
             "cache_keys": {"request_hash": "hash_banana_probe"},
             # the M4 creative prompt -- highest precedence on the composer
             # (render_driver reads creative.text_prompt at :2580)
             "creative": {"text_prompt": "the gunman levels his revolver at "
                                         "the radio console",
                          "source": "m4"}},
        ]},
    }


def test_video_funnel_transforms_and_stamps_receipt(monkeypatch):
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.delenv("OTR_BANANA_VIDEO", raising=False)
    led = _music_open_ledger()
    req = rd.build_request_from_shot(led["video"]["shots"][0], led)
    prompt = str(req.get("text_prompt") or "")
    assert "revolver" not in prompt.lower()
    assert "banana" in prompt.lower()
    obs = req.get("observability") or {}
    assert obs.get("banana_route") == "on"
    assert obs.get("banana_substitutions", 0) >= 1
    assert obs.get("banana_varieties")
    # seed derivation ran AFTER the transform and is still present
    assert "video_seed" in obs


def test_video_funnel_off_is_byte_identical(monkeypatch):
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.setenv("OTR_BANANA_VIDEO", "0")
    led_off = _music_open_ledger()
    req_off = rd.build_request_from_shot(led_off["video"]["shots"][0], led_off)
    monkeypatch.delenv("OTR_BANANA_VIDEO", raising=False)
    monkeypatch.setenv("OTR_BANANA_VIDEO", "1")
    led_on = _music_open_ledger()
    req_on = rd.build_request_from_shot(led_on["video"]["shots"][0], led_on)
    off_prompt = str(req_off.get("text_prompt") or "")
    assert "revolver" in off_prompt.lower()
    obs_off = req_off.get("observability") or {}
    assert obs_off.get("banana_route") == "off"
    assert obs_off.get("banana_substitutions") == 0
    # equal seeds across ON/OFF: the banana is the only difference
    assert (req_off.get("seed_bundle") == req_on.get("seed_bundle"))


def test_video_funnel_respects_fidelity_bank(monkeypatch):
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.delenv("OTR_BANANA_VIDEO", raising=False)
    monkeypatch.delenv("OTR_BANANA_INCLUDE_FIDELITY_BANKS", raising=False)
    led = _music_open_ledger(bank="shakespeare")
    req = rd.build_request_from_shot(led["video"]["shots"][0], led)
    assert "revolver" in str(req.get("text_prompt") or "").lower()
    assert (req.get("observability") or {}).get("banana_route") == "off"


# --------------------------------------------------------------------------- #
# the still funnel seam (the exact three lines, proven at the unit level)
# --------------------------------------------------------------------------- #

def test_still_seam_hash_covers_the_transform(monkeypatch):
    """The dispatcher hashes AFTER the transform, so flipping the switch
    re-mints cached stills instead of serving a stale gun. Proven on the same
    functions the funnel composes, in the funnel's order."""
    from nodes.otr_image_gen_dispatcher import _prompt_content_hash
    monkeypatch.delenv("OTR_BANANA_STILLS", raising=False)
    raw = "a man draws his revolver in a smoky office"
    res = b.apply(raw, variety_key="freeze_banana_probe")
    assert _prompt_content_hash(res.text) != _prompt_content_hash(raw)
    assert b.banana_gate({"source_bank": "original"}, lane="stills") is True
    assert b.banana_gate({"source_bank": "shakespeare"}, lane="stills") is False
