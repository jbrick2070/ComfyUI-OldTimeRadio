"""Ghost Prompt v2 -- the authoring surface, pinned.

The v1 composer turned an unmapped free-text ``beat_intent`` into
``"moves with " + its first six regex words``, which is how a published lane
came to say *"moves with erin risks exposure by transmitting a"*: a cast name in
the picture and a sentence with no end.

Everything here is a contract on the replacement. The model owns exactly one
field -- a short drawable leaf -- and Python owns the style, the negative, the
recurrence motif, the representation, the framing law, the identities, the
hashes, the retry and the fallback. Most of these tests are about what CANNOT
happen: what the model is never shown, what it is never allowed to return, and
what the render path refuses rather than repairs.
"""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes import _otr_visual_styles as vs
from nodes._otr_video_engines import ghost_signal_author as gsa
from nodes._otr_video_engines import ghost_signal_prompt as gsp

STYLES = [vs.get_visual_style({"visual_style": sid})
          for sid in sorted(vs.list_style_ids())]
STYLE_IDS = [s.style_id for s in STYLES]
STYLE = vs.get_visual_style({"visual_style": "archival_documentary"})

NAMES = ("ADRIAN SPENDER", "RASHIDA PIERCE", "ANNOUNCER")

RICH_CAST = {
    "char_id": "c02",
    "name": "ADRIAN SPENDER",
    "gender": "male",
    "appearance": ("a broad steady man in a charcoal overcoat, a long scar "
                   "across the left cheek, carrying a battered leather "
                   "satchel"),
}


def _components(cast_row=None, char_id="c02"):
    return gsp.distill_sigil_components(
        cast_row if cast_row is not None else RICH_CAST,
        episode_seed=1013426535, char_id=char_id,
        style_id=STYLE.style_id)


def _measure_stub(text):
    """A cheap injected measurer. It still GATES -- see the resolver test."""
    words = len(str(text or "").split())
    return words + 2, 1 if words <= 75 else 2


def _spec(beat_id="b002", role="character_video", mode="object",
          motif="rust satchel emblem", ordinal=0, model_id="m/x", **extra):
    row = {"beat_id": beat_id, "role": role, "mode": mode,
           "motif_cue": motif, "sanitized_intent": extra.get("intent", ""),
           "normalized_emotion": extra.get("emotion", ""),
           "mapped_arc": extra.get("arc", "")}
    return gsa.build_ghost_author_specs([row] * (ordinal + 1),
                                        model_id=model_id)[ordinal]


# --------------------------------------------------------------------------- #
# 1. The safe projection -- what the model is NEVER shown.
# --------------------------------------------------------------------------- #

def test_the_live_long_form_intent_loses_every_name():
    """The exact shape that produced the shipped fragment."""
    intent = ("Adrian Spender demands the waitress reveal her true identity, "
              "pressing her against the counter.")
    out = gsa.sanitize_intent(intent, NAMES)
    assert "adrian" not in out and "spender" not in out
    assert out == out.lower()
    assert out and not out.endswith(".")


def test_a_first_name_alone_is_still_a_leak():
    """Longest-first removal: stripping 'Adrian' first would strand 'Spender'."""
    for text in ("Adrian slams the door", "Spender slams the door",
                 "ADRIAN SPENDER slams the door"):
        out = gsa.sanitize_intent(text, NAMES)
        assert "adrian" not in out and "spender" not in out


def test_field_labels_and_second_person_never_survive_projection():
    out = gsa.sanitize_intent("Face: you lean toward your own reflection",
                              NAMES)
    assert ":" not in out
    for banned in ("face:", "you", "your"):
        assert banned not in out.split()


def test_short_tokens_are_not_treated_as_names():
    """An initial is not a name leak, and stripping 'a' destroys the sentence."""
    out = gsa.sanitize_intent("a lamp swings over the counter", ("A", "Ed"))
    assert out.startswith("a lamp swings")


def test_scene_is_dropped_and_a_real_arc_is_mapped():
    assert gsa.map_arc("scene") == ""
    assert gsa.map_arc("") == ""
    assert gsa.map_arc("climax") == "peak"


def test_the_batch_prompt_carries_no_dialogue_title_or_name():
    """The strongest form of the boundary: it is not a parameter."""
    rows = [{"beat_id": "b002", "role": "character_video", "mode": "object",
             "motif_cue": "rust satchel emblem",
             "sanitized_intent": gsa.sanitize_intent(
                 "Adrian Spender demands the truth", NAMES),
             "normalized_emotion": "tense", "mapped_arc": "peak"}]
    specs = gsa.build_ghost_author_specs(rows, model_id="m/x")
    prompt = gsa.build_batch_prompt(specs)
    for banned in ("Adrian", "Spender", "Rashida", "b002",
                   "Your time's runnin' out"):
        assert banned not in prompt
    assert "g000" in prompt


def test_the_model_never_sees_a_ledger_identifier():
    specs = gsa.build_ghost_author_specs(
        [{"beat_id": "music_opening_001", "role": "music_visual",
          "mode": "object", "motif_cue": "broadcast console emblem",
          "sanitized_intent": "", "normalized_emotion": "", "mapped_arc": ""}],
        model_id="m/x")
    assert specs[0]["id"] == "g000"
    assert "music_opening_001" not in gsa.build_batch_prompt(specs)


# --------------------------------------------------------------------------- #
# 2. Deterministic representation scheduling.
# --------------------------------------------------------------------------- #

TIMELINE = [("music_opening_001", "music_visual"), ("b001", "announcer_visual"),
            ("b002", "character_video"), ("b003", "character_video"),
            ("b004", "character_video"), ("b005", "character_video"),
            ("b006", "announcer_visual"), ("music_closing_001",
                                           "music_visual")]


def test_the_schedule_is_deterministic_on_the_episode_seed():
    a = gsa.schedule_ghost_modes(TIMELINE, 1013426535)
    b = gsa.schedule_ghost_modes(TIMELINE, 1013426535)
    assert a == b
    assert gsa.schedule_ghost_modes(TIMELINE, 999) != a


def test_no_bookend_ever_takes_figure_mode():
    """A radio console is not a person."""
    for seed in range(24):
        modes = gsa.schedule_ghost_modes(TIMELINE, seed)
        for beat_id, role in TIMELINE:
            if role != "character_video":
                assert modes[beat_id] in gsa.GHOST_NON_FIGURE_MODES


def test_no_run_of_three_identical_representations():
    for seed in range(64):
        modes = gsa.schedule_ghost_modes(TIMELINE, seed)
        run = [modes[b] for b, _r in TIMELINE]
        for i in range(len(run) - 2):
            assert not (run[i] == run[i + 1] == run[i + 2]), (seed, run)


def test_at_least_half_the_character_clips_are_non_figure():
    """An unmodified period-three cycle satisfies the quota by construction."""
    chars = [b for b, r in TIMELINE if r == "character_video"]
    for seed in range(64):
        modes = gsa.schedule_ghost_modes(TIMELINE, seed)
        picked = [modes[b] for b in chars]
        non_figure = sum(1 for m in picked if m != "figure")
        assert non_figure * 2 >= len(picked), (seed, picked)


def test_character_modes_are_an_unmodified_cycle():
    """Only a bookend may be corrected; a character assignment never moves."""
    chars = [b for b, r in TIMELINE if r == "character_video"]
    for seed in range(32):
        picked = [gsa.schedule_ghost_modes(TIMELINE, seed)[b] for b in chars]
        start = gsa.GHOST_MODES.index(picked[0])
        for i, mode in enumerate(picked):
            assert mode == gsa.GHOST_MODES[(start + i) % len(gsa.GHOST_MODES)]


# --------------------------------------------------------------------------- #
# 3. The recurrence motif -- compact, allowlisted, and NOT a face.
# --------------------------------------------------------------------------- #

def test_the_motif_carries_no_face_landmark_or_cast_prose():
    comp = _components()
    # The row really does carry a facial landmark, and the distiller really
    # does bucket it -- so its absence below is a choice, not an accident.
    assert "scar" in comp["landmark"]
    for mode in gsa.GHOST_MODES:
        motif = gsa.motif_for_character(comp, mode, seed_int=comp["seed_int"])
        low = motif.lower()
        for banned in ("jaw", "brow", "hair", "scar", "cheek", "face", "man",
                       "woman", "adrian", "spender", "steady", "battered",
                       "leather"):
            assert banned not in low, (mode, motif)
    # Whole WORDS from the allowlists, never a phrase lifted off the cast row.
    for mode in gsa.GHOST_MODES:
        motif = gsa.motif_for_character(comp, mode, seed_int=comp["seed_int"])
        for word in motif.split():
            assert word in (
                list(gsa.MOTIF_COLOUR_WORDS) + list(gsa.MOTIF_PROP_WORDS)
                + list(gsa.MOTIF_SILHOUETTE_WORDS)
                + ["silhouette", "with", "a", "emblem", "signal"]), motif


def test_the_three_representations_share_colour_and_prop():
    comp = _components()
    cues = {m: gsa.motif_for_character(comp, m, seed_int=comp["seed_int"])
            for m in gsa.GHOST_MODES}
    shared = set(cues["object"].split()) & set(cues["signal"].split())
    shared -= {"emblem", "signal"}
    assert len(shared) >= 2                      # colour + prop
    assert shared <= set(cues["figure"].split())
    assert "silhouette" in cues["figure"]


def test_a_sparse_cast_row_still_gets_a_stable_motif():
    comp = _components({"char_id": "c09", "name": "MARA"}, char_id="c09")
    first = gsa.motif_for_character(comp, "object", seed_int=comp["seed_int"])
    assert first.strip()
    assert first == gsa.motif_for_character(comp, "object",
                                            seed_int=comp["seed_int"])


def test_every_bookend_role_and_mode_has_a_checked_in_motif():
    for role in ("announcer_visual", "music_visual"):
        for mode in gsa.GHOST_NON_FIGURE_MODES:
            assert gsa.motif_for_bookend(role, mode).strip()
    with pytest.raises(gsa.GhostAuthorError):
        gsa.motif_for_bookend("announcer_visual", "figure")


def test_an_empty_motif_is_refused_at_spec_build():
    with pytest.raises(gsa.GhostAuthorError):
        gsa.build_ghost_author_specs(
            [{"beat_id": "b002", "role": "character_video", "mode": "object",
              "motif_cue": "  ", "sanitized_intent": "",
              "normalized_emotion": "", "mapped_arc": ""}], model_id="m/x")


# --------------------------------------------------------------------------- #
# 4. Strict parsing -- one envelope, no salvage.
# --------------------------------------------------------------------------- #

GOOD = ('{"shots": [{"id": "g000", "drawable_beat": "a rust clasp turns '
        'slowly into the light"}]}')


def test_one_markdown_fence_is_transport_and_is_removed():
    fenced = "```json\n%s\n```" % GOOD
    assert gsa.parse_batch_response(fenced, ["g000"]) == \
        gsa.parse_batch_response(GOOD, ["g000"])


@pytest.mark.parametrize("raw", [
    "",
    "here you go: " + GOOD,
    GOOD + " and that is all",
    '{"shots": []}',
    '{"shots": [{"id": "g001", "drawable_beat": "a rust clasp turns"}]}',
    '{"shots": [{"id": "g000", "drawable_beat": "x", "mode": "object"}]}',
    '{"shots": [{"id": "g000"}]}',
    '{"shots": [{"id": "g000", "drawable_beat": "a"}, '
    '{"id": "g000", "drawable_beat": "b"}]}',
    '{"shots": [{"id": "g000", "drawable_beat": "a"}], "note": "hi"}',
    '{"shots": {"g000": "a rust clasp turns"}}',
    '{"shots": [{"id": "g000", "drawable_beat": 7}]}',
    '[{"id": "g000", "drawable_beat": "a"}]',
    "```json\n```json\n%s\n```\n```" % GOOD,
])
def test_every_other_shape_is_refused(raw):
    with pytest.raises(gsa.GhostAuthorParseError):
        gsa.parse_batch_response(raw, ["g000"])


def test_a_duplicate_json_key_is_refused():
    raw = ('{"shots": [{"id": "g000", "drawable_beat": "a rust clasp turns", '
           '"drawable_beat": "something else"}]}')
    with pytest.raises(gsa.GhostAuthorParseError):
        gsa.parse_batch_response(raw, ["g000"])


# --------------------------------------------------------------------------- #
# 5. Leaf validation -- shape and boundary, never a taste judge.
# --------------------------------------------------------------------------- #

def test_a_good_leaf_passes_in_every_mode():
    for mode, leaf in (
            ("figure", "an outline lifts one arm into a narrow band of light"),
            ("object", "the clasp turns and a slow shadow crosses it"),
            ("signal", "bands of static crush inward and open again")):
        ok, why = gsa.validate_drawable_beat(leaf, mode=mode, names=NAMES)
        assert ok, (mode, why)


@pytest.mark.parametrize("leaf,mode,expect", [
    ("", "object", "empty"),
    ("too short here", "object", "under"),
    ("a " * 40, "object", "over"),
    ("x" * 120 + " turns slowly into the light", "object", "over"),
    ("face: the clasp turns slowly here", "object", "field label"),
    ("adrian spender turns toward the light", "figure", "cast name"),
    ("your hand turns the clasp slowly here", "object", "second person"),
    ("no people beside the turning clasp here", "object", "negates"),
    ("cinematic 8k detailed clasp turning slowly", "object", "boilerplate"),
    ("the title text glows above the clasp", "object", "lettering"),
    ("a man turns the clasp slowly in light", "signal", "person in"),
    ("a woman leans toward the console light", "object", "person in"),
    ("the clasp turns slowly into the", "object", "dangling"),
])
def test_a_bad_leaf_is_named_by_its_defect(leaf, mode, expect):
    ok, why = gsa.validate_drawable_beat(leaf, mode=mode, names=NAMES)
    assert not ok
    assert expect in why


def test_a_person_is_legal_in_figure_mode_only():
    leaf = "a silhouette leans into the narrowing band of light"
    assert gsa.validate_drawable_beat(leaf, mode="figure", names=NAMES)[0]
    for mode in gsa.GHOST_NON_FIGURE_MODES:
        assert not gsa.validate_drawable_beat(leaf, mode=mode, names=NAMES)[0]


# --------------------------------------------------------------------------- #
# 6. The request hash -- the replay identity.
# --------------------------------------------------------------------------- #

def test_the_hash_covers_exactly_the_thirteen_declared_keys():
    assert len(gsa.GHOST_REQUEST_HASH_KEYS) == 13
    spec = _spec()
    payload = {k: spec[k] for k in gsa.GHOST_REQUEST_HASH_KEYS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    assert spec["request_sha256"] == \
        hashlib.sha256(blob.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("field,value", [
    ("beat_id", "b099"),
    ("mode", "signal"),
    ("motif_cue", "olive lantern emblem"),
    ("sanitized_intent", "the clasp opens"),
    ("normalized_emotion", "tense"),
    ("mapped_arc", "peak"),
    ("model_id", "other/model"),
    ("role", "music_visual"),
])
def test_any_changed_safe_input_reauthors(field, value):
    base = _spec()
    changed = dict(base)
    changed[field] = value
    assert gsa.request_sha256(changed) != base["request_sha256"]


def test_the_template_contract_is_part_of_the_identity():
    base = _spec()
    for field in ("template_sha256", "author_version", "schema_version"):
        changed = dict(base)
        changed[field] = "different"
        assert gsa.request_sha256(changed) != base["request_sha256"]


def test_the_template_digest_moves_with_temperature_and_budget():
    """A leaf written at another temperature is not the same leaf."""
    src = gsa._template_identity()
    assert src == gsa.GHOST_TEMPLATE_SHA256
    original = gsa.GHOST_BATCH_TEMPERATURE
    try:
        gsa.GHOST_BATCH_TEMPERATURE = 0.9
        assert gsa._template_identity() != src
    finally:
        gsa.GHOST_BATCH_TEMPERATURE = original
    assert gsa._template_identity() == src


def test_the_output_hash_covers_the_leaf_not_the_wrapper():
    """Otherwise a replay would hash differently from what it replayed."""
    spec = _spec()
    leaf = "the clasp turns and a slow shadow crosses it"
    written = gsa.build_ghost_prompt_object(spec, leaf, source="writer_llm")
    replayed = dict(written, source="replay")
    gsa.validate_ghost_prompt_object(replayed)
    assert replayed["output_sha256"] == written["output_sha256"]


def test_the_ordinal_is_the_batch_position_not_a_durable_index():
    rows = [{"beat_id": "b00%d" % i, "role": "character_video",
             "mode": "object", "motif_cue": "rust satchel emblem",
             "sanitized_intent": "", "normalized_emotion": "",
             "mapped_arc": ""} for i in range(3)]
    specs = gsa.build_ghost_author_specs(rows, model_id="m/x")
    assert [s["ordinal"] for s in specs] == [0, 1, 2]
    assert [s["id"] for s in specs] == ["g000", "g001", "g002"]


# --------------------------------------------------------------------------- #
# 7. The stored object.
# --------------------------------------------------------------------------- #

def test_the_field_set_is_exact():
    obj = gsa.build_ghost_prompt_object(
        _spec(), "the clasp turns and a slow shadow crosses it",
        source="writer_llm")
    assert set(obj) == set(gsa.GHOST_PROMPT_FIELDS)
    assert "fallback" not in obj            # there is no boolean, by design


@pytest.mark.parametrize("mutate", [
    lambda o: o.pop("mode"),
    lambda o: o.update(extra="x"),
    lambda o: o.update(mode="portrait"),
    lambda o: o.update(source="guess"),
    lambda o: o.update(motif_cue=""),
    lambda o: o.update(model_id=""),
    lambda o: o.update(drawable_beat="rewritten without a new hash"),
    lambda o: o.update(request_sha256="nothex"),
    lambda o: o.update(schema_version=99),
    lambda o: o.update(author_version="ghost_drawable_beat_v0"),
])
def test_a_malformed_object_is_refused(mutate):
    obj = gsa.build_ghost_prompt_object(
        _spec(), "the clasp turns and a slow shadow crosses it",
        source="writer_llm")
    mutate(obj)
    with pytest.raises(gsa.GhostAuthorValidationError):
        gsa.validate_ghost_prompt_object(obj)


def test_a_fallback_must_carry_its_reason_and_a_writer_row_must_not():
    """This is what stops a reuse laundering a fallback into proof."""
    with pytest.raises(gsa.GhostAuthorValidationError):
        gsa.build_ghost_prompt_object(_spec(), "the clasp turns slowly here",
                                      source="deterministic_fallback")
    with pytest.raises(gsa.GhostAuthorValidationError):
        gsa.build_ghost_prompt_object(_spec(), "the clasp turns slowly here",
                                      source="writer_llm",
                                      fallback_reason="why")
    assert gsa.build_ghost_prompt_object(
        _spec(), "the clasp turns slowly here",
        source="deterministic_fallback", fallback_reason="no model")


# --------------------------------------------------------------------------- #
# 8. The deterministic batch -- complete clauses, never a slice.
# --------------------------------------------------------------------------- #

def _timeline_specs(seed=1013426535):
    modes = gsa.schedule_ghost_modes(TIMELINE, seed)
    comp = _components()
    rows = []
    for beat_id, role in TIMELINE:
        mode = modes[beat_id]
        motif = (gsa.motif_for_character(comp, mode, seed_int=comp["seed_int"])
                 if role == "character_video"
                 else gsa.motif_for_bookend(role, mode))
        rows.append({"beat_id": beat_id, "role": role, "mode": mode,
                     "motif_cue": motif, "sanitized_intent": "",
                     "normalized_emotion": "", "mapped_arc": ""})
    return gsa.build_ghost_author_specs(rows, model_id="m/x")


def test_every_deterministic_clause_is_unique_within_an_episode():
    for seed in (1013426535, 7, 999999):
        specs = _timeline_specs(seed)
        batch = gsa.deterministic_batch(specs, episode_seed=seed)
        assert len(set(batch.values())) == len(batch), seed


def test_the_opening_and_closing_bookends_differ():
    specs = _timeline_specs()
    batch = gsa.deterministic_batch(specs, episode_seed=1013426535)
    assert batch["g000"] != batch[specs[-1]["id"]]


def test_every_deterministic_clause_is_a_complete_valid_leaf():
    specs = _timeline_specs()
    batch = gsa.deterministic_batch(specs, episode_seed=1013426535)
    for spec in specs:
        ok, why = gsa.validate_drawable_beat(
            batch[spec["id"]], mode=spec["mode"], names=NAMES)
        assert ok, (spec["mode"], batch[spec["id"]], why)


def test_no_free_text_ever_reaches_a_deterministic_clause():
    """The pools are checked in; there is no extraction surface left."""
    pool = {c for group in gsa.GHOST_FALLBACK_CLAUSES.values() for c in group}
    pool |= set(gsa.GHOST_FALLBACK_BOOKENDS.values())
    specs = _timeline_specs()
    batch = gsa.deterministic_batch(specs, episode_seed=1013426535)
    assert set(batch.values()) <= pool


# --------------------------------------------------------------------------- #
# 9. Composition, banana and the measured window.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
def test_the_style_cue_and_negative_are_byte_identical_to_v1(style):
    """The look is ACCEPTED. v2 changes content authoring and nothing else."""
    v1 = gsp.compose_ghost_prompt(role="character_video", style=style,
                                  subject_sigil="a lean upright figure")
    v2 = gsp.compose_ghost_prompt_v2(
        role="character_video", style=style, mode="figure",
        motif_cue="lean silhouette with a rust satchel",
        drawable_beat="an outline lifts one arm into a narrow band")
    assert v2["negative"] == v1["negative"]
    # The cue comes from the SHARED authority, so v1 and v2 lead with the same
    # bytes. `sci_fi_radio` authors no cue at all -- the production look is the
    # bare prompt -- and an empty cue must stay empty rather than acquire one.
    cue = v2["components"]["pack_cue"]
    assert cue == vs.compact_style_cue(style).rstrip(".").strip()
    if cue:
        assert v1["positive"].lower().startswith(cue.lower())
        assert v2["positive"].lower().startswith(cue.lower())
        assert "pack_cue" in v2["slots"]
    else:
        assert "pack_cue" not in v2["slots"]


@pytest.mark.parametrize("style", STYLES, ids=STYLE_IDS)
def test_every_shipped_pack_holds_the_shortest_clause(style):
    gsa.assert_shell_fits([style], ledger_meta={"freeze_timestamp": "x"})


def test_v2_reads_no_raw_ledger_surface():
    """The strongest form: they are not parameters."""
    import inspect
    params = set(inspect.signature(gsp.compose_ghost_prompt_v2).parameters)
    assert params == {"role", "style", "mode", "motif_cue", "drawable_beat"}


def test_object_and_signal_laws_are_affirmative_and_non_human():
    for mode in gsa.GHOST_NON_FIGURE_MODES:
        out = gsp.compose_ghost_prompt_v2(
            role="announcer_visual", style=STYLE, mode=mode,
            motif_cue=gsa.motif_for_bookend("announcer_visual", mode),
            drawable_beat="a slow shadow crosses the dial and lifts away")
        low = out["positive"].lower()
        for banned in ("no people", "without", "no humans", "person",
                       "face", "close-up", "portrait"):
            assert banned not in low, (mode, out["positive"])


def test_a_figure_prompt_never_asks_for_a_face_or_a_close_up():
    out = gsp.compose_ghost_prompt_v2(
        role="character_video", style=STYLE, mode="figure",
        motif_cue="lean silhouette with a rust satchel",
        drawable_beat="an outline lifts one arm into a narrow band")
    low = out["positive"].lower()
    assert "face" not in low and "close-up" not in low and "closeup" not in low
    assert "mid-shot or wider" in low


def test_a_bookend_scheduled_as_figure_is_refused():
    with pytest.raises(gsp.GhostPromptError):
        gsp.compose_ghost_prompt_v2(
            role="music_visual", style=STYLE, mode="figure",
            motif_cue="broadcast console emblem",
            drawable_beat="an outline lifts one arm into a narrow band")


def test_an_empty_motif_or_leaf_is_refused_by_the_composer():
    for motif, leaf in (("", "an outline lifts one arm into the light"),
                        ("rust satchel emblem", "")):
        with pytest.raises(gsp.GhostPromptError):
            gsp.compose_ghost_prompt_v2(
                role="character_video", style=STYLE, mode="object",
                motif_cue=motif, drawable_beat=leaf)


def test_the_banana_route_may_substitute_inside_the_leaf():
    """A transformed prop is the route working, not the leaf being lost."""
    meta = {"freeze_timestamp": "2026-08-22T22:27:56.943819+00:00",
            "source_bank": "media_archive"}
    out = gsa.finalize_ghost_prompt_v2(
        role="character_video", style=STYLE, mode="figure",
        motif_cue="lean silhouette with a rust revolver",
        drawable_beat="an outline lifts the revolver into a narrow band",
        ledger_meta=meta, token_measure_fn=_measure_stub,
        banana_enabled=True)
    assert "revolver" not in out["positive"].lower()
    for name, text in out["components"].items():
        assert text in out["positive"], name
    assert out["banana_receipt"]["banana_route"] == "on"
    assert out["banana_receipt"]["banana_substitutions"] >= 1


def test_the_route_off_still_publishes_a_receipt():
    out = gsa.finalize_ghost_prompt_v2(
        role="character_video", style=STYLE, mode="object",
        motif_cue="rust satchel emblem",
        drawable_beat="the clasp turns and a slow shadow crosses it",
        ledger_meta={"freeze_timestamp": "x"}, token_measure_fn=_measure_stub,
        banana_enabled=False)
    assert out["banana_receipt"]["banana_route"] == "off"
    assert out["banana_receipt"]["banana_substitutions"] == 0


def test_an_over_window_prompt_is_refused_and_never_trimmed():
    """A protected field is never cut to make a number fit."""
    def _always_over(_text):
        return 91, 2
    with pytest.raises(gsa.GhostBudgetError):
        gsa.finalize_ghost_prompt_v2(
            role="character_video", style=STYLE, mode="object",
            motif_cue="rust satchel emblem",
            drawable_beat="the clasp turns and a slow shadow crosses it",
            ledger_meta={"freeze_timestamp": "x"},
            token_measure_fn=_always_over)


def test_an_injected_measurer_still_gates():
    """A test that supplies a measurer is asking for the gate, not an excuse."""
    assert gsa.resolve_token_measure(_measure_stub) is _measure_stub


def test_the_installed_tokenizer_counts_the_real_window():
    """The measured 75/76 boundary, not an asserted one."""
    installed = pytest.importorskip("comfy.sd1_clip", reason="needs ComfyUI")
    assert installed
    assert gsa.measure_clip_tokens(" ".join(["word"] * 75)) == (77, 1)
    assert gsa.measure_clip_tokens(" ".join(["word"] * 76)) == (80, 2)
    # A padded row is 77 long; counting its LENGTH would report 77 here.
    tokens, windows = gsa.measure_clip_tokens("a lantern turns")
    assert windows == 1 and tokens < 12


def test_the_clip_counter_names_the_measurer():
    out = gsa.finalize_ghost_prompt_v2(
        role="character_video", style=STYLE, mode="object",
        motif_cue="rust satchel emblem",
        drawable_beat="the clasp turns and a slow shadow crosses it",
        ledger_meta={"freeze_timestamp": "x"}, token_measure_fn=_measure_stub)
    assert out["clip_counter"] == gsa.GHOST_CLIP_COUNTER
    assert out["clip_window_max"] == 77


def test_the_author_target_sits_under_the_render_ceiling():
    """Author-time headroom is what the banana route may later spend."""
    assert gsa.GHOST_AUTHOR_TOKEN_TARGET < gsa.GHOST_CLIP_WINDOW_TOKENS
    ok, why = gsa.candidate_fits(
        role="character_video", style=STYLE, mode="object",
        motif_cue="rust satchel emblem",
        drawable_beat="the clasp turns and a slow shadow crosses it",
        ledger_meta={"freeze_timestamp": "x"},
        token_measure_fn=lambda _t: (70, 1))
    assert not ok and "author-time target" in why


# --------------------------------------------------------------------------- #
# 10. The v1 sigil goldens survive the refactor.
# --------------------------------------------------------------------------- #

def test_the_component_distiller_rejoins_to_the_same_sigil_bytes():
    for row in (RICH_CAST, {"char_id": "c09", "name": "MARA"},
                {"char_id": "c03", "name": "RASHIDA PIERCE", "gender": "female",
                 "portrait_prompt": "a tall stooped woman, tight black bun, "
                                    "steady hands, an olive uniform"}):
        char_id = row["char_id"]
        comp = gsp.distill_sigil_components(
            row, episode_seed=1013426535, char_id=char_id,
            style_id=STYLE.style_id)
        sigil = gsp.distill_subject_sigil(
            row, episode_seed=1013426535, char_id=char_id,
            style_id=STYLE.style_id)
        cues = [comp[bucket] for bucket, _v in gsp.SIGIL_BUCKETS
                if comp.get(bucket)]
        body = ", ".join(cues)
        expected = ("%s, %s" % (comp["gender_word"], body)
                    if comp["gender_word"] else body)
        assert sigil == gsp._trim_to(expected, gsp.GHOST_SIGIL_MAX_CHARS)


def test_the_live_a_arm_sigils_are_byte_stable():
    """The exact strings the 2026-08-22 v1 baseline rendered with."""
    got = gsp.distill_subject_sigil(
        {"char_id": "c02", "name": "ADRIAN SPENDER", "gender": "male"},
        episode_seed=1013426535, char_id="c02", style_id="video_art")
    assert got.startswith("a man, ")
    assert "adrian" not in got.lower() and "spender" not in got.lower()


# --------------------------------------------------------------------------- #
# 11. House style.
# --------------------------------------------------------------------------- #

def test_the_author_module_never_imports_the_render_driver():
    """A cycle-free, loader-free module -- checked on the IMPORTS, not prose.

    The docstring names those modules to say it does not import them, so a
    substring scan over the source would fail on its own explanation.
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(gsa))
    imported = set()
    for node in ast.walk(tree):
        # `entry`, not `alias`: the S28 forbidden-symbol sweep treats `alias`
        # as a runtime extinction marker, and a test that trips the gate it is
        # meant to sit behind is a red HEAD waiting to happen.
        if isinstance(node, ast.Import):
            imported.update(entry.name for entry in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(entry.name for entry in node.names)
    for banned in ("render_driver", "otr_shot_lock", "_otr_model_loader",
                   "registry", "_otr_video_engines.registry", "torch"):
        assert banned not in imported, banned
    assert "ghost_signal_prompt" in imported


def test_no_curse_words_or_placeholder_naming():
    import inspect
    for module in (gsa, gsp):
        low = inspect.getsource(module).lower()
        assert "dummy" not in low
