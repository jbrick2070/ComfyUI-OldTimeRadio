"""Ghost Half B: the batched author RANKS a subject from the closed list.

THE GAP THIS CLOSES. The deterministic tier (`_beat_mentions_object`) reaches a
beat only when its dialogue LITERALLY names one of the episode's key objects --
26.3% of beats, measured. The other ~74% fell to odometer cycling: the subject was
chosen by the beat's position in the episode, not its content. The operator's
2026-09-03 ruling chose the remedy -- extend the ONE batched Ghost author call so
a beat that names nothing still gets a real, photographable subject.

THE RULING'S HARD RULES, and each has a test below:
  * The beat RANKS the episode's own key objects. It never SOURCES a new noun.
  * A noun named only inside a negation is not an artifact.
  * Not a gate: a bad pick costs the pick, never the row or the batch.
  * One model call. No retry over subjects.

THE SHAPE, from the arc. `ghost_subject` is a NEW, OPTIONAL, SIBLING field next to
the closed `ghost_prompt` object -- not inside it. `GHOST_PROMPT_FIELDS`,
`validate_ghost_prompt_object` and both author versions are untouched, so a frozen
replay bundle simply lacks the key, tier 0 no-ops, and the render is byte-identical
to before. That is the whole reason it is a sibling: a version bump would fail every
frozen replay closed over a key it never had.

ENFORCEMENT IS STRUCTURAL, NOT A PROMPT PROMISE. `admit_ghost_subject` is an exact
case-fold membership check against the list the model was shown. An invented noun,
a negated one lifted from the line, a near-miss -- none of them is a legal answer
shape, so none can become the picture.
"""
from __future__ import annotations

import json

import pytest

from nodes._otr_video_engines import ghost_signal_author as gsa
from nodes._otr_video_engines.ghost_signal_author import (
    GhostAuthorParseError,
    admit_ghost_subject,
    build_batch_prompt,
    ghost_subject_candidates,
    parse_batch_response,
    resolve_crux_kernel,
)


META = {
    "key_objects": ["ledger", "pen", "lantern"],
    "story_brief_terms": {"setting": ["the archive"]},
}


# ---------------------------------------------------------------------------
# 1. Admission: the whole enforcement of RANK, never SOURCE
# ---------------------------------------------------------------------------
def test_an_exact_pick_is_admitted_as_the_list_spells_it():
    assert admit_ghost_subject("pen", ["ledger", "pen"]) == "pen"


def test_admission_is_case_and_whitespace_tolerant_but_nothing_looser():
    assert admit_ghost_subject("  Pen ", ["ledger", "pen"]) == "pen"
    assert admit_ghost_subject("PEN", ["ledger", "pen"]) == "pen"
    # A near-miss is not a match. Looser would be a second extractor.
    assert admit_ghost_subject("pens", ["ledger", "pen"]) == ""
    assert admit_ghost_subject("the pen", ["ledger", "pen"]) == ""


def test_an_INVENTED_noun_is_not_a_legal_answer():
    """The model may only hand back a member of the list it was shown."""
    assert admit_ghost_subject("truck", ["ledger", "pen"]) == ""


def test_the_NEGATED_noun_cannot_be_resurrected_through_the_subject():
    """The b002 case that the 2026-09-03 ruling exists to prevent: the line
    says the list 'isn't just some dusty list of truck routes'. Even if the
    model lifts 'truck' from that line and returns it as the subject, it is not
    in key_objects and is dropped."""
    objects = ["ledger", "pen"]
    assert admit_ghost_subject("truck routes", objects) == ""
    assert admit_ghost_subject("truck", objects) == ""


@pytest.mark.parametrize("raw", ["", None, "   ", 42, ["pen"], {"subject": "pen"}])
def test_admission_NEVER_raises_on_garbage(raw):
    assert admit_ghost_subject(raw, ["ledger", "pen"]) == ""


def test_admission_with_no_candidates_admits_nothing():
    assert admit_ghost_subject("pen", []) == ""
    assert admit_ghost_subject("pen", None) == ""


# ---------------------------------------------------------------------------
# 2. The candidate list is the resolver's own extraction
# ---------------------------------------------------------------------------
def test_candidates_are_the_same_terms_the_kernel_resolver_uses():
    """One extraction, shared, so author-time and render-time cannot drift."""
    assert ghost_subject_candidates(META) == ["ledger", "pen", "lantern"]


def test_candidates_are_bounded_by_the_term_limit():
    many = {"key_objects": ["o%d" % i for i in range(20)]}
    assert len(ghost_subject_candidates(many)) == gsa.GHOST_V3_TERM_LIMIT


def test_no_key_objects_means_no_candidates():
    assert ghost_subject_candidates({}) == []
    assert ghost_subject_candidates({"key_objects": []}) == []
    assert ghost_subject_candidates(None) == []


# ---------------------------------------------------------------------------
# 3. The prompt: spliced ONCE, and byte-identical when there is nothing to rank
# ---------------------------------------------------------------------------
SPECS = [
    {"id": "g000", "mode": "object", "motif_cue": "a brass key",
     "sanitized_intent": "opens the drawer"},
    {"id": "g001", "mode": "signal", "motif_cue": "a radio dial"},
]


def test_no_candidates_means_a_byte_identical_prompt():
    """A brief-failed episode's call must not change at all."""
    assert build_batch_prompt(SPECS) == build_batch_prompt(SPECS, subject_candidates=())
    assert gsa.GHOST_BATCH_SUBJECT_HEADER not in build_batch_prompt(SPECS)
    _head, shots = build_batch_prompt(SPECS).split(gsa.GHOST_BATCH_HEADER, 1)
    assert "subject" not in shots.lower()


def test_candidates_are_spliced_ONCE_above_the_shots():
    # "pen" is a substring of "opens" in the shot lines; use candidates that
    # cannot collide, so the assertion measures the splice and not a coincidence.
    prompt = build_batch_prompt(SPECS, subject_candidates=["ledger", "lantern"])
    assert prompt.count(gsa.GHOST_BATCH_SUBJECT_HEADER) == 1
    assert "%s ledger, lantern" % gsa.GHOST_BATCH_SUBJECT_HEADER in prompt
    # Once per episode, not per shot: the list appears before SHOTS: and the
    # per-shot lines carry no subject text.
    head, shots = prompt.split(gsa.GHOST_BATCH_HEADER, 1)
    assert gsa.GHOST_BATCH_SUBJECT_HEADER in head
    assert "ledger" not in shots and "lantern" not in shots


def test_the_rule_asks_for_a_CHOICE_not_a_noun():
    prompt = build_batch_prompt(SPECS, subject_candidates=["ledger"])
    assert "from SUBJECTS" in prompt
    assert "copied exactly as written" in prompt
    assert "Leave the key out" in prompt
    # SHOULD-FIX 2: it must not contradict rule 1's "exactly this shape".
    assert "extends the shape in rule 1" in prompt


# ---------------------------------------------------------------------------
# 4. The parser: an optional key that is optional only when asked for
# ---------------------------------------------------------------------------
def _envelope(rows):
    return json.dumps({"shots": rows})


def test_a_subject_is_collected_RAW_beside_the_leaf():
    subjects = {}
    leaves = parse_batch_response(_envelope([
        {"id": "g000", "drawable_beat": "a brass key turns in the drawer lock",
         "subject": "Ledger"},
        {"id": "g001", "drawable_beat": "a radio dial glows and slowly turns"},
    ]), ["g000", "g001"], subjects_out=subjects)
    assert set(leaves) == {"g000", "g001"}
    # RAW: not admitted here. Admission is the caller's job against the live
    # list, so case is preserved exactly as the model wrote it.
    assert subjects == {"g000": "Ledger"}


def test_a_row_without_a_subject_is_exactly_as_valid_as_before():
    subjects = {}
    leaves = parse_batch_response(_envelope([
        {"id": "g000", "drawable_beat": "a brass key turns in the drawer lock"},
    ]), ["g000"], subjects_out=subjects)
    assert "g000" in leaves and subjects == {}


def test_WITHOUT_subjects_out_a_subject_key_is_still_an_extra_field():
    """Every pre-Half-B caller and test keeps the strict shape."""
    with pytest.raises(GhostAuthorParseError, match="exactly id"):
        parse_batch_response(_envelope([
            {"id": "g000", "drawable_beat": "a brass key turns in the lock",
             "subject": "ledger"},
        ]), ["g000"])


def test_a_NON_STRING_subject_costs_the_hint_never_the_row():
    """Not a gate."""
    subjects = {}
    leaves = parse_batch_response(_envelope([
        {"id": "g000", "drawable_beat": "a brass key turns in the drawer lock",
         "subject": 42},
    ]), ["g000"], subjects_out=subjects)
    assert "g000" in leaves
    assert subjects == {}


def test_an_EMPTY_subject_is_ignored():
    subjects = {}
    parse_batch_response(_envelope([
        {"id": "g000", "drawable_beat": "a brass key turns in the drawer lock",
         "subject": "   "},
    ]), ["g000"], subjects_out=subjects)
    assert subjects == {}


def test_a_DUPLICATED_subject_key_is_dropped_not_fatal():
    """MUST-FIX round 2. `_object_pairs_hook` rejected ANY repeated key before
    the optional-field handling ever ran, so two subject keys on one row
    rejected the whole batch and burned the informed retry -- a gate. Two
    picks for one row is ambiguous, and the honest answer is no pick."""
    raw = '{"shots": [{"id": "g000", "drawable_beat": "a brass key turns in the drawer lock", "subject": "ledger", "subject": "pen"}]}'
    subjects = {}
    leaves = parse_batch_response(raw, ["g000"], subjects_out=subjects)
    assert "g000" in leaves
    assert subjects == {}, "an ambiguous pick must yield no pick"


def test_a_duplicated_LEAF_key_is_still_fatal():
    """Scope check: the tolerance is for the optional hint only."""
    raw = '{"shots": [{"id": "g000", "drawable_beat": "a", "drawable_beat": "b"}]}'
    with pytest.raises(GhostAuthorParseError, match="repeats the key"):
        parse_batch_response(raw, ["g000"], subjects_out={})


def test_a_duplicated_ID_key_is_still_fatal():
    raw = '{"shots": [{"id": "g000", "id": "g001", "drawable_beat": "a brass key turns in the drawer lock"}]}'
    with pytest.raises(GhostAuthorParseError, match="repeats the key"):
        parse_batch_response(raw, ["g000", "g001"], subjects_out={})


def test_the_leaf_contract_is_still_strict_with_subjects_enabled():
    """Loosening the key check for one optional key must not loosen anything
    else -- a missing drawable_beat is still fatal to the batch."""
    with pytest.raises(GhostAuthorParseError):
        parse_batch_response(_envelope([{"id": "g000", "subject": "ledger"}]),
                             ["g000"], subjects_out={})


# ---------------------------------------------------------------------------
# 5. Tier 0 in the kernel resolver
# ---------------------------------------------------------------------------
def test_an_ADMITTED_subject_is_tier_0():
    kernel, source = resolve_crux_kernel(
        META, ordinal=0, role="character_video", mode="object",
        beat_text="she said nothing at all", authored_subject="lantern")
    assert source == "authored_subject"
    assert kernel.startswith("lantern")


def test_a_CLEAN_beat_reference_outranks_the_authored_pick():
    """THE RULING'S ORDER. "The physical artifacts in the story, and ESPECIALLY
    IF REFERRED TO IN THE BEAT." The beat names 'ledger'; the author ranked
    'pen' from intent alone. The dialogue wins. The first cut of Half B had
    this backwards and asserted the inversion in this very test -- the round-2
    contrarian caught it against the ruling's own words."""
    kernel, source = resolve_crux_kernel(
        META, ordinal=0, role="character_video", mode="object",
        beat_text="she opened the ledger", authored_subject="pen")
    assert source == "key_object_in_beat"
    assert kernel.startswith("ledger")


def test_the_authored_pick_fills_the_gap_the_dialogue_leaves():
    """The 74% case: the beat names nothing, the author ranked 'pen', and
    that beats the odometer."""
    kernel, source = resolve_crux_kernel(
        META, ordinal=0, role="character_video", mode="object",
        beat_text="she waited and said nothing", authored_subject="pen")
    assert source == "authored_subject"
    assert kernel.startswith("pen")


def test_an_ABSENT_subject_falls_through_byte_identically():
    """Every pre-Half-B row. The output must be exactly what it was."""
    before = resolve_crux_kernel(
        META, ordinal=2, role="character_video", mode="object",
        beat_text="she opened the ledger")
    after = resolve_crux_kernel(
        META, ordinal=2, role="character_video", mode="object",
        beat_text="she opened the ledger", authored_subject="")
    assert before == after
    assert after[1] == "key_object_in_beat"


def test_an_OFF_LIST_subject_is_re_validated_against_the_LIVE_list():
    """A key_objects list edited after casting cannot leave a stale pick
    pointing at a noun the brief no longer lists."""
    kernel, source = resolve_crux_kernel(
        META, ordinal=1, role="character_video", mode="object",
        beat_text="", authored_subject="truck")
    assert source == "key_object"
    assert "truck" not in kernel


def test_tier_0_is_re_validated_case_insensitively():
    kernel, source = resolve_crux_kernel(
        META, ordinal=0, role="character_video", mode="object",
        authored_subject="LANTERN")
    assert source == "authored_subject"
    assert kernel.startswith("lantern")


def test_a_BOOKEND_keeps_the_radio_even_with_a_subject():
    """The radio stays on the announcer and music beds by operator rule; an
    authored subject on a bookend is ignored."""
    kernel, source = resolve_crux_kernel(
        META, ordinal=0, role="announcer_visual", mode="object",
        authored_subject="lantern")
    assert source == "bookend_radio"
    assert "lantern" not in kernel


def test_no_key_objects_means_tier_0_cannot_fire():
    """No list, nothing to be a member of."""
    kernel, source = resolve_crux_kernel(
        {"story_brief_terms": {"setting": ["the archive"]}},
        ordinal=0, role="character_video", mode="object",
        authored_subject="lantern")
    assert source == "setting"


# ---------------------------------------------------------------------------
# 6. The schema and the wiring
# ---------------------------------------------------------------------------
def test_ShotRow_declares_ghost_subject_as_optional_and_absent_by_default():
    from nodes._otr_video_engines.schemas import ShotRow

    assert "ghost_subject" in ShotRow.model_fields
    assert ShotRow(shot_id="s1").ghost_subject is None
    assert ShotRow(shot_id="s1", ghost_subject="pen").ghost_subject == "pen"


def test_the_closed_prompt_object_is_UNTOUCHED():
    """The whole reason it is a sibling field."""
    assert "ghost_subject" not in gsa.GHOST_PROMPT_FIELDS
    assert "subject" not in gsa.GHOST_PROMPT_FIELDS


def test_the_render_driver_hands_the_stored_subject_to_the_resolver():
    """A correct tier nothing calls is this repo's most repeated defect."""
    import inspect

    from nodes._otr_video_engines import render_driver

    src = inspect.getsource(render_driver)
    assert 'authored_subject=str(shot.get("ghost_subject") or "")' in src


def test_shot_lock_offers_candidates_and_admits_before_storing():
    import inspect

    from nodes import otr_shot_lock

    gen_src = inspect.getsource(otr_shot_lock._ghost_generate_batch)
    assert "ghost_subject_candidates(meta)" in gen_src
    assert "admit_ghost_subject(raw_pick, candidates)" in gen_src
    plan_src = inspect.getsource(otr_shot_lock.build_execution_plan)
    assert 'shots[-1]["ghost_subject"]' in plan_src

# ---------------------------------------------------------------------------
# 7. What the contrarian pass changed -- each was a real defect in the first cut
# ---------------------------------------------------------------------------
def test_the_hashed_per_shot_budget_is_UNCHANGED():
    """MUST-FIX 1. `_template_identity()` hashes GHOST_BATCH_PER_SHOT_TOKENS, so
    raising it would change every stored leaf's request hash and re-author
    every existing episode. The first cut did exactly that."""
    assert gsa.GHOST_BATCH_PER_SHOT_TOKENS == 48
    assert gsa.GHOST_BATCH_SUBJECT_TOKENS > 0


def test_the_subject_room_is_NOT_in_the_template_identity():
    """The whole point of a separate constant."""
    import inspect

    src = inspect.getsource(gsa._template_identity)
    assert "GHOST_BATCH_SUBJECT_TOKENS" not in src, (
        "the subject allowance is hashed; every pre-Half-B leaf will re-author")
    assert "GHOST_BATCH_PER_SHOT_TOKENS" in src, (
        "premise broken: the per-shot budget is no longer hashed at all")


def test_the_budget_grows_only_for_offered_rows():
    base = gsa.batch_output_tokens(5)
    assert gsa.batch_output_tokens(5, subject_rows=0) == base
    assert (gsa.batch_output_tokens(5, subject_rows=3)
            == base + 3 * gsa.GHOST_BATCH_SUBJECT_TOKENS)


def test_a_NEGATED_pick_is_refused_by_tier_0():
    """SHOULD-FIX 1. Membership proves the noun is the episode's own; only the
    beat text can say the beat disowns it. 'There is no truck here' ranked as
    truck is the b002 defect wearing the author's clothes."""
    meta = {"key_objects": ["truck", "ledger"],
            "story_brief_terms": {"setting": ["the yard"]}}
    # "isn't" is in _CRUX_NEGATION_CUES. NOTE, recorded rather than fixed here:
    # the bare "there is no X" phrasing is NOT in that list, so the guard --
    # the deterministic tier's, shipped this morning -- does not read it as a
    # negation. Widening the cue list changes which beats the LEXICAL tier
    # ranks and is its own change with its own blast radius.
    text = "it isn't a truck at all, only the ledger"
    # What tier 0 controls is ONE thing: it must not admit the negated pick.
    assert gsa._beat_object_stance(text, "truck") == "negated"
    kernel, source = resolve_crux_kernel(
        meta, ordinal=0, role="character_video", mode="object",
        beat_text=text, authored_subject="truck")
    assert source != "authored_subject", (
        "tier 0 admitted a pick the beat explicitly negates")
    # What happens NEXT is the deterministic ladder's business, not tier 0's.
    # (At ordinal 0 the odometer picks objects[0] regardless of the beat --
    # pre-existing behaviour, and asserting on it here would test the wrong
    # tier. The first draft of this test did exactly that.)


def test_an_ABSENT_pick_is_still_admitted():
    """The 74% case, and the whole reason tier 0 exists: the beat never names
    the object, the author ranked it anyway, and that is correct."""
    kernel, source = resolve_crux_kernel(
        META, ordinal=0, role="character_video", mode="object",
        beat_text="she waited a long time and said nothing",
        authored_subject="lantern")
    assert source == "authored_subject"


def test_the_stance_helper_distinguishes_all_three_states():
    stance = gsa._beat_object_stance
    assert stance("she opened the ledger", "ledger") == "clean"
    assert stance("it is not a ledger", "ledger") == "negated"
    assert stance("she said nothing", "ledger") == "absent"
    assert stance("", "ledger") == "absent"
    # The boolean contract is unchanged for every existing caller.
    assert gsa._beat_mentions_object("she opened the ledger", "ledger") is True
    assert gsa._beat_mentions_object("it is not a ledger", "ledger") is False
    assert gsa._beat_mentions_object("she said nothing", "ledger") is False


def test_the_subject_rule_reconciles_with_rule_1():
    """SHOULD-FIX 2: rule 1 said 'exactly this shape'; the subject rule now
    says it EXTENDS that shape rather than contradicting it."""
    assert "extends the shape in rule 1" in gsa.GHOST_BATCH_SUBJECT_RULE
    assert '"subject": "..."' in gsa.GHOST_BATCH_SUBJECT_RULE


def test_resolve_crux_kernel_delegates_to_the_shared_extraction():
    """OPTIONAL, taken: one extraction authority, so author and render cannot
    drift."""
    import inspect

    src = inspect.getsource(resolve_crux_kernel)
    assert "ghost_subject_candidates(meta)" in src


def test_the_PREFLIGHT_shot_carries_the_same_subject_as_the_row():
    """MUST-FIX 3. `build_request_from_shot` owns the v3 branch, so the
    cast-time preflight DOES reach the kernel resolver. A preflight shot
    without the subject would resolve a different kernel than the durable row."""
    import inspect

    from nodes import otr_shot_lock

    pre = inspect.getsource(otr_shot_lock._assert_family_inputs_satisfiable_cast_time)
    assert "ghost_subjects=None" in pre
    assert 'shot["ghost_subject"]' in pre
    plan = inspect.getsource(otr_shot_lock.build_execution_plan)
    # Kernel parity (2026-09-12) added a keyword after the subjects; the call
    # is now two lines and the second one is what proves the ordinal reaches
    # the preflight.
    assert ("ghost_prompts, ghost_subjects," in plan
            and "planned_ordinal=planned_ordinal" in plan), (
        "the plan builder no longer hands the preflight the subjects")


def test_an_EMPTY_candidate_list_never_hands_the_parser_None():
    """MUST-FIX 4. With None, a stray subject key from the model was an EXTRA
    FIELD that rejected the whole batch and burned the retry -- a gate by
    accident on exactly the episodes the feature should not touch."""
    import inspect

    from nodes import otr_shot_lock

    src = inspect.getsource(otr_shot_lock._ghost_generate_batch)
    assert "raw_subjects = {} if subjects_out is not None else None" in src
    assert "raw_subjects = {} if candidates else None" not in src


def test_a_stray_subject_with_NO_candidates_costs_nothing():
    """The behavioural half of MUST-FIX 4: the parser accepts it, and admission
    against an empty list drops it."""
    subjects = {}
    leaves = parse_batch_response(_envelope([
        {"id": "g000", "drawable_beat": "a brass key turns in the drawer lock",
         "subject": "ledger"},
    ]), ["g000"], subjects_out=subjects)
    assert "g000" in leaves
    assert admit_ghost_subject(subjects.get("g000"), []) == ""


def test_the_budget_covers_EVERY_offered_row():
    """SHOULD-FIX round 2. The prompt lets any row answer and rows carry no
    role flag, so bookend answers consume tokens whether or not they are kept.
    Budgeting only the rows we keep truncated batches."""
    import inspect

    from nodes import otr_shot_lock

    src = inspect.getsource(otr_shot_lock._ghost_generate_batch)
    assert "subject_rows = len(specs) if candidates else 0" in src


def test_the_admission_metric_reports_three_numbers():
    """SHOULD-FIX round 2. One ratio could not tell "the model invents" from
    "the model abstains", and counted bookend hints discarded on purpose."""
    import inspect

    from nodes import otr_shot_lock

    src = inspect.getsource(otr_shot_lock._ghost_generate_batch)
    assert "eligible=%d submitted=%d" in src and "admitted=%d" in src


def test_bookend_picks_are_DROPPED_at_admission():
    """CUT 1. The resolver returns the radio before it looks at a subject, so
    a pick on an announcer row is data the renderer ignores."""
    import inspect

    from nodes import otr_shot_lock

    src = inspect.getsource(otr_shot_lock._ghost_generate_batch)
    assert 'role_by_id.get(shot_id) != "character_video"' in src

# ---------------------------------------------------------------------------
# 8. Round 3 -- both reproduced by execution before they were fixed
# ---------------------------------------------------------------------------
def test_a_duplicated_subject_on_the_ENVELOPE_is_still_rejected():
    """MUST-FIX round 3. The tolerance lived in the pairs hook, which runs on
    every object -- so a duplicated subject on the ENVELOPE was swallowed and
    the "exactly one key 'shots'" check never saw it. The hook now preserves
    the duplicate as a sentinel; the envelope check sees an extra key."""
    raw = ('{"shots": [{"id": "g000", "drawable_beat": "a brass key turns in '
           'the drawer lock"}], "subject": "ledger", "subject": "clock"}')
    with pytest.raises(GhostAuthorParseError, match="exactly one key"):
        parse_batch_response(raw, ["g000"], subjects_out={})
    with pytest.raises(GhostAuthorParseError, match="exactly one key"):
        parse_batch_response(raw, ["g000"])


def test_a_duplicated_subject_on_a_LEGACY_row_is_still_rejected():
    """The other half of MUST-FIX round 3. With subjects_out=None a single
    subject key is an extra field and rejects; a DUPLICATED one used to slip
    past that check because the hook had already removed it. Now the sentinel
    stays in the row and the strict field set rejects it, same as a single."""
    raw = ('{"shots": [{"id": "g000", "drawable_beat": "a brass key turns in '
           'the drawer lock", "subject": "ledger", "subject": "clock"}]}')
    with pytest.raises(GhostAuthorParseError, match="exactly id"):
        parse_batch_response(raw, ["g000"])


def test_an_ambiguous_subject_is_dropped_ONLY_on_an_opted_in_row():
    """The tolerance, scoped to where it belongs."""
    raw = ('{"shots": [{"id": "g000", "drawable_beat": "a brass key turns in '
           'the drawer lock", "subject": "ledger", "subject": "clock"}]}')
    subjects = {}
    leaves = parse_batch_response(raw, ["g000"], subjects_out=subjects)
    assert "g000" in leaves
    assert subjects == {}
    assert gsa.AMBIGUOUS_SUBJECT not in subjects.values()


def test_the_sentinel_is_never_a_string():
    """A sentinel that could be mistaken for a pick would be a new way to
    source a noun."""
    assert not isinstance(gsa.AMBIGUOUS_SUBJECT, str)
    assert admit_ghost_subject(gsa.AMBIGUOUS_SUBJECT, ["ledger"]) == ""


def test_the_AUTHORED_pick_breaks_a_tie_among_clean_references():
    """MUST-FIX round 3, reproduced verbatim. The dialogue cleanly names BOTH
    candidates; the author ranked the clock. List order used to draw the
    ledger because it came first. The ruling calls reference a strong
    PREFERENCE, not a first-list-entry rule."""
    meta = {"key_objects": ["handwritten municipal ledger", "brass alarm clock"],
            "story_brief_terms": {"setting": ["the archive"]}}
    kernel, source = resolve_crux_kernel(
        meta, ordinal=0, role="character_video", mode="object",
        beat_text="the ledger can wait. stop the clock before the bell rings.",
        authored_subject="brass alarm clock")
    assert source == "key_object_in_beat"
    assert kernel.startswith("brass alarm clock"), kernel


def test_the_first_reference_still_wins_when_the_author_is_NOT_referenced():
    """The tie-break must not become "the author always wins"."""
    meta = {"key_objects": ["handwritten municipal ledger", "brass alarm clock",
                            "iron lantern"],
            "story_brief_terms": {"setting": ["the archive"]}}
    kernel, source = resolve_crux_kernel(
        meta, ordinal=0, role="character_video", mode="object",
        beat_text="the ledger can wait. stop the clock before the bell rings.",
        authored_subject="iron lantern")
    assert source == "key_object_in_beat"
    assert kernel.startswith("handwritten municipal ledger"), kernel


def test_the_tie_break_is_a_no_op_with_NO_authored_subject():
    """Byte-identical to a074f56e on every pre-Half-B row."""
    meta = {"key_objects": ["handwritten municipal ledger", "brass alarm clock"],
            "story_brief_terms": {"setting": ["the archive"]}}
    text = "the ledger can wait. stop the clock before the bell rings."
    before = resolve_crux_kernel(meta, ordinal=0, role="character_video",
                                 mode="object", beat_text=text)
    after = resolve_crux_kernel(meta, ordinal=0, role="character_video",
                                mode="object", beat_text=text,
                                authored_subject="")
    assert before == after
    assert before[0].startswith("handwritten municipal ledger")

# ---------------------------------------------------------------------------
# 9. Round 4 -- a case-fold collision, executed by the contrarian before fixing
# ---------------------------------------------------------------------------
def test_a_CASE_FOLD_COLLISION_cannot_return_the_negated_twin():
    """MUST-FIX round 4. "waﬄe" (ligature) and "waffle" casefold to the same
    string. `admit_ghost_subject("waffle")` returns the FIRST fold-match -- the
    ligature -- and a second case-fold comparison in the tie-break then
    accepted it against the clean reference "waffle", returning the candidate
    the beat NEGATES. Exact membership fixes it."""
    meta = {"key_objects": ["waﬄe", "waffle"],
            "story_brief_terms": {"setting": ["the kitchen"]}}
    kernel, source = resolve_crux_kernel(
        meta, ordinal=0, role="character_video", mode="object",
        beat_text="it isn't a waﬄe. it is a waffle.",
        authored_subject="waffle")
    assert source == "key_object_in_beat"
    assert kernel.startswith("waffle"), kernel
    assert not kernel.startswith("waﬄe"), (
        "the fold-collided, negated twin was returned")

def test_admission_prefers_the_EXACT_spelling_over_a_fold_twin():
    """The ROOT of the round-4 collision. `admit_ghost_subject` returned the
    FIRST fold-match, so a model that wrote the plain "waffle" got the ligature
    twin back -- and at the authored tier the negation guard then judged the
    twin's spelling rather than the one the beat disowned."""
    assert admit_ghost_subject("waffle", ["waﬄe", "waffle"]) == "waffle"
    assert admit_ghost_subject("waﬄe", ["waﬄe", "waffle"]) == "waﬄe"
    # The fold is still the fallback when there is no exact spelling.
    assert admit_ghost_subject("WAFFLE", ["waﬄe", "waffle"]) in ("waﬄe", "waffle")
    assert admit_ghost_subject("Pen", ["ledger", "pen"]) == "pen"


def test_the_authored_tier_cannot_admit_a_fold_twin_of_a_NEGATED_object():
    """Same collision, one tier over: the beat negates the plain spelling and
    the author wrote the plain spelling. With first-fold-match admission the
    ligature twin came back, its stance read "absent", and a disowned object
    was drawn."""
    meta = {"key_objects": ["waﬄe", "waffle"],
            "story_brief_terms": {"setting": ["the kitchen"]}}
    kernel, source = resolve_crux_kernel(
        meta, ordinal=0, role="character_video", mode="object",
        beat_text="it isn't a waffle at all",
        authored_subject="waffle")
    assert source != "authored_subject", (
        "the authored tier admitted a twin of an object the beat negates")
