"""Bug Bible 11.61 -- ONE naming authority, enforced at the boundary.

The defect: an upstream pass invents entity names, a downstream assigner
overrides them, and the per-record prompt is handed BOTH with no precedence, so
the model writes a description about somebody else. Measured live on this
archive at 46 unique rows across 30 episodes, 67 of 70 occurrences also copied
into the portrait prompt.

The Bug Bible's regression file records that ``11.61`` *"has no executable
assertion YET, deliberately"* and names verify step **(6)** -- the per-record
prompt builder must receive RECONCILED upstream text -- as the one statically
checkable half, blocked until the guard exists. The guard now exists, so
:func:`test_verify6_raw_brief_never_reaches_the_prompt_builder` is that
assertion.

Hermetic: no GPU, no ComfyUI import, every LLM call is a canned stub.

WHAT "GOES RED" MEANS HERE. Bible verify (5) demands the guard be proved by
DISABLING reconciliation and requiring the sweep to go red. Red means **a
non-empty guard finding**, never an episode exception -- the operator's standing
rule is that no mechanism in this item may block, reject, retire or fail an
episode, so a test that asserted a raise would be asserting a bug.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _otr_casting as _OTRC  # noqa: E402
import _otr_name_authority as _NA  # noqa: E402


# The real shape, taken from the live ledger this bug was reported from
# (signal_lost_rivers_embrace_20260817_233013): the pitch named two people, the
# pool assigned two different ones, and BOTH rows described the pitch's people.
UPSTREAM = ["CAPTAIN JONAS REED", "MARTHA HONEYWELL"]
REAL_BRIEF = (
    "We need a seasoned, gruff yet caring actor for CAPTAIN JONAS REED, who "
    "speaks with a weathered authority and carries the weight of the river and "
    "the clock. For MARTHA HONEYWELL, we're looking for someone who can convey "
    "determination, curiosity, and a hint of eccentricity in her speech."
)


class PromptCapturingStub:
    """Canned generator that records every prompt it is handed.

    ``echo_identities`` makes it paste an upstream name into its own output when
    it can still see one -- that is what a real model does with this prompt, and
    it is how the mutation test proves the guard fires rather than proving the
    generator behaved.
    """

    def __init__(self, echo_identities=(), description=None):
        self.prompts: list[str] = []
        self.echo_identities = list(echo_identities)
        self.description = description
        self.calls = 0

    def __call__(self, messages, *, temperature=0.0, max_new_tokens=0, **kwargs):
        self.calls += 1
        text = "\n".join(m.get("content", "") for m in messages)
        self.prompts.append(text)
        desc = self.description or "40s, harbour pilot. Face: square jaw, level brow."
        for identity in self.echo_identities:
            if identity.lower() in text.lower():
                desc = f"40s, {identity}. Face: square jaw, level brow."
                break
        return json.dumps({
            "character_description": desc,
            "speech_signature": "even, unhurried delivery",
        })


def _lock(stub, *, upstream=UPSTREAM, brief=REAL_BRIEF, **kw):
    import random
    return _OTRC.lock_cast(
        creative_fn=stub,
        num_characters=2,
        news_seed="a river ferry stalls at dusk",
        style="radio drama",
        rng=random.Random(1234),
        cast_seed=1234,
        force_lemmy=False,
        casting_brief=brief,
        upstream_identity_names=upstream,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Bible verify (6) -- the static boundary assertion the Bible was waiting for
# --------------------------------------------------------------------------- #

def test_verify6_raw_brief_never_reaches_the_prompt_builder():
    """The prompt builder receives RECONCILED text, never the raw brief.

    Walks to the real call site rather than trusting the helper: every prompt
    the generator was handed is inspected for every upstream identity and every
    surface form of it. This is the guard that stops a later refactor quietly restoring
    the two-authority prompt.
    """
    stub = PromptCapturingStub()
    _lock(stub)

    assert stub.prompts, "no prompt was captured -- the test proved nothing"
    for prompt in stub.prompts:
        for identity in UPSTREAM:
            # Empty roster ON PURPOSE: production suppresses forms the roster
            # owns, so passing [] makes this test STRICTER than production and
            # it can never miss a leak by inheriting a suppression.
            for surface_form in _NA.identity_aliases(identity, []):
                if len(surface_form) < 4:
                    continue
                assert surface_form.lower() not in prompt.lower(), (
                    f"superseded identity form {surface_form!r} reached the "
                    f"per-record prompt -- 11.61 verify (6) violated"
                )


def test_the_reconciled_brief_is_still_present_and_distinct():
    """Reconciliation must not gut the brief, and must keep people apart.

    A single shared placeholder for every identity would destroy which traits
    belong to whom and invite a blended description -- the same wrong-person
    failure wearing new clothes.
    """
    stub = PromptCapturingStub()
    _lock(stub)
    joined = "\n".join(stub.prompts)
    assert "CHARACTER A" in joined
    assert "CHARACTER B" in joined
    assert "weathered authority" in joined, "the brief's substance was lost"


# --------------------------------------------------------------------------- #
# Bible verify (5) -- disable the reconciliation, require a finding
# --------------------------------------------------------------------------- #

def test_guard_goes_red_when_reconciliation_is_disabled(monkeypatch):
    """Neuter reconciliation only; the guard must NAME the row and the surface.

    Without this, the cheap check gets adopted and the defect keeps shipping
    under a green gate -- 11.61 verify (3)/(5).
    """
    monkeypatch.setattr(_OTRC._NAMES, "reconcile_text",
                        lambda text, ids, roster: (text, {}))
    stub = PromptCapturingStub(echo_identities=UPSTREAM)
    cast, meta = _lock(stub)

    events = meta["name_authority"]["events"]
    assert events, "reconciliation was disabled and the guard stayed silent"
    for event in events:
        assert event["name"], "a finding that cannot name the row cannot repair it"
        assert event["surfaces"], "a finding must name the contaminated surface"
        assert event["rung"] in {"regenerated", "floor"}

    # And the episode still succeeded: red is a FINDING, never an exception.
    assert any(r["name"] == "ANNOUNCER" for r in cast), "ANNOUNCER row missing"
    assert len([r for r in cast if r["name"] != "ANNOUNCER"]) == 2, (
        "both open slots should still be filled after the guard fired"
    )
    for row in cast:
        assert row["character_description"].strip()


def test_no_contaminated_row_survives_into_the_locked_cast(monkeypatch):
    """The row the guard fired on must not still describe the other person."""
    monkeypatch.setattr(_OTRC._NAMES, "reconcile_text",
                        lambda text, ids, roster: (text, {}))
    stub = PromptCapturingStub(echo_identities=UPSTREAM)
    cast, _meta = _lock(stub)

    roster = [r["name"] for r in cast]
    sup = _NA.superseded_identities(UPSTREAM, roster)
    # Without this the test is VACUOUS: find_foreign_identities returns [] for
    # an empty identity list, so a broken superseded_identities would make every
    # assertion below pass while checking nothing.
    assert sup, "superseded_identities returned [] -- this test proves nothing"
    for row in cast:
        found = _NA.find_foreign_identities(
            {"character_description": row["character_description"],
             "speech_signature": row.get("speech_signature", "")},
            sup, roster,
        )
        assert not found, (
            f"row {row['name']} shipped carrying {[f.matched for f in found]}"
        )


# --------------------------------------------------------------------------- #
# The operator's rule: this mechanism may never fail an episode
# --------------------------------------------------------------------------- #

def test_a_generator_that_always_contaminates_still_yields_a_full_cast(monkeypatch):
    """Even when regeneration is ALSO contaminated, the episode completes.

    This drives the deterministic floor, which exists precisely so that no
    contaminated row can turn into a failed render.
    """
    monkeypatch.setattr(_OTRC._NAMES, "reconcile_text",
                        lambda text, ids, roster: (text, {}))

    class AlwaysDirty(PromptCapturingStub):
        def __call__(self, messages, **kwargs):
            super().__call__(messages, **kwargs)
            return json.dumps({
                "character_description": "40s, CAPTAIN JONAS REED. Face: square jaw.",
                "speech_signature": "MARTHA HONEYWELL speaks softly",
            })

    stub = AlwaysDirty()
    cast, meta = _lock(stub)

    assert [e["rung"] for e in meta["name_authority"]["events"]].count("floor") >= 1
    roster = [r["name"] for r in cast]
    sup = _NA.superseded_identities(UPSTREAM, roster)
    for row in cast:
        assert row["character_description"].strip()
        assert not _NA.find_foreign_identities(
            {"d": row["character_description"], "s": row.get("speech_signature", "")},
            sup, roster,
        ), "the deterministic floor is itself contaminated"


def test_a_raising_generator_on_the_retry_still_completes(monkeypatch):
    """A crash inside the clean-room retry falls to the floor, never upward."""
    monkeypatch.setattr(_OTRC._NAMES, "reconcile_text",
                        lambda text, ids, roster: (text, {}))
    state = {"n": 0}

    def flaky(messages, **kwargs):
        state["n"] += 1
        # Raise ONLY on the clean-room retry, identified by its empty
        # prior_cast (no "Cast so far" block). Raising on the PRIMARY call is a
        # different contract -- lock_cast legitimately surfaces that as
        # CastingFailedError, and this guard was never meant to swallow it.
        # Calls alternate primary, retry, primary, retry: reconciliation is
        # neutered above, so EVERY primary response is contaminated and every
        # one is followed by exactly one clean-room retry. Raising on the even
        # calls therefore breaks the RETRY only -- raising on a primary is a
        # different contract that lock_cast is entitled to surface as
        # CastingFailedError, and this guard never claimed to swallow it.
        if state["n"] % 2 == 0:
            raise RuntimeError("loader exploded")
        return json.dumps({
            "character_description": "40s, CAPTAIN JONAS REED. Face: square jaw.",
            "speech_signature": "even delivery",
        })

    cast, meta = _lock(flaky)
    assert any(e["rung"] == "floor" for e in meta["name_authority"]["events"])
    assert all(r["character_description"].strip() for r in cast)


# --------------------------------------------------------------------------- #
# Lanes that supply nothing must be byte-identical
# --------------------------------------------------------------------------- #

def test_no_upstream_identities_leaves_the_prompt_untouched():
    """Every lane without a structured identity list keeps its exact prompt."""
    with_none = PromptCapturingStub()
    _lock(with_none, upstream=None)
    assert "CAPTAIN JONAS REED" in "\n".join(with_none.prompts), (
        "the brief was altered on a lane that supplied no identities"
    )


def test_writer_merges_structured_identity_surfaces_and_dedupes_in_order():
    """The lane-neutral media surface joins, but never mines, legacy cast data."""
    from nodes import OTR_LedgerScriptWriter as writer

    meta = {
        "source_meta": {
            "selected_concept": {
                "cast": [
                    {"name": "Marta Vale"},
                    {"name": "Gil Neri"},
                    {"role": "unnamed archivist"},
                ],
            },
        },
        "news": {
            "casting_brief": "Dr. Amelia Hartley consults a Film Historian.",
            "upstream_identity_names": [
                " marta vale ", "Dr. Amelia Hartley", "GIL NERI",
            ],
        },
    }
    assert writer._upstream_identity_names(meta) == [
        "Marta Vale", "Gil Neri", "Dr. Amelia Hartley",
    ]

    # Structured-only is load-bearing: a title-cased occupation in prose must
    # not become a guessed person when the optional list is absent.
    assert writer._upstream_identity_names({
        "news": {
            "casting_brief": (
                "Dr. Amelia Hartley consults a Film Historian at Film Archive."
            ),
        },
    }) == []


def test_media_archive_meta_identities_are_reconciled_before_cast_prompts():
    """The known media-archive wrong person cannot reach a record prompt."""
    from nodes import OTR_LedgerScriptWriter as writer

    media_meta = {
        "news": {
            "upstream_identity_names": [
                "Dr. Amelia Hartley", "Professor Elias Venn",
            ],
        },
    }
    identities = writer._upstream_identity_names(media_meta)
    brief = (
        "Dr. Amelia Hartley is a meticulous preservationist with a warm "
        "contralto. Professor Elias Venn is a skeptical film historian with "
        "clipped delivery."
    )
    stub = PromptCapturingStub()
    _cast, cast_meta = _lock(stub, upstream=identities, brief=brief)

    assert cast_meta["name_authority"]["upstream_identities"] == identities
    assert stub.prompts, "no prompt was captured -- the test proved nothing"
    joined = "\n".join(stub.prompts)
    assert "CHARACTER A" in joined and "CHARACTER B" in joined
    for identity in identities:
        for surface_form in _NA.identity_aliases(identity, []):
            if len(surface_form) >= 4:
                assert surface_form.casefold() not in joined.casefold(), (
                    f"media-archive identity {surface_form!r} reached a "
                    "per-record cast prompt"
                )


def test_adaptation_names_are_never_redacted():
    """A roster that IS the source's cast supplies its names harmlessly.

    Fidelity to the source is the whole point of the adaptation lanes; this
    audit must never argue against it.
    """
    assert _NA.superseded_identities(
        ["MACBETH", "BANQUO"], ["LADY MACBETH", "BANQUO"]) == []
    text = "MACBETH broods while BANQUO watches."
    out, labels = _NA.reconcile_text(
        text, _NA.superseded_identities(["MACBETH"], ["MACBETH"]), ["MACBETH"])
    assert out == text and labels == {}


def test_a_row_is_never_flagged_for_its_own_name():
    """Local models routinely repeat the assigned name in the subject head.

    A detector that fires on the correct case is worse than none, because it
    trains everyone to ignore it.
    """
    roster = ["RICK STEINER", "OYA SATO"]
    # Layer 1: the roster owns the name, so it is never superseded.
    assert _NA.superseded_identities(["RICK STEINER"], roster) == []
    # Layer 2 must be proved SEPARATELY. Passing the (empty) result of
    # superseded_identities into the detector proves nothing at all: the
    # detector returns [] for an empty identity list, so that assertion would
    # hold even if the detector flagged every string it was ever given.
    assert _NA.find_foreign_identities(
        {"character_description": "30s, Rick Steiner, lead pilot."},
        ["RICK STEINER"], roster) == [], (
        "the detector flagged a row for its own assigned name"
    )


@pytest.mark.parametrize("identity,surface_form", [
    ("ELIZABETH 'LIZZIE' WALSH", "Lizzie"),
    ("EDWARD 'EDDIE' STONE", "Eddie"),
    ("CAPTAIN JONAS REED", "Jonas"),
    ("MARTHA HONEYWELL", "Honeywell"),
])
def test_aliases_are_detected_not_just_canonical_names(identity, surface_form):
    """Short forms are what the archive actually contains.

    Full-string matching scored a doubly-contaminated episode clean; these are
    the real surface forms measured on live ledgers.
    """
    roster = ["PHYLLIS TERWILLIGER"]
    found = _NA.find_foreign_identities(
        {"character_description": f"40s, {surface_form} Gray - the timekeeper."},
        [identity], roster)
    assert found, f"surface form {surface_form!r} of {identity!r} was not detected"


def test_titles_alone_are_not_identities():
    """'Captain' is colour, not a person; redacting it would maul the prose."""
    assert "captain" not in _NA.name_tokens(_NA.normalize_text("CAPTAIN JONAS REED"))


# --------------------------------------------------------------------------- #
# Contract tests requested by the QA pass (2026-08-20)
# --------------------------------------------------------------------------- #

def test_enforce_never_raises_even_when_the_generator_explodes():
    """The guard is a TOTAL function. Nothing escapes it, ever.

    This is the executable form of the operator's rule that no mechanism in this
    item may fail an episode. It is asserted here rather than with a runtime
    ``assert isinstance(...)``, because an assertion inside a function
    documented as never-raising would ITSELF be a raise path -- and would vanish
    under ``python -O``, so it would guarantee nothing where it mattered.
    """
    slot = _OTRC.EnsembleSlot(
        char_id="c02", name="ERIN BURNS", gender="female",
        timbre="warm", role="lead",
    )
    contaminated = _OTRC.CastingResponse(
        character_description="40s, CAPTAIN JONAS REED. Face: square jaw.",
        gender="female", voice_preset="", speech_signature="even delivery",
    )

    def exploding(messages, **kwargs):
        raise RuntimeError("loader exploded")

    out, event = _OTRC._enforce_name_authority(
        contaminated,
        generate_fn=exploding,
        slot=slot,
        news_seed="", style="radio drama", casting_brief="",
        superseded_names=list(UPSTREAM),
        roster_names=["ANNOUNCER", "ERIN BURNS", "ED HIBBERT"],
    )
    assert event["rung"] == "floor"
    assert "retry_error" in event
    # The returned object must still be a usable CastingResponse: the guard
    # copies a DescriptionResponse's two prose fields onto it, so both models
    # must keep carrying them.
    assert isinstance(out, _OTRC.CastingResponse)
    assert out.gender == "female", "a non-prose field was lost in the copy"
    assert out.character_description.strip()
    assert out.speech_signature.strip()
    assert "JONAS" not in out.character_description.upper()


def test_the_deterministic_floor_is_distinct_across_an_ensemble():
    """BUG-098 was ONE generic fallback painting one face for a whole cast.

    A floor that collapses to identical prose would recreate it, so the floor
    must vary with the Python-owned slot facts.
    """
    roles = ["lead", "foil", "authority", "witness"]
    floors = [
        _OTRC._deterministic_identity_floor(
            _OTRC.EnsembleSlot(char_id=f"c{i:02d}", name=f"NAME{i}",
                               gender="female", timbre=f"timbre{i}", role=role)
        )
        for i, role in enumerate(roles, start=2)
    ]
    assert len(set(floors)) == len(floors), (
        f"the deterministic floor collapsed to {len(set(floors))} distinct "
        f"descriptions across {len(floors)} rows -- BUG-098 all over again"
    )
    for text in floors:
        assert text.strip() and "Face:" in text


# --------------------------------------------------------------------------- #
# Regressions for the defects the Sol QA pass found (2026-08-20)
# --------------------------------------------------------------------------- #

def test_a_shared_surname_does_not_split_a_person_into_two_labels():
    """Two upstream people sharing a surname must stay two people, not three.

    Before the alias plan was resolved across the whole set, the first
    identity's bare surname was substituted first and ate the second person's
    full name: ``["JONAS REED", "MARTHA REED"]`` over
    *"JONAS REED argues with MARTHA REED."* produced
    *"CHARACTER A argues with CHARACTER B CHARACTER A."* -- rendering one person
    as two, the same harm as collapsing everyone into one shared token.
    """
    roster = ["ERIN BURNS", "ED HIBBERT"]
    out, labels = _NA.reconcile_text(
        "JONAS REED argues with MARTHA REED.", ["JONAS REED", "MARTHA REED"], roster)
    assert out == "CHARACTER A argues with CHARACTER B.", out
    assert len(set(labels.values())) == 2

    # And the ambiguous bare surname must not be admitted as a short form for
    # either of them, because it cannot identify which person it refers to.
    plan, _ = _NA.build_alias_plan(["JONAS REED", "MARTHA REED"], roster)
    assert not [row for row in plan if row[0].lower() == "reed"], (
        "an ambiguous shared surname was admitted to the alias plan"
    )


def test_an_unambiguous_short_form_is_still_admitted():
    """The collision guard must not disarm the detector for ordinary casts."""
    roster = ["ERIN BURNS"]
    plan, _ = _NA.build_alias_plan(["CAPTAIN JONAS REED", "MARTHA HONEYWELL"], roster)
    surfaces = {row[0].lower() for row in plan}
    assert "jonas" in surfaces and "honeywell" in surfaces, surfaces


def test_reconciliation_and_detection_share_one_resolved_plan():
    """What reconciliation removes is exactly what detection looks for.

    If detection admitted surfaces reconciliation does not remove, the guard
    would fire on text the boundary was never going to clean, and the archive
    sweep would be measuring a contract nobody ships.
    """
    roster = ["ERIN BURNS", "ED HIBBERT"]
    identities = ["JONAS REED", "MARTHA REED"]
    text = "JONAS REED argues with MARTHA REED."
    reconciled, _ = _NA.reconcile_text(text, identities, roster)
    assert not _NA.find_foreign_identities({"t": reconciled}, identities, roster), (
        "detection still sees an identity that reconciliation already removed"
    )
    # The assertion above alone is satisfied by ANY detector that happens to
    # find nothing here -- including one running its own divergent plan. So
    # assert the shared plan DIRECTLY: every surface detection would fire on
    # must be a surface reconciliation actually removes.
    plan, _labels = _NA.build_alias_plan(identities, roster)
    assert plan, "the alias plan is empty -- this test proves nothing"
    for surface, _label, _ident in plan:
        removed, _ = _NA.reconcile_text(f"X {surface} Y", identities, roster)
        assert surface.lower() not in removed.lower(), (
            f"detection admits {surface!r} but reconciliation leaves it in place"
        )


def test_the_prompt_seed_fallback_cannot_reinstate_an_identity():
    """An empty brief must not let the raw seed carry the names back in.

    `_build_user_prompt` falls back to a slice of news_seed when the brief is
    empty, so the seed is a second door into the same `Story:` line.
    """
    import random
    stub = PromptCapturingStub()
    # THE SEED MUST CARRY AN IDENTITY or this test proves nothing. With an empty
    # brief the prompt builder falls back to the seed -- but a seed with no name
    # in it stays clean whether or not the seed is reconciled, so the original
    # version of this test passed with seed reconciliation deleted.
    _OTRC.lock_cast(
        creative_fn=stub, num_characters=2,
        news_seed="CAPTAIN JONAS REED and MARTHA HONEYWELL stall the ferry at dusk",
        style="radio drama", rng=random.Random(1234), cast_seed=1234,
        force_lemmy=False, casting_brief="",
        upstream_identity_names=UPSTREAM,
    )
    joined = "\n".join(stub.prompts)
    assert joined.strip(), "no prompt captured -- the test proved nothing"
    for identity in UPSTREAM:
        assert identity.lower() not in joined.lower(), (
            f"{identity!r} reached the prompt through the news_seed fallback"
        )


# --------------------------------------------------------------------------- #
# Regressions for the round-2 QA findings (2026-08-20)
# --------------------------------------------------------------------------- #

def test_legitimate_relational_prose_is_never_flagged():
    """The one failure mode a guard must not have: making good output worse.

    Live artifact -- `signal_lost_the_flooded_hymn_20260714_181940` upstream
    names HIRAM BLEEK and the row ELLIE TERWILLIGER reads
    "40s, foil to Hiram's meticulous obsession." That describes ELLIE. A
    context-free match discards a healthy row and replaces it with a template.
    Bug Bible 11.61 names this class explicitly as one that must not be flagged.
    """
    roster = ["ELLIE TERWILLIGER", "ANNOUNCER"]
    assert not _NA.find_foreign_identities(
        {"character_description":
            "40s, foil to Hiram's meticulous obsession. Face: oval, hooded eyes."},
        ["HIRAM BLEEK"], roster)
    # relational connectors, not just possessives
    for prose in ("40s, rival to Hiram Bleek in every way.",
                  "50s, daughter of Hiram Bleek.",
                  "30s, confidante to Hiram Bleek."):
        assert not _NA.find_foreign_identities(
            {"character_description": prose}, ["HIRAM BLEEK"], roster), prose


def test_a_bare_identity_claim_is_still_flagged():
    """The relational carve-out must not disarm the real detection."""
    roster = ["OYA SATO", "ANNOUNCER"]
    for prose in ("30s, Henry 'Hank' Griswold. Face: square jaw.",
                  "40s, Lizzie Gray - The Timekeeper. Face: oval."):
        ident = ["HENRY 'HANK' GRISWOLD"] if "Griswold" in prose else ["ELIZABETH 'LIZZIE' GRAY"]
        assert _NA.find_foreign_identities(
            {"character_description": prose}, ident, roster), prose


def test_an_ordinary_word_surname_does_not_maul_innocent_prose():
    """`EDWARD STONE` must not rewrite "a stone wall" into a placeholder."""
    out, _ = _NA.reconcile_text(
        "A stone wall frames the harbour.", ["EDWARD STONE"], ["ERIN BURNS"])
    assert out == "A stone wall frames the harbour.", out
    # the full name is still removed
    out2, _ = _NA.reconcile_text(
        "EDWARD STONE waits.", ["EDWARD STONE"], ["ERIN BURNS"])
    assert "CHARACTER A" in out2 and "STONE" not in out2.upper()


def test_a_canonical_name_colliding_with_a_nickname_keeps_both_people():
    """Canonical forms must compete in the ambiguity check, not bypass it."""
    ids = ["ELIZABETH 'LIZZIE' WALSH", "LIZZIE"]
    out, labels = _NA.reconcile_text(
        "LIZZIE challenges ELIZABETH 'LIZZIE' WALSH.", ids, ["ERIN BURNS"])
    assert len(set(labels.values())) == 2
    assert out.count("CHARACTER A") == 1 and out.count("CHARACTER B") == 1, out


@pytest.mark.parametrize("identity,text", [
    ("ANA O'NEIL", "Ana O’Neil arrives."),
    ("MARY-JANE DOE", "Mary–Jane Doe arrives."),
])
def test_glyph_variants_are_matched_as_the_docstring_claims(identity, text):
    """normalize_text unifies these, so matching must tolerate them too.

    Matching the RAW surface silently ignored every glyph rule the module
    advertises, so wrong-person prose carrying a curly apostrophe or an en-dash
    passed both layers untouched.
    """
    roster = ["ERIN BURNS"]
    assert _NA.find_foreign_identities({"d": text}, [identity], roster)
    out, _ = _NA.reconcile_text(text, [identity], roster)
    assert "CHARACTER A" in out, out


def test_a_shared_token_does_not_grant_ownership_of_a_whole_person():
    """`SOM STONE` on the roster must not claim the foreign `DR. STONE`."""
    assert _NA.superseded_identities(["DR. STONE"], ["SOM STONE"]) == ["DR. STONE"]
    # and true ownership still holds
    assert _NA.superseded_identities(["MACBETH"], ["MACBETH"]) == []
    assert _NA.superseded_identities(["MACBETH"], ["LADY MACBETH"]) == []


def test_a_leaked_placeholder_is_caught_and_replaced():
    """The model can copy CHARACTER A out of the brief, so the guard must see it.

    Reconciliation puts the label in front of the model, which is the same
    free-text slot a real name used to land in. "30s, CHARACTER A." is not a
    wrong-PERSON description, so the identity detector is blind to it -- but it
    is obviously broken text that would be spoken, printed in the credits and
    painted into a portrait.
    """
    import random

    class LeaksTheLabel(PromptCapturingStub):
        def __call__(self, messages, **kwargs):
            super().__call__(messages, **kwargs)
            return json.dumps({
                "character_description": "30s, CHARACTER A. Face: square jaw.",
                "speech_signature": "even delivery",
            })

    stub = LeaksTheLabel()
    cast, meta = _OTRC.lock_cast(
        creative_fn=stub, num_characters=2, news_seed="a ferry stalls",
        style="radio drama", rng=random.Random(7), cast_seed=7,
        force_lemmy=False, casting_brief=REAL_BRIEF,
        upstream_identity_names=UPSTREAM,
    )
    events = meta["name_authority"]["events"]
    assert events, "a leaked placeholder shipped with no guard event"
    assert any(e.get("leaked_labels") for e in events)
    for row in cast:
        assert "CHARACTER A" not in row["character_description"], (
            "the placeholder shipped into the locked cast"
        )


def test_a_mention_does_not_shadow_a_later_claim():
    """Every occurrence is examined, not just the first one `search` returns.

    "foil to Jonas's obsession. But I am Jonas!" escaped detection entirely:
    the possessive mention came first, satisfied the claim test, and the real
    claim after it was never looked at.
    """
    roster = ["ELLIE TERWILLIGER"]
    found = _NA.find_foreign_identities(
        {"character_description": "foil to Jonas's obsession. But I am Jonas!"},
        ["JONAS REED"], roster)
    assert found, "a claim hiding behind an earlier mention was missed"


@pytest.mark.parametrize("prose", [
    "A loyal ally of Jonas's.",
    "She was a victim of Jonas's, once.",
    "40s, shaped by Jonas's long shadow.",
])
def test_a_possessive_is_a_mention_whatever_punctuation_follows(prose):
    """The possessive test must not depend on what comes after the `'s`.

    It was silently dead: a stray control character in the pattern meant it
    never matched anything, and the cases that looked correct were being caught
    by the relational-connector path instead.
    """
    assert not _NA.find_foreign_identities(
        {"character_description": prose}, ["JONAS REED"], ["ELLIE TERWILLIGER"]), prose


def test_no_source_file_carries_a_stray_control_character():
    """Heredoc-written source can smuggle in a real control byte.

    `\b` written through a shell heredoc became an actual backspace (0x08)
    inside a regex, which silently disabled the possessive check. A regex that
    never matches fails OPEN here -- it flags healthy prose -- so this is a
    correctness guard, not tidiness.
    """
    import io as _io
    targets = [
        _REPO_ROOT / "nodes" / "_otr_name_authority.py",
        _REPO_ROOT / "nodes" / "_otr_casting.py",
        _REPO_ROOT / "scripts" / "audit_wrong_person_census.py",
        Path(__file__),
    ]
    forbidden = {"\x00", "\x07", "\x08", "\x0b", "\x0c", "\x1b"}
    for path in targets:
        data = _io.open(path, encoding="utf-8").read()
        bad = sorted({hex(ord(c)) for c in data if c in forbidden})
        assert not bad, f"{path.name} carries control character(s) {bad}"
