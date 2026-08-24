"""THE CLEAN STAGE -- a model reads every line, and a model repairs it.

Operator, 2026-08-14: *"I need 3-5 LLM passes to clean the ledger."* and
*"I hate shims and assertions of py to fix things, I like LLM calls to ask it
to fix things."* So the shape is fixed and this module obeys it exactly:

    write the story once  ->  a model JUDGES each line
                          ->  a model REPAIRS what it named
                          ->  sealed ledger  ->  TTS

WHY THE JUDGE IS A MODEL AND NOT A PATTERN LIST
------------------------------------------------
The first cut of this pass gated the repair on a regex list, and the operator
killed it on exactly the right grounds: *"your shim can clean a contained
story but it won't fix the next one."* He is correct, and it is easy to show.
The list had `sighs` and `turns` in it; it did not have `closes`. So

    "The door closes behind him."

sails straight through a pattern list and gets read aloud. The lab's own
695-line `spoken_text_policy.py` is the same class of thing and says so in its
header -- *"a future fuzzy or model-assisted policy requires a new policy."*
Enumerating stage business is not a finite job, so it may not be a Python
job. What the operator asked for instead is the right primitive: *"I'd rather
a more intelligent LLM say 'do you see things acting, like a door closing?
that's not dialogue -- well, you need to make a rewrite pass'."*

So a MODEL answers the question "is every word of this line something the
character says out loud?" That question generalizes; a verb list does not.

THE PATTERN LIST SURVIVES AS A HINT, NEVER AS A GATE
-----------------------------------------------------
``_otr_spoken_text_policy`` still runs -- it is free, it is the same detector
the offline grader scores with, and it catches the blatant cases instantly.
Its findings are handed to the judge as EVIDENCE, labelled as often wrong.
A line is dirty if the judge says so OR the patterns do: a union, never a
veto. Nothing in Python decides whether the model's reading is admissible,
and nothing in Python decides whether the patterns' reading is. That is the
difference between two detectors and a shim.

THE REPAIR THINKS, IT DOES NOT STRIP
------------------------------------
Operator: *"make its best choice -- mix of removing the stage directions and
updating the dialogue. It should think of the best edit."* Deleting the
parenthetical and keeping the rest is the WRONG answer: it can leave a line
that no longer makes sense, or lose the beat the action carried. So the
repair model gets the line, who says it, the acts and beats SO FAR, and the
judge's own words about what is wrong -- and returns the best edit. Sometimes
the action becomes implied in what the character says. That judgement is the
model's, which is exactly why it may never be Python.

PYTHON'S ENTIRE JOB IN THIS MODULE
-----------------------------------
Choose which rows to ask about, carry the model's answer into the row, count
what happened, and stop after a bounded number of tries. It never writes,
edits, strips, or overrules a word of prose. The single line of code that
touches a row's text writes the MODEL'S returned string.

BOUNDED, INFORMED, AND IT NEVER STOPS THE RENDER
-------------------------------------------------
Each repair is TOLD what the judge found, and the judge re-reads the result
-- a retry that knows what was wrong is not the same cold roll again. After
the budget, the row SHIPS, the ledger records it as unclean, and the log says
so loudly. Never a silent pass; never a hard stop. An imperfect line beats a
dead episode.

WHERE IT RUNS
-------------
Once, at the one shared producer boundary --
``OTR_LedgerScriptWriter._run_writer_tail`` -- immediately before
``_otr_ledger_cleanup.run_ledger_cleanup``. Every source bank reaches that
boundary: the legacy writer path, and both news lanes via the pipeline-runner
dispatch. Running BEFORE the cleanup pass is deliberate, because that pass
re-stamps text metrics, so a row this module rewrites is measured after the
rewrite rather than before it.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, MutableMapping, Sequence

try:
    from . import _otr_spoken_text_policy as _POLICY
    from ._otr_text_metrics import set_line_text_metrics
except ImportError:  # pragma: no cover -- flat test/standalone load
    import _otr_spoken_text_policy as _POLICY  # type: ignore
    from _otr_text_metrics import set_line_text_metrics  # type: ignore

log = logging.getLogger("OTR.ledger_clean")

__all__ = [
    "LEDGER_CLEAN_VERSION",
    "UNCLEAN_COMPOSE_FLAG",
    "MISATTRIBUTED_COMPOSE_FLAG",
    "PROTECTED_FACT_COMPONENT_FLAG",
    "run_ledger_clean",
]

LEDGER_CLEAN_VERSION = "ledger_clean_v2"

#: A ROW THIS PASS MAY NOT REWRITE, BECAUSE PYTHON OWNS PART OF ITS TEXT.
#:
#: THE DEFECT THIS EXISTS FOR (PBUG-20260815-01). The closing announcer row is
#: a COMPOSITE: a model-authored bridge plus a deterministic, Python-appended
#: source fact. This pass had no concept of a protected span inside a row -- it
#: judged the row as one undifferentiated block and handed the whole thing to a
#: model to rewrite. `reel_of_mystery` b016 composed a factual Library of
#: Congress note naming three real films and SHIPPED *"Clarisse's gaze meets
#: the reel's enigmatic label"*: the fact was simply gone, while `meta` still
#: advertised that the episode had spoken it. Measured at 9 of 14 voiced rows
#: across three episodes in one overnight gate.
#:
#: ONE CONSTANT, TWO READERS. `_otr_line_composer.compose_news_coda` stamps it
#: at the moment it appends the fact; the loop below reads it before any judge
#: call. Two hand-spelled literals is `BUG_BIBLE.yaml` 12.86 -- a
#: producer/consumer mismatch where the emitter and the reader drift apart and
#: the guard silently stops guarding.
#:
#: STAMPED INDEPENDENTLY OF THE BRIDGE OUTCOME. Not derived from
#: `news_coda_bridge` / `news_coda_fact_only`: those describe whether the
#: bridge survived validation, which is decided AFTER cleaning would have run.
#: A flag that cannot identify the row before it is judged is useless to a
#: check whose whole job is to fire first.
#:
#: WHY A WHOLE-ROW SKIP RATHER THAN A CLEVERER SCOPE. The guarantee has to be
#: STRUCTURAL -- the protected text must be physically unable to enter the
#: model's edit surface. Asking a model nicely to preserve a fact leaves a
#: model deciding, and a post-hoc "did the fact survive" check ships the wrong
#: row and merely notices. Scoping by `speaker_role` was considered and
#: rejected: announcer rows are legitimately judged, and
#: `tests/test_ledger_clean_stage.py` pins `judge_calls == 3` on a synthetic
#: ledger whose announcer row carries no flags, so a role-keyed exemption
#: would drop it to 2.
PROTECTED_FACT_COMPONENT_FLAG = "protected_fact_component"

#: Stamped on a row that survived the whole repair budget still dirty, so the
#: defect is visible in the artifact and not only in the log.
UNCLEAN_COMPOSE_FLAG = "unclean_spoken_text"

#: Stamped on a row that still reads as another character's speech after the
#: bounded fix. Visible in the artifact, never a reason to kill a render.
MISATTRIBUTED_COMPOSE_FLAG = "misattributed_spoken_text"

#: Repair attempts per dirty row. Two, not nine: the second is INFORMED (the
#: judge has already read the first and said what survived), and a third
#: informed pass has never been the difference between a fixable line and a
#: hopeless one -- it is where grinding starts.
_MAX_ATTEMPTS = 2

#: One repaired line. A beat's worth of speech, never an act's.
_MAX_NEW_TOKENS = 320

#: The judge answers a yes/no and quotes the offending words. It never needs
#: room to write prose, and a tight budget is what keeps a per-row pass cheap.
_JUDGE_MAX_NEW_TOKENS = 160

#: THE RECIPE KNOBS. Module-level so `scripts/otr_clean_stage_lab.py` can
#: A/B them against a planted ledger and MEASURE the answer instead of
#: arguing about it. Operator, 2026-08-14: *"the repair needs to be model
#: agnostic -- do a bunch of A/Bs until you get something that works."* The
#: shipped defaults are whatever won the last A/B; they are not guesses.
#:
#: The prompt TEXT is deliberately not a knob -- one prompt per job for every
#: model tier is a standing law. What varies is how the job is run.
JUDGE_TEMPERATURE = 0.2
#: How many independent reads before a finding is acted on. 1 = trust the
#: first read. 2 = ask twice and keep only what BOTH reads name, which is the
#: classic cure for a noisy judge and costs one extra small call per row.
JUDGE_VOTES = 1
#: SPLIT THE LOAD BETWEEN CALLS. Operator, 2026-08-14: *"one pass creating
#: the briefing of the act and dialogue, so the repair pass can read that
#: brief and all it has to do is generate the repaired line -- splitting the
#: load between calls."*
#:
#: The reasoning is exactly right for a small model. The full repair prompt
#: carries the act, the brief, the surrounding lines, a numbered checklist
#: AND a how-to-edit passage -- and a 2B handed that much drops most of it.
#: With this on, the briefing pass has already done the reading, so the
#: repair call carries the brief, the line and the complaint, and its whole
#: job is to write one sentence.
REPAIR_READS_BRIEF_ONLY = False
#: ONE SENTENCE PER CALL instead of one line per call. This is the LAST
#: untested lever, and it is the one that has actually worked in this codebase
#: before: per-beat dialogue fixed the writer by making the JOB smaller, not
#: the prompt cleverer.
#:
#: Today the judge must do four things in one reply -- split the line,
#: classify every piece, quote each offender verbatim, and count the pieces.
#: A 2B drops most of a four-part instruction. With this on, PYTHON does the
#: splitting (mechanical -- sentence boundaries are not a judgement about
#: prose, and nothing is edited), and the model is left with one question it
#: can actually answer: is THIS sentence talk, or is it stage business?
#:
#: It costs about two small calls per line instead of one larger one. Whether
#: that trade wins is a measurement, not an opinion -- run the lab.
JUDGE_PER_SENTENCE = False
#: F2's CONTENT half -- "is this the wrong character's speech?" -- as its own
#: per-row model call. OFF until the lab says it earns its wall-clock, which
#: is the operator's own rule for this one: *"Fable had a good idea, but we
#: need a real laboratory test on Fable's design to see what works."*
JUDGE_ATTRIBUTION = False

#: How much of the episode a model is shown BEFORE the line. The window is the
#: point: a model that sees its own scene answers the line in front of it, and
#: one handed the whole line graph averages the episode instead.
_CONTEXT_ROWS = 12

#: And AFTER it. Operator, 2026-08-14: *"when correcting it needs to look at
#: the act and the lines before AND after."* He is right, and the writer's own
#: rule does not apply here: a BEAT job cannot see the future because the
#: future is not written yet, but this pass runs on a FINISHED episode where
#: the next line already exists. Rewriting a line without it is how you get an
#: edit the following line no longer answers.
_AFTER_ROWS = 4


def _rows(ledger_data: Mapping[str, Any], key: str) -> "list":
    value = ledger_data.get(key)
    return value if isinstance(value, list) else []


def _text_of(row: Mapping[str, Any]) -> str:
    value = row.get("text")
    return value if isinstance(value, str) else ""


def _beats_by_id(ledger_data: Mapping[str, Any]) -> "dict[str, Mapping]":
    return {
        str(row.get("beat_id")): row
        for row in _rows(ledger_data, "beats")
        if isinstance(row, Mapping)
    }


def _cast_names(ledger_data: Mapping[str, Any]) -> "dict[str, str]":
    return {
        str(row.get("char_id")): str(row.get("name") or "")
        for row in _rows(ledger_data, "cast")
        if isinstance(row, Mapping)
    }


def _is_voiced(row: Any) -> bool:
    """Is this row spoken aloud -- and is it even a row?

    THE isinstance GUARD IS LOAD-BEARING, not defensive habit. The CURRENT
    row was already guarded, but the NEIGHBOUR rows fed to `_lines_around`
    and to the context counters were sliced straight out of `lines[]` and
    handed here unchecked. One non-mapping entry in that array -- a stray
    string or null from an older or hand-edited ledger -- would raise
    AttributeError out of this pass, and `run_ledger_clean` is called
    UNWRAPPED from the writer tail, so it would kill the render. This pass
    may never do that. (`scripts/otr_ledger_view.grade()` already filters the
    same array the same way, so the shape is not hypothetical here.)
    """
    return (
        isinstance(row, Mapping)
        and row.get("speaker_role") in _POLICY.VOICED_ROLES
        and not bool(row.get("skip"))
        and bool(_text_of(row).strip())
    )


def _flag_unclean(row: MutableMapping[str, Any]) -> None:
    flags = row.get("compose_flags")
    if not isinstance(flags, list):
        flags = []
        row["compose_flags"] = flags
    if UNCLEAN_COMPOSE_FLAG not in flags:
        flags.append(UNCLEAN_COMPOSE_FLAG)


def _flag_misattributed(row: MutableMapping[str, Any]) -> None:
    flags = row.get("compose_flags")
    if not isinstance(flags, list):
        flags = []
        row["compose_flags"] = flags
    if MISATTRIBUTED_COMPOSE_FLAG not in flags:
        flags.append(MISATTRIBUTED_COMPOSE_FLAG)


def _lines_around(
    rows: "Sequence[Mapping[str, Any]]",
    index: int,
    beats: "Mapping[str, Mapping]",
) -> "list[str]":
    """The scene around this line: the rows before it, THIS LINE, then after.

    Both sides, because this pass runs on a FINISHED episode. A rewrite that
    cannot see the reply it has to set up produces a line the next one no
    longer answers -- and the next line is already written, so withholding it
    buys nothing. The beat's own intent rides along where the ledger carries
    one, because an edit has to know what the moment was FOR before it can
    decide which half of a line to keep.

    THE LINE ITSELF IS MARKED IN PLACE. A model handed a flat list of rows
    has to guess which one it is working on; the marker removes the guess.
    """
    window: "list[str]" = []
    for prior in rows[max(0, index - _CONTEXT_ROWS):index]:
        if not _is_voiced(prior):
            continue
        speaker = str(prior.get("speaker") or "").strip() or "UNKNOWN"
        window.append(f"{speaker}: {_text_of(prior).strip()}")

    here = rows[index]
    speaker = str(here.get("speaker") or "").strip() or "UNKNOWN"
    window.append(f">>> {speaker}: {_text_of(here).strip()}")

    for later in rows[index + 1:index + 1 + _AFTER_ROWS]:
        if not _is_voiced(later):
            continue
        speaker = str(later.get("speaker") or "").strip() or "UNKNOWN"
        window.append(f"{speaker}: {_text_of(later).strip()}")
    return window


def _story_field(
    row: Mapping[str, Any], beats: "Mapping[str, Mapping]", key: str,
) -> str:
    """Read a story field from wherever the producing lane actually put it.

    THE LINE ROW FIRST, and that ordering is the whole bug fix. Measured on a
    live `media_archive` episode 2026-08-14: `arc_phase` and `beat_intent`
    are present on **every** LINE row and on **no** beat row -- the writer
    lane's beats carry only transport (`beat_id`, `char_id`, `line_ids`,
    `scene_id`, `shot_id`, `start_s`, `dur_s`). The first cut of this
    function read the BEAT, so "WHERE THE STORY IS" came out empty on all 16
    rows and the judge and the repair both ran BLIND to the act. The unit
    tests missed it because their fixtures put the fields on the beat.

    The beat is kept as a fallback because the codex lane does populate beats
    -- so this reads the union rather than betting on either lane's shape.
    """
    value = str(row.get(key) or "").strip()
    if value:
        return value
    beat = beats.get(str(row.get("beat_id") or "")) or {}
    return str(beat.get(key) or "").strip()


def _where_the_story_is(
    row: Mapping[str, Any],
    beats: "Mapping[str, Mapping]",
    episode: str = "",
    act_briefs: "Mapping[str, str] | None" = None,
) -> str:
    """The act/beat the line lives in, as one short line of prompt.

    Kept OUT of the surrounding-rows block on purpose: one block in pure
    story order with an inline marker is the format a small model misreads
    least, and mixing commentary rows into it costs that clarity.
    """
    bits: "list[str]" = []
    if episode:
        bits.append(episode)
    arc = _story_field(row, beats, "arc_phase")
    if arc:
        bits.append(f"this is the {arc} of the story")
        # THE PASS'S OWN BRIEF. "rising" is a label; a sentence written from
        # the episode's own lines is what a small model can actually use to
        # judge whether a line belongs in this moment.
        brief = str((act_briefs or {}).get(arc) or "").strip()
        if brief:
            bits.append(f"what is going on here: {brief}")
    intent = _story_field(row, beats, "beat_intent")
    if intent:
        bits.append(f"this moment is meant to: {intent}")
    return " -- ".join(bits)


def _episode_context(ledger_data: Mapping[str, Any]) -> str:
    """One clause naming the whole episode, from what the ledger already has.

    The writer stamps a story contract and an arc shape on `meta`; using them
    costs nothing and no model call. Only what is actually recorded is used
    -- nothing here invents a premise the episode never had.
    """
    meta = ledger_data.get("meta")
    if not isinstance(meta, Mapping):
        return ""
    bits: "list[str]" = []
    contract = meta.get("story_contract")
    if isinstance(contract, Mapping):
        label = str(contract.get("label") or "").strip()
        if label:
            bits.append(f"the episode is a {label}")
    shape = str(meta.get("arc_shape") or "").strip()
    if shape:
        bits.append(f"its arc is {shape}")
    return ", ".join(bits)


def _numbered(findings: "Sequence[Mapping[str, str]]") -> str:
    """The judge's findings as the repair's must-all-be-gone checklist."""
    lines = []
    for n, finding in enumerate(findings, start=1):
        quote = str(finding.get("quote") or "").strip()
        why = str(finding.get("why") or "").strip()
        lines.append(f"  {n}. {quote!r}" + (f" -- {why}" if why else ""))
    return "\n".join(lines)


def _as_findings(findings: "Sequence[_POLICY.Finding]") -> "list[dict[str, str]]":
    """Pattern findings in the judge's own shape, so one path feeds the repair."""
    return [{"quote": f.detail, "why": f.kind} for f in findings]


#: THE FIVE THINGS a prompt in this pass is built to show the model, in a
#: FIXED order so the sight string means the same thing every time:
#:
#:     line  speaker  act  around  complaint
#:      1       1      1     1        1       -> "11111", it saw everything
#:      1       1      0     1        1       -> "11011", the ACT never landed
#:
#: `complaint` is repair-only and the act can be legitimately absent, so a
#: field this job never needed reads 1 -- nothing to see, nothing lost. A 0
#: ALWAYS means we built it and it did not arrive, which is the only thing
#: worth an alarm.
CONTEXT_FIELDS = ("line", "speaker", "act", "around", "complaint")

#: Digest length. Eight hex is four billion values -- ample to tell two
#: context blocks apart, short enough to read in a log line.
_SHA_CHARS = 8


def context_digest(*parts: str) -> str:
    """A short SHA-256 over the context we built, house style.

    Same idea the ledger already uses on a spoken row's
    ``text_for_tts_source_sha256``: a small digest is a cheap, exact name for
    "the bytes we meant", and it makes two runs comparable at a glance.
    """
    import hashlib

    joined = "\n".join(p for p in parts if p)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_SHA_CHARS]


def verify_context_landed(
    prompt: "Sequence[Mapping[str, str]]",
    required: "Mapping[str, str]",
) -> "dict[str, Any]":
    """Did the context we BUILT actually reach the prompt we SENT?

    THIS IS THE CHECK THAT WOULD HAVE CAUGHT TODAY'S TWO BUGS, and they were
    different bugs with the same shape:

      1. `arc_phase` / `beat_intent` were read off the wrong row, so the act
         block was built EMPTY and every prompt shipped without it.
      2. `where` was threaded into `_repair_prompt`'s signature and then
         never rendered into the message body -- built, handed over, silently
         dropped on the floor.

    A field-count check catches the first and MISSES THE SECOND, because the
    string existed; it just never made it into the bytes. Hashing what we
    meant and then looking for it in what we sent catches both, costs no
    model call, and cannot drift out of date the way a hand-written assertion
    does.

    Returns a receipt, never raises. A pass that is blind should say so in
    the ledger and keep rendering -- discovering blindness is not a reason to
    kill an episode.
    """
    sent = "\n".join(str(m.get("content") or "") for m in prompt)
    supplied = {
        name: str(value).strip() for name, value in required.items()
        if str(value or "").strip()
    }
    missing = [k for k, v in supplied.items() if v not in sent]

    # THE SIGHT STRING: one digit per field, ALWAYS in CONTEXT_FIELDS order.
    # 1 = I see it, 0 = I do not. Operator's format, 2026-08-14, and it is
    # the right one -- "10111" tells you at a glance that the SPEAKER never
    # landed, in a way no dict of booleans does. Fixed positions mean two
    # runs are comparable by eye and greppable in a log.
    #
    # A field this job never needed reads 1: there was nothing to see and
    # nothing was lost. A 0 ALWAYS means "we built it and it did not arrive",
    # which is the only thing worth an alarm.
    unknown = [k for k in required if k not in CONTEXT_FIELDS]
    if unknown:
        # A KEY WITH NO POSITION WOULD MAKE THE SIGHT STRING LIE -- it would
        # read 11111 while `ok` was False, which is worse than no check at
        # all. Caught here rather than shipped: this is a coding mistake, and
        # it is exactly the class this whole verification exists to surface.
        log.error(
            "[ledger_clean] context field(s) %s have no position in "
            "CONTEXT_FIELDS, so they cannot appear in the sight string. Add "
            "them to CONTEXT_FIELDS or use the existing names: %s",
            ", ".join(sorted(unknown)), " ".join(CONTEXT_FIELDS),
        )
    sight = "".join(
        "0" if name in missing else "1" for name in CONTEXT_FIELDS
    )
    return {
        "sha": context_digest(*(supplied[k] for k in sorted(supplied))),
        "sight": sight,
        "ok": bool(supplied) and not missing,
        "missing": missing,
    }


def _structured():
    """pydantic + the shared retry ladder, or None when unavailable."""
    try:
        from pydantic import BaseModel, Field

        try:
            from ._otr_structured_call import structured_call
        except ImportError:  # pragma: no cover -- flat load
            from _otr_structured_call import structured_call  # type: ignore
    except Exception:  # noqa: BLE001 -- no pydantic, no pass, no crash
        return None
    return BaseModel, Field, structured_call


# ---------------------------------------------------------------------------
# THE JUDGE -- a model reads the line and says what is not speech
# ---------------------------------------------------------------------------


def summarize_acts(
    ledger_data: Mapping[str, Any],
    *,
    slot_fn: "Callable[..., str]",
    cost: "MutableMapping[str, int] | None" = None,
) -> "dict[str, str]":
    """One short read of what is going on in each ACT of the finished episode.

    Operator, 2026-08-14: *"one LLM pass could read them and summarize the
    act again ... and if it says what act"*, then: *"maybe you say quickly
    summarize what's going on around this part of the dialogue based on the
    story act."*

    This is that pass, and it does two jobs at once:

      1. IT MAKES THE CONTEXT REAL. `arc_phase` is a one-word label --
         "rising" tells a 2B model almost nothing about what is happening.
         A sentence written from the episode's own lines does, and it is the
         difference between a repair that fits the moment and one that
         guesses at it.
      2. IT PROVES THE MODEL CAN SEE. A summary written FROM the lines is
         evidence the lines arrived; an empty or generic one is evidence they
         did not. That is the operator's own point -- asking the model what it
         can see beats counting Python variables, because a counter only ever
         proves a string was built.

    GROUPED BY ACT, which is what makes it affordable: one call per arc phase
    (about five an episode) instead of one per row (sixteen and up). Each row
    then carries the summary for the act it actually sits in, so the context
    is LOCAL without being per-row expensive.

    Never fatal. A phase that cannot be summarized simply carries no summary,
    and the pass runs on the labels alone exactly as before.
    """
    parts = _structured()
    if parts is None:
        return {}
    BaseModel, Field, structured_call = parts

    class _ActSummary(BaseModel):
        going_on: str = Field(min_length=1, max_length=160)

    rows = [
        r for r in _rows(ledger_data, "lines")
        if isinstance(r, Mapping) and _is_voiced(r)
    ]
    beats = _beats_by_id(ledger_data)
    grouped: "dict[str, list[str]]" = {}
    for row in rows:
        phase = _story_field(row, beats, "arc_phase") or "the episode"
        speaker = str(row.get("speaker") or "").strip() or "UNKNOWN"
        grouped.setdefault(phase, []).append(
            f"{speaker}: {_text_of(row).strip()}")

    summaries: "dict[str, str]" = {}
    for phase, lines in grouped.items():
        # COUNTED, because the writer tail's comment promises this pass
        # states its cost honestly -- and these briefing calls are real
        # spend that used to be invisible in the receipt.
        if cost is not None:
            cost["calls"] = cost.get("calls", 0) + 1
        prompt = [
            {
                "role": "system",
                "content": (
                    "You read part of a radio script and say what is going "
                    "on in it. You return JSON only."
                ),
            },
            {
                "role": "user",
                "content": "\n".join([
                    f"These are the lines from the {phase} of a radio "
                    "drama, in order:",
                    "",
                    "\n".join(lines[:40]),
                    "",
                    # SHORT ON PURPOSE. This brief is pasted into every judge
                    # and repair prompt for its act, so a rambling summary
                    # would cost tokens on every call and bury the line the
                    # model is actually meant to be looking at.
                    "In about TEN WORDS, say what is going on here: who "
                    "wants what from whom. Write only what these lines "
                    "actually show -- do not invent events.",
                    "",
                    'Answer JSON: {"going_on": "..."}',
                ]),
            },
        ]
        try:
            # LLM slot: creative -- reading drama for what it is about.
            result = structured_call(
                prompt=prompt,
                schema=_ActSummary,
                slot_fn=slot_fn,
                base_temperature=0.3,
                structural_retry_temperature=0.1,
                max_new_tokens=200,
                max_attempts=2,
                helper_name="ledger_clean_act_summary",
            )
        except Exception as exc:  # noqa: BLE001 -- context is a bonus
            log.warning(
                "[ledger_clean] could not summarize the %s (%s: %s); the "
                "pass runs on the arc labels alone for those rows",
                phase, type(exc).__name__, str(exc)[:160],
            )
            continue
        summaries[phase] = " ".join(str(result.going_on or "").split())[:160]
        log.info(
            "[ledger_clean] the %s: %s", phase, summaries[phase][:160],
        )
    return summaries


def _sentences(text: str) -> "list[str]":
    """Split a line into sentences, for ASKING -- never for editing.

    Python decides where the sentence boundaries are, which is mechanical:
    it makes no judgement about whether any of them is speech, it changes
    nothing, and the pieces are reassembled by the model, not by us. The
    whole point is to hand the model ONE small question at a time.
    """
    import re as _re

    parts = [
        p.strip() for p in
        _re.split(r"(?<=[.!?])\s+|(?<=\))\s+(?=[A-Z])", text or "")
        if p and p.strip()
    ]
    return parts or ([text.strip()] if text.strip() else [])


def _judge_by_sentence(
    *,
    slot_fn: "Callable[..., str]",
    speaker: str,
    text: str,
    hints: "Sequence[_POLICY.Finding]",
    lines_around: "Sequence[str]",
    where: str = "",
    sightings: "list[dict[str, Any]] | None" = None,
) -> "tuple[list[dict[str, str]], bool]":
    """Ask about ONE SENTENCE at a time -- the smallest honest job.

    The model no longer splits, counts, or transcribes: Python already knows
    which sentence is being asked about, so the quote is exact by
    construction and the only thing left is the judgement itself. That is the
    same move that fixed the writer -- shrink the job, keep the prompt.
    """
    pieces = _sentences(text)
    if len(pieces) <= 1:
        # Nothing to split. The whole-line judge is already the small job.
        return _judge_row(
            slot_fn=slot_fn, speaker=speaker, text=text, hints=hints,
            lines_around=lines_around, where=where, sightings=sightings,
        )

    found: "list[dict[str, str]]" = []
    reachable = False
    for piece in pieces:
        verdict, ok = _judge_row(
            slot_fn=slot_fn,
            speaker=speaker,
            text=piece,
            # The hints belong to the WHOLE line; only pass the ones whose
            # phrase actually falls inside this sentence, or the model is
            # being told about words it cannot see.
            hints=[h for h in hints if h.detail and h.detail in piece],
            lines_around=lines_around,
            where=where,
            sightings=sightings,
        )
        reachable = reachable or ok
        for entry in verdict:
            # The quote is the SENTENCE, exact by construction -- no verbatim
            # transcription for the model to get wrong.
            found.append({"quote": piece, "why": entry.get("why", "")})
            break
    return found, reachable


def _f2_judge(
    *,
    slot_fn: "Callable[..., str]",
    speaker: str,
    text: str,
    roster: "Sequence[str]",
    lines_around: "Sequence[str]",
    where: str = "",
) -> "tuple[bool, str, str]":
    """Does this line belong to the character it is assigned to?

    F2's CONTENT half, and it is the last failure class in the operator's
    acceptance test with no detector anywhere in the pipeline: *"I'm more
    concerned about not finding and fixing non-dialogue, or the WRONG
    CHARACTER'S SPEECH."*

    ITS OWN CALL, never bolted onto the F1 judge. A small model handed two
    questions degrades on both -- the same job-size law that took F1's recall
    from 13/15 to 15/15 by asking about one sentence instead of four things
    at once.

    The defect is NOT in the words. The identical sentence can be correct in
    one mouth and wrong in another, which is why the fixtures plant the same
    line twice and why no pattern could ever find this. Returns
    ``(belongs, likely_speaker, why)``.
    """
    parts = _structured()
    if parts is None:
        return True, "", ""
    BaseModel, Field, structured_call = parts

    class _Attribution(BaseModel):
        belongs_to_speaker: bool
        likely_speaker: str = Field(default="", max_length=80)
        why: str = Field(default="", max_length=200)

    prompt = [
        {
            "role": "system",
            "content": (
                "You check whether a line of radio dialogue belongs to the "
                "character who is speaking it. You return JSON only."
            ),
        },
        {
            "role": "user",
            "content": "\n".join([
                "In this script the line below is assigned to "
                f"{speaker or 'the announcer'}. Check whether it reads as "
                "that character's line. Most lines are fine -- when the line "
                "sits naturally in this character's mouth, say so and stop.",
                "",
                f"THE SPEAKER: {speaker or 'the announcer'}",
                f"THE LINE: {text}",
                "",
                "WHO IS IN THIS EPISODE: " + ", ".join(roster),
                "",
                f"WHERE THE STORY IS: {where}" if where else "",
                "THE LINES AROUND IT, in story order, this one marked >>>:",
                "\n".join(lines_around),
                "",
                "IT DOES NOT BELONG when:",
                f"- it addresses {speaker or 'the speaker'} BY NAME, or "
                "orders them about. Nobody addresses themselves;",
                "- it claims a role, job or authority that belongs to "
                "someone else in the cast;",
                "- it claims not to know something this character must know, "
                "or knows something only another character could.",
                "",
                "IT DOES BELONG -- say nothing -- when the character is:",
                "- naming or ordering SOMEONE ELSE. That is ordinary "
                "dialogue and is the most common thing in a script;",
                "- quoting or reporting what another person said;",
                "- agreeing, echoing, or finishing another's thought;",
                "- simply stating their own role correctly.",
                "",
                'Answer JSON: {"belongs_to_speaker": true} when it is fine. '
                'If it is not: {"belongs_to_speaker": false, '
                '"likely_speaker": "<who in the cast it really sounds like, '
                'or empty>", "why": "<a few words>"}',
            ]),
        },
    ]
    try:
        # LLM slot: creative -- reading a line against a roster is a
        # reader's judgement about character, not a mechanical check.
        result = structured_call(
            prompt=prompt,
            schema=_Attribution,
            slot_fn=slot_fn,
            base_temperature=JUDGE_TEMPERATURE,
            structural_retry_temperature=min(0.1, JUDGE_TEMPERATURE),
            max_new_tokens=_JUDGE_MAX_NEW_TOKENS,
            max_attempts=2,
            helper_name="ledger_clean_f2_judge",
        )
    except Exception as exc:  # noqa: BLE001 -- never fatal
        log.warning(
            "[ledger_clean] the attribution judge could not read a line "
            "(%s: %s); F2 is unchecked for it",
            type(exc).__name__, str(exc)[:160],
        )
        return True, "", ""
    if result.belongs_to_speaker:
        return True, "", ""
    return (
        False,
        " ".join(str(result.likely_speaker or "").split())[:80],
        " ".join(str(result.why or "").split())[:200],
    )


def _f2_fix(
    *,
    slot_fn: "Callable[..., str]",
    speaker: str,
    text: str,
    likely_speaker: str,
    why: str,
    lines_around: "Sequence[str]",
    where: str = "",
    previous_attempt: str = "",
) -> str:
    """Rewrite the line so it belongs to the STATED speaker.

    ONE behaviour, not a fork. The operator overruled a report-only design --
    *"we can judge, but someone needs to rewrite ... at some point it needs to
    be CORRECTED, not fail the whole thing"* -- and the correction is to move
    the WORDS, never the attribution: the beat's speaker is the contract every
    downstream consumer is already keyed to (voice casting, TTS voice,
    captions, credits), and `text` is the only field a model pass may own.
    Handing back "this is really Olivia's line" would be report-only wearing a
    hat; someone would still have to write it.
    """
    parts = _structured()
    if parts is None:
        return ""
    BaseModel, Field, structured_call = parts

    class _Fixed(BaseModel):
        text: str = Field(min_length=1, max_length=2000)

    body = [
        f"In this radio script the line below is assigned to {speaker}. A "
        f"reader checked it and found it does not read as {speaker}'s line "
        "-- it reads as someone else's words in their mouth. A voice actor "
        f"playing {speaker} will read it exactly as written, so it must "
        f"become a line {speaker} would truly say.",
        "",
        f"THE SPEAKER: {speaker}",
        f"THE LINE: {text}",
        "",
        "WHAT THE READER FOUND:",
        f"  {('It reads as ' + likely_speaker + chr(39) + 's line. ') if likely_speaker else ''}{why}",
        "",
    ]
    if where:
        body += [f"WHERE THE STORY IS: {where}", ""]
    if lines_around:
        body += [
            "THE LINES AROUND IT, in story order, yours marked >>>:",
            "\n".join(lines_around),
            "",
        ]
    if previous_attempt:
        body += [
            "YOUR PREVIOUS ATTEMPT still reads as someone else's line. Do "
            "not hand back a variation of it -- change what was named:",
            previous_attempt,
            "",
        ]
    body += [
        "HOW TO FIX IT:",
        f"- Give the words to {speaker}. Say only what {speaker} knows, "
        f"wants, and would say, in {speaker}'s way of talking.",
        "- The line must still do what the moment needs done. Do not lose "
        "the information or the turn it carries -- put it in this "
        "character's mouth.",
        f"- If the line calls {speaker} by name, that is the giveaway: a "
        "person does not address themselves. Aim the words at whoever "
        f"{speaker} is actually talking to.",
        "- Keep as many of the original words as the fix allows, and keep "
        "the line roughly its original length.",
        "",
        "Example of the move, shape only -- your line is the one above:",
        "  speaker: MALVOLIO",
        "  found: reads as OLIVIA's line -- it addresses Malvolio by name",
        "  before: Then step back, Malvolio, and let me see the letter.",
        "  after:  I will step back, my lady -- and the letter is yours to "
        "see.",
        "",
        f"Before you answer, read your line once as the actor playing "
        f"{speaker} will: every word must be something {speaker} says, to "
        "someone else, in this moment.",
        "",
        'Answer JSON: {"text": "the fixed line"}. The spoken line only -- no '
        "speaker name in front, no brackets, no quotation marks around the "
        "whole line.",
    ]
    try:
        # LLM slot: creative -- it is rewriting dialogue into a voice.
        result = structured_call(
            prompt=[
                {"role": "system", "content": (
                    "You rewrite one line of radio dialogue so that it "
                    "belongs to the character who speaks it. You return JSON "
                    "only."
                )},
                {"role": "user", "content": "\n".join(body)},
            ],
            schema=_Fixed,
            slot_fn=slot_fn,
            base_temperature=0.55,
            structural_retry_temperature=0.25,
            max_new_tokens=_MAX_NEW_TOKENS,
            max_attempts=2,
            helper_name="ledger_clean_f2_fix",
        )
    except Exception as exc:  # noqa: BLE001 -- the row ships flagged
        log.warning(
            "[ledger_clean] the attribution fix failed (%s: %s); the row "
            "keeps its text and is flagged",
            type(exc).__name__, str(exc)[:160],
        )
        return ""
    return " ".join(str(result.text or "").split())


def _judge_votes(**kwargs) -> "tuple[list[dict[str, str]], bool]":
    """Read the line ``JUDGE_VOTES`` times and keep only what every read names.

    A noisy judge is the measured failure mode on a small model: it condemned
    clean dialogue on roughly half the trap lines in the lab. Random noise
    does not survive being asked twice, while a real stage direction does --
    so agreement is the classic, model-agnostic cure, and it costs one extra
    SMALL call per row rather than a bigger model.

    One vote is a straight pass-through, so the default recipe pays nothing.
    """
    reader = _judge_by_sentence if JUDGE_PER_SENTENCE else _judge_row
    if JUDGE_VOTES <= 1:
        return reader(**kwargs)

    rounds: "list[list[dict[str, str]]]" = []
    reachable = False
    for _ in range(JUDGE_VOTES):
        found, ok = reader(**kwargs)
        reachable = reachable or ok
        if ok:
            rounds.append(found)
    if not rounds:
        return [], reachable

    # A finding survives only if EVERY read named the same words.
    agreed: "list[dict[str, str]]" = []
    for entry in rounds[0]:
        quote = entry["quote"].casefold()
        if all(
            any(other["quote"].casefold() == quote for other in later)
            for later in rounds[1:]
        ):
            agreed.append(entry)
    dropped = len(rounds[0]) - len(agreed)
    if dropped:
        log.info(
            "[ledger_clean] %d finding(s) were named by one read but not the "
            "other, so they were not acted on",
            dropped,
        )
    return agreed, reachable


def _judge_prompt(
    *,
    speaker: str,
    text: str,
    hints: "Sequence[_POLICY.Finding]",
    lines_around: "Sequence[str]",
    where: str = "",
) -> "list[dict[str, str]]":
    """One job, one small window -- the same prompt for every model tier.

    Standing law: vary the JOB SIZE, not the prompt text. A small local model
    can answer "is every piece of THIS ONE LINE talk"; it cannot answer
    "clean this ledger". A frontier model does the identical job better with
    the identical words. The lane picks the model; the prompt stays single.

    RECALIBRATED AFTER A LIVE LEG, 2026-08-14. The first cut opened by
    warning that lines "often smuggle stage business" and then listed five
    ways to be guilty. On a 2B that is a HUNT FRAME, and a hunt obliges: it
    condemned two lines of ordinary dialogue -- a man asking another to
    confirm a reference number, and a man arguing with someone he names --
    quoting the ENTIRE line as the offending segment both times. Three things
    fix it, and all three are framing rather than rules:
      * the clean answer is stated as the NORMAL outcome, up front;
      * TALK is defined BEFORE stage business, because on a small model the
        category it reads first becomes the default bucket, and the default
        must be innocence;
      * the three measured lines go in verbatim as calibration. A 2B imitates
        a worked example far more reliably than it applies a rule.
    """
    speaker_label = speaker or "the announcer"
    parts = [
        "Below is one line from a radio script, with the lines around it. A "
        "voice actor will read every word of THE LINE aloud. Most lines are "
        "just people talking -- when that is true, your answer is an empty "
        "list and you are done. Now and then a line carries a piece of stage "
        "business -- an action, a sound, a bit of scene description -- and "
        "that piece must be reported, because the actor would wrongly speak "
        "it.",
        "",
        f"THE SPEAKER: {speaker_label}",
        f"THE LINE: {text}",
        "",
    ]
    if where:
        parts += [f"WHERE THE STORY IS: {where}", ""]
    if lines_around:
        parts += [
            "THE LINES AROUND IT, in story order. The line you are judging "
            "is marked >>>:",
            "\n".join(lines_around),
            "",
        ]
    if hints:
        parts += [
            "A crude pattern-matcher also flagged the following. It is often "
            "WRONG. Check its claims like any other piece; never copy them "
            "as your answer:",
            "\n".join(f"  - {h.kind}: {h.detail}" for h in hints),
            "",
        ]
    parts += [
        "DO THIS, IN ORDER:",
        "1. Split THE LINE into pieces: every sentence, and every bracketed "
        "or parenthesized chunk, is a piece.",
        "2. Check each piece, first to last, all the way to the end: is it "
        "talk, or is it stage business?",
        "3. Report only the stage-business pieces, quoted exactly. If there "
        "are none -- and usually there are none -- return the empty list.",
        "",
        # TALK FIRST. On a small model the first category read becomes the
        # default, and the default has to be innocence. The talks-to-someone
        # test is the POSITIVE discriminator: it clears a question and an
        # argument in one clause each, while leaving pure scene description
        # nothing to grab. One-directional on purpose -- presence proves
        # talk, absence proves nothing, so the unpunctuated door line still
        # falls through to the reporter test below.
        "A piece is TALK -- normal, clean, report nothing -- when someone is "
        "talking to someone: asking, answering, arguing, ordering, "
        "explaining. If it says I, you, or we, asks a question, or names the "
        "person being spoken to, it is talk. Talk about objects, numbers, "
        "places, or the past is talk. Quoting another person is talk. Old, "
        "formal, or literary style is still talk -- style is never a fault.",
        "",
        # THE ANNOUNCER, and the lab is why this clause exists. Handed "The
        # reel returned to its can, and the vault closed", a 2B called it
        # stage business -- reasonably, by the rules as written, since it is
        # third person and nobody is addressed. But that is an ANNOUNCER
        # narrating to the listener, which is the announcer's entire job and
        # the oldest thing in radio. Without this, the pass rewrites every
        # open and every close on every bank.
        "THE ANNOUNCER IS A SPECIAL CASE, and it matters because every "
        "episode opens and closes with one. An announcer speaks TO THE "
        "LISTENER, and narrating in the third person is exactly their job: "
        "\"Tonight, from the lighthouse\" and \"And so the light went dark\" "
        "are an announcer doing radio, not stage business. If THE SPEAKER "
        "is the announcer or a narrator, third-person narration aimed at "
        "the audience is TALK. Report only script apparatus in their lines "
        "-- a bracketed direction, a NAME: label, a sound cue.",
        "",
        "A piece is STAGE BUSINESS when nobody is being spoken to and the "
        "script is describing the scene instead: someone performing an "
        "action (sighs, turns, throws), a sound or event in the room (a door "
        "closes, footsteps), a note on delivery (softly, pausing), people "
        "described from outside in the third person as it happens, or "
        "apparatus like a NAME: label. This kind can wear no brackets at "
        "all.",
        "",
        # THE WHOLE-LINE RULE, stated WITH its one exception. It cannot be
        # "never the whole line": the live catch that proved this pass was a
        # correct whole-line finding. The exception is precisely that shape
        # and nothing else's.
        "A finding is a PIECE of the line, almost never the whole line. If "
        "you are about to quote the entire line, stop and look again: a "
        "question, an argument, or a line that names the person being "
        "addressed is talk, and the right answer is the empty list. Quote "
        "the whole line only when it is scene description from its first "
        "word to its last and talks to no one.",
        "",
        # CALIBRATION IS THE STRONGEST LEVER ON A SMALL MODEL -- it imitates
        # a worked example far more reliably than it applies a rule. All four
        # of these are lines this pass actually got wrong in the lab or on
        # air, so each one buys a measured failure back.
        "CALIBRATION -- four real lines, judged right:",
        '- "The air in the studio crackles as Dale Bernard stares at Stone '
        'Tanaka." -- NOT speech, the whole line. Third person, describes the '
        "room, talks to no one.",
        '- "This shipment was listed as A.P.O. 86-574, reel two, right?" -- '
        "talk. A question to another person. Empty list.",
        "- \"It's more than just a reel number, Tanaka. We have to see "
        "what's inside.\" -- talk. He argues and names the man he is "
        "arguing with. Empty list.",
        # THE ONE THAT FAILS ON EVERY MODEL SIZE. Both the 2B and the 12B
        # condemned "She said the ward was closed before midnight" in the
        # lab -- a person REPORTING what someone else said reads as
        # narration unless the prompt says otherwise. Model-agnostic
        # failure, so it earns a calibration line rather than a tier tweak.
        "- \"She said the ward was closed before midnight.\" -- talk. He is "
        "telling someone what a third person told him. Reporting speech is "
        "speech, even in the past tense about someone absent. Empty list.",
        "",
        "Answer JSON, exactly this shape. Quotes must come from THE LINE "
        "above, never from the calibration lines:",
        '{"segments_read": <how many pieces you split THE LINE into>, '
        '"not_speech": [{"quote": "<the exact words>", "why": "<a few words: '
        'action / sound / delivery note / scene report / apparatus>"}]}',
        "",
        'All talk: {"segments_read": 2, "not_speech": []}',
        'One bad piece in a two-piece line: {"segments_read": 2, '
        '"not_speech": [{"quote": "(He sighs)", "why": "action"}]}',
    ]
    return [
        {
            "role": "system",
            "content": (
                "You check one line of a radio script. You say when it is "
                "all spoken words, and you report any piece of it that is "
                "not. You return JSON only."
            ),
        },
        {"role": "user", "content": "\n".join(parts)},
    ]


def _judge_row(
    *,
    slot_fn: "Callable[..., str]",
    speaker: str,
    text: str,
    hints: "Sequence[_POLICY.Finding]",
    lines_around: "Sequence[str]",
    where: str = "",
    sightings: "list[dict[str, Any]] | None" = None,
) -> "tuple[list[dict[str, str]], bool]":
    """Ask a model to walk the line and name EVERY piece that is not speech.

    Returns ``(not_speech, call_ok)``. ``not_speech`` is a list of
    ``{"quote": ..., "why": ...}`` -- empty means the model read it clean.
    ``call_ok`` is False when the model could not be reached at all, so the
    caller falls back to the pattern hints rather than declaring a line clean
    it never actually read.

    A LIST, NOT ONE FINDING, and that is the operator's requirement: *"it
    reads the WHOLE line and looks for even SEGMENTS that are not
    dialogue."* A line routinely carries stage business at BOTH ends with
    real dialogue between -- "(Montgomery sighs) I've already given you the
    reel ... (Montgomery sighs again)" -- and a judge that reports the first
    one leaves the second to be read aloud.
    """
    parts = _structured()
    if parts is None:
        return [], False
    BaseModel, Field, structured_call = parts

    class _NotSpeech(BaseModel):
        quote: str = Field(min_length=1, max_length=300)
        why: str = Field(default="", max_length=120)

    class _SpokenLineJudgement(BaseModel):
        # A model cannot report how many pieces it read without actually
        # splitting the line, so this integer is a three-token forcing
        # function for the walk -- and it makes a skim visible, because a
        # four-sentence line answered with segments_read=1 was not read.
        segments_read: int = Field(default=1, ge=1, le=64)
        not_speech: "list[_NotSpeech]" = Field(default_factory=list)

    haystack = " ".join(text.split()).casefold()

    def _validate(_result: "_SpokenLineJudgement") -> "str | None":
        # DELIBERATELY PERMISSIVE, and this is a correction of a real defect.
        # The first cut REJECTED a reply whose quote was not in the line, and
        # rejected a whole-line quote as a self-contradicted skim. Both are
        # true faults -- but rejecting cost the ladder, and when the ladder
        # exhausted the row lost its judge ENTIRELY and fell back to the
        # pattern floor. Measured in the lab: a 2B tripped one of the two on
        # most rows, so the strict version was quietly turning the model
        # judge OFF on the very episodes it was built for.
        #
        # A bad ENTRY is now dropped after the call instead of failing the
        # whole reply, which keeps every good finding in the same answer and
        # never costs the judge. Nothing here judges prose: it only decides
        # whether an answer is about the line we asked about.
        return None

    built = _judge_prompt(
        speaker=speaker,
        text=text,
        hints=hints,
        lines_around=lines_around,
        where=where,
    )
    # SHA VERIFY: the context we meant to show, looked for in the bytes we
    # are about to send. Cheap, exact, and it catches a field that was
    # threaded through the call and then never rendered.
    landed = verify_context_landed(built, {
        "line": text,
        "speaker": speaker,
        "act": where,
        "around": "\n".join(lines_around),
    })
    if sightings is not None:
        sightings.append(dict(landed, job="judge"))
    if not landed["ok"]:
        log.error(
            "[ledger_clean] THE JUDGE IS PARTLY BLIND on this line: %s was "
            "built but is not in the prompt that was sent (context sha %s). "
            "That is a wiring fault in this pass, not a model problem.",
            ", ".join(landed["missing"]), landed["sha"],
        )
    try:
        # LLM slot: creative -- it is reading DIALOGUE for what a listener
        # would hear performed, which is a reader's judgement about prose.
        result = structured_call(
            prompt=built,
            schema=_SpokenLineJudgement,
            slot_fn=slot_fn,
            base_temperature=JUDGE_TEMPERATURE,
            structural_retry_temperature=min(0.1, JUDGE_TEMPERATURE),
            post_validator=_validate,
            max_new_tokens=_JUDGE_MAX_NEW_TOKENS,
            max_attempts=2,
            helper_name="ledger_clean_line_judge",
        )
    except Exception as exc:  # noqa: BLE001 -- fall back to the hints
        log.warning(
            "[ledger_clean] the judge could not read a line (%s: %s); the "
            "pattern findings stand alone for it",
            type(exc).__name__, str(exc)[:200],
        )
        return [], False
    found: "list[dict[str, str]]" = []
    dropped: "list[str]" = []
    for entry in result.not_speech:
        quote = " ".join(str(entry.quote or "").split())
        if not quote:
            continue
        if quote.casefold() not in haystack:
            # NOT ABOUT THIS LINE. Measured in the lab: shown the lines
            # around the target, a 2B routinely quotes a NEIGHBOUR back --
            # it is reading the block as the subject. Acting on it would
            # send the repair after words this speaker never said, so the
            # entry goes; the rest of the answer stands.
            dropped.append(f"{quote!r} (not in this line)")
            continue
        if (
            result.segments_read >= 2
            and len(result.not_speech) == 1
            and len(quote.casefold()) >= len(haystack) - 2
        ):
            # SELF-CONTRADICTED: it says it split the line into pieces and
            # then condemns the whole line as one piece. That shape was the
            # exact signature of the measured false positives on clean
            # dialogue, so the entry goes and the line reads clean.
            dropped.append(f"{quote!r} (whole line, but claims a split)")
            continue
        found.append({
            "quote": quote,
            "why": " ".join(str(entry.why or "").split())[:120],
        })
    if dropped:
        log.info(
            "[ledger_clean] dropped %d judge finding(s) that were not usable "
            "for this line: %s",
            len(dropped), "; ".join(dropped)[:240],
        )
    return found, True


# ---------------------------------------------------------------------------
# THE REPAIR -- a model rewrites what the judge named
# ---------------------------------------------------------------------------


def _repair_prompt(
    *,
    speaker: str,
    text: str,
    complaint: str,
    lines_around: "Sequence[str]",
    previous_attempt: str = "",
    where: str = "",
) -> "list[dict[str, str]]":
    speaker_label = speaker or "the announcer"
    parts = [
        "A voice actor is about to read this line into a microphone, every "
        "word exactly as written. A reader has been through it and marked "
        "the pieces that are not speech -- there may be several. Hand back "
        "the line with ALL of them gone and the moment intact.",
        "",
        f"THE SPEAKER: {speaker_label}",
        f"THE LINE: {text}",
        "",
        # A NUMBERED CHECKLIST, not a prose complaint. The repair is graded
        # against an enumerated list, so a model that fixes item 1 and
        # forgets item 2 has failed a list it was literally handed.
        "NOT SPEECH -- every one of these must be gone from your rewrite:",
        complaint,
        "",
    ]
    if where:
        parts += [f"WHERE THE STORY IS: {where}", ""]
    if lines_around and REPAIR_READS_BRIEF_ONLY:
        # LOAD SPLIT: the briefing pass has already read the act, so this
        # call carries only the two lines the rewrite must sit between --
        # which is the constraint it cannot get from a summary -- and its
        # whole job becomes writing one sentence.
        marked = next(
            (i for i, s in enumerate(lines_around) if s.startswith(">>> ")), -1)
        neighbours = [
            s for s in lines_around[max(0, marked - 1):marked + 2] if s
        ]
        parts += [
            "IT SITS BETWEEN THESE, and must still fit between them "
            "(yours is marked >>>):",
            "\n".join(neighbours),
            "",
        ]
    elif lines_around:
        parts += [
            "THE LINES AROUND IT, in story order. The line you are "
            "rewriting is marked >>>. Your rewrite must sit in that slot:",
            "\n".join(lines_around),
            "",
        ]
    if previous_attempt:
        parts += [
            "YOUR PREVIOUS ATTEMPT was read again and still has a problem. "
            "Do not hand back a variation of it -- change what was named:",
            previous_attempt,
            "",
        ]
    parts += [
        "HOW TO EDIT -- take the marked pieces one at a time:",
        f"- Decide what each piece was doing for the moment. If it carried "
        f"something the scene needs -- a feeling, a beat, an event the "
        f"listener must know happened -- fold it into what {speaker_label} "
        f"SAYS, in {speaker_label}'s own voice. A sigh can become a weary "
        f"word. A door closing can become the speaker remarking that someone "
        f"is gone. A thrown object can become the speaker reacting to "
        f"catching it.",
        "- If it carried nothing the scene needs, remove it and smooth the "
        "join, so the sentence still reads as one thought.",
        "- Never just delete and staple. A stripped line often stops making "
        "sense or goes flat. You are making the best EDIT, not the shortest "
        "one.",
        # Without this clause a small model treats "remove the stage
        # direction" and "keep the length" as a contradiction, and resolves
        # it by keeping the stage direction rephrased as narration.
        "- Keep every part that already was speech. Keep the speaker's "
        "voice, and keep the line roughly its original length -- what you "
        "fold in makes up for what you take out.",
        "- Your rewrite has to fit its slot: it follows the line before it "
        "and is answered by the line after it. Do not contradict either "
        "one.",
        "",
        "Example of the move, shape only -- your line is the one above:",
        '  marked: "The door closes behind him." -- scene report',
        "  before: The door closes behind him. I told you he would not stay.",
        "  after:  There he goes -- door and all. I told you he would not "
        "stay.",
        "",
        # The judge's own standard, inside the repair. This is what cuts the
        # ping-pong between the two informed attempts.
        f"Before you answer, read your rewrite once, first word to last, the "
        f"way the voice actor will. If any word of it is not something "
        f"{speaker_label} says out loud, fix it. If any marked piece "
        f"survives in any form that is not spoken words, fix it.",
        "",
        'Answer JSON: {"text": "the rewritten line"}. The spoken line only '
        "-- no speaker label, no brackets or parentheses, no quotation marks "
        "wrapped around the whole line, no description of anyone doing "
        "anything.",
    ]
    return [
        {
            "role": "system",
            "content": (
                "You rewrite one line of radio dialogue so that every word "
                "of it is something the speaker says out loud. You return "
                "JSON only."
            ),
        },
        {"role": "user", "content": "\n".join(parts)},
    ]


def _call_repair(
    *,
    slot_fn: "Callable[..., str]",
    speaker: str,
    text: str,
    complaint: str,
    lines_around: "Sequence[str]",
    previous_attempt: str,
    where: str = "",
    sightings: "list[dict[str, Any]] | None" = None,
) -> str:
    """One bounded model call. Returns the rewritten line, or "" on failure --
    a repair that cannot run is never fatal to the render."""
    parts = _structured()
    if parts is None:
        return ""
    BaseModel, Field, structured_call = parts

    class _RepairedLine(BaseModel):
        text: str = Field(min_length=1, max_length=2000)

    def _validate(result: "_RepairedLine") -> "str | None":
        if not " ".join(str(result.text or "").split()):
            return "text is empty after whitespace normalization"
        return None

    built = _repair_prompt(
        speaker=speaker,
        text=text,
        complaint=complaint,
        lines_around=lines_around,
        previous_attempt=previous_attempt,
        where=where,
    )
    # SHA VERIFY, and this is the exact path where it earned itself: `where`
    # was threaded into this builder's signature and never rendered into the
    # body, so the act was carried all the way here and dropped.
    landed = verify_context_landed(built, {
        "line": text,
        "speaker": speaker,
        "act": where,
        "around": "\n".join(lines_around),
        "complaint": complaint,
    })
    if sightings is not None:
        sightings.append(dict(landed, job="repair"))
    if not landed["ok"]:
        log.error(
            "[ledger_clean] THE REPAIR IS PARTLY BLIND: %s was built but is "
            "not in the prompt that was sent (context sha %s). Wiring fault "
            "in this pass, not the model.",
            ", ".join(landed["missing"]), landed["sha"],
        )
    try:
        # LLM slot: creative -- it is rewriting DIALOGUE, so the tier that
        # wrote the line rewrites it and the repaired line still sounds like
        # its neighbours (operator ruling 2026-08-14).
        result = structured_call(
            prompt=built,
            schema=_RepairedLine,
            slot_fn=slot_fn,
            base_temperature=0.55,
            structural_retry_temperature=0.25,
            post_validator=_validate,
            max_new_tokens=_MAX_NEW_TOKENS,
            max_attempts=2,
            helper_name="ledger_clean_line_repair",
        )
    except Exception as exc:  # noqa: BLE001 -- the row ships flagged instead
        log.warning(
            "[ledger_clean] repair call failed (%s: %s); the row keeps its "
            "current text and is flagged rather than dropped",
            type(exc).__name__, str(exc)[:200],
        )
        return ""
    return " ".join(str(result.text or "").split())


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------


def run_ledger_clean(
    ledger_data: MutableMapping[str, Any],
    *,
    slot_fn: "Callable[..., str] | None" = None,
    bank_id: str = "",
) -> "dict[str, Any]":
    """A model reads every spoken row; a model repairs what it names.

    Returns the receipt it also stamps on ``meta.ledger_clean``. The receipt
    is the point on a dirty episode: it names every row that was touched,
    what the judge said about it in its own words, and what the repair did.

    F2 IS DETECTED AND REPORTED, NOT REPAIRED, and that is deliberate. Its
    metadata half -- a row that names no speaker, or disagrees with its beat
    or the cast -- is not fixed by rewriting prose; the beat already owns the
    answer, so a model call there would be asking a writer to repair a
    bookkeeping fault. Its CONTENT half -- one character speaking another's
    words -- is a reading of the whole episode rather than of one line, and
    is not this pass's job. Both are reported so a regression shows up in the
    artifact.
    """
    rows = _rows(ledger_data, "lines")
    beats = _beats_by_id(ledger_data)
    cast = _cast_names(ledger_data)
    admissible = _POLICY.repairable_kinds(bank_id)
    # Every prompt this run sends reports whether the context it was built
    # with actually landed in its bytes. Collected here so the ledger carries
    # one verdict for the episode instead of a log line nobody reads.
    sightings: "list[dict[str, Any]]" = []
    episode = _episode_context(ledger_data)
    # ONE read of the finished episode, act by act, BEFORE any judging. The
    # pass writes its own brief to rewrite against -- and the brief doubles
    # as the proof it could see the act, which is stronger evidence than any
    # Python field count (operator, 2026-08-14).
    brief_cost: "dict[str, int]" = {}
    act_briefs = (
        summarize_acts(ledger_data, slot_fn=slot_fn, cost=brief_cost)
        if slot_fn else {}
    )

    receipt: "dict[str, Any]" = {
        "version": LEDGER_CLEAN_VERSION,
        "policy": _POLICY.SPOKEN_TEXT_POLICY_ID,
        "source_bank": str(bank_id or ""),
        "judge": "model",
        "episode_context": episode,
        "act_briefs": dict(act_briefs),
        "admissible_pattern_kinds": sorted(admissible),
        # THE BLINDNESS TELEMETRY, and it exists because the pass WAS blind.
        # Operator, 2026-08-14: *"the key is we need to make sure in the
        # ComfyUI workflow it is actually seeing all the artifacts and
        # pointed to them -- act / act spine, characters, before and after
        # dialogue -- and not blind due to a coding error."* He was right:
        # `arc_phase` and `beat_intent` live on the LINE row, this pass read
        # the BEAT row, and every prompt shipped with an empty act. A green
        # unit suite cannot see that -- only a count taken from the real
        # artifact can. If any of these reads 0 on a live episode, the model
        # was working blind and the receipt says so in the ledger.
        "context_seen": {
            "rows_with_arc_phase": 0,
            "rows_with_beat_intent": 0,
            "rows_with_cast_name": 0,
            "rows_with_lines_before": 0,
            "rows_with_lines_after": 0,
            "rows_with_act_brief": 0,
            "acts_summarized": 0,
            "episode_context": "",
        },
        # Line ids skipped because Python owns part of their text. Its own
        # key, deliberately: `rows` means "the judge touched this", and the
        # D1 acceptance test asserts a protected row appears in NEITHER
        # `rows` nor any repair count.
        "protected_rows": [],
        "voiced_rows": 0,
        "judged_dirty": 0,
        "segments_named": 0,
        "pattern_only": 0,
        "judge_only": 0,
        "f1_rows": 0,
        "f2_rows": 0,
        "f2_content_rows": 0,
        "f2_reattributed": 0,
        "f2_reattributed_unverified": 0,
        "f2_unfixed": 0,
        "repaired": 0,
        "improved": 0,
        "unclean": 0,
        "no_model": 0,
        "found_by_both": 0,
        "model_calls": brief_cost.get("calls", 0),
        "briefing_calls": brief_cost.get("calls", 0),
        "rows": [],
        "f2": [],
    }

    for index, row in enumerate(rows):
        if not isinstance(row, MutableMapping) or not _is_voiced(row):
            continue
        receipt["voiced_rows"] += 1
        line_id = str(row.get("line_id") or "")
        speaker = str(row.get("speaker") or "").strip()

        # PYTHON OWNS PART OF THIS ROW. Hands off, before any judge call --
        # see PROTECTED_FACT_COMPONENT_FLAG. Recorded in its OWN list, never in
        # `rows`: an entry there means the judge read the row and something
        # rewrote it, which is exactly what must not have happened here.
        if PROTECTED_FACT_COMPONENT_FLAG in (row.get("compose_flags") or ()):
            receipt["protected_rows"].append(line_id)
            log.info(
                "[ledger_clean] %s carries a Python-owned fact component; "
                "skipped before judging (%s)",
                line_id or "<no line_id>", PROTECTED_FACT_COMPONENT_FLAG,
            )
            continue

        seen = receipt["context_seen"]
        seen["episode_context"] = episode
        if _story_field(row, beats, "arc_phase"):
            seen["rows_with_arc_phase"] += 1
        if _story_field(row, beats, "beat_intent"):
            seen["rows_with_beat_intent"] += 1
        if act_briefs.get(_story_field(row, beats, "arc_phase") or ""):
            seen["rows_with_act_brief"] += 1
        if cast.get(str(row.get("char_id") or "")):
            seen["rows_with_cast_name"] += 1
        if any(_is_voiced(r) for r in rows[max(0, index - _CONTEXT_ROWS):index]):
            seen["rows_with_lines_before"] += 1
        if any(_is_voiced(r) for r in rows[index + 1:index + 1 + _AFTER_ROWS]):
            seen["rows_with_lines_after"] += 1

        f2 = _POLICY.f2_findings(
            row,
            beats.get(str(row.get("beat_id") or "")),
            cast.get(str(row.get("char_id") or ""), ""),
        )
        if f2:
            receipt["f2_rows"] += 1
            receipt["f2"].append({
                "line_id": line_id,
                "findings": [{"kind": f.kind, "detail": f.detail} for f in f2],
            })

        # F2's CONTENT half, its own call, before F1 touches the text. It
        # runs first deliberately: if the line belongs to the wrong mouth,
        # fixing its stage business first would just polish the wrong
        # character's words.
        if JUDGE_ATTRIBUTION and slot_fn is not None:
            receipt["model_calls"] += 1
            belongs, likely, why = _f2_judge(
                slot_fn=slot_fn,
                speaker=speaker,
                text=_text_of(row),
                roster=[n for n in cast.values() if n],
                lines_around=_lines_around(rows, index, beats),
                where=_where_the_story_is(row, beats, episode, act_briefs),
            )
            if not belongs:
                receipt["f2_content_rows"] += 1
                receipt["f2"].append(_fix_attribution(
                    row=row, rows=rows, index=index, beats=beats,
                    episode=episode, act_briefs=act_briefs,
                    likely=likely, why=why,
                    roster=[n for n in cast.values() if n],
                    slot_fn=slot_fn, receipt=receipt,
                ))

        # The free detector runs first so the judge can be shown its evidence.
        # On the fidelity lanes the language kinds are inadmissible as a
        # TRIGGER -- the author's own third person is not a defect -- but the
        # judge still reads the line and its verdict counts everywhere.
        patterns = _POLICY.f1_findings(_text_of(row))
        hints = [f for f in patterns if f.kind in admissible]

        if slot_fn is None:
            if hints:
                receipt["f1_rows"] += 1
                receipt["pattern_only"] += 1
                # NOT "unclean": that number means "a model tried and could
                # not fix it". No model ran here at all, and conflating the
                # two made the receipt claim a failed repair that never
                # happened.
                receipt["no_model"] += 1
                _flag_unclean(row)
                receipt["rows"].append({
                    "line_id": line_id,
                    "outcome": "no_model",
                    "complaint": _as_findings(hints),
                })
            continue

        receipt["model_calls"] += JUDGE_VOTES
        judged, call_ok = _judge_votes(
            slot_fn=slot_fn,
            speaker=speaker,
            text=_text_of(row),
            hints=hints,
            lines_around=_lines_around(rows, index, beats),
            where=_where_the_story_is(row, beats, episode, act_briefs),
            sightings=sightings,
        )

        # A UNION, NOT A VETO. Whichever reader saw something, we act on it:
        # the model generalizes to the door nobody enumerated, the patterns
        # catch the blatant case instantly, and neither is allowed to
        # overrule the other.
        if judged and hints:
            source = "both"
            receipt["found_by_both"] += 1
        elif judged:
            source = "judge"
            receipt["judge_only"] += 1
        elif hints:
            source = "patterns"
            receipt["pattern_only"] += 1
        else:
            continue

        if judged:
            receipt["judged_dirty"] += 1
            receipt["segments_named"] += len(judged)
        receipt["f1_rows"] += 1
        complaint = judged or _as_findings(hints)

        receipt["rows"].append(_repair_row(
            row=row,
            rows=rows,
            index=index,
            beats=beats,
            complaint=complaint,
            source=source,
            episode=episode,
            act_briefs=act_briefs,
            judge_reachable=call_ok,
            admissible=admissible,
            slot_fn=slot_fn,
            receipt=receipt,
            sightings=sightings,
        ))

    receipt["context_seen"]["acts_summarized"] = len(act_briefs)
    # THE SHA VERDICT for the episode: did what we built reach what we sent?
    blind = [s for s in sightings if s["missing"]]
    # AND the per-prompt sight strings: a 0 in any prompt shows as a 0 for
    # the episode, so one glance at "11011" says the act went missing
    # somewhere without opening a single prompt.
    rolled = "".join(
        "1" if all(s["sight"][i] == "1" for s in sightings) else "0"
        for i in range(len(CONTEXT_FIELDS))
    ) if sightings else ""
    receipt["context_verified"] = {
        "sight": rolled,
        "sha": context_digest(*(s["sha"] for s in sightings)),
        "ok": bool(sightings) and not blind,
        "fields": list(CONTEXT_FIELDS),
        "prompts": len(sightings),
    }
    if blind:
        log.error(
            "[ledger_clean] CONTEXT DID NOT LAND on %d of %d prompt(s): %s "
            "was built and never reached the model. Wiring fault in this "
            "pass, not the model.",
            len(blind), len(sightings),
            ", ".join(sorted({n for s in blind for n in s["missing"]})),
        )
    elif sightings:
        log.info(
            "[ledger_clean] context %s sha %s across %d prompt(s) (%s)",
            rolled, receipt["context_verified"]["sha"], len(sightings),
            " ".join(CONTEXT_FIELDS),
        )
    _log_verdict(receipt)
    meta = ledger_data.setdefault("meta", {})
    if isinstance(meta, MutableMapping):
        meta["ledger_clean"] = receipt
    return receipt


def _repair_row(
    *,
    row: MutableMapping[str, Any],
    rows: "Sequence[Mapping[str, Any]]",
    index: int,
    beats: "Mapping[str, Mapping]",
    complaint: "Sequence[Mapping[str, str]]",
    source: str,
    episode: str,
    act_briefs: "Mapping[str, str]",
    judge_reachable: bool,
    admissible: "frozenset[str]",
    slot_fn: "Callable[..., str]",
    receipt: MutableMapping[str, Any],
    sightings: "list[dict[str, Any]] | None" = None,
) -> "dict[str, Any]":
    """Repair ONE row, bounded, the judge re-reading each attempt.

    Every segment the judge named goes into the SAME repair call, as a
    numbered checklist. One call fixes the whole line -- a per-segment loop
    would edit a line three times against three partial views of it and let
    each edit undo the last.
    """
    line_id = str(row.get("line_id") or "")
    speaker = str(row.get("speaker") or "").strip()
    original = _text_of(row)
    lines_around = _lines_around(rows, index, beats)
    where = _where_the_story_is(row, beats, episode, act_briefs)

    current = list(complaint)
    previous = ""
    attempts: "list[dict[str, Any]]" = []
    # PROGRESS, NOT PERFECTION. Measured in the lab on a 2B: the judge is
    # eager enough that it finds SOMETHING on almost every rewrite, so an
    # accept-only-when-spotless loop never converges -- every row burns its
    # whole budget and then ships the ORIGINAL, discarding rewrites that had
    # genuinely removed the stage direction. Keeping the best candidate turns
    # a grind into monotone improvement, and it is still the model's prose:
    # Python only picks which of the model's own answers to keep, exactly as
    # it already does when it accepts one.
    best_text = ""
    best_count = len(current)
    started_with = len(current)

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        receipt["model_calls"] += 1
        candidate = _call_repair(
            slot_fn=slot_fn,
            speaker=speaker,
            text=original,
            complaint=_numbered(current),
            lines_around=lines_around,
            previous_attempt=previous,
            where=where,
            sightings=sightings,
        )
        if not candidate:
            attempts.append({"attempt": attempt, "outcome": "call_failed"})
            break

        # READ IT BACK THE SAME WAY IT WAS READ THE FIRST TIME. A repair
        # graded by a weaker check than the one that condemned it is a repair
        # that can pass by moving the defect somewhere the check cannot see.
        still_patterns = [
            f for f in _POLICY.f1_findings(candidate) if f.kind in admissible
        ]
        still_judged: "list[dict[str, str]]" = []
        if judge_reachable:
            receipt["model_calls"] += 1
            still_judged, _ok = _judge_row(
                slot_fn=slot_fn,
                speaker=speaker,
                text=candidate,
                hints=still_patterns,
                lines_around=lines_around,
                where=where,
                sightings=sightings,
            )
        remaining = still_judged or _as_findings(still_patterns)
        attempts.append({
            "attempt": attempt,
            "outcome": (
                "clean" if not (still_judged or still_patterns)
                else "still_dirty"
            ),
            "remaining": remaining,
        })
        if not still_judged and not still_patterns:
            # THE CANONICAL OWNER sets the text, so `word_count` and
            # `char_count` move with it in one step. A bare `row["text"] = `
            # would leave the metrics describing the line this pass just
            # replaced, and every downstream consumer that budgets from them
            # would be reading a line nobody says.
            set_line_text_metrics(row, candidate)
            receipt["repaired"] += 1
            log.info(
                "[ledger_clean] %s repaired -- %s named %d segment(s) -- "
                "%r -> %r",
                line_id, source, len(complaint),
                original[:70], candidate[:70],
            )
            return {
                "line_id": line_id,
                "outcome": "repaired",
                "found_by": source,
                "complaint": list(complaint),
                "before": original,
                "after": candidate,
                "attempts": attempts,
            }
        if len(remaining) < best_count:
            best_text, best_count = candidate, len(remaining)
        previous = candidate
        current = remaining

    # BOUNDED, THEN TAKE THE BEST ANSWER AND FLAG IT. If some attempt left
    # the line strictly cleaner than it started, that rewrite SHIPS -- a line
    # with one problem left is better on air than a line with three, and
    # throwing it away to preserve the original was losing real repairs. The
    # flag stays either way, so nothing is quietly declared fixed.
    if best_text and best_count < started_with:
        set_line_text_metrics(row, best_text)
        receipt["improved"] += 1
        log.warning(
            "[ledger_clean] %s could not be made spotless in %d pass(es), but "
            "the best rewrite cut its problems from %d to %d -- SHIPPING that "
            "one, flagged: %r -> %r",
            line_id, len(attempts), started_with, best_count,
            original[:60], best_text[:60],
        )
        _flag_unclean(row)
        return {
            "line_id": line_id,
            "outcome": "improved",
            "found_by": source,
            "complaint": list(complaint),
            "before": original,
            "after": best_text,
            "remaining_count": best_count,
            "attempts": attempts,
        }

    _flag_unclean(row)
    receipt["unclean"] += 1
    log.error(
        "[ledger_clean] %s is STILL unclean after %d bounded repair pass(es): "
        "%s. The row SHIPS with compose_flag %r and the render continues -- "
        "a render is never killed for this.",
        line_id, len(attempts),
        "; ".join(str(f.get("quote") or "") for f in current) or "(nothing "
        "named)",
        UNCLEAN_COMPOSE_FLAG,
    )
    return {
        "line_id": line_id,
        "outcome": "unclean",
        "found_by": source,
        "complaint": list(complaint),
        "text": original,
        "attempts": attempts,
    }


def _fix_attribution(
    *,
    row: MutableMapping[str, Any],
    rows: "Sequence[Mapping[str, Any]]",
    index: int,
    beats: "Mapping[str, Mapping]",
    episode: str,
    act_briefs: "Mapping[str, str]",
    likely: str,
    why: str,
    roster: "Sequence[str]",
    slot_fn: "Callable[..., str]",
    receipt: MutableMapping[str, Any],
) -> "dict[str, Any]":
    """Rewrite one misattributed line into its stated speaker's voice.

    Bounded and fail-soft like everything else here: if the fix cannot be
    made, the row SHIPS with a flag and the render continues. A misattributed
    line on air is bad; a dead episode is worse.
    """
    line_id = str(row.get("line_id") or "")
    speaker = str(row.get("speaker") or "").strip()
    original = _text_of(row)
    lines_around = _lines_around(rows, index, beats)
    where = _where_the_story_is(row, beats, episode, act_briefs)

    previous = ""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        receipt["model_calls"] += 1
        candidate = _f2_fix(
            slot_fn=slot_fn, speaker=speaker, text=original,
            likely_speaker=likely, why=why,
            lines_around=lines_around, where=where,
            previous_attempt=previous,
        )
        if not candidate:
            break
        receipt["model_calls"] += 1
        belongs, _likely2, why2 = _f2_judge(
            slot_fn=slot_fn, speaker=speaker, text=candidate,
            roster=roster,
            lines_around=lines_around, where=where,
        )
        if belongs:
            set_line_text_metrics(row, candidate)
            receipt["f2_reattributed"] += 1
            log.info(
                "[ledger_clean] %s reattributed into %s's voice (%s) -- "
                "%r -> %r",
                line_id, speaker or "the speaker", why,
                original[:60], candidate[:60],
            )
            return {
                "line_id": line_id, "outcome": "reattributed",
                "why": why, "likely_speaker": likely,
                "before": original, "after": candidate,
            }
        previous, why = candidate, why2 or why

    # THE ORIGINAL IS KNOWN WRONG -- the judge said so, which is why we are
    # here. So a rewrite the judge merely failed to BLESS is not a coin flip
    # against a good line; it is a coin flip against a line already condemned,
    # aimed at the right mouth. Operator, 2026-08-14: *"at some point it needs
    # to be CORRECTED, not fail the whole thing."* It ships, and it ships
    # FLAGGED, so nothing is quietly declared fixed.
    #
    # Measured in the lab before this existed: the fix produced candidates on
    # every caught row and the judge blessed NONE of them, so F2 was
    # report-only in practice -- the exact behaviour the operator rejected.
    if previous:
        set_line_text_metrics(row, previous)
        _flag_misattributed(row)
        receipt["f2_reattributed_unverified"] += 1
        log.warning(
            "[ledger_clean] %s was rewritten for %s but the judge would not "
            "confirm it (%s). The REWRITE ships, flagged -- the original was "
            "already condemned: %r -> %r",
            line_id, speaker or "the speaker", why,
            original[:60], previous[:60],
        )
        return {
            "line_id": line_id, "outcome": "reattributed_unverified",
            "why": why, "likely_speaker": likely,
            "before": original, "after": previous,
        }

    _flag_misattributed(row)
    receipt["f2_unfixed"] += 1
    log.error(
        "[ledger_clean] %s STILL reads as the wrong character's speech after "
        "%d bounded pass(es): %s. The row SHIPS flagged and the render "
        "continues -- a render is never killed for this.",
        line_id, _MAX_ATTEMPTS, why,
    )
    return {
        "line_id": line_id, "outcome": "misattributed_unfixed",
        "why": why, "likely_speaker": likely, "text": original,
    }


def _log_verdict(receipt: Mapping[str, Any]) -> None:
    """One line the operator can read without opening the ledger."""
    if not receipt["f1_rows"] and not receipt["f2_rows"]:
        log.info(
            "[ledger_clean] %d voiced row(s) read by the judge, nothing to "
            "clean (%s)",
            receipt["voiced_rows"], receipt["policy"],
        )
        return
    log.info(
        "[ledger_clean] %d voiced row(s): %d carried something that is not "
        "speech (%d segment(s) named), F2 on %d. %d repaired, %d improved, "
        "%d still unclean, %d with no model, in %d model call(s) (%d of them "
        "briefing). Found by: judge only %d, patterns only %d, both %d.",
        receipt["voiced_rows"], receipt["f1_rows"],
        receipt["segments_named"], receipt["f2_rows"],
        receipt["repaired"], receipt["improved"], receipt["unclean"],
        receipt["no_model"], receipt["model_calls"],
        receipt["briefing_calls"],
        receipt["judge_only"], receipt["pattern_only"],
        receipt["found_by_both"],
    )
    if receipt["judge_only"]:
        log.info(
            "[ledger_clean] %d row(s) were caught ONLY by the judge -- no "
            "pattern would have found them. That is the reason the judge is "
            "a model.",
            receipt["judge_only"],
        )
    if receipt["f2_rows"]:
        log.error(
            "[ledger_clean] %d row(s) disagree about who is speaking -- this "
            "is F2 and it is a bookkeeping fault upstream, not something a "
            "rewrite can fix. See meta.ledger_clean.f2 for every one.",
            receipt["f2_rows"],
        )
