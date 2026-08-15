"""What "action in a spoken row" means -- ONE definition, shared by everyone.

Two consumers read this module and there must never be a third opinion:

  * ``scripts/otr_ledger_view.py`` GRADES a finished episode with it.
  * ``nodes/_otr_ledger_clean.py`` GATES the repair pass with it.

If the grader and the repair disagreed about what a defect is, a run could
repair a line the grader still fails, or ship a line the grader would have
caught. So the patterns live here and nowhere else.

WHAT IS GRADED, AND WHAT IS NOT
-------------------------------
Operator ruling 2026-08-14, and it is only two things:

    F1  ACTION IN THE LEDGER -- a spoken row carrying stage business,
        delivery notes or narration. Every sealed line becomes TTS audio,
        so a stage direction inside a line gets read aloud on air.
    F2  SPEAKER / LINE MISMATCH -- a row carrying another character's
        words, or carrying no speaker at all.

Nothing else is a failure. Prose quality is explicitly not a defect.

This module DETECTS and explains. It never edits a row: rewriting spoken
prose is a model's job, never Python's.

THESE PATTERNS ARE A FLOOR, NOT THE AUTHORITY
----------------------------------------------
Enumerating stage business is not a finite job. This list has ``sighs`` and
``turns`` in it and does not have ``closes``, so "The door closes behind him"
walks straight past it -- and the next story will invent something else the
list has never seen. Operator, 2026-08-14: *"your shim can clean a contained
story but it won't fix the next one."*

So in production the JUDGE IS A MODEL (``nodes/_otr_ledger_clean.py``), and
what is here is handed to it as evidence, labelled as often wrong. A line is
dirty if the model says so OR these patterns do -- a union, never a veto.
What this file is genuinely good at is being free and instant, which is why
it stays: it costs nothing to look, and the offline grader needs a verdict it
can compute over a thousand episodes without a GPU.

The patterns are deliberately noisy in one direction. A false positive costs
a human five seconds of reading; a false negative ships a stage direction to
the microphone. Every finding names the exact phrase it tripped on so a
reader -- or the judge model -- can dismiss it at a glance.

THE FIDELITY LANES
------------------
On ``shakespeare`` and ``public_domain`` the author's own language is carried
as written, so a third-person construction that came from the source is not a
defect -- the detector cannot tell those apart and over-reports there. What it
can still judge on those lanes is MARKUP: a bracketed stage direction or a
"NAME:" label is production apparatus, never an author's line, and it gets
read aloud exactly like the rest. So the fidelity lanes repair markup and
merely REPORT language.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, NamedTuple

__all__ = [
    "SPOKEN_TEXT_POLICY_ID",
    "FIDELITY_BANKS",
    "MARKUP_KINDS",
    "F1_PATTERNS",
    "Finding",
    "f1_findings",
    "f2_findings",
    "repairable_kinds",
    "summarize",
    "VOICED_ROLES",
]

#: Bumped when a pattern changes meaning, so a receipt names the rules it
#: was graded under rather than implying today's rules judged an old run.
SPOKEN_TEXT_POLICY_ID = "otr.spoken-text-only.v1"

#: Roles whose rows are actually delivered to a voice engine.
VOICED_ROLES = ("character", "announcer")

#: Banks that carry the source author's own language as written.
FIDELITY_BANKS = frozenset({"shakespeare", "public_domain"})


class Finding(NamedTuple):
    """One defect in one row: which rule fired, and on which words."""

    kind: str
    detail: str


# ---------------------------------------------------------------------------
# F1 -- action inside a row that is supposed to be pure speech
# ---------------------------------------------------------------------------
# Each pattern is one WAY narration leaks in, kept separate so a finding can
# say which one fired instead of "looks like prose".

F1_PATTERNS: "tuple[tuple[str, re.Pattern[str]], ...]" = (
    (
        "stage direction in brackets",
        re.compile(r"\([^)]{2,}\)|\[[^\]]{2,}\]|\*[^*]{2,}\*"),
    ),
    (
        "attribution verb",
        re.compile(
            r"\b(?:said|says|saying|murmurs?|murmured|whispers?|whispered"
            r"|shouts?|shouted|replies|replied|asks|asked|answers?|answered"
            r"|adds?|added|continues?|continued|mutters?|muttered|snaps?"
            r"|snapped|breathes?|breathed|calls? out|called out)\b",
            re.IGNORECASE,
        ),
    ),
    (
        # A BARE VERB IS NOT EVIDENCE. The first cut matched the verb alone
        # and fired six times out of six on a ledger that was clean: "We
        # stand at the precipice" (first person), "humanity to finally stand"
        # (an infinitive), "Then step back, Kaelen" (an imperative spoken to
        # another character), "it stands as our shield" (a metaphor). None of
        # those is stage business. What makes an action a STAGE DIRECTION is
        # a third-person subject DOING it, so the subject is now part of the
        # pattern -- and `to` in front rules out the infinitive.
        "third-person action",
        re.compile(
            r"(?<!to )\b(?:He|She|They|"
            # A capitalised word is only a SUBJECT if it is not the ordinary
            # way an English sentence opens. "Then step back" and "The vest
            # stands" are a command and a metaphor, not stage business.
            r"(?!(?:Then|But|And|When|If|Now|So|Let|Well|The|This|That|These"
            r"|Those|Here|There|Why|How|What|Who|Since|While|After|Before"
            r"|Today|Tonight|We|You|I|My|Your|Our|It|Its|Not|Just|Only|Even"
            r"|Maybe|Perhaps|Yes|No|Every|Each|Some|All)\b)"
            r"[A-Z][a-z]{2,})\s+"
            r"(?:turns?|turned|leans?|leaned|reaches?|reached|walks?|walked"
            r"|steps?|stepped|nods?|nodded|smiles?|smiled|frowns?|frowned"
            r"|glances?|glanced|stares?|stared|drums?|drummed|paces?|paced"
            r"|grabs?|grabbed|enters?|entered|leaves?|exits?|exited|stands?"
            r"|stood|sits?|sat|rises?|rose|sighs?|sighed|shrugs?|shrugged"
            r"|laughs?|laughed|pauses?|paused)\b",
        ),
    ),
    (
        "delivery note",
        re.compile(
            r"\b(?:her|his|their) (?:voice|eyes|hands?|face|gaze|tone"
            r"|breath|shoulders?|fingers?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        # STRAIGHT AND CURLY DOUBLE QUOTES ONLY. The first cut included the
        # curly SINGLE quotes, which in ordinary English are apostrophes --
        # so "It's designed to block the solar storms" read as quoted speech
        # embedded in prose. Two of the six false alarms were this.
        "quoted speech inside prose",
        re.compile("[\"“”].{8,}[\"“”]"),
    ),
    (
        "speaker label",
        re.compile(r"^\s*[A-Z][A-Z .'’-]{1,30}:", re.MULTILINE),
    ),
)

#: The kinds that are production APPARATUS rather than English. A source
#: author never wrote "(he turns away)" or "MACBETH:" into a character's
#: speech -- those come from a script format, so they are a defect on every
#: bank including the fidelity lanes, and they are read aloud if left in.
MARKUP_KINDS = frozenset({
    "stage direction in brackets",
    "speaker label",
})


def _first_match(text: str, pattern: "re.Pattern[str]") -> str:
    found = pattern.search(text)
    if not found:
        return ""
    snippet = " ".join(found.group(0).split())
    return snippet[:60]


def f1_findings(text: str) -> "list[Finding]":
    """Every F1 rule that fires on one row's text, with the phrase it hit."""
    findings: "list[Finding]" = []
    for kind, pattern in F1_PATTERNS:
        hit = _first_match(text, pattern)
        if hit:
            findings.append(Finding(kind, hit))
    return findings


def f2_findings(
    row: Mapping[str, Any],
    beat: "Mapping[str, Any] | None",
    cast_name: str = "",
) -> "list[Finding]":
    """Whether this row agrees with the beat and the cast about who speaks.

    This is the METADATA half of F2 and it is all Python can honestly judge.
    The content half -- a character speaking another character's words -- is
    a reading of the prose, and there is no detector for it, so nothing here
    pretends to find one.
    """
    speaker = str(row.get("speaker") or "").strip()
    owner = str((beat or {}).get("speaker") or "").strip()
    if not speaker:
        return [Finding(
            "no speaker", "the row does not say who is talking")]
    if owner and speaker != owner:
        return [Finding(
            "row disagrees with its beat",
            f"row says {speaker!r}, beat says {owner!r}")]
    if row.get("speaker_role") == "character" and cast_name:
        if speaker != cast_name:
            return [Finding(
                "row disagrees with the cast",
                f"row says {speaker!r}, cast says {cast_name!r}")]
    return []


def repairable_kinds(bank_id: str) -> "frozenset[str]":
    """Which F1 kinds may spend a model call on this bank.

    Everywhere else, all of them: the lane authored the line itself, so any
    narration in it is the writer's leak and is worth an informed rewrite.

    On the fidelity lanes only the MARKUP kinds, because the language kinds
    over-report there by design -- the source author's own third person is
    not a defect, and burning a repair call to "fix" Shakespeare would be
    the fidelity failure this project already ruled against. The language
    findings are still REPORTED on those lanes; they are a reading list.
    """
    if str(bank_id or "").strip() in FIDELITY_BANKS:
        return frozenset(MARKUP_KINDS)
    return frozenset(kind for kind, _pattern in F1_PATTERNS)


def summarize(findings: "Iterable[Finding]") -> str:
    """One human-readable line naming every rule that fired and its phrase."""
    return "; ".join(f"{f.kind}: {f.detail}" for f in findings)
