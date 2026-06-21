"""nodes/_otr_ledger_scrub.py -- Stream E: deterministic Ledger Scrub (Stage 4).

The final mechanical gate of the OTR story spine
(docs/otr-story-spine.md Section 3, "Stream E"). One pure-Python pass that
AGGREGATES the guards the pipeline already trusts -- it imports and CALLS
them, it does NOT reimplement them and it never invokes an LLM:

  * JSON / schema validity .......... the ledger dict round-trips through
                                      json.dumps / json.loads, and each
                                      lines[] row keeps its load-bearing keys.
  * cast-name leak filter (BUG-279/F3) `_otr_line_composer.detect_phantom_names`
                                      + `build_allowed_roster` -- the same
                                      proper-noun-vs-roster detector the
                                      composer uses; plus the composer's
                                      speaker-self / ANNOUNCER degenerate-leak
                                      shape (verbatim from compose_line, lines
                                      1680-1709). A leaked OTHER cast name or a
                                      leaked structural token mid-phrase is a
                                      MECHANICAL defect, not story content.
  * SFW / profanity validator ....... `_otr_stage3_validators.DEFAULT_PROFANITY_TERMS`
                                      + `CODE_SFW_VIOLATION`, matched with the
                                      exact whole-word / phrase semantics of
                                      `_otr_stage3_validators.validate_sfw`
                                      (word-boundary regex; "hell" fires on
                                      "go to hell" but not "hello"). The
                                      wordlist is imported LIVE so any term the
                                      main thread adds flows through with no
                                      edit here.
  * three cast assertions ........... `_otr_casting._assert_no_structural_tokens_in_cast`
                                      (:1580), `_assert_voice_preset_invariant`
                                      (:1466), `_assert_unique_bark_voices`
                                      (:1619).
  * line-level normalization ........ `_otr_line_composer.strip_line_formatting`
                                      (:141) for leaked prefixes / brackets /
                                      markdown / wrapping quotes, plus
                                      whitespace, smart-quote and
                                      stray-stage-direction cleanup; no
                                      empty / duplicate spoken lines; TTS-safe
                                      text; per-line + total length bounds.

SCOPE RULE (spine Sec 3, Stream E): the scrub NORMALIZES freely
(whitespace, quotes, punctuation, JSON-safety). It does NOT improve the
story, rewrite dull lines, or change the ending. When a *mechanical*
defect needs regeneration rather than normalization -- an empty spoken
line, or a leaked cast name / structural token mid-phrase that the
deterministic strips cannot repair in place -- the scrub raises a
NEEDS_EDITOR_REPAIR signal that consumes the single shared repair ONLY
if it has not already been used (the caller passes `repair_available`);
otherwise it fails closed.

  scrub_ledger(led, *, repair_available: bool) -> ScrubResult

ScrubResult is deterministic and carries no model. See the dataclass
below for the full shape.

--------------------------------------------------------------------------
SFW wordlist note (P8 / spine Sec 3 + Sec 6): the `damn(ed)` SFW addition
belongs to an EXISTING file, NOT here. The past-tense form ``"damned"`` is
absent from `DEFAULT_PROFANITY_TERMS` (the list carries "damn", "damnit",
"dammit", "goddamn" but not "damned"). The main thread should add the
string ``"damned"`` to:

    nodes/_otr_stage3_validators.py
        -> DEFAULT_PROFANITY_TERMS   (the list defined at ~line 140; place
                                      the new entry alongside "damn" at
                                      ~line 143)

This scrub imports that list live, so once the term lands the scrub
enforces it with zero change to this module.
--------------------------------------------------------------------------

PD3 (workflow JSON): N/A; this module adds no node surface, no widget,
and no output socket. It is a pure helper called in-process AFTER the
post-script LLM unload (spine D8), so it needs no resident model.

This module is DORMANT until the writer wires the Stage-4 call. It mirrors
the "dormant until wired" header convention of
nodes/_otr_constrained_generate.py:1-27.

NO LLM. No torch / GPU / network at import (heavy siblings are lazy-imported
inside functions). Deterministic: same ledger in -> same ScrubResult out.
"""
# LLM slot: none -- Stream E is a deterministic, NO-LLM mechanical gate.
# Reason: per spine Sec 3 (Stream E) + invariant 4, the final scrub is pure
# Python wiring existing guards; it routes no model call through either
# writer slot, so it carries no creative/technical tag.
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Bare leading stage-direction floor (2026-06-22). _otr_line_hygiene is a
# stdlib-only leaf (imports only `re`) -> no import cycle. Dual import keeps the
# module loadable both as a package node and standalone.
try:  # pragma: no cover - exercised by both import styles
    from ._otr_line_hygiene import (
        BARE_STAGE_FLOOR_ACTIVE,
        scrub_leading_stage_direction,
    )
except ImportError:  # pragma: no cover
    from _otr_line_hygiene import (  # type: ignore
        BARE_STAGE_FLOOR_ACTIVE,
        scrub_leading_stage_direction,
    )

__all__ = [
    "ScrubResult",
    "ScrubFinding",
    "NeedsEditorRepair",
    "scrub_ledger",
    "is_spoken_role",
]


# ---------------------------------------------------------------------------
# Tunables -- per-line + total length bounds (TTS / Bark timing safety).
#
# These mirror Stream D Guard 2 (spine Sec 3): one breath per line. They are
# generous ceilings, NOT a content target -- the scrub only flags a runaway
# spoken line, it never shortens for taste.
# ---------------------------------------------------------------------------

_MAX_WORDS_PER_LINE: int = 60      # generous one-breath runaway guard
_MAX_CHARS_PER_LINE: int = 400     # fixed char ceiling (Bark timing)
_TOTAL_WORDS_HARD_CEILING: int = 1200  # whole-episode spoken-word runaway guard

# Spoken roles whose text rides the TTS path and must be clean dialogue.
# music / sfx rows are render contracts, not speech -- the scrub never
# touches them (it never sees them as "spoken").
_SPOKEN_ROLES: frozenset[str] = frozenset({"character", "announcer"})


def is_spoken_role(role: Any) -> bool:
    """True when a ledger row's speaker_role is a voiced/dialogue role.

    Canonical single source of truth for "does this row's `text` ride the
    TTS + caption track". Character / announcer rows are voiced; music_open /
    music_inter / music_close / sfx rows are render contracts (S1: their
    `text` must never carry generic beat intent into the spoken/caption
    track). Defensive against None / non-str input.
    """
    return str(role or "") in _SPOKEN_ROLES

# Stage-direction shapes the scrub strips from a spoken field when they are
# clearly leaked cues rather than dialogue. Mirrors the composer's
# narration-leak family but is applied as a NORMALIZER (cut the cue, keep the
# speech), not a retry. A line that is NOTHING BUT a cue collapses to empty
# and is then handled as a mechanical defect.
_STAGE_DIRECTION_RES: Tuple[re.Pattern, ...] = (
    re.compile(r"\[[^\]]{1,80}\]"),      # [walks across]
    re.compile(r"\*[^*]{1,80}\*"),       # *sighs*
    re.compile(
        r"\(([^)]{1,80})\)",             # (sighs) / (pause) -- cue parens only
    ),
)

# Parenthetical cue verbs: a "(...)" group is treated as a leaked stage
# direction only when it reads like a cue. A real aside ("(I think)") is left
# alone. Conservative -- the scrub prefers to keep speech over cutting it.
_CUE_VERB_RE = re.compile(
    r"\b(?:sigh|pause|beat|laughs?|smiles?|gestures?|nods?|shrugs?|cough|"
    r"whispers?|mutters?|shouts?|exits?|enters?|walks?|turns?|stares?|"
    r"looks?\s+away)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result + signal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScrubFinding:
    """One deterministic scrub finding on the ledger.

    code     short machine code (see the CODE_* set used in this module)
    where    "lines[<i>]" / "cast" / "ledger" -- which part tripped
    detail   human-readable description
    line_id  owning line_id when the finding is line-scoped, else None
    """

    code: str
    where: str
    detail: str
    line_id: Optional[str] = None


@dataclass
class ScrubResult:
    """Deterministic outcome of scrub_ledger. Carries NO model.

    status            one of "PASS", "NEEDS_EDITOR_REPAIR", "FAIL".
                        * PASS                -- the ledger is clean, or was
                                                 made clean by in-place
                                                 normalization (see
                                                 `normalized`). Safe for TTS.
                        * NEEDS_EDITOR_REPAIR -- a MECHANICAL defect needs
                                                 regeneration AND the single
                                                 shared repair was still
                                                 available (consumed here).
                                                 The caller runs the one editor
                                                 repair on `repair_targets`.
                        * FAIL                -- a hard stop: a mechanical
                                                 defect needs regeneration but
                                                 the repair was already used
                                                 (fail-closed), OR an invariant
                                                 the scrub cannot normalize was
                                                 violated (bad JSON, cast guard,
                                                 SFW slip).
    ledger            the (possibly normalized-in-place) ledger dict. On PASS
                        with `normalized` set, this is the cleaned ledger the
                        caller should persist.
    findings          every ScrubFinding raised this pass (all severities).
    normalized        True iff the scrub mutated any spoken field in place.
    repair_consumed   True iff this call consumed the single shared repair.
    repair_targets    line_ids the editor repair should regenerate (only
                        populated when status == NEEDS_EDITOR_REPAIR).
    sfw_violations     spoken-text profanity matches (term, line_id). Non-empty
                        forces FAIL -- profanity is a hard PD4 ship-stop the
                        scrub does not normalize away.
    """

    status: str
    ledger: Dict[str, Any]
    findings: List[ScrubFinding] = field(default_factory=list)
    normalized: bool = False
    repair_consumed: bool = False
    repair_targets: List[str] = field(default_factory=list)
    sfw_violations: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True iff the ledger is shippable to TTS without a repair."""
        return self.status == "PASS"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "normalized": self.normalized,
            "repair_consumed": self.repair_consumed,
            "repair_targets": list(self.repair_targets),
            "sfw_violations": [list(v) for v in self.sfw_violations],
            "findings": [
                {
                    "code": f.code,
                    "where": f.where,
                    "detail": f.detail,
                    "line_id": f.line_id,
                }
                for f in self.findings
            ],
        }


class NeedsEditorRepair(RuntimeError):
    """Raised internally when a mechanical defect needs the one editor repair.

    scrub_ledger catches this and converts it into a ScrubResult
    (NEEDS_EDITOR_REPAIR when the repair was available, FAIL when not), so
    callers normally read `ScrubResult.status` rather than catching this.
    Exposed in __all__ for callers that prefer the exception surface.
    """

    def __init__(self, targets: List[str], findings: List[ScrubFinding]) -> None:
        self.targets = list(targets)
        self.findings = list(findings)
        super().__init__(
            "ledger scrub: mechanical defect needs the single editor repair "
            f"on {self.targets!r}"
        )


# Machine codes for findings (kept short + greppable in meta trails).
CODE_JSON_INVALID = "json_invalid"
CODE_LEDGER_SHAPE = "ledger_shape"
CODE_LINE_NORMALIZED = "line_normalized"
CODE_STAGE_DIRECTION = "stage_direction_stripped"
CODE_EMPTY_LINE = "empty_spoken_line"
CODE_DUP_LINE = "duplicate_spoken_line"
CODE_CAST_NAME_LEAK = "cast_name_leak"
CODE_OVERLONG_LINE = "overlong_spoken_line"
CODE_TOTAL_OVERLONG = "episode_overlong"
CODE_SFW_VIOLATION = "sfw_violation"
CODE_CAST_GUARD = "cast_guard"


# ---------------------------------------------------------------------------
# Lazy guard imports -- heavy siblings load INSIDE functions, never at module
# import. Each helper returns the imported object (or a tolerant fallback so a
# standalone / partial tree still runs the self-test).
# ---------------------------------------------------------------------------


def _imp_line_composer():
    """Import the composer guards (strip + roster + phantom detector)."""
    try:
        from . import _otr_line_composer as lc  # type: ignore
    except ImportError:  # pragma: no cover - standalone / test load
        import _otr_line_composer as lc  # type: ignore
    return lc


def _imp_casting():
    """Import the three cast assertions + the CastingFailedError type."""
    try:
        from . import _otr_casting as cast  # type: ignore
    except ImportError:  # pragma: no cover - standalone / test load
        import _otr_casting as cast  # type: ignore
    return cast


def _imp_sfw():
    """Import the live SFW wordlist + code. Falls back to a minimal list so a
    partial tree can still run the self-test (the real list is the source of
    truth in production)."""
    try:
        try:
            from . import _otr_stage3_validators as v  # type: ignore
        except ImportError:  # pragma: no cover - standalone / test load
            import _otr_stage3_validators as v  # type: ignore
        terms = list(getattr(v, "DEFAULT_PROFANITY_TERMS", []))
        code = getattr(v, "CODE_SFW_VIOLATION", CODE_SFW_VIOLATION)
        if terms:
            return terms, code
    except Exception:  # noqa: BLE001 - any import failure -> safe fallback
        pass
    # Minimal fallback wordlist (self-test only; production imports the live
    # DEFAULT_PROFANITY_TERMS, which is broader and the single source of truth).
    return (["damn", "hell", "goddamn"], CODE_SFW_VIOLATION)


def _imp_json():
    """Import the shared tolerant JSON object parser (round-trip helper)."""
    try:
        try:
            from . import _otr_json as j  # type: ignore
        except ImportError:  # pragma: no cover - standalone / test load
            import _otr_json as j  # type: ignore
        return j
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Normalization helpers (pure string; no story judgement)
# ---------------------------------------------------------------------------

# Smart-quote / dash normalization -> TTS-safe ASCII-ish punctuation. Bark and
# Kokoro read straight quotes more reliably than curly ones.
_SMART_MAP = {
    "“": '"', "”": '"',     # curly double quotes
    "‘": "'", "’": "'",     # curly single quotes / apostrophe
    "–": "-", "—": "-",     # en / em dash
    "…": "...",                  # ellipsis
    " ": " ",                    # non-breaking space
}
_SMART_RE = re.compile("|".join(re.escape(k) for k in _SMART_MAP))
_WS_RUN_RE = re.compile(r"[ \t\r\f\v]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:])")
_MULTI_PUNCT_RE = re.compile(r"([,;:])\1+")


def _normalize_whitespace_and_quotes(text: str) -> str:
    """Collapse whitespace, normalize smart quotes / dashes, tidy spacing
    around punctuation. Pure mechanical cleanup -- never changes words."""
    if not text:
        return ""
    s = _SMART_RE.sub(lambda m: _SMART_MAP[m.group(0)], text)
    # Collapse internal newlines to a single space (a spoken line is one
    # breath; a soft break is not meaningful to TTS).
    s = s.replace("\n", " ").replace("\r", " ")
    s = _WS_RUN_RE.sub(" ", s)
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = _MULTI_PUNCT_RE.sub(r"\1", s)
    return s.strip()


def _strip_stage_directions(text: str) -> Tuple[str, bool]:
    """Remove leaked stage-direction shapes from a spoken field.

    Returns (cleaned_text, stripped_any). Brackets / asterisk-actions are
    always cut; parenthetical groups are cut only when they read like a cue
    verb (so a genuine aside survives). NORMALIZER, not a retry: a line that
    is nothing but a cue collapses to empty and is handled downstream.
    """
    if not text:
        return "", False
    original = text
    out = text
    for rx in _STAGE_DIRECTION_RES:
        if rx.pattern.startswith(r"\("):
            # Parens: only strip cue-like groups.
            def _maybe(m: "re.Match") -> str:
                inner = m.group(1)
                return "" if _CUE_VERB_RE.search(inner) else m.group(0)
            out = rx.sub(_maybe, out)
        else:
            out = rx.sub("", out)
    out = _WS_RUN_RE.sub(" ", out).strip()
    # BARE (undelimited) leading stage direction -- the FREEZE-only floor
    # (2026-06-22). Runs AFTER the delimited strips on the already-cleaned text;
    # narrow + guarded (see _otr_line_hygiene.scrub_leading_stage_direction).
    # Gated on the precision-validated flag so the contract stays a no-op when
    # disabled. The Tuple[str,bool] return is preserved (callers unpack it).
    if BARE_STAGE_FLOOR_ACTIVE:
        bare = scrub_leading_stage_direction(out)
        if bare != out:
            out = _WS_RUN_RE.sub(" ", bare).strip()
    return out, (out != original)


# ---------------------------------------------------------------------------
# Cast-name leak detection (BUG-279/F3) -- wires the composer's detector.
# ---------------------------------------------------------------------------


def _speaker_self_or_announcer_leak(text: str, speaker: str) -> bool:
    """Verbatim port of the composer's degenerate-leak shape
    (_otr_line_composer.compose_line, lines 1680-1709): a spoken line that is
    NOTHING BUT the speaker's own label or the literal 'ANNOUNCER' is a leak,
    not dialogue. Saying ANOTHER cast member's name as a one-word line is
    legitimate drama and is NOT a leak here.
    """
    norm = (text or "").strip().rstrip(".!?,;:").upper()
    spk = (speaker or "").strip().upper()
    if not norm:
        return False
    return norm == spk or norm == "ANNOUNCER"


def _detect_cast_name_leak(
    lc,
    text: str,
    speaker: str,
    allowed_roster: "frozenset[str]",
) -> List[str]:
    """Return leaked-name tokens in `text` that are a MECHANICAL defect.

    Two layers, both deterministic:
      1. The composer's `detect_phantom_names` -- proper nouns not in the
         allowed roster (cast names + key terms + ANNOUNCER). A phantom is a
         leaked entity the locked cast never declared.
      2. The composer's speaker-self / ANNOUNCER degenerate-leak shape.

    The roster is built from the FULL cast, so a line that simply NAMES another
    real cast member is on-roster and clears -- per the composer, that is
    legitimate drama, not a leak. Only genuinely undeclared names + self/label
    leaks are returned.
    """
    leaks: List[str] = []
    try:
        phantoms = lc.detect_phantom_names(text, speaker, allowed_roster)
    except Exception:  # noqa: BLE001 - detector is defensive, but be safe
        phantoms = []
    for p in phantoms:
        if p not in leaks:
            leaks.append(p)
    if _speaker_self_or_announcer_leak(text, speaker):
        token = (speaker or "ANNOUNCER").strip().upper()
        if token not in leaks:
            leaks.append(token)
    return leaks


# ---------------------------------------------------------------------------
# SFW check -- reuses the live wordlist + validate_sfw word-boundary semantics.
# ---------------------------------------------------------------------------


def _sfw_violations_in(text: str, terms: List[str]) -> List[str]:
    """Return profanity terms present in `text`, using the EXACT match shape of
    `_otr_stage3_validators.validate_sfw`: case-insensitive; single tokens
    matched on word boundaries ("hell" hits "go to hell" but not "hello");
    multi-word entries matched as substrings. The term list is imported live so
    any entry the main thread adds (e.g. "damned") flows through with no edit
    here.
    """
    if not text:
        return []
    hits: List[str] = []
    low = text.lower()
    for term in terms:
        t = (term or "").strip().lower()
        if not t:
            continue
        if " " in t:
            if t in low and term not in hits:
                hits.append(term)
        else:
            if re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE):
                if term not in hits:
                    hits.append(term)
    return hits


# ---------------------------------------------------------------------------
# Cast assertions -- call the three _otr_casting guards.
# ---------------------------------------------------------------------------


def _run_cast_assertions(cast_obj, cast_rows: List[dict]) -> List[ScrubFinding]:
    """Call the three cast guards; convert any CastingFailedError into a
    ScrubFinding rather than letting it propagate. Returns the findings list
    (empty when all three pass). The assertions are authoritative -- the scrub
    only reports, it does not weaken them.
    """
    findings: List[ScrubFinding] = []
    CastErr = getattr(cast_obj, "CastingFailedError", Exception)
    guard_fns = (
        ("_assert_no_structural_tokens_in_cast",
         getattr(cast_obj, "_assert_no_structural_tokens_in_cast", None)),
        ("_assert_voice_preset_invariant",
         getattr(cast_obj, "_assert_voice_preset_invariant", None)),
        ("_assert_unique_bark_voices",
         getattr(cast_obj, "_assert_unique_bark_voices", None)),
    )
    for name, fn in guard_fns:
        if fn is None:
            continue
        try:
            fn(cast_rows)
        except CastErr as exc:
            findings.append(ScrubFinding(
                code=CODE_CAST_GUARD,
                where="cast",
                detail=f"{name} failed: {exc}",
            ))
        except Exception as exc:  # noqa: BLE001 - defensive; report, don't crash
            findings.append(ScrubFinding(
                code=CODE_CAST_GUARD,
                where="cast",
                detail=f"{name} raised {type(exc).__name__}: {exc}",
            ))
    return findings


# ---------------------------------------------------------------------------
# JSON round-trip + ledger-shape check.
# ---------------------------------------------------------------------------


def _check_json_round_trip(led: Dict[str, Any]) -> List[ScrubFinding]:
    """The ledger must serialize to JSON and parse back unchanged. Uses the
    shared tolerant object parser when present (consistency with the rest of
    the codebase) and stdlib json for the strict round-trip."""
    findings: List[ScrubFinding] = []
    try:
        text = json.dumps(led, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        findings.append(ScrubFinding(
            code=CODE_JSON_INVALID,
            where="ledger",
            detail=f"ledger is not JSON-serializable: {exc}",
        ))
        return findings
    try:
        back = json.loads(text)
    except (TypeError, ValueError) as exc:
        findings.append(ScrubFinding(
            code=CODE_JSON_INVALID,
            where="ledger",
            detail=f"ledger JSON did not parse back: {exc}",
        ))
        return findings
    if json.dumps(back, ensure_ascii=False, sort_keys=True) != text:
        findings.append(ScrubFinding(
            code=CODE_JSON_INVALID,
            where="ledger",
            detail="ledger did not survive a json round-trip byte-identical",
        ))
    # Tolerant parser sanity: the shared extractor should also read the dump.
    j = _imp_json()
    if j is not None:
        try:
            j.parse_first_json_object(text)
        except Exception as exc:  # noqa: BLE001
            findings.append(ScrubFinding(
                code=CODE_JSON_INVALID,
                where="ledger",
                detail=f"shared parse_first_json_object rejected the dump: {exc}",
            ))
    return findings


_REQUIRED_LINE_KEYS = ("line_id", "char_id", "text", "speaker_role")


def _check_ledger_shape(led: Dict[str, Any]) -> List[ScrubFinding]:
    """Confirm the ledger carries the load-bearing top-level arrays and that
    each line row keeps its render-contract keys. Shape only -- the scrub never
    invents or drops keys."""
    findings: List[ScrubFinding] = []
    if not isinstance(led, dict):
        findings.append(ScrubFinding(
            code=CODE_LEDGER_SHAPE, where="ledger",
            detail=f"ledger must be a dict, got {type(led).__name__}",
        ))
        return findings
    for key in ("cast", "lines"):
        if not isinstance(led.get(key), list):
            findings.append(ScrubFinding(
                code=CODE_LEDGER_SHAPE, where="ledger",
                detail=f"ledger[{key!r}] must be a list",
            ))
    for i, row in enumerate(led.get("lines", []) or []):
        if not isinstance(row, dict):
            findings.append(ScrubFinding(
                code=CODE_LEDGER_SHAPE, where=f"lines[{i}]",
                detail="line row must be a dict",
            ))
            continue
        for k in _REQUIRED_LINE_KEYS:
            if k not in row:
                findings.append(ScrubFinding(
                    code=CODE_LEDGER_SHAPE, where=f"lines[{i}]",
                    detail=f"line row missing required key {k!r}",
                    line_id=str(row.get("line_id") or ""),
                ))
        # `text` must be a string (or None -> normalizes to empty). A
        # dict / list / number text is a malformed render contract that no
        # editor repair can fix -- it is a shape violation, not a missing
        # line. Fail closed rather than silently coercing it to "".
        txt = row.get("text")
        if txt is not None and not isinstance(txt, str):
            findings.append(ScrubFinding(
                code=CODE_LEDGER_SHAPE, where=f"lines[{i}]",
                detail=(
                    f"line 'text' must be a string, got {type(txt).__name__}"
                ),
                line_id=str(row.get("line_id") or ""),
            ))
    return findings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scrub_ledger(led: Dict[str, Any], *, repair_available: bool) -> ScrubResult:
    """Deterministic Stage-4 scrub of an episode ledger. NO LLM.

    Pipeline (each step calls an EXISTING guard; none reimplements one):
      1. JSON round-trip + ledger-shape check.
      2. Three cast assertions (_otr_casting).
      3. Per spoken line (character / announcer rows only):
         a. strip_line_formatting (composer) -> drop leaked prefixes /
            brackets / markdown / wrapping quotes.
         b. strip leaked stage directions (NORMALIZER).
         c. normalize whitespace / smart quotes / punctuation -> TTS-safe.
         d. SFW / profanity check (live DEFAULT_PROFANITY_TERMS, validate_sfw
            semantics) -- a hit is a hard FAIL (PD4), never normalized away.
         e. cast-name leak check (composer detect_phantom_names + the
            speaker-self / ANNOUNCER degenerate-leak shape).
         f. per-line word + char bounds.
      4. Empty spoken line, or a cast-name leak the strips could not repair,
         is a MECHANICAL defect -> NEEDS_EDITOR_REPAIR (consumes the single
         shared repair if available, else FAIL). Duplicate spoken lines are
         reported but do not by themselves trigger a repair.
      5. Whole-episode spoken-word ceiling.

    The ledger is normalized IN PLACE (steps 3a-3c) -- on PASS the caller
    should persist the returned `ledger`.

    Args:
      led               the in-memory ledger dict (Ledger.data shape:
                        production_ledger.py). Read + normalized in place.
      repair_available  True iff the single shared editor repair (spine
                        invariant 5, "one repair, no loops") is still
                        unused. When a mechanical defect is found and this is
                        True the scrub consumes it (status
                        NEEDS_EDITOR_REPAIR); when False the scrub fails
                        closed (status FAIL).

    Returns:
      ScrubResult -- status PASS / NEEDS_EDITOR_REPAIR / FAIL, the normalized
      ledger, every finding, and the repair bookkeeping. Deterministic: the
      same ledger + flag always yield the same result.
    """
    findings: List[ScrubFinding] = []

    # ---- 1. JSON + shape -------------------------------------------------
    findings.extend(_check_json_round_trip(led))
    shape_findings = _check_ledger_shape(led)
    findings.extend(shape_findings)
    hard_structural = [
        f for f in findings
        if f.code in (CODE_JSON_INVALID, CODE_LEDGER_SHAPE)
    ]
    if hard_structural:
        # A malformed ledger cannot be normalized or repaired by an editor
        # pass -- fail closed regardless of the repair budget.
        return ScrubResult(
            status="FAIL", ledger=led, findings=findings,
        )

    cast_rows: List[dict] = list(led.get("cast", []) or [])
    lines: List[dict] = list(led.get("lines", []) or [])

    # ---- 2. cast assertions ---------------------------------------------
    cast_obj = _imp_casting()
    cast_findings = _run_cast_assertions(cast_obj, cast_rows)
    findings.extend(cast_findings)

    # ---- prep shared guards for the line loop ---------------------------
    lc = _imp_line_composer()
    sfw_terms, _sfw_code = _imp_sfw()
    # Full-cast roster (+ ANNOUNCER) so a line naming a real cast member is
    # on-roster and clears; only undeclared names register as leaks.
    try:
        allowed_roster = lc.build_allowed_roster(cast_rows)
    except Exception:  # noqa: BLE001 - defensive
        allowed_roster = frozenset(
            {(r.get("name") or "").upper() for r in cast_rows if isinstance(r, dict)}
            | {"ANNOUNCER"}
        )

    # name lookup for speaker resolution on a row (char_id -> NAME).
    name_by_cid: Dict[str, str] = {}
    for r in cast_rows:
        if isinstance(r, dict):
            cid = str(r.get("char_id") or "")
            nm = str(r.get("name") or "")
            if cid and nm:
                name_by_cid[cid] = nm

    normalized_any = False
    repair_targets: List[str] = []
    sfw_violations: List[Tuple[str, str]] = []
    seen_spoken: Dict[str, str] = {}   # normalized-text -> first line_id

    # ---- 3. per spoken line --------------------------------------------
    for i, row in enumerate(lines):
        if not isinstance(row, dict):
            continue
        role = str(row.get("speaker_role") or "character")
        if role not in _SPOKEN_ROLES:
            # music / sfx rows are render contracts, not speech. Never touched.
            continue
        line_id = str(row.get("line_id") or row.get("beat_id") or f"<idx{i}>")
        cid = str(row.get("char_id") or "")
        if role == "announcer":
            speaker = "ANNOUNCER"
        else:
            speaker = name_by_cid.get(cid) or str(row.get("speaker") or "")
        raw_text = row.get("text")
        raw_text = raw_text if isinstance(raw_text, str) else ""

        # 3a. composer format strip (leaked prefixes / brackets / md / quotes).
        cleaned = lc.strip_line_formatting(raw_text)
        # 3b. leaked stage directions.
        cleaned, stage_stripped = _strip_stage_directions(cleaned)
        if stage_stripped:
            findings.append(ScrubFinding(
                code=CODE_STAGE_DIRECTION, where=f"lines[{i}]",
                detail="stripped leaked stage direction from spoken field",
                line_id=line_id,
            ))
        # 3c. whitespace / smart-quote / punctuation normalization.
        cleaned = _normalize_whitespace_and_quotes(cleaned)

        if cleaned != raw_text:
            row["text"] = cleaned
            # Restamp the length fields so a bare/delimited strip does not leave
            # a stale word_count for downstream consumers (caption/timing).
            if "word_count" in row:
                row["word_count"] = len(cleaned.split())
            if "char_count" in row:
                row["char_count"] = len(cleaned)
            normalized_any = True
            findings.append(ScrubFinding(
                code=CODE_LINE_NORMALIZED, where=f"lines[{i}]",
                detail="normalized spoken text (whitespace/quotes/punct/strips)",
                line_id=line_id,
            ))

        # 3d. SFW -- hard PD4 stop, never normalized away.
        for term in _sfw_violations_in(cleaned, sfw_terms):
            sfw_violations.append((term, line_id))
            findings.append(ScrubFinding(
                code=CODE_SFW_VIOLATION, where=f"lines[{i}]",
                detail=f"profanity {term!r} in spoken text (PD4 hard stop)",
                line_id=line_id,
            ))

        # empty-line mechanical defect (post-strip). A spoken row with no
        # speech cannot ship to TTS and cannot be normalized -- it needs
        # regeneration.
        if not cleaned.strip():
            findings.append(ScrubFinding(
                code=CODE_EMPTY_LINE, where=f"lines[{i}]",
                detail="spoken line is empty after normalization",
                line_id=line_id,
            ))
            if line_id not in repair_targets:
                repair_targets.append(line_id)
            continue  # nothing further to check on an empty line

        # 3e. cast-name leak (BUG-279/F3). A leaked undeclared name or a
        # self/label leak mid-phrase is mechanical -> repair.
        leaks = _detect_cast_name_leak(lc, cleaned, speaker, allowed_roster)
        if leaks:
            findings.append(ScrubFinding(
                code=CODE_CAST_NAME_LEAK, where=f"lines[{i}]",
                detail=f"leaked name token(s) {leaks!r} in spoken text",
                line_id=line_id,
            ))
            if line_id not in repair_targets:
                repair_targets.append(line_id)

        # 3f. per-line length bounds (runaway guard; not a content target).
        wc = len(cleaned.split())
        cc = len(cleaned)
        if wc > _MAX_WORDS_PER_LINE or cc > _MAX_CHARS_PER_LINE:
            findings.append(ScrubFinding(
                code=CODE_OVERLONG_LINE, where=f"lines[{i}]",
                detail=(
                    f"spoken line over bounds ({wc} words / {cc} chars; "
                    f"caps {_MAX_WORDS_PER_LINE}/{_MAX_CHARS_PER_LINE})"
                ),
                line_id=line_id,
            ))

        # duplicate spoken line (exact, after normalization). Reported only --
        # an editor that cuts redundancy owns this; the scrub does not delete
        # story lines on its own.
        key = cleaned.strip().lower()
        if key in seen_spoken:
            findings.append(ScrubFinding(
                code=CODE_DUP_LINE, where=f"lines[{i}]",
                detail=(
                    f"duplicate of spoken line {seen_spoken[key]!r} "
                    "(reported; not auto-cut)"
                ),
                line_id=line_id,
            ))
        else:
            seen_spoken[key] = line_id

    # ---- 5. whole-episode spoken-word ceiling ---------------------------
    total_words = sum(
        len(str(r.get("text") or "").split())
        for r in lines
        if isinstance(r, dict)
        and str(r.get("speaker_role") or "character") in _SPOKEN_ROLES
    )
    if total_words > _TOTAL_WORDS_HARD_CEILING:
        findings.append(ScrubFinding(
            code=CODE_TOTAL_OVERLONG, where="ledger",
            detail=(
                f"episode spoken total {total_words} words exceeds hard "
                f"ceiling {_TOTAL_WORDS_HARD_CEILING}"
            ),
        ))

    # ---- decide status --------------------------------------------------
    # Hard, non-normalizable failures fail closed regardless of the repair
    # budget: SFW slip (PD4) or a tripped cast guard.
    if sfw_violations or cast_findings:
        return ScrubResult(
            status="FAIL", ledger=led, findings=findings,
            normalized=normalized_any, sfw_violations=sfw_violations,
        )

    # Mechanical defects (empty line / unrepairable leak) consume the one
    # shared repair if available, else fail closed (spine invariant 5).
    if repair_targets:
        if repair_available:
            return ScrubResult(
                status="NEEDS_EDITOR_REPAIR", ledger=led, findings=findings,
                normalized=normalized_any, repair_consumed=True,
                repair_targets=repair_targets, sfw_violations=sfw_violations,
            )
        return ScrubResult(
            status="FAIL", ledger=led, findings=findings,
            normalized=normalized_any, repair_consumed=False,
            repair_targets=repair_targets, sfw_violations=sfw_violations,
        )

    # Clean (possibly after in-place normalization). Reported-only findings
    # (overlong / duplicate / total-overlong) do not block TTS -- they are
    # advisory for the editor stage, which owns length and redundancy.
    return ScrubResult(
        status="PASS", ledger=led, findings=findings,
        normalized=normalized_any, sfw_violations=sfw_violations,
    )


# ---------------------------------------------------------------------------
# __main__ self-test -- exercises a clean ledger, a leaked-cast-name ledger,
# an empty-line ledger, and the repair_available=False fail-closed path.
# Zero network / GPU. Prints "SELF-TEST PASS: N/N".
# ---------------------------------------------------------------------------


def _build_clean_ledger() -> Dict[str, Any]:
    """A minimal but well-formed ledger: ANNOUNCER (Kokoro) + two Bark cast
    rows with distinct v2/ presets + clean spoken lines."""
    return {
        "schema_version": "l3-2026-05-14",
        "episode_id": "selftest_clean",
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER", "gender": "male",
             "tts_model": "kokoro", "voice_preset": "bm_george"},
            {"char_id": "c02", "name": "EDNA REEVES", "gender": "female",
             "tts_model": "bark", "voice_preset": "v2/en_speaker_4"},
            {"char_id": "c03", "name": "MILES HART", "gender": "male",
             "tts_model": "bark", "voice_preset": "v2/en_speaker_6"},
        ],
        "lines": [
            {"line_id": "b001", "beat_id": "b001", "char_id": "announcer",
             "speaker_role": "announcer",
             "text": "Tonight, a signal from the dark side of the moon."},
            {"line_id": "b002", "beat_id": "b002", "char_id": "c02",
             "speaker_role": "character",
             "text": "The relay is dead. We have minutes, not hours."},
            {"line_id": "b003", "beat_id": "b003", "char_id": "c03",
             "speaker_role": "character",
             "text": "Then we route power through the old array and pray."},
            {"line_id": "m001", "beat_id": "m001", "char_id": "music_open",
             "speaker_role": "music_open", "text": "tense orchestral sting"},
        ],
    }


def _run_self_test() -> Tuple[int, int]:
    passed = 0
    total = 0

    # -- Test 1: a clean ledger PASSES, no repair consumed. ---------------
    total += 1
    led1 = _build_clean_ledger()
    r1 = scrub_ledger(led1, repair_available=True)
    if r1.status == "PASS" and not r1.repair_consumed and r1.ok:
        passed += 1
        print("  [PASS] clean ledger -> PASS, no repair consumed")
    else:
        print(f"  [FAIL] clean ledger -> {r1.status} "
              f"(repair_consumed={r1.repair_consumed}); findings="
              f"{[f.code for f in r1.findings]}")

    # -- Test 2: normalization-only ledger (smart quotes + stray cue +
    #    whitespace) is cleaned IN PLACE and still PASSES. -----------------
    total += 1
    led2 = _build_clean_ledger()
    led2["lines"][1]["text"] = (
        "  The relay is dead.  [leans in] “We have minutes” — not hours.  "
    )
    r2 = scrub_ledger(led2, repair_available=True)
    cleaned_text = led2["lines"][1]["text"]
    if (r2.status == "PASS" and r2.normalized
            and "[" not in cleaned_text
            and "“" not in cleaned_text
            and "  " not in cleaned_text):
        passed += 1
        print("  [PASS] messy-but-recoverable ledger -> normalized + PASS")
    else:
        print(f"  [FAIL] normalization path -> {r2.status} "
              f"normalized={r2.normalized} text={cleaned_text!r}")

    # -- Test 3: leaked cast name (undeclared proper noun mid-phrase) ->
    #    NEEDS_EDITOR_REPAIR when the repair is available. -----------------
    total += 1
    led3 = _build_clean_ledger()
    # ZARKON is not on the roster -> a phantom (leaked) name.
    led3["lines"][2]["text"] = "Then we trust ZARKON to route the power."
    r3 = scrub_ledger(led3, repair_available=True)
    if (r3.status == "NEEDS_EDITOR_REPAIR" and r3.repair_consumed
            and "b003" in r3.repair_targets
            and any(f.code == CODE_CAST_NAME_LEAK for f in r3.findings)):
        passed += 1
        print("  [PASS] leaked cast name -> NEEDS_EDITOR_REPAIR (repair consumed)")
    else:
        print(f"  [FAIL] leaked-name path -> {r3.status} "
              f"targets={r3.repair_targets} "
              f"codes={[f.code for f in r3.findings]}")

    # -- Test 4: empty spoken line -> mechanical defect. With the repair
    #    available it is NEEDS_EDITOR_REPAIR. ------------------------------
    total += 1
    led4 = _build_clean_ledger()
    led4["lines"][1]["text"] = "   "   # whitespace-only -> empty after norm
    r4 = scrub_ledger(led4, repair_available=True)
    if (r4.status == "NEEDS_EDITOR_REPAIR" and r4.repair_consumed
            and "b002" in r4.repair_targets
            and any(f.code == CODE_EMPTY_LINE for f in r4.findings)):
        passed += 1
        print("  [PASS] empty spoken line -> NEEDS_EDITOR_REPAIR")
    else:
        print(f"  [FAIL] empty-line path -> {r4.status} "
              f"targets={r4.repair_targets} "
              f"codes={[f.code for f in r4.findings]}")

    # -- Test 5: same empty-line defect with NO repair available ->
    #    fail-closed (FAIL), repair NOT consumed. --------------------------
    total += 1
    led5 = _build_clean_ledger()
    led5["lines"][1]["text"] = "   "
    r5 = scrub_ledger(led5, repair_available=False)
    if (r5.status == "FAIL" and not r5.repair_consumed
            and "b002" in r5.repair_targets):
        passed += 1
        print("  [PASS] mechanical defect + no repair -> FAIL (fail-closed)")
    else:
        print(f"  [FAIL] fail-closed path -> {r5.status} "
              f"repair_consumed={r5.repair_consumed} "
              f"targets={r5.repair_targets}")

    # -- Test 6: SFW slip is a hard FAIL even with a repair available
    #    (profanity is never normalized away). -----------------------------
    total += 1
    led6 = _build_clean_ledger()
    led6["lines"][1]["text"] = "The damn relay is dead and we are out of time."
    r6 = scrub_ledger(led6, repair_available=True)
    if (r6.status == "FAIL" and r6.sfw_violations
            and any(t == "damn" for t, _ in r6.sfw_violations)):
        passed += 1
        print("  [PASS] SFW profanity -> FAIL (hard PD4 stop)")
    else:
        print(f"  [FAIL] SFW path -> {r6.status} "
              f"violations={r6.sfw_violations}")

    # -- Test 7: a real cast member named mid-line is NOT a leak (legitimate
    #    drama -- the roster includes the whole cast). -------------------
    total += 1
    led7 = _build_clean_ledger()
    led7["lines"][1]["text"] = "Miles, the relay is dead. Move now."
    r7 = scrub_ledger(led7, repair_available=True)
    if r7.status == "PASS" and not r7.repair_consumed:
        passed += 1
        print("  [PASS] naming a real cast member -> PASS (not a leak)")
    else:
        print(f"  [FAIL] cast-member-name path -> {r7.status} "
              f"codes={[f.code for f in r7.findings]}")

    # -- Test 8: malformed JSON (non-serializable value) -> FAIL,
    #    fail-closed regardless of repair budget. -------------------------
    total += 1
    led8 = _build_clean_ledger()
    led8["lines"][1]["text"] = {"not": "a string"}  # type: ignore[assignment]
    r8 = scrub_ledger(led8, repair_available=True)
    if r8.status == "FAIL" and any(
        f.code in (CODE_JSON_INVALID, CODE_LEDGER_SHAPE) for f in r8.findings
    ):
        passed += 1
        print("  [PASS] malformed ledger -> FAIL (fail-closed)")
    else:
        print(f"  [FAIL] malformed-ledger path -> {r8.status} "
              f"codes={[f.code for f in r8.findings]}")

    return passed, total


if __name__ == "__main__":
    print("OTR ledger-scrub self-test (deterministic, no LLM/GPU/network)")
    _passed, _total = _run_self_test()
    print(f"SELF-TEST PASS: {_passed}/{_total}")
    import sys as _sys
    _sys.exit(0 if _passed == _total else 1)
