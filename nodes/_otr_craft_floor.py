"""nodes/_otr_craft_floor.py -- Build 2 (2026-05-28).

DETERMINISTIC Tier-A integrity gate for the Story Room slot-formatted
output. PROMOTED from docs/sprint_drafts/build2/ and wired into
OTR_StoryRoomExtract.run per build2/WIRING_SPEC.md (Option A +
Placement 1). Helper module only -- not a node, not in _NODE_MODULES,
so ComfyUI never auto-imports it; OTR_StoryRoomExtract imports it lazily.

What this module is (per GO_FORWARD_PLAN_v10 Build 2 + critique 2):
    Format / integrity ONLY. No semantic judgment, no LLM call, no
    taste rubric. Same input -> same verdict, every time. The gate
    rejects broken/flat output; it cannot author good lines.

What it validates (Tier-A, all deterministic):
    1. SLOT_COUNT_MISMATCH  -- parsed row count != manifest slot count.
    2. SLOT_ORDER_MISMATCH  -- parsed slot ids, in order, != manifest
                               slot ids in order.
    3. SPEAKER_MISMATCH     -- a row's speaker != the manifest speaker
                               for that slot id.
    4. EMPTY_LINE           -- a voiced row carries empty text.
    5. BELOW_WORD_FLOOR     -- a row's word count is below the floor
                               parameter (default 4).
    6. PARSE_ERROR          -- a raw line is not parseable as
                               d###|SPEAKER: text (one block per slot).
    7. DUPLICATE_SLOT       -- the same slot id appears twice in the
                               parsed output (deterministic structural
                               break; surfaced so the order check has a
                               clean failure code rather than a confusing
                               count/order combination).

Contract assumptions (read from the live Build-1 code, see WIRING_SPEC):
    * The Story Room writer (Build 2 task 3a) EMITS one block per voiced
      slot in the form d001|SPEAKER: text. Exactly one ledger row per
      slot id. A block MAY contain internal pauses / multiple sentences,
      but it is ONE block terminated by the next d###| prefix or EOF
      (critique 8: one slot = exactly one committed text block).
    * The expected slot manifest is the ordered list of
      {slot_id, speaker} derived from the in-flight ledger's voiced
      lines (dialogue_slot_id + speaker), the same ordered list
      OTR_StoryRoomExtract already passes as voice_slot_ids and that
      OTR_StoryRoomCommit joins on. Slot ids match d\\d{3} (d001..dNNN).
    * Speaker is a cast name OR the literal 'ANNOUNCER' (bookends).
      Match is case-sensitive and exact, mirroring the commit join.

This module is PURE: no I/O, no GPU, no ComfyUI imports, no torch, no
pydantic. Word-count tokenization mirrors OTR_StoryRoomCommit's
convention ([A-Za-z][A-Za-z0-9'\\-]*) so a row that passes the floor
here produces the same word_count the commit node stamps.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation. No profanity. SFW.

PROMOTION NOTE (Build 2, 2026-05-28): `from __future__ import annotations`
was re-added on promotion (first import below) per repo convention.
Inside the nodes/ package the normal import path is used, so the Python
3.10 dataclasses string-annotation resolver race that the staged
file-path test loader hit does not apply here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


__all__ = [
    "FAILURE_CODES",
    "SLOT_LINE_RE",
    "DEFAULT_WORD_FLOOR",
    "SlotFailure",
    "CraftFloorResult",
    "ParsedSlot",
    "parse_slot_lines",
    "normalize_slot_line",
    "evaluate_tier_a",
]


# ---------------------------------------------------------------------------
# Failure-code vocabulary (machine-readable; stable strings)
# ---------------------------------------------------------------------------
#
# These codes are the public contract. The integration layer keys its
# red-graph messages and the regression test pins off these exact
# strings -- do not rename without updating WIRING_SPEC.md + the test.

SLOT_COUNT_MISMATCH = "SLOT_COUNT_MISMATCH"
SLOT_ORDER_MISMATCH = "SLOT_ORDER_MISMATCH"
SPEAKER_MISMATCH = "SPEAKER_MISMATCH"
EMPTY_LINE = "EMPTY_LINE"
BELOW_WORD_FLOOR = "BELOW_WORD_FLOOR"
PARSE_ERROR = "PARSE_ERROR"
DUPLICATE_SLOT = "DUPLICATE_SLOT"

FAILURE_CODES = (
    SLOT_COUNT_MISMATCH,
    SLOT_ORDER_MISMATCH,
    SPEAKER_MISMATCH,
    EMPTY_LINE,
    BELOW_WORD_FLOOR,
    PARSE_ERROR,
    DUPLICATE_SLOT,
)


# ---------------------------------------------------------------------------
# Parsing constants
# ---------------------------------------------------------------------------

# One block per slot. The prefix is d###|SPEAKER: then the line text.
# - slot id: d + exactly three digits (d001..dNNN), matching the
#   DialogueOnlyRow ^d\d{3}$ pattern already in the codebase.
# - speaker: everything up to the first colon AFTER the pipe, stripped.
#   Allows spaces, periods, apostrophes, hyphens (e.g. DR. MAEVE COLE).
#   Speaker validation against the manifest is exact; the parser only
#   splits, it does not judge.
# - text: the remainder after the first colon, stripped.
#
# The regex is anchored to the START of a block. A block ends at the
# next block-start prefix or end of input. Internal pauses (newlines,
# ellipses, sound directions) are preserved verbatim inside the text.
SLOT_LINE_RE = re.compile(
    r"^\s*(?P<slot_id>d\d{3})\s*\|\s*(?P<speaker>[^:|\n]+?)\s*:\s?(?P<text>.*)$",
    re.DOTALL,
)

# Word-count tokenizer. Mirrors OTR_StoryRoomCommit._commit_dialogue and
# production_ledger._word_count so the floor here agrees with the
# word_count the commit node stamps.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")

# Block-start detector for splitting a multi-block raw string. A new
# block begins at the start of a line that opens with d###|.
_BLOCK_START_RE = re.compile(r"(?m)^\s*d\d{3}\s*\|")

DEFAULT_WORD_FLOOR = 4


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParsedSlot:
    """One parsed d###|SPEAKER: text block.

    index is the zero-based position in the parsed output (draft order),
    used so the order check can report exactly which position diverged.
    raw is the original block text for forensic logging.
    """

    index: int
    slot_id: str
    speaker: str
    text: str
    raw: str = ""

    def as_tuple(self):
        """Return (slot_id, speaker, text) -- the parse_slot_lines shape."""
        return (self.slot_id, self.speaker, self.text)


@dataclass
class SlotFailure:
    """One Tier-A failure, machine-readable.

    code:    one of FAILURE_CODES.
    slot_id: the offending slot id when known (the draft's id, or the
             manifest id for an order/count divergence). May be '' for a
             whole-output failure like SLOT_COUNT_MISMATCH where no
             single slot owns the break.
    index:   zero-based draft position when known, else -1.
    detail:  short human-readable explanation. Deterministic text so the
             test can pin it; carries the concrete numbers / names.
    """

    code: str
    slot_id: str = ""
    index: int = -1
    detail: str = ""

    def to_dict(self):
        return {
            "code": self.code,
            "slot_id": self.slot_id,
            "index": self.index,
            "detail": self.detail,
        }


@dataclass
class CraftFloorResult:
    """Structured Tier-A verdict.

    passed:        True iff zero failures.
    failures:      ordered list of SlotFailure. Order is deterministic:
                   parse errors first (in draft order), then structural
                   (duplicate, count, order), then per-slot semantic-free
                   checks (speaker, empty, word floor) in draft order.
    parsed_slots:  the ParsedSlot list (empty when a parse error aborts
                   before a full parse is possible).
    word_floor:    the floor that was applied (echoed for the audit log).
    """

    passed: bool
    failures: List[SlotFailure] = field(default_factory=list)
    parsed_slots: List[ParsedSlot] = field(default_factory=list)
    word_floor: int = DEFAULT_WORD_FLOOR

    @property
    def failure_codes(self):
        """Ordered list of the failure code strings (for quick asserts)."""
        return [f.code for f in self.failures]

    def to_dict(self):
        return {
            "passed": bool(self.passed),
            "failures": [f.to_dict() for f in self.failures],
            "parsed_slots": [
                {
                    "index": s.index,
                    "slot_id": s.slot_id,
                    "speaker": s.speaker,
                    "text": s.text,
                }
                for s in self.parsed_slots
            ],
            "word_floor": int(self.word_floor),
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _split_blocks(raw):
    """Split a multi-block raw string into per-slot blocks.

    A block starts at a line opening with d###| and runs until the next
    such line or end of input. Leading content before the first block
    prefix is dropped (writer preamble guard). Returns the list of block
    strings in document order, each stripped of trailing whitespace but
    with internal newlines / pauses preserved.

    Deterministic: pure regex span walk, no model, no randomness.
    """
    text = raw or ""
    starts = [m.start() for m in _BLOCK_START_RE.finditer(text)]
    if not starts:
        # No block prefixes at all. Return the whole non-empty body as a
        # single "block" so the parser emits one PARSE_ERROR rather than
        # silently swallowing content.
        body = text.strip()
        return [body] if body else []
    blocks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
    return blocks


def parse_slot_lines(raw):
    """Parse Story Room slot output into (slot_id, speaker, text) tuples.

    Input form (one block per voiced slot):
        d001|ANNOUNCER: From the newsroom, a story.
        d002|REN BLACK: I will not sign that paper. ... Not tonight.
        d003|DR. MAEVE COLE: Then we both go down with it.

    A block may span multiple physical lines and carry internal pauses;
    it is terminated by the next d###| prefix or end of input. Text is
    returned verbatim (internal newlines preserved, edges stripped).

    Blocks that do not match d###|SPEAKER: text are SKIPPED here (they
    surface as PARSE_ERROR in evaluate_tier_a, which sees the same
    blocks). This helper is the public parse contract; it returns only
    the well-formed rows so callers that only want the structured rows
    do not have to filter. evaluate_tier_a does its own parse so it can
    attribute parse failures by position.

    PURE. Same input -> same output.
    """
    rows = []
    for block in _split_blocks(raw):
        m = SLOT_LINE_RE.match(block)
        if not m:
            continue
        rows.append((
            m.group("slot_id").strip(),
            m.group("speaker").strip(),
            m.group("text").strip(),
        ))
    return rows


def normalize_slot_line(slot_id, speaker, text):
    """Emit a canonical d###|SPEAKER: text block from its parts.

    The inverse of one parsed row. Useful for tests and for any layer
    that wants to re-serialize a normalized block. Edges of speaker and
    text are stripped; internal content is preserved verbatim.

    PURE.
    """
    sid = str(slot_id or "").strip()
    spk = str(speaker or "").strip()
    txt = str(text or "").strip()
    return sid + "|" + spk + ": " + txt


def _word_count(text):
    """Count words the same way OTR_StoryRoomCommit does.

    Tokenizes on a leading letter followed by letters/digits/apostrophe/
    hyphen. A line of pure punctuation or pauses counts as 0 words.
    """
    return len(_WORD_RE.findall(text or ""))


# ---------------------------------------------------------------------------
# Manifest normalization
# ---------------------------------------------------------------------------


def _normalize_manifest(manifest):
    """Normalize an expected-slot manifest into ordered (slot_id, speaker).

    Accepts:
      * a list of dicts: {"slot_id": "d001", "speaker": "REN BLACK"}.
        Also accepts the ledger-native key dialogue_slot_id as a synonym
        for slot_id so callers can pass ledger line dicts straight
        through.
      * a list of (slot_id, speaker) tuples / lists.

    Edges are stripped. Empty rows are dropped. Deterministic; preserves
    caller order.
    """
    out = []
    for entry in (manifest or []):
        if entry is None:
            continue
        if isinstance(entry, dict):
            sid = str(
                entry.get("slot_id")
                or entry.get("dialogue_slot_id")
                or ""
            ).strip()
            spk = str(entry.get("speaker") or "").strip()
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            sid = str(entry[0] or "").strip()
            spk = str(entry[1] or "").strip()
        else:
            # Unrecognized row shape -- skip; the count check will catch
            # any resulting divergence loudly rather than guessing.
            continue
        if sid:
            out.append((sid, spk))
    return out


# ---------------------------------------------------------------------------
# Tier-A evaluation
# ---------------------------------------------------------------------------


def evaluate_tier_a(raw, manifest, word_floor=DEFAULT_WORD_FLOOR):
    """Run the deterministic Tier-A gate.

    Args:
        raw: the Story Room writer's slot-formatted output -- one
            d###|SPEAKER: text block per voiced slot.
        manifest: ordered expected slots. List of
            {"slot_id"/"dialogue_slot_id", "speaker"} dicts OR
            (slot_id, speaker) tuples. The same ordered voiced-slot list
            OTR_StoryRoomExtract derives and OTR_StoryRoomCommit joins on.
        word_floor: minimum word count per voiced line. Default
            DEFAULT_WORD_FLOOR (4). A row at or above the floor passes
            that check.

    Returns:
        CraftFloorResult. passed is True iff zero failures. Failure order
        is deterministic (see CraftFloorResult docstring).

    Determinism guarantee: this function performs only regex parsing,
    list comparison, and integer counting. No randomness, no clock, no
    model. Same (raw, manifest, word_floor) -> identical CraftFloorResult.
    """
    floor = int(word_floor)
    expected = _normalize_manifest(manifest)
    expected_ids = [sid for sid, _ in expected]
    expected_speaker_by_id = {}
    for sid, spk in expected:
        # First occurrence wins; a duplicate manifest id is itself a
        # caller bug, but we stay deterministic by keeping the first.
        expected_speaker_by_id.setdefault(sid, spk)

    failures = []
    parsed = []

    # ---- Parse pass (attribute parse errors by position) ----
    blocks = _split_blocks(raw)
    for idx, block in enumerate(blocks):
        m = SLOT_LINE_RE.match(block)
        if not m:
            failures.append(SlotFailure(
                code=PARSE_ERROR,
                slot_id="",
                index=idx,
                detail=(
                    "block at position "
                    + str(idx)
                    + " did not match d###|SPEAKER: text (first 60 "
                    + "chars: " + repr(block[:60]) + ")"
                ),
            ))
            continue
        parsed.append(ParsedSlot(
            index=idx,
            slot_id=m.group("slot_id").strip(),
            speaker=m.group("speaker").strip(),
            text=m.group("text").strip(),
            raw=block,
        ))

    parsed_ids = [p.slot_id for p in parsed]

    # ---- Duplicate slot id (structural; deterministic) ----
    seen = set()
    dup_reported = set()
    for p in parsed:
        if p.slot_id in seen and p.slot_id not in dup_reported:
            failures.append(SlotFailure(
                code=DUPLICATE_SLOT,
                slot_id=p.slot_id,
                index=p.index,
                detail="slot id " + p.slot_id + " appears more than once",
            ))
            dup_reported.add(p.slot_id)
        seen.add(p.slot_id)

    # ---- Count check ----
    if len(parsed_ids) != len(expected_ids):
        failures.append(SlotFailure(
            code=SLOT_COUNT_MISMATCH,
            slot_id="",
            index=-1,
            detail=(
                "parsed " + str(len(parsed_ids)) + " slot block(s); "
                + "manifest expects " + str(len(expected_ids))
            ),
        ))

    # ---- Order check ----
    # Compared position-by-position over the overlap so we report the
    # first divergent index concretely; a pure list-inequality would
    # hide WHERE it broke.
    if parsed_ids != expected_ids:
        overlap = min(len(parsed_ids), len(expected_ids))
        first_bad = -1
        for i in range(overlap):
            if parsed_ids[i] != expected_ids[i]:
                first_bad = i
                break
        if first_bad == -1:
            # Same prefix, differing length -- the count check already
            # fired; still record an order failure for the tail so the
            # code vocabulary is unambiguous.
            first_bad = overlap
        got = parsed_ids[first_bad] if first_bad < len(parsed_ids) else "(none)"
        want = (
            expected_ids[first_bad]
            if first_bad < len(expected_ids) else "(none)"
        )
        failures.append(SlotFailure(
            code=SLOT_ORDER_MISMATCH,
            slot_id=want if want != "(none)" else got,
            index=first_bad,
            detail=(
                "slot order diverges at position " + str(first_bad)
                + ": got " + str(got) + ", manifest expects " + str(want)
            ),
        ))

    # ---- Per-slot checks: speaker, empty line, word floor ----
    # Walk parsed rows in draft order. Speaker is checked against the
    # manifest entry for THIS row's slot id (not its position) so a
    # correctly-named-but-misordered row is not double-flagged as a
    # speaker mismatch. Rows whose slot id is not in the manifest are
    # skipped for the speaker check (the order/count failure owns that
    # divergence); empty-line and word-floor still apply to every voiced
    # row regardless of manifest membership.
    for p in parsed:
        # Empty line.
        if not p.text.strip():
            failures.append(SlotFailure(
                code=EMPTY_LINE,
                slot_id=p.slot_id,
                index=p.index,
                detail="slot " + p.slot_id + " has empty voiced text",
            ))
            # An empty line is also below any positive floor, but we do
            # NOT also stamp BELOW_WORD_FLOOR for the same row --
            # EMPTY_LINE is the more specific code. Next row.
            continue

        # Speaker mismatch (only when the slot id is in the manifest).
        if p.slot_id in expected_speaker_by_id:
            want_spk = expected_speaker_by_id[p.slot_id]
            if p.speaker != want_spk:
                failures.append(SlotFailure(
                    code=SPEAKER_MISMATCH,
                    slot_id=p.slot_id,
                    index=p.index,
                    detail=(
                        "slot " + p.slot_id + " speaker "
                        + repr(p.speaker) + "; manifest expects "
                        + repr(want_spk)
                    ),
                ))

        # Word floor.
        wc = _word_count(p.text)
        if wc < floor:
            failures.append(SlotFailure(
                code=BELOW_WORD_FLOOR,
                slot_id=p.slot_id,
                index=p.index,
                detail=(
                    "slot " + p.slot_id + " has " + str(wc)
                    + " word(s); floor is " + str(floor)
                ),
            ))

    return CraftFloorResult(
        passed=(len(failures) == 0),
        failures=failures,
        parsed_slots=parsed,
        word_floor=floor,
    )
