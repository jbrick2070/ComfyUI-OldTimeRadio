"""Uncapped source document + pre-outline overview for source-owned lanes.

Why this module exists
----------------------
The fidelity packs instruct the model to CARRY the author's words, and until
now nothing put those words in front of it. Worse, the body the pipeline did
carry was a PREFIX: ``canonicalize_public_domain_text`` capped at 12,000
characters, so on a 25,200-word work the pre-outline authors -- the
interpreter that names the cast, the story contract, the outline -- read the
opening and inferred the rest. An author who is shown a prefix and told to be
faithful will invent the remainder and believe it complied.

This module owns the two artifacts that fix that, both deterministic and
model-free:

``SourceDocument``
    The COMPLETE canonicalized body, its hash, and the normalization version
    that produced it. It is a TRANSIENT runtime artifact: it travels beside
    the payload for the duration of a build and is never serialized into
    ``meta``, the ledger, or a prompt receipt -- receipts carry offsets and
    hashes, never body text.

``SourceOverview``
    Deterministic windows that COVER the whole body (not a prefix), plus a
    few role-tagged evidence spans. Built before the outline exists, so the
    pre-outline authors can be grounded in the whole work. Every span is
    validated against the body it claims to quote: ``text ==
    body[start_char:end_char]``, half-open Unicode character offsets.

Nothing here loads a model, touches the GPU, reads the network, or imports
anything heavy. Selection is deterministic: same body in, same spans out.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence

# Version constants. Bump NORMALIZATION_VERSION when the canonical body bytes
# for an unchanged source would change (that invalidates every stored offset
# and hash); bump OVERVIEW_VERSION when window/evidence selection changes
# (offsets stay valid, but a replay would pick different spans).
NORMALIZATION_VERSION = "otr_source_normalization_v1"
OVERVIEW_VERSION = "otr_source_overview_v1"

# Evidence roles the overview tags. Deterministic structural picks -- the
# opening establishes setting and who is present, the closing holds the
# ending, the quote-dense window is where the author's own dialogue lives.
EVIDENCE_OPENING = "opening"
EVIDENCE_CLOSING = "closing"
EVIDENCE_DIALOGUE = "dialogue"

# Quotation marks that plausibly OPEN or CLOSE quoted speech. Bare apostrophes
# are deliberately excluded: in "don't", "it's", "father's" the mark is
# possessive or elided, not a quote, and counting them ranked contraction-heavy
# NARRATION above real dialogue. Single quotes count only at a word boundary,
# which is where a quotation actually opens or closes.
_QUOTE_RE = re.compile(r"[\"“”]|(?<!\w)[‘']|['’](?!\w)")


class SourceDocumentError(RuntimeError):
    """A source document or overview could not be built or validated (loud)."""


def canonical_body_sha256(body: str) -> str:
    """Hash the canonical body exactly as stored.

    This is NOT the provenance sidecar's ``body_sha256``: that one covers
    normalized RAW bytes as fetched, before HTML-unescape and whitespace
    canonicalization. The two are not interchangeable and must never be
    compared. This hash pins the coordinate system that spans index into.
    """
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceSpan:
    """A half-open character range into a canonical body, plus its text.

    ``text`` is carried so a consumer can use the span without holding the
    document, but it is always verified against the body at construction --
    a span whose text has drifted from its offsets is refused by name rather
    than quietly quoting something the source does not say.

    ``text`` is ``repr=False`` ON PURPOSE: a span can hold thousands of words,
    and the default dataclass repr would dump them into any log line or
    traceback that formats a span. Identity (offsets + role) is what a reader
    of a log actually needs; the text is reachable explicitly when wanted.
    """

    start_char: int
    end_char: int
    text: str = field(repr=False)
    role: str = ""

    @property
    def char_count(self) -> int:
        return self.end_char - self.start_char

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _make_span(body: str, start: int, end: int, *, role: str = "") -> SourceSpan:
    if start < 0 or end > len(body) or start >= end:
        raise SourceDocumentError(
            f"span [{start}:{end}] is not a valid range into a "
            f"{len(body)}-character body (role={role!r})"
        )
    return SourceSpan(
        start_char=start, end_char=end, text=body[start:end], role=role)


@dataclass(frozen=True)
class SourceDocument:
    """The COMPLETE canonical body plus the identity spans index against.

    Transient by contract: never stamped into ``meta``, never written to a
    ledger, never persisted in a receipt. Callers that need durability store
    ``body_sha256`` + offsets and re-derive the text from the document.

    ``canonical_body`` is ``repr=False`` ON PURPOSE. "Never serialized" has to
    survive the paths nobody plans: a debug ``log.info("%s", result)``, an
    ``f"bad state: {doc!r}"`` in an exception, a traceback frame that formats
    locals. The default dataclass repr put the entire body -- up to 25,000
    words -- into every one of those. The repr now carries identity only, and
    a test pins it.
    """

    canonical_body: str = field(repr=False)
    body_sha256: str = ""
    normalization_version: str = ""
    source_ref: str = ""

    @property
    def char_count(self) -> int:
        return len(self.canonical_body)

    @property
    def word_count(self) -> int:
        return len(self.canonical_body.split())

    def span(self, start: int, end: int, *, role: str = "") -> SourceSpan:
        """Build a validated span into THIS body."""
        return _make_span(self.canonical_body, start, end, role=role)

    def verify_span(self, span: SourceSpan) -> None:
        """Refuse a span whose text no longer matches its own offsets."""
        actual = self.canonical_body[span.start_char:span.end_char]
        if actual != span.text:
            raise SourceDocumentError(
                f"span [{span.start_char}:{span.end_char}] (role="
                f"{span.role!r}) does not match the canonical body it claims "
                f"to quote"
            )


def build_source_document(
    canonical_body: str,
    *,
    source_ref: str = "",
    normalization_version: str = NORMALIZATION_VERSION,
) -> SourceDocument:
    """Wrap an already-normalized COMPLETE body as a hashed document.

    The caller owns normalization (each bank normalizes its own format); this
    owns identity. An empty body is refused -- a document with nothing in it
    would ground nothing while looking like grounding.
    """
    if not isinstance(canonical_body, str):
        raise SourceDocumentError(
            f"canonical body must be str, got {type(canonical_body).__name__}")
    if not canonical_body.strip():
        raise SourceDocumentError(
            f"canonical body is empty (source_ref={source_ref!r})")
    return SourceDocument(
        canonical_body=canonical_body,
        body_sha256=canonical_body_sha256(canonical_body),
        normalization_version=normalization_version,
        source_ref=source_ref,
    )


@dataclass(frozen=True)
class SourceOverview:
    """Whole-body coverage for the authors that run BEFORE the outline.

    ``windows`` tile the ENTIRE body in order with no gaps, so a consumer can
    state honestly which part of the work it read. ``evidence`` tags a few of
    those structural positions by role. Both are derived deterministically;
    no model participates.
    """

    body_sha256: str
    normalization_version: str
    overview_version: str
    windows: tuple[SourceSpan, ...] = field(default_factory=tuple)
    evidence: tuple[SourceSpan, ...] = field(default_factory=tuple)
    source_ref: str = ""
    # What the caller ASKED for, kept beside what it got. A body shorter than
    # 2 chars per window collapses to a single window; that is correct, but a
    # silent collapse reads downstream as "this work is one undivided block".
    # No silent caps: the receipt reports both numbers.
    requested_window_count: int = 0

    @property
    def covered_char_count(self) -> int:
        return sum(w.char_count for w in self.windows)

    def evidence_for(self, role: str) -> SourceSpan | None:
        for span in self.evidence:
            if span.role == role:
                return span
        return None

    def receipt(self) -> dict:
        """A body-free record safe to persist in the ledger.

        Offsets, counts, hashes and versions only -- never source text. This
        is what proves WHICH material grounded a build without duplicating the
        work into durable metadata.
        """
        return {
            "body_sha256": self.body_sha256,
            "normalization_version": self.normalization_version,
            "overview_version": self.overview_version,
            "source_ref": self.source_ref,
            "window_count": len(self.windows),
            "requested_window_count": self.requested_window_count,
            "window_count_collapsed": (
                self.requested_window_count > len(self.windows)
            ),
            "covered_char_count": self.covered_char_count,
            "dialogue_evidence_present": any(
                s.role == EVIDENCE_DIALOGUE for s in self.evidence
            ),
            "windows": [
                {"start_char": w.start_char, "end_char": w.end_char}
                for w in self.windows
            ],
            "evidence": [
                {
                    "role": s.role,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                }
                for s in self.evidence
            ],
        }


def _window_bounds(body: str, window_count: int) -> list[tuple[int, int]]:
    """Split a body into ``window_count`` ordered, gapless ranges.

    Cuts land on whitespace where one is available near the target so a window
    does not open mid-word; coverage is exact regardless -- every character
    belongs to exactly one window.

    FLOOR: a body shorter than two characters per requested window collapses
    to a SINGLE window. Slicing a 10-character work into 8 pieces produces
    fragments that ground nothing. The collapse is deliberate and is reported
    through ``SourceOverview.requested_window_count`` rather than hidden.
    """
    total = len(body)
    if window_count < 1:
        raise SourceDocumentError("window_count must be at least 1")
    if window_count == 1 or total < window_count * 2:
        return [(0, total)]

    target = total / window_count
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(1, window_count):
        cut = int(round(target * index))
        cut = max(start + 1, min(cut, total - (window_count - index)))
        # Nudge onto the nearest following space so windows start on a word.
        space = body.find(" ", cut)
        if space != -1 and space - cut <= max(1, int(target // 10)):
            cut = space + 1
        cut = max(start + 1, min(cut, total - (window_count - index)))
        bounds.append((start, cut))
        start = cut
    bounds.append((start, total))
    return bounds


def _densest_quote_window(windows: Sequence[SourceSpan]) -> SourceSpan | None:
    """The window carrying the most quotation marks per character.

    Returns ``None`` when the work contains no quoted speech at all. That
    matters: tagging a stretch of pure narration as ``dialogue`` would invite
    a consumer to treat the author's narration as quoted speech -- the exact
    class of confusion this whole module exists to remove. Absent evidence is
    reported as absent.

    Ties break on the EARLIER window so the pick is stable for a given body.
    """
    best: SourceSpan | None = None
    best_density = 0.0
    for window in windows:
        if window.char_count <= 0:
            continue
        hits = len(_QUOTE_RE.findall(window.text))
        if not hits:
            continue
        density = hits / window.char_count
        if best is None or density > best_density:
            best, best_density = window, density
    return best


def build_source_overview(
    document: SourceDocument,
    *,
    window_count: int = 8,
    overview_version: str = OVERVIEW_VERSION,
) -> SourceOverview:
    """Cover the whole document in ordered windows and tag evidence spans.

    Deterministic: the same document always yields the same windows and the
    same evidence picks. Coverage is total -- ``covered_char_count`` equals
    the document's own character count, which is the property that makes
    "the authors read the whole work" a checkable claim rather than a hope.
    """
    body = document.canonical_body
    windows = tuple(
        _make_span(body, start, end, role="window")
        for start, end in _window_bounds(body, window_count)
    )

    evidence = [
        _make_span(
            body, windows[0].start_char, windows[0].end_char,
            role=EVIDENCE_OPENING,
        ),
        _make_span(
            body, windows[-1].start_char, windows[-1].end_char,
            role=EVIDENCE_CLOSING,
        ),
    ]
    dialogue = _densest_quote_window(windows)
    if dialogue is not None:
        evidence.append(
            _make_span(
                body, dialogue.start_char, dialogue.end_char,
                role=EVIDENCE_DIALOGUE,
            )
        )

    overview = SourceOverview(
        body_sha256=document.body_sha256,
        normalization_version=document.normalization_version,
        overview_version=overview_version,
        windows=windows,
        evidence=tuple(evidence),
        source_ref=document.source_ref,
        requested_window_count=window_count,
    )
    if overview.covered_char_count != document.char_count:
        raise SourceDocumentError(
            f"overview covers {overview.covered_char_count} characters of a "
            f"{document.char_count}-character body -- coverage must be total"
        )
    return overview
