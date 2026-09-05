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
from typing import Sequence

# Version constants. Bump NORMALIZATION_VERSION when the canonical body bytes
# for an unchanged source would change (that invalidates every stored offset
# and hash); bump OVERVIEW_VERSION when window/evidence selection changes
# (offsets stay valid, but a replay would pick different spans).
NORMALIZATION_VERSION = "otr_source_normalization_v1"
# v2 (QA round 2): dialogue evidence is omitted entirely when a work carries
# no quoted speech, quotation is counted as balanced spans rather than loose
# marks, and the receipt gained requested_window_count / collapse and
# dialogue-presence flags. Selection and receipt shape both changed, so the
# version moves -- a v1 receipt and a v2 receipt are not comparable.
OVERVIEW_VERSION = "otr_source_overview_v2"

# Evidence roles the overview tags. Deterministic structural picks -- the
# opening establishes setting and who is present, the closing holds the
# ending, the quote-dense window is where the author's own dialogue lives.
EVIDENCE_OPENING = "opening"
EVIDENCE_CLOSING = "closing"
EVIDENCE_DIALOGUE = "dialogue"

# Counting individual marks cannot tell a quotation from an apostrophe: the
# first fix excluded "don't" and "father's" but still counted the mark in
# "the boys' club", "'tis", and "the '90s". Count BALANCED SPANS instead --
# an opener and a closer with text between them -- which is what a quotation
# actually is. Double quotes are unambiguous; single quotes must open at a
# word boundary and close at one, so a lone possessive or elision mark has
# nothing to pair with and scores nothing.
_DOUBLE_QUOTED_RE = re.compile(r"[\"“][^\"“”]{1,400}[\"”]")
# Single quotes need tighter bounds than double ones. A first attempt paired
# unrelated apostrophes ACROSS a work -- "'90s besides. the boys'" scans as a
# quotation if you only require boundaries. So: the opener may not be preceded
# by a word character (rules out possessives), may not be followed by a digit
# (rules out decades), the span is non-greedy, may not cross a line, and is
# length-capped. A lone possessive or elision mark then has nothing to pair
# with and contributes nothing.
_SINGLE_QUOTED_RE = re.compile(
    r"(?<!\w)[‘'](?!\d)[^‘’'\n]{1,200}?['’](?!\w)")


def _count_quoted_spans(text: str) -> int:
    """Balanced quoted passages in ``text`` -- not loose quotation marks."""
    return (len(_DOUBLE_QUOTED_RE.findall(text))
            + len(_SINGLE_QUOTED_RE.findall(text)))


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


class _Transient:
    """Base for artifacts that hold source text and must never be serialized.

    A dataclass was the obvious shape and the wrong one. ``repr=False`` hides
    a field from display, but ``dataclasses.asdict`` and ``astuple`` walk the
    field list regardless -- and because an overview's windows tile the whole
    work, ``asdict(overview)`` reconstructed the entire body. These are plain
    slotted objects instead, so every structural serializer REFUSES rather
    than leaks: not a dataclass (asdict/astuple raise TypeError), no
    ``__dict__`` (``vars()`` raises), and pickling is refused by name.

    The durable representation is ``SourceOverview.receipt()`` -- hashes,
    offsets and counts, never text.
    """

    __slots__ = ()

    def __setattr__(self, name, value):  # immutable after construction
        raise SourceDocumentError(
            f"{type(self).__name__} is immutable; build a new one")

    def __delattr__(self, name):
        raise SourceDocumentError(
            f"{type(self).__name__} is immutable")

    def _set(self, name, value) -> None:
        object.__setattr__(self, name, value)

    def __getstate__(self):
        raise SourceDocumentError(
            f"{type(self).__name__} is transient and refuses pickling; "
            f"persist the overview receipt (hashes + offsets) instead"
        )

    def __reduce__(self):
        raise SourceDocumentError(
            f"{type(self).__name__} is transient and refuses pickling; "
            f"persist the overview receipt (hashes + offsets) instead"
        )


class SourceSpan(_Transient):
    """A half-open character range into a canonical body, plus its text.

    ``text`` is carried so a consumer can use the span without holding the
    document, but the factory always slices it FROM the body, so text and
    offsets cannot disagree. Build spans with ``SourceDocument.span`` /
    ``_make_span``; a hand-built span is verifiable with ``verify_span``.

    The repr carries offsets and role only -- a span can hold thousands of
    words, and a log line is not where they belong.
    """

    __slots__ = ("start_char", "end_char", "text", "role")

    def __init__(self, start_char: int, end_char: int, text: str,
                 role: str = "") -> None:
        self._set("start_char", int(start_char))
        self._set("end_char", int(end_char))
        self._set("text", str(text))
        self._set("role", str(role))

    def __repr__(self) -> str:
        return (f"SourceSpan(start_char={self.start_char}, "
                f"end_char={self.end_char}, role={self.role!r})")

    def __eq__(self, other) -> bool:
        if not isinstance(other, SourceSpan):
            return NotImplemented
        return (self.start_char, self.end_char, self.text, self.role) == (
            other.start_char, other.end_char, other.text, other.role)

    def __hash__(self) -> int:
        return hash((self.start_char, self.end_char, self.role))

    @property
    def char_count(self) -> int:
        return self.end_char - self.start_char

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _make_span(body: str, start: int, end: int, *, role: str = "") -> SourceSpan:
    """Build a span whose text is TAKEN FROM the body -- the only safe way.

    Direct ``SourceSpan(...)`` construction can pair any offsets with any
    text; this factory slices the body itself, so text and offsets cannot
    disagree by construction rather than by later inspection.
    """
    if start < 0 or end > len(body) or start >= end:
        raise SourceDocumentError(
            f"span [{start}:{end}] is not a valid range into a "
            f"{len(body)}-character body (role={role!r})"
        )
    return SourceSpan(
        start_char=start, end_char=end, text=body[start:end], role=role)


def _assert_tiles(windows: Sequence[SourceSpan], total: int) -> None:
    """Refuse anything that is not a true ordered tiling of [0, total).

    Summing window lengths is NOT enough: a gap and an overlap of equal size
    cancel out and a body could be mis-covered while the total looked right.
    Check the actual boundaries.
    """
    if not windows:
        raise SourceDocumentError("overview produced no windows")
    if windows[0].start_char != 0:
        raise SourceDocumentError(
            f"coverage starts at {windows[0].start_char}, not 0")
    cursor = 0
    for window in windows:
        if window.end_char <= window.start_char:
            raise SourceDocumentError(
                f"window [{window.start_char}:{window.end_char}] is empty "
                f"or inverted"
            )
        if window.start_char != cursor:
            raise SourceDocumentError(
                f"coverage breaks at {cursor}: next window starts at "
                f"{window.start_char} ({'gap' if window.start_char > cursor else 'overlap'})"
            )
        cursor = window.end_char
    if cursor != total:
        raise SourceDocumentError(
            f"coverage ends at {cursor}, not {total}")


class SourceDocument(_Transient):
    """The COMPLETE canonical body plus the identity spans index against.

    Transient by contract: never stamped into ``meta``, never written to a
    ledger, never persisted in a receipt. Callers that need durability store
    ``body_sha256`` + offsets and re-derive the text from the document. See
    ``_Transient`` for why this is not a dataclass.

    Identity is REQUIRED, not defaulted, and the hash is checked against the
    body at construction -- so an identity-less or mislabelled document
    cannot exist, however it was built.
    """

    __slots__ = (
        "canonical_body", "body_sha256", "normalization_version", "source_ref",
    )

    def __init__(self, canonical_body: str, body_sha256: str,
                 normalization_version: str, source_ref: str = "") -> None:
        if not str(canonical_body or "").strip():
            raise SourceDocumentError(
                f"canonical body is empty (source_ref={source_ref!r})")
        if not body_sha256 or not normalization_version:
            raise SourceDocumentError(
                "a SourceDocument needs both body_sha256 and "
                "normalization_version; build it with build_source_document"
            )
        if body_sha256 != canonical_body_sha256(canonical_body):
            raise SourceDocumentError(
                f"body_sha256 does not hash this body "
                f"(source_ref={source_ref!r})"
            )
        self._set("canonical_body", canonical_body)
        self._set("body_sha256", body_sha256)
        self._set("normalization_version", normalization_version)
        self._set("source_ref", str(source_ref))

    def __repr__(self) -> str:
        return (f"SourceDocument(body_sha256={self.body_sha256[:12]!r}..., "
                f"chars={self.char_count}, "
                f"normalization_version={self.normalization_version!r}, "
                f"source_ref={self.source_ref!r})")

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
    if not normalization_version:
        raise SourceDocumentError("normalization_version is required")
    return SourceDocument(
        canonical_body=canonical_body,
        body_sha256=canonical_body_sha256(canonical_body),
        normalization_version=normalization_version,
        source_ref=source_ref,
    )


class SourceOverview(_Transient):
    """Whole-body coverage for the authors that run BEFORE the outline.

    ``windows`` tile the ENTIRE body in order with no gaps, so a consumer can
    state honestly which part of the work it read. ``evidence`` tags a few of
    those structural positions by role. Both are derived deterministically;
    no model participates.

    Transient like the document, and for a sharper reason: its windows hold
    every character of the body between them, so a structural serializer
    would reconstruct the whole work from them. ``receipt()`` is the durable
    form -- see ``_Transient``.

    ``requested_window_count`` keeps what the caller ASKED for beside what it
    got. A body shorter than two characters per window collapses to a single
    window; that is correct, but a silent collapse reads downstream as "this
    work is one undivided block". No silent caps.
    """

    __slots__ = (
        "body_sha256", "normalization_version", "overview_version",
        "windows", "evidence", "source_ref", "requested_window_count",
    )

    def __init__(self, body_sha256: str, normalization_version: str,
                 overview_version: str,
                 windows: tuple[SourceSpan, ...] = (),
                 evidence: tuple[SourceSpan, ...] = (),
                 source_ref: str = "",
                 requested_window_count: int = 0) -> None:
        self._set("body_sha256", body_sha256)
        self._set("normalization_version", normalization_version)
        self._set("overview_version", overview_version)
        self._set("windows", tuple(windows))
        self._set("evidence", tuple(evidence))
        self._set("source_ref", str(source_ref))
        self._set("requested_window_count", int(requested_window_count))

    def __repr__(self) -> str:
        return (f"SourceOverview(body_sha256={self.body_sha256[:12]!r}..., "
                f"windows={len(self.windows)}, evidence={len(self.evidence)}, "
                f"overview_version={self.overview_version!r}, "
                f"source_ref={self.source_ref!r})")

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
        hits = _count_quoted_spans(window.text)
        if not hits:
            continue
        density = hits / window.char_count
        if best is None or density > best_density:
            best, best_density = window, density
    return best
