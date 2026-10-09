"""Uncapped source document for source-owned lanes.

Why this module exists
----------------------
The fidelity packs instruct the model to CARRY the author's words, and until
now nothing put those words in front of it. Worse, the body the pipeline did
carry was a PREFIX: ``canonicalize_public_domain_text`` capped at 12,000
characters, so on a 25,200-word work the pre-outline authors -- the
interpreter that names the cast, the story contract, the outline -- read the
opening and inferred the rest. An author who is shown a prefix and told to be
faithful will invent the remainder and believe it complied.

This module owns the artifact that fixes that, deterministic and model-free:

``SourceDocument``
    The COMPLETE canonicalized body, its hash, and the normalization version
    that produced it. It is a TRANSIENT runtime artifact: it travels beside
    the payload for the duration of a build and is never serialized into
    ``meta``, the ledger, or a prompt receipt -- receipts carry offsets and
    hashes, never body text.

Nothing here loads a model, touches the GPU, reads the network, or imports
anything heavy.
"""
from __future__ import annotations

import hashlib

# Bump NORMALIZATION_VERSION when the canonical body bytes for an unchanged
# source would change -- that invalidates every stored offset and hash.
NORMALIZATION_VERSION = "otr_source_normalization_v1"


class SourceDocumentError(RuntimeError):
    """A source document could not be built or validated (loud)."""


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

    ``SourceOverview`` and its ``receipt()`` went with it on 2026-09-05; the
    rule it encoded stands for anything durable here: hashes, offsets and
    counts, never text.
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


class SourceDocument(_Transient):
    """The COMPLETE canonical body plus its identity: hash and normalization
    version.

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


