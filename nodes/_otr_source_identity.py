"""ONE typed bibliographic identity for every source-bearing lane.

WHY THIS EXISTS. Three lanes each knew who wrote the thing they adapted, and
none of it reached the announcer. `public_domain` carries `title`/`author` in
`source_meta`; `shakespeare` carries `play_title` and has NO author field at
all; `media_archive` carries a feed label and a post URL but never stamped the
post's own headline anywhere durable. Every consumer that wanted "name the
work" was therefore reaching into a different dict with a different shape, and
the one that mattered -- `_otr_provenance.spoken_coda_line` -- reached into
none of them and emitted a sentence keyed only on licence status. That is how
an episode adapted from "Gertrude the Governess" closed by saying "a work in
the public domain", and how a Macbeth scene got titled after The Tempest.

THE RULE, AND IT WAS SETTLED BY THE CORPUS RATHER THAN BY TASTE. Name the
WORK TITLE and the AUTHOR. Never the unit label. 60 of the 65 public-domain
units carry a `label` that differs from `title`, which looks like a corpus full
of anthologies until you read them: dracula is "Harker at the castle",
frankenstein is "The meeting on the ice", monkeys_paw is "Three wishes". Those
are SCENE DESCRIPTORS, not piece titles. A "piece, from container" coda would
have announced a passage as if it were a work on roughly 57 of 65 units. Only
three or four units (gertrude_governess, speckled_band, queer_feet) are
genuinely anthology pieces, and nothing distinguishes them from a scene
descriptor programmatically.

WHAT THIS MODULE IS NOT. It reads. It never fetches, never calls a model, and
never raises -- a missing field yields a DEGRADED identity that says so in its
own provenance map, so a caller can choose a shorter sentence and stamp a
receipt instead of guessing or dying. Field-level provenance is the point: a
later reader must be able to ask "where did this author string come from"
without re-deriving the lane's shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

__all__ = [
    "ADAPTATION_SOURCE_KINDS",
    "SOURCE_IDENTITY_VERSION",
    "SourceIdentity",
    "identity_from_meta",
]

#: Bump when the FIELD SET or the extraction rules change, never for wording.
#: Receipts carry it so a ledger read years from now can tell which rules ran.
SOURCE_IDENTITY_VERSION = "source_identity_v1"

#: Shakespeare's source rows carry no author column and never will -- the lane
#: vendors scenes out of one playwright's works. Naming him is a FACT about the
#: work, not an inference about the row, so it is a constant here rather than a
#: default smuggled into the extractor.
_SHAKESPEARE_AUTHOR = "William Shakespeare"

#: `source_meta.kind` / bank ids this module knows how to read. Anything else
#: yields an empty identity rather than a guess.
_KIND_PUBLIC_DOMAIN = "public_domain"
_KIND_SHAKESPEARE = "shakespeare"
_KIND_MEDIA_ARCHIVE = "media_archive"

#: The lanes that ADAPT a pre-existing work, and therefore the only lanes whose
#: `work_title` may be announced as something the episode performs.
#:
#: This exists because `work_title` carries two different meanings. On the
#: adaptation lanes it is the work being performed ("Twelfth Night"). On
#: `media_archive` it is the PUBLICATION the post came from ("Now See Hear!") --
#: measured live: 56 of 98 media_archive ledgers on disk carry a `source_label`.
#: Announcing "a scene from Now See Hear!" would invent a play that does not
#: exist, which is a worse fidelity defect than the wrong-play frame item F was
#: opened to fix. Any consumer that means "the work this episode performs" must
#: gate on this set rather than on the truthiness of `work_title`.
ADAPTATION_SOURCE_KINDS = frozenset({_KIND_PUBLIC_DOMAIN, _KIND_SHAKESPEARE})


def _clean(value: Any) -> str:
    """Trim to a usable string. `None`, numbers and junk all become ''."""
    if value is None:
        return ""
    return str(value).strip()


@dataclass(frozen=True)
class SourceIdentity:
    """What a lane can truthfully say about the thing it adapted.

    Every string field is either a real value or ``""`` -- never ``None`` and
    never a placeholder -- so a caller can test truthiness without a
    three-way branch. ``provenance`` names the exact meta path each populated
    field came from, which is what makes a wrong value fixable at its source
    instead of argued about downstream.
    """

    version: str = SOURCE_IDENTITY_VERSION
    source_kind: str = ""
    work_title: str = ""
    author: str = ""
    post_headline: str = ""
    canonical_url: str = ""
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def names_a_work(self) -> bool:
        """True when there is enough to name something out loud.

        A work title alone is enough for a spoken credit; an author alone is
        not, because "adapted from a work by H. G. Wells" is vaguer than the
        generic sentence it would replace.
        """
        return bool(self.work_title or self.post_headline)

    @property
    def is_degraded(self) -> bool:
        """True when a field this lane SHOULD have carried is missing.

        Degraded is not fatal and is not an error -- it is a receipt-worthy
        statement that the coda had to say less than the lane intended.
        """
        if not self.source_kind:
            return True
        if self.source_kind == _KIND_MEDIA_ARCHIVE:
            return not self.post_headline
        return not (self.work_title and self.author)

    def as_receipt(self) -> Dict[str, Any]:
        """The durable form. Field-level provenance rides along deliberately."""
        return {
            "version": self.version,
            "source_kind": self.source_kind,
            "work_title": self.work_title,
            "author": self.author,
            "post_headline": self.post_headline,
            "canonical_url": self.canonical_url,
            "degraded": self.is_degraded,
            "field_provenance": dict(self.provenance),
        }


def _canonical_url(
    meta: Mapping[str, Any], source_meta: Mapping[str, Any]
) -> "tuple[str, str]":
    """Where the source URL lives differs by lane. Return (url, meta path).

    THE FIDELITY LANES KEEP IT IN THE OTHER SIDEBAR. `source_meta_from_unit`
    (`_otr_public_domain_sources.py:553-563`) and `source_meta_from_scene`
    (`_otr_shakespeare_sources.py:428-457`) both emit bibliographic fields ONLY
    -- no `source_url`. That key lives in the sibling rights record built by
    `source_rights_from_unit` / `source_rights_from_scene`. The RSS lanes are
    the other way round: `_rss_source_fetch_result`
    (`_otr_source_payload.py:604-610`) puts `source_url` INSIDE `source_meta`.

    Reading only `source_meta` therefore returned "" on every public_domain and
    shakespeare episode -- the two banks this code actually runs on -- while
    looking correct against a hand-built fixture. Try the bibliographic sidecar
    first, then the rights sidecar, and RECORD WHICH ONE ANSWERED so the
    receipt's field provenance stays truthful rather than merely present.
    """
    url = _clean(source_meta.get("source_url"))
    if url:
        return url, "meta.source_meta.source_url"
    rights = meta.get("source_rights")
    if isinstance(rights, Mapping):
        url = _clean(rights.get("source_url"))
        if url:
            return url, "meta.source_rights.source_url"
    return "", ""


def _source_kind_of(meta: Mapping[str, Any], source_meta: Mapping[str, Any]) -> str:
    """Which extraction shape applies.

    Prefers the BANK, because that is the operator-facing identity and the
    thing every ruling was written against; falls back to `source_meta.kind`,
    which is what the RSS lanes stamp.
    """
    bank = _clean(meta.get("source_bank")).lower()
    if bank in (_KIND_PUBLIC_DOMAIN, _KIND_SHAKESPEARE, _KIND_MEDIA_ARCHIVE):
        return bank
    kind = _clean(source_meta.get("kind")).lower()
    if kind.startswith("media_archive"):
        return _KIND_MEDIA_ARCHIVE
    return ""


def identity_from_meta(meta: Any) -> SourceIdentity:
    """Read a ledger's ``meta`` and return what it can truthfully claim.

    Never raises. An unreadable or unknown-lane meta yields an empty identity
    whose ``is_degraded`` is True, which callers render as the generic coda.
    """
    if not isinstance(meta, Mapping):
        return SourceIdentity()
    source_meta = meta.get("source_meta")
    if not isinstance(source_meta, Mapping):
        source_meta = {}

    kind = _source_kind_of(meta, source_meta)
    prov: Dict[str, str] = {}
    url, url_path = _canonical_url(meta, source_meta)
    if url_path:
        prov["canonical_url"] = url_path

    if kind == _KIND_SHAKESPEARE:
        title = _clean(source_meta.get("play_title"))
        if title:
            prov["work_title"] = "meta.source_meta.play_title"
            prov["author"] = "constant (the lane vendors one playwright)"
        return SourceIdentity(
            source_kind=kind,
            work_title=title,
            author=_SHAKESPEARE_AUTHOR if title else "",
            canonical_url=url,
            provenance=prov,
        )

    if kind == _KIND_PUBLIC_DOMAIN:
        # `unit_label` is deliberately NOT read -- see the module docstring.
        title = _clean(source_meta.get("title"))
        author = _clean(source_meta.get("author"))
        if title:
            prov["work_title"] = "meta.source_meta.title"
        if author:
            prov["author"] = "meta.source_meta.author"
        return SourceIdentity(
            source_kind=kind,
            work_title=title,
            author=author,
            canonical_url=url,
            provenance=prov,
        )

    if kind == _KIND_MEDIA_ARCHIVE:
        # The post's own headline is the thing worth naming; the feed label
        # ("Now See Hear!") is the publication, and both are said aloud.
        headline = _clean(source_meta.get("post_headline"))
        publisher = _clean(source_meta.get("source_label"))
        if headline:
            prov["post_headline"] = "meta.source_meta.post_headline"
        if publisher:
            prov["work_title"] = "meta.source_meta.source_label"
        return SourceIdentity(
            source_kind=kind,
            work_title=publisher,
            post_headline=headline,
            canonical_url=url,
            provenance=prov,
        )

    return SourceIdentity(source_kind=kind)
