"""Every CONCRETE provider slug this pack ships must have provenance.

Chunk A dated and guarded the OpenRouter lane after `tencent/hy3:free` sat in
the dropdown until its promo ended and the slug stopped resolving. Five other
lanes kept the same shape -- a concrete version pin, no date, and nothing able
to notice it went stale (GO_FORWARD item 6).

WHAT THIS GUARD DOES AND DOES NOT CLAIM.

It does NOT claim the slugs are live. Nobody has checked most of them, and this
suite has no network. Pretending otherwise by stamping dates would be the
BUG-12.86 defect -- a field that reads as evidence and is not.

It DOES make the gap impossible to hide: every shipped concrete slug must carry
an entry in nodes/_otr_slug_provenance.py, either a real ISO date or an explicit
UNVERIFIED marker whose lane names the authority that could settle it. A NEW
undated slug fails the build; an old unverified one is counted out loud.

The distinction that matters: a missing entry is a slug nobody KNOWS about, an
UNVERIFIED entry is a slug everybody knows nobody has checked.
"""
from __future__ import annotations

import datetime
import os
import re
import sys

import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes import _otr_slug_provenance as prov  # noqa: E402


def _shipped_concrete_slugs() -> dict[str, str]:
    """Every concrete slug the pack OFFERS, mapped to where it came from.

    Imported from the live constants rather than a hand-kept list, so adding a
    slug to any of these lanes is what trips the guard -- not remembering to
    update a second list. Pointers are excluded: they resolve upstream and have
    no version claim to date.
    """
    found: dict[str, str] = {}

    def take(where: str, values) -> None:
        for v in values:
            if isinstance(v, str) and v and not prov.is_pointer(v):
                found.setdefault(v, where)

    from nodes import _otr_comfy_backend as comfy
    take("_otr_comfy_backend.COMFY_LLM_MODELS", comfy.COMFY_LLM_MODELS)

    from nodes._otr_google_api import models as gmodels
    for name in ("GOOGLE_API_LEGACY_TEXT_MODELS",
                 "GOOGLE_API_STABLE_TEXT_MODELS",
                 "GOOGLE_API_STATIC_TEXT_MODELS"):
        take(f"_otr_google_api.models.{name}", getattr(gmodels, name, ()))

    from nodes._otr_audio_engines import eng_cloud_elevenlabs as el
    take("eng_cloud_elevenlabs._SUPPORTED_MODELS", el._SUPPORTED_MODELS)

    from nodes._otr_image_engines import eng_cloud_image as ci
    for name in ("_NANO_MODELS", "_SEEDREAM_MODELS", "_KREA_MODELS",
                 "_LUMA_PHOTON_MODELS"):
        take(f"eng_cloud_image.{name}", getattr(ci, name, ()))

    from nodes._otr_image_engines import eng_google_image as gi
    take("eng_google_image.SUPPORTED_MODELS", gi.SUPPORTED_MODELS)

    return found


def test_every_shipped_concrete_slug_has_provenance():
    """THE GUARD. A new pin cannot be added without recording what is known
    about it -- which is the step that was missing when hy3 went stale."""
    shipped = _shipped_concrete_slugs()
    missing = {s: where for s, where in shipped.items()
               if s not in prov.SLUG_PROVENANCE}
    assert not missing, (
        f"{len(missing)} shipped slug(s) have no entry in "
        f"nodes/_otr_slug_provenance.py:\n" +
        "\n".join(f"  {s!r}  (from {w})" for s, w in sorted(missing.items())) +
        "\nAdd each with a verified-on date, or with UNVERIFIED if nobody has "
        "checked it. An entry is required; a date is not."
    )


def test_provenance_carries_no_slug_the_pack_no_longer_ships():
    """The other direction: a removed slug must not leave a stale entry
    implying the pack still offers it."""
    shipped = _shipped_concrete_slugs()
    orphans = [s for s in prov.SLUG_PROVENANCE if s not in shipped]
    assert not orphans, (
        f"provenance lists slug(s) the pack no longer ships: {sorted(orphans)}"
    )


def test_dates_are_real_iso_dates():
    """`fromisoformat`, not a regex: a regex accepts 2026-02-31."""
    for slug, (lane, when) in prov.SLUG_PROVENANCE.items():
        if when == prov.UNVERIFIED:
            continue
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", when), f"{slug}: {when!r}"
        datetime.date.fromisoformat(when)


def test_every_lane_names_an_authority():
    """An UNVERIFIED entry is only actionable if someone can say WHO would
    verify it. A lane with no authority turns a worklist back into a shrug."""
    lanes = {lane for lane, _ in prov.SLUG_PROVENANCE.values()}
    missing = sorted(lanes - set(prov.LANE_AUTHORITY))
    assert not missing, f"lanes with no recorded authority: {missing}"


def test_no_shipped_slug_carries_a_free_marker():
    """The hy3 defect, made unrepeatable in these lanes too."""
    for slug in _shipped_concrete_slugs():
        for marker in prov.FREE_MARKERS:
            assert marker not in slug, (
                f"{slug!r} carries {marker!r}: a ':free' id is a PRICE PROMISE "
                f"baked into an IDENTIFIER, and promises expire while "
                f"identifiers do not."
            )


def test_pointers_are_excluded_and_the_exclusion_is_real():
    """Guards the exclusion itself. If is_pointer() ever stopped recognising
    `-latest`, the Google pointers would demand nonsensical dates -- and if it
    over-matched, real pins would escape the guard entirely."""
    assert prov.is_pointer("~anthropic/claude-opus-latest")
    assert prov.is_pointer("gemini-flash-latest")
    assert not prov.is_pointer("gemini-2.5-pro")
    assert not prov.is_pointer("anthropic/claude-opus-4.7")
    # And the live Google list really does contain both shapes, so the
    # exclusion is exercised rather than theoretical.
    from nodes._otr_google_api import models as gmodels
    statics = list(gmodels.GOOGLE_API_STATIC_TEXT_MODELS)
    assert any(prov.is_pointer(s) for s in statics), statics
    assert any(not prov.is_pointer(s) for s in statics), statics


def test_the_unverified_backlog_is_visible(capsys):
    """Not a failure -- a report. Prints the worklist so it shows up in -s runs
    and in CI logs instead of decaying quietly inside a dict."""
    backlog = prov.unverified_slugs()
    by_lane: dict[str, int] = {}
    for lane in backlog.values():
        by_lane[lane] = by_lane.get(lane, 0) + 1
    print(f"\nUNVERIFIED provider slugs: {len(backlog)} "
          f"across {len(by_lane)} lane(s)")
    for lane in sorted(by_lane):
        print(f"  {lane:14s} {by_lane[lane]:3d}  authority: "
              f"{prov.LANE_AUTHORITY[lane]}")
    assert isinstance(backlog, dict)
