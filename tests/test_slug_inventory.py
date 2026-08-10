"""The shared selector mapping and the shared inventory.

WHAT THESE GUARD. The provenance guard could not see the model ids actually sent
to Google, because the pack offers image models by DISPLAY NAME and resolved
those names to ids inline at the invoke site. Two real ids were shipped with no
provenance entry at all. The fix is one shared mapping and one shared inventory;
these tests pin the properties that make sharing worth anything -- that there is
no second copy of the table, that the identity fallthrough survives, that a
provider id offered by two surfaces stays two records, and that none of it
reaches a video module.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_shared.google_image_model_ids import (  # noqa: E402
    SELECTOR_TO_MODEL_ID,
    resolve_selector_to_model_id,
)
from nodes._otr_shared.slug_inventory import (  # noqa: E402
    CATALOG_GOOGLE,
    EXEMPT_ENGINES,
    collect_inventory,
    google_catalog_records,
    provider_identities,
)

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# the shared mapping
# --------------------------------------------------------------------------

def test_mapped_selectors_resolve_to_their_wire_ids():
    assert (resolve_selector_to_model_id("Nano Banana 2 (Gemini 3.1 Flash Image)")
            == "gemini-3.1-flash-image-preview")
    assert resolve_selector_to_model_id("Nano Banana 2 Lite") == "gemini-3.1-flash-lite-image"


def test_unmapped_selector_resolves_to_itself():
    """The identity fallthrough is CONTRACT, not laziness. Most offered values
    are already concrete ids -- seedream, krea, photon -- and a resolver that
    raised on 'unknown' would break every one of those rows."""
    for concrete in ("seedream-4-5-251128", "photon-1", "Krea 2 Large",
                     "gemini-3.1-flash-image"):
        assert resolve_selector_to_model_id(concrete) == concrete


@pytest.mark.parametrize("bad", [None, 123, "", "   ", b"x"])
def test_resolver_rejects_input_that_is_not_a_selector(bad):
    """Unknown is fine; unusable is not."""
    with pytest.raises(ValueError):
        resolve_selector_to_model_id(bad)


def test_the_mapping_table_carries_no_identity_rows():
    """A row whose key equals its value is noise that will drift out of sync
    with the fallthrough. Keep the table to genuine display-name aliases."""
    identity = [k for k, v in SELECTOR_TO_MODEL_ID.items() if k == v]
    assert not identity, f"identity rows belong in the fallthrough, not the table: {identity}"


def test_invoke_site_uses_the_shared_resolver_and_keeps_no_second_copy():
    """THE POINT OF THE WHOLE MODULE. If cloud_media_invoke re-inlines the
    mapping, the guard goes blind again exactly as it was before -- so pin that
    the wire ids appear ONLY in the shared leaf."""
    invoke = (REPO / "nodes" / "_otr_shared" / "cloud_media_invoke.py").read_text(
        encoding="utf-8")
    assert "resolve_selector_to_model_id" in invoke, (
        "cloud_media_invoke must resolve through the shared leaf")
    for wire_id in SELECTOR_TO_MODEL_ID.values():
        assert wire_id not in invoke, (
            f"{wire_id!r} is hardcoded in cloud_media_invoke.py again -- that is "
            f"the second copy this module exists to delete")


# --------------------------------------------------------------------------
# the shared inventory
# --------------------------------------------------------------------------

def test_inventory_is_not_empty_and_every_record_is_well_formed():
    recs = collect_inventory()
    assert recs, "inventory collected nothing -- a vacuous guard passes everything"
    for r in recs:
        assert r.source_kind in ("engine", "static_lane"), r
        assert r.offered_value and isinstance(r.offered_value, str), r
        assert r.provider_id and not r.provider_id.startswith("models/"), r
        assert r.authority_lane, r
        assert r.catalog_target in (None, CATALOG_GOOGLE), r


def test_a_provider_id_offered_by_two_surfaces_stays_two_records():
    """The defect a flat {slug: where} dict hid. `gemini-3.1-flash-lite-image`
    is offered BOTH directly by eng_google_image and as the Comfy selector
    'Nano Banana 2 Lite' -- different auth and billing paths for one model."""
    recs = [r for r in collect_inventory()
            if r.provider_id == "gemini-3.1-flash-lite-image"]
    surfaces = {r.selection_surface for r in recs}
    assert len(recs) >= 2, f"expected the duplicate to survive collection, got {recs}"
    assert {"google_image", "comfy_nano"} <= surfaces, surfaces


def test_display_names_are_resolved_to_the_ids_actually_sent():
    """The two ids that were invisible to the old guard."""
    by_offered = {r.offered_value: r for r in collect_inventory()}
    nano = by_offered.get("Nano Banana 2 (Gemini 3.1 Flash Image)")
    assert nano is not None, "the Nano Banana selector left the inventory"
    assert nano.provider_id == "gemini-3.1-flash-image-preview"
    assert nano.offered_value != nano.provider_id, (
        "this row is the whole reason the inventory stores both")


def test_uniqueness_is_per_source_surface_and_offered_value():
    recs = collect_inventory()
    keys = [(r.source_key, r.selection_surface, r.offered_value) for r in recs]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"duplicate inventory rows: {sorted(dupes)}"
    # ...while provider_id is explicitly ALLOWED to repeat across sources.
    assert len({r.provider_id for r in recs}) < len(recs), (
        "expected at least one provider id offered by more than one source")


def test_google_projection_excludes_non_google_surfaces():
    """Without this projection the verifier would ask Google about ElevenLabs
    models and Comfy display selectors, then report them 'missing' -- a
    confident wrong answer."""
    google = google_catalog_records(collect_inventory())
    lanes = {r.authority_lane for r in google}
    assert "elevenlabs" not in lanes
    assert all(r.catalog_target == CATALOG_GOOGLE for r in google)
    assert google, "the Google projection collected nothing"
    # A Comfy SURFACE may still expose a Google provider id -- that is exactly
    # why catalog_target is separate from authority_lane.
    assert any(r.authority_lane == "comfy_image" for r in google), (
        "the nano rows resolve to Google ids and must remain checkable")


def test_provider_identities_are_the_pairs_provenance_keys_on():
    ids = provider_identities(collect_inventory())
    assert ("gemini-3.1-flash-image-preview", "comfy_image") in ids
    assert ("gemini-3.1-flash-lite-image", "google_image") in ids
    assert ("gemini-3.1-flash-lite-image", "comfy_image") in ids


def test_exempt_engines_record_a_REASON_not_just_a_name():
    """An allowlist of bare names decays into a shrug. `sonilo` is exempt
    because it ships a partner row rather than a model slug."""
    assert "sonilo" in EXEMPT_ENGINES
    for name, reason in EXEMPT_ENGINES.items():
        assert isinstance(reason, str) and len(reason) > 20, (name, reason)


# --------------------------------------------------------------------------
# the operator's scope ruling, enforced mechanically
# --------------------------------------------------------------------------

def test_collecting_the_inventory_imports_no_video_module():
    """The operator ruled 2026-08-09: no video. This proves it in a FRESH
    process rather than trusting naming discipline -- an import added three
    layers down would otherwise pass unnoticed."""
    script = (
        "import os,sys\n"
        "os.environ.setdefault('OTR_TEST_MODE','1')\n"
        "from nodes._otr_shared.slug_inventory import collect_inventory\n"
        "collect_inventory()\n"
        "bad=[m for m in sys.modules if '_otr_video_engines' in m]\n"
        "print('VIDEO_MODULES=%r' % (sorted(bad),))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=str(REPO),
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        "collecting the inventory pulled a video module:\n"
        f"{proc.stdout}\n{proc.stderr}")
