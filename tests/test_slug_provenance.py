"""Every CONCRETE provider model this pack ships must have provenance.

WHAT THIS GUARD CLAIMS, EXACTLY. It proves that **these listed lanes and these
registered engines are completely covered**. It does NOT claim the pack has no
unlisted model, and the claim is scoped to CHECKED-IN static and default model
selections -- `_otr_google_api/models.py` folds disk-cached ids into slot choices
at runtime, so no static enumeration can cover every runtime-selectable model.
Understating a guard is the cheap failure; overstating one is how BUG-12.86 got
promoted, and this suite exists because of that bug.

WHAT CHANGED 2026-08-10. The previous version of this file built its own inline
list from six hand-written imports, so it enforced "every slug reachable from
those six imports" while reading as a totality claim. It was green while
`gemini-3.1-flash-image-preview` -- the id `cloud_media_invoke` actually sends to
Google for the `Nano Banana 2` selector -- was invisible to it, because the guard
only ever saw the display NAME. It now imports `_otr_shared.slug_inventory`, the
same single collector the verifier uses, and is keyed on resolved provider ids.

It still does NOT claim the models are live. This suite has no network.
"""
from __future__ import annotations

import datetime
import os
import re
import subprocess
import sys

import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes import _otr_slug_provenance as prov                       # noqa: E402
from nodes._otr_shared import google_slug_verifier as core           # noqa: E402
from nodes._otr_shared import slug_inventory as inv                  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def records():
    return inv.collect_inventory()


@pytest.fixture(scope="module")
def shipped_pairs(records):
    """`(provider_id, lane)` for every non-pointer the pack offers."""
    return {(r.provider_id, r.authority_lane) for r in records
            if not prov.is_pointer(r.provider_id)}


# ===========================================================================
# THE GUARD, in both directions.
# ===========================================================================
def test_every_shipped_concrete_id_has_provenance(shipped_pairs):
    missing = sorted(shipped_pairs - set(prov.SLUG_PROVENANCE))
    assert not missing, (
        "%d shipped identity(ies) have no entry in nodes/_otr_slug_provenance.py:\n"
        % len(missing)
        + "\n".join("  %s  (lane %s)" % (pid, lane) for pid, lane in missing)
        + "\nAdd each with a verification_kind, and a date if one was earned."
    )


def test_provenance_carries_nothing_the_pack_no_longer_ships(shipped_pairs):
    orphans = sorted(set(prov.SLUG_PROVENANCE) - shipped_pairs)
    assert not orphans, "provenance lists identities no longer shipped: %s" % (orphans,)


def test_the_guard_now_sees_the_RESOLVED_id_not_the_display_name(records):
    """THE ORIGINAL DEFECT, pinned. The Nano Banana selector is a display name;
    the id that goes on the wire is what must carry provenance."""
    offered = {r.offered_value for r in records}
    assert "Nano Banana 2 (Gemini 3.1 Flash Image)" in offered
    # The display name is NOT what provenance is keyed on any more...
    assert not any(pid == "Nano Banana 2 (Gemini 3.1 Flash Image)"
                   for pid, _lane in prov.SLUG_PROVENANCE)
    # ...the resolved id is, and it is present.
    assert ("gemini-3.1-flash-image-preview", "comfy_image") in prov.SLUG_PROVENANCE


def test_both_display_mappings_reach_the_invoke_resolver():
    from nodes._otr_shared.google_image_model_ids import resolve_selector_to_model_id

    assert (resolve_selector_to_model_id("Nano Banana 2 (Gemini 3.1 Flash Image)")
            == "gemini-3.1-flash-image-preview")
    assert resolve_selector_to_model_id("Nano Banana 2 Lite") == "gemini-3.1-flash-lite-image"


def test_the_identity_fallthrough_is_preserved():
    """Raising on an unmapped selector would break every seedream / krea /
    photon row, none of which is mapped."""
    from nodes._otr_shared.google_image_model_ids import resolve_selector_to_model_id

    assert resolve_selector_to_model_id("seedream-4-0-250828") == "seedream-4-0-250828"
    assert resolve_selector_to_model_id("anything at all") == "anything at all"


def test_one_id_may_carry_two_authorities(records):
    """`gemini-3.1-flash-lite-image` arrives BOTH directly from eng_google_image
    and as the Comfy `Nano Banana 2 Lite` selector -- different auth and billing
    surfaces for one model. A slug-keyed table collapsed them."""
    lanes = {r.authority_lane for r in records
             if r.provider_id == "gemini-3.1-flash-lite-image"}
    assert {"google_image", "comfy_image"} <= lanes
    assert ("gemini-3.1-flash-lite-image", "google_image") in prov.SLUG_PROVENANCE
    assert ("gemini-3.1-flash-lite-image", "comfy_image") in prov.SLUG_PROVENANCE


# ===========================================================================
# Schema: kind and date must agree.
# ===========================================================================
def test_every_record_declares_a_defined_kind():
    for key, rec in prov.SLUG_PROVENANCE.items():
        assert rec.verification_kind in prov.VERIFICATION_KINDS, (key, rec)


def test_a_records_lane_agrees_with_its_own_key():
    for (pid, lane), rec in prov.SLUG_PROVENANCE.items():
        assert rec.authority_lane == lane, (pid, lane, rec.authority_lane)


def test_dated_kinds_carry_a_real_date_and_unverified_carries_none():
    for key, rec in prov.SLUG_PROVENANCE.items():
        if rec.verification_kind == prov.UNVERIFIED:
            assert rec.verified_on is None, (
                "%s is unverified but carries a date -- a date on an unverified "
                "row is a contradiction, not a bonus" % (key,))
        else:
            assert rec.verification_kind in prov.DATED_KINDS
            assert rec.verified_on, key
            # fromisoformat, not a regex: a regex accepts 2026-02-31.
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", rec.verified_on), (key, rec)
            datetime.date.fromisoformat(rec.verified_on)


def test_no_recorded_date_is_in_the_future():
    today = datetime.date.today()
    for key, rec in prov.SLUG_PROVENANCE.items():
        if not rec.verified_on:
            continue
        assert datetime.date.fromisoformat(rec.verified_on) <= today, (
            "%s is dated in the future -- that is malformed evidence" % (key,))


def test_the_kind_is_not_packed_into_the_date_string():
    """Both review lanes proposed `kind@YYYY-MM-DD`. Two facts in one string is
    the positional-parse defect this pack already has elsewhere."""
    for key, rec in prov.SLUG_PROVENANCE.items():
        assert "@" not in (rec.verified_on or ""), key


# ===========================================================================
# The five protections carried forward from the old schema.
# ===========================================================================
def test_no_shipped_id_carries_a_free_marker(records):
    """The hy3 defect: a ':free' id is a PRICE PROMISE baked into an IDENTIFIER,
    and promises expire while identifiers do not."""
    for rec in records:
        for marker in prov.FREE_MARKERS:
            assert marker not in rec.provider_id, (rec.provider_id, marker)
            assert marker not in rec.offered_value, (rec.offered_value, marker)


def test_pointers_are_excluded_and_the_exclusion_is_real(records):
    assert prov.is_pointer("~anthropic/claude-opus-latest")
    assert prov.is_pointer("gemini-flash-latest")
    assert not prov.is_pointer("gemini-2.5-pro")
    assert not prov.is_pointer("anthropic/claude-opus-4.7")
    # Both shapes really are shipped, so the exclusion is exercised.
    provider_ids = {r.provider_id for r in records}
    assert any(prov.is_pointer(p) for p in provider_ids)
    assert any(not prov.is_pointer(p) for p in provider_ids)


def test_pointers_are_never_version_dated():
    for pid, _lane in prov.SLUG_PROVENANCE:
        assert not prov.is_pointer(pid), (
            "%r is a pointer and must not be dated -- it has no version claim "
            "to go stale" % pid)


def test_every_lane_names_an_authority(records):
    lanes = {r.authority_lane for r in records}
    missing = sorted(lanes - set(prov.LANE_AUTHORITY))
    assert not missing, "lanes with no recorded authority: %s" % missing


def test_lane_authority_lists_nothing_unused(records):
    lanes = {r.authority_lane for r in records}
    assert not sorted(set(prov.LANE_AUTHORITY) - lanes)


# ===========================================================================
# Preview rules.
# ===========================================================================
@pytest.mark.parametrize("pid,expected", [
    ("lyria-3-clip-preview", True),            # suffix
    ("gemini-2.5-flash-preview-tts", True),    # infix
    ("gemini-3.1-flash-tts-preview", True),    # infix, reversed
    ("gemini-3.1-flash-image-preview", True),
    ("gemini-2.5-flash", False),
    ("previewer-1", False),                    # substring, NOT a token
    ("a-preview-b", True),
    ("preview", True),
    ("", False),
    (None, False),
])
def test_preview_detection_is_a_token_boundary_never_a_substring(pid, expected):
    assert prov.has_preview_token(pid) is expected


def test_a_preview_id_may_not_rest_at_unverified_unless_it_is_a_named_exception():
    offenders = []
    for key, rec in prov.SLUG_PROVENANCE.items():
        pid, _lane = key
        if not prov.has_preview_token(pid):
            continue
        if rec.verification_kind != prov.UNVERIFIED:
            continue
        if key in prov.PREVIEW_DATE_EXCEPTIONS:
            continue
        offenders.append(key)
    assert not offenders, (
        "preview id(s) resting at unverified with no named exception: %s"
        % offenders)


def test_the_exception_list_is_data_with_a_reason_not_a_silent_skip():
    assert prov.PREVIEW_DATE_EXCEPTIONS, (
        "an empty exception list would make the preview rule unshippable on day "
        "one, and a rule nobody can ship gets deleted instead of obeyed")
    for key, reason in prov.PREVIEW_DATE_EXCEPTIONS.items():
        assert key in prov.SLUG_PROVENANCE, key
        assert prov.has_preview_token(key[0]), key
        assert len(reason) > 40, ("a reason must actually explain", key)


def test_the_only_exception_is_the_escalated_vertex_proxy_id():
    """If this ever grows, someone widened an exception list instead of doing
    the verification -- which is the failure this rule is guarding."""
    assert set(prov.PREVIEW_DATE_EXCEPTIONS) == {
        ("gemini-3.1-flash-image-preview", "comfy_image")}


def test_preview_to_stable_is_directed_and_both_ends_are_really_shipped(records):
    provider_ids = {r.provider_id for r in records}
    for preview, stable in prov.PREVIEW_TO_STABLE.items():
        assert prov.has_preview_token(preview), preview
        assert not prov.has_preview_token(stable), stable
        assert preview in provider_ids, preview
        assert stable in provider_ids, stable


def test_twins_on_ONE_surface_are_banned_and_cross_surface_pairs_are_only_reported(records):
    """`eng_cloud_image` bills through the Comfy partner path and
    `eng_google_image` through a BYO key -- different auth and billing surfaces.
    A global ban would delete a valid one, so the ban is PER SURFACE."""
    by_surface: dict = {}
    for r in records:
        by_surface.setdefault(r.selection_surface, set()).add(r.provider_id)

    same_surface = []
    cross_surface = []
    for preview, stable in prov.PREVIEW_TO_STABLE.items():
        for surface, ids in by_surface.items():
            if preview in ids and stable in ids:
                same_surface.append((surface, preview, stable))
        p_surfaces = {s for s, ids in by_surface.items() if preview in ids}
        s_surfaces = {s for s, ids in by_surface.items() if stable in ids}
        if p_surfaces and s_surfaces and not (p_surfaces & s_surfaces):
            cross_surface.append((preview, stable))

    assert not same_surface, (
        "a preview and its stable twin are offered on ONE surface, so the "
        "operator can pick the preview by accident: %s" % same_surface)
    # The real pair IS cross-surface, so the reporting branch is exercised
    # rather than theoretical.
    assert cross_surface == [("gemini-3.1-flash-image-preview",
                              "gemini-3.1-flash-image")]


# ===========================================================================
# Completeness cross-check -- FAIL-CLOSED, never vacuous (BUG-12.87).
# ===========================================================================
#: Every registered audio+image engine maps to the collector that covers it, or
#: to an explicit exemption. NEVER inferred from an absent `native` flag:
#: `eng_cloud_image` declares `native` zero times, so `getattr(e, "native", True)`
#: would classify the whole cloud image lane as local and let it escape.
ENGINE_COVERAGE = {
    # --- audio: local engines ship presets or weights, not provider slugs ----
    "bark": "local", "kokoro": "local", "chatterbox": "local",
    "dia": "local", "indextts2": "local", "musicgen": "local",
    "stable_audio_3": "local", "stable_audio_music": "local",
    # --- audio: cloud engines ship concrete provider model ids ---------------
    "elevenlabs": "collected", "google_tts": "collected",
    "google_lyria": "collected",
    "sonilo": "exempt",
    # --- image: local ---------------------------------------------------------
    "z_image_turbo": "local", "flux_gen1": "local", "flux2_klein": "local",
    "lumina_image": "local",
    # --- image: cloud ---------------------------------------------------------
    "google_image": "collected", "cloud_nano_banana_2": "collected",
    "cloud_seedream_2": "collected", "cloud_krea_2_turbo": "collected",
    "cloud_luma_photon_flash": "collected",
    # cloud_flux_pro and ideo route through eng_cloud_image but ship no
    # concrete model collection of their own -- same shape as sonilo.
    "cloud_flux_pro": "exempt", "ideo": "exempt",
}


def _registered_engines():
    """Import the PACKAGES, not just the registry modules: registration happens
    through import-time side effects, so reading `registry.py` alone can observe
    a partial or empty registry."""
    import nodes._otr_audio_engines  # noqa: F401  -- imported for registration
    import nodes._otr_image_engines  # noqa: F401
    from nodes._otr_audio_engines import registry as areg
    from nodes._otr_image_engines import registry as ireg

    return (set(areg._REGISTRY)
            | set(ireg._IMAGE_REGISTRY.all_engine_names()))


def test_the_registry_is_not_empty_so_this_check_cannot_pass_vacuously():
    """BUG-12.87 -- a gate reporting success from its own skip path. Guarded
    imports mean a missing optional dep silently drops an engine, and a loop over
    an empty collection would PASS."""
    found = _registered_engines()
    # PINNED, not a floor with slack: guarded imports mean a missing optional
    # dep silently drops an engine, and a loop over a shrunken set still PASSES.
    # A strict subset must fail, so the expected set is written down.
    expected = set(ENGINE_COVERAGE)
    assert found == expected, (
        "registered engines differ from the pinned set.\n"
        "  dropped (guarded import failed?): %s\n"
        "  new (needs a coverage decision):  %s"
        % (sorted(expected - found), sorted(found - expected)))


def test_every_registered_engine_is_collected_or_explicitly_exempt():
    found = _registered_engines()
    unmapped = sorted(found - set(ENGINE_COVERAGE))
    assert not unmapped, (
        "engine(s) with no coverage decision: %s. Add each to ENGINE_COVERAGE "
        "as collected / local / exempt -- never infer it from a missing "
        "attribute." % unmapped)


def test_the_exemption_is_keyed_on_a_reason_not_a_bare_name():
    assert "sonilo" in inv.EXEMPT_ENGINES
    assert "partner row" in inv.EXEMPT_ENGINES["sonilo"]


def test_static_lanes_are_enumerated_separately_from_engines(records):
    """A registered engine name cannot represent every source: the Comfy LLM
    list and the Google text tuples are module constants with no adapter."""
    kinds = {r.source_kind for r in records}
    assert kinds == {"engine", "static_lane"}
    static = {r.source_key for r in records if r.source_kind == "static_lane"}
    assert any("COMFY_LLM_MODELS" in s for s in static)
    assert any("_otr_google_api.models." in s for s in static)


def test_inventory_uniqueness_is_per_source_surface_and_offered_value(records):
    seen = set()
    for r in records:
        key = (r.source_key, r.selection_surface, r.offered_value)
        assert key not in seen, "duplicate inventory record: %s" % (key,)
        seen.add(key)


def test_collecting_the_inventory_imports_no_video_module():
    """A FRESH PROCESS, because this one has already imported half the tree.
    Operator scope for this chunk was 'no video'."""
    code = (
        "import sys;"
        "sys.path.insert(0, %r);"
        "from nodes._otr_shared.slug_inventory import collect_inventory;"
        "collect_inventory();"
        "bad=[m for m in sys.modules if 'video' in m];"
        "print('VIDEO_MODULES=' + repr(bad))" % REPO_ROOT
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=180, cwd=REPO_ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "VIDEO_MODULES=[]" in out.stdout, out.stdout[-2000:]


# ===========================================================================
# Verifier core. All I/O injected -- these tests touch no socket.
# ===========================================================================
def _page(names, token=None):
    payload = {"models": [{"name": n} for n in names]}
    if token:
        payload["nextPageToken"] = token
    return payload


class _Getter:
    """Records every path it was asked for, so pagination is provable."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.paths = []

    def __call__(self, path):
        self.paths.append(path)
        nxt = self.pages.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


@pytest.mark.parametrize("raw,expected", [
    ("models/gemini-2.5-pro", "gemini-2.5-pro"),
    ("gemini-2.5-pro", "gemini-2.5-pro"),
    ("models/models/x", "models/x"),           # exactly ONE prefix stripped
    ("  models/x  ", "x"),
    ("", None), ("   ", None), (None, None), (123, None),
])
def test_normalization_strips_exactly_one_prefix(raw, expected):
    assert core.normalize_model_name(raw) == expected


def test_a_terminal_page_is_a_complete_fetch():
    got = core.fetch_catalog(_Getter([_page(["models/a", "b"])]))
    assert got.complete is True
    assert got.ids == frozenset({"a", "b"})
    assert got.pages == 1


def test_pagination_holds_pageSize_across_every_token():
    getter = _Getter([_page(["a"], token="T1"), _page(["b"], token="T 2"),
                      _page(["c"])])
    got = core.fetch_catalog(getter)
    assert got.complete is True
    assert got.ids == frozenset({"a", "b", "c"})
    assert len(getter.paths) == 3
    for path in getter.paths:
        assert "pageSize=200" in path
    assert "pageToken=T1" in getter.paths[1]
    assert "pageToken=T%202" in getter.paths[2], "token must be URL-encoded"


def test_a_never_terminating_token_is_incomplete_and_never_missing():
    getter = _Getter([_page(["a"], token="T")] * 60)
    got = core.fetch_catalog(getter, max_pages=5)
    assert got.complete is False
    assert got.exit_code == core.EXIT_MALFORMED_CATALOG


def test_a_partial_fetch_can_never_authorize_a_missing_verdict(records):
    incomplete = core.CatalogFetch(frozenset(), False, 1)
    report = core.build_report(
        records, prov.SLUG_PROVENANCE, catalog=incomplete,
        as_of=datetime.date(2026, 8, 10), is_pointer=prov.is_pointer,
        unverified_kind=prov.UNVERIFIED)
    assert report.missing == ()
    assert report.exit_code == core.EXIT_OK


def _real_client_error(status):
    """An exception shaped EXACTLY as the live client raises it.

    THIS FUNCTION IS THE POINT OF THE TEST. A previous version hand-rolled an
    exception carrying a `status` attribute -- a field no exception in this
    codebase actually has -- so the retry law was asserted against a fiction. It
    passed while auth failures were being retried three times in production.
    Building the exception through the client's own `_attach_evidence` means the
    test cannot drift from the shape the verifier will really meet.
    """
    from nodes._otr_google_api.client import _attach_evidence, GoogleAPIError

    return _attach_evidence(GoogleAPIError("boom %s" % status), status=status)


def test_the_retry_fixture_matches_the_real_client_shape():
    """Guards the guard. If the client stops stamping these fields, every retry
    assertion below silently becomes vacuous."""
    exc = _real_client_error(401)
    assert hasattr(exc, "retryable"), "client no longer stamps `retryable`"
    assert hasattr(exc, "http_status"), "client no longer stamps `http_status`"
    assert exc.retryable is False and exc.http_status == 401
    assert _real_client_error(503).retryable is True
    assert _real_client_error(None).retryable is True


@pytest.mark.parametrize("status,attempts", [
    (None, core.MAX_ATTEMPTS),    # transport -> retried
    (429, core.MAX_ATTEMPTS),     # rate limit -> retried
    (503, core.MAX_ATTEMPTS),     # 5xx -> retried
    (401, 1),                     # auth -> NEVER retried
    (403, 1),
    (400, 1),                     # request shape -> not retryable either
    (404, 1),
])
def test_retry_law(status, attempts):
    getter = _Getter([_real_client_error(status)] * (core.MAX_ATTEMPTS + 1))
    got = core.fetch_catalog(getter, sleep=lambda _s: None)
    assert len(getter.paths) == attempts, (
        "status %s was attempted %d time(s), expected %d"
        % (status, len(getter.paths), attempts))
    assert got.exit_code == core.EXIT_AUTH_OR_TRANSPORT


def test_an_exception_with_no_client_evidence_is_treated_as_transport():
    """A bare exception from somewhere unexpected still gets one more go rather
    than being classified as a permanent failure on no information."""
    getter = _Getter([RuntimeError("no evidence at all")] * 10)
    core.fetch_catalog(getter, sleep=lambda _s: None)
    assert len(getter.paths) == core.MAX_ATTEMPTS


def test_retry_exhaustion_is_never_converted_into_missing(records):
    getter = _Getter([_real_client_error(503)] * 10)
    catalog = core.fetch_catalog(getter, sleep=lambda _s: None)
    report = core.build_report(
        records, prov.SLUG_PROVENANCE, catalog=catalog,
        as_of=datetime.date(2026, 8, 10), is_pointer=prov.is_pointer,
        unverified_kind=prov.UNVERIFIED)
    assert report.missing == ()
    assert report.exit_code == core.EXIT_AUTH_OR_TRANSPORT


@pytest.mark.parametrize("payload", [
    {"models": "not a list"}, {"models": None}, {}, [],
])
def test_a_malformed_catalog_is_its_own_exit_code(payload):
    got = core.fetch_catalog(_Getter([payload]))
    assert got.exit_code == core.EXIT_MALFORMED_CATALOG


def test_shipped_ids_absent_from_a_complete_catalog_exit_2(records):
    catalog = core.CatalogFetch(frozenset({"nothing-we-ship"}), True, 1)
    report = core.build_report(
        records, prov.SLUG_PROVENANCE, catalog=catalog,
        as_of=datetime.date(2026, 8, 10), is_pointer=prov.is_pointer,
        unverified_kind=prov.UNVERIFIED)
    assert report.exit_code == core.EXIT_SHIPPED_IDS_MISSING
    assert report.missing


def test_unshipped_catalog_entries_are_not_a_finding(records):
    """Google listing a model this pack does not offer says nothing about
    this pack."""
    everything = {r.provider_id for r in records}
    catalog = core.CatalogFetch(frozenset(everything | {"some-other-model"}), True, 1)
    report = core.build_report(
        records, prov.SLUG_PROVENANCE, catalog=catalog,
        as_of=datetime.date(2026, 8, 10), is_pointer=prov.is_pointer,
        unverified_kind=prov.UNVERIFIED)
    assert report.missing == ()
    assert report.exit_code == core.EXIT_OK


def test_provenance_orphans_are_reported_separately_from_missing(records):
    doctored = dict(prov.SLUG_PROVENANCE)
    doctored[("a-model-we-dropped", "google_api")] = prov.ProvenanceRecord(
        "google_api", prov.UNVERIFIED, None)
    catalog = core.CatalogFetch(
        frozenset({r.provider_id for r in records}), True, 1)
    report = core.build_report(
        records, doctored, catalog=catalog, as_of=datetime.date(2026, 8, 10),
        is_pointer=prov.is_pointer, unverified_kind=prov.UNVERIFIED)
    assert ("a-model-we-dropped", "google_api") in report.orphaned
    assert report.missing == ()


def test_pointers_are_audited_separately_and_never_dated(records):
    catalog = core.CatalogFetch(
        frozenset({r.provider_id for r in records}), True, 1)
    report = core.build_report(
        records, prov.SLUG_PROVENANCE, catalog=catalog,
        as_of=datetime.date(2026, 8, 10), is_pointer=prov.is_pointer,
        unverified_kind=prov.UNVERIFIED)
    assert "gemini-flash-latest" in report.pointers
    for pid, _lane in prov.SLUG_PROVENANCE:
        assert pid not in report.pointers


# --- age -------------------------------------------------------------------
def test_age_is_a_pure_function_of_an_injected_as_of():
    assert core.age_in_days("2026-08-01", datetime.date(2026, 8, 11)) == 10
    assert core.age_in_days(None, datetime.date(2026, 8, 11)) is None
    assert core.age_in_days("not-a-date", datetime.date(2026, 8, 11)) is None


def test_an_ancient_date_never_fails(records):
    """A calendar-triggered red would fire on a fresh clone in 2027, on a box
    nobody touched -- and the first thing anyone would do is delete the check."""
    aged = {("gemini-2.5-pro", "google_api"): prov.ProvenanceRecord(
        "google_api", prov.CATALOG_LISTED, "2019-01-01")}
    report = core.build_report(
        records, aged, catalog=None, as_of=datetime.date(2030, 1, 1),
        is_pointer=prov.is_pointer, unverified_kind=prov.UNVERIFIED)
    assert report.exit_code == core.EXIT_OK
    assert report.ages and report.ages[0][1] > 4000


def test_a_future_date_is_reported_as_malformed(records):
    future = {("gemini-2.5-pro", "google_api"): prov.ProvenanceRecord(
        "google_api", prov.CATALOG_LISTED, "2030-01-01")}
    report = core.build_report(
        records, future, catalog=None, as_of=datetime.date(2026, 8, 10),
        is_pointer=prov.is_pointer, unverified_kind=prov.UNVERIFIED)
    assert ("gemini-2.5-pro", "google_api") in report.future_dated


def test_the_report_names_the_backlog_and_says_dates_were_not_written(records):
    report = core.build_report(
        records, prov.SLUG_PROVENANCE, catalog=None,
        as_of=datetime.date(2026, 8, 10), is_pointer=prov.is_pointer,
        unverified_kind=prov.UNVERIFIED)
    text = core.format_report(report, catalog=None,
                              lane_authority=prov.LANE_AUTHORITY)
    assert "UNVERIFIED backlog" in text
    assert "No dates were written" in text
    assert "NEVER fails" in text


def test_the_verifier_core_performs_no_network_access(monkeypatch, records):
    """Belt and braces on the injection contract: break sockets outright and
    run everything the unit tests run."""
    import socket

    def no_sockets(*a, **k):               # pragma: no cover
        raise AssertionError("verifier core attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_sockets, raising=False)
    monkeypatch.setattr(socket, "create_connection", no_sockets, raising=False)

    catalog = core.fetch_catalog(_Getter([_page(["models/x"])]))
    core.build_report(records, prov.SLUG_PROVENANCE, catalog=catalog,
                      as_of=datetime.date(2026, 8, 10),
                      is_pointer=prov.is_pointer,
                      unverified_kind=prov.UNVERIFIED)


def test_the_verifier_script_is_not_collected_by_pytest():
    assert os.path.exists(os.path.join(REPO_ROOT, "scripts", "verify_google_slugs.py"))
    assert not os.path.exists(os.path.join(REPO_ROOT, "tests", "verify_google_slugs.py"))


def test_the_exit_codes_are_pinned():
    """Callers and CI switch on these numbers, so they may not drift."""
    assert (core.EXIT_OK, core.EXIT_SHIPPED_IDS_MISSING, core.EXIT_NOT_RUN,
            core.EXIT_AUTH_OR_TRANSPORT, core.EXIT_MALFORMED_CATALOG) == (0, 2, 3, 4, 5)


# ===========================================================================
# Coverage statement, said out loud.
# ===========================================================================
def test_both_modules_state_the_limit_of_the_claim():
    for path in ("nodes/_otr_slug_provenance.py",
                 "nodes/_otr_shared/slug_inventory.py"):
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as fh:
            text = fh.read()
        assert "checked-in" in text.lower() or "checked-in constants" in text.lower(), path


def test_provenance_still_has_exactly_the_importers_we_think():
    """What makes the schema cheap to change -- and it will not stay true by
    accident."""
    # Matches an IMPORT, not a mention: `google_slug_verifier` names the module
    # in prose to explain why it will not write dates back, and a prose mention
    # is not a coupling.
    importer = re.compile(r"^\s*(?:from\s+\S*\s+)?import\s+.*_otr_slug_provenance"
                          r"|^\s*from\s+\S*_otr_slug_provenance\s+import",
                          re.MULTILINE)
    hits = []
    for root, _dirs, files in os.walk(REPO_ROOT):
        if any(p in root for p in (".git", "__pycache__", ".claude", "kibitz-runs")):
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            full = os.path.join(root, name)
            with open(full, encoding="utf-8", errors="ignore") as fh:
                if importer.search(fh.read()):
                    hits.append(os.path.relpath(full, REPO_ROOT).replace("\\", "/"))
    hits = sorted(h for h in hits if not h.endswith("_otr_slug_provenance.py"))
    assert hits == ["scripts/verify_google_slugs.py",
                    "tests/test_slug_provenance.py"], hits
