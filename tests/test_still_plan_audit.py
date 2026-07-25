"""S1 post-registration audit for the per-model still plans.

Chunk S1 of the per-model still-plans build
(``docs/2026-07-25-still-plans-locked-build-spec.md``, locked at
``84328aa1``). Every registered video engine adapter must own a
``still_plan`` class attribute; the plan must validate against the closed
schema in :mod:`nodes._otr_shared.still_plan_helpers`.

Spec section 6 -- "**Validation is a POST-REGISTRATION AUDIT, never
decorator-time.**" This test lives OUTSIDE the guarded
``nodes/_otr_video_engines/__init__.py`` import wrappers so a plan defect
surfaces here (a specific engine id + validator message) instead of silently
deleting the engine from the menu.

Section 6 sets the invariant:

  The audit runs outside those guards and compares THREE sets for exact
  equality: ``registry.CAPABILITIES`` keys, ``all_engine_names()``, and the
  set of ids owning a valid ``still_plan``.

  **A missing plan is UNKNOWN and fails closed. An explicit ``()`` means
  "needs no images".** Collapsing the two would make a forgotten
  declaration look like a visualizer.

Pure CPU: no ComfyUI runtime, no GPU, no image model, no LLM. Nothing reads
the plan for production at S1; S2 wires it into the seven consumers
atomically. This test is a fence -- it is the reason a plan typo in an
adapter class body cannot be silently dropped by the ``try/except`` in
``nodes/_otr_video_engines/__init__.py:94-209``.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Bootstrap: matches the S0a fixture bootstrap so imports resolve identically
# whether run under pytest or as a script.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")
if "folder_paths" not in sys.modules:
    _fp_stub = types.ModuleType("folder_paths")
    _fp_stub.get_output_directory = lambda: str(Path.cwd())  # noqa: E731
    _fp_stub.get_filename_list = lambda *a, **kw: []  # noqa: E731
    sys.modules["folder_paths"] = _fp_stub


import nodes._otr_video_engines as _video_pkg  # noqa: F401,E402
from nodes._otr_shared import still_plan_helpers as _sp  # noqa: E402
from nodes._otr_video_engines import registry as _vreg  # noqa: E402


def _all_registered_names():
    """Every registered video engine id, sorted (matches
    ``tests/test_still_plan_parity.py::_all_registered_video_engines``)."""
    return sorted(_vreg._VIDEO_REGISTRY._registry)


def _capability_names():
    """Keys of ``registry.CAPABILITIES`` sorted -- the CAPABILITIES table is
    what the shared capability profiles derive per-profile enable-sets from,
    so a row here without a registered engine (or vice versa) means the
    ``registry IS the menu`` invariant is broken."""
    return sorted(_vreg.CAPABILITIES.keys())


def _valid_plan_owner_names():
    """Every registered engine whose ``still_plan`` attribute exists and
    validates. Uses :func:`still_plan_helpers.engine_has_valid_still_plan`
    for the SET answer; ``test_every_registered_engine_owns_valid_still_plan``
    calls :func:`validate_still_plan` separately to surface the specific
    failure message per engine on a diff (rather than a bare
    'not equal' set diff)."""
    return sorted(
        name for name, eng in _vreg._VIDEO_REGISTRY._registry.items()
        if _sp.engine_has_valid_still_plan(eng))


def test_capabilities_match_registered_engines():
    """The ``registry IS the menu`` invariant that pre-dates S1: every
    ``CAPABILITIES`` row has a registered engine and every registered engine
    has a ``CAPABILITIES`` row. If S1 breaks this by accidentally deleting
    an adapter import (e.g. via a ``ValueError`` in a class body swallowed
    by ``nodes/_otr_video_engines/__init__.py:94-209``), this test names
    the specific id that disappeared."""
    assert _capability_names() == _all_registered_names()


def test_every_registered_engine_owns_valid_still_plan():
    """Spec section 6, the S1 audit invariant:

      ``registry.CAPABILITIES.keys() == all_engine_names() == valid-plan owners``

    Fails LOUD with the specific engine id + validator message on the FIRST
    plan defect (so pytest's short diff is diagnostic instead of a bare
    'set A != set B'). This is the fence against a plan typo silently
    deleting the engine from the menu: the ``try/except`` in
    ``nodes/_otr_video_engines/__init__.py:94-209`` swallows a ClassBody
    ``ValueError`` and drops the adapter, so a plan typo would present as
    'that model disappeared' -- and this test would catch that as a MISSING
    id in the plan-owner set, then the follow-up call names the actual
    ValueError so the operator sees WHICH field failed the schema.
    """
    all_ids = _all_registered_names()
    plan_owners = _valid_plan_owner_names()
    if all_ids != plan_owners:
        missing = [n for n in all_ids if n not in plan_owners]
        # Re-validate each missing engine to surface a specific ValueError.
        details = []
        for name in missing:
            eng = _vreg._VIDEO_REGISTRY._registry.get(name)
            if eng is None:
                details.append("%s: NOT REGISTERED" % name)
                continue
            if not _sp.engine_has_still_plan(eng):
                details.append(
                    "%s: NO still_plan class attribute (spec section 6 -- "
                    "MISSING is UNKNOWN and fails closed; explicit () means "
                    "needs-no-images)" % name)
                continue
            try:
                _sp.validate_still_plan(eng.still_plan)
            except ValueError as exc:
                details.append("%s: %s" % (name, exc))
            else:
                details.append("%s: unexpected -- validated after failing"
                               % name)
        assert False, (
            "S1 audit failed. Registered engines without a valid still_plan:"
            "\n  " + "\n  ".join(details))


def test_missing_still_plan_is_distinct_from_empty_tuple():
    """The audit's UNKNOWN vs "needs no images" boundary (spec section 6):

      A missing plan is UNKNOWN and fails closed. An explicit () means
      "needs no images". Collapsing the two would make a forgotten
      declaration look like a visualizer.

    Prove both branches concretely at HEAD: at least one registered engine
    (a visualizer) declares ``still_plan = ()``, and a fabricated engine
    with no ``still_plan`` attribute at all is REJECTED by the audit
    helpers.
    """
    # Explicit () is a VALID declaration (needs no images).
    class _EmptyPlanEngine:
        still_plan = ()
    assert _sp.engine_has_still_plan(_EmptyPlanEngine())
    assert _sp.engine_has_valid_still_plan(_EmptyPlanEngine())

    # A missing attribute is NOT a valid declaration -- the audit treats it
    # as UNKNOWN and rejects it.
    class _NoPlanEngine:
        pass
    assert not _sp.engine_has_still_plan(_NoPlanEngine())
    assert not _sp.engine_has_valid_still_plan(_NoPlanEngine())

    # And a HEAD visualizer proves the explicit () branch is exercised in
    # production: viz_camera / viz_green / viz_mxc_cpu / viz_mxc_mandala all
    # declare still_plan = () per spec section 3 (Shape C -- "nothing").
    at_least_one_empty = False
    for name in ("viz_camera", "viz_green", "viz_mxc_cpu", "viz_mxc_mandala"):
        if not _vreg._VIDEO_REGISTRY.is_registered(name):
            continue
        eng = _vreg._VIDEO_REGISTRY._registry[name]
        assert _sp.engine_has_valid_still_plan(eng), name
        assert eng.still_plan == (), name
        at_least_one_empty = True
    assert at_least_one_empty, (
        "No visualizer engine declares still_plan = () -- the audit's "
        "explicit-empty branch has no HEAD coverage")


def test_resolve_row_aspect_reads_engine_render_aspect():
    """The ONE aspect resolver every reader must go through
    (:func:`still_plan_helpers.resolve_row_aspect`). ``inherit_engine`` is
    NEVER compared directly (spec section 5), and unknown engine aspect
    resolves to portrait (the safe legacy look at HEAD, matching
    ``OTR_VideoDirector._role_aspects``). This test proves the resolver
    honours ``wide`` on both the ``render_aspect`` key (S1 caller shape)
    and the ``engine_render_aspect`` key (S0b's future ``engine_facts``
    shape) so downstream is not sensitive to which key S0b picks."""
    row = _sp.StillPlanRow(
        kind="portrait", cardinality="per_subject",
        target_class="portrait", aspect="inherit_engine", required="never",
        framing_geometry="", style_tail_policy="full")
    # inherit_engine + wide engine -> wide (both key shapes accepted).
    assert _sp.resolve_row_aspect(row, {"render_aspect": "wide"}) == "wide"
    assert _sp.resolve_row_aspect(
        row, {"engine_render_aspect": "wide"}) == "wide"
    # inherit_engine + portrait engine -> portrait.
    assert _sp.resolve_row_aspect(
        row, {"render_aspect": "portrait"}) == "portrait"
    # inherit_engine + missing key -> portrait (safe legacy look).
    assert _sp.resolve_row_aspect(row, {}) == "portrait"
    # Explicit wide / portrait rows short-circuit and never consult
    # engine_facts at all.
    wide_row = row._replace(aspect="wide")
    portrait_row = row._replace(aspect="portrait")
    assert _sp.resolve_row_aspect(wide_row, {}) == "wide"
    assert _sp.resolve_row_aspect(portrait_row, {}) == "portrait"
    assert _sp.resolve_row_aspect(
        wide_row, {"render_aspect": "portrait"}) == "wide"


def test_still_plan_schema_row_field_validation():
    """The closed schema (spec section 5) rejects out-of-enum tokens on
    every field. Uses one probe per field so an accidental widening of any
    enum trips this test."""
    valid_row = _sp.StillPlanRow(
        kind="portrait", cardinality="per_subject",
        target_class="portrait", aspect="inherit_engine", required="never",
        framing_geometry="", style_tail_policy="full")
    # The valid row validates cleanly.
    _sp.validate_still_plan_row(valid_row)
    # Now flip each closed-enum field to an invalid token and prove the
    # validator names the specific field.
    for field, bad_value in (
            ("kind", "not_a_kind"),
            ("cardinality", "per_episode"),
            ("target_class", "letterbox"),
            ("aspect", "square"),
            ("required", "sometimes"),
            ("style_tail_policy", "minimal_unclean")):
        bad_row = valid_row._replace(**{field: bad_value})
        try:
            _sp.validate_still_plan_row(bad_row)
        except ValueError as exc:
            assert field in str(exc), (field, exc)
        else:
            assert False, (
                "invalid %s=%r validated unexpectedly" % (field, bad_value))
    # framing_geometry must be a str (may be empty).
    try:
        _sp.validate_still_plan_row(valid_row._replace(framing_geometry=None))
    except ValueError as exc:
        assert "framing_geometry" in str(exc), exc
    else:
        assert False, "framing_geometry=None validated unexpectedly"


def test_still_plan_top_level_must_be_a_tuple():
    """Whole-plan validation surface: the top level must be a tuple. A
    single row wrapped in a list would silently be treated as a plan by a
    duck-typed consumer; the audit rejects that at S1."""
    row = _sp.StillPlanRow(
        kind="portrait", cardinality="per_subject",
        target_class="portrait", aspect="inherit_engine", required="never",
        framing_geometry="", style_tail_policy="full")
    _sp.validate_still_plan(())
    _sp.validate_still_plan((row,))
    try:
        _sp.validate_still_plan([row])  # a list is NOT a tuple
    except ValueError as exc:
        assert "tuple" in str(exc), exc
    else:
        assert False, "list-of-rows validated unexpectedly"
