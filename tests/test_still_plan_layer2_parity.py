"""S1b: the plan's layer-2 text must BE the producer's geometry, verbatim.

Chunk S1b of the per-model still-plans build
(``docs/2026-07-25-still-plans-locked-build-spec.md``, locked at ``84328aa1``;
corrected plan ``docs/2026-07-25-still-plans-r4-corrected-plan.md``).

S1 declared 31 ``still_plan`` attributes whose ``framing_geometry`` strings
were PARAPHRASES of the producer's real layer-2 text. Spec section 5 makes
that field the literal layer-2 prompt TEXT ("authored TEXT -- the ONE free
field"), so wiring S1 as it stood would have silently degraded every prompt in
the system: ``mesh_fodder`` had lost the whole clay-blob clause,
``scene_background_plate`` had lost "no people, no subject, no characters",
and the ``portrait`` row was the EMPTY STRING on 19 engines.

This file is the fence that keeps the transplant honest. It pins THREE
invariants, all of which are about DRIFT rather than taste -- none of them
judges prose, so none of them can collide with THE LAW:

1. **Every row's ``framing_geometry`` IS the producer's constant** for that
   row's ``kind`` (and, for ``portrait``, the engine's shipped
   ``render_aspect`` / talking register). Not "contains", not "looks like" --
   equality. If someone reworders the producer's geometry without updating the
   plans, or vice versa, this test names the engine and the row.

2. **Layer 2 never carries layer-3 LOOK vocabulary.** The chunk-A1 split
   (``otr_meta_brief_image_prompt.py:96-104``) makes geometry ENGINE-SAFETY
   framing owned by Python and the LOOK segment (costume / environment /
   lighting) PACK-OWNED (``VisualStyle.portrait_look`` /
   ``portrait_look_talking`` / ``plate_look``). A plan row that swallowed a
   ``*_LOOK_DEFAULT`` would hard-code the sci_fi_radio pack's look into an
   engine and quietly take a decision away from the style authority -- spec
   section 4: a plan "may only contribute layer 2 ... it may never decide
   style".

3. **No plan object is shared by engines of DIFFERENT shipped aspects.** The
   producer picks between ``PORTRAIT_GEOMETRY`` and ``WIDE_PORTRAIT_GEOMETRY``
   by aspect (``_style_anchor_for_aspect``), so ONE static portrait string
   cannot serve both. At S1 all four HuMo variants shared one plan object
   across two aspects; S1b splits it. This invariant is what stops that
   regrowing -- and it is also the spec's own "no inheritance or shared
   defaults" rule, made executable.

Pure CPU: no ComfyUI runtime, no GPU, no image model, no LLM. Nothing reads
the plan for production at S1b either -- S2 wires it in.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Bootstrap: identical to tests/test_still_plan_audit.py so imports resolve the
# same whether run under pytest or as a script.
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
from nodes import _otr_story_brief_helpers as _sbh  # noqa: E402
from nodes import otr_meta_brief_image_prompt as _mb  # noqa: E402
from nodes._otr_video_engines import registry as _vreg  # noqa: E402


#: kind -> the producer constant that IS that kind's layer 2. ``portrait`` is
#: resolved separately because the producer switches on aspect + talking.
_KIND_GEOMETRY = {
    "scene_open": _sbh.STILL_FRAMING_OPEN,
    "scene_beat": _sbh.STILL_FRAMING_SCENE_BEAT,
    "scene_character": _sbh.STILL_FRAMING_SCENE_CHARACTER,
    "mesh_fodder": _mb.MESH_FODDER_POS_SCAFFOLD,
    "scene_background_plate": _mb.BACKGROUND_PLATE_GEOMETRY,
}

#: The pack-owned LOOK segments. None of these may appear inside a plan row.
_LOOK_SEGMENTS = (
    _mb.PORTRAIT_LOOK_DEFAULT,
    _mb.TALKING_PORTRAIT_LOOK_DEFAULT,
    _mb.PLATE_LOOK_DEFAULT,
)


def _registered():
    return sorted(_vreg._VIDEO_REGISTRY._registry.items())


def _expected_portrait_geometry(engine, row):
    """The producer's portrait geometry for one row.

    Three cases, in the order the producer resolves them:

    - ``per_bookend_role`` is the LTX radio face. The producer mints it at
      ``otr_meta_brief_image_prompt.py:1782-1790`` via
      ``build_radio_host_prompt(meta, "wide", radio_host_style=
      "ltx_radio_mouth")``, and that branch calls
      ``_style_anchor_for_aspect("wide", style=...)`` with NO talking flag --
      so it is the WIDE portrait anchor, not the talking close-up bust. This
      case is checked FIRST because the row is also
      ``required="when_engine_talking"`` and would otherwise fall into the
      talking branch below.
    - a row gated on the talking register is the CAST portrait, which DOES pass
      ``talking=True`` (``otr_meta_brief_image_prompt.py:1166``).
    - everything else follows the engine's shipped aspect: wide gets the
      head-and-shoulders medium because a three-quarter body shot decapitates
      the subject in a short landscape frame (the 2026-06-17 operator catch).
    """
    if row.cardinality == "per_bookend_role":
        return _mb.WIDE_PORTRAIT_GEOMETRY
    if row.required == "when_engine_talking":
        return _mb.TALKING_PORTRAIT_GEOMETRY
    aspect = getattr(engine, "render_aspect", None)
    if aspect == "portrait":
        return _mb.PORTRAIT_GEOMETRY
    return _mb.WIDE_PORTRAIT_GEOMETRY


def test_every_plan_row_carries_the_producers_geometry_verbatim():
    """Invariant 1 -- equality, per engine, per row. The failure message names
    the engine id, the row kind and both strings, because a near-miss reword is
    exactly the defect that a set-difference assertion would hide."""
    mismatches = []
    for name, engine in _registered():
        plan = getattr(engine, "still_plan", ())
        for i, row in enumerate(plan):
            if row.kind == "portrait":
                expected = _expected_portrait_geometry(engine, row)
            else:
                expected = _KIND_GEOMETRY[row.kind]
            if row.framing_geometry != expected:
                mismatches.append(
                    "%s row %d (kind=%s, required=%s):\n"
                    "      PLAN: %r\n"
                    "  PRODUCER: %r"
                    % (name, i, row.kind, row.required,
                       row.framing_geometry, expected))
    assert not mismatches, (
        "still_plan framing_geometry has drifted from the producer's layer-2 "
        "geometry constants (S1b transplant):\n\n" + "\n\n".join(mismatches))


def test_no_plan_row_carries_pack_owned_look_vocabulary():
    """Invariant 2 -- the chunk-A1 geometry/LOOK boundary, made executable. A
    plan row that swallowed a LOOK segment would hard-code one style pack's
    look into an engine and take the decision away from the style authority."""
    leaks = []
    for name, engine in _registered():
        for i, row in enumerate(getattr(engine, "still_plan", ())):
            for look in _LOOK_SEGMENTS:
                if look and look in row.framing_geometry:
                    leaks.append(
                        "%s row %d (kind=%s) contains pack-owned LOOK %r"
                        % (name, i, row.kind, look))
    assert not leaks, (
        "layer-2 framing_geometry must never carry layer-3 LOOK vocabulary "
        "(spec section 4: a plan may only contribute layer 2 and may never "
        "decide style):\n  " + "\n  ".join(leaks))


def test_no_plan_object_is_shared_across_differing_aspects():
    """Invariant 3 -- the MUST-SPLIT fence. Two engines may share a plan OBJECT
    only if they ship the SAME ``render_aspect``; otherwise one static portrait
    string would have to serve both a portrait and a wide renderer."""
    by_object = {}
    for name, engine in _registered():
        plan = getattr(engine, "still_plan", None)
        if not plan:
            continue
        rec = by_object.setdefault(id(plan), {"engines": [], "aspects": set()})
        rec["engines"].append(name)
        rec["aspects"].add(getattr(engine, "render_aspect", None))
    bad = ["%s share one plan object but ship aspects %s"
           % (", ".join(sorted(rec["engines"])), sorted(map(str, rec["aspects"])))
           for rec in by_object.values() if len(rec["aspects"]) > 1]
    assert not bad, (
        "a still_plan object shared across differing shipped aspects cannot "
        "carry one portrait geometry (spec section 6: no inheritance or "
        "shared defaults):\n  " + "\n  ".join(bad))


def test_portrait_rows_are_non_empty_on_every_engine_that_has_one():
    """The specific S1 regression this chunk closes: the ``portrait`` row was
    the EMPTY STRING on 19 engines, so layer 2 vanished for every portrait the
    plan would have authored."""
    empty = [
        "%s row %d (kind=portrait, required=%s)" % (name, i, row.required)
        for name, engine in _registered()
        for i, row in enumerate(getattr(engine, "still_plan", ()))
        if row.kind == "portrait" and not row.framing_geometry.strip()
    ]
    assert not empty, (
        "portrait rows with no layer-2 text:\n  " + "\n  ".join(empty))
