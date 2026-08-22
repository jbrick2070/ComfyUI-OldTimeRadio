"""The VIDEO lane decides whether a portrait is minted at all.

Operator, 2026-08-22: *"still_word never needs a portrait, only words"*, and
*"the video lanes must be telling the image lanes what to do -- we built it in
each video lane."* He is right on both counts, and the second one names the
existing seam: `derive_image_prompts` is already handed FOUR lane-derived role
sets -- `still_aspects` (dimensions), `mesh_fodder_roles` (kind),
`talking_roles` (framing) and `still_word_roles` (which composer). Portraits
were the one image kind minted unconditionally, outside that mechanism.

`_portrait_free_roles_from_policy` is the fifth member of that family, and its
truth is the lane's OWN `still_plan` declaration -- never a hardcoded engine
name. The cheap family has declared `kind="portrait" required="never"` since it
was written; nothing read it (`still_plan_helpers`: *"Nothing in this module
reads the plan for production"*). So every still_word episode has minted a
portrait per cast member that no consumer on that lane ever loads: free on an
engine that will draw a face, FATAL on `ideogram4_local`, which returns a safety
placeholder for a person close-up and killed a live leg on 2026-08-22.

Pure CPU: no ComfyUI runtime, no GPU, no image model.
"""

from __future__ import annotations

import json

import pytest

from nodes import _otr_video_engines  # noqa: F401 -- self-registration
from nodes._otr_video_engines import registry as vreg
from nodes._otr_shared import still_plan_helpers as sp
from nodes.otr_meta_brief_image_prompt import _portrait_free_roles_from_policy


ALL_ROLES = {"announcer_visual", "music_visual", "character_video"}


def _policy(announcer, music, character):
    """The `image_policy_json` shape OTR_ImageDirector forwards. `video_models`
    is keyed by SLOT name, not role name -- the role->slot join is `role_slots`'
    job and getting it backwards silently yields NO exemptions."""
    return json.dumps({"video_models": {
        "announcer_video_model": {"engine_id": announcer},
        "music_video_model": {"engine_id": music},
        "character_video_model": {"engine_id": character},
    }})


# --------------------------------------------------------------------------- #
# The declaration is the truth
# --------------------------------------------------------------------------- #
def test_still_word_declares_it_never_needs_a_portrait():
    """The row that has been correct and unread. If this ever flips, the whole
    premise of the skip is gone and the test that notices should be this one."""
    plan = getattr(vreg.get_engine("still_word"), "still_plan", None) or ()
    portrait = [r for r in plan if str(getattr(r, "kind", "")) == sp.KIND_PORTRAIT]
    assert len(portrait) == 1
    assert str(portrait[0].required) == sp.REQUIRED_NEVER


@pytest.mark.parametrize("engine", ("still_word", "still_flat", "still_pan",
                                    "still_motion"))
def test_the_whole_still_family_is_portrait_free_on_every_role(engine):
    assert _portrait_free_roles_from_policy(_policy(engine, engine, engine)) == ALL_ROLES


def test_humo_KEEPS_its_character_portrait():
    """The counter-case that proves the mechanism reads reality rather than
    just saying yes. HuMo is `audio_driven_face`: it drives a talking mouth from
    the character portrait, so its plan declares portrait `always` and
    `render_driver` keeps `init_image = portrait` for it (it is NOT in
    `_SCENE_INIT_FAMILIES`). A blanket skip would have starved it."""
    free = _portrait_free_roles_from_policy(_policy("humo", "humo", "humo"))
    assert "character_video" not in free, "HuMo character portraits must survive"


def test_wan_ti2v_is_portrait_free_because_the_scene_still_overrides_it():
    """`wan_ti2v` is family `image_to_video`, which IS in
    `_SCENE_INIT_FAMILIES`, so render_driver replaces the portrait init with the
    per-beat scene still. Its plan says portrait `never` and that matches what
    the renderer actually does -- the declaration is finer-grained than "i2v
    lanes need a portrait" would suggest."""
    free = _portrait_free_roles_from_policy(_policy("wan_ti2v", "wan_ti2v", "wan_ti2v"))
    assert free == ALL_ROLES


def test_a_mixed_episode_is_judged_per_role_not_per_episode():
    """The hazard a per-episode skip would create: one role on a portrait-hungry
    lane must keep its portraits even when the others are portrait-free."""
    free = _portrait_free_roles_from_policy(_policy("still_word", "still_word", "humo"))
    assert "character_video" not in free
    assert {"announcer_visual", "music_visual"} <= free


# --------------------------------------------------------------------------- #
# Fail SAFE, never silently
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("policy_json", ["{not json", "", "null", "[]", "{}",
                                         '{"video_models": "nope"}'])
def test_a_broken_policy_grants_no_exemption(policy_json):
    """No exemption == the portrait IS minted == the pre-existing behaviour.
    A resolver failure must never DELETE an asset; the safe direction is the
    wasteful one."""
    assert _portrait_free_roles_from_policy(policy_json) == set()


def test_an_unregistered_engine_grants_no_exemption():
    assert _portrait_free_roles_from_policy(
        _policy("no_such_engine", "no_such_engine", "no_such_engine")) == set()


def test_an_engine_with_no_portrait_row_grants_no_exemption():
    """`ltx25_video` ships three scene rows and NO portrait row. Absence is not
    a declaration of `never`, so it mints -- conservative on purpose."""
    plan = getattr(vreg.get_engine("ltx25_video"), "still_plan", None) or ()
    assert not [r for r in plan if str(getattr(r, "kind", "")) == sp.KIND_PORTRAIT]
    assert _portrait_free_roles_from_policy(
        _policy("ltx25_video", "ltx25_video", "ltx25_video")) == set()


def test_it_reads_the_declaration_rather_than_naming_engines():
    """The point of sourcing from `still_plan`: a new lane -- an ultra-low-VRAM
    animatediff, anything -- is covered the day it declares its plan, with no
    edit to this helper. Proven by asserting the helper's own module text names
    no engine id."""
    import ast
    import inspect
    import textwrap

    from nodes import otr_meta_brief_image_prompt as mbp

    fn = ast.parse(textwrap.dedent(
        inspect.getsource(mbp._portrait_free_roles_from_policy))).body[0]
    # Drop the docstring node -- it deliberately CITES engine names as examples.
    # Only the executable body may not name one. (A source-text split on quotes
    # is the fragile way to do this and it is why this test failed first time.)
    body = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)
                           and isinstance(fn.body[0].value.value, str)) else fn.body
    code = chr(10).join(ast.unparse(node) for node in body)
    for engine_id in ("still_word", "still_flat", "wan_ti2v", "humo"):
        assert engine_id not in code, f"{engine_id} is hardcoded in the logic"


# --------------------------------------------------------------------------- #
# End to end through the enumerator
# --------------------------------------------------------------------------- #
def test_the_portrait_objects_actually_disappear():
    """The helper is only half the fix; the enumerator has to honour it."""
    from nodes.otr_meta_brief_image_prompt import derive_image_prompts

    cast = [{"char_id": "c01", "name": "JANE", "gender": "female",
             "portrait_prompt": "a tall weathered spacer with a scar"},
            {"char_id": "c02", "name": "GRIFFIN", "gender": "male",
             "portrait_prompt": "a broad man in a soaked overcoat"}]
    meta = {"episode_id": "ep_pf", "episode_title": "The Toll",
            "story_brief_terms": {"setting": ["a dockside station"]}}

    with_portraits, _ = derive_image_prompts(cast, meta, portrait_free_roles=set())
    without, _ = derive_image_prompts(cast, meta, portrait_free_roles=ALL_ROLES)

    def portraits(payload):
        return [o for o in (payload.get("objects") or [])
                if str(o.get("kind")) == "portrait"]

    assert portraits(with_portraits), "baseline must mint portraits"
    assert portraits(without) == [], "portrait-free roles must mint none"
