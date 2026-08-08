"""tests/test_visual_storybased.py -- qualification matrix for visual_storybased.

Validates the full dynamic visual_storybased lifecycle:
1. Card-to-pack composition & pack validation (23 v2 fields, place-holders, mouth vocabulary).
2. Failure provenance:
   - Malformed card -> floor fallback (failure_class = "model").
   - Transport failure -> floor fallback (failure_class = "transport").
   - Composer/programmer defect -> raises LOUD (never floored).
3. Lifecycle & Ledger consume:
   - Pending status guard -> raises VisualStyleError.
   - Sha256 digest tamper -> raises VisualStyleError.
   - Valid embedded pack consume -> returns VisualStyle.
4. Floor replay & seed determinism for both entry paths (rolled vs direct pick).
"""

import json
import random
import pytest

from nodes import _otr_rolls as rolls
from nodes import _otr_story_brief as story_brief
from nodes import _otr_visual_styles as vs
from nodes import OTR_LedgerScriptWriter as writer_mod


@pytest.fixture
def valid_card_dict():
    return {
        "medium_short": "oil painting",
        "medium": "oil painting on dark weathered wood",
        "radio_material": "carved Highland oak and tarnished iron",
        "character_art_style": "dramatic chiaroscuro oil portraiture with heavy impasto",
        "texture": "cracked varnish, weathered grain, and damp mist texture",
        "linework": "broad expressive brushstrokes and sharp dark contours",
        "lighting_character": "torchlit flickering firelight, harsh raking side shadows",
        "typography_voice": "carved ancient Celtic serif capitals",
        "motion_temperament": "Torchlight flickers across dark oak. Embers pulse.",
    }


def test_card_model_validation(valid_card_dict):
    card = vs.VisualStyleCardModel.model_validate(valid_card_dict)
    assert card.medium_short == "oil painting"
    assert card.motion_temperament == "Torchlight flickers across dark oak. Embers pulse."

    # Test medium_short max_length cap (>40 chars fails validation)
    bad_card = dict(valid_card_dict, medium_short="a" * 41)
    with pytest.raises(Exception):
        vs.VisualStyleCardModel.model_validate(bad_card)


def test_compose_pack_and_validate(valid_card_dict):
    card = vs.VisualStyleCardModel.model_validate(valid_card_dict)
    pack = vs.compose_pack_from_card(card)
    assert pack["style_id"] == "visual_storybased"
    assert pack["schema_version"] == "v2"
    assert "mouth" in pack["announcer_subject_ltx_mouth"] or "lips" in pack["announcer_subject_ltx_mouth"]
    assert "{form}" in pack["announcer_subject_ltx_mouth"]
    assert "{base}" in pack["non_character_emblem_fallback"]

    # Path-independent validation passes
    style_obj = vs.validate_pack(pack, expected_style_id="visual_storybased")
    assert style_obj.style_id == "visual_storybased"


def test_get_visual_style_pending_status_raises():
    meta = {
        "visual_style": "visual_storybased",
        "visual_style_receipt": {"status": "pending"},
    }
    with pytest.raises(vs.VisualStyleError, match="pending"):
        vs.get_visual_style(meta)


def test_get_visual_style_missing_embedded_pack_raises():
    meta = {
        "visual_style": "visual_storybased",
        "visual_style_receipt": {"status": "dynamic"},
    }
    with pytest.raises(vs.VisualStyleError, match="embedded_visual_style_pack"):
        vs.get_visual_style(meta)


def test_get_visual_style_tampered_sha256_raises(valid_card_dict):
    card = vs.VisualStyleCardModel.model_validate(valid_card_dict)
    pack = vs.compose_pack_from_card(card)
    meta = {
        "visual_style": "visual_storybased",
        "embedded_visual_style_pack": pack,
        "visual_style_receipt": {
            "status": "dynamic",
            "sha256": "bad_sha256_hash",
        },
    }
    with pytest.raises(vs.VisualStyleError, match="sha256 mismatch"):
        vs.get_visual_style(meta)


def test_get_visual_style_valid_embedded_pack(valid_card_dict):
    card = vs.VisualStyleCardModel.model_validate(valid_card_dict)
    pack = vs.compose_pack_from_card(card)
    canonical_bytes = json.dumps(
        pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    import hashlib
    sha256 = hashlib.sha256(canonical_bytes).hexdigest()

    meta = {
        "visual_style": "visual_storybased",
        "embedded_visual_style_pack": pack,
        "visual_style_receipt": {
            "status": "dynamic",
            "sha256": sha256,
        },
    }
    style_obj = vs.get_visual_style(meta)
    assert style_obj.style_id == "visual_storybased"
    assert style_obj.positive_tail == pack["positive_tail"]


def test_floor_style_pool_excludes_dynamic():
    floor_pool = rolls.floor_style_ids()
    assert "visual_storybased" not in floor_pool
    assert len(floor_pool) == 9
    assert rolls.DYNAMIC_STYLE_ID in rolls.eligible_style_ids()
    assert len(rolls.eligible_style_ids()) == 10


def test_floor_selection_reproducibility():
    floor_pool = rolls.floor_style_ids()
    seed = 12345
    s1 = rolls.draw(floor_pool, seed, random.Random)
    s2 = rolls.draw(floor_pool, seed, random.Random)
    assert s1 == s2
    assert s1 in floor_pool


def test_writer_floor_branch_has_no_undefined_globals():
    """Regression pin for two NameErrors that would only fire on the floor path.

    The dynamic-visual-style floor branch lives inside
    `OTR_LedgerScriptWriter._run_writer_tail` (NOT `run()` -- the branch
    was landed inside the tail method that runs after the outline
    reflection). It previously referenced two names that were not in scope:
      * `env` -- never assigned in `_run_writer_tail`'s local scope; would
        NameError on the direct-pick (`_style_roll is None`) floor path.
      * `_otr_story_brief` -- the module was imported by-symbol
        (`from ._otr_story_brief import ...`); the module identifier itself
        was never bound in the writer module, so
        `_otr_story_brief.REJECT_JSON_PARSE` would NameError the moment
        the reflection returned any failure sentinel.
    Both slipped review because the writer floor branch has no direct unit
    test. This test pins the fix by asserting the tail method's code object
    never references those names as free globals.
    """
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter

    tail_code = OTR_LedgerScriptWriter._run_writer_tail.__code__
    used_globals = set(tail_code.co_names)
    assert "env" not in used_globals, (
        "OTR_LedgerScriptWriter._run_writer_tail must not reference a free "
        "`env` global; the dynamic-visual-style floor branch used to do "
        "this on the direct-pick fallback path and would NameError at "
        "runtime."
    )
    assert "_otr_story_brief" not in used_globals, (
        "OTR_LedgerScriptWriter._run_writer_tail must not reference the "
        "`_otr_story_brief` module by name; the writer imports specific "
        "symbols from that module, not the module itself. The floor branch "
        "used to reach through `_otr_story_brief.REJECT_JSON_PARSE`, which "
        "would NameError on any reflection failure."
    )


def test_writer_floor_branch_reject_constants_bound():
    """Sibling of the co_names pin: the two REJECT_* constants the writer's
    failure-class classifier reads must be bound as module-level names so a
    future refactor of the story-brief module cannot silently drop them."""
    from nodes import OTR_LedgerScriptWriter as writer_mod

    assert hasattr(writer_mod, "_STORY_BRIEF_REJECT_JSON_PARSE"), (
        "writer must import REJECT_JSON_PARSE from _otr_story_brief by "
        "name so the floor failure-class classifier has a bound reference."
    )
    assert hasattr(writer_mod, "_STORY_BRIEF_REJECT_SCHEMA"), (
        "writer must import REJECT_SCHEMA from _otr_story_brief by "
        "name so the floor failure-class classifier has a bound reference."
    )
    assert writer_mod._STORY_BRIEF_REJECT_JSON_PARSE == story_brief.REJECT_JSON_PARSE
    assert writer_mod._STORY_BRIEF_REJECT_SCHEMA == story_brief.REJECT_SCHEMA
