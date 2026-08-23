"""THE IMAGE-GEN PREFLIGHT MATRIX -- `docs/IMAGE_GEN_PREFLIGHT.md` as executable law.

The house pattern (`SOURCE_BANK_PREFLIGHT.md`, and the S8c video suite in
`tests/test_lane_preflight_matrix.py`): a preflight document is never written
before the code that enforces it, so this file exists first and the doc
narrates it.

WHY AN IMAGE-GEN SIBLING AT ALL, given the video matrix already exists.
`still_flat`, `still_word`, `still_pan` and `still_motion` are VIDEO dropdown
options -- they are lanes on `OTR_VideoDirector`'s video slots and the video
preflight already gates them (operator, 2026-08-21: *"still technically is a
video lane option"*). What had no gate of its own is the OTHER dropdown: the
per-role IMAGE model that actually mints the picture those lanes hold. That
subsystem has its own registry (`nodes/_otr_image_engines`), its own menu, its
own cache key and its own licence posture, and none of it is video.

Every gate below sweeps the LIVE registry rather than a whitelist, so a new
image engine is covered the day it registers.

NOT RE-ASSERTED HERE, because they are already enforced elsewhere and one
invariant deserves one owner: registry/CAPABILITIES parity for the image
registry lives in `tests/test_capability_profiles.py` (`set(ireg.CAPABILITIES)
== set(ireg.all_engine_names())`) and `tests/test_cloud_image_adapters.py`.

Pure CPU: no ComfyUI runtime, no GPU, no image model, no network.
"""

from __future__ import annotations

import pytest

from nodes import _otr_image_engines  # noqa: F401 -- adapters self-register on import
from nodes import otr_image_gen_dispatcher as disp
from nodes._otr_image_engines import registry as ireg
from nodes.otr_video_director import ADD_CUSTOM, _image_model_combo


#: The three per-role image slots OTR_VideoDirector exposes, and the object role
#: each one serves. Kept as the literal pair the operator actually sees, so a
#: silent rename of either side fails here.
_SLOT_FOR_ROLE = {
    "announcer_visual": "announcer_image_model",
    "music_visual": "music_image_model",
    "character_video": "character_image_model",
}


def _registered():
    return sorted(ireg.all_engine_names())


# --------------------------------------------------------------------------- #
# Gate IG1 -- declarations are explicit, never a caller's fallback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("engine_id", _registered())
def test_IG1_1_every_engine_declares_its_engine_version(engine_id):
    """IG1.1 `engine_version` is DECLARED on the adapter.

    The still cache key is `(role, object_id, prompt_hash, seed, engine_id,
    engine_version)` and the dispatcher reads the version as
    `getattr(engine, "engine_version", "1")`. An engine that never declares it
    therefore gets "1" from the fallback and has no way to invalidate its own
    cached stills -- the operator would have to change the prompt or the seed to
    get a different picture. That is worst on a PARTNER row, where the provider
    can move the model behind a stable id and nothing in this repo changes.

    Origin: 2026-08-21 census -- seven of eleven registered engines (the six
    `_CloudImageBase` rows plus `google_image`) declared no version. Declaring
    "1" explicitly is byte-identical to the fallback they were already getting,
    so the fix moved no cache key; it removed the silence.
    """
    version = getattr(ireg.get_engine(engine_id), "engine_version", None)
    assert isinstance(version, str) and version.strip(), (
        f"{engine_id} does not declare engine_version -- the dispatcher's "
        f"getattr fallback would silently pin it to '1' forever")


@pytest.mark.parametrize("engine_id", _registered())
def test_IG1_2_every_engine_declares_its_licence_posture(engine_id):
    """IG1.2 `commercial_clean` is declared as a real bool, never absent.

    Absent reads as False at some call sites and as "unknown" at others, and the
    honest answer differs per row: `flux_gen1` is False (Flux.1-dev is BFL
    non-commercial) while the Apache-2.0 locals are True. A missing declaration
    would let a non-commercial engine pass as clean by omission.
    """
    clean = getattr(ireg.get_engine(engine_id), "commercial_clean", None)
    assert isinstance(clean, bool), (
        f"{engine_id} must declare commercial_clean as a bool, got {clean!r}")


@pytest.mark.parametrize("engine_id", _registered())
def test_IG1_3_every_engine_declares_the_prompt_contract(engine_id):
    """IG1.3 `required_inputs` declares `text_prompt` -- the reduced
    prompt -> image contract every still composer writes against."""
    required = tuple(getattr(ireg.get_engine(engine_id), "required_inputs", ()) or ())
    assert "text_prompt" in required, (engine_id, required)


# --------------------------------------------------------------------------- #
# Gate IG2 -- the registry IS the menu
# --------------------------------------------------------------------------- #
def test_IG2_1_every_registered_engine_is_selectable_exactly_once():
    """IG2.1 The per-role dropdown is registry-derived, with no filter.

    C4, 2026-06-29 removed the validated-subset gate: every registered image
    engine is SELECTABLE and validation is the operator's manual process. So the
    menu must be the registry plus the custom sentinel -- nothing dropped (an
    engine that exists but cannot be chosen is dead code the operator is never
    told about) and nothing duplicated.
    """
    menu = _image_model_combo()
    assert menu[-1] == ADD_CUSTOM, "the custom escape hatch must remain last"
    offered = menu[:-1]
    assert len(offered) == len(set(offered)), f"duplicate menu entries: {offered}"
    assert sorted(offered) == _registered()


@pytest.mark.parametrize("role,slot", sorted(_SLOT_FOR_ROLE.items()))
def test_IG2_2_every_engine_declares_every_role_the_menu_offers_it_for(role, slot):
    """IG2.2 The menu offers every engine in all three slots, so every engine
    must DECLARE all three roles.

    The dropdowns are not per-role filtered: one unfiltered list is built and
    used for the announcer, music and character slots alike. An engine that
    declared only two roles would still be selectable in the third and would
    fail at render, after the episode had already been written and voiced.

    DECLARES, NOT SERVES (2026-08-22). This assertion reads `roles` off the
    registry; it renders nothing and cannot. `ideogram4_local` declares all
    three, passes this test, and then REFUSED at render on real card prose --
    so calling this "serves" was a promise the code never kept. The honest
    strengthening is a per-KIND declaration surface, which no image engine has
    today and which wants an arc; see the gap note in
    `docs/IMAGE_GEN_PREFLIGHT.md` under IG2.2.
    """
    assert set(ireg.engines_for_role(role)) == set(_registered()), (role, slot)


# --------------------------------------------------------------------------- #
# Gate IG3 -- the operator's pick is what renders
# --------------------------------------------------------------------------- #
def test_IG3_1_the_three_slots_resolve_independently_and_verbatim():
    """IG3.1 Each role mints on the model the operator picked FOR THAT ROLE.

    The image-side twin of the video preflight's G3.6. Three different engines
    in the three slots must come back as three different engines, each from its
    own slot and with `fallback_used` false -- no shared default, no silent
    substitution, no leakage of the character pick into the named slots.
    """
    picks = {"announcer_image_model": "z_image_turbo",
             "music_image_model": "flux2_klein",
             "character_image_model": "lumina_image"}
    policy = {"image_models": dict(picks)}
    for role, slot in _SLOT_FOR_ROLE.items():
        engine_id, resolved_slot, fell_back = disp.resolve_engine_for_role(policy, role)
        assert (engine_id, resolved_slot, fell_back) == (picks[slot], slot, False), role


def test_IG3_2_a_named_slot_left_empty_fails_loud_instead_of_substituting():
    """IG3.2 A dedicated slot that is PRESENT but blank is an operator config
    error and must raise -- never quietly borrow the character model.

    Origin: the 2026-07-03 no-fallback rip. A silent substitution renders a
    whole episode on a model the operator did not choose and never mentions it.
    """
    policy = {"image_models": {"announcer_image_model": "",
                               "character_image_model": "z_image_turbo"}}
    with pytest.raises(ValueError, match="PRESENT but EMPTY"):
        disp.resolve_engine_for_role(policy, "announcer_visual")
