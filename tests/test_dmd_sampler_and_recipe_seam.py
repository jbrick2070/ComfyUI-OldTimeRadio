"""The DMD restart sampler's ONE home + the wan_ti2v recipe seam.

Two changes land together here and each pins a defect the kibitz arc found:

* r2 F2 -- the transition used to live in the diagnostic bench helper. A
  production engine may not depend on it, and two copies of a sampling transition
  is how a silent divergence ships.
* r2 MF2 -- the recipe accessors read MODULE-level constants, so a subclass
  declaring its own recipe would have silently rendered with wan_ti2v's and
  stamped its own receipt on the result.
"""

import ast
import io
import os

from pathlib import Path

import pytest

from nodes._otr_video_engines import dmd_sampler as _DS
from nodes._otr_video_engines import eng_wan_ti2v as _WT

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# The sampler has exactly ONE owner
# --------------------------------------------------------------------------- #
def test_pack_owns_the_dmd_node_key():
    assert _DS.DMD_NODE_KEY == "OTR_DMDRestartSamplerSelect"
    assert _DS.NODE_CLASS_MAPPINGS[_DS.DMD_NODE_KEY] is _DS.OTR_DMDRestartSamplerSelect
    assert _DS.DMD_NODE_KEY in _DS.NODE_DISPLAY_NAME_MAPPINGS


def test_bench_helper_no_longer_registers_the_key():
    """A duplicate registration under one node key is ambiguous.

    This used to compare production against the VENDORED bench helper, which
    also declared ComfyUI nodes: the DMD sampler had to live in exactly one of
    them. The operator retired every bake-off on 2026-08-23, so the second
    declarer is gone and the guard is now ABSENCE -- there is no other tree that
    can claim this node key. `OTR_DMDRestartSamplerSelect` itself is unaffected:
    it moved OUT of the bakeoff category into production before any of this, and
    the rest of this file exercises it directly."""
    helper = Path(_REPO) / "scripts" / "bench_helper"
    assert not helper.exists(), (
        "scripts/bench_helper is back -- it declared ComfyUI nodes of its own, "
        "so a returning copy can silently re-register this node key against "
        "the production one in nodes/_otr_video_engines/dmd_sampler.py")
    assert not (Path(_REPO) / "scripts" / "bench_graphs").exists()


def test_root_registration_points_at_the_pack_module():
    root = io.open(os.path.join(_REPO, "__init__.py"), encoding="utf-8").read()
    assert ('"OTR_DMDRestartSamplerSelect": (".nodes._otr_video_engines.dmd_sampler"'
            in root)


def test_sampler_module_is_cold_import_clean():
    """No module-scope torch / comfy / tqdm -- this imports at pack load."""
    src = io.open(_DS.__file__, encoding="utf-8").read()
    heavy = {"torch", "comfy", "tqdm"}
    for node in ast.parse(src).body:          # TOP LEVEL only
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in heavy, a.name
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in heavy, node.module


def test_node_contract_is_preserved():
    cls = _DS.OTR_DMDRestartSamplerSelect
    assert cls.RETURN_TYPES == ("SAMPLER",)
    assert cls.RETURN_NAMES == ("SAMPLER",)
    assert cls.FUNCTION == "get_sampler"
    assert cls.INPUT_TYPES() == {"required": {}}
    # Moved OUT of the bakeoff category -- this is production machinery now.
    assert cls.CATEGORY == "OTR/video"
    assert "bakeoff" not in cls.CATEGORY.lower()


# --------------------------------------------------------------------------- #
# The fallback is now a refusal (r2 F2b)
# --------------------------------------------------------------------------- #
class _ModelSamplingMissing:
    """get_model_object raises -- the case that used to log FALLBACK-flow."""

    class inner_model:
        class model_patcher:
            @staticmethod
            def get_model_object(_name):
                raise KeyError("model_sampling")


class _NoiseScalingMissing:
    class inner_model:
        class model_patcher:
            @staticmethod
            def get_model_object(_name):
                return object()          # present, but no noise_scaling()


@pytest.mark.parametrize("model, needle", [
    (_ModelSamplingMissing, "could not resolve 'model_sampling'"),
    (_NoiseScalingMissing, "no callable noise_scaling"),
])
def test_sampler_refuses_by_name_before_step_1(model, needle):
    """It must REFUSE, not approximate.

    The old path stamped ``noise_scaling=FALLBACK-flow`` and rendered anyway with
    an inline CONST approximation -- loud, but the bench would have graded that a
    PASS for a transition that was never taken from the loaded model.

    Reachable with x=None / sigmas=None on purpose: the refusal fires before any
    tensor work, so 'before step 1' is literal and this test needs no GPU."""
    with pytest.raises(_DS.DmdModelSamplingError) as exc:
        _DS._dmd_restart_sampler(model, None, None)
    assert needle in str(exc.value)
    assert "NO FALLBACK" in str(exc.value)


def test_no_fallback_flow_path_remains_anywhere():
    # The vendored bench helper was the second file scanned here until the
    # operator retired every bake-off (2026-08-23). The guard is unchanged in
    # substance -- it always mattered for the PRODUCTION sampler, and that is
    # the file that survived.
    src = io.open(_DS.__file__, encoding="utf-8").read()
    assert "FALLBACK-flow" not in src.replace(
        "noise_scaling=FALLBACK-flow", "")  # only the comment may name it


# --------------------------------------------------------------------------- #
# The recipe seam (r2 MF2)
# --------------------------------------------------------------------------- #
def test_seam_is_behaviour_preserving_for_the_incumbent():
    """Each class-level owner points at the SAME object the methods read before."""
    assert _WT.WanTi2vEngine.recipe_id is _WT.RECIPE_WAN_TI2V
    assert _WT.WanTi2vEngine.recipe_data is _WT.WAN_TI2V_RECIPE
    assert _WT.WanTi2vEngine.recipe_env_keys is _WT._RECIPE_ENV_KEYS
    assert _WT.WanTi2vEngine.prequalification_env is _WT.PREQUALIFICATION_ENV
    assert _WT.WanTi2vEngine.max_frames_env == "OTR_WAN_TI2V_MAX_FRAMES"


def test_recipe_accessors_read_the_seam_not_module_globals():
    """THE DEFECT THIS CLOSES: a subclass declaring its own recipe used to be
    ignored, because the accessors read the module name directly."""

    class _Subclass(_WT.WanTi2vEngine):
        name = "seam_probe"
        recipe_data = dict(_WT.WAN_TI2V_RECIPE, steps=3, cfg=1.0,
                           negative="probe-negative", vae_tile=128)
        recipe_id = "seam_probe_v1"
        prequalification_env = "OTR_SEAM_PROBE_PREQUALIFICATION"

    sub = _Subclass()
    resolved = sub._resolve_render_config()
    assert resolved["steps"] == 3, "subclass recipe ignored -- seam not wired"
    assert resolved["cfg"] == 1.0
    assert sub._negative_prompt() == "probe-negative"
    assert sub._tile_geometry("vae_tile") == 128

    # ...and the incumbent is untouched by the subclass existing.
    inc = _WT.WanTi2vEngine()._resolve_render_config()
    assert inc["steps"] == _WT.WAN_TI2V_RECIPE["steps"]
    assert inc["cfg"] == _WT.WAN_TI2V_RECIPE["cfg"]


def test_max_frames_env_is_per_adapter():
    """A subclass must not inherit the WAN tier's ceiling channel: one key for
    two tiers lets one engine's ceiling silently cap another."""

    class _Subclass(_WT.WanTi2vEngine):
        name = "seam_probe"
        max_frames_env = "OTR_SEAM_PROBE_MAX_FRAMES"

    assert _Subclass.max_frames_env != _WT.WanTi2vEngine.max_frames_env


def test_planned_length_refusals_name_the_actual_engine():
    """The refusal text used to hardcode 'wan_ti2v', so a subclass would have
    refused under the wrong engine's name."""

    class _Subclass(_WT.WanTi2vEngine):
        name = "seam_probe"

    from nodes._otr_video_engines import wrapper_bridge as _wb
    with pytest.raises(_wb.GraphExecutionError) as exc:
        _Subclass()._planned_length(18)      # 18 is not on the 4n+1 ladder
    assert "seam_probe" in str(exc.value)
    assert "wan_ti2v was handed" not in str(exc.value)
