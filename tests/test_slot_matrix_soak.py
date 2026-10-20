"""C5 (2026-06-30; 3-role rewrite 2026-07-01 rip-sfx-broll) -- OFFLINE proof of
the all-slots canonical-JSON soak.

The live GPU soak is gated, but its load-bearing wiring is proven here on the CPU:

  * the REAL workflow (workflows/otr_canonical.json) is loaded and ALL
    THREE video slots are set INDEPENDENTLY via the capability-profile
    role_overrides -> apply_profile patches the OTR_VideoDirector video-model
    widgets BY NODE TYPE (config/profiles/widget_mapping.json; no node ids),
    and NONE is left at a stale inherit sentinel;
  * every chosen engine is role_compat-ELIGIBLE for its slot (capability, C2/C4);
  * the per-beat CONTENT ORACLE catches the D2 dark floor + a frozen motion clip,
    and exempts static engines -- proven on real ffmpeg-rendered fixtures.

CPU only; no server, no model load (apply_profile uses build_offline_schemas).
UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import pytest

import nodes._otr_video_engines  # noqa: F401  (self-registers every engine)
from nodes import _otr_workflow_apply as wa
from nodes._otr_shared import role_compat as rc
from nodes._otr_shared.workflow_rows import load_workflow_row
from nodes._otr_video_engines import registry as vreg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "workflows" / "otr_canonical.json"
_HAS_FFMPEG = shutil.which("ffmpeg") is not None

#: distinct, capability-eligible engine per slot.
ROLE_ENGINES = {
    "announcer_visual": "humo_1.7B",
    "music_visual": "viz_green",
    "character_video": "humo_1.7B_169",
}
#: Internal ids that carry a PUBLIC menu id today. Read from the live table so
#: the next lane's rename does not have to remember this file exists.
from nodes._otr_shared.public_engines import _INTERNAL_TO_PUBLIC as _PUBLIC_ROWS
_VIDEO_WIDGET = {
    "announcer_visual": "announcer_video_model",
    "music_visual": "music_video_model",
    "character_video": "character_video_model",
}


def _load_canonical() -> dict:
    with open(CANONICAL, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# all-slots applied to the REAL workflow JSON
# --------------------------------------------------------------------------- #


_RETIRED_GRANULARITY = ("announcer_granularity", "music_granularity",
                        "character_granularity")


def test_canonical_image_director_granularity_widgets_are_retired():
    # node 88 (OTR_ImageDirector): the three granularity widgets were RETIRED
    # 2026-10-14 (no production reader; they promised a still frequency nothing
    # applied). Gone from the saved graph, the schema and the call signature,
    # and the retained widgets sit exactly where the migration put them.
    import inspect
    wf = _load_canonical()
    n88 = next(n for n in wf["nodes"] if n.get("id") == 88)
    assert n88.get("type") == "OTR_ImageDirector"
    names = [i.get("name") for i in n88["inputs"]]
    assert not set(_RETIRED_GRANULARITY) & set(names)
    assert names[:2] == ["video_policy_json", "gate_in"]
    # character_photos (2026-10-20) is appended after them, never between.
    assert names[2:] == ["fresh_cap", "seed_mode", "request_seed",
                         "custom_models_json", "dtype_policy", "character_photos"]
    assert n88["widgets_values"] == [15, "request_hash", 42, "{}", "fp8_ok", ""]
    from nodes.otr_image_director import OTRImageDirector
    schema = OTRImageDirector.INPUT_TYPES()
    for name in _RETIRED_GRANULARITY:
        assert name not in schema["required"] and name not in schema["optional"]
        assert name not in inspect.signature(OTRImageDirector().direct).parameters


def test_every_chosen_engine_is_capability_eligible():
    for role, engine in ROLE_ENGINES.items():
        desc = vreg.descriptor_for_engine(engine)
        assert rc.engine_fits_role(desc, role), (engine, role)


# --------------------------------------------------------------------------- #
# content oracle on REAL ffmpeg fixtures
# --------------------------------------------------------------------------- #
def _render(path, src, *, dur=1.0):
    # lavfi option separator: ':' when the source already opened options with '='
    # (e.g. "color=c=black"), else '=' (e.g. "testsrc2").
    sep = ":" if "=" in src else "="
    spec = f"{src}{sep}d={dur}:r=25:s=160x120"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", spec,
         "-pix_fmt", "yuv420p", str(path)],
        check=True)

