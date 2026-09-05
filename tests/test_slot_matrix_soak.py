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
from nodes._otr_shared import slot_matrix as sm
from nodes._otr_shared.capability_profiles import load_profile
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


def test_image_keys_are_exactly_the_three():
    # 2026-07-04 rename: the third image role key is character_image (guards the
    # silent-drop path in build_all_role_profile).
    assert sm.IMAGE_KEYS == ("announcer_image", "music_image", "character_image")


def test_canonical_json_character_granularity_widget():
    # node 88 (OTR_ImageDirector) granularity widget renamed to character_granularity:
    # all three name fields agree, the value stays positional, and the code side matches.
    import inspect
    wf = _load_canonical()
    n88 = next(n for n in wf["nodes"] if n.get("id") == 88)
    assert n88.get("type") == "OTR_ImageDirector"
    inp = next(i for i in n88["inputs"] if i.get("name") == "character_granularity")
    assert inp["localized_name"] == "character_granularity"
    assert inp["widget"]["name"] == "character_granularity"
    assert n88["widgets_values"][2] == "per_object"          # positional value unchanged
    from nodes.otr_image_director import OTRImageDirector
    assert "character_granularity" in OTRImageDirector.INPUT_TYPES()["required"]
    sig = inspect.signature(OTRImageDirector().direct)
    assert "character_granularity" in sig.parameters


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

