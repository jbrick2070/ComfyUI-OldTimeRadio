# -*- coding: utf-8 -*-
"""GUARD tests for the 2026-08-06 rip-sfx BED cleanbreak.

Sibling to ``test_rip_sfx_broll_guard.py`` (which guards the 2026-07-01 ROLE
rip and is deliberately untouched). This file pins the BED rip: the five
SFX-bed engines are gone, the bed compiler and its helper chain are gone, the
mux is an unconditional ``-c:a copy`` passthrough, and none of the deleted
wire fields can be EMITTED -- attribute checks alone would pass on a module
that stopped exporting a symbol while still writing the field, so emitted rows
are checked too.

THE CLOSED FORBIDDEN SET is the contract: every deleted surface is listed, not
a sample. NARROW EXPLICIT EXEMPTIONS: the 2026-07-01 role tombstones
(``speaker_role="sfx"`` guards), the ``[ENV|SFX|MUSIC:]`` text sanitizers, and
``RETIRED_ENGINE_IDS`` itself -- the retired ids are DATA consulted by the
guard, never a row in CAPABILITIES, never an importable adapter.

Contract: docs/2026-08-06-BUILD-SPEC-rip-sfx.md sections 8 and 10.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import pathlib

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes import otr_master_audio_mux as MUX
from nodes import _otr_story_brief_helpers as BRIEF
from nodes._otr_shared import cloud_media_canonical as CMC
from nodes._otr_shared.public_engines import (
    _LEGACY_ENGINE_ALIASES,
    _PUBLIC_ENGINES,
    RETIRED_ENGINE_IDS,
)
from nodes._otr_video_engines import eng_cloud_video as ECV
from nodes._otr_video_engines import render_driver as RD
from nodes._otr_video_engines import registry as vreg

_REPO = pathlib.Path(__file__).resolve().parent.parent

#: The retired engine ids, restated here INDEPENDENTLY of public_engines so a
#: weakened RETIRED_ENGINE_IDS cannot weaken this guard. Five from the
#: 2026-08-06 SFX-bed rip; five more from the 2026-08-23 dormant-3D retirement
#: (lean-mean order 4). Growing this set is deliberate and append-only, exactly
#: like the production set it mirrors.
_RETIRED_IDS = frozenset({
    # The Ghost narrowing, 2026-08-23 ("delete any animatediff that are not
    # haunted"). Listed here because this guard asserts the WHOLE retired set is
    # unregistered and unaliased -- a retired id that still resolves through the
    # public menu or a legacy alias is a stale selection that silently works.
    "animatediff15_video",
    "animatediff15_v2_video",
    "animatediff15_v3_video",
    "animatediff15_h3_video",
    "animatediff15_h5_video",
    "cloud_vidu_q2_pro_fast_720p_sfx",
    "google_vid_sfx_omni",
    "google_vid_sfx_veo_fast",
    "google_vid_sfx_veo_lite",
    "google_vid_sfx_veo_pro",
    "triposg_talk",
    "hunyuan3d_talk",
    "trellis_talk",
    "triposr",
    "still_parallax",
})

#: Deleted symbols per module -- the CLOSED set. A symbol reappearing on its
#: module is the bed creeping back.
_FORBIDDEN_ATTRS = {
    MUX: (
        "compile_sfx_bed_from_manifest", "_sfx_gain", "DEFAULT_SFX_BED_GAIN",
        "_decoded_pcm_sha", "_clamp01", "_numeric",
    ),
    CMC: (
        "extract_sfx_bed_from_provider_video", "_normalize_sfx_stem_audio",
        "_sfx_loudnorm_params", "SFX_LOUDNESS_REFERENCE_SOURCE",
        "_env_float", "_rms_dbfs", "_provider_video_path",
    ),
    BRIEF: ("append_sfx_audio_safety_clause", "SFX_AUDIO_SAFETY_CLAUSE"),
    RD: ("_GOOGLE_VIDEO_SFX_ENGINES",),
    ECV: ("CloudViduQ2ProFast720pSfxEngine", "ViduQ2ProFast720pSfx"),
}

#: Deleted env keys -- ALL seven, not just the bed gain. Scanned as source
#: text across nodes/, scripts/, tools/ (no tombstone carries them).
_FORBIDDEN_ENV_KEYS = (
    "OTR_SFX_BED_GAIN",
    "OTR_SFX_STEM_TARGET_RMS_DBFS",
    "OTR_SFX_STEM_MAX_BOOST_DB",
    "OTR_SFX_STEM_MAX_CUT_DB",
    "OTR_SFX_STEM_GATE_DBFS",
    "OTR_SFX_STEM_PEAK_CEILING",
    "OTR_GOOGLE_VEO_SFX_SUBMIT_MIN_INTERVAL_S",
)

#: The five deleted wire fields.
_FORBIDDEN_WIRE_FIELDS = (
    "sfx_stem_path", "sfx_duration_s", "sfx_sha256", "sfx_bed_path", "sfx_gain",
)


# ---------------------------------------------------------------------------
# (a) registry: no sfx engine id, no wants_provider_sfx, retired ids barred
# ---------------------------------------------------------------------------

def test_no_registered_engine_id_contains_sfx():
    # BOTH checks are required: the four google_vid_sfx_* engines never set
    # wants_provider_sfx, so the flag check alone would miss them, and a future
    # sfx lane might set the flag on a non-sfx-named id.
    for name in vreg.all_engine_names():
        assert "sfx" not in name.lower(), name
    for name in vreg.CAPABILITIES:
        assert "sfx" not in name.lower(), name


def test_no_registered_engine_declares_wants_provider_sfx():
    for name in vreg.all_engine_names():
        eng = vreg.get_engine(name)
        assert not hasattr(eng, "wants_provider_sfx"), name


def test_the_retired_ids_are_not_registered_or_aliased():
    assert RETIRED_ENGINE_IDS == _RETIRED_IDS
    live = set(vreg.all_engine_names()) | set(vreg.CAPABILITIES)
    assert not (live & _RETIRED_IDS), sorted(live & _RETIRED_IDS)
    # never ALIASED either: no public menu id and no legacy alias may resolve
    # to (or from) a retired id.
    for table in (_PUBLIC_ENGINES, _LEGACY_ENGINE_ALIASES):
        assert not (set(table) & _RETIRED_IDS), table
        assert not (set(table.values()) & _RETIRED_IDS), table


def test_the_sfx_engine_module_is_gone():
    assert importlib.util.find_spec(
        "nodes._otr_video_engines.eng_google_vid_sfx") is None
    assert not (_REPO / "nodes" / "_otr_video_engines"
                / "eng_google_vid_sfx.py").exists()


# ---------------------------------------------------------------------------
# (b) module attributes: the closed forbidden symbol set
# ---------------------------------------------------------------------------

def test_no_deleted_symbol_survives_on_its_module():
    for module, symbols in _FORBIDDEN_ATTRS.items():
        for symbol in symbols:
            assert not hasattr(module, symbol), (module.__name__, symbol)
            assert symbol not in getattr(module, "__all__", ()), (
                module.__name__, symbol)


def test_the_mux_node_class_lost_its_bed_helpers():
    assert not hasattr(MUX.OTRMasterAudioMux, "_default_sfx_bed_out")


# ---------------------------------------------------------------------------
# (c) emitted rows and signatures -- not just attributes
# ---------------------------------------------------------------------------

def test_mux_master_audio_signature_has_no_sfx_parameters():
    params = set(inspect.signature(MUX.mux_master_audio).parameters)
    assert "sfx_bed_path" not in params
    assert "sfx_gain" not in params


def test_build_clip_manifest_emits_no_sfx_fields_even_from_a_poisoned_clip():
    """The producer side is gone, but a stale/foreign clip dict could still
    carry the fields -- the manifest builder must not FORWARD them."""
    result = {
        "ledger": {"video": {
            "video_revision": 1,
            "fps": 25,
            "canonical_canvas": {"w": 1472, "h": 832},
            "shots": [
                {"shot_id": "shot_poisoned", "source_line_ids": ["b001"],
                 "role": "music_visual", "engine_id": "google_veo_video",
                 "family": "text_to_video", "target_frame_count": 50,
                 "start_s": 0.0},
            ]}},
        "clips": {
            "shot_poisoned": {
                "engine_id": "google_veo_video",
                "family": "text_to_video",
                "frame_count": 50,
                "path": __file__,
                "sfx_stem_path": __file__,
                "sfx_duration_s": 2.0,
                "sfx_sha256": "a" * 64,
            },
        },
    }
    manifest = RD.build_clip_manifest(result, episode_id="ep_guard")
    for row in manifest["clips"]:
        for field in _FORBIDDEN_WIRE_FIELDS:
            assert field not in row, (row.get("shot_id"), field)


def test_the_mux_source_carries_no_sfx_mixed_report_line():
    src = inspect.getsource(MUX)
    assert "audio_mode=sfx_mixed" not in src
    assert "audio_mode=master_copy" in src   # the survivor, unconditional


# ---------------------------------------------------------------------------
# (d) env keys: deleted everywhere in the production tree
# ---------------------------------------------------------------------------

def test_no_production_source_reads_a_deleted_sfx_env_key():
    offenders = []
    for top in ("nodes", "scripts", "tools"):
        for path in sorted((_REPO / top).rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for key in _FORBIDDEN_ENV_KEYS:
                if key in text:
                    offenders.append((str(path.relative_to(_REPO)), key))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# (e) the vestigial connector stays a connector (section 6 -- decided)
# ---------------------------------------------------------------------------

def test_clip_manifest_json_stays_wired_but_retired():
    """clip_manifest_json / link 278 STAY WIRED (terminal-node risk asymmetry,
    twice contested, decided). The input must exist, be connector-only, and
    say plainly that it is retired -- never invent a use."""
    spec = MUX.OTRMasterAudioMux.INPUT_TYPES()["optional"]["clip_manifest_json"]
    assert spec[0] == "STRING"
    assert spec[1]["forceInput"] is True
    assert "retired" in spec[1]["tooltip"].lower()
    params = set(inspect.signature(MUX.OTRMasterAudioMux.mux).parameters)
    assert "clip_manifest_json" in params    # accepted, hashed, unused
