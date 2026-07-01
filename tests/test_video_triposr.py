"""triposr -- the lower-tier MIT 3D mesher (UNREGISTERED dark scaffold, C3).

UNREGISTERED 2026-06-29 (C3 -- "registry IS the menu"): triposr renders
NotImplementedError, so it is no longer registered/selectable and carries no
CAPABILITIES row. The SOURCE class stays on disk (it returns when a real forward
ships) and still has its identity + still fails LOUD (RuntimeError on load,
NotImplementedError on render) when instantiated directly.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import schemas
from nodes._otr_video_engines.eng_triposr import TripoSREngine


def test_triposr_unregistered_not_selectable():
    # C3: dropped from the registry + the full dropdown; no CAPABILITIES row.
    assert not vreg.is_registered("triposr")
    assert "triposr" not in vreg.all_engine_names()
    assert "triposr" not in vreg.CAPABILITIES


def test_triposr_source_identity_intact():
    # The source class is preserved (returns when built): a STATIC mesher
    # (turntable motion only, NEVER lip-sync), MIT, degrades to still_motion
    # (still_parallax retired 2026-06-30, item 2 rip-out).
    eng = TripoSREngine()
    assert eng.name == "triposr"
    assert eng.family == "image_to_video" and eng.family in schemas.FAMILIES
    assert "audio_ref" not in eng.required_inputs   # no lip-sync
    assert eng.commercial_clean is True             # MIT
    assert eng.default_roles == ()
    assert eng.fallback_engine == "still_motion"


def test_triposr_source_load_and_render_are_dark():
    eng = TripoSREngine()
    with pytest.raises(RuntimeError):
        eng.load()
    with pytest.raises(NotImplementedError):
        eng.render_clip(request=None, prepared=None)
