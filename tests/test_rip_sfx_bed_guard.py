# -*- coding: utf-8 -*-
"""The vestigial `clip_manifest_json` connector left over from the 2026-08-06
rip-sfx BED cleanbreak.

The SFX bed itself is gone and the mux is an unconditional ``-c:a copy``
passthrough. The one thing that stays is the connector the shipped workflows
still wire into the mux node (link 278): the input must exist, be
connector-only, and say plainly that it is retired.

Contract: docs/2026-08-06-BUILD-SPEC-rip-sfx.md section 6.
"""
from __future__ import annotations

import inspect

from nodes import otr_master_audio_mux as MUX


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
