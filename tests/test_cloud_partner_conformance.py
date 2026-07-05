"""B5 profile->schema CONFORMANCE guard (owed since the video-team kwargs
warning, 2026-07-02).

Two invariants over ``nodes/_otr_shared/partner_nodes.yaml``:

1. COVERAGE -- every BILLED row (``api_node: true``) is either served by a
   registered adapter (an engine whose ``node_key`` matches) OR carries an
   EXPLICIT xfail entry (nothing silently unadaptered). AUX rows
   (``api_node: false`` -- e.g. the ElevenLabs voice_selector) are excluded:
   they never route through invoke_partner_node and never get an adapter.

2. CONFORMANCE -- every kwarg an adapter EMITS (its ``_partner_inputs``) is
   DECLARED in that row's pinned schema (``required`` U ``optional``). Hidden
   auth inputs are injected by the bridge, never by the adapter, so they are
   NOT in the emitted set. Rows whose adapter is an honest dark row / needs a
   live comfy_api type offline are listed in KNOWN_NONBUILDABLE (the video
   suite pins those separately).

Covers video + image + cloud audio (the ElevenLabs TTS adapter, cloud-audio S1).
"""
import pytest

from nodes._otr_shared.cloud_media_invoke import partner_rows
from nodes._otr_image_engines import registry as ireg
from nodes._otr_video_engines import registry as vreg
from nodes._otr_audio_engines import registry as areg

# Rows with NO adapter yet -- each an EXPLICIT, reasoned xfail (nothing silent).
KNOWN_UNADAPTERED = {
    # cloud_ideogram_v4 is now served by the `ideo` adapter (S1+1); ideo_word
    # (words specialist, same node_key) lands next.
    # cloud_elevenlabs_tts is now served by the `elevenlabs` adapter (cloud-audio
    # S1, 2026-07-03); cloud_sonilo_music by the `sonilo` adapter (cloud-audio S5,
    # 2026-07-03) -- both xfails REMOVED here.
    "cloud_elevenlabs_flash": "same class, no separate adapter (tier via params)",
    "cloud_stability_audio": "documented next music candidate, no adapter (CHEAP-cand)",
}

# Adapters whose _partner_inputs cannot build OFFLINE (honest dark row, or needs
# a live comfy_api input type). Conformance for these is pinned in the
# per-lane suites (test_cloud_video_adapters.py).
KNOWN_NONBUILDABLE = {
    "cloud_seedance_2": "honest dark row until the V3-expansion pin",
    "cloud_kling_lipsync": "needs comfy_api VideoFromFile (offline-unavailable)",
}


def _billed_rows():
    return {k: r for k, r in partner_rows().items() if r.get("api_node")}


def _engines_by_node_key():
    """node_key -> LIST of engines across ALL cloud registries (image + video +
    audio). The audio registry exposes its map as ``_REGISTRY`` (name -> engine),
    the image/video registries via ``all_engine_names()`` + ``get_engine()``.

    A dict-of-LISTS, NOT last-wins: when two adapters legitimately share a
    node_key (e.g. ``ideo`` + a future ``ideo_word`` both on cloud_ideogram_v4,
    or the ElevenLabs tier variants both on cloud_elevenlabs_tts), EVERY adapter
    on that key gets its emitted kwargs verified -- not just whichever registered
    last (the B5 conformance debt, 2026-07-03)."""
    out = {}

    def _add(eng):
        nk = getattr(eng, "node_key", None)
        if nk:
            out.setdefault(nk, []).append(eng)

    for reg in (ireg, vreg):
        for name in reg.all_engine_names():
            _add(reg.get_engine(name))
    for eng in areg._REGISTRY.values():           # cloud-audio adapters (elevenlabs)
        _add(eng)
    return out


def _fixture_request(tmp_path):
    from PIL import Image
    import numpy as np
    import soundfile as sf
    png = tmp_path / "init.png"
    Image.new("RGB", (64, 48), (180, 90, 30)).save(str(png))
    wav = tmp_path / "voice.wav"
    t = np.linspace(0, 1.0, 16000, dtype="float32")
    sf.write(str(wav), (0.2 * np.sin(2 * 3.14159 * 220 * t)), 16000)
    return {
        "shot_id": "shot_x", "prompt": "an old radio on a wooden desk",
        "text_prompt": "an old radio on a wooden desk",
        "init_image": str(png), "audio_ref": str(wav),
        "seed": 7, "seed_bundle": {"request_seed": 7},
        "width": 832, "height": 448,
        "canvas": {"w": 832, "h": 448, "fps": 25},
    }


@pytest.mark.parametrize("node_key", sorted(_billed_rows()))
def test_billed_row_has_adapter_or_explicit_xfail(node_key):
    engines = _engines_by_node_key()
    if node_key in engines:
        return
    if node_key in KNOWN_UNADAPTERED:
        pytest.xfail(f"{node_key}: no adapter yet -- {KNOWN_UNADAPTERED[node_key]}")
    raise AssertionError(
        f"billed row {node_key!r} has NO adapter and NO explicit xfail entry "
        f"-- add an adapter or a KNOWN_UNADAPTERED reason (nothing silent)")


@pytest.mark.parametrize("node_key", sorted(_billed_rows()))
def test_emitted_kwargs_are_declared(node_key, tmp_path, monkeypatch):
    # EVERY buildable adapter sharing this node_key is verified (dict-of-lists),
    # not just whichever registered last (B5 conformance debt fixed 2026-07-05).
    engines = [e for e in _engines_by_node_key().get(node_key, [])
               if hasattr(e, "_partner_inputs")]
    if not engines:
        pytest.skip(f"{node_key}: no buildable adapter (coverage test owns this)")
    # give the V3 video rows the env they demand so they can build offline.
    monkeypatch.setenv("OTR_CLOUD_WAN_MODEL", "wan-2.2-i2v")
    row = partner_rows()[node_key]
    declared = set((row["inputs"].get("required") or {})) \
        | set((row["inputs"].get("optional") or {}))
    for eng in engines:
        try:
            emitted = set(eng._partner_inputs(_fixture_request(tmp_path)))
        except (RuntimeError, ImportError):
            assert node_key in KNOWN_NONBUILDABLE, (
                f"{node_key} ({type(eng).__name__}): _partner_inputs raised "
                f"offline but is not in KNOWN_NONBUILDABLE -- pin it or fix it")
            continue
        undeclared = emitted - declared
        assert not undeclared, (
            f"{node_key} ({type(eng).__name__}): adapter emits kwargs NOT in "
            f"the pinned schema: {sorted(undeclared)} "
            f"(declared: {sorted(declared)})")


def test_v3_model_ids_never_forward_placeholder():
    """The V3 image rows must resolve a real model id, never the placeholder."""
    from nodes._otr_shared.cloud_model_ids import (
        DYNAMICCOMBO_PLACEHOLDER, resolve_model_id)
    for node_key in ("cloud_nano_banana_2", "cloud_seedream_2"):
        rid = resolve_model_id(node_key)
        assert rid and rid != DYNAMICCOMBO_PLACEHOLDER
