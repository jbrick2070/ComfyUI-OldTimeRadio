"""BUG 3 (2026-06-20): the forensic model/engine/system detail the treatment
records must ALSO scroll on the on-screen rolling credits (the production
dossier). Tests the data builder + that the HUD renders it without clipping."""
from nodes import video_engine as ve


def _led():
    return {
        "meta": {
            "gen_params_initial": {
                "creative_writing_model": "openrouter:~anthropic/claude-opus-latest",
                "technical_model": "mistralai/Mistral-Nemo-Instruct-2407",
                "creativity": "balanced", "temperature": 0.8, "top_p": 0.95,
                "target_words": 320,
            },
            "slot_transitions": 3,
            "total_word_count": 305, "character_word_count": 210,
            "announcer_word_count": 95,
            "news": {"script_brief": "UCLA lab integrity vs commercialization",
                     "key_terms": ["UCLA", "integrity", "startup"]},
            "dramatic_state": {"dramatic_question": "Will Hayes expose the data?",
                               "character_a_wants": "protect the lab",
                               "character_b_wants": "ship the product",
                               "ending_change": "Hayes goes public"},
            "render_engines": {"by_role": {"character_video": {"still_flat": 3},
                                           "announcer_visual": {"still_flat": 2}},
                               "histogram": {"still_flat": 6}},
        },
        "cast": [{"name": "HAYES VANCE", "char_id": "c01",
                  "voice_preset": "v2/en_speaker_3", "voice_engine": "bark"}],
        "images": {"images": [
            {"role": "character_video", "engine_id": "z_image_turbo"},
            {"role": "character_video", "engine_id": "z_image_turbo"},
            {"role": "announcer_visual", "engine_id": "flux_gen1"}]},
        "lines": [{"line_id": "b002", "speaker_role": "character", "char_id": "c01",
                   "text": "We can't ship this."}],
    }


def _headers(dossier):
    return [s["header"] for s in dossier]


def test_dossier_has_writer_llm_block():
    dossier = ve._build_hud_dossier(_led())
    assert "WRITER / LLM CONFIG" in _headers(dossier)
    body = "\n".join(
        l for s in dossier if s["header"] == "WRITER / LLM CONFIG" for l in s["lines"])
    assert "claude-opus-latest" in body          # creative slot model on-screen
    assert "Mistral-Nemo" in body                # technical slot model on-screen
    assert "320" in body and "305" in body       # target vs actual words


def test_dossier_has_story_spine_block():
    dossier = ve._build_hud_dossier(_led())
    body = "\n".join(
        l for s in dossier if s["header"] == "STORY SPINE" for l in s["lines"])
    assert "UCLA lab integrity" in body
    assert "protect the lab" in body and "ship the product" in body


# RENDER ENGINES dossier tests RETIRED (credits enrichment 2026-07-03). The
# per-role image/video engine receipts were ripped out of node-12's
# _build_hud_dossier: they do not exist at node-12 time (it renders BEFORE the
# image dispatcher 91 / video render batch 92, so the block only ever printed
# stale/"(not recorded)" data). The receipts now render LATE from the DURABLE
# ledger in OTR_CreditsRoll; the 3 relocated tests are the node's spec now:
#   tests/test_credits_roll_spec.py::test_roll_has_motion_and_images_blocks_video_and_image
#   tests/test_credits_roll_spec.py::test_roll_image_engines_from_meta_is_the_only_source
#   (meta-primary + meta-over-legacy folded into the single "only source" test)
# node-12's dossier keeps only what is TRUE at its early time (writer / story
# spine / system / resolved-openrouter), covered by the tests below.


def test_dossier_has_system_block_techie_stats():
    # SYSTEM always present (techie-friendly); fields degrade to (unknown) but
    # the labels are always there.
    dossier = ve._build_hud_dossier(_led())
    assert "SYSTEM" in _headers(dossier)
    body = "\n".join(
        l for s in dossier if s["header"] == "SYSTEM" for l in s["lines"])
    for label in ("Host:", "CPU:", "RAM:", "GPU:", "CUDA:", "torch"):
        assert label in body


def test_resolved_openrouter_block_when_snapshot_present(monkeypatch):
    import nodes._otr_openrouter_backend as bk
    monkeypatch.setattr(
        bk, "resolved_models_snapshot",
        lambda: {"openrouter:~anthropic/claude-opus-latest":
                 {"resolved": "anthropic/claude-opus-4-1", "calls": 4,
                  "cost_usd": 0.0123}}, raising=False)
    dossier = ve._build_hud_dossier(_led())
    assert "RESOLVED (OPENROUTER)" in _headers(dossier)
    body = "\n".join(
        l for s in dossier if s["header"] == "RESOLVED (OPENROUTER)" for l in s["lines"])
    assert "claude-opus-4-1" in body and "$0.0123" in body


def test_parse_hud_data_carries_dossier_and_renders():
    data = ve._parse_hud_data("Glimpse Through Glass", _led(), {}, "sci-fi", "",
                              '["UCLA lab integrity"]', 90.0, 1472, 832)
    assert data.get("dossier"), "dossier missing from HUD data"
    r = ve._TelemetryHUDRenderer(1472, 832, 25, data)
    n = r.hud_frames()
    img = r.render(int(n * 0.5), n)             # a mid frame draws dossier+scroll
    assert img.size == (1472, 832)
    # the pre-rendered scrollable canvas is tall enough to hold dossier+transcript
    assert r._right_h > 832


def test_dossier_never_raises_on_empty_ledger():
    assert isinstance(ve._build_hud_dossier({}), list)
    assert isinstance(ve._build_hud_dossier(None), list)
