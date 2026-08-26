"""`ideogram4_local` -- the focused tests the preflight matrix does NOT cover.

The image preflight matrix sweeps DECLARATIONS and menu behaviour, so this
engine is picked up there automatically. What it cannot cover is the part of
this adapter that is failure-sensitive: the caption transform, the graph
topology, and the refusal classifier. Every threshold and literal asserted here
was DERIVED from captured artifacts during the 2026-08-21/22 campaign, never
guessed -- see `docs/2026-08-21-ideogram4-verdict.md`.

Pure CPU: no ComfyUI runtime, no GPU, no weights, no network.
"""

from __future__ import annotations

import ast
import json

import numpy as np
import pytest

from nodes import _otr_image_engines  # noqa: F401 -- self-registration
from nodes._otr_image_engines import registry as ireg
from nodes._otr_image_engines import ideogram4_local as ideo
from nodes import _otr_visual_styles as vs
from nodes.otr_meta_brief_image_prompt import compose_still_word_prompt


ALL_PACKS = ("anime", "archival_documentary", "cartoon", "paper_origami",
             "recur_frac", "sci_fi_radio", "shakespeare_stage_realism",
             "storybook_engraving", "video_art")

#: Every prohibition phrase OTR splices in, on either route. On this topology the
#: guider's negative is a ZEROED positive, so each of these is positive
#: conditioning -- "no logos" is what returned painted on a card as "NO MISCOS".
FORBIDDEN_IN_CAPTION = ("only the quoted words", "no other text", "no logos",
                        "no captions", "no lettering", "no on-screen text")


def _engine():
    return ireg.get_engine("ideogram4_local")


def _styled_prose(pack, role, text):
    """Exactly what the dispatcher hands the engine: composed prose with the
    pack's style cue FRONT-ANCHORED (`prefix_style_cue`, applied before the
    engine call). Route detection must survive that prefix."""
    style = vs.resolve_visual_style(pack)
    meta = {"episode_id": "ep_t", "episode_title": "The Phonograph's Secret",
            "visual_style": pack,
            "story_brief_terms": {"setting": ["a dock"], "atmosphere": ["tense"]}}
    line = {"line_id": "b1", "speaker_role": "character", "text": text}
    return vs.prefix_style_cue(style, compose_still_word_prompt(meta, role, line))


# --------------------------------------------------------------------------- #
# The caption transform
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pack", ALL_PACKS)
def test_word_route_survives_the_style_prefix_on_every_pack(pack):
    """The dispatcher FRONT-ANCHORS the pack's style cue before calling the
    engine. An anchored match would drop every styled episode's word cards into
    the wordless fallback and ship cards with no words on them."""
    text = "I've got our chats recorded, Marshal."
    cap = ideo.build_caption(_styled_prose(pack, "character_video", text), 1472, 832)
    elements = cap["compositional_deconstruction"]["elements"]
    assert len(elements) == 1, pack
    assert elements[0]["text"] == text, pack


@pytest.mark.parametrize("pack", ALL_PACKS)
def test_music_cards_carry_no_text_element_and_keep_their_subject(pack):
    """The music card is contractually WORDLESS, so it gets `elements: []` --
    and the evoked title must still reach `high_level_description`, or the model
    renders an abstract image of nothing in particular."""
    cap = ideo.build_caption(_styled_prose(pack, "music_visual", ""), 1472, 832)
    assert cap["compositional_deconstruction"]["elements"] == [], pack
    assert "Phonograph" in cap["high_level_description"], pack


@pytest.mark.parametrize("pack", ALL_PACKS)
@pytest.mark.parametrize("role", ("character_video", "music_visual"))
def test_no_prohibition_reaches_the_model_on_either_route(pack, role):
    """Bible 12.126. The composer appends `only the quoted words, no other text,
    no logos, no captions` to word cards and every pack appends `no lettering`
    to music cards. With no negative channel those ASK for the thing they
    forbid, so none may survive into the caption."""
    prose = _styled_prose(pack, role, "I've got our chats recorded, Marshal.")
    blob = json.dumps(ideo.build_caption(prose, 1472, 832), ensure_ascii=False)
    leaked = [p for p in FORBIDDEN_IN_CAPTION if p in blob.lower()]
    assert leaked == [], (pack, role, leaked)


def test_a_spoken_line_containing_a_guard_phrase_is_never_edited():
    """THE ORDERING GUARANTEE. The card text is extracted and removed BEFORE the
    scrub, so a line that legitimately contains a guard phrase survives intact.
    Script is never edited -- a scrub-first implementation would silently rewrite
    the words on screen."""
    hostile = "No captions, no excuses. We go live at nine."
    cap = ideo.build_caption(
        _styled_prose("sci_fi_radio", "character_video", hostile), 1472, 832)
    element = cap["compositional_deconstruction"]["elements"][0]
    assert element["text"] == hostile
    # ...and the atmosphere around it is still scrubbed.
    background = cap["compositional_deconstruction"]["background"].lower()
    assert "no logos" not in background


def test_caption_is_exactly_three_top_level_keys_in_the_schema_order():
    """The vendor's caption contract, read out of the template's own magic-prompt
    subgraph. A FOREIGN key is not ignored -- it is rendered onto the card."""
    cap = ideo.build_caption(
        _styled_prose("anime", "character_video", "Hold the line."), 1472, 832)
    assert list(cap) == ["aspect_ratio", "high_level_description",
                         "compositional_deconstruction"]


def test_caption_json_is_single_line_minified():
    """Minified means the JSON SEPARATORS are tight. It cannot mean `", "` is
    absent from the string -- prose values legitimately contain commas followed
    by spaces, so asserting that would test the sentence, not the format."""
    blob = ideo.caption_json(
        _styled_prose("anime", "character_video", "Hold the line."), 1472, 832)
    assert "\n" not in blob
    assert blob == json.dumps(json.loads(blob), ensure_ascii=False,
                              separators=(",", ":"))


def test_unrecognised_prose_is_wrapped_never_passed_through_raw():
    """Raw prose is what this model REFUSED 6 of 6 times. A portrait or scene
    still reaching this engine must still arrive as schema."""
    cap = ideo.build_caption("a weathered spacer in a cramped orbital corridor",
                             1472, 832)
    assert list(cap)[0] == "aspect_ratio"
    assert cap["compositional_deconstruction"]["elements"] == []


# --------------------------------------------------------------------------- #
# Canvas + aspect
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("w,h,expected", [
    (1472, 832, "16:9"),      # the production still canvas -- exact reduction
                              # would give 23:13, which the schema never saw
    (1920, 1088, "16:9"),
    (1024, 1024, "1:1"),
    (832, 1216, "2:3"),
    (0, 0, "16:9"),           # missing dimensions must not crash
])
def test_aspect_ratio_buckets_to_a_standard_never_raw_pixels(w, h, expected):
    assert ideo.canonical_aspect(w, h) == expected


@pytest.mark.parametrize("raw,snapped", [(1472, 1472), (832, 832), (1080, 1088),
                                         (100, 256), (0, 256)])
def test_canvas_snaps_to_16_with_a_256_floor(raw, snapped):
    assert ideo._snap(raw) == snapped


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #
def _graph():
    engine = _engine()
    params = engine._params({"prompt": 'a title card displaying the words "Hi."',
                             "seed": 42, "w": 1472, "h": 832,
                             "object_id": "still_b1"})
    return engine, params, engine._build_graph(params, lambda n, s: [n, s])


def test_every_wire_points_at_a_declared_node_and_every_node_has_a_candidate():
    """A graph that references a node it never declares dies at run_graph with a
    missing wire source; a node with no candidate class dies as unresolved."""
    engine, _params, graph = _graph()
    candidates = engine._node_candidates()
    for node, spec in graph.items():
        assert spec["class"] in candidates, node
        for key, value in spec["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph, f"{node}.{key} -> {value[0]}"


def test_the_loaders_and_the_latent_edge_are_present():
    """The three that a prose description of this topology keeps dropping."""
    _engine_, _params, graph = _graph()
    assert {"clip", "vae", "latent"} <= set(graph)
    assert graph["sample"]["inputs"]["latent_image"] == ["latent", 0]
    assert graph["pos"]["inputs"]["clip"] == ["clip", 0]
    assert graph["decode"]["inputs"]["vae"] == ["vae", 0]


def test_the_negative_branch_is_the_zeroed_positive():
    """This is WHY no prohibition text can act here. If this ever becomes a real
    text negative, the guard-stripping above is no longer required -- and this
    test failing is the signal to revisit it."""
    _engine_, _params, graph = _graph()
    assert graph["guider"]["inputs"]["negative"] == ["zero", 0]
    assert graph["zero"]["inputs"]["conditioning"] == ["pos", 0]
    assert graph["guider"]["inputs"]["model"] == ["cfg_override", 0]
    assert graph["guider"]["inputs"]["model_negative"] == ["unet_uncond", 0]


def test_the_recipe_literals_are_pinned():
    """Pinned so a drive-by tweak is a test failure rather than a silent
    re-render of every card. mu 0.5 is a RECORDED deviation from the vendor
    Default preset (0.0) -- see the module docstring."""
    _engine_, params, graph = _graph()
    assert (params["steps"], params["mu"], params["std"]) == (20, 0.5, 1.75)
    assert graph["guider"]["inputs"]["cfg"] == 7.0
    assert graph["cfg_override"]["inputs"]["cfg"] == 3.0
    assert graph["cfg_override"]["inputs"]["start_percent"] == 0.7
    assert graph["sampler"]["inputs"]["sampler_name"] == "euler"


def test_the_text_bbox_is_the_measured_moderate_box():
    """8 measured cells: a two-thirds-height box drew invented page furniture on
    4 of 4 designed-artifact cards; no box at all broke the spelling. This is the
    only setting measured clean on both axes."""
    assert ideo.TEXT_BBOX == [200, 60, 700, 940]
    cap = ideo.build_caption(
        _styled_prose("anime", "character_video", "Hold the line."), 1472, 832)
    assert cap["compositional_deconstruction"]["elements"][0]["bbox"] == [200, 60, 700, 940]


# --------------------------------------------------------------------------- #
# The refusal classifier
# --------------------------------------------------------------------------- #
def test_a_captured_refusal_card_is_classified_as_a_refusal():
    """Refusals measured min 68-87 / std 9.9-10.7: a flat pale placeholder at the
    exact requested dimensions, delivered with host status SUCCESS."""
    rng = np.random.default_rng(0)
    frame = (108 + rng.normal(0, 10, (832, 1472, 3))).clip(70, 150).astype(np.uint8)
    refused, minimum, std = ideo.classify_refusal(frame)
    assert refused and minimum > 50 and std < 15


def test_a_real_card_is_not_classified_as_a_refusal():
    """Real cards measured min 0-1 / std 27-41 -- pale lettering on a near-black
    ground, so the floor sits at zero and the spread is wide."""
    frame = np.zeros((832, 1472, 3), dtype=np.uint8)
    frame[300:500, 100:1300] = 230          # the lettering
    refused, minimum, std = ideo.classify_refusal(frame)
    assert not refused and minimum == 0 and std > 15


def test_the_all_black_guard_would_MISS_a_refusal():
    """Bible 12.125's third verify condition: the new detector must be shown NOT
    to be redundant with the generic empty-output guard."""
    rng = np.random.default_rng(1)
    frame = (108 + rng.normal(0, 10, (64, 64, 3))).clip(70, 150).astype(np.uint8)
    assert frame.any(), "a refusal card is not black -- that is the whole problem"
    assert ideo.classify_refusal(frame)[0]


def test_the_refusal_error_is_local_not_the_dispatchers():
    """Importing `ImageRenderError` from the dispatcher would be a cycle (the
    dispatcher imports this package at module scope) and, because this adapter's
    own import is GUARDED, the cycle would be swallowed and the engine would
    silently fail to register. The dispatcher wraps adapter exceptions anyway."""
    assert issubclass(ideo.Ideogram4RefusalError, RuntimeError)
    source = (ideo.__file__ or "")
    assert source, "engine module must have a file"
    with open(source, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    # An AST walk, not a substring search: the module DOCUMENTS why it does not
    # import the dispatcher, and a naive grep fails on its own explanation.
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)
    assert not any("otr_image_gen_dispatcher" in n for n in imported), imported


# --------------------------------------------------------------------------- #
# Declarations that the matrix cannot see
# --------------------------------------------------------------------------- #
def test_engine_version_moves_when_a_weight_override_moves(monkeypatch):
    """Model overrides change the rendered pixels but touch no other term of the
    still cache key, so without this a quant swap serves yesterday's images."""
    engine = _engine()
    before = engine.engine_version
    monkeypatch.setenv(ideo.COND_UNET_ENV, "ideogram4_some_other_quant.safetensors")
    assert engine.engine_version != before


def test_it_is_a_local_engine_and_takes_the_gpu_lease():
    """`native` / `node_key` must stay ABSENT: either one routes this to the
    cloud no-lease path, and 11 GB of experts would load without the lease."""
    engine = _engine()
    assert not hasattr(engine, "node_key")
    assert getattr(engine, "native", True) is True


def test_it_is_opt_in_and_does_not_displace_the_shipped_default():
    engine = _engine()
    assert engine.default_roles == ()
    assert ireg.get_engine("z_image_turbo").name == "z_image_turbo"


def test_an_empty_atmosphere_leaves_no_dangling_separator():
    """The degenerate case: prose that is ONLY the card clause. The composed
    sentence must not open with "; " or close with "," -- every token here is
    positive conditioning, so stray punctuation is noise the encoder reads.

    Production cannot reach this (the composer always splices style, era and
    grade tails after the card clause), so this is defence in depth for a direct
    caller and for any future composer that trims its tails.
    """
    cap = ideo.build_caption('a title card displaying the words "Hi."', 1472, 832)
    hld = cap["high_level_description"]
    background = cap["compositional_deconstruction"]["background"]
    assert not hld.rstrip().endswith((",", ";", ":")), hld
    assert not background.lstrip().startswith((",", ";", ":")), background
    # ...and the card itself is untouched by the tidying.
    assert cap["compositional_deconstruction"]["elements"][0]["text"] == "Hi."


def test_the_artifact_list_is_resolved_once_per_params_build(monkeypatch):
    """`_params` used to call `resolve_all_artifacts()` once per NAME, and each
    call re-probes all four artifacts through folder_paths -- 16 filesystem
    probes to read four basenames."""
    calls = []
    real = ideo.resolve_all_artifacts
    monkeypatch.setattr(ideo, "resolve_all_artifacts",
                        lambda: (calls.append(1), real())[1])
    _engine()._params({"prompt": 'a title card displaying the words "Hi."',
                       "seed": 1, "w": 1472, "h": 832, "object_id": "o"})
    assert len(calls) == 1, f"resolved {len(calls)} times"


# --------------------------------------------------------------------------- #
# PBUG: the punctuation tidier was injecting a CONTROL CHARACTER
# --------------------------------------------------------------------------- #

def test_tidy_preserves_punctuation_instead_of_injecting_a_control_character():
    """THE BUG THIS PINS, found by an r2 review pass on 2026-08-26.

    `_tidy`'s two punctuation rules capture a group and were meant to put it
    back with the `\1` backreference. What was actually stored in both
    replacement literals was a bare **U+0001**, so every match DELETED the
    captured punctuation and injected a C0 control character in its place.

    Measured before the fix::

        "a warm revelation., sepia tones" -> "a warm revelation\x01 sepia tones"

    That mattered because the refused prompt from the 2026-08-26 sweep ends in
    exactly that shape (`...revelation., sepia tones`), so the JSON caption
    handed to Ideogram carried a control character inside it. A control
    character in a structured caption is a plausible refusal driver on its own,
    and it reached the model on EVERY route, not just the scene fallthrough.

    Asserted as a PROPERTY -- no C0 control characters survive `_tidy` -- rather
    than as one golden string, so the guard cannot be satisfied by fixing only
    the one input that was noticed.
    """
    cases = [
        "a warm revelation., sepia tones",
        "anime style. , huge high-contrast",
        "a dusty archive room ,a glowing radio",
        "one clause; , another",
        "a label: , a value",
    ]
    for raw in cases:
        out = ideo._tidy(raw)
        controls = sorted({c for c in out if ord(c) < 32})
        assert not controls, (
            "_tidy(%r) -> %r injected control character(s) %r -- the "
            "replacement literal is not the \1 backreference"
            % (raw, out, controls))

    # and the punctuation is genuinely KEPT, not merely non-control
    assert ideo._tidy("a warm revelation., sepia tones") ==         "a warm revelation. sepia tones"
    assert ideo._tidy("anime style. , huge high-contrast") ==         "anime style. huge high-contrast"


def test_tidy_does_not_weld_two_words_together_at_the_style_seam():
    """THE SECOND HALF, caught in review after the control-character fix.

    Restoring the backreference alone gives ``r"\\1"``, which consumes the
    whitespace the comma was carrying: "anime style. ,huge contrast" closes up
    to "anime style.huge contrast" -- two words welded into one token. So the
    rule eats the trailing whitespace deliberately and re-emits a single space.

    "cartoon style. , a man" is the REAL seam this function exists for: it is
    what the style prefix produces once the card clause is cut out of the
    middle of the comma-joined string. Pinned by name rather than left to a
    synthetic case.
    """
    assert ideo._tidy("anime style. ,huge contrast") == "anime style. huge contrast"
    assert ideo._tidy("cartoon style. , a man") == "cartoon style. a man"
    for raw in ("anime style. ,huge", "cartoon style. , a man",
                "a warm revelation., sepia", "one clause; , another",
                "a label: , a value"):
        assert "  " not in ideo._tidy(raw), (
            "_tidy(%r) -> %r left doubled whitespace" % (raw, ideo._tidy(raw)))


def test_no_control_characters_reach_the_caption_on_any_route():
    """The property that actually matters: whatever route runs, the JSON the
    model receives is free of C0 controls. `_tidy` feeds all three routes, so a
    regression there is invisible until a model declines the card."""
    prose = ("a title card displaying the words \"HOLD THE LINE\", a dusty "
             "archive room ,warm revelation., sepia tones")
    for probe in (prose,
                  'an abstract picture evoking "Signal Lost", a room ,dusk.',
                  "a vintage tube radio glowing warmly, worn dials ,dusk."):
        blob = ideo.caption_json(probe, 1920, 1080)
        controls = sorted({c for c in blob if ord(c) < 32})
        assert not controls, (
            "caption_json(%r) carries control character(s) %r" % (probe, controls))
