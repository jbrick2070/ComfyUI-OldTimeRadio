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
from nodes._otr_shared import still_plan_helpers as sp
from nodes._otr_video_engines import coverage_plan as cp


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


def _scene_prose(pack="sci_fi_radio"):
    """A composed SCENE as the three non-`still_word` lanes hand it over: a
    comma-joined description behind the same front-anchored style cue. It
    matches NEITHER composer anchor -- which is exactly why routing on prose
    text alone left `still_flat` / `still_pan` / `still_motion` unrouted."""
    style = vs.resolve_visual_style(pack)
    return vs.prefix_style_cue(
        style,
        "a weathered spacer at a cramped orbital console, worn dials glowing, "
        "one hand raised, a low tense hum, amber key light")


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
    """The music card is contractually WORDLESS -- and the evoked title must
    still reach `high_level_description`, or the model renders an abstract image
    of nothing in particular.

    THE ASSERTION MOVED, THE CONTRACT DID NOT (2026-08-26). This used to say
    `elements == []`, which is STRICTER than the rule the test's own name
    states: the card must carry no TEXT element. Emptiness was an incidental
    way of expressing that, and it stopped being true when the music card
    gained one `type: "obj"` anchor.

    WHY IT GAINED ONE. Measured on a live leg: this engine rendered 6 of 8
    stills in one episode -- every card carrying words -- and refused BOTH
    wordless music bookends on different seeds (min 78/80, std 10.5, against a
    real card's min~0 std 27-41), on captions verified free of control
    characters, face language, prohibitions and duplicated fields. The only
    structural difference left was `elements: []`. A display-typography model
    given nothing to anchor on is being asked for the one thing it is not for.
    The operator's ruling is that ideogram stays selectable for the music role
    and must PRODUCE AN OUTPUT, so the card gets a concrete OBJECT instead of
    lettering it is contractually forbidden.

    So the assertion is now the NAME's rule, stated directly and made stronger
    in the dimension that matters: no `text` element may EVER appear here, and
    the object anchor is pinned so it cannot silently become one.
    """
    cap = ideo.build_caption(_styled_prose(pack, "music_visual", ""), 1472, 832)
    els = cap["compositional_deconstruction"]["elements"]
    assert all(e["type"] != "text" for e in els), (pack, els)
    assert [e["type"] for e in els] == ["obj"], (pack, els)
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


# --------------------------------------------------------------------------- #
# THE LENS: the caption transform routes on the request's METADATA
#
# `ideogram4_local` is the only image engine that TRANSFORMS its prompt, so it
# is the only one that owes a translation layer. Until 2026-08-26 that layer
# routed by searching the prose for two literal anchors, and both anchors are
# minted by ONE composer entry (`compose_still_word_prompt`) serving ONE lane.
# Every other still lane matched neither and fell into the unrouted fallthrough
# -- the branch whose own docstring conceded that raw prose is what this model
# refuses. The dispatcher was already handing over `kind` and `role`
# (otr_image_gen_dispatcher, the request dict); the adapter never read them.
# --------------------------------------------------------------------------- #

#: Every live still kind and the route it must take. Written out one by one
#: rather than derived, so a kind silently changing route is a diff a reader
#: sees rather than a formula that quietly still passes.
KIND_ROUTE_EXPECTATIONS = (
    (sp.KIND_PORTRAIT, ideo.ROUTE_PORTRAIT),
    (sp.KIND_SCENE_OPEN, ideo.ROUTE_SCENE),
    (sp.KIND_SCENE_BEAT, ideo.ROUTE_SCENE),
    (sp.KIND_SCENE_CHARACTER, ideo.ROUTE_SCENE),
    (sp.KIND_MESH_FODDER, ideo.ROUTE_SCENE),
    (sp.KIND_SCENE_BACKGROUND_PLATE, ideo.ROUTE_SCENE),
    (cp.JUMP_STILL_KIND, ideo.ROUTE_SCENE),
)


def test_the_kind_table_covers_the_live_vocabulary():
    """The routing table repeats its literals instead of importing them, because
    the engine module is cold-import clean (V-12) and the video package
    self-registers on import. This test pays that import cost HERE instead, and
    is the thing that fails when the two definitions drift.

    Six kinds are the closed enum in `still_plan_helpers`; the seventh is
    `coverage_plan.JUMP_STILL_KIND`, deliberately not a `scene_*` token so
    segment stills stay invisible to the beat-indexed consumers."""
    live = set(sp.VALID_KINDS) | {cp.JUMP_STILL_KIND}
    assert set(ideo.OBJECT_KIND_ROUTES) == live, (
        "table %r vs live vocabulary %r"
        % (sorted(ideo.OBJECT_KIND_ROUTES), sorted(live)))
    # ...and the expectations above are exhaustive over that same vocabulary.
    assert {k for k, _route in KIND_ROUTE_EXPECTATIONS} == live


@pytest.mark.parametrize("kind,expected", KIND_ROUTE_EXPECTATIONS)
def test_every_live_still_kind_takes_an_explicitly_mapped_route(kind, expected):
    """One route assertion per object kind. Scene prose carries neither composer
    anchor, so this is the decision `kind` alone is making."""
    assert ideo.caption_route(_scene_prose(), kind=kind) == expected


def test_a_portrait_request_reaches_the_portrait_route_at_all():
    """It could not before: with no metadata read, a portrait was
    indistinguishable from a scene and both landed in the same fallthrough.
    `ideogram4_local` returns a safety placeholder for a person close-up (it
    killed a live leg on 2026-08-22), so knowing a request IS a portrait is the
    prerequisite for ever treating one differently."""
    cap = ideo.build_caption(_scene_prose(), 1472, 832,
                             kind=sp.KIND_PORTRAIT, role="character_video")
    assert ideo.caption_route(_scene_prose(), kind=sp.KIND_PORTRAIT) == \
        ideo.ROUTE_PORTRAIT
    assert list(cap) == ["aspect_ratio", "high_level_description",
                         "compositional_deconstruction"]


def test_an_unmapped_kind_fails_loudly_and_names_the_lane():
    """NO FALLBACK. A kind nobody routed must not slide into the scene caption --
    that silent misroute is the whole defect being repaired, and letting the
    next new kind repeat it would be choosing the same bug twice. The message
    carries the role because that is how an operator finds which lane minted
    the row."""
    with pytest.raises(ValueError) as excinfo:
        ideo.build_caption(_scene_prose(), 1472, 832,
                           kind="scene_montage", role="music_visual")
    message = str(excinfo.value)
    assert "scene_montage" in message
    assert "music_visual" in message
    # the known vocabulary is listed, so the reader is told what IS mapped
    assert sp.KIND_SCENE_BEAT in message


def test_an_unmapped_kind_fails_on_the_word_lane_too():
    """The kind is resolved BEFORE the prose anchors are consulted, on purpose.
    A hole in the table is a hole whether or not this particular prose happened
    to carry a composer anchor, and a hole that only surfaces on some lanes is
    the kind that ships."""
    word = _styled_prose("anime", "character_video", "Hold the line.")
    with pytest.raises(ValueError):
        ideo.build_caption(word, 1472, 832, kind="scene_montage")


def test_an_absent_kind_is_the_scene_route_and_never_an_error():
    """An ABSENT kind is not an unknown kind. `build_caption(prose, w, h)` stays
    a valid positional call -- the refusal repro script and most of this file
    call it that way -- and a caller with no metadata gets the route it already
    had."""
    prose = _scene_prose()
    assert ideo.caption_route(prose) == ideo.ROUTE_SCENE
    assert ideo.caption_route(prose, kind="", role="") == ideo.ROUTE_SCENE
    cap = ideo.build_caption(prose, 1472, 832)          # positional, no metadata
    assert cap["compositional_deconstruction"]["elements"] == []


def test_the_word_anchor_outranks_the_kind_word_cards_actually_wear():
    """THE REASON THE PROSE ANCHORS KEEP PRIORITY. A word card's ledger row wears
    the shared cheap-family `scene_character` kind -- it inherits face framing
    while actually minting typography from the spoken line. Routing on kind
    first would send every word card to the scene caption and ship cards with no
    words on them."""
    text = "I've got our chats recorded, Marshal."
    prose = _styled_prose("sci_fi_radio", "character_video", text)
    assert ideo.caption_route(prose, kind=sp.KIND_SCENE_CHARACTER,
                              role="character_video") == ideo.ROUTE_WORD
    cap = ideo.build_caption(prose, 1472, 832, kind=sp.KIND_SCENE_CHARACTER,
                             role="character_video")
    elements = cap["compositional_deconstruction"]["elements"]
    assert len(elements) == 1 and elements[0]["text"] == text


def test_the_title_anchor_outranks_the_object_kind_as_well():
    """Same rule on the music card: it is contractually WORDLESS and its subject
    is the episode title, whatever kind its row happens to carry."""
    prose = _styled_prose("anime", "music_visual", "")
    assert ideo.caption_route(prose, kind=sp.KIND_SCENE_BEAT,
                              role="music_visual") == ideo.ROUTE_TITLE
    cap = ideo.build_caption(prose, 1472, 832, kind=sp.KIND_SCENE_BEAT,
                             role="music_visual")
    assert "Phonograph" in cap["high_level_description"]
    # No TEXT element, ever -- see the music-card test above for why this is no
    # longer spelled `== []`.
    els = cap["compositional_deconstruction"]["elements"]
    assert all(e["type"] != "text" for e in els), els


@pytest.mark.parametrize("kind", (sp.KIND_PORTRAIT, sp.KIND_SCENE_BEAT,
                                  sp.KIND_SCENE_BACKGROUND_PLATE,
                                  cp.JUMP_STILL_KIND))
def test_the_wrapped_routes_stop_pasting_the_input_into_two_fields(kind):
    """BUG 2. The fallthrough used to put the IDENTICAL string into
    `high_level_description` AND `compositional_deconstruction.background` --
    the input pasted twice, which is not a deconstruction and merely told the
    model the same thing over again.

    The fix is an EMPTY background, not a richer guess: the composer emits a
    comma-joined five-layer string behind a style prefix, which is a convention
    and not a grammar, so re-extracting a setting from it would mis-fire and
    invent one. Empty invents nothing."""
    cap = ideo.build_caption(_scene_prose(), 1472, 832, kind=kind)
    deconstruction = cap["compositional_deconstruction"]
    description = cap["high_level_description"]
    assert description, "the scrubbed prose must still reach the model"
    assert deconstruction["background"] == ""
    assert deconstruction["background"] != description, kind
    assert deconstruction["elements"] == []
    # the vendor shape stays minimal -- a FOREIGN key is not ignored by this
    # model, it is rendered onto the card
    assert set(deconstruction) == {"background", "elements"}


def test_params_threads_the_request_kind_and_role_into_the_caption():
    """Changing only `build_caption` would leave the lens dead: `_params` is what
    the render path calls, and it read only `prompt`. An unmapped kind reaching
    the transform through `_params` is the proof the metadata now travels."""
    engine = _engine()
    with pytest.raises(ValueError) as excinfo:
        engine._params({"prompt": _scene_prose(), "seed": 1,
                        "w": 1472, "h": 832, "object_id": "still_b1",
                        "kind": "scene_montage", "role": "music_visual"})
    assert "scene_montage" in str(excinfo.value)
    assert "music_visual" in str(excinfo.value)


@pytest.mark.parametrize("kind,_expected", KIND_ROUTE_EXPECTATIONS)
def test_params_builds_a_caption_for_every_live_kind(kind, _expected):
    """The other half: every kind the dispatcher can actually send must survive
    the round trip and arrive as minified schema, never raw prose."""
    params = _engine()._params({"prompt": _scene_prose(), "seed": 1,
                                "w": 1472, "h": 832, "object_id": "still_b1",
                                "kind": kind, "role": "character_video"})
    caption = json.loads(params["prompt"])
    assert list(caption) == ["aspect_ratio", "high_level_description",
                             "compositional_deconstruction"]
    assert "\n" not in params["prompt"]


def test_the_base_engine_version_moved_so_stale_stills_cannot_be_served():
    """REQUIRED by the routing change, not bookkeeping. The dispatcher's still
    cache key is `(role, object_id, prompt_hash, seed, engine_id,
    engine_version)`, and `prompt_hash` is computed from OTR's PROSE before this
    adapter ever runs -- so a change that alters only the caption this adapter
    builds is invisible to every other term. Without the bump, every still
    minted under the old blind fallthrough would be served from cache forever
    and the fix would never reach a rendered frame."""
    engine = _engine()
    assert engine.base_engine_version == "2"
    assert engine.engine_version.startswith("2.")


def test_the_music_card_anchor_is_an_object_and_never_lettering():
    """THE NEW GUARANTEE, pinned on its own so it cannot erode.

    The music bookend may not carry words (operator 2026-07-04), and this engine
    refuses a card with nothing to anchor on (measured 2026-08-26). Both hold at
    once only if the anchor is an OBJECT. If someone later "helpfully" turns it
    into a text element to stop a refusal, the wordless contract breaks silently
    and the card starts spelling the episode title on screen.

    The bbox order is also pinned. An `obj` bbox is [x, y, w, h] while a `text`
    bbox is [y1, x1, y2, x2]; getting that backwards misplaces the subject
    instead of failing, so it is asserted rather than trusted.
    """
    cap = ideo.build_caption(
        _styled_prose("sci_fi_radio", "music_visual", ""), 1472, 832,
        kind=sp.KIND_SCENE_BEAT, role="music_visual")
    els = cap["compositional_deconstruction"]["elements"]
    assert len(els) == 1 and els[0]["type"] == "obj", els
    assert "text" not in els[0], "an obj element must carry no text field"
    assert els[0]["bbox"] == list(ideo.MUSIC_OBJECT_BBOX)
    assert ideo.MUSIC_OBJECT_BBOX is not ideo.TEXT_BBOX
    # and the whole caption still carries no lettering instruction anywhere
    import json as _json
    blob = _json.dumps(cap, ensure_ascii=False).lower()
    for banned in ("the words", "lettering reading", "spelled exactly"):
        assert banned not in blob, banned


# --- precision ladder (2026-09-03) -----------------------------------------
# The engine used to demand exactly `ideogram4_nvfp4_mixed.safetensors`. nvfp4
# is a Blackwell format, so an AMD / Mac / 3060 box holding the perfectly good
# fp8 or int8 build from the SAME ungated Comfy-Org/Ideogram-4 repo was refused
# with "missing: ideogram4_nvfp4_mix..." -- which reads as "this lane needs a
# Blackwell card" and is false. Only the DEFAULT was Blackwell.

class _FakeFolderPaths:
    """Minimal stand-in for ComfyUI's folder_paths with a chosen shelf."""

    def __init__(self, shelf):
        self._shelf = shelf

    def get_filename_list(self, category):
        return list(self._shelf.get(category, ()))

    def get_full_path(self, category, name):
        return ("/models/%s/%s" % (category, name)
                if name in self._shelf.get(category, ()) else None)


def _with_shelf(monkeypatch, shelf):
    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", _FakeFolderPaths(shelf))
    for env in ("OTR_IDEOGRAM4_COND_UNET", "OTR_IDEOGRAM4_UNCOND_UNET",
                "OTR_IDEOGRAM4_CLIP", "OTR_IDEOGRAM4_VAE"):
        monkeypatch.delenv(env, raising=False)
    from nodes._otr_image_engines import ideogram4_local as ig
    return ig.resolve_all_artifacts()


_FP8_ONLY = {
    "diffusion_models": ("ideogram4_fp8_scaled.safetensors",
                         "ideogram4_unconditional_fp8_scaled.safetensors"),
    "text_encoders": ("qwen3vl_8b_fp8_scaled.safetensors",),
    "vae": ("flux2-vae.safetensors",),
}
_NVFP4_ONLY = {
    "diffusion_models": ("ideogram4_nvfp4_mixed.safetensors",
                         "ideogram4_unconditional_nvfp4_mixed.safetensors"),
    "text_encoders": ("qwen3vl_8b_nvfp4.safetensors",),
    "vae": ("flux2-vae.safetensors",),
}


def test_an_fp8_only_box_resolves_instead_of_being_refused(monkeypatch):
    """THE REGRESSION. A non-Blackwell machine with the fp8 build installed."""
    got = _with_shelf(monkeypatch, _FP8_ONLY)
    assert all(verified for _n, verified, _c in got), got
    names = [n for n, _v, _c in got]
    assert names[0] == "ideogram4_fp8_scaled.safetensors"
    assert names[1] == "ideogram4_unconditional_fp8_scaled.safetensors"
    assert names[2] == "qwen3vl_8b_fp8_scaled.safetensors"


def test_a_blackwell_box_still_prefers_nvfp4(monkeypatch):
    """The ladder is smallest-first, so nvfp4 wins where it exists -- the
    5.5 GB build is both lighter and faster on the card that supports it."""
    got = _with_shelf(monkeypatch, _NVFP4_ONLY)
    assert all(verified for _n, verified, _c in got), got
    assert [n for n, _v, _c in got][0] == "ideogram4_nvfp4_mixed.safetensors"


def test_nvfp4_wins_when_both_precisions_are_installed(monkeypatch):
    both = {k: tuple(_NVFP4_ONLY.get(k, ())) + tuple(_FP8_ONLY.get(k, ()))
            for k in ("diffusion_models", "text_encoders", "vae")}
    got = _with_shelf(monkeypatch, both)
    assert [n for n, _v, _c in got][0] == "ideogram4_nvfp4_mixed.safetensors"


def test_an_env_override_beats_the_whole_ladder(monkeypatch):
    """The override names a mirror or a quant we have never heard of; second-
    guessing it against a ladder would defeat why it exists."""
    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", _FakeFolderPaths(_NVFP4_ONLY))
    monkeypatch.setenv("OTR_IDEOGRAM4_COND_UNET", "some_mirror_q4.safetensors")
    from nodes._otr_image_engines import ideogram4_local as ig
    assert ig.resolve_all_artifacts()[0][0] == "some_mirror_q4.safetensors"


def test_precision_is_part_of_the_engine_version(monkeypatch):
    """A precision swap must bust the still cache: the resolved basenames feed
    the effective version, so two machines on different builds cannot serve
    each other's cached stills."""
    a = [n for n, _v, _c in _with_shelf(monkeypatch, _NVFP4_ONLY)]
    b = [n for n, _v, _c in _with_shelf(monkeypatch, _FP8_ONLY)]
    assert a != b, "different precisions must yield different version inputs"
