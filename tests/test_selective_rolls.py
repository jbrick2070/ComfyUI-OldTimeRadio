"""Selective rolls (operator, 2026-09-26): a chosen pool for the visual-style
roll, and a language roll with its own pool.

The roll draw, seed and receipt were already pool-agnostic; what is new is
the POOL (a native multi-select COMBO per surface, appended), the language
roll itself (resolved AFTER the bank, honouring each language row's
`source_bank_exclusions`), the floor fallback following the roll's own pool,
and the API converters sending a list value the way the frontend does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from nodes import _otr_episode_languages as LANG  # noqa: E402
from nodes import _otr_rolls as ROLLS  # noqa: E402
from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as W  # noqa: E402

CANONICAL = REPO / "workflows" / "otr_canonical.json"
LANGUAGES = tuple(c for c in LANG.dropdown_choices() if c != LANG.OFF_LABEL)


# ---------------------------------------------------------------------------
# the pool parser
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [None, "", "   ", [], ()])
def test_an_empty_pool_is_no_pool(raw):
    assert ROLLS.parse_roll_pool(raw, valid_ids=("a", "b"), surface="s") == ()


@pytest.mark.parametrize("raw", [
    ["b", "a", "b"],              # the native multi-select value
    '["b", "a", "b"]',            # a headless --set JSON list
    "b, a ,b",                    # a headless --set typed by a person
])
def test_every_value_shape_parses_to_the_same_pool(raw):
    got = ROLLS.parse_roll_pool(raw, valid_ids=("a", "b", "c"), surface="s")
    assert got == ("b", "a")      # first-seen order, no repeats


def test_an_unknown_id_is_refused_loud_never_dropped():
    with pytest.raises(ROLLS.RollError, match="'zzz'"):
        ROLLS.parse_roll_pool(["a", "zzz"], valid_ids=("a",), surface="s")


def test_a_refused_id_is_refused_by_name():
    with pytest.raises(ROLLS.RollError, match="cannot be rolled"):
        ROLLS.parse_roll_pool(["Off"], valid_ids=("Off", "English"),
                              surface="episode_language", refused=("Off",))


@pytest.mark.parametrize("raw", ["[not json", 7, {"a": 1}])
def test_a_malformed_pool_is_refused(raw):
    with pytest.raises(ROLLS.RollError):
        ROLLS.parse_roll_pool(raw, valid_ids=("a",), surface="s")


# ---------------------------------------------------------------------------
# the visual-style roll with a pool
# ---------------------------------------------------------------------------
def test_a_pool_beside_a_manual_style_is_ignored():
    """The manual path stays byte-identical: no receipt, pool unread."""
    assert ROLLS.resolve_style_selection("anime", pool=["cartoon", "video_art"],
                                         env={}) == ("anime", None)


def test_an_empty_style_pool_is_the_whole_list_roll():
    _sel, rec = ROLLS.resolve_style_selection(ROLLS.STYLE_SENTINEL, pool=[], env={})
    assert rec.eligible_order == ROLLS.eligible_style_ids()


def test_a_style_pool_of_one_is_a_pick():
    assert ROLLS.resolve_style_selection(
        ROLLS.STYLE_SENTINEL, pool=["video_art"], env={}) == ("video_art", None)


def test_a_style_pool_rolls_among_exactly_those_and_says_so():
    pool = ["video_art", "anime", "cartoon"]
    seen = set()
    for seed in range(40):
        sel, rec = ROLLS.resolve_style_selection(
            ROLLS.STYLE_SENTINEL, pool=pool,
            env={ROLLS.STYLE_SEED_ENV: str(seed)})
        assert rec.eligible_order == ("anime", "cartoon", "video_art")
        assert rec.surface == "visual_style" and sel == rec.selected
        seen.add(sel)
    assert seen == set(pool)          # every member reachable, nothing else


def test_a_seeded_style_pool_roll_replays():
    env = {ROLLS.STYLE_SEED_ENV: "12345"}
    a = ROLLS.resolve_style_selection(ROLLS.STYLE_SENTINEL, pool=["anime", "cartoon"], env=env)
    b = ROLLS.resolve_style_selection(ROLLS.STYLE_SENTINEL, pool=["cartoon", "anime"], env=env)
    assert a[0] == b[0]               # the receipt order is sorted, so input order cannot matter


# ---------------------------------------------------------------------------
# the dynamic lane's floor follows the roll's own pool
# ---------------------------------------------------------------------------
def test_the_floor_for_the_whole_list_roll_is_unchanged():
    _sel, rec = ROLLS.resolve_style_selection(ROLLS.STYLE_SENTINEL, env={})
    assert ROLLS.floor_style_order(rec) == ROLLS.floor_style_ids()
    assert ROLLS.floor_style_order(None) == ROLLS.floor_style_ids()


def test_the_floor_for_a_pool_stays_inside_the_pool():
    _sel, rec = ROLLS.resolve_style_selection(
        ROLLS.STYLE_SENTINEL,
        pool=[ROLLS.DYNAMIC_STYLE_ID, "anime", "cartoon"], env={})
    assert ROLLS.floor_style_order(rec) == ("anime", "cartoon")


# ---------------------------------------------------------------------------
# the language roll
# ---------------------------------------------------------------------------
def test_a_manual_language_is_untouched():
    assert ROLLS.resolve_language_selection(
        "French", pool=["Spanish", "Italian"], env={}) == ("French", None)


def test_the_whole_language_roll_never_lands_on_off():
    for seed in range(60):
        sel, rec = ROLLS.resolve_language_selection(
            ROLLS.LANGUAGE_SENTINEL, env={ROLLS.LANGUAGE_SEED_ENV: str(seed)})
        assert sel != LANG.OFF_LABEL and sel in LANGUAGES
        assert rec.eligible_order == tuple(sorted(LANGUAGES))
        assert rec.surface == "episode_language"
        assert rec.requested == ROLLS.LANGUAGE_SENTINEL
        assert LANG.resolve_label(sel).row is not None   # a real, stampable row


def test_the_roll_label_has_one_spelling():
    assert ROLLS.LANGUAGE_SENTINEL == LANG.ROLL_LABEL


def test_off_in_a_language_pool_is_refused():
    with pytest.raises(ROLLS.RollError, match="cannot be rolled"):
        ROLLS.resolve_language_selection(
            ROLLS.LANGUAGE_SENTINEL, pool=["Off", "French"], env={})


def test_a_language_pool_of_one_is_a_pick():
    assert ROLLS.resolve_language_selection(
        ROLLS.LANGUAGE_SENTINEL, pool=["Japanese"], env={}) == ("Japanese", None)


def test_a_language_pool_rolls_among_exactly_those_and_replays():
    env = {ROLLS.LANGUAGE_SEED_ENV: "99"}
    sel, rec = ROLLS.resolve_language_selection(
        ROLLS.LANGUAGE_SENTINEL, pool=["Spanish", "French"], env=env)
    assert rec.eligible_order == ("French", "Spanish") and sel in ("French", "Spanish")
    again, _ = ROLLS.resolve_language_selection(
        ROLLS.LANGUAGE_SENTINEL, pool=["French", "Spanish"], env=env)
    assert again == sel


class _Row:
    def __init__(self, excluded):
        self.admission = {"source_bank_exclusions": list(excluded)}


def test_the_language_roll_honours_a_rows_bank_exclusion(monkeypatch):
    """Empty on every shipped row (2026-09-18: every lane takes every
    language), but it is the gate -- a roll must not land on a refusal."""
    real = LANG.row_by_label
    monkeypatch.setattr(LANG, "row_by_label", lambda label, **kw: (
        _Row(["shakespeare"]) if label == "French" else real(label, **kw)))
    for seed in range(30):
        sel, rec = ROLLS.resolve_language_selection(
            ROLLS.LANGUAGE_SENTINEL, pool=["French", "Spanish", "Italian"],
            source_bank_id="shakespeare", env={ROLLS.LANGUAGE_SEED_ENV: str(seed)})
        assert sel != "French"
        assert rec.eligible_order == ("Italian", "Spanish")


def test_a_pool_the_bank_excludes_entirely_fails_loud(monkeypatch):
    monkeypatch.setattr(LANG, "row_by_label", lambda label, **kw: _Row(["shakespeare"]))
    with pytest.raises(ROLLS.RollError, match="excludes source bank 'shakespeare'"):
        ROLLS.resolve_language_selection(
            ROLLS.LANGUAGE_SENTINEL, pool=["French", "Spanish"],
            source_bank_id="shakespeare", env={})


def test_the_language_roll_never_mutates_process_env():
    before = dict(__import__("os").environ)
    ROLLS.resolve_language_selection(ROLLS.LANGUAGE_SENTINEL, env={})
    assert dict(__import__("os").environ) == before


# ---------------------------------------------------------------------------
# the story bank's pool (2026-09-27)
# ---------------------------------------------------------------------------
def test_a_pool_beside_a_manual_bank_is_ignored():
    assert ROLLS.resolve_bank_selection(
        "original", pool="shakespeare, my_story", env={}) == ("original", None)


def test_an_empty_bank_pool_is_the_whole_eligible_roll():
    _sel, rec = ROLLS.resolve_bank_selection(ROLLS.BANK_SENTINEL, pool="", env={})
    assert rec.eligible_order == ROLLS.eligible_bank_ids()


def test_a_bank_pool_of_one_is_a_pick_and_may_keep_a_pinned_source():
    """One name is the manual path, so a pinned source_ref is coherent."""
    assert ROLLS.resolve_bank_selection(
        ROLLS.BANK_SENTINEL, pool="shakespeare", source_ref="macbeth",
        env={}) == ("shakespeare", None)


def test_a_bank_pool_rolls_among_exactly_those_and_says_so():
    pool = "original, public_domain, shakespeare"
    seen = set()
    for seed in range(40):
        sel, rec = ROLLS.resolve_bank_selection(
            ROLLS.BANK_SENTINEL, pool=pool,
            env={ROLLS.BANK_SEED_ENV: str(seed)})
        assert rec.eligible_order == ("original", "public_domain", "shakespeare")
        assert rec.surface == "source_bank" and sel == rec.selected
        seen.add(sel)
    assert seen == {"original", "public_domain", "shakespeare"}


def test_a_bank_pool_roll_still_refuses_a_pinned_source():
    with pytest.raises(ROLLS.RollError):
        ROLLS.resolve_bank_selection(
            ROLLS.BANK_SENTINEL, pool="original, shakespeare",
            source_ref="macbeth", env={})


def test_an_unknown_bank_in_the_pool_is_refused_loud():
    with pytest.raises(ROLLS.RollError):
        ROLLS.resolve_bank_selection(
            ROLLS.BANK_SENTINEL, pool="original, not_a_bank", env={})


# ---------------------------------------------------------------------------
# the writer surface: trailing typed-list pools
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,names", [
    ("style_roll_pool", list(ROLLS.eligible_style_ids())),
    ("language_roll_pool", list(LANGUAGES)),
    ("bank_roll_pool", list(ROLLS.eligible_bank_ids())),
])
def test_each_pool_is_a_typed_list_the_app_view_can_draw(name, names):
    """A STRING, not a multi-select: the app view (frontend 1.52.7) drew the
    2.3.8 multi-select as a blank canvas, so it could not be used there. The
    tooltip names every valid entry, and never Off."""
    kind, meta = W.INPUT_TYPES()["optional"][name]
    assert kind == "STRING" and meta["default"] == ""
    assert "multiselect" not in meta and "multi_select" not in meta
    for entry in names:
        assert entry in meta["tooltip"]
    assert "Choose from: Off" not in meta["tooltip"]


@pytest.mark.parametrize("saved", [[], ["anime", "cartoon"], ["Español"]])
def test_a_2_3_8_saved_list_survives_cores_str_coercion(saved):
    """A graph saved by 2.3.8 holds a LIST here. The frontend wraps it as
    {"__value__": [...]}, core unwraps it and then runs str() on every STRING
    input (execution.py) -- so the node receives Python's repr, not JSON."""
    coerced = str(saved)                  # exactly what core hands run()
    if saved == ["Español"]:
        with pytest.raises(ROLLS.RollError):   # still refused: not a style
            ROLLS.parse_roll_pool(coerced, valid_ids=ROLLS.eligible_style_ids(), surface="s")
        return
    got = ROLLS.parse_roll_pool(coerced, valid_ids=ROLLS.eligible_style_ids(), surface="s")
    assert got == tuple(saved)


def test_bracketed_text_that_is_no_list_is_refused_loud():
    for text in ("[anime, cartoon", "[__import__('os')]", "[1+1]"):
        with pytest.raises(ROLLS.RollError):
            ROLLS.parse_roll_pool(text, valid_ids=ROLLS.eligible_style_ids(), surface="s")


def test_a_typed_pool_and_a_saved_list_parse_the_same():
    """A graph saved by 2.3.8 holds a LIST in this slot; it still reads."""
    typed = ROLLS.parse_roll_pool("video_art; anime\ncartoon",
                                  valid_ids=ROLLS.eligible_style_ids(), surface="s")
    saved = ROLLS.parse_roll_pool(["video_art", "anime", "cartoon"],
                                  valid_ids=ROLLS.eligible_style_ids(), surface="s")
    assert typed == saved == ("video_art", "anime", "cartoon")


@pytest.mark.parametrize("typed,expected", [
    ("english, spanish", ("English", "Spanish")),
    ("ENGLISH;french", ("English", "French")),
    ("  Japanese  ", ("Japanese",)),
])
def test_a_language_pool_ignores_case(typed, expected):
    """Operator, 2026-09-28: people type into a text box. "english" is
    English; the whole run must not stop over a capital letter."""
    assert ROLLS.parse_roll_pool(typed, valid_ids=LANGUAGES,
                                 surface="episode_language") == expected


@pytest.mark.parametrize("typed,expected", [
    ("video art, Recur-Frac", ("video_art", "recur_frac")),
    ("VIDEO_ART", ("video_art",)),
])
def test_a_style_pool_ignores_case_spaces_and_hyphens(typed, expected):
    assert ROLLS.parse_roll_pool(typed, valid_ids=ROLLS.eligible_style_ids(),
                                 surface="visual_style") == expected


def test_a_bank_pool_ignores_case_and_spaces():
    assert ROLLS.parse_roll_pool("Public Domain, SHAKESPEARE",
                                 valid_ids=ROLLS.eligible_bank_ids(),
                                 surface="source_bank") == ("public_domain", "shakespeare")


def test_forgiving_is_not_guessing():
    """Case and separators are forgiven; a different name is not."""
    with pytest.raises(ROLLS.RollError, match="Espanol"):
        ROLLS.parse_roll_pool("Espanol", valid_ids=LANGUAGES, surface="episode_language")
    with pytest.raises(ROLLS.RollError):
        ROLLS.parse_roll_pool("off", valid_ids=LANGUAGES, surface="episode_language",
                              refused=(LANG.OFF_LABEL,))


def test_the_roll_resolvers_return_canonical_ids_for_forgiving_input():
    sel, rec = ROLLS.resolve_style_selection(ROLLS.STYLE_SENTINEL, pool="Video Art, anime", env={})
    assert rec.eligible_order == ("anime", "video_art") and sel in rec.eligible_order
    assert ROLLS.resolve_bank_selection(ROLLS.BANK_SENTINEL, pool="Original", env={}) == ("original", None)


def test_the_canonical_saves_every_pool_as_a_trailing_empty_slot():
    wf = json.loads(CANONICAL.read_text(encoding="utf-8"))
    node = next(n for n in wf["nodes"] if n["type"] == "OTR_LedgerScriptWriter")
    assert node["widgets_values"][-3:] == ["", "", ""]
    assert [i["name"] for i in node["inputs"][-3:]] == [
        "style_roll_pool", "language_roll_pool", "bank_roll_pool"]
    assert all(i["link"] is None and i["widget"]["name"] == i["name"]
               for i in node["inputs"][-3:])


def test_every_roll_in_the_app_form_has_its_pool_right_under_it():
    """Operator, 2026-09-27: the three rolls must look alike in the app
    view -- the dropdown with its roll row, then its typed pool."""
    form = json.loads((REPO / "config" / "app_mode.json").read_text(
        encoding="utf-8"))["form"]
    names = [row[1] for row in form if row[0] == "OTR_LedgerScriptWriter"]
    for dropdown, pool in (("episode_language", "language_roll_pool"),
                           ("source_bank", "bank_roll_pool"),
                           ("visual_style", "style_roll_pool")):
        assert names[names.index(dropdown) + 1] == pool


def test_the_writer_wires_both_pools_and_the_language_roll_at_their_real_sites():
    """The helpers are proven above; this proves run() CALLS them, in the
    order that matters: the language roll after the bank is bound, before
    the admission gate, and its receipt stamped beside the bank's."""
    src = (REPO / "nodes" / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    body = src[src.index("    def run(\n"):]
    style = body.index("visual_style, pool=style_roll_pool)")
    bank_bound = body.index("_source_bank_row = _otr_story_routing.require_runnable_bank(source_bank)")
    language = body.index("_ROLLS.resolve_language_selection(")
    gate = body.index("_EPLANG.check_source_bank_admission(")
    assert style < bank_bound < language < gate
    assert "pool=language_roll_pool" in body[language:gate]
    assert 'meta["language_roll"] = _language_roll.to_meta()' in body
    bank = body.index("_ROLLS.resolve_bank_selection(")
    assert "pool=bank_roll_pool" in body[bank:style]


def test_the_floor_fallback_reads_the_rolls_own_pool():
    src = (REPO / "nodes" / "_otr_writer_tail.py").read_text(encoding="utf-8")
    assert "_ROLLS.floor_style_order(ctx.style_roll)" in src
    assert "_ROLLS.floor_style_ids()" not in src


# ---------------------------------------------------------------------------
# the API converters send a list the way the frontend does
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("which", ["script", "package"])
def test_a_list_widget_value_goes_out_wrapped(which):
    """A bare list in an API prompt is read as a LINK; the frontend's
    graphToPrompt sends {"__value__": [...]} and core unwraps it."""
    if which == "script":
        import otr_api as conv
        from nodes._otr_workflow_apply import build_offline_schemas
        schemas = build_offline_schemas()
    else:
        from nodes import _otr_workflow_apply as conv
        schemas = conv.build_offline_schemas()
    wf = json.loads(CANONICAL.read_text(encoding="utf-8"))
    node = next(n for n in wf["nodes"] if n["type"] == "OTR_LedgerScriptWriter")
    node["widgets_values"][-3] = ["anime", "cartoon"]
    prompt = conv.workflow_to_api_prompt(wf, schemas)
    inputs = prompt[str(node["id"])]["inputs"]
    assert inputs["style_roll_pool"] == {"__value__": ["anime", "cartoon"]}
    assert inputs["language_roll_pool"] == ""             # a string goes out as-is
    assert isinstance(inputs["episode_language"], str)   # scalars untouched


@pytest.mark.parametrize("which", ["script", "package"])
def test_both_patchers_take_a_typed_pool(which):
    """The package's patcher and the script's both accept the typed list; an
    unknown name is refused at run time by parse_roll_pool (tested above)."""
    from nodes import _otr_workflow_apply as WA
    schemas = WA.build_offline_schemas()
    wf = json.loads(CANONICAL.read_text(encoding="utf-8"))
    node = next(n for n in wf["nodes"] if n["type"] == "OTR_LedgerScriptWriter")
    if which == "script":
        import otr_api as mod
    else:
        mod = WA
    mod.patch_creative(wf, node["id"], "style_roll_pool", "anime, cartoon", schemas)
    assert node["widgets_values"][-3] == "anime, cartoon"
    mod.patch_creative(wf, node["id"], "bank_roll_pool", "original", schemas)
    assert node["widgets_values"][-1] == "original"


def test_set_carries_a_pool_through_the_headless_runner(tmp_path):
    import contextlib
    import io

    import otr_canonical_api_run as canonical
    dump = tmp_path / "prompt.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = canonical.main([
            "--offline-schemas", "--dry-run",
            "--set", 'OTR_LedgerScriptWriter.visual_style="roll (any style)"',
            "--set", "OTR_LedgerScriptWriter.style_roll_pool=anime, video_art",
            "--dump-prompt", str(dump),
        ])
    assert rc == 0
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = next(n for n in prompt.values()
                  if n.get("class_type") == "OTR_LedgerScriptWriter")
    assert writer["inputs"]["style_roll_pool"] == "anime, video_art"
    assert writer["inputs"]["visual_style"] == ROLLS.STYLE_SENTINEL
