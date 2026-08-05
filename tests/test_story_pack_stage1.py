"""Stage 1 story-pack foundation tests (multi-modal story schema).

Proves: (a) the science pack is BYTE-IDENTICAL to the live prompt constants
(runtime-import equality), (b) exact seam-key set, (c) the fail-loud matrix,
(d) accessor semantics, (e) DORMANCY -- no production node consumes the loader
yet, so the sci-fi run is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_story_pack as sp
from nodes import _otr_compose_exchange as EX
from nodes import _otr_line_composer as L
from nodes import _otr_outline as O

REPO = Path(__file__).resolve().parents[1]
PACK_PATH = REPO / "nodes" / "story_packs" / "media_archive" / "media_restoration_adventure.json"

EXPECTED_SEAMS = frozenset({
    "outline_macro_system", "outline_phase_system", "outline_beat_system",
    "line_composer_system", "exchange_system", "coda_system",
    "announcer_intro_system", "announcer_intro_safe_system", "announcer_outro_system",
})


@pytest.fixture
def pack():
    sp._PACK_CACHE.clear()
    return sp.load_pack(PACK_PATH)


# -- (a) byte-identity + (b) exact seam set --------------------------------

def test_allowlist_equals_authored_set():
    assert sp.PRODUCTION_SEAM_ALLOWLIST == EXPECTED_SEAMS


def test_pack_seam_keys_exact(pack):
    assert set(pack.prompt_stages) == EXPECTED_SEAMS


def test_pack_metadata(pack):
    assert pack.source_bank_id == "media_archive"
    assert pack.story_model_id == "media_restoration_adventure"
    assert pack.story_pipeline_id == "legacy_many_pass"
    assert pack.schema_version == "v2.0"


# -- (c) fail-loud matrix ---------------------------------------------------

def _valid() -> dict:
    return {
        "source_bank_id": "b",
        "story_model_id": "m",
        "story_pipeline_id": "p",
        "schema_version": "v2.0",
        "prompt_stages": {"line_composer_system": "hi"},
    }


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "pack.json"
    p.write_text(text, encoding="utf-8")
    return p


def _write_obj(tmp_path: Path, obj: dict) -> Path:
    return _write(tmp_path, json.dumps(obj, ensure_ascii=False))


def test_missing_file():
    sp._PACK_CACHE.clear()
    with pytest.raises(sp.StoryPackNotFoundError):
        sp.load_pack(REPO / "nodes" / "story_packs" / "does_not_exist.json")


def test_malformed_json(tmp_path):
    sp._PACK_CACHE.clear()
    with pytest.raises(sp.StoryPackParseError):
        sp.load_pack(_write(tmp_path, "{ not json"))


def test_duplicate_key_top_level(tmp_path):
    sp._PACK_CACHE.clear()
    text = ('{"source_bank_id":"a","source_bank_id":"b",'
            '"story_model_id":"m","story_pipeline_id":"p",'
            '"schema_version":"v2.0","prompt_stages":{}}')
    with pytest.raises(sp.StoryPackParseError):
        sp.load_pack(_write(tmp_path, text))


def test_duplicate_key_nested(tmp_path):
    sp._PACK_CACHE.clear()
    text = ('{"source_bank_id":"a","story_model_id":"m","story_pipeline_id":"p",'
            '"schema_version":"v2.0","prompt_stages":'
            '{"line_composer_system":"x","line_composer_system":"y"}}')
    with pytest.raises(sp.StoryPackParseError):
        sp.load_pack(_write(tmp_path, text))


def test_unknown_top_level_key(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["bogus_field"] = 1
    with pytest.raises(sp.StoryPackValidationError):
        sp.load_pack(_write_obj(tmp_path, obj))


def test_missing_required_key(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    del obj["schema_version"]
    with pytest.raises(sp.StoryPackValidationError):
        sp.load_pack(_write_obj(tmp_path, obj))


def test_unknown_schema_version(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["schema_version"] = "v9.9"
    with pytest.raises(sp.StoryPackValidationError):
        sp.load_pack(_write_obj(tmp_path, obj))


def test_unknown_seam_key(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["prompt_stages"] = {"not_a_real_seam": "x"}
    with pytest.raises(sp.UnknownSeamError):
        sp.load_pack(_write_obj(tmp_path, obj))


def test_whitespace_only_seam_value(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["prompt_stages"] = {"line_composer_system": "   \n\t"}
    with pytest.raises(sp.StoryPackValidationError):
        sp.load_pack(_write_obj(tmp_path, obj))


# -- (d) accessor semantics -------------------------------------------------

def test_get_pack_prompt_present(pack):
    val = pack.prompt_stages["coda_system"]
    assert sp.get_pack_prompt(pack, "coda_system") == val
    assert sp.get_pack_prompt_or_none(pack, "coda_system") == val


def test_accessor_absent_seam(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()  # only line_composer_system present
    p = sp.load_pack(_write_obj(tmp_path, obj))
    assert sp.get_pack_prompt_or_none(p, "coda_system") is None
    with pytest.raises(sp.StoryPackValidationError):
        sp.get_pack_prompt(p, "coda_system")


def test_accessor_unknown_seam_raises(pack):
    with pytest.raises(sp.UnknownSeamError):
        sp.get_pack_prompt(pack, "not_a_real_seam")
    with pytest.raises(sp.UnknownSeamError):
        sp.get_pack_prompt_or_none(pack, "not_a_real_seam")


def test_or_none_preserves_exact_content(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["prompt_stages"] = {"line_composer_system": "  padded content  "}
    p = sp.load_pack(_write_obj(tmp_path, obj))
    # non-empty after strip -> returns the ORIGINAL untrimmed bytes
    assert sp.get_pack_prompt_or_none(p, "line_composer_system") == "  padded content  "


# -- (e) sanctioned-consumer guard (Stage 1b) -------------------------------

def test_only_sanctioned_consumer_uses_loader():
    """Stage 1b wires exactly ONE production consumer: the creative prompt
    router (line_composer_system). Any OTHER production file consuming the loader
    is unexpected and must be reviewed (catches accidental scope creep)."""
    allowed = {"_otr_story_pack.py", "_otr_creative_prompt_router.py",
               "_otr_story_routing.py"}
    offenders = []
    for py in (REPO / "nodes").rglob("*.py"):
        if py.name in allowed:
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if "_otr_story_pack" in text or "get_pack_prompt" in text:
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"unexpected loader consumers beyond {allowed}: {offenders}"


def test_load_pack_with_seams_sole_caller_is_routing():
    """The permissive loader (load_pack_with_seams) must have EXACTLY ONE
    production caller: the routing layer. Anywhere else it would bypass the
    strict production-seam admission control."""
    offenders = []
    for py in (REPO / "nodes").rglob("*.py"):
        if py.name in {"_otr_story_pack.py", "_otr_story_routing.py"}:
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if "load_pack_with_seams" in text:
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, (
        f"load_pack_with_seams called outside _otr_story_routing.py: {offenders}"
    )


# -- (f) Stage 1b wiring: router sources the seam from the pack -------------

def test_stage1b_router_sources_line_composer_from_pack(pack):
    """The router returns line_composer_system from the JSON pack of its
    default lane (media_archive) -> the seam is pack-owned, not a Python
    constant."""
    from nodes import _otr_creative_prompt_router as router
    from nodes import _otr_model_catalog as cat

    modern = [m.repo_id for m in cat.CURATED_LLM_MODELS
              if m.prompt_profile == "modern"]
    assert modern, "expected at least one modern curated model"
    out = router.resolve_creative_system_prompt(modern[0], "line_composer_system")
    assert out == pack.prompt_stages["line_composer_system"]
    # outline stays object-identical (not pack-sourced in Stage 1b).
    out_outline = router.resolve_creative_system_prompt(modern[0], "outline")
    assert out_outline is O._SYSTEM_PROMPT


def test_stage1b_router_fail_loud_on_missing_pack(monkeypatch, tmp_path):
    """No fallback at the router: if the routing registry / science pack is
    missing, resolving the migrated seam RAISES (never silently reverts to a
    Python constant). Stage 2: routed via _otr_story_routing, so we point the
    routing layer at an empty story_packs root."""
    from nodes import _otr_creative_prompt_router as router
    from nodes import _otr_model_catalog as cat
    from nodes import _otr_story_routing as routing

    routing._clear_caches()
    monkeypatch.setattr(routing, "_STORY_PACKS_ROOT", tmp_path / "story_packs")
    modern = next(m.repo_id for m in cat.CURATED_LLM_MODELS
                  if m.prompt_profile == "modern")
    with pytest.raises(routing.StoryRoutingError):
        router.resolve_creative_system_prompt(modern, "line_composer_system")
    routing._clear_caches()


# -- (g) Stage 2 precondition: NO swallow around the outline resolver -------

def test_outline_resolver_call_not_swallowed():
    """Fable forward-note (2026-07-05): the old `except Exception ->
    period_system_overlay = None` around the outline resolver call would become
    a SILENT FALLBACK once the outline seam is pack-sourced. Pin its removal:
    no resolve_creative_system_prompt call in _otr_outline.py may sit inside a
    try/except."""
    import ast

    src = (REPO / "nodes" / "_otr_outline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # annotate parents
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]

    swallowed = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "resolve_creative_system_prompt"):
            anc = getattr(node, "_parent", None)
            while anc is not None:
                if isinstance(anc, ast.Try):
                    swallowed.append(node.lineno)
                    break
                anc = getattr(anc, "_parent", None)
    assert not swallowed, (
        f"resolve_creative_system_prompt call(s) at line(s) {swallowed} in "
        f"_otr_outline.py are wrapped in try/except -- the swallow is a "
        f"hidden fallback; resolver/pack errors must fail the episode loud"
    )


# ---------------------------------------------------------------------------
# 2026-08-05: the 08-03 guardrail rip deleted the list item "blood, guns,
# knives, and graphic violence" but left its separating comma standing in front
# of the full stop, so FOUR live prompt strings ended mid-list --
# "...unrelated framing story, changed ending,." A model reads these on every
# fidelity episode, and a sentence that stops at a comma reads as a truncated
# instruction. Both adaptation packs, both prompt stages each.
# ---------------------------------------------------------------------------
def _pack_files():
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "nodes" / "story_packs"
    return sorted(root.glob("*/*.json"))


def test_no_prompt_stage_ends_a_sentence_on_a_comma():
    """Sweeps EVERY stage of EVERY pack, not just the four known sites --
    otherwise the next clause removal reintroduces exactly this.
    """
    import json

    offenders = []
    for path in _pack_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        stages = data.get("prompt_stages")
        if not isinstance(stages, dict):
            continue
        for stage, text in stages.items():
            if not isinstance(text, str):
                continue
            if ",." in text:
                i = text.index(",.")
                offenders.append(
                    "%s::%s ...%s" % (path.name, stage, text[max(0, i - 60):i + 2]))
    assert not offenders, (
        "prompt stage ends a sentence on a dangling comma:\n  "
        + "\n  ".join(offenders))


def test_the_packs_are_still_valid_json_with_prompt_stages():
    """Guards the sweep above: if the packs stopped parsing, or stopped
    carrying prompt_stages, the sweep would pass by finding nothing.
    """
    import json

    files = _pack_files()
    assert files, "no story packs found -- the sweep would be vacuous"
    staged = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("prompt_stages"), dict) and data["prompt_stages"]:
            staged += 1
    assert staged >= 2, f"only {staged} pack(s) carry prompt_stages"
