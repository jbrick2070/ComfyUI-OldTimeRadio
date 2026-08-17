"""The style traceroute's VIDEO side -- D-BIS finding 1 (2026-08-17).

`otr_style_traceroute.py` had ZERO test coverage while being the tool the
operator's "we can't have negative prompts conflicting with any visual style"
ruling is checked with. This pins the half added for D-BIS: the AST reader that
pulls a video engine's negative without importing it, the motion_registers
phrase test, and -- the row that actually earns its keep -- the MEASURED result,
so a pack reword or a recipe change cannot silently move the number.

It also pins the deliberate NON-behaviour: `--strict` must stay scoped to STILL
effective fights. The video negatives are frozen recipe, so gating CI on them
would be a build-breaker by construction.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "otr_style_traceroute.py"


def _load():
    spec = importlib.util.spec_from_file_location("otr_style_traceroute",
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load()


class _Pack:
    """The only surface find_video_fights reads."""

    def __init__(self, registers):
        self.motion_registers = registers


# --------------------------------------------------------------------------- #
# _literal_assign -- read a constant without importing the render stack
# --------------------------------------------------------------------------- #

def test_literal_assign_reads_a_plain_string(tmp_path):
    src = tmp_path / "m.py"
    src.write_text('_NEG = ("low quality, "\n        "blurry")\n',
                   encoding="utf-8")
    assert tr._literal_assign(str(src), "_NEG") == "low quality, blurry"


def test_literal_assign_unwraps_a_call(tmp_path):
    """The eng_cloud_video shape: visual_safety_negative("...")."""
    src = tmp_path / "m.py"
    src.write_text('_NEG = visual_safety_negative("jump cuts, flicker")\n',
                   encoding="utf-8")
    assert tr._literal_assign(str(src), "_NEG") == "jump cuts, flicker"


@pytest.mark.parametrize("body, name", [
    ('_OTHER = "x"\n', "_NEG"),          # name absent
    ('_NEG = 5\n', "_NEG"),              # not a string
    ('_NEG = SOME_CONST\n', "_NEG"),     # not a literal
    ('def (\n', "_NEG"),                 # unparseable
])
def test_literal_assign_degrades_to_empty(tmp_path, body, name):
    """This tool reports and must never fail a build, so an unreadable
    constant is a skipped row, never an exception."""
    src = tmp_path / "m.py"
    src.write_text(body, encoding="utf-8")
    assert tr._literal_assign(str(src), name) == ""


def test_literal_assign_missing_file_is_empty():
    assert tr._literal_assign(str(_REPO / "no_such_file.py"), "_NEG") == ""


# --------------------------------------------------------------------------- #
# find_video_fights
# --------------------------------------------------------------------------- #

def test_hyphen_variant_is_caught():
    """The single widest conflict in the tree: the engine writes "whip pans",
    every pack writes "whip-pans". A strict literal test misses all four."""
    pack = _Pack({"music_open": "Dial whip-pans across frequencies."})
    assert tr.find_video_fights("jump cuts, whip pans, jitter", pack) \
        == ["whip pans"]


def test_plural_and_spacing_variants():
    pack = _Pack({"announcer": "Candlelight flickers across polished wood."})
    assert tr.find_video_fights("flicker", pack) == ["flicker"]


def test_artifact_terms_are_never_fights():
    """"text" and "watermark" are anti-ARTIFACT, not anti-STYLE -- a pack may
    carry them even where the word appears in its own positive."""
    pack = _Pack({"announcer": "Drifting text and a watermark shimmer."})
    assert tr.find_video_fights("text, watermark", pack) == []


def test_no_motion_registers_is_no_fight():
    assert tr.find_video_fights("whip pans", _Pack({})) == []
    assert tr.find_video_fights("", _Pack({"a": "whip pans"})) == []


def test_unrelated_negative_is_clean():
    pack = _Pack({"announcer": "Dial needle glides. Tubes glow steadily."})
    assert tr.find_video_fights("low quality, blurry, deformed", pack) == []


# --------------------------------------------------------------------------- #
# The MEASURED result -- the pin that catches a reword or a recipe change
# --------------------------------------------------------------------------- #

def _style(sid):
    return tr.vs.resolve_visual_style(sid)


def _all_style_ids():
    import os
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(tr.vs._VISUAL_STYLES_ROOT)
                  if f.endswith(".json"))


def _wan_negative():
    return tr._literal_assign(
        str(_REPO / "nodes/_otr_video_engines/eng_cloud_video.py"),
        "_WAN_NEGATIVE_DEFAULT")


@pytest.mark.parametrize("sid", _all_style_ids())
def test_no_pack_fights_the_cloud_wan_negative(sid):
    """THE INVARIANT, after the 2026-08-17 subject-first rewrite.

    These registers used to collide: four packs asked for a dial that
    "whip-pans" while the negative bans "whip pans", and
    `shakespeare_stage_realism` asked for candlelight that "flickers" while the
    negative bans "flicker". Both were CAMERA-crew vocabulary borrowed to
    describe a diegetic object, which is what made them collide at all -- the
    negative means a camera whip pan and the pack meant the needle.

    The registers now describe what the RADIO does and leave the camera to the
    engine, so the collision cannot recur by wording. Zero, for every pack.
    """
    negative = _wan_negative()
    assert negative, "the cloud Wan negative must stay AST-readable"
    assert tr.find_video_fights(negative, _style(sid)) == []


@pytest.mark.parametrize("label, relpath, const", [
    e for e in tr.VIDEO_NEGATIVE_SOURCES if e[0] != "eng_cloud_video"])
def test_every_local_video_engine_is_clean(label, relpath, const):
    """The exposure is confined to the opt-in CLOUD lane. The local engines'
    negatives are pure quality/artifact terms, and that is worth pinning: it is
    the reason this finding is narrow."""
    negative = tr._literal_assign(str(_REPO / relpath), const)
    assert negative, "%s negative must stay AST-readable" % label
    for sid in _all_style_ids():
        assert tr.find_video_fights(negative, _style(sid)) == [], (
            "%s now conflicts with %s" % (label, sid))


# --------------------------------------------------------------------------- #
# The SEEDANCE softeners -- the live defect the panel surfaced (2026-08-17)
# --------------------------------------------------------------------------- #

def _seedance_softeners():
    """The shipped softener table, read statically from the cloud engine."""
    import ast as _ast
    import re as _re
    src = (_REPO / "nodes/_otr_video_engines/eng_cloud_video.py").read_text(
        encoding="utf-8")
    out = []
    for node in _ast.parse(src).body:
        if isinstance(node, _ast.Assign) and any(
                getattr(t, "id", "") == "_SEEDANCE_PROMPT_SOFTENERS"
                for t in node.targets):
            for elt in node.value.elts:
                out.append((elt.elts[0].value,
                            _re.compile(elt.elts[1].args[0].value, _re.I),
                            elt.elts[2].value))
    return out


def test_the_softener_table_is_still_readable():
    """If this breaks, the test below is silently passing on an empty list."""
    assert len(_seedance_softeners()) >= 5


@pytest.mark.parametrize("sid", _all_style_ids())
def test_no_register_trips_a_seedance_softener(sid):
    """THE REAL DEFECT, and why the registers were rewritten subject-first.

    The Seedance cloud lane rewrites register text at compose time
    (`_SEEDANCE_PROMPT_SOFTENERS`, eng_cloud_video.py). It is a blind regex
    pass, so it produced genuinely broken prompts from perfectly good writing:

        cartoon.music_open  "Dial whip-pans wildly."
                         -> "Dial slowly sweeps wildly."      (a contradiction)
        sci_fi_radio.music_open  four substitutions in one line, including
            "vibrates aggressively" -> "vibrates subtly"      (energy REVERSED)
            "cold to white-hot"     -> "cold to bright warm glow" (broken)

    That is an engine overriding the episode's own style -- the same class as
    PBUG-20260817-01, on the video side, and it was already shipping. The
    registers no longer contain any phrase this table matches, so the pass is a
    no-op on our own packs and the authored words are what reach the provider.

    This does NOT touch the softener table, which is frozen recipe. It pins the
    packs, which are ours.
    """
    style = _style(sid)
    for key, value in (getattr(style, "motion_registers", None) or {}).items():
        for name, pattern, replacement in _seedance_softeners():
            assert not pattern.search(value), (
                "%s.%s trips the %r softener -> it would ship as %r"
                % (sid, key, name, pattern.sub(replacement, value)))


# --------------------------------------------------------------------------- #
# The deliberate non-behaviour
# --------------------------------------------------------------------------- #

def test_strict_is_not_gated_on_video_conflicts(capsys):
    """Video conflicts are REPORTED, never gated. The video negatives are
    frozen recipe, so --strict failing on them would be a build-breaker by
    construction. STILL effective fights are 0, so --strict must exit 0 even
    though the video section reports five conflicts."""
    rc = tr.main(["--strict"])
    out = capsys.readouterr().out
    assert "VIDEO-SIDE" in out
    assert "VIDEO conflicts:" in out
    assert rc == 0
