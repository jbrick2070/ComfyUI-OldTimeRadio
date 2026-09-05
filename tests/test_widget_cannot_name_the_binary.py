"""A node WIDGET may not name the binary this pack spawns.

THE DEFECT THIS CLOSES, reproduced against the real modules on 2026-09-04.
Five shipped nodes expose an ``ffmpeg`` STRING widget. A widget value arrives in
the body of ComfyUI's ``/prompt`` request -- unauthenticated by default -- and is
whatever a downloaded workflow JSON says. It reached ``argv[0]``:

    widget -> _ffmpeg_bin(ffmpeg) -> resolve_ffmpeg(preferred)
           -> _explicit()  honoured it (it carried a directory)
           -> _usable()    honoured it (the file existed)
           -> proc.py      allowed it (argv[0]'s BASENAME is "ffmpeg")
           -> otr_proc.run EXECUTED IT

Measured with ``OTR_FFMPEG`` pinned to a real binary, a widget value of
``<tmp>\\ffmpeg.exe`` BEAT the operator's pin, and ``resolve_ffprobe(ffmpeg=...)``
produced a SECOND attacker binary through the sibling rule.

WHERE THE FIX LIVES, and why not the two obvious places:

* NOT in ``_explicit``: provenance is invisible there. Trusted internal callers
  legitimately pass directory-bearing arguments -- ``blend()`` resolves ffmpeg
  and threads that RESOLVED path through ``_probe_dims`` -> ``probe_raw`` ->
  ``resolve_ffprobe(ffmpeg=...)``.
* NOT in ``_ffmpeg_bin``: ``otr_master_audio_mux`` deliberately hands its
  ALREADY-RESOLVED binary to ``audio_pcm_sha`` so the byte-identity proof cannot
  resolve differently from the encode that just ran.
* NOT as an "argv[0] must be absolute" gate in ``proc.py``: that owner
  deliberately admits bare ``git`` and ``nvidia-smi``
  (``production_ledger.py``, ``_otr_ledger.py``, ``_otr_sys_specs.py``), each
  wrapped in ``except Exception`` -> "unknown". Such a gate would blank the
  ledger's commit stamp on EVERY episode, with a green run and a published obs
  artifact, invisible to the whole suite.

So: the widget is discarded at each node's EXECUTE METHOD, and the resolvers
were made to return an ABSOLUTE path or ``None``.
"""
from __future__ import annotations

import ast
import os
import pathlib
import shutil

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The five nodes whose `ffmpeg` widget used to reach argv[0].
WIDGET_NODES = {
    "nodes/otr_caption_burn.py": "OTR_CaptionBurn",
    "nodes/otr_master_audio_mux.py": "OTR_MasterAudioMux",
    "nodes/otr_post_upscale_procgen_blend.py": "OTR_PostUpscaleProcgenBlend",
    "nodes/otr_scene_aware_scopes.py": "OTR_SceneAwareScopes",
    "nodes/otr_silent_composite.py": "OTR_SilentComposite",
}

HOSTILE = r"C:\Users\victim\Downloads\payload\ffmpeg.exe"


# --------------------------------------------------------------------------- #
# the helper itself
# --------------------------------------------------------------------------- #
def _helper():
    from nodes._otr_shared.ffmpeg import widget_ffmpeg_is_ignored
    return widget_ffmpeg_is_ignored


@pytest.mark.parametrize("value", [
    HOSTILE,
    r"\\attacker-host\share\ffmpeg.exe",      # UNC, no local file needed
    "./ffmpeg",                               # cwd-relative
    "ffmpeg-7.1",                             # a non-default BARE name
    "/usr/local/evil/ffmpeg",
])
def test_no_widget_value_survives_the_node_boundary(value):
    """Whatever it says, the answer is "no preference"."""
    assert _helper()(value, "OTR_Probe") == ""


@pytest.mark.parametrize("value", ["ffmpeg", "ffmpeg.exe", "", None, "  "])
def test_the_ordinary_default_is_accepted_silently(value):
    """Every shipped graph carries the bare default; it must not warn."""
    assert _helper()(value, "OTR_Probe") == ""


def test_a_junk_widget_value_does_not_raise():
    """A node must not die because a workflow put nonsense in a text box."""
    for junk in (123, [], {"a": 1}, object()):
        assert _helper()(junk, "OTR_Probe") == ""


def test_the_operator_is_told_once_and_only_once(caplog):
    """An operator who typed a path learns it is dead -- but a soak does not
    print it on every beat."""
    from nodes._otr_shared import ffmpeg as owner
    owner._WIDGET_IGNORED_WARNED.discard("OTR_OnceProbe")
    with caplog.at_level("WARNING"):
        for _ in range(5):
            owner.widget_ffmpeg_is_ignored(HOSTILE, "OTR_OnceProbe")
    said = [r for r in caplog.records if "OTR_OnceProbe" in r.getMessage()]
    assert len(said) == 1, [r.getMessage() for r in said]
    assert "OTR_FFMPEG" in said[0].getMessage(), "must name the real channel"


# --------------------------------------------------------------------------- #
# every node actually calls it -- a helper nobody invokes is not a fix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel,node", sorted(WIDGET_NODES.items()))
def test_every_widget_node_discards_at_its_execute_method(rel, node):
    src = (REPO / rel).read_text(encoding="utf-8")
    assert 'widget_ffmpeg_is_ignored(ffmpeg, "%s")' % node in src, (
        "%s never discards its ffmpeg widget. The value reaches argv[0] "
        "through _ffmpeg_bin/resolve_ffmpeg unless the EXECUTE METHOD drops "
        "it first." % rel)


@pytest.mark.parametrize("rel,node", sorted(WIDGET_NODES.items()))
def test_the_widget_stays_declared_and_stays_a_parameter(rel, node):
    """The field is KEPT on purpose. Removing it would shift `widgets_values`,
    the `inputs` descriptors and every later link `dst_slot` across every
    shipped workflow -- a separate, scheduled migration. It is also what
    tests/test_input_types_signature_parity.py requires."""
    src = (REPO / rel).read_text(encoding="utf-8")
    assert '"ffmpeg": ("STRING"' in src, rel


@pytest.mark.parametrize("rel", sorted(WIDGET_NODES))
def test_the_tooltip_no_longer_advertises_the_removed_power(rel):
    src = (REPO / rel).read_text(encoding="utf-8")
    assert "widget's value if it runs" not in src, (
        "%s still tells the operator the widget picks the binary" % rel)


def test_the_scopes_node_severs_before_BOTH_consumers():
    """`otr_scene_aware_scopes` has TWO doors, and the encoder one was missed
    by the first three drafts of this fix: `:450` probes with
    `resolve_ffprobe(ffmpeg=...)`, and the encode path hands the same value to
    `scope_draw.encode_silent_mp4` -> `find_ffmpeg` -> `resolve_ffmpeg`."""
    src = (REPO / "nodes/otr_scene_aware_scopes.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "render_scopes")
    discarded_at = None
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "widget_ffmpeg_is_ignored"):
            discarded_at = node.lineno
            break
    assert discarded_at, "render_scopes never discards the widget"
    consumers = [n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") in
                 ("resolve_ffprobe", "encode_silent_mp4")]
    assert consumers, "neither consumer found -- has this node been rewired?"
    assert discarded_at < min(consumers), (
        "the widget is discarded at line %d but a consumer runs at line %d"
        % (discarded_at, min(consumers)))


# --------------------------------------------------------------------------- #
# no fallback may reflect raw input back into argv[0]
# --------------------------------------------------------------------------- #
def test_no_resolver_wrapper_reflects_its_argument():
    """The bypass that made the first version of this fix a no-op: a wrapper
    that answered `resolve(x) or x` handed a REJECTED value straight back to
    argv[0]. Two sites did it -- one of them aliased, so a grep for the
    resolver's NAME missed it."""
    import re
    offenders = []
    pattern = re.compile(r"\)\s+or\s+\(?\s*(?:str\()?\s*(cand|ffmpeg|ffprobe)\b")
    for path in (REPO / "nodes").rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8",
                                                errors="replace").splitlines(), 1):
            if "resolve_ff" in line or "_resolve(" in line or "_ffmpeg_bin(" in line:
                if pattern.search(line):
                    offenders.append("%s:%d %s"
                                     % (path.relative_to(REPO), i, line.strip()))
    assert not offenders, "a fallback reflects its argument:\n" + "\n".join(offenders)


def test_the_blend_answers_empty_rather_than_echoing(monkeypatch):
    from nodes import otr_post_upscale_procgen_blend as pu
    from nodes._otr_shared import ffmpeg as ffm
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("OTR_FFMPEG", raising=False)
    monkeypatch.setattr(ffm, "_WINDOWS_INSTALL_CANDIDATES", ())
    for value in (HOSTILE, "ffmpeg", "", "anything at all"):
        assert pu._ffmpeg_bin(value) == "", value


# --------------------------------------------------------------------------- #
# the resolvers answer with an ABSOLUTE path, or nothing
# --------------------------------------------------------------------------- #
def test_a_resolver_never_answers_with_a_relative_name(tmp_path, monkeypatch):
    """A bare answer is spawned relative, and Windows CreateProcess searches
    the cwd -- so `resolve_ffmpeg()` returning the string 'ffmpeg' while a file
    of that name sat beside the server WAS the hazard."""
    from nodes._otr_shared import ffmpeg as ffm
    from nodes._otr_shared import ffprobe as ffp

    real = tmp_path / "real" / "ffmpeg.exe"
    real.parent.mkdir()
    real.write_bytes(b"")
    monkeypatch.setattr(shutil, "which",
                        lambda name: str(real) if "ffmpeg" in name else None)
    monkeypatch.delenv("OTR_FFMPEG", raising=False)
    monkeypatch.setattr(ffm, "_WINDOWS_INSTALL_CANDIDATES", ())

    got = ffm.resolve_ffmpeg()
    assert got is None or os.path.isabs(got), got
    assert ffp._usable("ffmpeg") in (None, str(real))


def test_an_implicit_cwd_hit_is_refused(tmp_path, monkeypatch):
    """THE MECHANISM: CPython inserts the literal `os.curdir` on Windows unless
    `NoDefaultCurrentDirectoryInExePath` is set, so a cwd hit comes back
    RELATIVE while every real PATH directory yields an absolute answer.

    The env var MUST be deleted here: this developer box happens to set it, so
    without the delenv this test passes vacuously and the guard would ship
    unproven (Fable gate, 2026-09-04)."""
    from nodes._otr_shared import ffprobe as ffp
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: r".\ffmpeg.exe")
    got = ffp._which_no_cwd("ffmpeg")
    # It must not TAKE the cwd hit. It MAY still find a real one on PATH --
    # refusing outright would break a box whose only ffmpeg is on PATH the
    # moment a file of that name appeared beside the server.
    assert got != r".\ffmpeg.exe"
    assert got is None or os.path.isabs(got), got
    if got:
        assert os.path.dirname(os.path.abspath(got)) != os.path.abspath(os.getcwd())


def test_a_directory_bearing_relative_path_is_refused(tmp_path, monkeypatch):
    """`bin/ffmpeg` resolves against the process cwd just as a bare name does."""
    from nodes._otr_shared import ffprobe as ffp
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ffmpeg.exe").write_bytes(b"")
    assert ffp._usable(r"bin\ffmpeg.exe") is None
    assert ffp._usable("bin/ffmpeg.exe") is None


def test_an_absolute_path_is_still_honoured(tmp_path):
    """Trusted callers -- an operator pin, a resolved sibling, a Windows
    install dir -- all supply absolute paths, and must keep working."""
    from nodes._otr_shared import ffprobe as ffp
    real = tmp_path / "ffmpeg.exe"
    real.write_bytes(b"")
    assert ffp._usable(str(real)) == str(real)


# --------------------------------------------------------------------------- #
# the filtergraph basename
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("char", [",", ";", ":", "=", "[", "]", "'"])
def test_filtergraph_syntax_in_a_caption_filename_is_refused(char):
    from nodes import otr_caption_burn as cb
    with pytest.raises(ValueError):
        cb._ass_filter_arg("C:\\ep\\bad%sname.ass" % char)


def test_a_backslash_is_rejected_by_the_validator_itself():
    """A backslash cannot reach a BASENAME through a path -- it IS the
    separator, so `Path(...).name` never contains one. It is tested on the
    validator directly, and stays in the reject set because it is ffmpeg's own
    escape character and the validator is shared with the blend node's copy."""
    from nodes import otr_caption_burn as cb
    with pytest.raises(ValueError):
        cb._reject_filtergraph_syntax("bad\\name.ass")


@pytest.mark.parametrize("name", ["episode.ass", "my episode 01.ass",
                                  "ep-01_final.ass", "a.b.ass"])
def test_an_ordinary_caption_filename_still_works(name):
    """Spaces, dots and dashes are legal in filenames and harmless in a graph.
    Every real episode stem is slugified to [a-z0-9_] anyway, so this guard can
    never fire on a normal render."""
    from nodes import otr_caption_burn as cb
    got, _cwd = cb._ass_filter_arg("C:\\ep\\" + name)
    assert got == name


def test_both_copies_of_the_filter_arg_builder_are_guarded():
    """There are TWO `_ass_filter_arg`s and FOUR `ass={name}` interpolations.
    Guarding only the caption node leaves three sites open."""
    for rel in ("nodes/otr_caption_burn.py",
                "nodes/otr_post_upscale_procgen_blend.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "def _ass_filter_arg(" in src, rel
        assert "_reject_filtergraph_syntax(" in src, (
            "%s builds an ass= argument without validating it" % rel)


# --------------------------------------------------------------------------- #
# the no-auth route
# --------------------------------------------------------------------------- #
def test_the_ledger_route_serves_no_wildcard_cors():
    """`GET /otr/latest_ledger` is registered on EVERY install with no
    authentication and answers with the whole ledger. A wildcard is what makes
    that readable cross-origin, so any site visited while ComfyUI runs could
    take it."""
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert '"Access-Control-Allow-Origin": "*"' not in src


def test_the_ledger_route_discloses_no_absolute_path():
    """It used to answer with `fullpath` -- the operator's own directory tree,
    Windows username included -- and with `str(exc)` on failure, which names
    the file it could not open. Both went to an unauthenticated caller
    (2026-09-05)."""
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    i = src.index('@_otr_PromptServer.instance.routes.get("/otr/latest_ledger")')
    handler = src[i:src.index("routes.options", i)]
    assert '"fullpath"' not in handler
    assert '"reason": str(exc)' not in handler
    assert '"filename"' in handler, "the basename still identifies the episode"
