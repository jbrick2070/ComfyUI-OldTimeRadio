"""The TERMINAL FRAME: what a CHAIN successor begins on.

Multi-clip coverage chunk 6c. Segment N+1 needs segment N's last frame
SYNCHRONOUSLY -- inside the render loop, not from a post-episode pass -- or the
chain cannot be a chain.

The extractor decodes the whole clip and lets ``-update 1`` overwrite, rather
than seeking with ``-sseof``. The test that matters is the one that proves it
picks the LAST frame and not merely A frame: the fixture paints a different
grey per frame, so "did we get frame N-1" is checkable rather than assumed.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _ramp_frames(n, value_step=20):
    """``n`` frames, each a FLAT grey one step brighter than the last."""
    import numpy as np
    return [np.full((64, 64, 3), (i + 1) * value_step, dtype=np.uint8)
            for i in range(n)]


# ---------------------------------------------------------------------------
# The pure command builder
# ---------------------------------------------------------------------------

def test_the_command_updates_ONE_output_rather_than_seeking():
    cmd = wb.ffmpeg_terminal_frame_cmd("in.mp4", "out.png")
    assert "-update" in cmd and cmd[cmd.index("-update") + 1] == "1"
    assert "-an" in cmd, "a terminal frame must never carry audio (V-1)"
    assert cmd[-1] == "out.png"
    # A tail SEEK is exactly what this must not do -- it has nothing to land on
    # in a 9-frame segment.
    assert "-sseof" not in cmd


def test_the_command_honours_an_explicit_ffmpeg_binary():
    cmd = wb.ffmpeg_terminal_frame_cmd("in.mp4", "out.png", ffmpeg="/opt/ffmpeg")
    assert cmd[0] == "/opt/ffmpeg"


# ---------------------------------------------------------------------------
# The real extraction
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_it_extracts_the_LAST_frame_not_merely_A_frame():
    """The whole point. Each frame is a different grey; the extracted image
    must be the brightest one."""
    import numpy as np
    from PIL import Image

    frames = _ramp_frames(8)
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "ramp.mp4")
        path, n = wb.encode_frames_to_silent_mp4(frames, clip, 25)
        assert n == 8
        out = os.path.join(tmp, "terminal.png")
        wb.extract_terminal_frame(path, out)
        got = np.asarray(Image.open(out).convert("RGB"))
        # h264 is lossy, so compare to the EXPECTED level with tolerance, and
        # prove it is nearer the last frame than the first.
        mean = float(got.mean())
        assert abs(mean - 160.0) < 12.0, mean          # frame 8 == 8*20
        assert abs(mean - 20.0) > 100.0, "that is the FIRST frame"


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_it_works_on_a_SHORT_clip_where_a_tail_seek_would_miss():
    """A 3-frame segment is well under the one-second tail an ``-sseof -1``
    seek assumes exists."""
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "short.mp4")
        path, _n = wb.encode_frames_to_silent_mp4(_ramp_frames(3, 60), clip, 25)
        out = os.path.join(tmp, "terminal.png")
        wb.extract_terminal_frame(path, out)
        mean = float(np.asarray(Image.open(out).convert("RGB")).mean())
        assert abs(mean - 180.0) < 12.0, mean          # frame 3 == 3*60


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_the_extracted_frame_matches_the_clip_GEOMETRY():
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "geom.mp4")
        path, _n = wb.encode_frames_to_silent_mp4(_ramp_frames(4), clip, 25)
        out = os.path.join(tmp, "terminal.png")
        wb.extract_terminal_frame(path, out)
        assert Image.open(out).size == (64, 64)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")
def test_a_clip_ffmpeg_cannot_read_is_TERMINAL(monkeypatch):
    """The MESSAGE is asserted, not just the type. Without it this passed on a
    box with no ffmpeg at all -- "ffmpeg not found" and "ffmpeg refused this
    input" are the same exception class, so the type alone cannot tell a real
    refusal from a broken install, and an unconditional raise anywhere in
    run_ffmpeg would satisfy it too."""
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "nope.mp4")
        out = os.path.join(tmp, "terminal.png")
        with pytest.raises(wb.GraphExecutionError, match="ffmpeg failed rc="):
            wb.extract_terminal_frame(missing, out)


def test_a_SILENT_SUCCESS_that_wrote_nothing_is_TERMINAL(monkeypatch):
    """ffmpeg exits 0 for an input it decoded zero frames from.

    A 0-byte file handed on as the next segment's init image is a black frame
    at the cut with a clean exit code in front of it -- which is the failure
    mode this build keeps having to remove, not a new one.
    """
    monkeypatch.setattr(wb, "run_ffmpeg", lambda cmd: cmd)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "never_written.png")
        with pytest.raises(wb.GraphExecutionError, match="no usable image"):
            wb.extract_terminal_frame("whatever.mp4", out)


def test_a_ZERO_BYTE_output_is_TERMINAL(monkeypatch):
    monkeypatch.setattr(wb, "run_ffmpeg", lambda cmd: cmd)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "empty.png")
        open(out, "wb").close()
        with pytest.raises(wb.GraphExecutionError, match="0 bytes"):
            wb.extract_terminal_frame("whatever.mp4", out)


# ---------------------------------------------------------------------------
# The in-tree tmp allocator now reserves any suffix, not just .mp4
# ---------------------------------------------------------------------------

def test_the_allocator_reserves_a_png_in_the_same_tier(monkeypatch, tmp_path):
    from nodes import _otr_paths as _paths
    from nodes._otr_video_engines import _tmp as engine_tmp

    shared = tmp_path / "otr" / "episodes" / "_shared" / "tmp"
    monkeypatch.setattr(_paths, "otr_shared_tmp_dir", lambda: shared)
    path = engine_tmp.otr_engine_tmp_path("otr_terminal_", ".png")
    assert path.endswith(".png")
    assert str(shared) in path
    assert not os.path.exists(path), "the allocator hands back a FREE path"


def test_the_mp4_allocator_is_unchanged(monkeypatch, tmp_path):
    from nodes import _otr_paths as _paths
    from nodes._otr_video_engines import _tmp as engine_tmp

    shared = tmp_path / "otr" / "episodes" / "_shared" / "tmp"
    monkeypatch.setattr(_paths, "otr_shared_tmp_dir", lambda: shared)
    path = engine_tmp.otr_engine_tmp_mp4("otr_clip_")
    assert path.endswith(".mp4")
    assert str(shared) in path


# ---------------------------------------------------------------------------
# A5-lite (2026-07-27): the encoder boundary asserts its dtype.
#
# Cut as a LIVE bug -- every producer feeds an exact-size uint8 buffer through
# images_to_uint8, and ffmpeg raises on a short write. The residual this closes
# is a future wider-dtype caller: the rawvideo pipe is 8-bit, so float32 sends
# four times the bytes for the same frames, and the frame_count the encoder
# returns is taken from the ARRAY rather than from what ffmpeg wrote -- so that
# clip would carry a clean receipt describing a length it does not have.
# ---------------------------------------------------------------------------

def test_the_encoder_refuses_a_non_uint8_batch():
    import numpy as np

    frames = np.zeros((4, 8, 8, 3), dtype=np.float32)
    with pytest.raises(wb.GraphExecutionError) as excinfo:
        wb.encode_frames_to_silent_mp4(frames, "unused.mp4", 25)
    message = str(excinfo.value)
    assert "uint8" in message
    assert "float32" in message
    assert "4x the bytes" in message
    assert "images_to_uint8" in message


def test_the_encoder_still_accepts_what_the_real_converter_produces():
    """The refusal must not reject the shipped path. images_to_uint8 is the
    converter every producer goes through, so its OUTPUT is the contract --
    asserted here rather than a hand-made uint8 array, so a change to the
    converter's dtype shows up as a failure instead of a silent divergence."""
    import numpy as np

    torch = pytest.importorskip("torch")
    images = torch.zeros((2, 8, 8, 3), dtype=torch.float32)
    converted = wb.images_to_uint8(images)
    assert converted.dtype == np.uint8
    # Shape/dtype are accepted: the encode itself needs ffmpeg, so drive only
    # the guards by pointing at a path ffmpeg would refuse to write.
    if not _HAS_FFMPEG:
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ok.mp4")
        _path, n = wb.encode_frames_to_silent_mp4(converted, out, 25)
        assert n == 2


# ---------------------------------------------------------------------------
# The frame_count asymmetry (2026-07-27): an adapter that ENCODES a clip must
# also PROVE it.
#
# CanonicalClip.frame_count is documented as "the integer timing authority",
# but it is decode-counted truth only for assembled multi-segment beats -- for
# a single-render beat it is whatever the adapter declared. Four of the seven
# adapters that emit an mp4 never ffprobed what they wrote, so their clips
# reached the ledger entirely self-declared. The two derivations agree today
# only because every producer pipes exact bytes, which is a property of the
# current producers rather than a contract anything enforced.
# ---------------------------------------------------------------------------

#: The source markers that evidence each proof half. The count half accepts
#: EITHER: proven_frame_count reads the muxer's own header off a stream read
#: that already happens, and ffprobe_counted_frames DECODES -- the stronger of
#: the two, and the right one at the assembly boundary where a concatenated
#: container's header can disagree with its own picture data. Requiring only
#: the cheap one would have failed wan_shared for proving MORE.
_ENCODER_PROOF_MARKER = {
    "count": ("proven_frame_count(", "ffprobe_counted_frames("),
    "contract": ("validate_silent_clip_contract(",),
}
#: Entry points that start a child. `popen` is lowercase because the process
#: OWNER (nodes/_otr_shared/proc.py) spells its Popen wrapper that way; without
#: it every engine that migrates to the owner would fall out of this sweep and
#: the encoder inventory would quietly shrink to the un-migrated stragglers.
_SPAWN_CALLS = ("Popen", "popen", "run", "call", "check_call", "check_output")

#: The module that owns the spawn. A migrated engine imports it ALIASED (see
#: the owner's docstring), so the alias is resolved from each module's own
#: imports below, exactly the way `subprocess` already is.
_PROC_OWNER_MODULE = "proc"
_PROC_OWNER_PACKAGE = "_otr_shared"
# KNOWN LIMIT of the sweep below, recorded rather than left to be rediscovered:
# the codec flag is matched as a STRING CONSTANT, so a flag assembled at
# runtime (an f-string, a concatenation, "-c:%s" % stream) is invisible to it,
# as is the stream-index spelling "-c:0". Nothing in the tree does that today.
# An encoder that ever needs to must be pinned in _ENTRY_POINT_PROOFS by hand,
# and the inventory test below is what makes that a visible decision.


def _has_proof(src, half):
    """Does ``src`` CALL one of the accepted proofs for ``half``.

    A substring search was wrong here, and wan_shared.py is the proof: it
    DEFINES both ``ffprobe_counted_frames`` and ``validate_silent_clip_contract``,
    so `def ffprobe_counted_frames(` satisfied a whole-file match on its own.
    The module was excused for proving nothing, on the strength of owning the
    helper -- deleting its real call and its `counted != expect_frames`
    comparison would have left both gates green. Matching CALLS cannot be
    satisfied by a definition, a docstring or a comment."""
    calls = _called_names(src)
    return any(marker.rstrip("(") in calls
               for marker in _ENCODER_PROOF_MARKER[half])


def _function_source(nodes_dir, module, fn_name):
    """The source of ONE function, so a proof claim is checked where it is
    claimed rather than anywhere in the file."""
    import ast

    path = next(nodes_dir.rglob(module))
    src = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == fn_name):
            return ast.get_source_segment(src, node) or ""
    return ""


#: Encoder ENTRY POINTS whose own body already establishes part of the proof,
#: so a caller does not owe that half. VERIFIED against the source below, never
#: merely trusted: an entry that stops proving what it claims here would
#: silently excuse every one of its callers.
_ENTRY_POINT_PROOFS = {
    # Proves the COUNT inside (2026-07-28, 48e3c6fb) -- it returns
    # proven_frame_count(...) rather than len(frames). The colour/stream half
    # is deliberately NOT here: it stays with the adapter, which is the one
    # that knows the engine fps the contract is asserted against.
    ("wrapper_bridge.py", "encode_frames_to_silent_mp4"): {"count"},
    # NOTE run_ffmpeg is deliberately ABSENT: it carries no codec of its own
    # (it drives terminal-frame extraction as readily as a clip encode), so
    # the sweep does not classify it as an encode at all. A caller that builds
    # a still/floor command and hands it to run_ffmpeg is billed for the
    # BUILDER it called -- ffmpeg_still_motion_cmd and friends -- which is the
    # function that actually names the codec.
    # THE SECOND ENCODER (filed 2026-07-28): a hand-rolled Popen wrapper that
    # four live viz_* engines write their clips through. Its callers owe both
    # halves, which is exactly what this file could not see before.
    ("scope_draw.py", "encode_silent_mp4"): set(),
}
#: Encoder entry points that prove their own output in a DIFFERENT dialect, so
#: the marker check cannot see it and the source evidence is pinned instead.
_SELF_PROVING_ENTRY_POINTS = {
    # Re-probes the file it just wrote, refuses on a failed audio strip, and
    # measures actual_duration_s from the OUTPUT rather than the provider's
    # claim. A cloud clip is not piped from an array, so there is no declared
    # count for a count proof to check against.
    ("cloud_media_canonical.py", "canonicalize_video"):
        "post = _ffprobe_streams(",
    # The 2026-08-29 beat-clip audio mux. Its VIDEO stream is `-c:v copy` --
    # a verbatim stream copy of a beat mp4 whose colour contract and frame
    # count `assemble_beat_segments` proved moments earlier, so a re-encode
    # proof would re-prove unchanged bytes. What it CAN change -- the stream
    # census and the container timing -- it proves itself, in its own
    # dialect: it probes the file it just wrote, refuses unless the census is
    # exactly one video plus one audio stream, and refuses when the two
    # durations disagree beyond one frame plus AAC granularity. It also swaps
    # in via a temp sibling, so a failed proof leaves the proven original.
    ("foley_stems.py", "mux_native_audio_into_beat_clip"):
        "doc = _fp.probe_json(",
}


def _clip_encoder_entry_points(nodes_dir):
    """Every function under ``nodes/`` that BUILDS AND RUNS an ffmpeg ENCODE.

    Structural, not a call-spelling list -- which is the whole point of this
    rewrite. The previous version grepped for two literal substrings,
    ``encode_frames_to_silent_mp4(`` and ``run_ffmpeg(``, so it was blind to
    any third way of spelling the same act. And there was one: four live
    engines (viz_green / viz_camera / viz_mxc_mandala / viz_mxc_cpu) write
    their clips through ``nodes/_otr_shared/scope_draw.py::encode_silent_mp4``,
    a spelling on neither list, so ``encoders - provers`` was vacuously empty
    over all four.

    The marker used instead is the ffmpeg CLI itself: a command that WRITES
    video must name a video codec (``-c:v`` / ``-vcodec`` / ``-codec:v``), and
    a command that only reads -- ffprobe, a duration probe, git -- never does.
    A new encoder cannot dodge that and still be an encoder. The flag is
    chased from the SPAWNING function through same-module calls, because it is
    routinely built a helper or two away (wrapper_bridge spawns in
    ``encode_frames_to_silent_mp4``, assembles in ``ffmpeg_silent_mp4_cmd`` and
    holds the flag in ``_bt709_encode_args``), and the result is reported per
    spawning FUNCTION -- the granularity a caller actually calls.
    """
    import ast

    entry_points = {}
    for path in sorted(nodes_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Whatever names THIS module binds subprocess to -- resolved from its
        # own imports rather than guessed. The first draft tested
        # `"sp" in name`, which is false for "subprocess", so the inventory
        # came back empty and BOTH gates passed vacuously. A hard-coded alias
        # list would have been one more spelling to be blind to. ImportFrom is
        # read too (`from subprocess import Popen` binds a BARE name); no file
        # under nodes/ does that today, and that is precisely why it has to be
        # covered now rather than after the first one does.
        # The process OWNER is resolved the same way and for the same reason:
        # it is imported at four different depths (`from . import proc as
        # otr_proc` inside _otr_shared, `from .._otr_shared import ...` in an
        # engine subpackage, `from ._otr_shared import ...` in a top-level node,
        # and the flat `from _otr_shared import ...` every test path takes), and
        # a hard-coded `otr_proc` would go blind the first time one of those is
        # spelled differently. `module` never carries the leading dot, so the
        # relative forms are told apart by `level`.
        mod_aliases, bare_spawn_names = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name == "subprocess":
                        mod_aliases.add(imported.asname or "subprocess")
                    elif imported.name == _PROC_OWNER_MODULE:
                        mod_aliases.add(imported.asname or _PROC_OWNER_MODULE)
            elif isinstance(node, ast.ImportFrom) and node.module == "subprocess":
                for imported in node.names:
                    if imported.name in _SPAWN_CALLS:
                        bare_spawn_names.add(imported.asname or imported.name)
            elif isinstance(node, ast.ImportFrom) and (
                    (node.module or "").endswith(_PROC_OWNER_PACKAGE)
                    # `from . import proc` means the owner ONLY from inside the
                    # owner's own package. Anywhere else the same line would
                    # name some other sibling module that happens to be called
                    # `proc`, and crediting it here would be the sweep going
                    # blind in the opposite direction.
                    or (node.module is None and node.level
                        and path.parent.name == _PROC_OWNER_PACKAGE)):
                for imported in node.names:
                    if imported.name == _PROC_OWNER_MODULE:
                        mod_aliases.add(imported.asname or _PROC_OWNER_MODULE)
        # NO early `continue` here. A module can BUILD an ffmpeg encode command
        # without spawning anything -- that is what wrapper_bridge's
        # ffmpeg_*_cmd builders do -- and skipping the whole file for want of
        # an `import subprocess` would make the builder invisible AND silently
        # un-bill every engine that calls it. That is the original blindness
        # in a new costume: split "build the argv" and "run it" across two
        # files and the gate stops seeing either.
        own_strings, own_calls = {}, {}
        spawners, returners = set(), set()
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            strings, calls = set(), set()
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    returners.add(fn.name)
                elif (isinstance(sub, ast.Constant)
                        and isinstance(sub.value, str)):
                    strings.add(sub.value)
                elif isinstance(sub, ast.Call):
                    func = sub.func
                    if isinstance(func, ast.Attribute):
                        calls.add(func.attr)
                        if (func.attr in _SPAWN_CALLS
                                and isinstance(func.value, ast.Name)
                                and func.value.id in mod_aliases):
                            spawners.add(fn.name)
                    elif isinstance(func, ast.Name):
                        calls.add(func.id)
                        if func.id in bare_spawn_names:
                            spawners.add(fn.name)
            own_strings[fn.name] = strings
            own_calls[fn.name] = calls

        # An encode NAMES A VIDEO CODEC, and the flag is routinely built one
        # or two helpers away from the spawn -- wrapper_bridge spawns in
        # encode_frames_to_silent_mp4, assembles the arg list in
        # ffmpeg_silent_mp4_cmd and holds the flag in _bt709_encode_args. So
        # the flag is chased transitively through SAME-MODULE calls from the
        # spawning function. Classifying per module instead over-collects
        # badly: it credited cloud_media_canonical's AUDIO extraction and
        # every has_nvenc probe as clip encoders.
        def _reaches_a_codec(start):
            seen, stack = set(), [start]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                if own_strings.get(cur, set()) & {"-c:v", "-vcodec",
                                                  "-codec:v"}:
                    return True
                stack.extend(own_calls.get(cur, set()) & own_strings.keys())
            return False

        # Both ways of reaching an encode count, because an engine can do
        # either: RUN the encode itself (a SPAWNER), or BUILD the encode
        # command and hand it to a generic runner (a RETURNER -- it hands its
        # caller the argv). cheap_families does the second: it assembles
        # ffmpeg_still_motion_cmd / _still_static_cmd / _lavfi_floor_cmd and
        # passes them to run_ffmpeg, which carries no codec of its own.
        # Counting spawners alone would have dropped the four still_* engines
        # this gate was widened to catch on 2026-07-27 -- a NARROWING wearing
        # a widening's name. "Returns a value" is the structural property; the
        # first draft tested `name.endswith("_cmd")`, which is a house naming
        # convention and so just another spelling to be blind to.
        for fn_name in sorted(set(spawners) | set(returners)):
            if _reaches_a_codec(fn_name):
                entry_points.setdefault(path.name, set()).add(fn_name)
    return entry_points


def _called_names(src):
    """Every function name this source actually CALLS, by AST.

    Substring matching was tried first and was wrong in both directions: a
    module that merely DEFINES ``__enter__`` contains the text ``__enter__(``
    and was billed for a clip it never writes.

    Dedented first so this works on a single function's source segment as
    readily as on a whole module."""
    import ast
    import textwrap

    names = set()
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.add(node.func.id)
    return names


def _proof_debts(engine_src, engine_name, entry_points, half):
    """Which encoder entry points ``engine_src`` calls that leave it owing
    ``half``. Empty when the engine writes no clip, or when every encoder it
    reaches already establishes that half."""
    calls = _called_names(engine_src)
    owed = set()
    for module, fns in entry_points.items():
        if module == engine_name:
            continue                       # the encoder itself, not a client
        for fn in fns:
            if fn not in calls:
                continue
            if (module, fn) in _SELF_PROVING_ENTRY_POINTS:
                continue
            if half in _ENTRY_POINT_PROOFS.get((module, fn), set()):
                continue
            owed.add("%s::%s" % (module, fn))
    return owed


def test_the_encoder_inventory_is_derived_and_still_proves_what_it_claims():
    """The inventory of clip encoders is derived from source, and every claim
    made about one is checked against that source.

    This is the half that keeps the roster gate below from going confidently
    blind again. ``_ENTRY_POINT_PROOFS`` excuses a caller from a proof half on
    the grounds that the encoder already establishes it; if that ever stops
    being true, every caller is silently excused."""
    import pathlib

    nodes_dir = pathlib.Path(__file__).resolve().parents[1] / "nodes"
    found = _clip_encoder_entry_points(nodes_dir)
    assert found, "found no clip encoders at all -- the sweep is broken"
    # The second encoder must be VISIBLE to the sweep. This is the exact
    # blindness being closed, so it is asserted rather than assumed.
    assert "encode_silent_mp4" in found.get("scope_draw.py", set()), (
        "the sweep no longer sees scope_draw.py::encode_silent_mp4 -- the "
        "second encoder is invisible again")
    assert "encode_frames_to_silent_mp4" in found.get("wrapper_bridge.py",
                                                      set())

    for (module, fn), halves in _ENTRY_POINT_PROOFS.items():
        assert fn in found.get(module, set()), (
            "%s::%s is pinned as an encoder entry point but the sweep no "
            "longer finds it there" % (module, fn))
        for half in halves:
            # Scoped to the CREDITED FUNCTION, not the whole file. A file-wide
            # substring would let the count proof be credited to
            # ffmpeg_silent_mp4_cmd -- a pure argv builder that never calls it
            # -- because the marker exists elsewhere in wrapper_bridge.py.
            assert _has_proof(_function_source(nodes_dir, module, fn), half), (
                "%s::%s is credited with the %s proof, which excuses every "
                "one of its callers from it, and none of %r is in THAT "
                "function" % (module, fn, half, _ENCODER_PROOF_MARKER[half]))
    for (module, fn), evidence in _SELF_PROVING_ENTRY_POINTS.items():
        # The membership assertion its sibling loop has. Without it, an
        # encoder that stops being FOUND keeps its exemption: the evidence
        # string survives an ffmpeg call moving to another file, so this test
        # stays green while every caller silently stops being billed.
        # eng_cloud_video.py has no proof of its own and depends entirely on
        # this exemption, so that is not hypothetical.
        assert fn in found.get(module, set()), (
            "%s::%s is exempted as self-proving but the sweep no longer finds "
            "it -- its callers are now unbilled AND unchecked" % (module, fn))
        assert evidence in _function_source(nodes_dir, module, fn), (
            "%s::%s is credited with proving its own output via %r, which "
            "excuses its callers, and that evidence is gone"
            % (module, fn, evidence))


def test_every_adapter_that_writes_a_clip_proves_its_COLOUR_CONTRACT():
    """An engine that writes a clip must prove the clip's colour/stream
    contract -- at itself, or inside the encoder it calls.

    Derived from the SOURCE, not from a hand-kept list. The bug report named
    two adapters (eng_humo, eng_ltx_av) and listed eng_ltx_video among those
    that already probed; sweeping every encoder found eng_ltx_video did NOT, on
    either of its two recipe paths, and that eng_still_parallax was missing
    from the report altogether. A later widening caught cheap_families.py,
    which is not named eng_* and builds its mp4 from an ffmpeg arg list. This
    version stops asking HOW the clip was written at all -- it asks who reaches
    an ffmpeg encode, by any spelling."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    entry_points = _clip_encoder_entry_points(root / "nodes")
    engines = root / "nodes" / "_otr_video_engines"

    unproven, billed = {}, set()
    for path in sorted(engines.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        owed = _proof_debts(src, path.name, entry_points, "contract")
        if not owed:
            continue
        billed.add(path.name)
        if not _has_proof(src, "contract"):
            unproven[path.name] = sorted(owed)

    # WITHOUT THIS the test passes vacuously the day the roster stops billing
    # anyone -- `unproven == {}` is satisfied just as well by "no engine writes
    # a clip" as by "every engine proves it", and that is the precise failure
    # this whole file is a response to. These four reach the encode by three
    # different routes: a viz engine through the second encoder, an ltx
    # adapter through the first, and cheap_families by building the argv
    # itself and handing it to a generic runner.
    for expected in ("eng_viz_camera.py", "eng_visualizer.py",
                     "eng_ltx_video.py", "cheap_families.py"):
        assert expected in billed, (
            "%s writes a clip and the roster no longer bills it for the "
            "colour contract -- the gate has gone blind again, not clean"
            % expected)

    assert unproven == {}, (
        "these engines write a clip through an ffmpeg encode and never prove "
        "the clip's colour/stream contract: %r" % (unproven,))


def test_DEFINING_a_proof_helper_is_not_the_same_as_calling_it():
    """The gate's own logic, pinned.

    ``wan_shared.py`` owns both proof helpers, so the first version of
    ``_has_proof`` -- a whole-file substring search -- was satisfied by their
    ``def`` lines alone: the one module that could regress its real
    ``counted != expect_frames`` comparison was also the one module the gates
    could not notice it in. Found by the pre-push QA fan-out."""
    assert not _has_proof("def ffprobe_counted_frames(path):\n    return 0\n",
                          "count")
    assert not _has_proof("# calls ffprobe_counted_frames( eventually\n",
                          "count")
    assert _has_proof("def go(p):\n    return ffprobe_counted_frames(p)\n",
                      "count")
    assert _has_proof("def go(p):\n    return proven_frame_count(p, 3)\n",
                      "count")
    assert not _has_proof(
        "def validate_silent_clip_contract(fields, fps):\n    return None\n",
        "contract")


def test_every_adapter_that_writes_a_clip_proves_its_FRAME_COUNT():
    """The other half. An engine that writes a clip must prove how many frames
    the FILE holds -- at itself, or inside the encoder it calls.

    Landed one chunk after the colour half, and strictly on top of it: the
    roster is the same, the rule is the same, only the debt is new. Splitting
    them that way is what let the colour gate land GREEN while this one was
    still red on cheap_families -- a gate is allowed to arrive later, never to
    be narrowed so it arrives clean.

    ``frame_count`` is documented as the integer timing authority the
    composite positions a beat by, so an engine that declares it without ever
    reading the file is asserting a timing fact about a clip it never
    measured."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    entry_points = _clip_encoder_entry_points(root / "nodes")
    engines = root / "nodes" / "_otr_video_engines"

    unproven, billed = {}, set()
    for path in sorted(engines.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        owed = _proof_debts(src, path.name, entry_points, "count")
        if not owed:
            continue
        billed.add(path.name)
        if not _has_proof(src, "count"):
            unproven[path.name] = sorted(owed)

    # Same anti-vacuity guard as the colour gate: `unproven == {}` is equally
    # satisfied by "nobody writes a clip".
    for expected in ("eng_viz_camera.py", "eng_visualizer.py",
                     "cheap_families.py", "wan_shared.py"):
        assert expected in billed, (
            "%s writes a clip and the roster no longer bills it for the frame "
            "count -- the gate has gone blind again, not clean" % expected)

    assert unproven == {}, (
        "these engines write a clip through an ffmpeg encode and never prove "
        "how many frames it holds: %r" % (unproven,))


def test_the_proof_runs_AFTER_the_encode_in_every_adapter():
    """Order matters: ffprobing before the file is written would raise on a
    path that does not exist yet, which is a different failure wearing the
    same name. Pinned per encode site, not per file.

    The pairing is asserted in BOTH directions, and the second direction was
    added because the mutation round caught the first version of this test
    being decorative: checking only that some proof FOLLOWS each encode stays
    green when a proof is inserted BEFORE one, which is the exact ordering
    defect the test is named for."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    entry_points = _clip_encoder_entry_points(root / "nodes")
    engines = root / "nodes" / "_otr_video_engines"

    # The encode spellings come from the DERIVED roster, not from one literal.
    # This test used to regex `encode_frames_to_silent_mp4\(` alone -- so it
    # was blind to every engine that writes through the SECOND encoder, which
    # is the whole family the file above was rewritten to see. It would have
    # stayed green with the proof moved BEFORE the encode in all four viz_*
    # engines: the exact defect it is named for, in the exact place this
    # chunk was working. Found by the pre-push QA fan-out, twice, independently.
    checked, files = 0, set()
    for path in sorted(engines.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if path.name in entry_points:
            continue                       # the encoder itself, not a client
        # Exactly the encode sites this engine is BILLED for -- reusing the
        # same debt calculation the contract gate uses, so the two cannot
        # disagree about who writes a clip. Sweeping every entry point in the
        # tree instead would demand a contract proof from the cloud adapters,
        # whose encoder proves its own output in its own dialect.
        owed = _proof_debts(src, path.name, entry_points, "contract")
        if not owed:
            continue
        spellings = sorted({billed.split("::")[1] for billed in owed})
        encode_re = re.compile(r"\b(?:%s)\("
                               % "|".join(re.escape(s) for s in spellings))
        encodes = [m.start() for m in encode_re.finditer(src)]
        proofs = [m.start() for m
                  in re.finditer(r"validate_silent_clip_contract\(", src)]
        assert encodes, (
            "%s is billed for %r but no call site was found -- the ordering "
            "sweep and the debt calculation disagree" % (path.name, owed))
        files.add(path.name)
        assert len(proofs) >= 1, (
            "%s: %d encode site(s) and no proof at all"
            % (path.name, len(encodes)))
        for encode_at in encodes:
            assert any(p > encode_at for p in proofs), (
                "%s: an encode at offset %d has no proof after it"
                % (path.name, encode_at))
            checked += 1
        for proof_at in proofs:
            assert any(e < proof_at for e in encodes), (
                "%s: a proof at offset %d runs before any encode -- it would "
                "ffprobe a path that does not exist yet"
                % (path.name, proof_at))
    # Named, not counted: a bare `checked >= 7` goes quietly green when the
    # sweep stops seeing a whole family and some other file makes up the number.
    for expected in ("eng_visualizer.py", "eng_viz_camera.py",
                     "eng_viz_mandala.py", "eng_viz_rainbow.py",
                     "eng_humo.py", "cheap_families.py"):
        assert expected in files, (
            "%s writes a clip and this ordering proof no longer covers it"
            % expected)
    assert checked >= 7, checked
