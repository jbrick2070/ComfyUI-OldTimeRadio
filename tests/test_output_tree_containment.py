"""tests/test_output_tree_containment.py

THE REGISTRY PUNCH LIST (2026-09-05, GO_FORWARD item 2.9).

A node declares a path in ``INPUT_TYPES``; ComfyUI's unauthenticated
``POST /prompt`` sets any literal for it (``forceInput`` is a frontend hint the
server never reads, and a default value is irrelevant because the caller sends
their own). Before this change set the only guard on those values was
``reject_remote_path``, which refuses a UNC/URL spelling and passes every
ordinary local absolute path -- so a caller chose an arbitrary destination and
ffmpeg/``shutil``/``os.replace`` wrote there. The Comfy Registry bans exactly
that shape as ``policy-v0.2: PATH_TRAVERSAL`` (35 bans in the surveyed corpus).

These pin the containment itself and the two token whitelists. The node-level
call sites are pinned by source assertion rather than by executing a render:
the point is that the guard is CALLED at the execute method, and a full render
needs ffmpeg and a GPU.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from nodes import _otr_paths as P


# --------------------------------------------------------------------------
# the helper
# --------------------------------------------------------------------------
class TestConfineToOutputTree:
    def test_in_tree_path_is_accepted_and_returned_resolved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
        good = tmp_path / "otr" / "episodes" / "ep1" / "final.mp4"
        assert P.confine_to_output_tree(str(good), "output_path") == str(good.resolve())

    def test_empty_is_passed_through_because_it_means_use_the_default(self):
        assert P.confine_to_output_tree("", "output_path") == ""
        assert P.confine_to_output_tree(None, "output_path") == ""

    def test_an_arbitrary_local_absolute_path_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv("OTR_EXTRA_OUTPUT_ROOTS", "")  # conftest declares the tmp base
        outside = tmp_path.parent / "not_the_output_tree" / "evil.mp4"
        with pytest.raises(P.OtrPathContractError):
            P.confine_to_output_tree(str(outside), "output_path")

    def test_a_traversal_segment_is_refused_by_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
        with pytest.raises(P.OtrPathContractError, match="traversal"):
            P.confine_to_output_tree(str(tmp_path / ".." / "evil.mp4"), "output_path")
        with pytest.raises(P.OtrPathContractError, match="traversal"):
            P.confine_to_output_tree("../../evil.mp4", "output_path")

    def test_the_obs_root_is_permitted_because_the_operator_declared_it(
            self, tmp_path, monkeypatch):
        """`$OTR_OBS_DIR` may point anywhere; the mux publishes there BY DESIGN,
        so containment that refused it would break the success signal."""
        monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "out"))
        obs = tmp_path / "elsewhere" / "obs"
        monkeypatch.setenv("OTR_OBS_DIR", str(obs))
        assert P.confine_to_output_tree(str(obs / "ep_procgen_blended.mp4"), "output_path")

    def test_comfyui_own_input_dir_is_permitted(self, monkeypatch):
        """A graph that captions a video sitting in ComfyUI's input dir puts its
        output beside it, and this pack stages engine assets there itself.
        Permitting it hands an attacker nothing: core ComfyUI's own
        unauthenticated POST /upload/image already writes into that directory."""
        target = P.comfy_input_dir() / "some_clip_captioned.mp4"
        assert P.confine_to_output_tree(str(target), "output_path")

    def test_the_env_allowlist_widens_it(self, tmp_path, monkeypatch):
        """An env var is OPERATOR configuration -- a /prompt body cannot set one --
        so this widens the contract without widening the attack surface."""
        monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "out"))
        monkeypatch.setenv("OTR_EXTRA_OUTPUT_ROOTS", "")  # conftest declares the tmp base
        extra = tmp_path / "extra"
        target = extra / "x.mp4"
        with pytest.raises(P.OtrPathContractError):
            P.confine_to_output_tree(str(target), "output_path")
        monkeypatch.setenv("OTR_EXTRA_OUTPUT_ROOTS", str(extra))
        assert P.confine_to_output_tree(str(target), "output_path") == str(target.resolve())

    def test_the_allowlist_takes_several_roots(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "out"))
        a, b = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv("OTR_EXTRA_OUTPUT_ROOTS", "%s;%s" % (a, b))
        assert P.confine_to_output_tree(str(b / "x.mp4"), "output_path")

    def test_the_root_itself_is_permitted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
        assert P.confine_to_output_tree(str(tmp_path), "output_dir")

    def test_the_field_name_is_in_the_message(self, tmp_path, monkeypatch):
        """The operator reads the log, not the traceback."""
        monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
        monkeypatch.setenv("OTR_EXTRA_OUTPUT_ROOTS", "")  # conftest declares the tmp base
        with pytest.raises(P.OtrPathContractError, match="silent_video_path"):
            P.confine_to_output_tree(str(tmp_path.parent / "x.mp4"), "silent_video_path")

    def test_it_refuses_a_remote_path_itself_before_resolving(self):
        """`Path(...).resolve()` OPENS an SMB session for a UNC value -- measured
        at 42 seconds against an unreachable host, and an NTLM handshake against
        a reachable one. Callers refuse first, but this function must not
        perform the coercion it was written to sit beside if one forgets."""
        B = chr(92)  # built, not escaped: a backslash literal in a test is a trap
        for value in (B + B + "attacker" + B + "share" + B + "x.mp4",
                      "//attacker/share/x.mp4",
                      B + B + "?" + B + "C:" + B + "x.mp4",
                      "smb://attacker/share/x.mp4"):
            with pytest.raises(P.OtrPathContractError) as got:
                P.confine_to_output_tree(value, "output_path")
            assert "local path" in str(got.value), (
                "%r reached the resolver instead of the remote refusal" % value)

    def test_it_is_exported(self):
        assert "confine_to_output_tree" in P.__all__
        assert "confine_to_output_tree_paths" in P.__all__


# --------------------------------------------------------------------------
# the call sites -- source pins (a render needs ffmpeg and a GPU)
# --------------------------------------------------------------------------
_NODES = Path(__file__).resolve().parents[1] / "nodes"

#: node file -> (the expression the guard is called on, the widgets it covers).
#: THE GUARD SITS ON THE COMPUTED DESTINATION, not on the raw widget: a call
#: that passes through without writing must not be refused, and the empty-widget
#: case (where a `_default_out` derives the destination from an INPUT path) has
#: to be covered by the same line.
CONFINED_DESTINATIONS = {
    "otr_caption_burn.py": ("confine_to_output_tree(out,", ("output_path", "video_path")),
    "otr_master_audio_mux.py": ("confine_to_output_tree(out,", ("output_path",)),
    "otr_silent_composite.py": ("confine_to_output_tree(out,", ("output_path", "base_video_path")),
    "otr_post_upscale_procgen_blend.py": (
        "confine_to_output_tree(str(output_path),", ("source_mp4_path",)),
    "otr_credits_roll.py": ("confine_to_output_tree(out,", ("video_path",)),
    # scene_sequencer is NOT here on purpose: its `output_dir` sink was DELETED
    # rather than confined. See the two tests below.
}


def test_the_sequencer_creates_no_caller_named_directory():
    """`os.makedirs(output_dir)` on a workflow STRING let an unauthenticated
    caller create directory trees anywhere. Confining it would have been the
    wrong shape: the value is inert (nothing below reads it), so a guard could
    only ever REFUSE and kill a render. The sink is gone instead."""
    src = (_NODES / "scene_sequencer.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
    assert "makedirs(output_dir" not in code
    for line in code.split("\n"):
        assert "output_dir" not in line or "def sequence" in line or '"output_dir"' in line \
            or "output_dir=DEFAULT_OUT" in line or "DEFAULT_OUT" in line, \
            "output_dir gained a live use again: %s" % line.strip()


def test_no_shipped_default_hardcodes_this_developers_home():
    """A default that is only correct on the machine it was written on is a
    portability defect. `scene_sequencer.DEFAULT_OUT` was
    `~/Documents/ComfyUI/output/otr/audio` -- wrong on a registry install, on
    the 8 GB box, and on any two-tree split, and invisible to a suite that runs
    where the guess happens to be true."""
    from nodes import scene_sequencer
    assert scene_sequencer.DEFAULT_OUT == ""
    for name in ("scene_sequencer.py", "otr_caption_burn.py", "otr_master_audio_mux.py",
                 "otr_silent_composite.py", "otr_credits_roll.py",
                 "otr_post_upscale_procgen_blend.py"):
        src = (_NODES / name).read_text(encoding="utf-8")
        code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
        assert 'expanduser("~")' not in code, (
            "%s builds a shipped path from this machine's home directory" % name)


@pytest.mark.parametrize("filename,spec", sorted(CONFINED_DESTINATIONS.items()))
def test_every_write_destination_is_confined(filename, spec):
    call, fields = spec
    src = (_NODES / filename).read_text(encoding="utf-8")
    assert call in src, (
        "%s writes to a caller-steered location; the containment call %r is "
        "missing (or moved off the computed destination)" % (filename, call))
    for field in fields:
        assert field in src, "%s: %s is gone -- re-check the guard" % (filename, field)


def test_the_guard_runs_after_the_remote_refusal_not_before():
    """Resolving first would launder a UNC spelling into a local-looking path,
    so `reject_remote_path` has to see the value as typed."""
    for filename in ("otr_caption_burn.py", "otr_master_audio_mux.py",
                     "otr_silent_composite.py", "otr_post_upscale_procgen_blend.py",
                     "otr_credits_roll.py"):
        src = (_NODES / filename).read_text(encoding="utf-8")
        assert src.index("reject_remote_paths(") < src.index("confine_to_output_tree("), (
            "%s confines before it refuses a remote path" % filename)


def test_the_blend_rejects_a_remote_scopes_path():
    """`scopes_mp4_path` was missing from the node's own remote-refusal list
    while being resolved and statted four lines later."""
    src = (_NODES / "otr_post_upscale_procgen_blend.py").read_text(encoding="utf-8")
    i = src.index("reject_remote_paths(")
    assert "scopes_mp4_path" in src[i:i + 400]


def test_the_render_batch_rejects_remote_paths_at_all():
    """This node had no remote refusal whatsoever while staging three
    caller-named paths into ComfyUI's input dir."""
    src = (_NODES / "otr_video_render_batch.py").read_text(encoding="utf-8")
    assert "reject_remote_paths(" in src
    for field in ("portrait_path", "audio_path", "master_audio_path"):
        assert field in src


# --------------------------------------------------------------------------
# token whitelists: a caller-supplied label may name a FILE, never a LOCATION
# --------------------------------------------------------------------------
def test_the_scopes_episode_id_becomes_a_safe_filename_token():
    src = (_NODES / "otr_scene_aware_scopes.py").read_text(encoding="utf-8")
    i = src.index('key = ')
    window = src[i:i + 400]
    assert "_re.sub" in window and "A-Za-z0-9_.-" in window, (
        "the manifest's episode_id is joined into a filename and must be "
        "reduced to a token first")


def test_the_render_batch_engine_becomes_a_safe_filename_token():
    src = (_NODES / "otr_video_render_batch.py").read_text(encoding="utf-8")
    # Anchor on the ASSIGNMENT, not the bare literal -- the comment above it
    # quotes the old form, so the literal appears twice.
    i = src.index('name = "node_single_%s.json"')
    window = src[i:i + 300]
    assert "_re.sub" in window and "A-Za-z0-9_.-" in window, (
        "the engine label is joined into a report filename and must be "
        "reduced to a token first")


def test_a_profile_id_names_a_file_not_a_path():
    from nodes._otr_shared import capability_profiles as cp
    for bad in ("../../../etc/hosts", r"..\..\x", "a/b", "a\\b"):
        with pytest.raises(cp.ProfileError, match="names a file"):
            cp.load_profile(bad)


# --------------------------------------------------------------------------
# the arbitrary-file-read sibling the earlier fix missed
# --------------------------------------------------------------------------
class TestTrustedStillSource:
    """`render_driver._still_spine_materialize_row` copied a ledger-carried
    `pool_path` into a `/view`-served directory with only an `isfile` check --
    the sibling of the chain closed in the image dispatcher."""

    def _rd(self):
        from nodes._otr_video_engines import render_driver as rd
        return rd

    def _png(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 120)
        return path

    def test_a_still_inside_the_otr_tree_is_trusted(self, tmp_path):
        rd = self._rd()
        stills = tmp_path / "otr" / "episodes" / "ep1" / "stills"
        src = self._png(tmp_path / "otr" / "pool" / "a.png")
        assert rd._trusted_still_source(str(src), str(stills)) is True

    def test_a_file_outside_the_tree_is_refused(self, tmp_path):
        rd = self._rd()
        stills = tmp_path / "otr" / "episodes" / "ep1" / "stills"
        outside = self._png(tmp_path / "elsewhere" / "secret.png")
        assert rd._trusted_still_source(str(outside), str(stills)) is False

    def test_a_non_png_inside_the_tree_is_refused(self, tmp_path):
        rd = self._rd()
        stills = tmp_path / "otr" / "episodes" / "ep1" / "stills"
        doc = tmp_path / "otr" / "pool" / "not_an_image.txt"
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_bytes(b"secrets, but long enough to pass the size floor" * 4)
        assert rd._trusted_still_source(str(doc), str(stills)) is False

    def test_a_remote_path_is_refused_without_statting_it(self, tmp_path):
        rd = self._rd()
        stills = tmp_path / "otr" / "episodes" / "ep1" / "stills"
        assert rd._trusted_still_source(r"\\attacker\share\x.png", str(stills)) is False

    def test_the_probe_never_raises(self, tmp_path):
        rd = self._rd()
        stills = tmp_path / "otr" / "episodes" / "ep1" / "stills"
        for junk in ("", None, 0, "\x00", "?" * 400):
            assert rd._trusted_still_source(junk, str(stills)) is False

    def test_the_materializer_uses_the_trust_probe(self):
        src = (_NODES / "_otr_video_engines" / "render_driver.py").read_text(encoding="utf-8")
        i = src.index("def _still_spine_materialize_row")
        window = src[i:i + 2000]
        assert "_trusted_still_source(" in window, (
            "the copier must gate its source, not merely check isfile()")


# --------------------------------------------------------------------------
# the ledger a voice run rewrites is the one it opened
# --------------------------------------------------------------------------
def test_the_voice_stamp_prefers_the_in_flight_ledger():
    """`meta.paths.ledger_path` arrives in the workflow's own JSON and reached
    `save_ledger_safe` -> `os.replace`, i.e. an arbitrary JSON overwrite."""
    src = (_NODES / "_otr_voice_node_common.py").read_text(encoding="utf-8")
    i = src.index("def _persist_ledger_stamps")
    window = src[i:i + 3000]
    assert "in_flight_ledger_path()" in window
    assert "confine_to_output_tree" in window


# --------------------------------------------------------------------------
# the routes
# --------------------------------------------------------------------------
def test_no_post_route_ships():
    """Both POST render-harness routes are removed. They read caller-supplied
    paths from an unauthenticated body and started a background GPU render --
    the registry's second-largest ban class -- and nothing shipped called them."""
    src = (Path(__file__).resolve().parents[1] / "__init__.py").read_text(encoding="utf-8")
    assert "routes.post" not in src
    assert src.count("routes.get") == 1, "only the read-only ledger GET should remain"


def test_the_composite_fingerprint_hook_checks_every_step():
    """Steps 1 and 2 got the remote check when the hook was fixed; step 3 (the
    sibling master WAV listdir) did not, on the very file that commit named."""
    src = (_NODES / "otr_silent_composite.py").read_text(encoding="utf-8")
    i = src.index("# 3. Sibling master WAV")
    window = src[i:i + 800]
    assert "_is_remote(base_video_path)" in window
    assert "os.listdir" in window, "anchor drifted: this step should still listdir"


# --------------------------------------------------------------------------
# the legacy voice reference -- a UNC path arriving inside a JSON document
# --------------------------------------------------------------------------
def test_a_remote_voice_reference_is_refused_before_it_is_statted():
    """A cast row with no `voice_route` is accepted as a LEGACY reference
    (`_otr_voice_route.py:1074`), and `resolve_voice_ref_path` passes an
    already-absolute value straight through (`base.py:140`) -- so a
    `ledger_json` naming a UNC share reached `os.path.exists` at eleven call
    sites and Windows authenticated to the host the workflow chose. Found
    2026-09-05 by GPT-6 Astra; the same coercion class `79dc9828` closed at the
    ffmpeg nodes, missed here because the value arrives inside JSON rather than
    as a widget."""
    from nodes._otr_voice_node_common import _resolve_ref_to_disk

    b = chr(92)                       # built, never escaped
    assert _resolve_ref_to_disk(b + b + "attacker" + b + "share" + b + "x.wav") is None
    assert _resolve_ref_to_disk("//attacker/share/x.wav") is None
    assert _resolve_ref_to_disk("") is None
    # ...and the legitimate shapes still resolve. A mapped drive is a drive
    # letter, not a UNC spelling, so the operator's transfer drive keeps working.
    assert _resolve_ref_to_disk("models/TTS/refs/indextts2/ix_male_warm.wav")
    assert _resolve_ref_to_disk("U:" + b + "refs" + b + "x.wav")


def test_the_guard_lives_in_the_one_resolver():
    """Eleven call sites reach this resolver and every one already treats None
    as 'asked for and not found'. Guarding each caller instead would be eleven
    chances to miss one -- which is how this site survived the first pass."""
    src = (_NODES / "_otr_voice_node_common.py").read_text(encoding="utf-8")
    i = src.index("def _resolve_ref_to_disk")
    body = src[i:i + 2200]
    assert "is_remote_path" in body, (
        "the remote refusal belongs inside _resolve_ref_to_disk")
