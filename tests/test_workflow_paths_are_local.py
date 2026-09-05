"""A path that arrives from a workflow must name something on THIS machine.

THE DEFECT. Several nodes take a filesystem path from an ordinary editable
STRING widget, and a workflow JSON arrives over ComfyUI's ``/prompt`` endpoint,
which is unauthenticated by default. The worst of them is
``OTR_LedgerScriptWriter``'s ``replay_from``: it is handed VERBATIM to
``production_ledger.import_replay_bundle`` -> ``load_replay_manifest``, whose
first act is ``os.path.isfile(<bundle>/replay_manifest.json)``.

On Windows the ordinary file APIs open a UNC path transparently. So statting
``\\\\attacker-host\\share\\...`` makes the machine perform an SMB negotiate and
authenticate to a host the WORKFLOW chose, handing over NTLM material -- before
any containment or digest check in this pack has run. Nothing needs to exist
locally first, which makes it strictly cheaper to reach than the pack's other
path defects. If the attacker instead supplies a directory they control, they
also author the manifest, so its own SHA-256 entries validate THEIR bundle, and
``shutil.copyfile`` then copies their files into an episode workspace that is
eligible for publication.

WHAT THIS GUARD IS NOT. It is not containment and not a traversal check --
``_validate_episode_id`` and ``_validate_contract`` already do those jobs for
the paths they cover. This asks only "does this path leave the machine", which
is why it can be applied to a value the operator legitimately points anywhere
on their own disk.
"""
from __future__ import annotations

import pathlib

import pytest

from nodes._otr_paths import OtrPathContractError, reject_remote_path

REPO = pathlib.Path(__file__).resolve().parents[1]

REMOTE = [
    r"\\attacker-host\share",
    r"\\attacker-host\share\bundle",
    "//attacker/share",
    r"\\?\C:\windows",          # Windows device namespace
    r"\\.\pipe\x",
    "http://evil.example/x",
    "https://evil.example/x",
    "file:///etc/passwd",
    "smb://host/share",
    "ftp://host/x",
]

LOCAL = [
    r"C:\Users\someone\Documents\ComfyUI\output\otr\episodes\ep1",
    r"D:\bundles\my episode",
    r"C:\ComfyUI-Models\LLM",
    "otr/episodes/ep1",
    "/home/u/otr/ep1",
    "relative/dir",
    "",
    "   ",
]


@pytest.mark.parametrize("value", REMOTE)
def test_a_remote_path_is_refused(value):
    with pytest.raises(OtrPathContractError):
        reject_remote_path(value, "replay_from")


@pytest.mark.parametrize("value", LOCAL)
def test_an_ordinary_local_path_is_accepted(value):
    """A drive letter is NOT a URL scheme -- `C:\\...` must keep working, and
    so must every relative and POSIX path. This guard must never be the reason
    a normal render stops."""
    assert reject_remote_path(value, "replay_from") == value.strip()


def test_the_refusal_names_the_field_and_the_value():
    """An operator who typed a share path deserves to know which input."""
    with pytest.raises(OtrPathContractError) as caught:
        reject_remote_path(r"\\host\share", "replay_from")
    message = str(caught.value)
    assert "replay_from" in message
    assert "host" in message


# --------------------------------------------------------------------------- #
# the sinks actually call it -- a helper nobody invokes is not a fix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel,why", [
    ("nodes/production_ledger.py",
     "replay_from -> import_replay_bundle / load_replay_manifest: the SMB "
     "authentication happens on the first isfile(), so the check must precede it"),
    ("nodes/_otr_workflow_validator.py",
     "workflow_json_path documents 'absolute -> taken as-is' and then opens it"),
])
def test_the_sink_refuses_remote_paths(rel, why):
    src = (REPO / rel).read_text(encoding="utf-8")
    assert "_reject_remote(" in src, "%s does not refuse a remote path (%s)" % (rel, why)


def test_the_replay_import_refuses_a_unc_bundle():
    """End to end on the real function: it must refuse BEFORE touching the
    filesystem, so no SMB session is ever opened."""
    from nodes import production_ledger as pl
    with pytest.raises(OtrPathContractError):
        pl.load_replay_manifest(r"\\attacker-host\share\bundle")


def test_the_workflow_validator_refuses_a_unc_path():
    from nodes import _otr_workflow_validator as wv
    with pytest.raises(OtrPathContractError):
        wv._resolve_workflow_path(r"\\attacker-host\share\graph.json")


def test_an_ordinary_workflow_path_still_resolves():
    """The canonical default and a plain relative path must be unaffected."""
    from nodes import _otr_workflow_validator as wv
    assert wv._resolve_workflow_path("") == wv._DEFAULT_WORKFLOW_PATH
    got = wv._resolve_workflow_path("workflows/otr_canonical.json")
    assert got.is_absolute()


# --------------------------------------------------------------------------- #
# the guard lives at the WIDGET boundary, NOT at the spawn
# --------------------------------------------------------------------------- #
def test_the_spawn_owner_does_NOT_carry_this_rule():
    """A spawn-level UNC rule was written, measured, and REMOVED.

    A mapped network drive BECOMES a UNC path once resolved: on the development
    box, resolving the mapped drive U: yields a UNC path on the 4060 transfer
    host. And blend() resolves its inputs before handing them to ffmpeg, so the
    rule refused a LEGITIMATE render on any install whose output lives on a
    mapped drive -- including the operator's own transfer drive.

    Provenance is not knowable at the spawn: by then a hostile widget value and
    a resolved local path look identical. The refusal belongs where the value
    ARRIVES, which is what reject_remote_paths does at each execute method."""
    src = (REPO / "nodes/_otr_shared/proc.py").read_text(encoding="utf-8")
    assert "_no_remote_arguments" not in src, (
        "the spawn-level UNC rule is back; it breaks mapped-drive installs")


@pytest.mark.parametrize("node,rel", [
    ("OTR_CaptionBurn", "nodes/otr_caption_burn.py"),
    ("OTR_MasterAudioMux", "nodes/otr_master_audio_mux.py"),
    ("OTR_SilentComposite", "nodes/otr_silent_composite.py"),
    ("OTR_PostUpscaleProcgenBlend", "nodes/otr_post_upscale_procgen_blend.py"),
])
def test_each_media_node_refuses_remote_path_inputs(node, rel):
    src = (REPO / rel).read_text(encoding="utf-8")
    assert "reject_remote_paths(" in src, (
        "%s accepts a workflow path without asking whether it leaves the "
        "machine. The stat happens BEFORE any spawn, so a downstream guard "
        "cannot help." % node)


def test_a_mapped_drive_style_local_path_is_never_refused():
    """The regression this whole relocation exists for."""
    for value in (r"U:\OTR-BACKUP\ep.mp4", r"Z:\obs\ep.mp4",
                  r"E:\OTR-BACKUP\ep.mp4"):
        assert reject_remote_path(value, "output_path") == value


def test_the_plural_form_names_the_offending_field():
    from nodes._otr_paths import reject_remote_paths
    reject_remote_paths(video_path="C:/a.mp4", output_path="C:/b.mp4")
    with pytest.raises(OtrPathContractError) as caught:
        reject_remote_paths(video_path="C:/a.mp4",
                            output_path=r"\\evil\share\b.mp4")
    assert "output_path" in str(caught.value)
