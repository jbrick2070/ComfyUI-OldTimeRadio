"""A derived path is confined on its own spelling, not through its parent or
its principal.

Four claims from the 2026-10-30 registry-sinks review (Codex Astra, grounded
by executed probes). The episode stills directory passed a text-only check, so
a junction already sitting at the episode or its stills folder carried the
dispatcher's copies and manifest outside the output tree. The caption `.ass`,
the credits frames and concat list, and the composite `.qa.json` are derived
beside a confined output, each its own leaf a planted link could redirect. A
directory clip's listed frames were opened and handed to ffmpeg with no check
of their own. And two fingerprint hooks proved a relative spelling anchored at
the output root, then stat'ed the raw spelling anchored at the process cwd.
Link tests skip where the box cannot create links.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import tempfile
from pathlib import Path

import pytest


def _link_file(target: Path, link: Path) -> bool:
    try:
        os.symlink(str(target), str(link))
        return True
    except (OSError, NotImplementedError):
        return False


def _link_dir(target: Path, link: Path) -> bool:
    """A directory link. On Windows a JUNCTION first: it is what the walks
    follow (`rglob` descends into a junction and not into a symlinked
    directory on Python 3.12), so it is the shape that escaped."""
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
            return True
        except OSError:
            pass
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


@pytest.fixture
def outside():
    """A folder in no read or write root: pytest's tmp tree IS a declared
    root, so this sits beside it in the system temp directory."""
    d = Path(tempfile.mkdtemp(prefix="otr_outside_"))
    try:
        yield d
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# --- RANK 4: the episode stills directory ----------------------------------- #

def test_a_linked_episode_dir_is_refused_by_the_stills_resolver(tmp_path, outside):
    from nodes._otr_shared import portrait_ledger as PLG
    out = tmp_path / "out"
    eps = out / "otr" / "episodes"
    eps.mkdir(parents=True)
    assert PLG.episode_stills_dir("ep0", output_dir=str(out)) == eps / "ep0" / "stills"
    if not _link_dir(outside, eps / "ep1"):
        pytest.skip("this box cannot create a directory link")
    with pytest.raises(PLG.PortraitUnresolved):
        PLG.episode_stills_dir("ep1", output_dir=str(out))
    (eps / "ep2").mkdir()
    assert _link_dir(outside, eps / "ep2" / "stills")
    with pytest.raises(PLG.PortraitUnresolved):
        PLG.episode_stills_dir("ep2", output_dir=str(out))


def test_the_episode_copy_keeps_a_hash_with_separators_inside(tmp_path):
    from nodes import otr_image_gen_dispatcher as D
    ep_dir = tmp_path / "otr" / "episodes" / "ep1" / "stills"
    ep_dir.mkdir(parents=True)
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG src")
    dst = Path(D._materialize_episode_copy(str(src), str(ep_dir), "p", "\\..\\..\\zz"))
    assert dst.parent == ep_dir and dst.read_bytes() == b"\x89PNG src"
    assert not list(tmp_path.glob("**/zz*"))
    hexed = D._materialize_episode_copy(str(src), str(ep_dir), "q", "ab" * 32)
    assert Path(hexed).name == "q_" + "ab" * 6 + ".png"


def test_the_episode_copy_refuses_a_linked_stills_dir(tmp_path, outside):
    from nodes import otr_image_gen_dispatcher as D
    link = tmp_path / "otr" / "episodes" / "ep1" / "stills"
    link.parent.mkdir(parents=True)
    if not _link_dir(outside, link):
        pytest.skip("this box cannot create a directory link")
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG src")
    with pytest.raises(D.ImageRenderError):
        D._materialize_episode_copy(str(src), str(link), "p", "ab" * 32)
    assert not list(outside.glob("*.png"))


# --- RANK 5: derived sidecars ---------------------------------------------- #

def test_every_derived_sidecar_is_confined_at_its_site():
    """The wiring, asserted at the real sites: the sidecar names are derived
    beside a confined output and each goes through the write core before its
    write or its ffmpeg argv. Behaviour of the core itself is pinned in
    tests/test_output_tree_containment.py and tests/test_linked_paths_are_refused.py."""
    from nodes import otr_caption_burn as CB
    from nodes import otr_credits_roll as CR
    from nodes import otr_silent_composite as SC
    burn = inspect.getsource(CB.burn_captions_on_video)
    assert burn.index('_confine_sidecar(_ass_sidecar_path(out_path)') < burn.index("_build_ass(")
    credits = inspect.getsource(CR)
    for leaf in ('confine_to_output_tree(credits_clip, "credits clip")',
                 'confine_to_output_tree(backdrop_png, "credits backdrop")',
                 '_confine_frame(out_path + ".base.png"',
                 '_confine_frame(out_path + ".scroll.png"',
                 '_confine_list(compact_scratch('):
        assert leaf in credits, leaf
    assert credits.index('_confine_frame(out_path + ".base.png"') < credits.index('.save(base_png)')
    composite = inspect.getsource(SC.assemble_silent_timeline)
    assert '_confine_qa(out_path + ".qa.json"' in composite


def test_a_planted_sibling_link_is_refused_by_the_write_core(tmp_path, outside):
    """What the sites above rely on: `out` passes, the sibling at a planted
    link does not."""
    from nodes._otr_paths import OtrPathContractError, confine_to_output_tree
    out = tmp_path / "episode" / "final.mp4"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"mp4")
    victim = outside / "victim.ass"
    victim.write_text("do not touch", encoding="utf-8")
    sidecar = out.parent / "final.ass"
    if not _link_file(victim, sidecar):
        pytest.skip("this box cannot create a file symlink")
    assert confine_to_output_tree(str(out), "video_path") == str(out)
    with pytest.raises(OtrPathContractError):
        confine_to_output_tree(str(sidecar), "caption .ass")


# --- RANK 10: listed frames ------------------------------------------------ #

def test_a_linked_frame_is_not_a_directory_clip_frame(tmp_path, outside):
    from nodes._otr_video_engines import directory_clip as DC
    clip = tmp_path / "clip"
    clip.mkdir()
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    (clip / "0001.png").write_bytes(png)
    assert DC.list_directory_frames(str(clip)) == [str(clip / "0001.png")]
    away = outside / "0002.png"
    away.write_bytes(png)
    if not _link_file(away, clip / "0002.png"):
        pytest.skip("this box cannot create a file symlink")
    with pytest.raises(ValueError, match="link"):
        DC.list_directory_frames(str(clip))


# --- RANK 11: fingerprint spellings ---------------------------------------- #

def test_the_fingerprint_hashes_only_the_proven_spelling(tmp_path, monkeypatch):
    from nodes import otr_master_audio_mux as MUX
    out = tmp_path / "output"
    (out / "otr" / "obs").mkdir(parents=True)
    cwd = tmp_path / "cwd"
    (cwd / "otr" / "obs").mkdir(parents=True)
    (cwd / "otr" / "obs" / "secret.wav").write_bytes(b"RIFF cwd secret")
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(out))
    monkeypatch.chdir(cwd)
    assert MUX._quiet_file_sha256("otr/obs/secret.wav") == ""      # not the cwd file
    (out / "otr" / "obs" / "secret.wav").write_bytes(b"RIFF proven")
    assert MUX._quiet_file_sha256("otr/obs/secret.wav") == hashlib.sha256(b"RIFF proven").hexdigest()


def test_child_inside_follows_the_link_not_the_name(tmp_path, outside):
    from nodes._otr_paths import child_inside
    (tmp_path / "real.txt").write_text("x", encoding="utf-8")
    assert child_inside(tmp_path / "real.txt", tmp_path)
    assert not child_inside(outside / "x.txt", tmp_path)
    (outside / "away.txt").write_text("y", encoding="utf-8")
    if not _link_file(outside / "away.txt", tmp_path / "link.txt"):
        pytest.skip("this box cannot create a file symlink")
    assert not child_inside(tmp_path / "link.txt", tmp_path)


def test_two_sources_without_a_digest_never_share_a_name(tmp_path):
    """Codex review of 502aeeda6: the `nodigest` fallback collided, so the
    second source kept the first one's bytes. The fallback is the source's
    own digest."""
    from nodes import otr_image_gen_dispatcher as D
    ep_dir = tmp_path / "otr" / "episodes" / "ep1" / "stills"
    ep_dir.mkdir(parents=True)
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"\x89PNG a")
    b.write_bytes(b"\x89PNG b")
    da = Path(D._materialize_episode_copy(str(a), str(ep_dir), "same", ""))
    db = Path(D._materialize_episode_copy(str(b), str(ep_dir), "same", "x"))
    assert da != db and da.read_bytes() == b"\x89PNG a" and db.read_bytes() == b"\x89PNG b"
    mixed = Path(D._materialize_episode_copy(str(a), str(ep_dir), "m", "zzabcdef123456"))
    assert mixed.name != "m_abcdef123456.png"


def test_a_refused_still_destination_degrades_at_every_fail_soft_site():
    """Codex review of 502aeeda6: the materialization callers caught only
    OSError, so a confinement refusal (ImageRenderError, a ValueError for the
    manifest leaf) aborted the dispatch instead of pointing the row at the
    pool copy, as their own warnings advertise."""
    import inspect
    from nodes import otr_image_gen_dispatcher as D
    src = inspect.getsource(D)
    assert src.count("except (OSError, ImageRenderError) as exc:") == 2
    assert "except (OSError, ValueError) as exc:        # ValueError: a refused leaf" in src
    assert "except OSError as exc:\n        ctx[\"warnings\"].append(" not in src
