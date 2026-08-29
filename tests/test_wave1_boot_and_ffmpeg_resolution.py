"""Wave 1 regressions: boot-contract IDENTIFICATION and ffmpeg RESOLUTION.

Every test here pins a defect that failed LATE -- after a boot, or after an
expensive render -- which is exactly why a cheap CPU test is worth having.

The boot half: `contract_from_running_server` used to match reserve-VRAM by
EQUALITY while `check_running_server` has always used a FLOOR, and it ignored
Sage entirely. A server booted with more reserve than H3 asks for therefore
failed to identify as H3, and H3 refused its own valid boot.

The ffmpeg half: OTR ships an install shape where ffmpeg may be reachable ONLY
through `OTR_FFMPEG`. Four resolvers ignored it -- the mux's own identity
proof, the scopes encoder, the shared video wrapper bridge, and cloud video
canonicalization -- so work completed and then failed at the boundary.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import boot_contracts as bc


def _state(reserve, pinned, sage):
    return {"available": True, "reserve_vram_gb": reserve,
            "disable_pinned_memory": pinned, "sage_attention": sage}


# --------------------------------------------------------------------------- #
# boot-contract identification
# --------------------------------------------------------------------------- #
def test_reserving_MORE_than_h3_asks_still_identifies_as_h3(monkeypatch):
    """THE REFUSAL THIS FIXES. `--reserve-vram 16` is inside the envelope H3
    was measured in (12), and the satisfaction check has always agreed. Only
    IDENTIFICATION disagreed, so H3 rejected a boot that honoured it."""
    monkeypatch.setattr(bc, "running_server_boot_state",
                        lambda: _state(16.0, True, False))
    assert bc.contract_from_running_server() == bc.H3


def test_a_server_that_satisfies_two_contracts_reports_the_most_specific(
        monkeypatch):
    """reserve 12 + pinned-off satisfies the humo diet's 2.921 floor as well as
    H3's 12. The answer must be the MOST constrained boot the server can be,
    not whichever contract name happened to sort first."""
    monkeypatch.setattr(bc, "running_server_boot_state",
                        lambda: _state(12.0, True, False))
    assert bc.contract_from_running_server() == bc.H3


def test_sage_known_ACTIVE_disqualifies_the_sage_free_contract(monkeypatch):
    """H3 pins `sage_attention: False` because its recorded sm_120 behaviour
    under active Sage is CORRUPT OUTPUT. A server with Sage known on is not an
    H3 boot, however its VRAM knobs look."""
    monkeypatch.setattr(bc, "running_server_boot_state",
                        lambda: _state(12.0, True, True))
    assert bc.contract_from_running_server() != bc.H3


def test_sage_UNKNOWN_stays_a_candidate_so_the_named_check_can_speak(
        monkeypatch):
    """Identification is not the gate. An unverifiable Sage state must still
    reach `assert_running_server`, whose refusal names the real reason -- an
    unverifiable clamp may not be assumed on a lane Sage silently corrupts."""
    monkeypatch.setattr(bc, "running_server_boot_state",
                        lambda: _state(12.0, True, None))
    assert bc.contract_from_running_server() == bc.H3
    with pytest.raises(bc.BootContractError):
        bc.assert_running_server(bc.H3)


def test_a_stock_server_is_still_not_an_h3_boot(monkeypatch):
    """The guard doing its job: no reserve, no pinned clamp -> default."""
    monkeypatch.setattr(bc, "running_server_boot_state",
                        lambda: _state(None, False, None))
    assert bc.contract_from_running_server() == bc.DEFAULT


def test_h3_asserts_its_OWN_contract_not_the_stripped_policy(monkeypatch):
    """On a real leg the policy adapters see is rebuilt from the ledger's
    `video` section and carries no `launch`, so `contract_for_profile` answers
    `default` -- which constrains nothing. Asserting THAT verified nothing while
    reading like a defense. H3 declares exactly one compatible contract, so its
    second check must name it."""
    from nodes._otr_video_engines.eng_minimax_h3 import MiniMaxH3VideoEngine

    engine = MiniMaxH3VideoEngine()
    assert tuple(engine.compatible_boot_contracts) == (bc.H3,)

    seen = []
    monkeypatch.setattr(bc, "assert_running_server",
                        lambda name, state=None: seen.append(name))
    monkeypatch.setattr(bc, "check_engine_against_profile",
                        lambda engine, profile: [])
    engine._assert_boot_contract({})  # a stripped, launch-less production policy
    assert seen == [bc.H3]


# --------------------------------------------------------------------------- #
# ffmpeg resolution -- explicit -> OTR_FFMPEG -> PATH, everywhere
# --------------------------------------------------------------------------- #
def test_the_wrapper_bridge_honours_OTR_FFMPEG(monkeypatch, tmp_path):
    """The widest blast radius: every heavy lane encodes through this bridge,
    and the cheap-family preflight VALIDATES this env var before a render the
    bridge then ran with a literal 'ffmpeg'."""
    from nodes._otr_video_engines import wrapper_bridge as wb

    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("OTR_FFMPEG", str(fake))
    monkeypatch.setattr(wb.shutil, "which", lambda name: None)
    assert wb.resolve_ffmpeg("ffmpeg") == str(fake)
    # argv[0] of an already-built command is rewritten, so every pure builder
    # is covered without changing one tested arg list.
    assert wb._with_resolved_ffmpeg(["ffmpeg", "-y"])[0] == str(fake)


def test_an_explicit_ffmpeg_still_wins_over_the_env(monkeypatch, tmp_path):
    from nodes._otr_video_engines import wrapper_bridge as wb

    explicit = tmp_path / "explicit.exe"
    explicit.write_bytes(b"")
    other = tmp_path / "env.exe"
    other.write_bytes(b"")
    monkeypatch.setenv("OTR_FFMPEG", str(other))
    monkeypatch.setattr(wb.shutil, "which", lambda name: None)
    assert wb.resolve_ffmpeg(str(explicit)) == str(explicit)


def test_scope_draw_honours_OTR_FFMPEG(monkeypatch, tmp_path):
    """The scopes node ships a tooltip promising this exact order."""
    from nodes._otr_shared import scope_draw as sd

    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("OTR_FFMPEG", str(fake))
    monkeypatch.setattr(sd.shutil, "which", lambda name: None)
    assert sd.find_ffmpeg("ffmpeg") == str(fake)


def test_the_mux_identity_proof_resolves_like_the_encode(monkeypatch, tmp_path):
    """`audio_pcm_sha` called `shutil.which("ffmpeg")` directly. On an env-only
    box the mux ENCODED FINE and this returned '', so the fail-closed identity
    assertion destroyed a finished episode at the last boundary."""
    from nodes import otr_master_audio_mux as mux

    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv("OTR_FFMPEG", str(fake))
    monkeypatch.setattr(mux.shutil, "which", lambda name: None)

    seen = {}

    def _fake_run(cmd, **kw):
        seen["argv0"] = cmd[0]
        raise RuntimeError("stop after argv is proven")

    monkeypatch.setattr(mux.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError):
        mux.audio_pcm_sha(str(tmp_path / "x.wav"))
    assert seen["argv0"] == str(fake)


# --------------------------------------------------------------------------- #
# fail-soft contracts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_an_infinite_planning_ceiling_reads_as_unpinned(value):
    """Python's own JSON parser accepts bare `Infinity`, so a profile or ledger
    can carry it -- and `int(inf)` raises OverflowError, which the helper's
    "never raises" promise did not survive."""
    from nodes._otr_video_engines.frame_contract import (
        normalized_planning_ceiling,
    )

    assert normalized_planning_ceiling(value) == 0


def test_the_joint_av_receipts_reach_the_durable_trace():
    """Stamped on the request and declared in the schema, but omitted from the
    node-92 projection allowlist -- so the published /history evidence never
    carried what a joint-AV beat asked for or was heard to say."""
    import inspect

    from nodes._otr_video_engines import render_driver

    src = inspect.getsource(render_driver.run_episode)
    for key in ("joint_av_prompt", "joint_av_sounds", "joint_av_identity_leak"):
        assert key in src, "%s is stamped but never projected into the trace" % key
