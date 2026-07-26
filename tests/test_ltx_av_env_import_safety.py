"""A malformed env value may not take an adapter's IMPORT down (chunk 7b-1).

``eng_ltx_av`` parsed four environment variables at MODULE SCOPE with a bare
``int()`` / ``float()``. A typo in any of them raised ``ValueError`` during
import, so the adapter never registered -- and ``frame_contract_for`` answers
``SINGLE_ONLY`` for an adapter it cannot reach, because its resolution is
wrapped in a broad ``except Exception`` (``frame_contract.py:243-247``).

The failure therefore did NOT look like a failure. The engine disappeared, its
lane silently reverted to unbounded single-clip behaviour, and nothing in the
log named the variable. That is the swallowed-fail-closed shape chunk 1a's QA
panel found four times over, arriving through a different door.

Both r2 kibitz seats (agy Gemini 3.6 Flash High, codex) reached this
independently from
``docs/2026-07-27-multiclip-7b-fork-problem-statement.md`` section 5.4.

SCOPE. This pins the CRASH only. Whether a WELL-FORMED env value may move a
declared ceiling is the 7b fork and is settled elsewhere -- see
``docs/2026-07-27-multiclip-7b-fork-r3-coding-plan.md``. A test here that
asserted a well-formed override were refused would be pre-judging that fork.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every module-scope environment read in ``eng_ltx_av``, with a value that the
#: pre-fix bare cast could not parse. Table-driven on purpose: the defect was
#: never specific to the frame ceiling, and a test that pinned only the one
#: variable the panel happened to name would leave three identical landmines.
MALFORMED = [
    ("OTR_LTX_AV_MAX_FRAMES", "not-a-number"),
    ("OTR_LTX_AV_STEPS", "eight"),
    ("OTR_LTX_AV_CFG", "3.0.1"),
    ("OTR_LTX_AV_I2V_STRENGTH", ""),
]

#: The declared defaults these must fall back to. ``497`` is also the literal
#: ``frame_contract`` declares (``eng_ltx_av.py:1229``), which is what makes a
#: silent fallback to something else a contract violation rather than a nit.
DEFAULTS = {
    "_LTX_AV_MAX_FRAMES": 497,
    "_LTX_AV_STEPS": 8,
    "_LTX_AV_CFG": 3.0,
    "_LTX_AV_I2V_STRENGTH": 1.0,
}


def _import_with(env_overrides):
    """Import the adapter in a FRESH interpreter under ``env_overrides``.

    A subprocess rather than ``importlib.reload``: the module body calls
    ``registry.register``, so reloading it inside the test session would mutate
    the shared registry every other test in the run reads. The cold-import
    checks elsewhere in this suite use the same idiom
    (``tests/test_video_wrapper_bridge.py:245-254``).
    """
    import os

    code = (
        "import nodes._otr_video_engines.eng_ltx_av as m;"
        "print('MAX', m._LTX_AV_MAX_FRAMES);"
        "print('STEPS', m._LTX_AV_STEPS);"
        "print('CFG', m._LTX_AV_CFG);"
        "print('STRENGTH', m._LTX_AV_I2V_STRENGTH)"
    )
    env = dict(os.environ)
    env.update(env_overrides)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, env=env)


def _parsed(stdout):
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("MAX", "STEPS", "CFG", "STRENGTH"):
            out[parts[0]] = float(parts[1])
    return out


@pytest.mark.parametrize("name,bad", MALFORMED)
def test_a_malformed_env_value_does_not_crash_the_import(name, bad):
    """THE MUTATION TARGET. Restore the bare ``int()`` / ``float()`` at module
    scope and this must fail with a non-zero exit and a ``ValueError`` in
    stderr -- not merely a changed number."""
    r = _import_with({name: bad})
    assert r.returncode == 0, (
        "%s=%r took the import of eng_ltx_av down:\n%s\n%s"
        % (name, bad, r.stdout, r.stderr))
    assert "ValueError" not in r.stderr, (
        "%s=%r raised at import instead of falling back:\n%s"
        % (name, bad, r.stderr))


@pytest.mark.parametrize("name,bad", MALFORMED)
def test_a_malformed_env_value_falls_back_to_the_declared_default(name, bad):
    """A survived import is not enough -- it must land on the DECLARED number.

    Falling back to 0, or to the empty-string coercion, would leave the adapter
    importable and its ceiling wrong, which is worse than the crash: the crash
    at least stopped.
    """
    r = _import_with({name: bad})
    assert r.returncode == 0, r.stderr
    got = _parsed(r.stdout)
    assert got, "no constants printed:\n%s\n%s" % (r.stdout, r.stderr)
    assert got["MAX"] == DEFAULTS["_LTX_AV_MAX_FRAMES"]
    assert got["STEPS"] == DEFAULTS["_LTX_AV_STEPS"]
    assert got["CFG"] == DEFAULTS["_LTX_AV_CFG"]
    assert got["STRENGTH"] == DEFAULTS["_LTX_AV_I2V_STRENGTH"]


def test_a_well_formed_env_value_is_still_honoured():
    """The fix must not quietly become a refusal.

    Refusing a well-formed override is Option A of the 7b fork, and Option A
    was CUT -- it breaks the documented low-VRAM operator knobs. If this ever
    starts failing, someone settled the fork inside a crash fix.
    """
    r = _import_with({"OTR_LTX_AV_STEPS": "12"})
    assert r.returncode == 0, r.stderr
    assert _parsed(r.stdout)["STEPS"] == 12


def test_the_engine_still_registers_under_a_malformed_env():
    """The consequence that made the crash invisible, pinned directly.

    An unimportable adapter is not absent from the registry -- it is present
    and resolves to ``SINGLE_ONLY``, which reads as 'this engine has no
    ceiling' rather than 'this engine failed to load'.
    """
    code = (
        "import nodes._otr_video_engines as pkg;"
        "from nodes._otr_video_engines import registry as vreg;"
        "from nodes._otr_video_engines import frame_contract as fc;"
        "e = vreg.get_engine('ltx_audio_in');"
        "print('MAXF', fc.frame_contract_for(e).max_frames)"
    )
    import os

    env = dict(os.environ)
    env["OTR_LTX_AV_MAX_FRAMES"] = "not-a-number"
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, "%s\n%s" % (r.stdout, r.stderr)
    assert "MAXF 497" in r.stdout, (
        "ltx_audio_in did not keep its declared 497 ceiling under a malformed "
        "env -- a 0 here means the adapter fell through to SINGLE_ONLY:\n%s\n%s"
        % (r.stdout, r.stderr))
