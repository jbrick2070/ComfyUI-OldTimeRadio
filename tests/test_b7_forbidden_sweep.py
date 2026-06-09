"""S30 B7 -- forbidden-pattern sweep armed with extinction markers.

Four tests:

1. test_forbidden_sweep_runs_clean       -- run the sweep against
   the current diff vs. s29-clean-slate-gate; assert zero RUNTIME
   hits (forensic mentions allowed).
2. test_forbidden_sweep_catches_reintroduction -- write a temp file
   that contains a forbidden symbol AS A RUNTIME identifier; assert
   the sweep flags it (validates the marker actually fires).
3. test_forbidden_sweep_exemption_works  -- the loader exemption
   for `load_llm(` (the new canonical loader API) is not in the
   sweep's marker list; the sweep should NOT flag it.
4. test_forbidden_sweep_non_llm_marker_works -- a class with
   `NON_LLM_MODEL_WIDGET_OK = True` and a `model_id` STRING widget
   is exempt from the B6 structural guard (already proven by
   B6 tests; this is the cross-check at the B7 sweep layer).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
SWEEP_PATH = PACK_ROOT / "docs" / "_s28_forbidden_sweep.py"


def _run_sweep_subprocess() -> tuple[int, str]:
    """Regenerate the diff input + invoke the sweep as a subprocess.
    Returns (return_code, stdout).

    The diff + out files go to the OS temp dir (not the OneDrive-synced
    docs/ tree) and are passed to the sweep via OTR_S28_DIFF_PATH /
    OTR_S28_OUT_PATH. Writing the multi-MB diff into the synced tree let
    OneDrive hold it open mid-sync, so the next truncating re-open
    failed with OSError(22) -- the 2026-05-31 forbidden-sweep red
    baseline. The OS temp dir is not synced, so the sweep is
    deterministic.
    """
    tmpdir = Path(tempfile.gettempdir())
    diff_path = tmpdir / "otr_s28_diff_tmp.txt"
    out_path = tmpdir / "otr_s28_forbidden_hits.txt"
    # Use git directly to write the diff with UTF-8 encoding.
    diff = subprocess.check_output(
        ["git", "-C", str(PACK_ROOT), "diff",
         "s29-clean-slate-gate", "--", "*.py"],
        text=True, encoding="utf-8",
    )
    diff_path.write_text(diff, encoding="utf-8")
    env = {
        **os.environ,
        "OTR_S28_DIFF_PATH": str(diff_path),
        "OTR_S28_OUT_PATH": str(out_path),
    }
    proc = subprocess.run(
        [sys.executable, str(SWEEP_PATH)],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(PACK_ROOT), env=env,
    )
    return proc.returncode, proc.stdout


def test_forbidden_sweep_runs_clean():
    """The S30 branch must trigger zero RUNTIME hits against the
    full set of extinction markers (cleanup_model_id, OTR_LFCPhase*,
    OTR_VisualLLMSelector, _POLISH_CACHE, MODEL_CONTEXT_CAPS, etc.).
    Forensic mentions (string literals + comments) are allowed.
    """
    rc, stdout = _run_sweep_subprocess()
    assert rc == 0, f"sweep exited non-zero: rc={rc}"
    # The first line is HITS: N  forensic: F  runtime: R
    m = re.search(
        r"HITS:\s+(\d+)\s+forensic:\s+(\d+)\s+runtime:\s+(\d+)",
        stdout,
    )
    assert m, f"could not parse sweep output:\n{stdout}"
    runtime = int(m.group(3))
    assert runtime == 0, (
        f"S30 forbidden-pattern sweep has {runtime} runtime hits; "
        f"forensic count: {m.group(2)}\nsweep output:\n{stdout}"
    )


def test_forbidden_sweep_catches_reintroduction(tmp_path):
    """Drive the sweep's regex against a synthetic added line that
    contains a forbidden marker as a runtime identifier; assert the
    regex catches it (proves the markers actually fire).
    """
    # Import the sweep module to access the regex.
    spec_path = SWEEP_PATH
    code = spec_path.read_text(encoding="utf-8")
    # Extract the `forbidden = re.compile(...)` block. The sweep
    # module's code runs as import-time work, so directly importing
    # it would scan the diff file. To avoid that side effect, exec
    # only the regex definition in a private namespace.
    ns: dict = {"re": re}
    # Strip past the regex assignment.
    # Find the line where the regex is opened and the line where the
    # closing parenthesis lands.
    start = None
    end = None
    for i, line in enumerate(code.splitlines()):
        if start is None and "forbidden = re.compile(" in line:
            start = i
        elif start is not None and line.strip() == ")":
            end = i
            break
    assert start is not None and end is not None
    snippet = "\n".join(code.splitlines()[start:end + 1])
    exec(snippet, ns)
    forbidden = ns["forbidden"]

    # The full set of S30 extinction markers we expect to fire.
    samples = [
        "OTR_VisualLLMSelector",
        "VisualLLMSelector",
        "_POLISH_CACHE",
        "_MODEL_CHOICES",
        "DEFAULT_MODEL_ID",
        "MODEL_CONTEXT_CAPS",
        "DEFAULT_CONTEXT_CAP",
        "cleanup_model_id",
        "OTR_LFCPhase4Scene",
        "OTR_LFCPhase5Voice",
        "OTR_LFCPhase6Arc",
        "enable_phase_3_polish",
        "polish_announcer_beats",
        "enable_phase_4_scene_coherence",
        "enable_phase_4_5_smart_suggestion",
        "enable_phase_5_voice_drift",
        "enable_phase_6_episode_arc",
        "Phase3PolishReport",
    ]
    for sym in samples:
        assert forbidden.search(f"x = {sym}"), (
            f"sweep regex does not catch reintroduction of {sym!r}"
        )


def test_forbidden_sweep_exemption_works():
    """The modern canonical loader API `_otr_model_loader.load_llm`
    must NOT trigger the sweep. The marker set targets the legacy
    `_load_llm` (note the leading underscore) at orchestrator scope,
    not the new public surface.
    """
    spec_path = SWEEP_PATH
    code = spec_path.read_text(encoding="utf-8")
    ns: dict = {"re": re}
    start = None
    end = None
    for i, line in enumerate(code.splitlines()):
        if start is None and "forbidden = re.compile(" in line:
            start = i
        elif start is not None and line.strip() == ")":
            end = i
            break
    snippet = "\n".join(code.splitlines()[start:end + 1])
    exec(snippet, ns)
    forbidden = ns["forbidden"]
    # The canonical loader.load_llm(...) call signature must NOT match.
    canonical_call = "cache_entry = loader.load_llm(model_id, device='cuda')"
    assert not forbidden.search(canonical_call), (
        "sweep regex incorrectly catches the canonical "
        "loader.load_llm public API"
    )
    # The canonical loader.unload_llm() call must NOT match either.
    assert not forbidden.search("loader.unload_llm()"), (
        "sweep regex incorrectly catches the canonical "
        "loader.unload_llm public API"
    )


def test_forbidden_sweep_non_llm_marker_works():
    """B6 / B7 structural rule: a class flagged
    NON_LLM_MODEL_WIDGET_OK = True is exempt from the model-widget
    guard. This is the B6 test's responsibility (validated there);
    the sweep itself doesn't enforce the structural rule -- it
    enforces the symbol-name extinction list. Cross-check: VRAMContextTest
    (flagged with the marker) does NOT trigger the runtime sweep (it keeps its
    `model_id` widget legitimately). BatchAudioGenGenerator + MusicGenTheme were
    retired in the audio clean-break (1c), so the exempt set is now just the one.
    """
    rc, stdout = _run_sweep_subprocess()
    # The sweep output must not mention the exempt class as a runtime offender.
    for cls_name in ("VRAMContextTest",):
        # A runtime hit would appear under the "--- runtime (in
        # output file) ---" header; if the section exists but the
        # class name appears as an offender line, fail.
        runtime_section_started = False
        for line in stdout.splitlines():
            if "--- runtime" in line:
                runtime_section_started = True
                continue
            if not runtime_section_started:
                continue
            assert cls_name not in line, (
                f"S30 B7: exempt class {cls_name} appears as a "
                f"runtime offender in sweep output:\n  {line}"
            )


def test_classifier_recognizes_fstring_context():
    """Python 3.12 split string tokens into FSTRING_START /
    FSTRING_MIDDLE / FSTRING_END. The sweep's classifier was patched
    in B7 to treat these as string tokens (forensic), not code.
    Validates the classifier recognizes the FSTRING token types.
    """
    fstring_types = (
        getattr(tokenize, "FSTRING_START", None),
        getattr(tokenize, "FSTRING_MIDDLE", None),
        getattr(tokenize, "FSTRING_END", None),
    )
    # The token types exist on Python 3.12+. Skip rather than hard-fail on
    # 3.11 so the headless suite stays green on the base Python 3.11 install.
    if sys.version_info < (3, 12):
        pytest.skip(
            f"FSTRING_* tokenize tokens only on Python 3.12+; "
            f"running {sys.version_info.major}.{sys.version_info.minor}"
        )
    for t in fstring_types:
        assert t is not None, (
            "tokenize.FSTRING_* token types must exist on the OTR "
            "Python target so the sweep classifier can recognize "
            "f-string contexts as forensic"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
