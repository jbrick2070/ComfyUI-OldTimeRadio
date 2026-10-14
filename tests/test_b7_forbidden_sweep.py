"""S30 B7 -- forbidden-pattern sweep.

Two tests:

1. test_forbidden_sweep_runs_clean       -- run the sweep against
   the current diff vs. s29-clean-slate-gate; assert zero RUNTIME
   hits (forensic mentions allowed).
2. test_classifier_recognizes_fstring_context -- the sweep's line
   classifier treats Python 3.12 f-string tokens as string context.
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
SWEEP_PATH = PACK_ROOT / "tests" / "_s28_forbidden_sweep.py"


def _untracked_python_as_diff() -> str:
    """New .py files that git diff cannot see yet, rendered as added lines.

    The sweep reads a DIFF, and `git diff` covers TRACKED files only. So a new
    module or test file was invisible to this gate for exactly as long as it
    stayed untracked -- it passed green, and then turned HEAD red on its first
    commit with nothing else changed. That has now cost two red HEADs.

    EVERY unignored untracked `*.py`, matching the tracked half of the sweep,
    which already covers `*.py` repo-wide. An earlier draft limited this to
    `nodes/` and `tests/` to avoid "widening the gate to every untouched file";
    that rationale does not apply -- `git ls-files --others` lists only
    UNTRACKED files, never the untouched tracked tree, so the narrow scope
    bought nothing and left runtime Python under `config/`, `scripts/` and the
    repo root blind. `--exclude-standard` honours .gitignore, so tmp/ scratch
    stays out.

    FAILS LOUD. A git, read or decode error raises rather than returning an
    empty string: a gate that silently sweeps nothing is worse than no gate,
    which is exactly the failure this whole helper exists to close.
    """
    listed = subprocess.check_output(
        ["git", "-C", str(PACK_ROOT), "ls-files", "--others",
         "--exclude-standard", "--", "*.py"],
        text=True, encoding="utf-8",
    )
    chunks = []
    for rel in sorted({line.strip() for line in listed.splitlines() if line.strip()}):
        path = PACK_ROOT / rel
        body = path.read_text(encoding="utf-8")
        # Same shape `git diff` emits for a new file, so the sweep's existing
        # parser needs no change: every line arrives as an addition.
        #
        # The @@ header is REQUIRED, not decoration. The sweep tracks the
        # current line number through `hunk_re`, and a `+` line outside any
        # hunk is silently dropped -- an earlier draft of this helper omitted
        # it, produced a diff that looked perfectly correct, and swept exactly
        # nothing.
        lines = body.splitlines()
        chunks.append(
            "diff --git a/%s b/%s\nnew file mode 100644\n--- /dev/null\n+++ b/%s\n"
            "@@ -0,0 +1,%d @@\n" % (rel, rel, rel, len(lines))
            + "".join("+%s\n" % line for line in lines)
        )
    return "".join(chunks)


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
    diff += _untracked_python_as_diff()
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
    """The branch must trigger zero RUNTIME hits against the forbidden
    patterns (back-compat shims, aliases, hardcoded developer paths,
    period-locked language, ...). Forensic mentions (string literals +
    comments) are allowed.
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


def _sweep_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_s28_forbidden_sweep_under_test", SWEEP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # import-safe: the sweep itself is main()
    return mod


_PLANTED = (
    "def f():\n"                                                  # 1
    "    \"\"\"A docstring naming a back-compat path.\"\"\"\n"    # 2
    "    # a comment about the old shim\n"                         # 3
    "    msg = \"no back-compat here\"\n"                          # 4
    "    out = generate_text(\"hello\")\n"                         # 5
    "    note = f\"retired: {len(msg)} back-compat paths\"\n"      # 6
    "    return generate_text(msg)  # call\n"                      # 7
)


@pytest.mark.parametrize("lineno, expected", [
    (2, "forensic"),   # docstring
    (3, "forensic"),   # comment
    (4, "forensic"),   # plain string literal
    (5, "code"),       # a real call on a line that ALSO holds a string
    (6, "forensic"),   # the literal part of an f-string
    (7, "code"),       # a real call on a line that also holds a comment
])
def test_the_classifier_judges_each_hit_where_it_sits(tmp_path, lineno,
                                                       expected):
    """Calls the sweep's own classifier. Line 5 is the case that used to slip
    through: any string token on a line made the WHOLE line forensic, so an
    executable hit beside a string literal never reached the runtime list."""
    sw = _sweep_module()
    planted = tmp_path / "planted.py"
    planted.write_text(_PLANTED, encoding="utf-8")
    spans = sw.string_and_comment_spans(planted)
    text = _PLANTED.splitlines()[lineno - 1]
    assert sw.forbidden.search(text), "fixture line %d must hit" % lineno
    assert sw.classify_hit(spans, lineno, text) == expected


def test_an_unreadable_file_counts_as_code(tmp_path):
    """A gate that cannot tokenize a file must not wave its hits through."""
    sw = _sweep_module()
    broken = tmp_path / "broken.py"
    broken.write_text("x = (\n  'back-compat'\n", encoding="utf-8")
    assert sw.string_and_comment_spans(broken) is None
    assert sw.classify_hit(None, 2, "  'back-compat'") == "code"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
