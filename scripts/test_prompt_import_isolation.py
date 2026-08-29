"""scripts/test_prompt_import_isolation.py

Subprocess guard for the Sprint H headless prompt tester.

Confirms that importing any OTR LLM prompt entrypoint module does NOT
transitively pull `comfy.*` or `folder_paths`. The tester runs in a
plain venv WITHOUT ComfyUI's node loader -- any transitive ComfyUI
import would crash the run at first call.

Why subprocess: static grep misses transitive imports. In-process
sys.modules stubbing pollutes the tester's own namespace. A child
process with the sentinel pre-installed gives a clean answer.

Exit codes:
  0  all 5 prompt modules import cleanly with sentinels installed
  1  at least one module accessed a forbidden import surface
  2  child process crashed for an unrelated reason

Usage:
  & C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe ^
    scripts\\test_prompt_import_isolation.py

Per Sprint H Go-Forward Plan section 1.1 (docs/2026-05-17-sprint-h-
go-forward-plan.md).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Active prompt entrypoint modules. Order does not matter for the guard.
# (_otr_style_picker was retired with the style-picker rip; the module no
# longer exists, so listing it here would fail the guard on a ghost.)
PROMPT_MODULES = (
    "nodes._otr_outline",
    "nodes._otr_casting",
    "nodes._otr_line_composer",
    "nodes._otr_story_brief",
    "nodes.news_interpreter",
)

# Forbidden top-level package names. The child process installs a
# sentinel in sys.modules under each name; any attribute access on
# the sentinel raises ImportIsolationError and exits 1.
FORBIDDEN = ("comfy", "folder_paths")


_CHILD_SCRIPT = r"""
import importlib
import sys
import traceback


class ImportIsolationError(RuntimeError):
    pass


class _Forbidden:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise ImportIsolationError(
            "forbidden import surface accessed: %s.%s" % (self._name, attr)
        )

    def __call__(self, *a, **kw):
        raise ImportIsolationError(
            "forbidden import surface called: %s(...)" % self._name
        )


FORBIDDEN_NAMES = {forbidden!r}
PROMPT_MODULES = {modules!r}

# Install sentinels under both the top-level name and any common
# attribute-access pattern (e.g. "from comfy.utils import X").
for name in FORBIDDEN_NAMES:
    sys.modules[name] = _Forbidden(name)

results = []
exit_code = 0
for mod_name in PROMPT_MODULES:
    try:
        importlib.import_module(mod_name)
        results.append((mod_name, "OK", ""))
    except ImportIsolationError as exc:
        results.append((mod_name, "FORBIDDEN", str(exc)))
        exit_code = 1
    except Exception as exc:
        # Any OTHER exception (ImportError on a real missing dep,
        # SyntaxError on a malformed module, etc.) is also a fail
        # -- the prompt module must be importable in a plain venv.
        tb = traceback.format_exc(limit=3)
        results.append((mod_name, "ERROR", repr(exc) + " // " + tb.splitlines()[-1]))
        exit_code = 1

import json
print(json.dumps(results))
sys.exit(exit_code)
"""


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    venv_py = Path(r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe")
    py = str(venv_py) if venv_py.exists() else sys.executable

    child_src = _CHILD_SCRIPT.format(forbidden=FORBIDDEN, modules=PROMPT_MODULES)

    env = os.environ.copy()
    # Insert repo root so `nodes.*` resolves regardless of cwd.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_root) + (os.pathsep + existing_pp if existing_pp else "")
    )

    try:
        proc = subprocess.run(
            [py, "-c", child_src],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("[isolation] FAIL: child process timeout (60s)", file=sys.stderr)
        return 2
    except Exception as exc:
        print("[isolation] FAIL: subprocess crashed: %r" % (exc,), file=sys.stderr)
        return 2

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()

    # Parse the JSON line the child emits as its final stdout line.
    results = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("["):
            try:
                results = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if results is None:
        print("[isolation] FAIL: could not parse child output", file=sys.stderr)
        print("--- child stdout ---", file=sys.stderr)
        print(out, file=sys.stderr)
        print("--- child stderr ---", file=sys.stderr)
        print(err, file=sys.stderr)
        return 2

    n_total = len(results)
    n_ok = sum(1 for _, status, _ in results if status == "OK")
    n_bad = n_total - n_ok

    print("=" * 64)
    print("Sprint H import isolation guard")
    print("=" * 64)
    print("Sentinels installed for: %s" % ", ".join(FORBIDDEN))
    print("Modules tested: %d" % n_total)
    print()
    for mod_name, status, detail in results:
        marker = "PASS" if status == "OK" else "FAIL"
        if detail:
            print("  [%s] %-30s  %s  %s" % (marker, mod_name, status, detail))
        else:
            print("  [%s] %-30s  %s" % (marker, mod_name, status))
    print()
    print("Summary: %d/%d clean" % (n_ok, n_total))
    if n_bad:
        print("EXIT 1: %d module(s) accessed a forbidden import surface" % n_bad)
        return 1
    print("EXIT 0: all prompt modules import in a plain venv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
