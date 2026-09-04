# Cross-check: my responses to a gpt-6-astra pre-publish review

**You are the SECOND lane. Astra reviewed the shipped env/proc single-owner
migration before it is published to the Comfy Registry; I acted on its findings.
Check MY RESPONSES, and check whether Astra was right in the first place.** A
published version string is burned permanently, so this is the last cheap moment.

Disagree freely -- I would rather find out here.

## Astra's findings and exactly what I did

**MUST-FIX 1 -- standalone provisioning broken. ACCEPTED, CONFIRMED, FIXED.**
Astra said `nodes/_otr_model_catalog.py` and `nodes/_otr_voice_bank.py` no longer
import when a `scripts/` helper loads them BY PATH under a non-package name.
I verified in fresh subprocesses against a pre-migration git worktree:

    module                          before -> after
    _otr_model_catalog              LOADS  -> ModuleNotFoundError   (regression)
    _otr_voice_bank                 LOADS  -> ModuleNotFoundError   (regression)
    _otr_kokoro_voice_prefetch      LOADS  -> ModuleNotFoundError   (regression)
    otr_post_upscale_procgen_blend  LOADS  -> ModuleNotFoundError   (regression)
    _otr_gguf_backend               ImportError both  (pre-existing, not mine)
    _otr_ledger                     fails both        (pre-existing, not mine)
    _otr_ledger_clean               fails both        (pre-existing, not mine)

Fix: the three modules gained a `_NODES_DIR` sys.path insert BEFORE the owner
ladder -- the same pattern `otr_post_upscale_procgen_blend.py` already carried,
with a comment saying why. The blend file needed only its import MOVED below the
bootstrap it already had. New guard `tests/test_standalone_module_loads.py` runs
each in a FRESH SUBPROCESS, because the in-process suite cannot see this class.

**QUESTION FOR YOU:** is the sys.path insert the right fix, or does it create a
DOUBLE MODULE INSTANCE hazard? The blend file's own comment warns that the flat
spelling "resolves through this insert even inside the package and yields a
SECOND module instance of the owner it names". I preserved package-relative FIRST
in every ladder for exactly that reason -- but check me. A second `env`/`proc`
module object would be a real defect: `monkeypatch` on one would not reach the
other.

**SHOULD-FIX 2 -- `executable=` bypasses the allowlist. ACCEPTED, FIXED.**
`run(["ffmpeg"], executable="cmd.exe")` passed the allowlist on "ffmpeg" and
would have launched cmd.exe. Both entry points now refuse a non-None
`executable=`. `executable=None` is still accepted (subprocess's own default; a
caller passing it through from its own signature must not break).

**SHOULD-FIX 3 -- ratchet bypass spellings. PARTIALLY ACCEPTED.**
I covered the two with a live path: a CALL through the identity re-export
(`otr_proc.Popen(argv)` -- the raw stdlib class, no allowlist, no shell or
executable refusal) and a REBOUND spawn (`run = subprocess.run`). I did NOT cover
`getattr`, `importlib.import_module`, or star-imports, and instead recorded the
guard's limits in a `KNOWN LIMITS` comment with a test asserting it is present.
Astra's own CUT section recommended exactly that. **Check that the re-export is
still legal as an annotation and in an `except` clause** -- flagging those would
push modules back to importing subprocess, which defeats the migration.

**SHOULD-FIX 4 -- document the gpu_residency behaviour change. Already done**
before the review, in `tests/test_gpu_residency_pid_liveness.py` and the commit
body: Windows without psutil now warns once and returns True, disabling stale
lease reclamation.

## What I want from you specifically

1. **Is any of my four sys.path fixes wrong?** Especially the double-instance
   question above. Prove it either way from the code.
2. **Did Astra miss anything in the same class?** Other modules a `scripts/`
   helper loads by path that I did not test.
3. **Is the `executable=` refusal too strict?** Any legitimate caller pattern it
   breaks.
4. **Anything else that would embarrass a published version.**

Ground every claim in the real files. Say CONFIRMED / MISREAD / UNVERIFIABLE.
Suite is green, so a claim that something is broken must explain why.

## The diff under review

```diff
diff --git a/nodes/_otr_kokoro_voice_prefetch.py b/nodes/_otr_kokoro_voice_prefetch.py
index 0863e8a0..bfeba8d9 100644
--- a/nodes/_otr_kokoro_voice_prefetch.py
+++ b/nodes/_otr_kokoro_voice_prefetch.py
@@ -31,9 +31,22 @@ from __future__ import annotations
 import logging
 import os
 
+import os as _os_boot
+import sys as _sys_boot
+
+# STANDALONE LOAD. `scripts/otr_provision.py`, `otr_make_portable_voice_bank.py`
+# and friends load this file by PATH under a non-package name, deliberately --
+# "without importing the ComfyUI node package". Neither ladder rung below can
+# resolve then: the relative one has no parent package and the flat one needs
+# `nodes/` on sys.path, which those loaders do not add. Same insert, and the
+# same reason, as `otr_post_upscale_procgen_blend.py`.
+_NODES_DIR = _os_boot.path.dirname(_os_boot.path.abspath(__file__))
+if _NODES_DIR not in _sys_boot.path:
+    _sys_boot.path.insert(0, _NODES_DIR)
+
 try:
     from ._otr_shared import env as otr_env
-except ImportError:  # pragma: no cover -- flat test imports
+except ImportError:  # pragma: no cover -- flat / standalone load
     from _otr_shared import env as otr_env  # type: ignore
 
 log = logging.getLogger("OTR")
diff --git a/nodes/_otr_model_catalog.py b/nodes/_otr_model_catalog.py
index 4b015daa..e046abdd 100644
--- a/nodes/_otr_model_catalog.py
+++ b/nodes/_otr_model_catalog.py
@@ -23,9 +23,22 @@ from dataclasses import dataclass
 from pathlib import Path
 from typing import Literal
 
+import os as _os_boot
+import sys as _sys_boot
+
+# STANDALONE LOAD. `scripts/otr_provision.py`, `otr_make_portable_voice_bank.py`
+# and friends load this file by PATH under a non-package name, deliberately --
+# "without importing the ComfyUI node package". Neither ladder rung below can
+# resolve then: the relative one has no parent package and the flat one needs
+# `nodes/` on sys.path, which those loaders do not add. Same insert, and the
+# same reason, as `otr_post_upscale_procgen_blend.py`.
+_NODES_DIR = _os_boot.path.dirname(_os_boot.path.abspath(__file__))
+if _NODES_DIR not in _sys_boot.path:
+    _sys_boot.path.insert(0, _NODES_DIR)
+
 try:
     from ._otr_shared import env as otr_env
-except ImportError:  # pragma: no cover -- flat test imports
+except ImportError:  # pragma: no cover -- flat / standalone load
     from _otr_shared import env as otr_env  # type: ignore
 
 # ---------------------------------------------------------------------------
diff --git a/nodes/_otr_shared/proc.py b/nodes/_otr_shared/proc.py
index dcc299e7..8a9cbdd6 100644
--- a/nodes/_otr_shared/proc.py
+++ b/nodes/_otr_shared/proc.py
@@ -142,6 +142,16 @@ def _no_shell(kwargs: dict) -> None:
     if kwargs.get("shell"):
         raise ExecutableNotAllowed(
             "shell=True is refused: it re-parses an argv list through a shell")
+    # `executable=` REPLACES the binary that actually runs while argv[0] keeps
+    # its old value -- so `run(["ffmpeg"], executable="cmd.exe")` would pass the
+    # allowlist on "ffmpeg" and launch cmd.exe. The check above it would be
+    # decoration. Nothing in this pack passes it; refusing it keeps the
+    # allowlist a boundary rather than a suggestion.
+    if kwargs.get("executable") is not None:
+        raise ExecutableNotAllowed(
+            "executable= is refused: it replaces the binary that runs while "
+            "argv[0] -- the thing the allowlist checked -- stays unchanged. "
+            "Put the real program in argv[0].")
 
 
 def run(argv: Sequence[Any], **kwargs) -> subprocess.CompletedProcess:
diff --git a/nodes/_otr_voice_bank.py b/nodes/_otr_voice_bank.py
index 8c62599d..9b382134 100644
--- a/nodes/_otr_voice_bank.py
+++ b/nodes/_otr_voice_bank.py
@@ -33,9 +33,22 @@ import random
 from dataclasses import dataclass
 from typing import List, Optional, Tuple
 
+import os as _os_boot
+import sys as _sys_boot
+
+# STANDALONE LOAD. `scripts/otr_provision.py`, `otr_make_portable_voice_bank.py`
+# and friends load this file by PATH under a non-package name, deliberately --
+# "without importing the ComfyUI node package". Neither ladder rung below can
+# resolve then: the relative one has no parent package and the flat one needs
+# `nodes/` on sys.path, which those loaders do not add. Same insert, and the
+# same reason, as `otr_post_upscale_procgen_blend.py`.
+_NODES_DIR = _os_boot.path.dirname(_os_boot.path.abspath(__file__))
+if _NODES_DIR not in _sys_boot.path:
+    _sys_boot.path.insert(0, _NODES_DIR)
+
 try:
     from ._otr_shared import env as otr_env
-except ImportError:  # pragma: no cover -- flat test imports
+except ImportError:  # pragma: no cover -- flat / standalone load
     from _otr_shared import env as otr_env  # type: ignore
 
 log = logging.getLogger("OTR")
diff --git a/nodes/otr_post_upscale_procgen_blend.py b/nodes/otr_post_upscale_procgen_blend.py
index 75730bfe..6e8fe09c 100644
--- a/nodes/otr_post_upscale_procgen_blend.py
+++ b/nodes/otr_post_upscale_procgen_blend.py
@@ -46,11 +46,6 @@ import sys
 from pathlib import Path
 from typing import Optional
 
-try:
-    from ._otr_shared import proc as otr_proc
-except ImportError:  # pragma: no cover -- flat test imports
-    from _otr_shared import proc as otr_proc  # type: ignore
-
 # Ensure sibling node modules (e.g. _otr_shared) resolve when this file is
 # loaded FLAT by ComfyUI's custom-node loader -- the flat fallbacks of the
 # two try/except imports below need it. Package-relative comes FIRST in
@@ -61,6 +56,11 @@ _NODES_DIR = os.path.dirname(os.path.abspath(__file__))
 if _NODES_DIR not in sys.path:
     sys.path.insert(0, _NODES_DIR)
 
+try:
+    from ._otr_shared import proc as otr_proc  # noqa: E402
+except ImportError:  # pragma: no cover -- flat / standalone load
+    from _otr_shared import proc as otr_proc  # type: ignore  # noqa: E402
+
 try:
     from ._otr_shared.ffmpeg import resolve_ffmpeg  # noqa: E402
 except ImportError:  # pragma: no cover -- flat (sys.path) load
diff --git a/tests/test_process_single_owner.py b/tests/test_process_single_owner.py
index fa1d595b..7c17e018 100644
--- a/tests/test_process_single_owner.py
+++ b/tests/test_process_single_owner.py
@@ -67,6 +67,7 @@ BLOCKED = {
         "the migration rides along with it, never ahead of it."),
 }
 
+
 def _subprocess_names(tree):
     """(module aliases, {bound name: spawn call}) resolved from this module.
 
@@ -87,6 +88,29 @@ def _subprocess_names(tree):
     return modules, bare
 
 
+def _owner_names(tree):
+    """Whatever name this module binds ``nodes/_otr_shared/proc.py`` to.
+
+    Needed because the owner RE-EXPORTS ``Popen`` as an identity alias, so
+    ``otr_proc.Popen(argv)`` is the raw stdlib class and skips the allowlist and
+    the shell/executable refusals entirely. The re-export has to stay -- an
+    ``except`` clause and a type annotation both need it -- so the guard has to
+    tell an annotation from a CALL. Found by a gpt-6-astra review, 2026-09-04."""
+    names = set()
+    for node in ast.walk(tree):
+        if isinstance(node, ast.Import):
+            for imported in node.names:
+                if imported.name == "proc":
+                    names.add(imported.asname or "proc")
+        elif isinstance(node, ast.ImportFrom):
+            module = node.module or ""
+            if module.endswith("_otr_shared") or (not module and node.level):
+                for imported in node.names:
+                    if imported.name == "proc":
+                        names.add(imported.asname or "proc")
+    return names
+
+
 def _os_spawn_names(tree):
     """(``os`` module names, {bound name: os spawn call}).
 
@@ -114,13 +138,30 @@ def _offenders(tree, rel):
     spawns, and the owner re-exports those names so they stay legal."""
     modules, bare = _subprocess_names(tree)
     os_modules, os_bare = _os_spawn_names(tree)
+    owners = _owner_names(tree)
     out = []
+    # `x = subprocess.run` then `x(...)` -- the rebinding Astra demonstrated.
+    # Flagged at the ASSIGNMENT, which is where it is readable.
+    for node in ast.walk(tree):
+        if (isinstance(node, ast.Assign)
+                and isinstance(node.value, ast.Attribute)
+                and node.value.attr in _SPAWN_CALLS
+                and isinstance(node.value.value, ast.Name)
+                and node.value.value.id in modules):
+            out.append(f"{rel}:{node.lineno} rebinds "
+                       f"{node.value.value.id}.{node.value.attr} to a local name")
     for node in ast.walk(tree):
         if not isinstance(node, ast.Call):
             continue
         func = node.func
         if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
-            if func.attr in _SPAWN_CALLS and func.value.id in modules:
+            if func.attr == "Popen" and func.value.id in owners:
+                # CALLING the re-export, not annotating with it. This is the
+                # raw stdlib class: no allowlist, no shell/executable refusal.
+                out.append(f"{rel}:{node.lineno} calls {func.value.id}.Popen() "
+                           "-- the raw re-export; use "
+                           f"{func.value.id}.popen() so the boundary applies")
+            elif func.attr in _SPAWN_CALLS and func.value.id in modules:
                 out.append(f"{rel}:{node.lineno} calls "
                            f"{func.value.id}.{func.attr}()")
             elif func.attr in _OS_SPAWN_CALLS and func.value.id in os_modules:
@@ -214,3 +255,48 @@ def test_every_blocked_file_is_still_pending_and_still_offends():
             "%s is BLOCKED but not PENDING -- if it was migrated anyway, the "
             "reason above says what that costs" % rel)
         assert (REPO / rel).is_file(), rel
+
+
+def test_the_finder_catches_a_call_through_the_RE_EXPORT():
+    """`otr_proc.Popen(...)` is the raw stdlib class -- no allowlist, no shell or
+    executable refusal. The re-export itself must STAY (an `except` clause and a
+    type annotation both need it), so the rule keys on the CALL, not the name."""
+    for src in (
+            "from . import proc as otr_proc\notr_proc.Popen(['ffmpeg'])\n",
+            "from ._otr_shared import proc as otr_proc\notr_proc.Popen(a)\n",
+            "from _otr_shared import proc as p\np.Popen(a)\n",
+    ):
+        assert _find(src), src
+
+
+def test_the_re_export_is_still_legal_as_an_ANNOTATION_and_an_EXCEPT():
+    """The whole reason it is re-exported. Flagging these would push modules back
+    to importing subprocess just to name a type or catch an error."""
+    for src in (
+            "from . import proc as otr_proc\ndef f() -> otr_proc.Popen: ...\n",
+            "from . import proc as otr_proc\nx: otr_proc.Popen = None\n",
+            "from . import proc as otr_proc\ntry:\n    pass\n"
+            "except otr_proc.CalledProcessError:\n    pass\n",
+            "from . import proc as otr_proc\nx = otr_proc.PIPE\n",
+            "from . import proc as otr_proc\notr_proc.popen(['ffmpeg'])\n",
+    ):
+        assert _find(src) == [], src
+
+
+def test_the_finder_catches_a_REBOUND_spawn_function():
+    """`run = subprocess.run` then `run(argv)` -- flagged at the ASSIGNMENT,
+    which is where a reader can actually see it."""
+    assert _find("import subprocess\nrun = subprocess.run\nrun(['ffmpeg'])\n")
+    assert _find("import subprocess as sp\np = sp.Popen\n")
+
+
+# KNOWN LIMITS, stated rather than implied. This guard reads SPELLINGS, not
+# dataflow: `getattr(subprocess, "run")`, `importlib.import_module("subprocess")`
+# and a star-import would each get past it. That is deliberate -- a general
+# reflection analyser is not worth building for this migration, which was
+# gpt-6-astra's own recommendation on 2026-09-04 -- and none of those spellings
+# appears anywhere in the tree today. If one ever does, it belongs here as a
+# named case, not as a rewrite of the finder.
+def test_the_known_limits_are_recorded_next_to_the_guard():
+    import pathlib
+    assert "KNOWN LIMITS" in pathlib.Path(__file__).read_text(encoding="utf-8")
diff --git a/tests/test_shared_env_and_proc_owners.py b/tests/test_shared_env_and_proc_owners.py
index 622935ef..04217581 100644
--- a/tests/test_shared_env_and_proc_owners.py
+++ b/tests/test_shared_env_and_proc_owners.py
@@ -210,6 +210,26 @@ def test_an_empty_argv_is_refused_by_NAME_not_by_IndexError():
     assert "empty" in str(exc.value)
 
 
+@pytest.mark.parametrize("spawn", ["run", "popen"])
+def test_executable_override_is_refused_on_both_entry_points(spawn):
+    """`executable=` swaps the binary that actually runs while argv[0] keeps the
+    name the allowlist checked, so `(["ffmpeg"], executable="cmd.exe")` would
+    pass the check and launch something else. Found by a gpt-6-astra review of
+    the shipped migration; nothing in the pack passes it, which is exactly why
+    it has to be refused before something does."""
+    with pytest.raises(otr_proc.ExecutableNotAllowed) as exc:
+        getattr(otr_proc, spawn)(["ffmpeg", "-version"], executable="cmd.exe")
+    assert "executable=" in str(exc.value)
+
+
+def test_an_explicit_executable_None_is_still_fine():
+    """subprocess's own default. Refusing it would break a caller that passes
+    the parameter through from its own signature."""
+    done = otr_proc.run([sys.executable, "-c", "pass"], executable=None,
+                        stdout=otr_proc.DEVNULL, stderr=otr_proc.DEVNULL)
+    assert done.returncode == 0
+
+
 @pytest.mark.parametrize("spawn", ["run", "popen"])
 def test_shell_true_is_refused_on_both_entry_points(spawn):
     with pytest.raises(otr_proc.ExecutableNotAllowed) as exc:

```
