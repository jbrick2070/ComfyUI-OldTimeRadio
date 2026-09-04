"""ONE owner spawns a process: ``nodes/_otr_shared/proc.py``.

The registry scan of alpha.17 carried 35 `python_command_injection_risk`
findings -- this rule fires once per CALL SITE -- across twenty files. A human
reading that report has to judge thirty-five spawns to learn that this pack runs
ffmpeg. Collapsing them to the owner makes it two, and gives the pack one place
that answers "what may this process launch".

WHAT THIS GUARD ASSERTS: no file under ``nodes/`` calls ``subprocess.run``,
``Popen``, ``call``, ``check_call`` or ``check_output``, or ``os.system`` /
``os.popen``, for itself. It says nothing about what any caller RUNS -- every
caller keeps its own argv, because what a caller runs is part of what the caller
does.

NOT FINDINGS, deliberately: ``subprocess.PIPE`` and ``DEVNULL`` constants, a
``CompletedProcess`` annotation, and ``except subprocess.CalledProcessError``.
The owner re-exports all of those as IDENTITY aliases precisely so a migrated
module can drop its ``import subprocess`` without breaking an ``except`` clause,
and a rule that flagged them would push modules back to importing subprocess to
catch an error.

It is a shrinking named-set ratchet for the same reason the env guard is: the
migration runs over several commits across two boxes, so the guard ships first
and carries the not-yet-migrated files by name. See ``tests/fixtures/ratchet.py``.
"""
from __future__ import annotations

import ast

from tests.fixtures.ratchet import REPO, assert_ratchet, scan

NODES = REPO / "nodes"
OWNER = NODES / "_otr_shared" / "proc.py"

ROOTS = (NODES, REPO / "__init__.py", REPO / "prestartup_script.py")

#: The subprocess entry points that start a child.
_SPAWN_CALLS = frozenset({"run", "Popen", "call", "check_call", "check_output"})

#: The ``os`` spellings of the same act. ``os.popen`` runs through a shell, which
#: is the thing the owner refuses outright.
_OS_SPAWN_CALLS = frozenset({"system", "popen"})

ALLOWED = {
    "nodes/_otr_shared/proc.py": "the owner itself -- its two execution sites",
}

#: Not yet migrated. Shrinks in the same commit as each batch; never grows.
PENDING = {
    "nodes/_otr_audio_engines/eng_indextts2.py",
}


#: Files that still offend and MUST NOT be migrated yet, with the reason and
#: what unblocks each. They stay in PENDING because they do still offend; this
#: table exists so a later batch does not sweep one up mechanically and pay a
#: cost nobody priced.
BLOCKED = {
    "nodes/_otr_audio_engines/eng_indextts2.py": (
        "named in nodes/_otr_voice_route.py RUNTIME_FINGERPRINT_SOURCES, so "
        "ANY byte changed here moves the adapter's sha256 and DEMOTES the "
        "shipped Lemmy voice route to the ordinary draw until a GPU "
        "re-audition re-qualifies it. Migrating it cost 6 voice tests on "
        "2026-09-04 and would have cost the operator a voice he approved by "
        "ear. That module's own history records the same thing happening once "
        "before, for a COMMENT. UNBLOCKS: the next re-audition of that route -- "
        "the migration rides along with it, never ahead of it."),
}

def _subprocess_names(tree):
    """(module aliases, {bound name: spawn call}) resolved from this module.

    ``import subprocess as sp`` and ``from subprocess import Popen as P`` are
    both spellings of the same act, and a hard-coded list of bound names is
    one more thing to be blind to."""
    modules, bare = set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "subprocess":
                    modules.add(imported.asname or "subprocess")
        elif (isinstance(node, ast.ImportFrom) and node.module == "subprocess"
                and not node.level):
            for imported in node.names:
                if imported.name in _SPAWN_CALLS:
                    bare[imported.asname or imported.name] = imported.name
    return modules, bare


def _os_spawn_names(tree):
    """(``os`` module names, {bound name: os spawn call}).

    The second half covers ``from os import system`` -- nothing in the tree
    spells it that way today, which is exactly why it is covered now rather
    than after the first one does. The env guard already reads its half of
    ``os`` this way; leaving this one asymmetric would be a hole shaped like a
    single import line."""
    modules, bare = set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "os" or imported.name.startswith("os."):
                    modules.add(imported.asname or "os")
        elif (isinstance(node, ast.ImportFrom) and node.module == "os"
                and not node.level):
            for imported in node.names:
                if imported.name in _OS_SPAWN_CALLS:
                    bare[imported.asname or imported.name] = imported.name
    return modules, bare


def _offenders(tree, rel):
    """Only CALLS. A constant, an annotation and an ``except`` clause are not
    spawns, and the owner re-exports those names so they stay legal."""
    modules, bare = _subprocess_names(tree)
    os_modules, os_bare = _os_spawn_names(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.attr in _SPAWN_CALLS and func.value.id in modules:
                out.append(f"{rel}:{node.lineno} calls "
                           f"{func.value.id}.{func.attr}()")
            elif func.attr in _OS_SPAWN_CALLS and func.value.id in os_modules:
                out.append(f"{rel}:{node.lineno} calls "
                           f"{func.value.id}.{func.attr}()")
        elif isinstance(func, ast.Name):
            if func.id in bare:
                out.append(f"{rel}:{node.lineno} calls "
                           f"subprocess.{bare[func.id]}() (bound as {func.id})")
            elif func.id in os_bare:
                out.append(f"{rel}:{node.lineno} calls "
                           f"os.{os_bare[func.id]}() (bound as {func.id})")
    return out


_HINT = ("ask nodes/_otr_shared/proc.py instead: `otr_proc.run(argv, ...)` and "
         "`otr_proc.popen(argv, ...)` forward every keyword and return the real "
         "subprocess objects. Import it ALIASED at your own depth -- `proc` is a "
         "live local name in eleven files. Use the MODULE form so a test can "
         "still patch `<module>.otr_proc.run`.")


def test_the_owner_exists():
    assert OWNER.is_file(), OWNER


def test_the_allowlist_says_why_for_every_entry():
    for rel, reason in ALLOWED.items():
        assert (REPO / rel).is_file(), rel
        assert reason.strip(), rel


def test_no_second_spawner_under_nodes():
    assert_ratchet(scan(ROOTS, _offenders), set(PENDING) | set(ALLOWED),
                   owner_hint=_HINT)


def test_the_pending_set_and_the_allowlist_do_not_overlap():
    assert not (set(PENDING) & set(ALLOWED))


# --------------------------------------------------------------------------- #
# the finder itself, on synthetic source
# --------------------------------------------------------------------------- #
def _find(src):
    return _offenders(ast.parse(src), "probe.py")


def test_the_finder_catches_every_spawn_spelling():
    for src in (
            "import subprocess\nsubprocess.run(['ffmpeg'])\n",
            "import subprocess\nsubprocess.Popen(['ffmpeg'])\n",
            "import subprocess\nsubprocess.call(['ffmpeg'])\n",
            "import subprocess\nsubprocess.check_call(['ffmpeg'])\n",
            "import subprocess\nsubprocess.check_output(['ffmpeg'])\n",
            "import subprocess as sp\nsp.run(['ffmpeg'])\n",
            "from subprocess import Popen\nPopen(['ffmpeg'])\n",
            "from subprocess import run as r\nr(['ffmpeg'])\n",
            "import os\nos.system('ffmpeg')\n",
            "import os\nos.popen('ffmpeg')\n",
            "from os import system\nsystem('ffmpeg')\n",
            "from os import popen as p\np('ffmpeg')\n",
    ):
        assert _find(src), src


def test_the_finder_leaves_the_re_exported_names_alone():
    """These are exactly what the owner re-exports, so flagging them would push
    a migrated module back to importing subprocess to catch its own errors."""
    for src in (
            "import subprocess\nx = subprocess.PIPE\n",
            "import subprocess\nx = subprocess.DEVNULL\n",
            "import subprocess\ndef f() -> subprocess.CompletedProcess: ...\n",
            "import subprocess\ntry:\n    pass\n"
            "except subprocess.CalledProcessError:\n    pass\n",
            "import subprocess\ntry:\n    pass\n"
            "except subprocess.TimeoutExpired:\n    pass\n",
            "from . import proc as otr_proc\notr_proc.run(['ffmpeg'])\n",
            "s = 'subprocess.run(argv)'\n",
            "import os\nos.path.run\n",
    ):
        assert _find(src) == [], src


def test_every_blocked_file_is_still_pending_and_still_offends():
    """A blocked file that stopped offending, or quietly left PENDING, means
    this table is describing a world that no longer exists."""
    for rel, reason in BLOCKED.items():
        assert reason.strip(), rel
        assert rel in PENDING, (
            "%s is BLOCKED but not PENDING -- if it was migrated anyway, the "
            "reason above says what that costs" % rel)
        assert (REPO / rel).is_file(), rel
