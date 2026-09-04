"""ONE owner answers what an environment knob says: ``nodes/_otr_shared/env.py``.

The registry scan of alpha.17 carried 103 `python_environment_manipulation`
findings -- the rule fires once per FILE -- and every one of them was tagged
`credential-access`. A human asked to review that report has to read a hundred
and three lines to learn that this pack reads its own `OTR_*` knobs. Collapsing
the spelling to one module makes it one line, and makes "what does this knob
mean" a question with one place to look.

WHAT THIS GUARD ASSERTS, and it is only the spelling: no file under ``nodes/``
(plus the two root modules, which a ``nodes/`` rglob cannot see) reaches
``os.environ``, ``os.getenv``, ``os.putenv`` or ``os.unsetenv`` for itself. It
says nothing about defaults, casts or precedence -- those stay at the call site,
which is exactly what makes a hundred-file migration safe to do in one pass.

WHY IT IS A RATCHET AND SHIPS FIRST. The migration touches a hundred files over
several commits, and two boxes push to ``v2.0-alpha``. A guard that arrived on
the last day would protect nothing in between. So it ships with the owners,
carrying the not-yet-migrated files by NAME, and asserts both directions every
run: nothing outside the set offends, and every file inside it still does. A
shrinking COUNT would let one file be fixed while another grew a new read; a
named set cannot, because the new offender is outside it.

AST, NEVER TEXT. ``os`` is aliased in six files (``__init__.py:510`` binds it as
``_otr_ro``), the subscript form is a read (``eng_humo.py``, ``eng_mesh_stage.py``),
and this module's own docstring says ``os.environ`` several times -- a source grep
cannot tell a call from prose about a call, and the version that cries wolf is
the version that gets deleted.
"""
from __future__ import annotations

import ast

from tests.fixtures.ratchet import REPO, assert_ratchet, scan

NODES = REPO / "nodes"
OWNER = NODES / "_otr_shared" / "env.py"

#: Everything under nodes/, plus the two root modules by explicit path. Both
#: ship in the registry zip and both touch the environment.
ROOTS = (NODES, REPO / "__init__.py", REPO / "prestartup_script.py")

#: The ``os`` attributes that read or write the process environment. ``environb``
#: is deliberately absent: it does not exist on Windows and nothing here uses it;
#: if a POSIX-only site ever wants it, it belongs on this list with the rest.
_ENV_ATTRS = frozenset({"environ", "getenv", "putenv", "unsetenv"})

#: Files that keep their own spelling, each with the reason the owner cannot
#: serve them. Retiring one shrinks this dict; a new one is a reviewed decision.
ALLOWED = {
    "nodes/_otr_shared/env.py": "the owner itself",
    "prestartup_script.py": (
        "runs before the pack is a package -- there is no package context to "
        "import an owner from -- so it keeps its inline writes by decision, "
        "and stays exactly one finding"),
}

#: Files not yet migrated to the owner. This set SHRINKS, in the same commit as
#: the batch that migrates its members, until it is empty. It never grows: a new
#: `os.environ` site under nodes/ lands outside it and fails on the next run,
#: including on the other box's next pull, which is the intended behaviour.
PENDING = {
    "nodes/_otr_audio_engines/eng_indextts2.py",
    "nodes/_otr_writer_heartbeat.py",
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
    "nodes/_otr_writer_heartbeat.py": (
        "a LEAF by contract: tests/test_writer_heartbeat_is_visible.py asserts "
        "it contains no `from ._otr` import at all, because a pack import "
        "reintroduces the cycle that once left two of three local generate "
        "transports running BLIND (no streamer, no visible progress). Migrating "
        "it added `from ._otr_shared import env as otr_env` and turned that "
        "guard red on 2026-09-04. The owner IS a stdlib-only leaf and could not "
        "actually cycle, but the guard is deliberately stricter than the hazard "
        "and weakening a safeguard to remove one finding is the trade the "
        "operator has already refused elsewhere. UNBLOCKS: never, unless that "
        "guard is deliberately narrowed on its own merits -- which is its own "
        "design item, not a migration detail."),
}


def _os_aliases(tree):
    """Every name this module binds the ``os`` MODULE to.

    ``import os`` binds ``os``; ``import os as _otr_ro`` (``__init__.py:510``)
    binds ``_otr_ro``; ``import os.path`` binds ``os`` as well. Resolving this
    from the module's own imports is what keeps the rule from being one more
    spelling to be blind to."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "os" or imported.name.startswith("os."):
                    names.add(imported.asname or "os")
    return names


def _names_from_os(tree):
    """``from os import environ as X`` -> {X: "environ"}.

    Nothing under ``nodes/`` does this today, and that is precisely why it is
    covered now rather than after the first one does."""
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os" and not node.level:
            for imported in node.names:
                if imported.name in _ENV_ATTRS:
                    bound[imported.asname or imported.name] = imported.name
    return bound


def _offenders(tree, rel):
    """Every site in ``tree`` that decides an environment read for itself.

    A bare local named ``environ`` is NOT one. ``motion_common.py`` binds
    ``environ = os.environ if env is None else env`` to a CALLER-SUPPLIED
    mapping and then reads that local; a name-only rule would flag every one of
    those reads falsely. The ``os.environ`` on that line's right-hand side is a
    finding, and it is the only one there."""
    aliases = _os_aliases(tree)
    from_os = _names_from_os(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os" and not node.level:
            out.extend(f"{rel}:{node.lineno} imports os.{a.name} directly"
                       for a in node.names if a.name in _ENV_ATTRS)
        elif (isinstance(node, ast.Attribute) and node.attr in _ENV_ATTRS
                and isinstance(node.value, ast.Name) and node.value.id in aliases):
            # Load OR Store: `os.environ["X"]` on either side of an `=` is a
            # site, and a Load-only rule misses the subscript writes.
            out.append(f"{rel}:{node.lineno} reads {node.value.id}.{node.attr}")
        elif isinstance(node, ast.Name) and node.id in from_os:
            out.append(f"{rel}:{node.lineno} uses os.{from_os[node.id]} "
                       f"(bound as {node.id})")
    return out


_HINT = ("ask nodes/_otr_shared/env.py instead: `otr_env.get(name, default)` "
         "keeps your default and your cast exactly where they are. Import it "
         "ALIASED at your own depth -- `env` is a live parameter name.")


def test_the_owner_exists():
    assert OWNER.is_file(), OWNER


def test_the_allowlist_says_why_for_every_entry():
    """An exception with no reason is an exception nobody can retire."""
    for rel, reason in ALLOWED.items():
        assert (REPO / rel).is_file(), rel
        assert reason.strip(), rel


def test_no_second_owner_of_the_environment_under_nodes():
    assert_ratchet(scan(ROOTS, _offenders), set(PENDING) | set(ALLOWED),
                   owner_hint=_HINT)


def test_the_pending_set_and_the_allowlist_do_not_overlap():
    """A file cannot be both permanently excused and temporarily behind."""
    assert not (set(PENDING) & set(ALLOWED))


# --------------------------------------------------------------------------- #
# the finder itself, on synthetic source -- a guard that can go blind is worse
# than no guard, because it reads as coverage
# --------------------------------------------------------------------------- #
def _find(src):
    return _offenders(ast.parse(src), "probe.py")


def test_the_finder_catches_every_spelling_in_the_tree():
    for src in (
            "import os\nos.environ.get('A')\n",
            "import os\nos.environ['A'] = '1'\n",          # Store, by subscript
            "import os\nx = os.environ['A']\n",            # Load, by subscript
            "import os\nos.environ.pop('A', None)\n",
            "import os\nos.environ.setdefault('A', '1')\n",
            "import os\nos.getenv('A')\n",
            "import os\nos.putenv('A', '1')\n",
            "import os\nos.unsetenv('A')\n",
            "import os as _otr_ro\n_otr_ro.environ.get('A')\n",   # renamed os
            "import os.path as _p\nimport os\nos.environ.get('A')\n",
            "from os import environ\nenviron.get('A')\n",
            "from os import getenv as ge\nge('A')\n",
    ):
        assert _find(src), src


def test_the_finder_does_not_cry_wolf():
    for src in (
            "environ = {}\nenviron.get('A')\n",            # a bare local
            "def f(environ):\n    return environ.get('A')\n",
            "import os\nx = os.path.join('a', 'b')\nos.name\n",
            "s = 'os.environ.get'\n",                      # prose about a call
            "'''os.environ.get(\"A\") in a docstring'''\n",
            "from . import env as otr_env\notr_env.get('A')\n",   # the migrated form
            # the caller-supplied mapping motion_common actually binds
            "import _m\ndef f(env=None):\n"
            "    environ = _m.d if env is None else env\n"
            "    return environ.get('A')\n",
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
