"""A node WIDGET may not name the binary this pack spawns.

THE DEFECT THIS CLOSES, reproduced against the real modules on 2026-09-04.
Five shipped nodes exposed an ``ffmpeg`` STRING widget. A widget value arrives
in the body of ComfyUI's ``/prompt`` request -- unauthenticated by default --
and is whatever a downloaded workflow JSON says. It reached ``argv[0]``:

    widget -> _ffmpeg_bin(ffmpeg) -> resolve_ffmpeg(preferred)
           -> _explicit()  honoured it (it carried a directory)
           -> _usable()    honoured it (the file existed)
           -> proc.py      allowed it (argv[0]'s BASENAME is "ffmpeg")
           -> otr_proc.run EXECUTED IT

Measured with ``OTR_FFMPEG`` pinned to a real binary, a widget value of
``<tmp>\\ffmpeg.exe`` BEAT the operator's pin, and ``resolve_ffprobe(ffmpeg=...)``
produced a SECOND attacker binary through the sibling rule.

WHERE THE FIX LIVES, and why not the two obvious places:

* NOT in ``_explicit``: provenance is invisible there. Trusted internal callers
  legitimately pass directory-bearing arguments -- ``blend()`` resolves ffmpeg
  and threads that RESOLVED path through ``_probe_dims`` -> ``probe_raw`` ->
  ``resolve_ffprobe(ffmpeg=...)``.
* NOT in ``_ffmpeg_bin``: ``otr_master_audio_mux`` deliberately hands its
  ALREADY-RESOLVED binary to ``audio_pcm_sha`` so the byte-identity proof cannot
  resolve differently from the encode that just ran.
* NOT as an "argv[0] must be absolute" gate in ``proc.py``: that owner
  deliberately admits bare ``git`` and ``nvidia-smi``
  (``production_ledger.py``, ``_otr_ledger.py``, ``_otr_sys_specs.py``), each
  wrapped in ``except Exception`` -> "unknown". Such a gate would blank the
  ledger's commit stamp on EVERY episode, with a green run and a published obs
  artifact, invisible to the whole suite.

So the 2026-09-04 fix DISCARDED the widget at each node's EXECUTE METHOD, and
the resolvers were made to return an ABSOLUTE path or ``None``.

THE WIDGET ITSELF WAS REMOVED ON 2026-09-13, and that is what the first half of
this file now asserts. The field is no longer DECLARED by any of the five node
classes, so ComfyUI never passes it and there is nothing left to sanitise: the
channel is CLOSED by non-declaration rather than guarded by a discard, which is
strictly stronger. A discard can be forgotten at a sixth call site; an input
that is not declared cannot be supplied at all. ``widget_ffmpeg_is_ignored``
was removed with it, deliberately -- a sanitiser left lying around invites
someone to re-add the widget and "handle" it, which is the weaker design this
change replaced. ``OTR_FFMPEG`` remains the one way to pin a binary.

THE RESOLVER RULES ABOVE ARE UNCHANGED, and every one of their tests stays.
The resolvers are still reachable from the pack's own callers, so
absolute-path-or-``None`` is still what keeps a cwd hit or a bare name out of
``argv[0]``.

FOUR GAPS AN INDEPENDENT REVIEW FOUND IN THIS FILE, all closed here:

1. The blend's live security test asserted ``_ffmpeg_bin(x) == ""`` while
   neutralising ONE of four fallbacks, so it was really claiming "this box has
   no ffmpeg at all" -- false on any box with imageio-ffmpeg installed, and the
   property it meant to prove (the answer is never the ARGUMENT) went untested.
   It is now two tests: the non-echo one, which needs no neutralisation and so
   always runs, and the empty-answer one, which neutralises every fallback and
   proves it did.
2. Removing the widget left four of the five nodes binding ``ffmpeg = ""`` with
   nothing asserting it -- only the scopes node kept a binding check. The walk
   is now parametrized over all five.
3. The live-class guard parametrized over ``NODE_CLASS_MAPPINGS``, which
   ``__init__.py`` builds node-by-node in its own try/except: a missing optional
   dependency SHRINKS that list and the guard still reports green over the
   survivors. A floor now fails instead.
4. The sanitiser scan swallowed a ``SyntaxError``, read ``nodes/`` only, and
   walked names but not strings, so ``getattr(mod, "widget_ffmpeg_is_ignored")``
   was invisible to it.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import pathlib
import shutil
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The five nodes whose `ffmpeg` widget used to reach argv[0].
WIDGET_NODES = {
    "nodes/otr_caption_burn.py": "OTR_CaptionBurn",
    "nodes/otr_master_audio_mux.py": "OTR_MasterAudioMux",
    "nodes/otr_post_upscale_procgen_blend.py": "OTR_PostUpscaleProcgenBlend",
    "nodes/otr_scene_aware_scopes.py": "OTR_SceneAwareScopes",
    "nodes/otr_silent_composite.py": "OTR_SilentComposite",
}

#: The helper that was removed with the widget, spelled once.
SANITISER = "widget_ffmpeg_is_ignored"

#: How many declared nodes may be absent from `NODE_CLASS_MAPPINGS` before the
#: registry has COLLAPSED rather than merely thinned. `__init__.py` imports each
#: node in its own try/except and prints `Skipped '<name>': <reason>`, so a box
#: without some optional dependency legitimately registers fewer than the
#: manifest declares -- but a walk over three survivors is reporting green over
#: almost nothing. Read against the live manifest so adding a node raises the
#: floor by itself; 25 declared - 5 = 20, which is the same floor
#: `tests/test_node_list_manifest.py` pins for the same "empty cannot look like
#: agreement" reason.
MAX_OPTIONAL_SKIPS = 5

#: A directory no install puts a binary in, so every value below names
#: something that does NOT exist on any box this suite runs on. That matters:
#: an absolute path to a file that DOES exist is honoured by design (trusted
#: internal callers pass resolved paths, see the header), so a test built on an
#: existing file would assert the opposite of the contract. What is being
#: proven here is narrower and is the real property: a value the resolver
#: REFUSED is never handed back.
HOSTILE_DIR = r"C:\Users\victim\Downloads\payload"
HOSTILE = HOSTILE_DIR + r"\ffmpeg.exe"

HOSTILE_VALUES = (
    HOSTILE,                                        # the measured 2026-09-04 shape
    HOSTILE_DIR + "/ffmpeg",                        # mixed separators, no suffix
    HOSTILE_DIR.replace("\\", "/") + "/ffmpeg.exe",  # posix spelling
    "ffmpeg",                                       # the bare name
    "ffmpeg.exe",
    r".\ffmpeg.exe",                                # an explicit cwd hit
    "bin/ffmpeg",                                   # relative, directory-bearing
    "anything at all",
)


def _mappings() -> dict:
    """The pack's registered nodes, imported the way ComfyUI imports them.

    Same route as tests/test_input_types_signature_parity.py -- the package
    directory is hyphenated, so it is imported by name from its parent rather
    than through an ordinary `import`.
    """
    sys.path.insert(0, str(REPO.parent))
    try:
        pkg = importlib.import_module(REPO.name)
    finally:
        sys.path.pop(0)
    return dict(getattr(pkg, "NODE_CLASS_MAPPINGS", {}) or {})


NODE_CLASS_MAPPINGS = _mappings()


def _declared_input_names(cls) -> set:
    spec = cls.INPUT_TYPES()
    names = set()
    for section in ("required", "optional", "hidden"):
        block = spec.get(section) or {}
        if isinstance(block, dict):
            names |= set(block.keys())
    return names


#: Every tree that can hold a Python definition or reference this guard must
#: see. `tests/` is the one deliberate exclusion -- see the docstring.
_SCANNED_TREES = ("nodes", "scripts", "config", "tools")


def _python_sources() -> list:
    """Every .py that could define or reference the removed sanitiser.

    AN EARLIER VERSION OF THIS SCANNED `nodes/` + `scripts/` + the repo root and
    justified it as "every .py this pack SHIPS AND RUNS". That justification was
    INVERTED AT BOTH ENDS, and `.comfyignore` says so in its own words:

      * `scripts/` DOES NOT SHIP. `.comfyignore` excludes the whole tree --
        "NOTHING IN THE PACK IMPORTS scripts/" -- so it is dev harness, not
        shipped surface. Worth scanning anyway, because it runs on our boxes.
      * `config/` DOES SHIP and was NOT SCANNED. A contrarian planted the helper
        at `config/cast_pools.py` and this guard passed, blind.

    So the criterion is not "does it ship" and not "does it run" -- it is "could
    a definition or a reference live here", which is every tracked Python tree
    except the one below. Measured when written: nodes 284, scripts 123,
    config 2, tools 6, root 2.

    `tests/` is excluded ON PURPOSE and it is the one honest exclusion: this
    file names the removed helper in its own assertions, and a test that named
    it would fail at runtime rather than resurrect it.
    """
    found = []
    for folder in _SCANNED_TREES:
        root = REPO / folder
        if root.is_dir():
            found += sorted(root.rglob("*.py"))
    found += sorted(REPO.glob("*.py"))
    return [p for p in found if "__pycache__" not in p.parts]


def _execute_def(rel, cls):
    """The AST of `cls`'s own execute method, found through the LIVE class.

    Keyed on `cls.__name__` and `cls.FUNCTION` rather than a literal, so a free
    function of the same name elsewhere in the module cannot be walked by
    mistake and a renamed method is a loud failure rather than a quiet skip.
    The registry key and the class name differ here (`OTR_CaptionBurn` vs
    `OTRCaptionBurn`), which is exactly why this asks the class.
    """
    path = REPO / rel
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    klass = next((n for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and n.name == cls.__name__), None)
    assert klass is not None, (
        "%s defines no class %s -- this walk cannot find the execute method"
        % (rel, cls.__name__))
    defs = [n for n in klass.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == cls.FUNCTION]
    assert len(defs) == 1, (
        "%s.%s declares FUNCTION=%r and %d method(s) of that name"
        % (rel, cls.__name__, cls.FUNCTION, len(defs)))
    return defs[0]


def _is_empty_string_constant(value) -> bool:
    return (isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value == "")


#: The pack's own answer to "which binary", and the only thing other than the
#: empty constant a node's `ffmpeg` local may hold.
TRUSTED_RESOLVERS = ("_ffmpeg_bin", "resolve_ffmpeg")


def _is_trusted_resolution(value) -> bool:
    """`ffmpeg = _ffmpeg_bin(ffmpeg)` -- the resolver's answer, computed from
    the already-empty local and constants, so nothing new enters.

    The blend node does exactly this at its own boundary: it binds the empty
    constant, then immediately replaces it with the resolved absolute path
    because every helper below it takes the binary as an argument. That is a
    rebinding of a value this test has ALREADY proven empty, not a new channel.
    """
    if not isinstance(value, ast.Call):
        return False
    name = getattr(value.func, "id", None) or getattr(value.func, "attr", None)
    if name not in TRUSTED_RESOLVERS:
        return False
    supplied = list(value.args) + [kw.value for kw in value.keywords]
    return all((isinstance(a, ast.Name) and a.id == "ffmpeg")
               or isinstance(a, ast.Constant) for a in supplied)


# --------------------------------------------------------------------------- #
# the channel is CLOSED -- the widget is not declared, not accepted, and the
# sanitiser that used to stand in for its absence is gone
# --------------------------------------------------------------------------- #
def test_the_registry_this_guard_walks_cannot_silently_shrink():
    """`NODE_CLASS_MAPPINGS` is ENVIRONMENT-DEPENDENT, and the guard below
    parametrizes over it.

    `__init__.py` imports each node inside its own try/except -- deliberate
    partial-install resilience -- and DROPS one whose import raises. So on a box
    missing an optional dependency the parametrized list quietly shortens and
    the guard still reports green, over whatever happened to survive. If every
    import failed it would collect nothing at all and pass in silence. A guard
    that can shrink to nothing is not a guard, so the shrink is the failure.
    """
    declared = json.loads((REPO / "node_list.json").read_text(encoding="utf-8"))
    registered = set(NODE_CLASS_MAPPINGS)

    absent = sorted(set(WIDGET_NODES.values()) - registered)
    assert not absent, (
        "the node(s) this whole file is about are not registered: %s. These "
        "five are the classes that declared the `ffmpeg` widget, so a run "
        "without them proves nothing about the channel that mattered. Read the "
        "`[OldTimeRadio] Skipped '<name>': <reason>` lines the loader prints "
        "and fix the import -- do not let the guard pass over the remainder."
        % ", ".join(absent))

    floor = max(len(WIDGET_NODES), len(declared) - MAX_OPTIONAL_SKIPS)
    assert len(registered) >= floor, (
        "only %d of the %d nodes node_list.json declares are registered, below "
        "the floor of %d. That is a collapsed registry rather than a thinned "
        "one, and every test in this file that parametrizes over the live "
        "mappings is now asking about a rump of the pack."
        % (len(registered), len(declared), floor))


@pytest.mark.parametrize("rel", sorted(WIDGET_NODES))
def test_no_file_still_declares_the_widget(rel):
    """The declaration is what made the value arrive. Removed 2026-09-13.

    Parametrized over the CONSTANT above rather than over the live registry, so
    this half of the proof holds even on a box where a node fails to import.
    """
    src = (REPO / rel).read_text(encoding="utf-8")
    assert '"ffmpeg": ("STRING"' not in src, (
        "%s declares an ffmpeg STRING widget again. A widget value arrives "
        "from an unauthenticated /prompt body and used to reach argv[0]; "
        "OTR_FFMPEG is the only channel that may name a binary." % rel)


@pytest.mark.parametrize("node_name", sorted(NODE_CLASS_MAPPINGS))
def test_no_node_class_declares_an_input_named_ffmpeg(node_name):
    """Asked of the LIVE class, not of the source text.

    A grep for `"ffmpeg": ("STRING"` sees only a literal spelling. INPUT_TYPES
    is a classmethod that may build its dict from a loop, a constant, or a
    profile, so a computed key would pass the grep and still hand ComfyUI an
    `ffmpeg` keyword argument. This asks the pack what it actually declares.

    The list it walks can shrink; `test_the_registry_this_guard_walks_cannot_
    silently_shrink` above is what stops that being silent.
    """
    cls = NODE_CLASS_MAPPINGS[node_name]
    if not hasattr(cls, "INPUT_TYPES"):
        pytest.skip("%s declares no INPUT_TYPES" % node_name)
    declared = _declared_input_names(cls)
    assert "ffmpeg" not in declared, (
        "%s declares an input named 'ffmpeg'. ComfyUI passes declared inputs "
        "as keyword arguments straight from the /prompt body, which is how "
        "the value reached argv[0] before 2026-09-04." % node_name)


@pytest.mark.parametrize("rel,node", sorted(WIDGET_NODES.items()))
def test_the_execute_method_accepts_no_ffmpeg_keyword(rel, node):
    """The plumbing may not come back quietly behind a missing declaration.

    Removing the declaration alone leaves a parameter that any future
    `INPUT_TYPES` edit -- or any internal caller -- can fill again. Closing
    BOTH ends means there is nowhere for a binary name to enter.
    """
    cls = NODE_CLASS_MAPPINGS[node]
    fn = getattr(cls, cls.FUNCTION, None)
    assert fn is not None, "%s.FUNCTION=%r names no method" % (node, cls.FUNCTION)
    sig = inspect.signature(fn)
    assert "ffmpeg" not in sig.parameters, (
        "%s.%s() still accepts an `ffmpeg` parameter (%s). The declaration is "
        "gone, so nothing fills it today -- but the plumbing underneath is "
        "what carried the value to argv[0], and leaving it wired means one "
        "INPUT_TYPES line reopens the channel."
        % (rel, cls.FUNCTION, sig))
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in sig.parameters.values()), (
        "%s.%s() takes **kwargs, which absorbs an `ffmpeg` keyword without "
        "naming it -- this test can no longer see the channel." % (rel, cls.FUNCTION))


@pytest.mark.parametrize("rel,node", sorted(WIDGET_NODES.items()))
def test_every_widget_node_binds_the_local_to_the_empty_constant(rel, node):
    """WHAT the local named `ffmpeg` HOLDS, in all five files.

    The deleted `test_every_widget_node_discards_at_its_execute_method` pinned
    this and its replacement did not: proving no `ffmpeg` PARAMETER exists says
    nothing about what the body then binds to that name, and four of the five
    bind it with nothing watching (`otr_caption_burn:535`,
    `otr_master_audio_mux:1704`, `otr_post_upscale_procgen_blend:914`,
    `otr_silent_composite:1764`). Only the scopes node kept a check.

    THE PROPERTY: the first binding is the empty constant, it happens before
    the name is read, and any later binding is the pack's own resolver fed by
    that already-empty local -- which is what the blend does at :929, because
    every helper below it takes the binary as an argument.

    Line order stands in for execution order here, as it did in the scopes walk
    this generalises. It is a proxy, and it is the right one: the bindings are
    straight-line statements at the top of each method, and a future edit that
    made the order conditional would be exactly the change worth failing on.
    """
    cls = NODE_CLASS_MAPPINGS[node]
    fn = _execute_def(rel, cls)
    where = "%s.%s()" % (rel, cls.FUNCTION)

    params = [a.arg for a in list(fn.args.posonlyargs) + list(fn.args.args)
              + list(fn.args.kwonlyargs)]
    assert "ffmpeg" not in params, (
        "%s accepts an `ffmpeg` parameter again -- ComfyUI would fill it from "
        "the /prompt body the moment it is declared" % where)
    assert fn.args.kwarg is None, (
        "%s takes **kwargs, which can carry an `ffmpeg` value past this check "
        "and into every consumer below it" % where)

    scoped = [n for n in ast.walk(fn)
              if isinstance(n, (ast.Global, ast.Nonlocal)) and "ffmpeg" in n.names]
    assert not scoped, (
        "%s declares `ffmpeg` global or nonlocal at line %s -- the name then "
        "belongs to another scope and nothing here can prove what it holds"
        % (where, [n.lineno for n in scoped]))

    stores, loads = [], []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id == "ffmpeg":
            (stores if isinstance(n.ctx, (ast.Store, ast.Del)) else loads).append(n)

    bound = {}
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "ffmpeg"):
            bound[id(n.targets[0])] = n.value
        elif (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                and n.target.id == "ffmpeg" and n.value is not None):
            bound[id(n.target)] = n.value

    if not stores:
        # A future rewrite may drop the local entirely, and that is FINE -- a
        # name that is never bound and never read is the strongest form of
        # closed. What is not fine is reading it from somewhere else.
        assert not loads, (
            "%s reads a name `ffmpeg` it never binds -- it is coming from an "
            "enclosing or module scope, and every consumer is downstream of "
            "whatever that holds (first read at line %d)"
            % (where, min(n.lineno for n in loads)))
        return

    opaque = sorted(n.lineno for n in stores if id(n) not in bound)
    assert not opaque, (
        "%s binds `ffmpeg` at line(s) %s by a form this walk cannot read -- a "
        "tuple unpack, a loop target, a `with ... as`, a walrus, an augmented "
        "assignment or a `del`. Only a plain `ffmpeg = <expr>` can be proven "
        "to hold the empty constant, so spell it that way." % (where, opaque))

    first = min(stores, key=lambda n: n.lineno)
    assert _is_empty_string_constant(bound[id(first)]), (
        "%s binds `ffmpeg` to something other than the empty constant at line "
        "%d. That first binding is the whole severing: whatever it holds is "
        "what reaches the ffmpeg resolver and then argv[0]."
        % (where, first.lineno))

    for n in stores:
        value = bound[id(n)]
        assert (_is_empty_string_constant(value)
                or _is_trusted_resolution(value)), (
            "%s rebinds `ffmpeg` at line %d to something that is neither the "
            "empty constant nor a call to %s fed by the empty local. Only the "
            "pack's own resolver may name the binary."
            % (where, n.lineno, " or ".join(TRUSTED_RESOLVERS)))

    assert loads, (
        "%s binds `ffmpeg` at line %d and never reads it. The binding is dead, "
        "so the ordering proof below it is vacuous -- either a consumer was "
        "removed and the local should go with it, or one was renamed."
        % (where, first.lineno))
    assert first.lineno < min(n.lineno for n in loads), (
        "%s reads `ffmpeg` at line %d but does not bind it until line %d"
        % (where, min(n.lineno for n in loads), first.lineno))


def test_the_sanitiser_itself_is_gone():
    """A discard helper with no widget to discard is an invitation.

    It reads as permission to re-declare the widget and "handle" it at the
    node boundary, which is exactly the weaker design 2026-09-13 replaced:
    sanitising a channel is a promise kept at every call site, closing it is a
    property of the declaration.
    """
    from nodes._otr_shared import ffmpeg as ffm
    assert not hasattr(ffm, SANITISER), (
        "nodes/_otr_shared/ffmpeg.py still exports %s. The widget it sanitised "
        "no longer exists; the helper must go with it so nobody re-adds the "
        "widget and points at the helper as cover." % SANITISER)
    src = (REPO / "nodes/_otr_shared/ffmpeg.py").read_text(encoding="utf-8")
    assert "def %s" % SANITISER not in src

    # Asked of the PARSED tree, not of the text. The owner module keeps a
    # tombstone comment explaining what stood there and why non-declaration
    # replaced it -- that history is the point of the comment, and a substring
    # scan would read it as a survivor.
    #
    # STRINGS ARE WALKED TOO. `getattr(mod, "widget_ffmpeg_is_ignored")` reaches
    # the helper without ever spelling it as a Name, so a name-only walk is
    # blind to the one shape a re-adder is most likely to reach for when the
    # import no longer resolves.
    sources = _python_sources()
    covered = {p.relative_to(REPO).as_posix() for p in sources}
    assert {"__init__.py", "nodes/_otr_shared/ffmpeg.py"} <= covered, (
        "the scan missed the loader or the module that owned the helper; it is "
        "globbing the wrong tree and would pass over anything")
    assert len(sources) > len(WIDGET_NODES), (
        "only %d source files found -- a scan that reaches almost nothing "
        "reports clean for the wrong reason" % len(sources))

    unparseable, live = [], []
    for path in sources:
        rel = path.relative_to(REPO).as_posix()
        try:
            # `filename=` so a SyntaxWarning raised while parsing the pack --
            # an invalid escape in some other module's docstring, say -- names
            # that module instead of reporting "<unknown>:38" from this test.
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                             filename=str(path))
        except SyntaxError as exc:
            # NOT `continue`. A file this scan cannot read is a file the removed
            # sanitiser could be hiding in, and skipping it turns an unreadable
            # module into a clean bill of health.
            unparseable.append("%s: %s" % (rel, exc))
            continue
        for node in ast.walk(tree):
            named = (
                (isinstance(node, ast.Name) and node.id)
                or (isinstance(node, ast.Attribute) and node.attr)
                or (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name)
                or (isinstance(node, ast.alias)
                    and (node.asname or node.name))
            )
            if named == SANITISER:
                live.append("%s:%d" % (rel, node.lineno))
            elif (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and SANITISER in node.value):
                live.append("%s:%d (named in a string)" % (rel, node.lineno))
    assert not unparseable, (
        "this scan could not parse:\n  %s\nIt is the scan that proves the "
        "removed sanitiser is nowhere, so an unreadable file is a hole in the "
        "proof, not a file to skip." % "\n  ".join(sorted(unparseable)))
    assert not live, (
        "the removed sanitiser is still defined, imported, called or named at: "
        "%s" % ", ".join(sorted(live)))


@pytest.mark.parametrize("rel", sorted(WIDGET_NODES))
def test_the_tooltip_no_longer_advertises_the_removed_power(rel):
    src = (REPO / rel).read_text(encoding="utf-8")
    assert "widget's value if it runs" not in src, (
        "%s still tells the operator the widget picks the binary" % rel)


def test_the_scopes_node_severs_before_BOTH_consumers():
    """`otr_scene_aware_scopes` has TWO doors, and the encoder one was missed
    by the first three drafts of the 2026-09-04 fix: `:491` probes with
    `resolve_ffprobe(ffmpeg=...)`, and the encode path hands the same value to
    `scope_draw.encode_silent_mp4` -> `find_ffmpeg` -> `resolve_ffmpeg`.

    Both doors are still there and both still take a binary name. What changed
    on 2026-09-13 is where that name can come from, and the parametrized walk
    above owns that half now. What stays HERE is the door count: this node is
    the one where severing before ONE consumer was not enough, so losing sight
    of either is the specific regression worth naming.
    """
    rel = "nodes/otr_scene_aware_scopes.py"
    fn = _execute_def(rel, NODE_CLASS_MAPPINGS[WIDGET_NODES[rel]])

    doors = {}
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") in ("resolve_ffprobe",
                                                    "encode_silent_mp4")):
            doors.setdefault(n.func.attr, n.lineno)
    assert set(doors) == {"resolve_ffprobe", "encode_silent_mp4"}, (
        "render_scopes should reach BOTH consumers and this walk found %s -- "
        "has the node been rewired? Severing before one of the two is the "
        "defect this test was written for." % (sorted(doors) or "neither"))

    bindings = [n.lineno for n in ast.walk(fn)
                if isinstance(n, ast.Name) and n.id == "ffmpeg"
                and isinstance(n.ctx, ast.Store)]
    assert bindings, "render_scopes no longer binds `ffmpeg` at all"
    assert min(bindings) < min(doors.values()), (
        "`ffmpeg` is bound at line %d but a consumer runs at line %d"
        % (min(bindings), min(doors.values())))


# --------------------------------------------------------------------------- #
# no fallback may reflect raw input back into argv[0]
# --------------------------------------------------------------------------- #
def test_no_resolver_wrapper_reflects_its_argument():
    """The bypass that made the first version of this fix a no-op: a wrapper
    that answered `resolve(x) or x` handed a REJECTED value straight back to
    argv[0]. Two sites did it -- one of them aliased, so a grep for the
    resolver's NAME missed it."""
    import re
    offenders = []
    pattern = re.compile(r"\)\s+or\s+\(?\s*(?:str\()?\s*(cand|ffmpeg|ffprobe)\b")
    for path in (REPO / "nodes").rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8",
                                                errors="replace").splitlines(), 1):
            if "resolve_ff" in line or "_resolve(" in line or "_ffmpeg_bin(" in line:
                if pattern.search(line):
                    offenders.append("%s:%d %s"
                                     % (path.relative_to(REPO), i, line.strip()))
    assert not offenders, "a fallback reflects its argument:\n" + "\n".join(offenders)


def _starve_the_resolver(monkeypatch):
    """Leave `resolve_ffmpeg` with nothing to find, at all SIX of its steps.

    The order it walks is: the preferred value, `OTR_FFMPEG`, `PATH`, the
    Windows and macOS install candidates, `ffmpeg-downloader`, and the binary
    imageio-ffmpeg bundles. Neutralising one of them -- which is what this
    file used to do -- leaves five live, and on a box with imageio-ffmpeg
    installed the last one answers.
    """
    from nodes._otr_shared import ffmpeg as ffm
    monkeypatch.setattr(shutil, "which", lambda name, *a, **kw: None)
    monkeypatch.delenv("OTR_FFMPEG", raising=False)
    monkeypatch.setattr(ffm, "_WINDOWS_INSTALL_CANDIDATES", ())
    monkeypatch.setattr(ffm, "_MACOS_INSTALL_CANDIDATES", ())
    monkeypatch.setattr(ffm, "_downloaded_ffmpeg", lambda: None)
    monkeypatch.setattr(ffm, "_imageio_ffmpeg", lambda: None)
    return ffm


@pytest.mark.parametrize("starved", [False, True],
                         ids=["as-this-box-is", "nothing-to-find"])
@pytest.mark.parametrize("value", HOSTILE_VALUES)
def test_the_blend_never_answers_with_the_value_it_was_given(value, starved,
                                                             monkeypatch):
    """THE SECURITY PROPERTY, stated so that it runs on every box.

    `_ffmpeg_bin` is `resolve_ffmpeg(x) or ""`, and the defect class it guards
    against is the one-character variant `resolve_ffmpeg(x) or x` -- a wrapper
    that hands a REJECTED value straight back to argv[0]. So the claim is "the
    answer is never the argument", not "this box has no ffmpeg".

    Those are not the same claim, and the difference is why this test exists.
    Its predecessor asserted the answer was `""` while neutralising one of six
    fallbacks, so on any box with imageio-ffmpeg installed it failed -- not
    because the pack echoed anything, but because the box HAD an ffmpeg and the
    assertion had mistaken that for a defect. A test that cannot run where the
    dependency is installed proves the property nowhere.

    BOTH ARMS ARE NEEDED, and a negative control is what proved it. On a box
    where resolution SUCCEEDS, `resolve(x) or x` never reaches its second
    operand, so the `as-this-box-is` arm cannot see that defect -- what it does
    see is an answer built from the argument (a sibling inside the caller's
    directory, a basename echoed back relative). The `nothing-to-find` arm is
    where the resolver genuinely refuses, which is the only condition under
    which the reflecting fallback fires at all.

    THE ASSERTION IS THE SAME IN BOTH, and deliberately so: unlike `== ""` it
    stays TRUE and meaningful if the starvation is ever incomplete. A seventh
    fallback appearing would make this test answer a real binary rather than
    fail a claim about the box.
    """
    from nodes import otr_post_upscale_procgen_blend as pu
    if starved:
        _starve_the_resolver(monkeypatch)
    got = pu._ffmpeg_bin(value)

    assert got != value, (
        "_ffmpeg_bin echoed its own argument %r back to the caller, which is "
        "the value argv[0] is built from" % value)
    if not got:
        return                       # "this box has no ffmpeg" -- a fact, not an echo
    assert os.path.isabs(got), (
        "_ffmpeg_bin(%r) answered the relative path %r; Windows CreateProcess "
        "searches the cwd, so a relative answer is a cwd hit waiting to "
        "happen" % (value, got))
    assert os.path.isfile(got), (
        "_ffmpeg_bin(%r) answered %r, which is not a file on this box -- a "
        "real resolution names something that exists, so this came from the "
        "argument rather than from the search" % (value, got))
    assert os.path.abspath(got) != os.path.abspath(value), (
        "_ffmpeg_bin(%r) answered %r, the same path by another spelling"
        % (value, got))
    if os.path.dirname(value):
        assert (os.path.normcase(os.path.dirname(os.path.abspath(got)))
                != os.path.normcase(os.path.dirname(os.path.abspath(value)))), (
            "_ffmpeg_bin(%r) answered %r -- a DIFFERENT name inside the "
            "directory the caller supplied, which is still the caller choosing "
            "the binary" % (value, got))


def test_the_blend_answers_empty_when_this_box_has_no_ffmpeg(monkeypatch):
    """The other half: when nothing resolves, the answer is `""`.

    `resolve_ffmpeg` has SIX steps -- the preferred value, `OTR_FFMPEG`, PATH,
    the Windows and macOS install candidates, `ffmpeg-downloader`, and the
    binary imageio-ffmpeg bundles. Neutralising one of them and asserting `""`
    is a claim about the box rather than about the code, so all of them are
    neutralised here and the neutralisation is then PROVEN before the empty
    answer is asserted -- otherwise a seventh fallback would quietly turn this
    test back into the one it replaced.
    """
    from nodes import otr_post_upscale_procgen_blend as pu
    ffm = _starve_the_resolver(monkeypatch)

    assert ffm.resolve_ffmpeg() is None, (
        "a fallback this test does not neutralise still answers %r, so the "
        "empty case below would be proving nothing about the code. "
        "resolve_ffmpeg has grown a step past preferred / OTR_FFMPEG / PATH / "
        "the install candidates / ffmpeg-downloader / imageio-ffmpeg -- "
        "neutralise that one here too." % (ffm.resolve_ffmpeg(),))

    for value in HOSTILE_VALUES + ("",):
        assert pu._ffmpeg_bin(value) == "", value


# --------------------------------------------------------------------------- #
# the resolvers answer with an ABSOLUTE path, or nothing
# --------------------------------------------------------------------------- #
def test_a_resolver_never_answers_with_a_relative_name(tmp_path, monkeypatch):
    """A bare answer is spawned relative, and Windows CreateProcess searches
    the cwd -- so `resolve_ffmpeg()` returning the string 'ffmpeg' while a file
    of that name sat beside the server WAS the hazard."""
    from nodes._otr_shared import ffmpeg as ffm
    from nodes._otr_shared import ffprobe as ffp

    real = tmp_path / "real" / "ffmpeg.exe"
    real.parent.mkdir()
    real.write_bytes(b"")
    monkeypatch.setattr(shutil, "which",
                        lambda name: str(real) if "ffmpeg" in name else None)
    monkeypatch.delenv("OTR_FFMPEG", raising=False)
    monkeypatch.setattr(ffm, "_WINDOWS_INSTALL_CANDIDATES", ())

    got = ffm.resolve_ffmpeg()
    assert got is None or os.path.isabs(got), got
    assert ffp._usable("ffmpeg") in (None, str(real))


def test_an_implicit_cwd_hit_is_refused(tmp_path, monkeypatch):
    """THE MECHANISM: CPython inserts the literal `os.curdir` on Windows unless
    `NoDefaultCurrentDirectoryInExePath` is set, so a cwd hit comes back
    RELATIVE while every real PATH directory yields an absolute answer.

    The env var MUST be deleted here: this developer box happens to set it, so
    without the delenv this test passes vacuously and the guard would ship
    unproven (Fable gate, 2026-09-04)."""
    from nodes._otr_shared import ffprobe as ffp
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: r".\ffmpeg.exe")
    got = ffp._which_no_cwd("ffmpeg")
    # It must not TAKE the cwd hit. It MAY still find a real one on PATH --
    # refusing outright would break a box whose only ffmpeg is on PATH the
    # moment a file of that name appeared beside the server.
    assert got != r".\ffmpeg.exe"
    assert got is None or os.path.isabs(got), got
    if got:
        assert os.path.dirname(os.path.abspath(got)) != os.path.abspath(os.getcwd())


def test_a_directory_bearing_relative_path_is_refused(tmp_path, monkeypatch):
    """`bin/ffmpeg` resolves against the process cwd just as a bare name does."""
    from nodes._otr_shared import ffprobe as ffp
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ffmpeg.exe").write_bytes(b"")
    assert ffp._usable(r"bin\ffmpeg.exe") is None
    assert ffp._usable("bin/ffmpeg.exe") is None


def test_an_absolute_path_is_still_honoured(tmp_path):
    """Trusted callers -- an operator pin, a resolved sibling, a Windows
    install dir -- all supply absolute paths, and must keep working."""
    from nodes._otr_shared import ffprobe as ffp
    real = tmp_path / "ffmpeg.exe"
    real.write_bytes(b"")
    assert ffp._usable(str(real)) == str(real)


# --------------------------------------------------------------------------- #
# the filtergraph basename
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("char", [",", ";", ":", "=", "[", "]", "'"])
def test_filtergraph_syntax_in_a_caption_filename_is_refused(char):
    from nodes import otr_caption_burn as cb
    with pytest.raises(ValueError):
        cb._ass_filter_arg("C:\\ep\\bad%sname.ass" % char)


def test_a_backslash_is_rejected_by_the_validator_itself():
    """A backslash cannot reach a BASENAME through a path -- it IS the
    separator, so `Path(...).name` never contains one. It is tested on the
    validator directly, and stays in the reject set because it is ffmpeg's own
    escape character and the validator is shared with the blend node's copy."""
    from nodes import otr_caption_burn as cb
    with pytest.raises(ValueError):
        cb._reject_filtergraph_syntax("bad\\name.ass")


@pytest.mark.parametrize("name", ["episode.ass", "my episode 01.ass",
                                  "ep-01_final.ass", "a.b.ass"])
def test_an_ordinary_caption_filename_still_works(name):
    """Spaces, dots and dashes are legal in filenames and harmless in a graph.
    Every real episode stem is slugified to [a-z0-9_] anyway, so this guard can
    never fire on a normal render."""
    from nodes import otr_caption_burn as cb
    got, _cwd = cb._ass_filter_arg("C:\\ep\\" + name)
    assert got == name


def test_both_copies_of_the_filter_arg_builder_are_guarded():
    """There are TWO `_ass_filter_arg`s and FOUR `ass={name}` interpolations.
    Guarding only the caption node leaves three sites open."""
    for rel in ("nodes/otr_caption_burn.py",
                "nodes/otr_post_upscale_procgen_blend.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "def _ass_filter_arg(" in src, rel
        assert "_reject_filtergraph_syntax(" in src, (
            "%s builds an ass= argument without validating it" % rel)


# --------------------------------------------------------------------------- #
# the no-auth route
# --------------------------------------------------------------------------- #
def test_the_ledger_route_serves_no_wildcard_cors():
    """`GET /otr/latest_ledger` is registered on EVERY install with no
    authentication and answers with the whole ledger. A wildcard is what makes
    that readable cross-origin, so any site visited while ComfyUI runs could
    take it."""
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert '"Access-Control-Allow-Origin": "*"' not in src


def test_the_ledger_route_discloses_no_absolute_path():
    """It used to answer with `fullpath` -- the operator's own directory tree,
    Windows username included -- and with `str(exc)` on failure, which names
    the file it could not open. Both went to an unauthenticated caller
    (2026-09-05)."""
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    i = src.index('@_otr_PromptServer.instance.routes.get("/otr/latest_ledger")')
    handler = src[i:src.index("routes.options", i)]
    assert '"fullpath"' not in handler
    assert '"reason": str(exc)' not in handler
    assert '"filename"' in handler, "the basename still identifies the episode"


def _scrub_from_shipped_source():
    """Load the shipped `_otr_scrub_paths` out of `__init__.py`.

    Exec'd from source rather than imported, the same way
    `test_http_render_route_gate` used to load the route block: importing the
    package pulls in ComfyUI's `server` module, which does not exist here.
    """
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    start = src.index("    import re as _otr_re")
    end = src.index('    @_otr_PromptServer.instance.routes.get("/otr/latest_ledger")')
    body = src[start:end].split("\n")
    block = "\n".join(l[4:] if l.startswith("    ") else l for l in body)
    ns: dict = {}
    exec(block, ns)                                   # noqa: S102 -- shipped code
    return ns["_otr_scrub_paths"]


def test_the_ledger_route_scrubs_paths_from_the_whole_document():
    """Removing the top-level `fullpath` was NOT enough, and the first pass at
    this stopped there. The route returns the ENTIRE ledger, and one live
    episode ledger carried 75 absolute paths inside it: `meta.paths` has ten
    keys, and every still, cue and final asset carries its own. The scrub runs
    on the serialized response and the on-disk record is untouched."""
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert '"ledger": _otr_scrub_paths(ledger)' in src, (
        "the response must serialize a scrubbed projection, not the raw ledger")
    scrub = _scrub_from_shipped_source()

    b = chr(92)          # a backslash is BUILT here, never escaped in source
    cases = [
        ("C:" + b + "Users" + b + "jeffr" + b + "x.mp4", "x.mp4"),
        ("C:/Users/jeffr/y.png", "y.png"),
        ("/home/j/a/b.wav", "b.wav"),
        (b + b + "host" + b + "share" + b + "f.png", "f.png"),
        ("relative/path.txt", "relative/path.txt"),
        ("just a sentence about C: drives", "just a sentence about C: drives"),
        ("", ""),
    ]
    for probe, want in cases:
        assert scrub(probe) == want, (
            "%r -> %r, want %r" % (probe, scrub(probe), want))


def test_the_scrub_never_mutates_the_ledger_it_was_given():
    """The on-disk record is the production artifact; this is a projection for
    one HTTP reader. Non-path values must survive untouched, or the route would
    be lying about the episode rather than merely hiding its location."""
    scrub = _scrub_from_shipped_source()
    b = chr(92)
    doc = {
        "meta": {"paths": {"audio_dir": "C:" + b + "out" + b + "audio"}},
        "lines": [{"id": "l1", "wav": "C:" + b + "out" + b + "l1.wav",
                   "n": 3, "ok": True, "none": None}],
    }
    frozen = json.dumps(doc, sort_keys=True)
    out = scrub(doc)
    assert out["meta"]["paths"]["audio_dir"] == "audio"
    assert out["lines"][0]["wav"] == "l1.wav"
    assert out["lines"][0]["n"] == 3
    assert out["lines"][0]["ok"] is True
    assert out["lines"][0]["none"] is None
    assert json.dumps(doc, sort_keys=True) == frozen, (
        "the input document was mutated; the response must be a copy")


def test_the_scrub_is_depth_bounded():
    """It walks a document read from disk inside an HTTP handler, so unbounded
    recursion would be a denial of service the route hands out for free."""
    scrub = _scrub_from_shipped_source()
    deep: dict = {}
    cursor = deep
    for _ in range(60):
        cursor["n"] = {}
        cursor = cursor["n"]
    assert isinstance(scrub(deep), dict)
