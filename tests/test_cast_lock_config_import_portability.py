"""``config`` imports inside ``nodes/`` must be relative-first, never bare.

PBUG-20260825-02. ``CastLock._assign_bark_announcer`` imported the cast-pools
module with a bare ``from config import cast_pools`` -- no relative attempt
first, no fallback. That works only when the repo root happens to be on
``sys.path`` as a bare entry, which is true under pytest and under
``otr_canonical_api_run.py`` (both add the repo root themselves), but NOT under
a real ComfyUI installation, where the pack is loaded as a submodule of
``custom_nodes`` and its own root is never added to ``sys.path`` directly.

So every proof leg run against the API script tonight passed by ACCIDENT, and
the bug reproduced on the very first real ComfyUI Desktop render on the 4060 --
18 minutes into a run, at the last step (announcer voice casting), with
``ModuleNotFoundError: No module named 'config'``.

This is not a one-off: the same two-tier shape (relative attempt, absolute
fallback) already existed at 10+ other ``cast_pools`` call sites across
``nodes/_otr_casting.py``, ``nodes/_otr_voice_bank.py``,
``nodes/_otr_scifi_news_pro.py``, and in another function of ``cast_lock.py``
itself (since removed) -- proof that the pattern was known and simply not
applied here. This test pins the shape so it cannot regress at any of the
guarded call sites, and would have failed on the bug as filed.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Every production file with a cast_pools import, and every function inside it
# that imports cast_pools. A new call site added here without the guard is
# exactly the PBUG-20260825-02 shape.
_FILES = (
    "nodes/cast_lock.py",
    "nodes/_otr_casting.py",
    "nodes/_otr_voice_bank.py",
    "nodes/_otr_scifi_news_pro.py",
    "nodes/_otr_voice_node_common.py",
)


def _functions_importing_cast_pools(tree: ast.AST):
    """Map enclosing-function-name -> list of ImportFrom nodes that import
    `cast_pools`, for every such import in the module (module-level imports
    are keyed under "<module level>")."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    def enclosing_function(node):
        cur = node
        while id(cur) in parents:
            cur = parents[id(cur)]
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
        return None

    hits: dict[str, list[ast.ImportFrom]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        # BOTH SPELLINGS. `from config import cast_pools` puts cast_pools in
        # the NAMES; `from config.cast_pools import f` puts it in the MODULE.
        # Matching only the first is how four sites of PBUG-20260825-02 were
        # added on 2026-09-24 without this test noticing.
        module = (node.module or "")
        root = module.lstrip(".")
        names = {a.name for a in node.names}
        is_cast_pools = (
            root in ("config", "cast_pools") and "cast_pools" in names
        ) or root in ("config.cast_pools",)
        if is_cast_pools:
            fn = enclosing_function(node)
            key = fn.name if fn is not None else "<module level>"
            hits.setdefault(key, []).append(node)
    return hits


def test_every_cast_pools_import_site_is_guarded():
    """No bare `from config import cast_pools` may exist outside a `try` whose
    `except ImportError` (or `except (ImportError, ValueError)`) falls back to
    the other import form. Mirrors the shape at cast_lock.py:49-55."""
    violations = []

    for rel in _FILES:
        path = REPO_ROOT / rel
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))

        # Map each `ast.Try` node's line RANGE so we can ask "is this import
        # inside some try block whose handler falls back to the sibling
        # import form".
        try_blocks = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]

        hits = _functions_importing_cast_pools(tree)
        for fn_name, imports in hits.items():
            for imp in imports:
                guarded = False
                for tb in try_blocks:
                    body_lines = {
                        s.lineno for s in ast.walk(tb) if hasattr(s, "lineno")
                    }
                    if imp.lineno not in body_lines:
                        continue
                    # It sits inside a try. The try must have at least one
                    # ImportError-catching handler (directly, or nested via a
                    # second try inside the handler -- the fallback pattern).
                    for handler in tb.handlers:
                        exc = handler.type
                        names = set()
                        if isinstance(exc, ast.Name):
                            names.add(exc.id)
                        elif isinstance(exc, ast.Tuple):
                            names.update(
                                e.id for e in exc.elts if isinstance(e, ast.Name)
                            )
                        if "ImportError" in names:
                            guarded = True
                    if guarded:
                        break
                if not guarded:
                    violations.append(
                        f"{rel}:{imp.lineno} in {fn_name}() -- bare "
                        f"'from {imp.module} import cast_pools', not inside "
                        "a try/except ImportError fallback"
                    )

    assert not violations, (
        "unguarded cast_pools import(s) found -- these work by accident under "
        "pytest/otr_canonical_api_run.py (repo root on sys.path) and crash "
        "under a real ComfyUI install (PBUG-20260825-02):\n"
        + "\n".join(violations)
    )


def test_assign_bark_announcer_specifically_has_the_fallback():
    """Direct regression pin for the exact function that broke on the 4060."""
    src = (REPO_ROOT / "nodes/cast_lock.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_assign_bark_announcer"
    )
    fn_src = ast.get_source_segment(src, target)
    assert fn_src is not None

    assert "from ..config import cast_pools" in fn_src, (
        "the relative-first import is gone from _assign_bark_announcer"
    )
    assert "from config import cast_pools" in fn_src, (
        "the absolute fallback import is gone from _assign_bark_announcer"
    )
    # The bare, unguarded form (no leading whitespace before `from config`,
    # i.e. not nested inside the except block) must not be present.
    bare = [
        line for line in fn_src.splitlines()
        if line.strip() == "from config import cast_pools as _POOLS  # type: ignore"
        and not line.startswith((" " * 12, "\t\t\t"))
    ]
    assert not bare, f"found an unguarded absolute import: {bare}"


def test_the_FIRST_import_attempt_at_every_site_is_the_relative_one():
    """A fallback chain proves nothing if its first rung is unreachable.

    THIS IS THE ASSERTION THE OTHER TEST SHOULD HAVE BEEN. On 2026-09-24 four
    recurring-character sites were added, each correctly wrapped in
    try/except ImportError with a fallback -- and each tried `config.cast_pools`
    first and bare `cast_pools` second. Under a real ComfyUI package load
    NEITHER resolves, so every one of them silently took its `return ""` /
    `return 0` path and the whole cutover was dormant in production while this
    suite stayed green.

    The reachable form under a package load is the RELATIVE one, because the
    loader imports the pack by file location and adds nothing to sys.path.
    So: whatever else a site tries, it tries `..config` first.
    """
    offenders = []
    for rel in _FILES:
        path = REPO_ROOT / rel
        if not path.exists():           # a file removed by a later change
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for fn_name, imports in _functions_importing_cast_pools(tree).items():
            first = min(imports, key=lambda n: n.lineno)
            if first.level == 0:
                offenders.append(
                    "%s::%s line %d: first attempt is absolute (%r)"
                    % (rel, fn_name, first.lineno, first.module))
    assert not offenders, (
        "a cast_pools import site tries an absolute form FIRST:\n  "
        + "\n  ".join(offenders))


def test_the_recurring_character_table_resolves_under_a_real_package_load():
    """Load the pack as ComfyUI does and prove the cutover actually runs.

    ComfyUI imports a custom-node pack BY FILE LOCATION under its folder name
    and adds nothing to sys.path. Under pytest the repo root IS on sys.path,
    which is the accident that made every other test here pass while the
    feature did not exist at runtime. This removes that accident.

    RUN IN A SUBPROCESS, deliberately. Doing the sys.path surgery in-process
    would leak into whatever test ran next and make this file's result depend
    on collection order -- trading one invisible defect for another.
    """
    import json
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent(
        """
        import importlib.util, json, sys
        from pathlib import Path
        REPO = Path(sys.argv[1])
        sys.path = [p for p in sys.path
                    if p and Path(p).resolve() != REPO.resolve()]
        out = {}
        try:
            import config                       # noqa: F401
            out["repo_root_still_on_path"] = True
        except ImportError:
            out["repo_root_still_on_path"] = False
        name = "OTR_pkgload_probe"
        spec = importlib.util.spec_from_file_location(
            name, REPO / "__init__.py",
            submodule_search_locations=[str(REPO)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        cl = importlib.import_module(name + ".nodes.cast_lock")
        out["key"] = cl._recurring_character_key(
            {"char_id": "c02", "name": "LEMMY", "gender": "male"})
        bcv = importlib.import_module(name + ".nodes.batch_character_voices")
        out["is_changed"] = bcv.BatchCharacterVoices.IS_CHANGED(
            script_json=json.dumps({
                "meta": {"episode_seed": 42},
                "cast": [{"char_id": "c02", "name": "LEMMY", "gender": "male",
                          "voice_ref_id": "bm_george",
                          "voice_engine": "kokoro"}],
                "lines": []}),
            engine="kokoro")
        print("OTRPROBE" + json.dumps(out))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, str(REPO_ROOT)],
        capture_output=True, text=True, timeout=300)
    line = next((l for l in proc.stdout.splitlines()
                 if l.startswith("OTRPROBE")), None)
    assert line, (
        "the package-load probe produced no verdict.\nstdout:\n%s\nstderr:\n%s"
        % (proc.stdout[-2000:], proc.stderr[-2000:]))
    out = json.loads(line[len("OTRPROBE"):])

    assert out["repo_root_still_on_path"] is False, (
        "the probe did not actually simulate a package load -- the repo root "
        "is still importable, so this test would pass on the broken code")
    assert out["key"] == "LEMMY", (
        "the recurring-character table is DORMANT under a package load: "
        "_recurring_character_key returned %r, so the recurring voice never "
        "applies on a real install and the character is cast on an ordinary "
        "drawn voice" % (out["key"],))
    assert out["is_changed"] != "static", (
        "IS_CHANGED returned 'static' for a ledger holding the recurring row, "
        "so the reference bytes behind that voice can change without the graph "
        "noticing")
