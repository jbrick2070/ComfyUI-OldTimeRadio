"""The shipped pack registers no HTTP route on ComfyUI's server.

Registry version 2.3.51 was Flagged with a signed verdict,
``policy-v0.7: information-disclosure, too-many-findings`` (read 2026-11-10).
The one thing in that zip that served data to a caller was
``GET /otr/latest_ledger``: registered on every install, no login, and it
answered with the whole episode ledger -- story, cast, prompts, model ids and
the run's settings -- to anything that could reach the ComfyUI port. Paths
were scrubbed to basenames, but the document itself still went out. Nothing
in the shipped pack called it; its only reader was an internal live-tail page.

The registry's own record shows what clears this class: deno-custom-nodes
carried the same verdict on four versions in a row, and 0.7.127, which only
deleted its unauthenticated resource-monitor route, passed the automated
checks and went Active with no human review.

So the route is gone, and this test keeps the shipped set route-free. The
real render path is the canonical workflow through ComfyUI's own ``/prompt``;
a development reader for a ledger belongs in ``scripts/`` (outside the zip)
or reads the file on disk.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _shipped_python_files():
    spec = importlib.util.spec_from_file_location(
        "otr_registry_scan_oracle", REPO / "scripts" / "otr_registry_scan_oracle.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    return [p for p in oracle._shipped_files(False) if p.endswith(".py")]


#: aiohttp's router and application methods that hang a handler on a path.
_ROUTER_METHODS = frozenset((
    "add_routes", "add_route", "add_get", "add_post", "add_put", "add_patch",
    "add_delete", "add_head", "add_options", "add_view", "add_static",
    "add_resource",
))


def _route_sites(tree: ast.AST) -> list:
    """Every way a pack hangs a handler on aiohttp: any ``.routes`` reach,
    whatever the receiver (``PromptServer.instance.routes``, or
    ``ps = PromptServer.instance`` then ``ps.routes``), ``getattr(x,
    "routes")``, a ``RouteTableDef``, and a router or app ``add_*`` call
    (Sonnet review of adf158352). Nothing in the shipped set names
    ``.routes`` for any other reason, so the wide net costs no false hit."""
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("routes", "RouteTableDef"):
            sites.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id == "RouteTableDef":
            sites.append(node.lineno)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in _ROUTER_METHODS:
                sites.append(node.lineno)
            elif (isinstance(fn, ast.Name) and fn.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == "routes"):
                sites.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if any(a.name == "RouteTableDef" for a in node.names):
                sites.append(node.lineno)
    return sorted(set(sites))


def test_the_shipped_set_is_read_by_this_test():
    """Wiring: the finder must actually reach the files that ship, the
    package entry point first among them."""
    shipped = _shipped_python_files()
    assert "__init__.py" in shipped
    assert "prestartup_script.py" in shipped
    assert any(p.startswith("nodes/") for p in shipped)


def test_no_shipped_python_file_registers_an_http_route():
    offenders = []
    for rel in _shipped_python_files():
        src = (REPO / rel).read_text(encoding="utf-8")
        for line in _route_sites(ast.parse(src, filename=rel)):
            offenders.append("%s:%d" % (rel, line))
    assert offenders == [], (
        "a shipped file registers an HTTP route on ComfyUI's server. Every "
        "such route is reachable without a login, and an unauthenticated "
        "route that serves data is what Flagged 2.3.51 "
        "(policy-v0.7: information-disclosure). Sites:\n  "
        + "\n  ".join(offenders))


def test_the_finder_sees_a_decorated_route():
    """The finder is proven on the exact spelling the deleted route used."""
    probe = (
        "from server import PromptServer\n"
        "@PromptServer.instance.routes.get('/x')\n"
        "async def h(request):\n"
        "    return None\n"
        "app.router.add_routes([])\n"
    )
    assert _route_sites(ast.parse(probe)) == [2, 5]


def test_the_finder_sees_every_other_registration_spelling():
    """Each line below registers a route a different way; each must be seen,
    and an unrelated ``add_*`` call must not be."""
    probe = (
        "ps = PromptServer.instance\n"                 # 1: no hit by itself
        "@ps.routes.post('/a')\n"                      # 2
        "async def a(r): pass\n"                       # 3
        "r = getattr(ps, 'routes')\n"                  # 4
        "from aiohttp.web import RouteTableDef\n"      # 5
        "t = web.RouteTableDef()\n"                    # 6
        "app.router.add_put('/b', a)\n"                # 7
        "app.router.add_view('/c', a)\n"               # 8
        "parser.add_argument('--x')\n"                 # 9: not a route
    )
    assert _route_sites(ast.parse(probe)) == [2, 4, 5, 6, 7, 8]
