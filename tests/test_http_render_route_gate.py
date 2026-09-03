"""``OTR_ENABLE_HTTP_RENDER_ROUTES`` gate on the two POST harness routes.

Registry manual review draft, option (a)
(docs/2026-09-02-registry-manual-review-request.md): ``__init__.py`` used to
register ``POST /otr/video_render_single`` and ``POST /otr/video_render_soak``
unconditionally. Both read caller-supplied file paths out of an
unauthenticated JSON body and start a background render thread -- the shape
the registry banned another pack for under ``policy-v0.2:
UNAUTHENTICATED_SIDE_EFFECT``. Nothing that ships calls them; they are a
hand-built GPU-gate harness for the operator to poll during dev.

This execs the EXACT bytes of that block out of the real ``__init__.py`` --
not a re-typed copy -- against fake ``server``/``aiohttp`` modules that just
record what got registered. A re-typed copy would keep passing after someone
edited the real file and broke the gate; this cannot drift from what ships.
"""
from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path

import pytest

_INIT_PATH = Path(__file__).resolve().parents[1] / "__init__.py"


def _route_gate_block() -> str:
    src = _INIT_PATH.read_text(encoding="utf-8")
    start = src.index("# HTTP routes: POST /otr/video_render_single")
    start = src.rindex("# ===", 0, start)
    end = src.rindex("# ===", 0, src.index("# OH-3 janitor"))
    block = src[start:end]
    # NOT just "the route names appear somewhere" -- that string also matches
    # the header COMMENT, so a boundary drift that shrank `block` down to just
    # the comment would still pass a substring check and then silently prove
    # nothing (the exec'd block would register zero routes either way, and
    # test_default_and_any_non_one_value_registers_nothing would pass for the
    # wrong reason). Require the actual decorator syntax, twice.
    assert block.count('@_otr_PS2.instance.routes.post(') == 2, (
        "block slicing drifted -- expected exactly the two POST route "
        "decorators, found a different shape:\n%s" % block)
    ast.parse(block)  # the isolated slice is itself valid Python
    return block


class _FakeRoutes:
    def __init__(self):
        self.registered: list[tuple[str, str]] = []

    def post(self, path):
        def deco(fn):
            self.registered.append(("POST", path))
            return fn
        return deco

    def get(self, path):
        def deco(fn):
            self.registered.append(("GET", path))
            return fn
        return deco

    def options(self, path):
        def deco(fn):
            self.registered.append(("OPTIONS", path))
            return fn
        return deco


def _exec_route_block(monkeypatch, env_value):
    fake_instance = types.SimpleNamespace(routes=_FakeRoutes())
    fake_server_mod = types.ModuleType("server")
    fake_server_mod.PromptServer = types.SimpleNamespace(instance=fake_instance)
    fake_web_mod = types.ModuleType("aiohttp.web")
    fake_web_mod.json_response = lambda *a, **k: None
    fake_web_mod.Response = lambda *a, **k: None
    fake_aiohttp_mod = types.ModuleType("aiohttp")
    fake_aiohttp_mod.web = fake_web_mod

    monkeypatch.setitem(sys.modules, "server", fake_server_mod)
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp_mod)
    monkeypatch.setitem(sys.modules, "aiohttp.web", fake_web_mod)
    if env_value is None:
        monkeypatch.delenv("OTR_ENABLE_HTTP_RENDER_ROUTES", raising=False)
    else:
        monkeypatch.setenv("OTR_ENABLE_HTTP_RENDER_ROUTES", env_value)

    exec(compile(_route_gate_block(), "<route-gate-block>", "exec"), {})
    return fake_instance.routes.registered


@pytest.mark.parametrize("env_value", [None, "0", "false", "no"])
def test_default_and_any_non_one_value_registers_nothing(monkeypatch, env_value):
    assert _exec_route_block(monkeypatch, env_value) == []


def test_1_registers_exactly_the_two_POST_routes_and_nothing_else(monkeypatch):
    regs = _exec_route_block(monkeypatch, "1")
    assert set(regs) == {
        ("POST", "/otr/video_render_single"),
        ("POST", "/otr/video_render_soak"),
    }


def test_the_flag_check_matches_the_house_convention():
    """Same shape as OTR_ENABLE_COMFY_CREDITS: `os.environ.get(name, "0") == "1"`.

    Pinned as text so a future edit that swaps in "on"/"off" or a truthy-string
    parser (a real behaviour change, not a typo) is a deliberate, reviewed
    decision rather than something that slips through unnoticed.
    """
    block = _route_gate_block()
    assert 'environ.get("OTR_ENABLE_HTTP_RENDER_ROUTES", "0") == "1"' in block
