"""The TYPE name of a pack exception carries no folder path.

FOUND LIVE, 2026-11-27, on a CPU ComfyUI running this pack: a prompt that
named a file outside the allowed folders came back from /history with a clean
message and this beside it:

    "exception_type": "<the full install folder>/custom_nodes/
                       ComfyUI-OldTimeRadio.nodes._otr_paths.OtrPathContractError"

ComfyUI imports a custom node pack under a module name that IS the pack
folder's absolute path, and reports a failed node's exception type as
``__module__ + "." + __qualname__`` -- to the prompt's history and the
websocket, with no login. So every exception class the pack defines carried
the install folder, and on most machines the account name, in its own name.
The message redaction never saw it, and no static review of the published zip
caught it. It shipped through 2.3.64.

Backslashes are built with ``chr(92)``. CPU only. UTF-8 no BOM.
"""
from __future__ import annotations

import os
import sys
import traceback
import types

import pytest

import nodes._otr_error_text as mod
from nodes._otr_error_text import clean_module_name, guard_node_class, redact_exception

SEP = chr(92)
WINDOWS_NAME = SEP.join(("C:", "Users", "someone", "ComfyUI", "custom_nodes",
                         "ComfyUI-OldTimeRadio.nodes._otr_paths"))
POSIX_NAME = "/home/someone/ComfyUI/custom_nodes/comfyui-old-time-radio.nodes._otr_paths"


def _full_type_name(klass) -> str:
    """ComfyUI core's own rule (execution.full_type_name), repeated here."""
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__
    return module + "." + klass.__qualname__


@pytest.mark.parametrize("name, expected", [
    (WINDOWS_NAME, "ComfyUI-OldTimeRadio.nodes._otr_paths"),
    (POSIX_NAME, "comfyui-old-time-radio.nodes._otr_paths"),
    ("nodes._otr_paths", "nodes._otr_paths"),
    ("builtins", "builtins"),
    ("custom_nodes.ComfyUI-OldTimeRadio", "custom_nodes.ComfyUI-OldTimeRadio"),
    (None, None),
])
def test_a_module_name_loses_its_folders_and_nothing_else(name, expected):
    assert clean_module_name(name) == expected


@pytest.mark.parametrize("module_name", [WINDOWS_NAME, POSIX_NAME])
def test_a_pack_exception_leaving_a_node_reports_a_clean_type_name(module_name):
    PathContractError = type("PathContractError", (ValueError,), {"__module__": module_name})

    class _Node:
        FUNCTION = "run"

        def run(self):
            raise PathContractError("refused")

    guard_node_class(_Node)
    with pytest.raises(PathContractError) as caught:
        _Node().run()
    reported = _full_type_name(type(caught.value))
    assert "someone" not in reported
    assert reported.endswith("nodes._otr_paths.PathContractError")
    last_line = traceback.format_exception_only(type(caught.value), caught.value)[-1]
    assert "someone" not in last_line
    assert isinstance(caught.value, ValueError)           # same class, same bases


def test_the_cause_of_an_exception_gets_a_clean_type_name_too():
    Inner = type("Inner", (OSError,), {"__module__": WINDOWS_NAME})
    Outer = type("Outer", (RuntimeError,), {"__module__": WINDOWS_NAME})
    try:
        try:
            raise Inner("inner")
        except Inner as err:
            raise Outer("outer") from err
    except Outer as top:
        redact_exception(top)
        text = "".join(traceback.format_exception(type(top), top, top.__traceback__))
    assert "someone" not in _full_type_name(Outer) + _full_type_name(Inner)
    assert "Users" not in text.replace(__file__, "")


def test_builtin_and_ordinary_classes_are_left_exactly_as_they_are():
    redact_exception(ValueError("x"))
    assert ValueError.__module__ == "builtins"
    Plain = type("Plain", (Exception,), {"__module__": "nodes._otr_paths"})
    redact_exception(Plain("x"))
    assert Plain.__module__ == "nodes._otr_paths"


def test_every_exception_class_the_pack_has_loaded_is_cleaned_at_load(monkeypatch):
    """ComfyUI also reports the type name of an exception raised while it
    VALIDATES a prompt, which never passes through a node's own function. So
    the pack's exception classes are cleaned once, when the nodes are guarded."""
    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
    inside = types.ModuleType(WINDOWS_NAME)
    inside.__file__ = os.path.join(pack_root, "nodes", "_otr_example.py")
    inside.PackError = type("PackError", (Exception,), {"__module__": WINDOWS_NAME})
    inside.NotAnError = type("NotAnError", (object,), {"__module__": WINDOWS_NAME})
    outside = types.ModuleType("another_pack")
    outside.__file__ = os.path.join(os.path.dirname(pack_root), "another_pack", "x.py")
    outside.TheirError = type("TheirError", (Exception,), {"__module__": POSIX_NAME})
    monkeypatch.setitem(sys.modules, WINDOWS_NAME, inside)
    monkeypatch.setitem(sys.modules, "another_pack", outside)

    assert mod.guard_node_classes([]) == 0
    assert inside.PackError.__module__ == "ComfyUI-OldTimeRadio.nodes._otr_paths"
    assert inside.NotAnError.__module__ == WINDOWS_NAME        # only exceptions
    assert outside.TheirError.__module__ == POSIX_NAME         # only this pack
    assert mod.clean_pack_exception_modules() == 0             # nothing left to do
