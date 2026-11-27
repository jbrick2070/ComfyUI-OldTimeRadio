"""Nothing a node declares names a folder on this machine.

ComfyUI serves every node's INPUT_TYPES (widget defaults, dropdown choices,
tooltips), DESCRIPTION, CATEGORY and output names at GET /object_info with no
login. A value computed from the filesystem -- a default built from
``Path(__file__)``, a models folder, a file listing with its folders -- hands
the install location, and on most machines the account name, to anyone who can
reach the server.

THE DEFECT THIS PINS, and it shipped through 2.3.64: OTR_WorkflowValidator
declared ``"default": str(_DEFAULT_WORKFLOW_PATH)``, the absolute path of the
pack's own workflows/otr_canonical.json, while every shipped workflow saved the
widget as "" and the runtime already read "" as the canonical graph.

The nodes are imported the way ComfyUI imports them, and every declared value
is checked, so a new node or widget is covered without touching this file.
CPU only. UTF-8 no BOM.
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: A drive-letter or UNC path, or a rooted POSIX path under a real top level.
#: HTTP routes ("/api/prompt"), ratios ("16/9") and relative paths are not
#: matched; a URL's "//" is excluded by the look-behind.
_ABSOLUTE = re.compile(
    r"(?<![\w])[A-Za-z]:[\\/]"
    r"|\\\\[^\\/\s]+[\\/]"
    r"|(?<![\w.:/\\~-])/(?:home|Users|Volumes|root|tmp|var|opt|srv|mnt|media|private|"
    r"usr|etc|workspace|content)/"
)

_DECLARED = ("DESCRIPTION", "CATEGORY", "RETURN_NAMES", "OUTPUT_TOOLTIPS",
             "RETURN_TYPES", "FUNCTION", "DISPLAY_NAME", "SEARCH_ALIASES")


def _mappings() -> dict:
    sys.path.insert(0, str(_REPO_ROOT.parent))
    try:
        pkg = importlib.import_module(_REPO_ROOT.name)
    finally:
        sys.path.pop(0)
    return dict(getattr(pkg, "NODE_CLASS_MAPPINGS", {}) or {})


NODE_CLASS_MAPPINGS = _mappings()


def _served_text(cls) -> str:
    """Everything /object_info would serve for one node, as one string."""
    served = {"input": cls.INPUT_TYPES()}
    for name in _DECLARED:
        value = getattr(cls, name, None)
        if value is not None and not callable(value):
            served[name] = value
    return json.dumps(served, default=str, ensure_ascii=False)


def _machine_strings() -> list:
    """This machine's own locations in every spelling a JSON dump may carry."""
    out = []
    for folder in (_REPO_ROOT, Path.home()):
        plain = str(folder)
        for spelling in (plain, plain.replace("\\", "/"), plain.replace("\\", "\\\\")):
            if len(spelling) > 3:
                out.append(spelling.lower())
    return out


def test_the_pack_registers_nodes():
    assert len(NODE_CLASS_MAPPINGS) >= 20, sorted(NODE_CLASS_MAPPINGS)


@pytest.mark.parametrize("node_id", sorted(NODE_CLASS_MAPPINGS))
def test_no_declared_value_is_an_absolute_path(node_id):
    text = _served_text(NODE_CLASS_MAPPINGS[node_id])
    found = _ABSOLUTE.search(text)
    assert found is None, (
        "%s serves an absolute path at /object_info: ...%s..."
        % (node_id, text[max(0, found.start() - 60):found.start() + 100]))
    lowered = text.lower()
    for location in _machine_strings():
        assert location not in lowered, (
            "%s serves this machine's folder %r at /object_info" % (node_id, location))


def test_the_detector_sees_the_shapes_it_claims():
    """The check above cannot pass by matching nothing."""
    for sample in (r'"default": "C:\\Users\\someone\\pack\\workflows\\x.json"',
                   '"default": "C:/Users/someone/x.json"',
                   '"default": "/home/someone/ComfyUI/x.json"',
                   '"default": "/Users/someone/x.json"',
                   r'"default": "\\\\nas\\share\\x.json"'):
        assert _ABSOLUTE.search(sample), sample
    for sample in ('"tooltip": "POST /api/prompt, ratio 16/9, otr/obs/ep.mp4"',
                   '"tooltip": "see https://huggingface.co/Qwen/Qwen3.5-4B"',
                   '"default": "workflows/otr_canonical.json"'):
        assert not _ABSOLUTE.search(sample), sample
