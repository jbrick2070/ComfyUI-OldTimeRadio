"""The pack does not go looking for a Hugging Face token.

Through 2.3.64 the pack's ``__init__`` read ``HF_TOKEN`` out of the Windows
user registry at every ComfyUI start and exported it into the whole process
(``HF_TOKEN`` and ``HUGGING_FACE_HUB_TOKEN``), where every other pack and
every child process could read it -- for gated models the pack stopped
shipping in 2.3.60. Three reviewers of the published zip named it.

Now a token is read only from where the user put it for this process
(``HF_TOKEN`` in the environment, or the Hub's own login file), only on the
execution path, and it is passed to the one download that needs it.

The token used here is a made-up string. CPU only. UTF-8 no BOM.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

from nodes import _otr_hf_auth as auth

REPO = Path(__file__).resolve().parents[1]
_FAKE = "hf_" + "notARealToken" * 2


def _registry_that_must_not_be_opened(monkeypatch):
    opened = []
    fake = types.ModuleType("winreg")
    fake.HKEY_CURRENT_USER = object()
    fake.KEY_READ = 0

    def _open(*args, **kwargs):
        opened.append(args)
        raise AssertionError("the registry was opened for a token")
    fake.OpenKey = _open
    fake.QueryValueEx = _open
    monkeypatch.setitem(sys.modules, "winreg", fake)
    return opened


def test_the_resolver_reads_the_environment_and_nothing_else(monkeypatch):
    opened = _registry_that_must_not_be_opened(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", _FAKE)
    assert auth.resolve_hf_token() == _FAKE
    monkeypatch.delenv("HF_TOKEN")
    assert auth.resolve_hf_token() is None
    assert opened == []


def test_the_runtime_resolver_never_opens_the_registry(monkeypatch, tmp_path):
    opened = _registry_that_must_not_be_opened(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(get_token=lambda: None))
    assert auth.resolve_hf_token_runtime() is None
    assert opened == []


def test_finding_a_token_changes_nothing_in_the_process_environment(monkeypatch, tmp_path):
    """Resolving is a read. Nothing is exported for other packs to find."""
    import os

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(get_token=lambda: _FAKE))
    before = dict(os.environ)
    assert auth.resolve_hf_token_runtime() == _FAKE
    assert dict(os.environ) == before


def test_the_pack_entry_points_do_not_touch_the_token():
    """No token is resolved at import: not in __init__, not at prestartup."""
    for name in ("__init__.py", "prestartup_script.py"):
        tree = ast.parse((REPO / name).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        imported = {alias.name for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        for banned in ("ensure_hf_token", "resolve_hf_token", "resolve_hf_token_runtime"):
            assert banned not in names | imported, "%s resolves a token at load" % name
    assert not (REPO / "nodes" / "_otr_shared" / "hf_token.py").exists()


def test_no_shipped_module_reads_a_token_from_the_registry():
    """`winreg` stays in use for two non-credential facts (the HF cache folder
    and the long-path switch). Wherever it is imported, the file must not
    name the token."""
    offenders = []
    for path in list((REPO / "nodes").rglob("*.py")) + [
            REPO / "__init__.py", REPO / "prestartup_script.py", REPO / "otr_hf_home_pin.py"]:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "import winreg" in text and "HF_TOKEN" in text:
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == []
