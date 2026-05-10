"""Phase 3 story_orchestrator import smoke (Gate 5b)."""
import sys
import pathlib
import importlib

_THIS = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
_CUSTOM_NODES = _REPO_ROOT.parent
_COMFY_ROOT = _CUSTOM_NODES.parent
sys.path.insert(0, str(_COMFY_ROOT))
sys.path.insert(0, str(_CUSTOM_NODES))

_PKG = _REPO_ROOT.name
_so = importlib.import_module(_PKG + ".nodes.story_orchestrator")
_legacy = importlib.import_module(_PKG + ".nodes._otr_legacy_writer")

assert _so.LLMScriptWriter is not None, "LLMScriptWriter not resolvable"
assert _so.LLMScriptWriter is _legacy.LegacyLLMScriptWriter, \
    "shim drift: _so.LLMScriptWriter is not LegacyLLMScriptWriter"

print("story_orchestrator import OK")
print("LLMScriptWriter resolves to:", _so.LLMScriptWriter.__name__)
print("class lives in module:", _so.LLMScriptWriter.__module__)
print("shim assertion: PASS")
