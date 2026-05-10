"""Phase 3 schema validation gate.

The OTR repo's directory name has a hyphen (ComfyUI-OldTimeRadio) which
Python's `import` statement cannot parse. We use importlib.import_module()
which accepts arbitrary strings, the same trick ComfyUI itself uses to
load this package. This preserves the package context so the modules'
relative imports (from .project_state, etc.) resolve correctly.
"""
import sys
import pathlib
import importlib

_THIS = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent              # tests/file -> repo root
_CUSTOM_NODES = _REPO_ROOT.parent             # custom_nodes/
_COMFY_ROOT = _CUSTOM_NODES.parent            # ComfyUI/
sys.path.insert(0, str(_COMFY_ROOT))
sys.path.insert(0, str(_CUSTOM_NODES))

_PKG = _REPO_ROOT.name                        # "ComfyUI-OldTimeRadio"

_ledger_mod = importlib.import_module(_PKG + ".nodes.OTR_LedgerScriptWriter")
_legacy_mod = importlib.import_module(_PKG + ".nodes._otr_legacy_writer")
_so = importlib.import_module(_PKG + ".nodes.story_orchestrator")


def validate(node_cls, name):
    it = node_cls.INPUT_TYPES()
    assert isinstance(it, dict), name + ".INPUT_TYPES not a dict"
    for bucket in ("required", "optional", "hidden"):
        if bucket not in it:
            continue
        assert isinstance(it[bucket], dict), name + "." + bucket + " not dict"
        for wname, spec in it[bucket].items():
            assert isinstance(wname, str), \
                name + "." + bucket + ": bad name " + repr(wname)
            assert isinstance(spec, tuple) and len(spec) >= 1, \
                name + "." + bucket + "." + wname + ": bad spec " + repr(spec)
            assert spec[0] is not None, \
                name + "." + bucket + "." + wname + ": type is None"
    req = len(it.get("required", {}))
    opt = len(it.get("optional", {}))
    print("  " + name + ": OK (" + str(req) + " required, "
          + str(opt) + " optional)")


validate(_ledger_mod.OTR_LedgerScriptWriter, "OTR_LedgerScriptWriter")
validate(_legacy_mod.LegacyLLMScriptWriter, "LegacyLLMScriptWriter")

assert _so.LLMScriptWriter is _legacy_mod.LegacyLLMScriptWriter, \
    "shim broken: story_orchestrator.LLMScriptWriter is not LegacyLLMScriptWriter"
print("  shim: OK")

for cls, name in [
    (_legacy_mod.LegacyLLMScriptWriter, "Legacy"),
    (_ledger_mod.OTR_LedgerScriptWriter, "Ledger"),
]:
    assert cls.RETURN_TYPES == ("STRING", "STRING", "STRING", "INT")
    assert cls.RETURN_NAMES == (
        "script_text", "script_json", "news_used", "estimated_minutes"
    )
    print("  " + name + ": RETURN contract OK")

print("SCHEMA GATE PASS")
