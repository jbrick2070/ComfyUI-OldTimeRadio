"""Phase 3 saved-workflow binding gate.

Uses importlib.import_module() to handle the hyphenated package name
(ComfyUI-OldTimeRadio cannot be parsed by `import` statements).
"""
import sys
import pathlib
import importlib
import json

_THIS = pathlib.Path(__file__).resolve()
_REPO_ROOT = _THIS.parent.parent
_CUSTOM_NODES = _REPO_ROOT.parent
_COMFY_ROOT = _CUSTOM_NODES.parent
sys.path.insert(0, str(_COMFY_ROOT))
sys.path.insert(0, str(_CUSTOM_NODES))

_PKG = _REPO_ROOT.name
_legacy_mod = importlib.import_module(_PKG + ".nodes._otr_legacy_writer")

WF = pathlib.Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\otr_scifi_16gb_full.json")
wf = json.loads(WF.read_text(encoding="utf-8"))
node = next(n for n in wf["nodes"] if n.get("type") == "OTR_LLMScriptWriter")
saved = node.get("widgets_values", [])

it = _legacy_mod.LegacyLLMScriptWriter.INPUT_TYPES()
all_widgets = (
    list(it.get("required", {}).keys())
    + list(it.get("optional", {}).keys())
)

print("Saved widgets_values: " + str(len(saved)))
print("Current INPUT_TYPES widgets: " + str(len(all_widgets)))

if len(saved) > len(all_widgets):
    raise AssertionError(
        "Saved has " + str(len(saved)) + " values; current declares "
        + str(len(all_widgets)) + ". Saved workflow will FAIL to load."
    )
if len(saved) < len(all_widgets):
    print("  WARN: trailing-default drift (acceptable)")

print("WORKFLOW BINDING GATE PASS")
