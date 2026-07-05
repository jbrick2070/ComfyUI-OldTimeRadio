"""ComfyUI import smoke: node mappings load and dropdown choices discover
from the registry (no ComfyUI required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nodes  # noqa: E402

mappings = sorted(nodes.NODE_CLASS_MAPPINGS)
banks = nodes._bank_choices()
models = nodes._model_choices()
styles = nodes._style_choices()
pipelines = nodes._pipeline_choices()

assert mappings == [
    "OTR_BridgeArtifactEmit", "OTR_StoryPackPreview", "OTR_UpstreamStoryLabValidator",
], mappings
assert "media_archive" in banks and "custom_source_bank" in banks
assert "simple_4_prompt_experimental" not in [m for m in models if m != "auto"], (
    "experimental pack must not appear in narrative model choices"
)
assert "archival_documentary" in styles and "auto" in styles
assert "legacy_many_pass" in pipelines

report = nodes.OTR_UpstreamStoryLabValidator().validate()[0]
assert report.startswith("OK Upstream Story Lab v2"), report[:80]

print("smoke OK")
print("nodes:", mappings)
print("banks:", banks)
print("styles:", styles)
