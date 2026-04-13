"""
One-shot workflow sanitizer for baseline capture.

Restores the workflow from git HEAD, strips emoji from dropdown values,
and writes back clean JSON. Run once before baseline capture.

Usage:
  python tests/v2/_fix_workflow.py
"""
import json
import os
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
WF_PATH = os.path.join(REPO, "workflows", "otr_scifi_16gb_full.json")

# Step 1: Restore from git to undo any corruption
print("Restoring workflow from git HEAD...")
subprocess.run(
    ["git", "checkout", "--", "workflows/otr_scifi_16gb_full.json"],
    cwd=REPO, check=True,
)
print("  Restored.")

# Step 2: Load and fix
print("Loading workflow...")
with open(WF_PATH, "r", encoding="utf-8") as f:
    raw = f.read()

# Emoji -> [EMOJI] / [FAST] mapping
# These match the dropdown options in story_orchestrator.py INPUT_TYPES
EMOJI_MAP = {
    "\U0001f4fb test (1 min)": "[EMOJI] test (1 min)",
    "\U0001f4fb standard (12 min)": "[EMOJI] standard (12 min)",
    "\U0001f4fb long (15 min)": "[EMOJI] long (15 min)",
    "\U0001f4fb epic (20 min)": "[EMOJI] epic (20 min)",
    "\U0001f4fb custom": "[EMOJI] custom",
}

replaced = 0
for old, new in EMOJI_MAP.items():
    if old in raw:
        raw = raw.replace(old, new)
        replaced += 1
        print(f"  Replaced: {repr(old[:30])} -> {new}")

if replaced == 0:
    print("  No emoji replacements needed.")

# Step 3: Validate JSON
print("Validating JSON...")
wf = json.loads(raw)
assert "nodes" in wf and "links" in wf
print(f"  Valid: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

# Step 4: Write back (Python UTF-8, no BOM)
print("Writing clean workflow...")
with open(WF_PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(raw)

print(f"\nDone. Workflow is clean at:\n  {WF_PATH}")
