"""One-shot git commit and push for Phase 0 work."""
import subprocess
import sys
import os

REPO = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"

def run(cmd):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True, timeout=60)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.stderr.strip():
        print(r.stderr.strip())
    return r.returncode

# Clear any stale lock
lock = os.path.join(REPO, ".git", "index.lock")
if os.path.exists(lock):
    os.remove(lock)
    print("Cleared stale index.lock")

# Stage real changes only (not CRLF noise)
files = [
    "CLAUDE.md",
    # nodes/v2_preview.py removed (sidecar architecture pending)
    "tests/test_core.py",
    "BUG_LOG.md",
    "otr_v2/__init__.py",
    "otr_v2/subprocess_runners/__init__.py",
    "otr_v2/schema/visual_plan.schema.json",
    "tests/v2/__init__.py",
    "tests/v2/test_audio_byte_identical.py",
    "tests/v2/_run_baseline.py",
    "tests/v2/_fix_workflow.py",
]

for f in files:
    rc = run(f'git add "{f}"')
    if rc != 0:
        print(f"Failed to add {f}")

# Remove deleted test workflow
run("git rm --cached workflows/otr_scifi_16gb_test.json")

# Commit
run('git commit -m "Phase 0: audio regression baseline, widget-drift fixes, v2 scaffolding"')

# Push using token
TOKEN = sys.argv[1] if len(sys.argv) > 1 else ""
if TOKEN:
    remote = f"https://jbrick2070:{TOKEN}@github.com/jbrick2070/ComfyUI-OldTimeRadio.git"
    run(f'git push "{remote}" v2.0-alpha')
else:
    run("git push origin v2.0-alpha")
