"""Scratch: verify GitHub root matches local and no stale QA docs exist."""
import json, os

# Parse the GitHub API response
api_file = r"C:\Users\jeffr\.gemini\antigravity\brain\27c109fc-1fce-4b67-a0b0-4651da98acbb\.system_generated\steps\185\content.md"
with open(api_file, "r", encoding="utf-8") as f:
    raw = f.read()
data = json.loads(raw.split("---\n\n")[1].strip())

print("=== FILES ON GITHUB (main branch) ===")
for item in data:
    kind = "DIR " if item["type"] == "dir" else "FILE"
    size = item.get("size", 0)
    name = item["name"]
    print(f"  {kind}  {name:35s}  {size:>8} bytes")

# Check stale docs
stale_names = [
    "QA_PEER_REVIEW_GUIDE.md",
    "CLAUDE_v1.3_LOCKED.md",
    "V1.3_OOM_EMERGENCY.md",
    "VRAM_QA.md",
    "FAILED_SCRIPT_DUMP.txt",
]
github_names = [item["name"] for item in data]
found_stale = [s for s in stale_names if s in github_names]

print()
if found_stale:
    print(f"WARNING: Stale files STILL on GitHub: {found_stale}")
else:
    print("CONFIRMED: All stale QA/debug docs deleted from GitHub. CLEAN.")

# Verify key files present
required = ["CLAUDE.md", "ROADMAP.md", "__init__.py", "README.md"]
missing = [r for r in required if r not in github_names]
if missing:
    print(f"WARNING: Required files missing from GitHub: {missing}")
else:
    print("CONFIRMED: All required files present on GitHub.")

# Compare local root files vs GitHub
local_root = os.listdir(".")
local_files = sorted([f for f in local_root if os.path.isfile(f)])
github_files = sorted([item["name"] for item in data if item["type"] == "file"])

only_local = [f for f in local_files if f not in github_files and not f.startswith(".")]
only_github = [f for f in github_files if f not in local_files]

print()
if only_local:
    print(f"Local-only (not on GitHub): {only_local}")
else:
    print("No untracked root files. CLEAN.")
if only_github:
    print(f"GitHub-only (not local): {only_github}")
