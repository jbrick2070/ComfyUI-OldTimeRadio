"""Scratch script: audit node registrations in __init__.py vs node files."""
import re, os

init_path = "__init__.py"
with open(init_path, "r", encoding="utf-8") as f:
    init_text = f.read()

registered = re.findall(r'"(OTR_\w+)"', init_text)

node_dir = "nodes"
node_files_classes = []
for fname in os.listdir(node_dir):
    if fname.endswith(".py") and not fname.startswith("_"):
        fpath = os.path.join(node_dir, fname)
        with open(fpath, "r", encoding="utf-8") as nf:
            text = nf.read()
        classes = re.findall(r'"(OTR_\w+)"', text)
        for c in classes:
            node_files_classes.append((fname, c))

print("=== Node Registration Audit ===")
for fname, cls in sorted(node_files_classes):
    status = "OK" if cls in registered else "MISSING from __init__.py"
    print(f"  {cls:45s} ({fname}) -> {status}")
print(f"\nTotal node classes: {len(node_files_classes)}, Registered in __init__: {len(registered)}")

missing = [c for _, c in node_files_classes if c not in registered]
if missing:
    print(f"\nWARNING: {len(missing)} unregistered node(s)!")
else:
    print("\nAll nodes registered. CLEAN.")
