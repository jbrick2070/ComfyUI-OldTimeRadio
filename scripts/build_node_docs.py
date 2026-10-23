"""Write the node help pages ComfyUI shows for a node's ? button.

ComfyUI's frontend looks for ``WEB_DIRECTORY/docs/<NodeClass>.md`` (and
``<NodeClass>/<locale>.md`` beside it) and shows that page in the node's help
panel in place of the bare DESCRIPTION-plus-tooltips view. This writes one page
per registered node into ``js/docs/``, joining two sources:

* ``config/node_docs.json`` -- the hand-written half: what the node is for and
  where it sits, and which shipped guides explain it;
* the node's own definition -- DESCRIPTION, every input with its tooltip,
  every output -- read from the class, so the page cannot drift from the node.

``tests/test_node_docs_are_generated.py`` fails when a committed page differs
from what this writes: edit the config or the node, run this, commit both.

    python scripts/build_node_docs.py          # write js/docs/*.md
    python scripts/build_node_docs.py --check  # exit 1 when any page differs
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "js" / "docs"
CONFIG = REPO / "config" / "node_docs.json"
GUIDES = "https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/"

#: The scalar widget kinds, in the words the page uses for them.
_KINDS = {"INT": "a whole number", "FLOAT": "a number", "BOOLEAN": "on or off",
          "STRING": "text"}


def _prepare_runtime() -> None:
    """Import the pack the way its test suite does: no GPU, test mode, and a
    stand-in for ComfyUI's ``folder_paths`` whose file lists are empty."""
    os.environ.setdefault("OTR_TEST_MODE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if "folder_paths" not in sys.modules:
        stub = types.ModuleType("folder_paths")
        out = tempfile.mkdtemp(prefix="otr_node_docs_")
        stub.get_output_directory = lambda: out  # noqa: E731
        stub.get_filename_list = lambda *args, **kwargs: []  # noqa: E731
        sys.modules["folder_paths"] = stub
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def node_table() -> dict:
    """``{key: (module_path, class_name, display_name)}`` as the package
    ``__init__`` builds it: the literal table, plus the class registry's rows
    whose modules exist -- read, not executed, so no ComfyUI boot runs here."""
    tree = ast.parse((REPO / "__init__.py").read_text(encoding="utf-8"))
    table: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_NODE_MODULES" for t in node.targets):
            table.update(ast.literal_eval(node.value))
    from nodes._otr_class_registry import new_node_modules_table
    for key, value in new_node_modules_table().items():
        rel = value[0].lstrip(".").replace(".", os.sep) + ".py"
        if (REPO / rel).exists():
            table[key] = value
    return table


def load_class(module_path: str, class_name: str):
    module = importlib.import_module(module_path.lstrip("."))
    return getattr(module, class_name)


def _kind(spec) -> str:
    """What an input takes, in words, from its INPUT_TYPES spec."""
    kind = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(kind, (list, tuple)):
        return "a choice from the dropdown"
    if opts.get("forceInput"):
        # A widget type kept as a link socket on purpose (Codex QA, 2026-10-23).
        return f"a connection from another node (`{kind}`)"
    if kind == "STRING" and opts.get("multiline"):
        return "text, as many lines as you like"
    if kind in _KINDS:
        words = _KINDS[kind]
        if kind in ("INT", "FLOAT") and ("min" in opts or "max" in opts):
            lo, hi = opts.get("min"), opts.get("max")
            if lo is not None and hi is not None:
                words += f" from {lo} to {hi}"
            elif lo is not None:
                words += f", at least {lo}"
            elif hi is not None:
                words += f", at most {hi}"
        return words
    return f"a connection from another node (`{kind}`)"


def _default(spec):
    opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    if "default" not in opts:
        return None
    value = opts["default"]
    if opts.get("forceInput"):
        return None
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        if not value.strip():
            return "empty"
        if os.path.isabs(value) or "\\" in value or value.startswith(("./", "../", "~")):
            # A path, which a node derives from its own checkout: the page
            # must read the same on every machine (Codex QA, 2026-10-23). A
            # model slug with a colon or a slash in it is not one.
            return "a path on this machine"
        return f"`{value}`"
    return f"`{value}`"


def _form_labels() -> dict:
    cfg = json.loads((REPO / "config" / "app_mode.json").read_text(encoding="utf-8"))
    return dict(cfg.get("labels") or {})


def render(key: str, entry: dict, display_name: str, cls, labels: dict) -> str:
    """One node's page."""
    title = display_name.strip()
    lines = [f"<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the "
             f"node's own definition. Edit those, then run the script. -->",
             f"# {title}", "", entry["intro"].strip(), ""]
    description = (getattr(cls, "DESCRIPTION", "") or "").strip()
    if description:
        lines += ["## What it does", "", description, ""]
    spec = cls.INPUT_TYPES()
    sections = [("required", spec.get("required") or {}), ("optional", spec.get("optional") or {})]
    if any(inputs for _, inputs in sections):
        lines += ["## Inputs", ""]
        if entry.get("touch") == "yes":
            lines += ["On the shipped workflows these are the widgets a person changes; the "
                      "ones on the app form are named as the form names them.", ""]
        else:
            lines += ["The shipped workflows set these; most people never change them.", ""]
        for section, inputs in sections:
            if not inputs:
                continue
            if section == "optional":
                lines += ["### Optional", ""]
            for name, item in inputs.items():
                opts = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], dict) else {}
                label = labels.get(f"{key}.{name}")
                head = f"**{label}** (`{name}`)" if label else f"**`{name}`**"
                facts = _kind(item)
                default = _default(item)
                if default is not None:
                    facts += f"; starts as {default}"
                lines += [f"{head} -- {facts}.", ""]
                tooltip = str(opts.get("tooltip") or "").strip()
                if tooltip:
                    lines += [tooltip, ""]
    returns = tuple(getattr(cls, "RETURN_TYPES", ()) or ())
    names = tuple(getattr(cls, "RETURN_NAMES", ()) or ())
    if returns:
        lines += ["## Outputs", ""]
        for i, rtype in enumerate(returns):
            name = names[i] if i < len(names) else rtype
            lines.append(f"- `{name}` (`{rtype}`)")
        lines.append("")
    if getattr(cls, "OUTPUT_NODE", False):
        lines += ["This is an output node: the workflow runs to reach it.", ""]
    lines += ["## Read more", ""]
    for guide in entry.get("guides") or []:
        lines.append(f"- [{guide}]({GUIDES}{guide})")
    lines += ["", f"Class `{key}`, category `{getattr(cls, 'CATEGORY', '')}`.", ""]
    return "\n".join(lines)


def render_all() -> dict:
    """``{key: page}`` for every registered node."""
    _prepare_runtime()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config.pop("_comment", None)
    table = node_table()
    missing = sorted(set(table) - set(config))
    extra = sorted(set(config) - set(table))
    if missing or extra:
        raise SystemExit(f"config/node_docs.json: no entry for {missing}; "
                         f"entries for nodes that do not exist: {extra}")
    labels = _form_labels()
    pages = {}
    for key, (module_path, class_name, display_name) in sorted(table.items()):
        cls = load_class(module_path, class_name)
        pages[key] = render(key, config[key], display_name, cls, labels)
    return pages


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report differences; write nothing")
    args = ap.parse_args(argv)
    pages = render_all()
    DOCS.mkdir(parents=True, exist_ok=True)
    stale = sorted(p.name for p in DOCS.glob("*.md") if p.stem not in pages)
    differing = []
    for key, page in pages.items():
        path = DOCS / f"{key}.md"
        current = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else None
        if current != page:
            differing.append(key)
            if not args.check:
                path.write_bytes(page.encode("utf-8"))
    if args.check:
        for name in differing:
            print(f"  differs: js/docs/{name}.md")
        for name in stale:
            print(f"  stale: js/docs/{name}")
        print(f"check: {len(pages)} pages, {len(differing)} differ, {len(stale)} stale")
        return 1 if differing or stale else 0
    for name in stale:
        (DOCS / name).unlink()
    print(f"wrote {len(differing)} of {len(pages)} pages; removed {len(stale)} stale")
    return 0


if __name__ == "__main__":
    sys.exit(main())
