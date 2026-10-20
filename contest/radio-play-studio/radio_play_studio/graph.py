"""Workflow file -> API graph, and the typed rules for the few inputs a person sets.

The Comfy SDK runs an API-format graph. The pack ships a litegraph (canvas)
workflow, so this module does the conversion against the node definitions of
the ComfyUI that will actually run it: a widget's saved position means nothing
without the node's input order, and only the live server knows that.

`workflow_to_api_prompt` is a standard-library port of
`nodes/_otr_workflow_apply.py::workflow_to_api_prompt` in the pack. It is a copy
on purpose, so this app runs without the pack's Python (and without torch) on
its path; `tests/test_contest_app_converter_parity.py` in the repo root pins the
two to the same output.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

#: Canvas notes. The frontend never sends them, so neither do we.
NOTE_NODE_TYPES = frozenset({"Note", "MarkdownNote"})

_WIDGET_PRIMITIVE_TYPES = frozenset({"STRING", "INT", "FLOAT", "BOOLEAN", "BOOL", "COMBO"})
_COMPANION_INT_WIDGETS = ("seed", "noise_seed")
_COMPANION_SUFFIX = "__control_after_generate"
_COMPANION_VALUES = frozenset({"fixed", "randomize", "increment", "decrement"})
#: A picker declared with an upload button saves one more slot right after it
#: ("image", "video" or "audio"), as core's LoadImage saves ["file.png", "image"].
_UPLOAD_SUFFIX = "__upload"
_UPLOAD_VALUES = frozenset({"image", "video", "audio"})
_UPLOAD_KEYS = ("image_upload", "video_upload", "audio_upload")


class GraphError(ValueError):
    """The workflow cannot be turned into a graph this server will accept."""


class ServerUnreachable(RuntimeError):
    """ComfyUI did not answer `/object_info`."""


# ---------------------------------------------------------------------------
# Reading the workflow and the server
# ---------------------------------------------------------------------------
def load_workflow(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fetch_object_info(base_url: str, timeout: float = 120.0) -> dict:
    """GET `<base_url>/object_info`: every node type the server has and what
    each one accepts. A ComfyUI that is rendering took 25 to 39 s to answer on
    the 5080 (2026-10-19), so the timeout is well above that."""
    url = base_url.rstrip("/") + "/object_info"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ServerUnreachable(
            f"could not read {url} ({exc}). Is ComfyUI running at {base_url}?"
        ) from exc


# ---------------------------------------------------------------------------
# Saved widget positions
# ---------------------------------------------------------------------------
def _is_widget_backed(spec: Any) -> bool:
    type_def = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    if isinstance(type_def, (list, tuple)):  # dropdown choices
        return True
    return isinstance(type_def, str) and type_def in _WIDGET_PRIMITIVE_TYPES


def _has_force_input(spec: Any) -> bool:
    if not isinstance(spec, (list, tuple)) or len(spec) < 2:
        return False
    opts = spec[1]
    return isinstance(opts, dict) and bool(opts.get("forceInput"))


def _declared_inputs(node_type: str, schemas: dict) -> list:
    schema = (schemas[node_type].get("input") or {})
    return list((schema.get("required") or {}).items()) + list((schema.get("optional") or {}).items())


def serialized_slot_names(node_type: str, schemas: dict) -> list:
    """The widget names, in the order `widgets_values` was saved: link-only
    inputs dropped, and the seed widgets' hidden control companion added."""
    if node_type not in schemas:
        raise KeyError(f"node type {node_type!r} is not on the server")
    slots: list = []
    for name, spec in _declared_inputs(node_type, schemas):
        if not _is_widget_backed(spec) or _has_force_input(spec):
            continue
        slots.append(name)
        type_def = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
        if (isinstance(type_def, str) and type_def.upper() == "INT"
                and name in _COMPANION_INT_WIDGETS):
            slots.append(name + _COMPANION_SUFFIX)
        opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
        if isinstance(opts, dict) and any(opts.get(k) for k in _UPLOAD_KEYS):
            slots.append(name + _UPLOAD_SUFFIX)
    return slots


def workflow_to_api_prompt(workflow: dict, schemas: dict) -> dict:
    """Canvas workflow -> API graph `{node_id: {class_type, inputs}}`.

    Fails loud rather than send a graph the server would misread: a node type
    the server does not have (the pack is not installed, or is a different
    version) and a `widgets_values` list that does not fit the node's inputs
    both raise `GraphError` naming the node."""
    link_map: dict = {}
    for link in workflow.get("links") or []:
        link_map[link[0]] = [str(link[1]), link[2]]

    prompt: dict = {}
    for node in workflow.get("nodes") or []:
        node_id = str(node["id"])
        node_type = node["type"]
        if node_type in NOTE_NODE_TYPES:
            continue
        if node_type not in schemas:
            raise GraphError(
                f"node {node_id} is a {node_type!r}, which this ComfyUI does not have. "
                f"Install the Old-Time Radio pack (comfyui-old-time-radio) and restart ComfyUI."
            )
        inputs: dict = {}
        linked: set = set()
        for socket in node.get("inputs") or []:
            if socket.get("link") is not None:
                inputs[socket["name"]] = link_map.get(socket["link"])
                linked.add(socket["name"])

        slots = serialized_slot_names(node_type, schemas)
        values = node.get("widgets_values") or []
        linked_widgets = sum(1 for name in slots if name in linked)
        if len(values) == len(slots):
            linked_keeps_slot = True
        elif len(values) == len(slots) - linked_widgets or not values:
            linked_keeps_slot = False
        else:
            raise GraphError(
                f"node {node_id} ({node_type}): the workflow saved {len(values)} widget "
                f"values but this server's node takes {len(slots)} "
                f"({linked_widgets} linked). The pack and the workflow are different "
                f"versions: update the pack and restart ComfyUI."
            )

        cursor = 0
        for name in slots:
            if name.endswith(_COMPANION_SUFFIX) or name.endswith(_UPLOAD_SUFFIX):
                allowed = _UPLOAD_VALUES if name.endswith(_UPLOAD_SUFFIX) else _COMPANION_VALUES
                if cursor < len(values):
                    if values[cursor] not in allowed:
                        raise GraphError(
                            f"node {node_id} ({node_type}): value {values[cursor]!r} at "
                            f"position {cursor} should be one of {sorted(allowed)}; "
                            f"the widget order has drifted."
                        )
                    cursor += 1
                continue
            if name in linked:
                if linked_keeps_slot and cursor < len(values):
                    cursor += 1
                continue
            if cursor < len(values):
                value = values[cursor]
                # A list widget value goes out wrapped, exactly as the
                # frontend sends it: a bare list in an API graph reads as a link.
                inputs[name] = {"__value__": value} if isinstance(value, list) else value
                cursor += 1
        prompt[node_id] = {"class_type": node_type, "inputs": inputs}
    return prompt


def node_id_for_type(workflow: dict, node_type: str) -> str:
    """The id of the ONE node of this type. Zero or several is an error: the app
    addresses nodes by type so it survives a re-saved workflow's new ids."""
    ids = [str(n["id"]) for n in workflow.get("nodes") or [] if n.get("type") == node_type]
    if len(ids) != 1:
        raise GraphError(
            f"expected exactly one {node_type} in the workflow, found {len(ids)}"
        )
    return ids[0]


# ---------------------------------------------------------------------------
# What an input accepts
# ---------------------------------------------------------------------------
def has_input(schemas: dict, node_type: str, name: str) -> bool:
    """True when the server's `node_type` takes an input called `name`."""
    schema = (schemas.get(node_type) or {}).get("input") or {}
    return any(name in (schema.get(section) or {}) for section in ("required", "optional"))


def describe_input(schemas: dict, node_type: str, name: str) -> dict:
    """One input's rules, normalised across the two shapes a server reports a
    dropdown in (a bare list, or `("COMBO", {"options": [...]})`)."""
    if node_type not in schemas:
        raise GraphError(f"node type {node_type!r} is not on the server")
    spec = None
    schema = schemas[node_type].get("input") or {}
    for section in ("required", "optional"):
        if name in (schema.get(section) or {}):
            spec = schema[section][name]
            break
    if spec is None:
        raise GraphError(f"{node_type} has no input named {name!r}")

    type_def = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
    info: dict = {"default": opts.get("default"), "tooltip": opts.get("tooltip", "")}
    if isinstance(type_def, (list, tuple)):
        info.update(kind="choice", choices=list(type_def))
    elif type_def == "COMBO":
        info.update(kind="choice", choices=list(opts.get("options") or []))
    elif type_def == "INT":
        info.update(kind="int", min=opts.get("min"), max=opts.get("max"))
    elif type_def == "FLOAT":
        info.update(kind="float", min=opts.get("min"), max=opts.get("max"))
    elif type_def in ("BOOLEAN", "BOOL"):
        info.update(kind="bool")
    elif type_def == "STRING":
        info.update(kind="text", multiline=bool(opts.get("multiline")),
                    placeholder=str(opts.get("placeholder") or ""))
    else:
        raise GraphError(f"{node_type}.{name} is a {type_def!r}, not something a form can set")
    return info


def coerce_value(info: dict, value: Any, label: str) -> Any:
    """`value` as the server expects it, or `ValueError` saying what was wrong."""
    kind = info["kind"]
    if kind == "choice":
        for choice in info["choices"]:
            if value == choice or str(value) == str(choice):
                return choice
        raise ValueError(f"{label}: {value!r} is not one of {info['choices']}")
    if kind in ("int", "float"):
        if isinstance(value, bool):
            raise ValueError(f"{label}: expected a number, got {value!r}")
        try:
            number = float(value) if kind == "float" else value
            if kind == "int":
                # 2.7 is not a whole number of anything; do not quietly make it 2.
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError("not a whole number")
                number = int(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{label}: expected a number, got {value!r}") from None
        if kind == "float" and not math.isfinite(number):
            raise ValueError(f"{label}: expected a number, got {value!r}")
        lo, hi = info.get("min"), info.get("max")
        if (lo is not None and number < lo) or (hi is not None and number > hi):
            raise ValueError(f"{label}: {number} is outside {lo}..{hi}")
        return number
    if kind == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError(f"{label}: expected true or false, got {value!r}")
    if not isinstance(value, str):
        raise ValueError(f"{label}: expected text, got {value!r}")
    return value
