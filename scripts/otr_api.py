"""otr_api.py -- ComfyUI HTTP API helpers for OTR workflow JSONs.

BUG-LOCAL-002 fix (2026-05-02). Replaces scripts/soak_operator.py and
scripts/supersoaker.py, both of which carried stale `WV_*` positional
widget indices that no longer matched the live OTR_LLMScriptWriter node
(`episode_title` and `num_characters` widgets were added later, shifting
every downstream index off by 1-2 slots).

This module exposes:

  * `load_workflow(path)` -- read a UI-format workflow JSON.
  * `fetch_schemas()`     -- GET /object_info, returns the schema dict.
  * `patch_widget_by_name(workflow, node_id, widget_name, value, schemas)`
        -- writes to the slot named `widget_name`, regardless of position.
  * `workflow_to_api_prompt(workflow, schemas)`
        -- convert UI workflow JSON -> API prompt dict expected by /prompt.
  * `submit_prompt(api_prompt) -> prompt_id`
  * `poll_history(prompt_id, timeout_s=1800, on_tick=None)
        -> tuple[str, str]`
                -- returns ("SUCCESS"|"FAIL"|"TIMEOUT", error_message_if_any).

The module reads `COMFYUI_URL` from the env (default
`http://127.0.0.1:8000`) so callers can target a non-default host.

Determinism note: schemas come from the LIVE ComfyUI process, so
`patch_widget_by_name` is robust against future widget-order changes.
This is the deliberate fix for BUG-LOCAL-002.
"""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from typing import Any, Callable

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000")
DEFAULT_POLL_S = 5
DEFAULT_TIMEOUT_S = 1800

# Widget-capable input types. Mirrors soak_operator's set; any spec whose
# type is one of these (or a `list` of literal choices, i.e. a dropdown)
# consumes a slot in `widgets_values`.
_WIDGET_PRIMITIVE_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN", "BOOL", "COMBO"}


# ---------------------------------------------------------------------------
# Workflow IO
# ---------------------------------------------------------------------------
def load_workflow(path: str) -> dict:
    """Read a UI-format workflow JSON from disk and return a deep copy."""
    with open(path, "r", encoding="utf-8") as f:
        return copy.deepcopy(json.load(f))


def fetch_schemas() -> dict:
    """GET /object_info and return the schema dict.

    The result is cached only for the duration of one request -- callers
    that mutate widget order in the same Python process should re-fetch.
    """
    resp = requests.get(f"{COMFYUI_URL}/object_info", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------
def _is_widget_backed(spec: Any) -> bool:
    """Return True if a schema spec represents a widget-capable input.

    Widget-backed: primitives (STRING/INT/FLOAT/BOOLEAN/COMBO) and dropdowns
    (a Python list of literal choices). Custom socket types (PROJECT_STATE,
    LATENT, MODEL, ...) are NOT widget-backed and never consume a slot in
    `widgets_values`.
    """
    type_def = (
        spec[0]
        if isinstance(spec, (list, tuple)) and len(spec) > 0
        else spec
    )
    if isinstance(type_def, list):  # dropdown
        return True
    if isinstance(type_def, str) and type_def in _WIDGET_PRIMITIVE_TYPES:
        return True
    return False


def _spec_for(node_type: str, widget_name: str, schemas: dict) -> Any:
    """Return the raw input-spec for a (node_type, widget_name) pair.

    Looks at the schema's required+optional dicts; returns the spec value
    (a tuple/list/str) so callers can introspect type + choices. Raises
    KeyError if the widget is not declared on the node.
    """
    if node_type not in schemas:
        raise KeyError(
            f"node_type {node_type!r} not in /object_info schemas"
        )
    schema = schemas[node_type].get("input", {}) or {}
    required = schema.get("required", {}) or {}
    optional = schema.get("optional", {}) or {}
    if widget_name in required:
        return required[widget_name]
    if widget_name in optional:
        return optional[widget_name]
    raise KeyError(
        f"widget {widget_name!r} not declared on node_type {node_type!r}"
    )


def _validate_widget_value(
    node_type: str,
    widget_name: str,
    spec: Any,
    value: Any,
) -> None:
    """Assert `value` is compatible with the declared widget `spec`.

    BUG-LOCAL-002 follow-up (round-robin recommendation 2026-05-02): the
    name-keyed patcher protects against widget-position drift, but does not
    catch a caller passing the wrong VALUE shape (e.g. `True` for a STRING
    field, or `"medium"` for an INT). This helper adds a light type/range
    guardrail: COMBO values must be in the declared choice list, INT/FLOAT/
    BOOL/STRING must match Python types.

    Permissive on `None` (treated as "use default"). Permissive on numeric
    coercion (int -> FLOAT is fine; bool -> INT is rejected because bool is
    a subclass of int and we want to catch True/False being mis-routed).
    Raises ValueError on mismatch with a clear message naming the node + widget.
    """
    if value is None:
        return  # caller deliberately omitting -- let ComfyUI use the default

    type_def = (
        spec[0]
        if isinstance(spec, (list, tuple)) and len(spec) > 0
        else spec
    )

    # Dropdown / COMBO -- type_def is the list of choices.
    if isinstance(type_def, list):
        if value not in type_def:
            raise ValueError(
                f"widget {widget_name!r} on node_type {node_type!r} is a "
                f"COMBO with choices {type_def!r}; got {value!r} which is "
                f"not in the choice list."
            )
        return

    if not isinstance(type_def, str):
        # Unknown spec shape -- skip validation rather than refuse.
        return

    t = type_def.upper()

    if t == "BOOLEAN" or t == "BOOL":
        if not isinstance(value, bool):
            raise ValueError(
                f"widget {widget_name!r} on node_type {node_type!r} is "
                f"{t}; expected bool, got {type(value).__name__} ({value!r})"
            )
        return

    if t == "INT":
        # Reject bool (which is a subclass of int) because mistaken
        # True/False routed into an INT field is a real bug class.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"widget {widget_name!r} on node_type {node_type!r} is "
                f"INT; expected int, got {type(value).__name__} ({value!r})"
            )
        return

    if t == "FLOAT":
        # Accept int -> float coercion (caller passing 0 to a FLOAT field
        # is a normal Python idiom); reject bool.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"widget {widget_name!r} on node_type {node_type!r} is "
                f"FLOAT; expected number, got {type(value).__name__} "
                f"({value!r})"
            )
        return

    if t == "STRING":
        if not isinstance(value, str):
            raise ValueError(
                f"widget {widget_name!r} on node_type {node_type!r} is "
                f"STRING; expected str, got {type(value).__name__} "
                f"({value!r})"
            )
        return

    if t == "COMBO":
        # COMBO with no inline choices -- e.g. dynamic enum populated at
        # runtime. Skip choice validation.
        return

    # Unknown primitive -- skip.


def _ordered_widget_names(node_type: str, schemas: dict) -> list[str]:
    """Return the widget-backed input names for a node type, in declaration order.

    Order matches `widgets_values` slot mapping: required-then-optional, only
    widget-backed entries.
    """
    if node_type not in schemas:
        raise KeyError(
            f"node_type {node_type!r} not present in /object_info schemas. "
            f"Is the custom node loaded?"
        )
    schema = schemas[node_type].get("input", {}) or {}
    required = schema.get("required", {}) or {}
    optional = schema.get("optional", {}) or {}
    ordered = list(required.items()) + list(optional.items())
    return [name for name, spec in ordered if _is_widget_backed(spec)]


# ---------------------------------------------------------------------------
# Widget patching by NAME (not position)
# ---------------------------------------------------------------------------
def patch_widget_by_name(
    workflow: dict,
    node_id: int,
    widget_name: str,
    value: Any,
    schemas: dict,
) -> None:
    """Set the value of a widget on a node by its declared name.

    Looks up the widget's slot index from the live `/object_info` schemas,
    so callers don't need to know positional indices. Raises if the node or
    widget cannot be located -- silent miss is the historical bug we are
    explicitly killing here.
    """
    target_node = None
    for node in workflow.get("nodes", []):
        if node.get("id") == node_id:
            target_node = node
            break
    if target_node is None:
        raise KeyError(f"node id={node_id} not in workflow")

    node_type = target_node.get("type")
    widget_names = _ordered_widget_names(node_type, schemas)

    if widget_name not in widget_names:
        raise KeyError(
            f"widget {widget_name!r} not in widget order for {node_type!r}. "
            f"Known widgets: {widget_names!r}"
        )

    # BUG-LOCAL-002 follow-up (round-robin recommendation 2026-05-02):
    # validate the value's TYPE/CHOICES against the declared widget spec
    # before writing. Wrong-type writes are still legal at the JSON level
    # (the workflow is just text), but they manifest as silent runtime
    # degradation -- e.g. a bool written to a STRING field becomes the
    # literal string "True" downstream. Refuse loudly here so the call site
    # gets a clear error instead of producing an episode that drifts on a
    # mistyped widget value.
    _spec = _spec_for(node_type, widget_name, schemas)
    _validate_widget_value(node_type, widget_name, _spec, value)

    # BUG-LOCAL-002 follow-up round 2 (round-robin verdict 2026-05-02):
    # `None` is the documented "use the node's default" sentinel. Returning
    # early here -- AFTER validation has accepted it -- prevents a literal
    # `null` from being written into the workflow's widgets_values slot.
    # Many ComfyUI core nodes treat `null` as a parser error rather than
    # a default-fallback, so patching a value of `None` previously could
    # crash the run at queue-time. Match the documented behavior in the
    # _validate_widget_value docstring: None means "leave the slot alone".
    if value is None:
        return

    # ComfyUI's UI saves widgets_values in either "stripped" or "preserved"
    # mode depending on which widgets were converted to sockets. For pure
    # widget patching (no socket conversion on this node), the index in the
    # stored array equals the index in widget_names. If this node has any
    # widget converted to a socket (`linked_names` is non-empty for that
    # widget), the stored array may be one slot shorter per linked widget.
    # We probe the actual length and pick the simplest correct mapping.
    linked_names = {
        inp["name"]
        for inp in target_node.get("inputs", []) or []
        if inp.get("link") is not None and inp.get("name") in widget_names
    }

    wv = target_node.setdefault("widgets_values", [])
    target_idx = widget_names.index(widget_name)

    if widget_name in linked_names:
        # Trying to patch a widget that's been converted to an input socket
        # is a usage error: the value will come from the link, not from
        # widgets_values. Refuse loudly.
        raise ValueError(
            f"widget {widget_name!r} on node {node_id} has been converted "
            f"to an input socket; cannot patch via widgets_values. "
            f"Edit the upstream linked node instead."
        )

    # If linked widgets keep placeholder slots, our positional index is
    # already correct (slots include the placeholders). If they're stripped,
    # subtract the count of linked widgets that come before our target.
    linked_widget_count = sum(1 for n in widget_names if n in linked_names)
    if len(wv) == len(widget_names):
        # "preserved" mode -- use index as-is
        slot = target_idx
    elif len(wv) == len(widget_names) - linked_widget_count:
        # "stripped" mode -- subtract leading linked widgets
        leading_linked = sum(
            1 for n in widget_names[:target_idx] if n in linked_names
        )
        slot = target_idx - leading_linked
    else:
        # Ambiguous (trailing unset optionals, manual edits). Bail with a
        # clear error rather than silently writing to the wrong slot.
        raise ValueError(
            f"widgets_values length mismatch on node {node_id} ({node_type}): "
            f"len(wv)={len(wv)} vs "
            f"len(widget_names)={len(widget_names)} "
            f"(linked={linked_widget_count}). Refusing to patch by name."
        )

    # Pad widgets_values if our target slot is past the end (rare with
    # stripped mode + late optional widgets). Pad with None; the API
    # converter will drop these for unmapped slots anyway.
    while len(wv) <= slot:
        wv.append(None)
    wv[slot] = value


# ---------------------------------------------------------------------------
# UI workflow JSON  ->  /prompt API format
# ---------------------------------------------------------------------------
def workflow_to_api_prompt(workflow: dict, schemas: dict) -> dict:
    """Convert ComfyUI UI-format workflow JSON to the API prompt dict.

    Ported verbatim (with comments) from soak_operator's working converter,
    which carries the BUG-LOCAL-027 + BUG-LOCAL-029 fixes for socket-only
    inputs and "stripped" vs "preserved" widgets_values shapes.
    """
    # Build link map: link_id -> [source_node_id, source_slot]
    link_map: dict[int, list] = {}
    for lnk in workflow.get("links", []) or []:
        link_id, src_node, src_slot = lnk[0], lnk[1], lnk[2]
        link_map[link_id] = [str(src_node), src_slot]

    prompt: dict[str, Any] = {}
    for node in workflow.get("nodes", []):
        nid = str(node["id"])
        ntype = node["type"]

        inputs: dict[str, Any] = {}
        linked_names: set[str] = set()
        for inp in node.get("inputs", []) or []:
            if inp.get("link") is not None:
                inputs[inp["name"]] = link_map.get(inp["link"])
                linked_names.add(inp["name"])

        if ntype in schemas:
            schema = schemas[ntype].get("input", {}) or {}
            required = schema.get("required", {}) or {}
            optional = schema.get("optional", {}) or {}
            ordered_params = list(required.items()) + list(optional.items())

            wv = node.get("widgets_values", []) or []

            widget_backed_params = [
                (p, spec) for p, spec in ordered_params
                if _is_widget_backed(spec)
            ]
            linked_widget_count = sum(
                1 for p, _ in widget_backed_params if p in linked_names
            )
            unlinked_widget_count = len(widget_backed_params) - linked_widget_count

            if len(wv) == len(widget_backed_params) and linked_widget_count > 0:
                linked_keeps_slot = True
            elif len(wv) == unlinked_widget_count:
                linked_keeps_slot = False
            else:
                linked_keeps_slot = False  # safer default

            wv_idx = 0
            for param, spec in ordered_params:
                widget_backed = _is_widget_backed(spec)

                if param in linked_names:
                    if linked_keeps_slot and widget_backed:
                        if wv_idx < len(wv):
                            wv_idx += 1
                    continue

                if not widget_backed:
                    continue

                if wv_idx < len(wv):
                    inputs[param] = wv[wv_idx]
                    wv_idx += 1

        prompt[nid] = {"class_type": ntype, "inputs": inputs}
    return prompt


# ---------------------------------------------------------------------------
# Submit + poll
# ---------------------------------------------------------------------------
def submit_prompt(api_prompt: dict, client_id: str | None = None) -> str:
    """POST the API prompt to /prompt and return the prompt_id."""
    if client_id is None:
        client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": api_prompt, "client_id": client_id},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"POST /prompt -> HTTP {resp.status_code}: {resp.text[:500]}"
        )
    body = resp.json()
    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(
            f"submit_prompt: response missing prompt_id: {body!r}"
        )
    return prompt_id


def poll_history(
    prompt_id: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    poll_s: int = DEFAULT_POLL_S,
    on_tick: Callable[[float, dict], None] | None = None,
) -> tuple[str, str]:
    """Poll /history/<prompt_id> until completed/error/timeout.

    Returns (status, error_message). status is "SUCCESS", "FAIL", or
    "TIMEOUT". error_message is non-empty only on FAIL.
    `on_tick(elapsed_s, status_dict)` fires once per poll for callers that
    want to interleave their own log tail.
    """
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=10
            ).json()
        except Exception:
            r = {}
        status = (r.get(prompt_id, {}) or {}).get("status", {}) or {}
        if on_tick:
            try:
                on_tick(time.time() - start, status)
            except Exception:
                pass
        if status.get("completed", False):
            return ("SUCCESS", "")
        if status.get("status_str") == "error":
            return ("FAIL", str(status.get("messages", "execution error"))[:500])
        time.sleep(poll_s)
    return ("TIMEOUT", "")


def queue_snapshot() -> tuple[int, int]:
    """Return (running_count, pending_count) from /queue. Best-effort."""
    try:
        q = requests.get(f"{COMFYUI_URL}/queue", timeout=10).json()
        return len(q.get("queue_running", []) or []), len(q.get("queue_pending", []) or [])
    except Exception:
        return -1, -1


def cancel_queue() -> bool:
    """POST /queue {"clear": true} + /interrupt. Best-effort, returns True on 200/200."""
    ok = True
    try:
        r1 = requests.post(
            f"{COMFYUI_URL}/queue",
            json={"clear": True},
            timeout=5,
        )
        ok &= (r1.status_code == 200)
    except Exception:
        ok = False
    try:
        r2 = requests.post(f"{COMFYUI_URL}/interrupt", timeout=5)
        ok &= (r2.status_code == 200)
    except Exception:
        ok = False
    return ok


__all__ = [
    "COMFYUI_URL",
    "load_workflow",
    "fetch_schemas",
    "patch_widget_by_name",
    "workflow_to_api_prompt",
    "submit_prompt",
    "poll_history",
    "queue_snapshot",
    "cancel_queue",
]
