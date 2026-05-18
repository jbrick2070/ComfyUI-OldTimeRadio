"""otr_api.py -- ComfyUI HTTP API helpers for OTR workflow JSONs.

BUG-LOCAL-002 fix (2026-05-02). Replaces scripts/soak_operator.py and
scripts/supersoaker.py, both of which carried stale `WV_*` positional
widget indices that no longer matched the live OTR_LedgerScriptWriter node
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

    Note: this returns ONLY schema-declared widgets. It does NOT include
    ComfyUI's client-side `control_after_generate` companion that the UI
    auto-injects next to seed widgets. For the actual stored slot layout in
    `widgets_values`, use `_serialized_slot_names` instead. Other introspection
    consumers (tests, schema audits) keep using this pure-schema variant.
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


# Companion-bearing widget names. ComfyUI's UI auto-injects a
# `control_after_generate` companion widget IMMEDIATELY after any INT
# widget whose schema name is one of these. The companion is a hidden
# COMBO (choices = ["fixed", "increment", "decrement", "randomize"]) and
# its saved value lives at slot `<widget>_idx + 1` in widgets_values.
# It is NOT declared in /object_info, so `_ordered_widget_names` does
# not see it -- but `widgets_values` saved by the editor always
# includes it.
_COMPANION_INT_WIDGETS = ("seed", "noise_seed")

# Acceptable saved values for the `control_after_generate` companion
# widget. Anything else at a companion position is widget drift, NOT a
# legitimate companion -- the converter raises rather than silently
# pushing the stray value into a downstream declared input.
_COMPANION_VALUES = frozenset({"fixed", "randomize", "increment", "decrement"})


def _spec_has_force_input(spec: Any) -> bool:
    """Return True if a schema spec carries `forceInput: True` in its opts.

    Background (Jeffrey 2026-05-17 round-robin Reading D / node 20):
    ComfyUI treats widget-backed inputs with `forceInput=True` as
    socket-only -- the UI never renders them as a widget, and
    `widgets_values` never carries a slot for them, regardless of
    whether the input is linked. This is symmetric to the
    `control_after_generate` companion: it's a ComfyUI client-side
    serialization rule that `/object_info` does not encode visibly
    (the flag is reachable but the slot-omission consequence is not).

    Surfaced on `OTR_VideoPlan` node 20 where `freeze_done_gate` is
    declared STRING + forceInput=True. Live save: 6 slots. Mapper
    pre-fix said 7. Conversion refused at length mismatch.
    """
    if not isinstance(spec, (list, tuple)) or len(spec) < 2:
        return False
    opts = spec[1]
    if not isinstance(opts, dict):
        return False
    return bool(opts.get("forceInput"))


def _serialized_slot_names(node_type: str, schemas: dict) -> list[str]:
    """Return the widget-name list that maps to the SAVED `widgets_values`
    layout 1-to-1, including ComfyUI's `control_after_generate` companion
    next to each seed widget AND excluding inputs flagged `forceInput=True`.

    Mapping:
        For each declared widget-backed input in declaration order:
            * If the input is flagged `forceInput=True`, SKIP -- never
              occupies a `widgets_values` slot.
            * Otherwise yield the widget's name.
            * If the widget is an INT and its name is one of
              `_COMPANION_INT_WIDGETS`, yield a synthetic companion
              entry named `f"{widget_name}__control_after_generate"`
              immediately after.

    The synthetic companion name is NEVER a real schema name -- it is a
    deliberate fabrication so callers can talk about the slot
    unambiguously (e.g. tests pinning the saved `"fixed"` value).

    Result: for a clean-saved workflow JSON, `len(_serialized_slot_names)`
    equals `len(node.widgets_values)` exactly. Drift between this list
    length and the saved array length is a structural problem; the
    patcher raises rather than silently misalign.

    Background (Jeffrey 2026-05-17 round-robin Reading C + D): two
    ComfyUI client-side serialization rules that the live
    `/object_info` schemas do not encode visibly. The mapper inverts
    BOTH against the declared widget set to recover the actual saved
    layout 1-to-1:
        - INT `seed` / `noise_seed` widgets gain a hidden
          `control_after_generate` companion slot (Reading C, commit
          c2c06e9 + 8df3d0a).
        - `forceInput=True` widget-backed inputs LOSE their widget
          slot entirely (Reading D, this revision).
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

    slots: list[str] = []
    for name, spec in ordered:
        if not _is_widget_backed(spec):
            continue
        # Reading D (Jeffrey 2026-05-17 round-robin section 2):
        # forceInput=True declarations are socket-only; never occupy
        # a widgets_values slot. Applied BEFORE companion expansion so
        # a hypothetical forceInput-flagged seed widget wouldn't drag
        # in a phantom companion slot either.
        if _spec_has_force_input(spec):
            continue
        slots.append(name)
        # Companion injection: INT seed widgets carry a hidden
        # control_after_generate COMBO. Type lookup is the same shape
        # used by `_is_widget_backed` -- spec is (type, opts) tuple or
        # bare type.
        type_def = (
            spec[0]
            if isinstance(spec, (list, tuple)) and len(spec) > 0
            else spec
        )
        if (
            isinstance(type_def, str)
            and type_def.upper() == "INT"
            and name in _COMPANION_INT_WIDGETS
        ):
            slots.append(f"{name}__control_after_generate")
    return slots


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
    # mode depending on which widgets were converted to sockets. The SAVED
    # layout also includes ComfyUI's auto-injected `control_after_generate`
    # companion next to any seed widget (see `_serialized_slot_names`).
    # We map widget_name -> slot index against the serialized layout so the
    # companion slot is correctly skipped over.
    serialized_slots = _serialized_slot_names(node_type, schemas)

    linked_names = {
        inp["name"]
        for inp in target_node.get("inputs", []) or []
        if inp.get("link") is not None and inp.get("name") in widget_names
    }

    wv = target_node.setdefault("widgets_values", [])
    target_idx = serialized_slots.index(widget_name)

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
    # Companions are NEVER socketed, so they are NEVER in linked_names; the
    # linked-count math is keyed to slot-occupying widgets only.
    #
    # Reading D fix (Jeffrey 2026-05-17, node 20): count only linked widgets
    # that survive the serialized_slots filter -- forceInput=True declarations
    # are linked-only by design but never occupy a slot, so they must NOT
    # reduce the expected length in either preserved or stripped mode.
    linked_widget_count = sum(
        1 for n in serialized_slots if n in linked_names
    )
    if len(wv) == len(serialized_slots):
        # "preserved" mode (companion-aware) -- use serialized index as-is
        slot = target_idx
    elif len(wv) == len(serialized_slots) - linked_widget_count:
        # "stripped" mode (companion-aware) -- subtract leading linked widgets
        # encountered before the target in the SERIALIZED layout. Companions
        # are skipped during this count because they are not in linked_names.
        leading_linked = sum(
            1 for n in serialized_slots[:target_idx] if n in linked_names
        )
        slot = target_idx - leading_linked
    else:
        # Ambiguous (trailing unset optionals, manual edits, unexpected
        # extra slots outside the seed-companion position). Bail with a
        # clear error rather than silently writing to the wrong slot.
        # The narrow loosening over the historical check accepts ONLY the
        # +1-per-seed companion model; anything else still rejects.
        raise ValueError(
            f"widgets_values length mismatch on node {node_id} ({node_type}): "
            f"len(wv)={len(wv)} vs "
            f"len(serialized_slots)={len(serialized_slots)} "
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

    Ported (with comments) from soak_operator's working converter, which
    carries the BUG-LOCAL-027 + BUG-LOCAL-029 fixes for socket-only
    inputs and "stripped" vs "preserved" widgets_values shapes.

    Sprint H §3.7 sibling fix (Jeffrey 2026-05-17): walk the SERIALIZED
    slot layout from `_serialized_slot_names` rather than the bare
    declared widget list. Companion slots auto-injected next to seed
    widgets are recognised, validated against the
    `control_after_generate` vocabulary, and skipped -- they MUST NOT
    map into a downstream declared input. Pre-fix the companion's
    `"fixed"` value bled into the next declared widget and produced
    `inputs["creative_writing_model"]="fixed"`,
    `inputs["seed"]=""`, `inputs["clip_length"]="fixed"`, etc.
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
            widget_names = _ordered_widget_names(ntype, schemas)
            serialized_slots = _serialized_slot_names(ntype, schemas)
            companion_slot_names = {
                n for n in serialized_slots
                if n.endswith("__control_after_generate")
            }

            wv = node.get("widgets_values", []) or []

            # Reading D fix (Jeffrey 2026-05-17, node 20): count only
            # linked widgets that survive the serialized_slots filter --
            # forceInput=True inputs never occupy a slot, so they must
            # NOT reduce the expected length.
            linked_widget_count = sum(
                1 for n in serialized_slots if n in linked_names
            )
            if len(wv) == len(serialized_slots):
                # Preserved mode: companion slots present + all
                # widget-backed declared inputs occupy their slots
                # (linked widgets keep placeholders).
                linked_keeps_slot = True
            elif len(wv) == len(serialized_slots) - linked_widget_count:
                # Stripped mode: linked widgets had their slot dropped
                # at save time. Companions remain.
                linked_keeps_slot = False
            elif len(wv) == 0:
                # Pure-socket node with no widget-backed inputs (or all
                # widgets converted to sockets and stripped). Nothing
                # to assign from widgets_values; links already supplied
                # via the link_map walk above.
                linked_keeps_slot = False
            else:
                # Fail loud per Jeffrey 2026-05-17 spec: unexpected
                # extra slot, missing non-linked slot, or other length
                # drift. Refuse to silently misalign the API prompt.
                raise ValueError(
                    f"widgets_values length mismatch on node {nid} "
                    f"({ntype}): len(wv)={len(wv)} vs "
                    f"len(serialized_slots)={len(serialized_slots)} "
                    f"(linked={linked_widget_count}). Refusing API prompt "
                    f"conversion."
                )

            wv_idx = 0
            for slot_name in serialized_slots:
                if slot_name in companion_slot_names:
                    # Companion slot. Validate value vocabulary so a
                    # misplaced/drifted companion does not silently
                    # propagate. Consume the slot but DO NOT write to
                    # inputs -- companions are not declared inputs.
                    if wv_idx < len(wv):
                        val = wv[wv_idx]
                        if val not in _COMPANION_VALUES:
                            raise ValueError(
                                f"node {nid} ({ntype}) companion slot at "
                                f"position {wv_idx} expected a "
                                f"control_after_generate value "
                                f"(one of "
                                f"{sorted(_COMPANION_VALUES)!r}); got "
                                f"{val!r}. Workflow widget drift; refusing "
                                f"API prompt conversion."
                            )
                        wv_idx += 1
                    continue

                if slot_name in linked_names:
                    # Linked widget. Value comes from the upstream node
                    # via the link, NOT from widgets_values. If the
                    # save preserved the placeholder slot, advance past
                    # it; otherwise stripped mode already collapsed
                    # the array length.
                    if linked_keeps_slot and wv_idx < len(wv):
                        wv_idx += 1
                    continue

                if wv_idx < len(wv):
                    inputs[slot_name] = wv[wv_idx]
                    wv_idx += 1

        prompt[nid] = {"class_type": ntype, "inputs": inputs}
    return prompt


# ---------------------------------------------------------------------------
# Submit + poll
# ---------------------------------------------------------------------------
def submit_prompt(api_prompt: dict, client_id: str | None = None) -> str:
    """POST the API prompt to /prompt and return the prompt_id.

    Sprint H §3.7 hardening (Jeffrey 2026-05-17): ComfyUI returns
    HTTP 200 even when the prompt has validation errors -- it accepts
    the POST, refuses to execute the invalid nodes, and the resulting
    history entry returns status_str='success' with zero outputs. That
    looks like a success to a naive caller. Inspect the response body
    BEFORE returning a prompt_id:
        * truthy `error` field -> raise.
        * non-empty `node_errors` dict -> raise.
          (Empty `node_errors: {}` is the success shape and accepted.)
    """
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

    error = body.get("error")
    if error:
        raise RuntimeError(
            f"submit_prompt: ComfyUI returned error: {error!r}"
        )
    node_errors = body.get("node_errors")
    if node_errors:
        # `node_errors` is a non-empty dict only when at least one node
        # failed validation. Surface at POST time so the caller
        # classifies as graph_widget without polling a zombie prompt_id.
        raise RuntimeError(
            f"submit_prompt: ComfyUI returned node_errors: {node_errors!r}"
        )

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
