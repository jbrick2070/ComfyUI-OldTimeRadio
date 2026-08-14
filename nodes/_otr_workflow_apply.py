"""The ONE applier -- GATE B S2 of the switchable-workflow architecture.

Spec: docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md
(sections 4, 5, 6); sequencing: docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md
section 0 (GATE B). This module is the structural kill for the headless
patch-list drift (the captions/credits/LTX-open bug): ONE applier consumed by
the generator, the headless scripts and CI -- nobody hand-codes widget patch
lists again.

Two strictly split operations (only the first lives here tonight):
  * ``apply_profile(workflow, profile)`` -- PURE semantic widget patching.
    Never stamps, never touches node-63 paths.
  * ``emit_snapshot`` -- generator-only (S3, not yet built): writes the node-63
    artifact path + stamp.

OFFLINE schemas: ``build_offline_schemas()`` adapts real ``INPUT_TYPES()`` into
the ``/object_info`` response shape, replicating ``_serialized_slot_names``
semantics (forceInput consumes no slot; INT seed/noise_seed companions). It is
tested against ``scripts/otr_api._serialized_slot_names`` for every OTR node
type (tests/test_workflow_apply.py), so CI runs with no live ComfyUI; the live
``/object_info`` cross-check stays in the soak lane.

PROMOTION NOTE: the patch-by-NAME machinery below is promoted from
``scripts/otr_api.py`` (the BUG-LOCAL-002 lineage). The script keeps its copy
for the live-HTTP path; the parity tests pin the two against each other until
the script is converted to import from here (tracked follow-up). The promoted
core is import-light (stdlib only -- scripts/otr_api.py needs ``requests``,
which a bare ComfyUI venv may lack).

Whitelist (headless creative knobs, enforced by ``patch_creative``):
``target_words``/``num_characters``/``act_count``/``request_seed`` + seed-policy
fields, prompt/title text fields, and the writer model slots (admissible via the
OpenRouter/Comfy admit-paths).
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Optional

from ._otr_shared.capability_profiles import (
    ProfileError,
    load_profile,
    load_widget_mapping,
)

log = logging.getLogger("OTR.workflow_apply")

__all__ = [
    "build_offline_schemas",
    "serialized_slot_names",
    "ordered_widget_names",
    "patch_widget_by_name",
    "workflow_to_api_prompt",
    "apply_profile",
    "patch_creative",
    "CREATIVE_WHITELIST",
]

# ---------------------------------------------------------------------------
# Offline /object_info adapter
# ---------------------------------------------------------------------------


def _node_class_mappings() -> dict:
    """The package NODE_CLASS_MAPPINGS, resolved for both ComfyUI-runtime and
    bare-pytest import shapes (mirrors _otr_workflow_validator.validate; the
    sys.path fallback mirrors tests/test_init_aliases_empty.py)."""
    try:
        from .. import NODE_CLASS_MAPPINGS as ncm  # type: ignore
        return ncm
    except (ImportError, ValueError):
        pass
    import importlib
    import os
    import sys
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_name = os.path.basename(repo_root)
    parent = os.path.dirname(repo_root)
    inserted = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        inserted = True
    try:
        pkg = importlib.import_module(pkg_name)
        return getattr(pkg, "NODE_CLASS_MAPPINGS", {})
    except Exception as e:  # noqa: BLE001 -- environment-dependent package import
        log.warning("_node_class_mappings: import of %r failed: %r", pkg_name, e)
        return {}
    finally:
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def build_offline_schemas(ncm: Optional[dict] = None) -> dict:
    """Adapt real ``INPUT_TYPES()`` into the ``/object_info`` response shape.

    Node types whose ``INPUT_TYPES()`` raises are SKIPPED (logged) rather than
    fatal -- the applier then fails loudly only if a MANAGED type is missing.
    """
    ncm = ncm if ncm is not None else _node_class_mappings()
    schemas: dict[str, Any] = {}
    for ntype, cls in ncm.items():
        if not hasattr(cls, "INPUT_TYPES"):
            continue
        try:
            it = cls.INPUT_TYPES() or {}
        except Exception as e:  # noqa: BLE001 -- environment-dependent INPUT_TYPES
            log.warning("build_offline_schemas: %s.INPUT_TYPES() raised %r; skipped", ntype, e)
            continue
        schemas[ntype] = {"input": it}
    return schemas


# ---------------------------------------------------------------------------
# Promoted patch-by-NAME machinery (scripts/otr_api.py lineage; parity-tested)
# ---------------------------------------------------------------------------
_WIDGET_PRIMITIVE_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN", "BOOL", "COMBO"}
_COMPANION_INT_WIDGETS = ("seed", "noise_seed")
_COMPANION_VALUES = frozenset({"fixed", "randomize", "increment", "decrement"})

_OPENROUTER_SLOT_WIDGETS = frozenset({
    "openrouter_slot_a_model", "openrouter_slot_b_model",
})
_COMFY_SLOT_WIDGETS = frozenset({
    "comfy_slot_a_model", "comfy_slot_b_model",
})

# OTR_VideoDirector / OTR_ImageDirector per-role engine COMBOs. As of 2026-06-17
# these dropdowns DISPLAY only GPU-validated engines (the tested-only gate), but
# the applier is a TRUSTED programmatic path -- a capability profile (e.g.
# 8gb_lite) legitimately selects a CPU-floor / untested engine for its tier, so
# the applier validates these widgets against the FULL registry, not the gated
# display list. Mirrors the openrouter/comfy admissibility escapes above.
_VIDEO_DIRECTOR_WIDGETS = frozenset({
    # Three first-class video slots (2026-07-03: legacy catch-all video slot retired).
    "announcer_video_model", "music_video_model", "character_video_model",
})
_IMAGE_DIRECTOR_WIDGETS = frozenset({
    "announcer_image_model", "music_image_model", "character_image_model",
})


def _is_widget_backed(spec: Any) -> bool:
    type_def = spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else spec
    if isinstance(type_def, (list, tuple)):  # dropdown choices
        return True
    return isinstance(type_def, str) and type_def in _WIDGET_PRIMITIVE_TYPES


def _spec_has_force_input(spec: Any) -> bool:
    if not isinstance(spec, (list, tuple)) or len(spec) < 2:
        return False
    opts = spec[1]
    return isinstance(opts, dict) and bool(opts.get("forceInput"))


def _spec_for(node_type: str, widget_name: str, schemas: dict) -> Any:
    if node_type not in schemas:
        raise KeyError(f"node_type {node_type!r} not in schemas")
    schema = schemas[node_type].get("input", {}) or {}
    for section in ("required", "optional"):
        d = schema.get(section, {}) or {}
        if widget_name in d:
            return d[widget_name]
    raise KeyError(f"widget {widget_name!r} not declared on node_type {node_type!r}")


def ordered_widget_names(node_type: str, schemas: dict) -> list:
    if node_type not in schemas:
        raise KeyError(f"node_type {node_type!r} not present in schemas")
    schema = schemas[node_type].get("input", {}) or {}
    required = schema.get("required", {}) or {}
    optional = schema.get("optional", {}) or {}
    ordered = list(required.items()) + list(optional.items())
    return [name for name, spec in ordered if _is_widget_backed(spec)]


def serialized_slot_names(node_type: str, schemas: dict) -> list:
    """The widget-name list mapping 1-to-1 to SAVED ``widgets_values`` --
    forceInput inputs dropped, INT seed/noise_seed companions injected.
    Promoted verbatim from ``scripts/otr_api._serialized_slot_names``."""
    if node_type not in schemas:
        raise KeyError(f"node_type {node_type!r} not present in schemas")
    schema = schemas[node_type].get("input", {}) or {}
    required = schema.get("required", {}) or {}
    optional = schema.get("optional", {}) or {}
    ordered = list(required.items()) + list(optional.items())

    slots: list = []
    for name, spec in ordered:
        if not _is_widget_backed(spec):
            continue
        if _spec_has_force_input(spec):
            continue
        slots.append(name)
        type_def = spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else spec
        if (isinstance(type_def, str) and type_def.upper() == "INT"
                and name in _COMPANION_INT_WIDGETS):
            slots.append(f"{name}__control_after_generate")
    return slots


def _is_openrouter_admissible(widget_name: str, value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if widget_name in _OPENROUTER_SLOT_WIDGETS:
        return True
    return value.startswith("openrouter:")


def _is_engine_director_admissible(widget_name: str, value: Any) -> bool:
    """A director engine COMBO accepts ANY registered engine (or the custom
    sentinel), even one HIDDEN from the tested-only display dropdown.

    The display gate (validated-only) is a UI concern; the applier validates a
    trusted profile selection against the full registry so the 8GB / floor tiers
    keep working. Registry imports are dep-free (cold-import safe)."""
    if not isinstance(value, str) or not value:
        return False
    if value == "+ Add Custom Model":
        return True
    if widget_name in _VIDEO_DIRECTOR_WIDGETS:
        from ._otr_video_engines import registry as _vreg
        from ._otr_shared.public_engines import resolve_engine_id
        # The video dropdown DISPLAYS public/aspect-labelled engines
        # ("wan_8gb (16:9)", "humo (portrait)"), and that label is what a fresh save
        # stores. Resolve the label / PUBLIC id / LEGACY id back to the concrete
        # internal engine id before the registry check (mirrors
        # OTR_VideoDirector._engine_id_from_pick). A bare internal value passes through.
        return resolve_engine_id(value) in _vreg.all_engine_names()
    if widget_name in _IMAGE_DIRECTOR_WIDGETS:
        from ._otr_image_engines import registry as _ireg
        return value in _ireg.all_engine_names()
    return False


#: The two LLM picker widgets whose menu options now carry a VRAM badge
#: (``'google/gemma-4-12b-it (11.9 GB)'``). Profiles, CLI flags and saved graphs
#: store the BARE repo id, so both the validator and the writer need the same
#: bare<->badged tolerance the video director already has for ``'wan_8gb (16:9)'``.
_LLM_MODEL_WIDGETS = frozenset({"creative_writing_model", "technical_model"})


def _is_llm_model_admissible(widget_name: str, value: Any, type_def: Any) -> bool:
    """A BARE repo id is admissible against a VRAM-badged choice list.

    Without this, adding the badge would reject every profile that names a
    creative model -- the badge is display text, not a new identity."""
    if widget_name not in _LLM_MODEL_WIDGETS or not isinstance(value, str):
        return False
    from ._otr_model_catalog import _strip_label_suffix
    bare = _strip_label_suffix(value)
    return any(_strip_label_suffix(str(c)) == bare for c in (type_def or ()))


def _is_comfy_admissible(widget_name: str, value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if widget_name in _COMFY_SLOT_WIDGETS:
        return True
    return value.startswith("comfy:")


def _validate_widget_value(node_type: str, widget_name: str, spec: Any, value: Any) -> None:
    if value is None:
        return
    type_def = spec[0] if isinstance(spec, (list, tuple)) and len(spec) > 0 else spec

    if isinstance(type_def, (list, tuple)):
        if value not in type_def:
            if _is_openrouter_admissible(widget_name, value):
                return
            if _is_comfy_admissible(widget_name, value):
                return
            if _is_engine_director_admissible(widget_name, value):
                return
            if _is_llm_model_admissible(widget_name, value, type_def):
                return
            raise ValueError(
                f"widget {widget_name!r} on node_type {node_type!r} is a COMBO "
                f"with choices {list(type_def)!r}; got {value!r} which is not in "
                f"the choice list."
            )
        return
    if not isinstance(type_def, str):
        return
    t = type_def.upper()
    if t in ("BOOLEAN", "BOOL"):
        if not isinstance(value, bool):
            raise ValueError(f"widget {widget_name!r} on {node_type!r} is {t}; got {value!r}")
        return
    if t == "INT":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"widget {widget_name!r} on {node_type!r} is INT; got {value!r}")
        return
    if t == "FLOAT":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"widget {widget_name!r} on {node_type!r} is FLOAT; got {value!r}")
        return
    if t == "STRING":
        if not isinstance(value, str):
            raise ValueError(f"widget {widget_name!r} on {node_type!r} is STRING; got {value!r}")
        return
    # COMBO with no inline choices / unknown primitive: skip.


def _node_by_type(workflow: dict, node_type: str) -> dict:
    """Find the SINGLE node of ``node_type``; raise if absent or ambiguous
    (the mapping's unique-match assertion, enforced at apply time too)."""
    found = [n for n in workflow.get("nodes", []) if n.get("type") == node_type]
    if len(found) != 1:
        raise ProfileError(
            f"apply_profile: node type {node_type!r} occurs {len(found)}x in the "
            f"workflow (the widget mapping requires exactly 1)"
        )
    return found[0]


def patch_widget_by_name(workflow: dict, node_id: int, widget_name: str,
                         value: Any, schemas: dict) -> None:
    """Set a widget value by declared NAME (never position). Promoted from
    ``scripts/otr_api.patch_widget_by_name`` -- same semantics, same refusals."""
    target_node = None
    for node in workflow.get("nodes", []):
        if node.get("id") == node_id:
            target_node = node
            break
    if target_node is None:
        raise KeyError(f"node id={node_id} not in workflow")
    _patch_node_widget(target_node, widget_name, value, schemas)


def _patch_node_widget(target_node: dict, widget_name: str, value: Any,
                       schemas: dict) -> None:
    node_type = target_node.get("type")
    node_id = target_node.get("id")
    widget_names = ordered_widget_names(node_type, schemas)
    if widget_name not in widget_names:
        raise KeyError(
            f"widget {widget_name!r} not in widget order for {node_type!r}. "
            f"Known widgets: {widget_names!r}"
        )
    spec = _spec_for(node_type, widget_name, schemas)
    _validate_widget_value(node_type, widget_name, spec, value)
    if value is None:
        return

    serialized_slots = serialized_slot_names(node_type, schemas)
    linked_names = {
        inp["name"]
        for inp in target_node.get("inputs", []) or []
        if inp.get("link") is not None and inp.get("name") in widget_names
    }
    if widget_name in linked_names:
        raise ValueError(
            f"widget {widget_name!r} on node {node_id} has been converted to an "
            f"input socket; cannot patch via widgets_values."
        )
    wv = target_node.setdefault("widgets_values", [])
    target_idx = serialized_slots.index(widget_name)
    linked_widget_count = sum(1 for n in serialized_slots if n in linked_names)
    if len(wv) == len(serialized_slots):
        slot = target_idx
    elif len(wv) == len(serialized_slots) - linked_widget_count:
        leading_linked = sum(1 for n in serialized_slots[:target_idx] if n in linked_names)
        slot = target_idx - leading_linked
    else:
        raise ValueError(
            f"widgets_values length mismatch on node {node_id} ({node_type}): "
            f"len(wv)={len(wv)} vs len(serialized_slots)={len(serialized_slots)} "
            f"(linked={linked_widget_count}). Refusing to patch by name."
        )
    while len(wv) <= slot:
        wv.append(None)
    wv[slot] = value


def workflow_to_api_prompt(workflow: dict, schemas: dict) -> dict:
    """UI workflow JSON -> API prompt dict. Promoted verbatim from
    ``scripts/otr_api.workflow_to_api_prompt`` (companion-aware, fail-loud);
    the S2 identity/parity CI gates compare THESE dicts."""
    link_map: dict[int, list] = {}
    for lnk in workflow.get("links", []) or []:
        link_id, src_node, src_slot = lnk[0], lnk[1], lnk[2]
        link_map[link_id] = [str(src_node), src_slot]

    prompt: dict[str, Any] = {}
    for node in workflow.get("nodes", []):
        nid = str(node["id"])
        ntype = node["type"]
        inputs: dict[str, Any] = {}
        linked_names: set = set()
        for inp in node.get("inputs", []) or []:
            if inp.get("link") is not None:
                inputs[inp["name"]] = link_map.get(inp["link"])
                linked_names.add(inp["name"])

        if ntype in schemas:
            serialized_slots = serialized_slot_names(ntype, schemas)
            companion_slot_names = {
                n for n in serialized_slots if n.endswith("__control_after_generate")
            }
            wv = node.get("widgets_values", []) or []
            linked_widget_count = sum(1 for n in serialized_slots if n in linked_names)
            if len(wv) == len(serialized_slots):
                linked_keeps_slot = True
            elif len(wv) == len(serialized_slots) - linked_widget_count:
                linked_keeps_slot = False
            elif len(wv) == 0:
                linked_keeps_slot = False
            else:
                raise ValueError(
                    f"widgets_values length mismatch on node {nid} ({ntype}): "
                    f"len(wv)={len(wv)} vs len(serialized_slots)="
                    f"{len(serialized_slots)} (linked={linked_widget_count}). "
                    f"Refusing API prompt conversion."
                )
            wv_idx = 0
            for slot_name in serialized_slots:
                if slot_name in companion_slot_names:
                    if wv_idx < len(wv):
                        val = wv[wv_idx]
                        if val not in _COMPANION_VALUES:
                            raise ValueError(
                                f"node {nid} ({ntype}) companion slot at position "
                                f"{wv_idx} expected a control_after_generate value; "
                                f"got {val!r}. Workflow widget drift; refusing."
                            )
                        wv_idx += 1
                    continue
                if slot_name in linked_names:
                    if linked_keeps_slot and wv_idx < len(wv):
                        wv_idx += 1
                    continue
                if wv_idx < len(wv):
                    inputs[slot_name] = wv[wv_idx]
                    wv_idx += 1
        prompt[nid] = {"class_type": ntype, "inputs": inputs}
    return prompt


# ---------------------------------------------------------------------------
# semantic master_hash -- the S5 drift tripwire (generator + validator share
# THIS normalizer; docs/2026-07-09-platform-portability-final.md section 1)
# ---------------------------------------------------------------------------
def semantic_master_hash(workflow: dict, mapping: Optional[dict] = None,
                         schemas: Optional[dict] = None) -> str:
    """sha256 over node types + links (sorted) + the profile-MANAGED widget
    VALUES only.

    Excluded by construction: pos/size/UI keys (never read), the node-63
    stamp widgets (only ``managed`` targets count -- the stamps live under
    ``emit_only``), and every creative widget (title/premise/seeds stay
    free to edit without tripping the wire). A byte hash would flag every
    cosmetic save; THIS hash flags exactly the changes a platform variant
    must not drift on: graph shape, wiring, and profile-managed values.
    """
    import hashlib

    mapping = mapping if mapping is not None else load_widget_mapping()
    schemas = schemas if schemas is not None else build_offline_schemas()

    managed_targets = set()
    for entry in mapping["managed"].values():
        for node_type, widget in entry["targets"]:
            managed_targets.add((node_type, widget))

    types_part = sorted(
        (int(n["id"]), str(n["type"])) for n in workflow.get("nodes", []))
    links_part = sorted(
        tuple(l) for l in (workflow.get("links") or []) if isinstance(l, list))

    managed_part = []
    for node in workflow.get("nodes", []):
        ntype = node["type"]
        if ntype not in schemas:
            continue
        slots = serialized_slot_names(ntype, schemas)
        linked = {i.get("name") for i in (node.get("inputs") or [])
                  if i.get("link") is not None}
        wv = node.get("widgets_values", []) or []
        if len(wv) == len(slots):
            keeps = True
        elif len(wv) == len(slots) - sum(1 for s in slots if s in linked):
            keeps = False
        else:
            continue  # malformed vectors are the validator's job, not the hash's
        idx = 0
        for slot in slots:
            if slot in linked and not keeps:
                continue
            if idx >= len(wv):
                break
            if (ntype, slot) in managed_targets:
                managed_part.append((ntype, slot, wv[idx]))
            idx += 1
    managed_part.sort()

    payload = json.dumps(
        {"types": types_part, "links": links_part, "managed": managed_part},
        ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# apply_profile -- the ONE applier
# ---------------------------------------------------------------------------
def _flatten_profile_values(profile: dict) -> dict:
    flat: dict[str, Any] = {}
    for section in ("role_overrides", "slot_overrides", "features"):
        for k, v in profile.get(section, {}).items():
            flat[f"{section}.{k}"] = v
    sp = profile.get("seed_policy", {})
    if "request_seed" in sp:
        flat["seed_policy.request_seed"] = sp["request_seed"]
    if "seed_mode" in sp:
        flat["seed_policy.seed_mode"] = sp["seed_mode"]
    # S2 platform-portability (2026-07-10): widget-mapped v2 sections whose
    # targets EXIST today. llm model keys -> the writer's 8 existing model
    # widgets; render.* -> the fps/canvas/composite/frame-budget widget set.
    # The llm RUNTIME fields (device/attn/quant/gguf/...) and the
    # video/image/audio policy fields flatten in S5 WITH their widgets --
    # flattening them earlier would fail apply (no mapping target), and a
    # mapping target must never name a widget that does not exist yet.
    llm = profile.get("llm", {})
    for k in ("creative_model", "technical_model",
              "openrouter_slot_a_model", "openrouter_slot_b_model",
              "comfy_slot_a_model", "comfy_slot_b_model",
              "google_api_slot_a_model", "google_api_slot_b_model",
              # S5: the six runtime-policy widgets (writer slots 28-33).
              "device", "attn_impl", "quant_policy", "vram_ceiling_gb",
              "gguf_n_ctx", "gguf_quant"):
        if k in llm:
            flat[f"llm.{k}"] = llm[k]
    rend = profile.get("render", {})
    for k in ("fps", "canvas_w", "canvas_h", "composite_res",
              "composite_w", "composite_h", "frame_budget", "beats"):
        if k in rend:
            flat[f"render.{k}"] = rend[k]
    # S5: director + castlock policy widgets.
    vid = profile.get("video", {})
    if "device_policy" in vid:
        flat["video.device_policy"] = vid["device_policy"]
    if "dtype_policy" in vid:
        flat["video.dtype_policy"] = vid["dtype_policy"]
    # 2026-07-24 (WAN 8GB launch contract): OPTIONAL per-tier render-length
    # ceiling. Absent = unpinned, so no other tier's applied widget vector moves.
    if "max_render_frames" in vid:
        flat["video.max_render_frames"] = vid["max_render_frames"]
    img = profile.get("image", {})
    if "dtype_policy" in img:
        flat["image.dtype_policy"] = img["dtype_policy"]
    aud = profile.get("audio", {})
    if "voice_device" in aud:
        flat["audio.voice_device"] = aud["voice_device"]
    # Queue item 8 (2026-08-08): upscale_stage section, only-if-present.
    # If absent, nothing flattens -> applier never patches the widgets ->
    # canonical's "off"/"cpu" defaults survive (registry-supplied default).
    us = profile.get("upscale_stage")
    if us is not None:
        if "engine" in us:
            flat["upscale_stage.engine"] = us["engine"]
        if "device" in us:
            flat["upscale_stage.device"] = us["device"]
    return flat


def _director_option_value(node_type: str, widget: str, value: Any) -> Any:
    """For an OTR_VideoDirector role widget selecting one of the four PUBLIC-aliased
    tier engines, write the EXACT live menu option (the public label) -- so a
    generated variant stores what the UI would save and round-trips (e.g. profile
    ``wan_ti2v`` -> ``'wan_8gb (16:9)'``). Video-tiers (2026-07-20), boundary 5.

    SCOPED to the four renamed tier engines (``_INTERNAL_TO_PUBLIC``): their id
    CHANGED, so a stored bare internal id would not match any visible menu option.
    Every OTHER engine keeps its existing stored form (its bare id -- still admissible
    via the resolver), so non-tier variants do NOT churn (additive only). The
    ``OTR_VideoRenderBatch.engine`` widget and every non-director target keep the raw
    INTERNAL value. ADD_CUSTOM / empty / unregistered pass through raw."""
    if node_type != "OTR_VideoDirector" or widget not in _VIDEO_DIRECTOR_WIDGETS:
        return value
    if not isinstance(value, str) or not value or value == "+ Add Custom Model":
        return value
    from ._otr_shared.public_engines import resolve_engine_id, _INTERNAL_TO_PUBLIC
    from ._otr_video_engines import registry as _vreg
    internal = resolve_engine_id(value)
    if internal not in _INTERNAL_TO_PUBLIC or not _vreg.is_registered(internal):
        return value
    from .otr_video_director import exact_menu_option_for
    return exact_menu_option_for(internal)


def _llm_option_value(node_type: str, widget: str, value: Any,
                      schemas: Optional[dict]) -> Any:
    """Write the EXACT live menu option for an LLM picker, badge included.

    The sibling of :func:`_director_option_value`. A profile stores the bare
    repo id (``'google/gemma-4-12b-it'``); the picker now shows it with its VRAM
    cost (``'google/gemma-4-12b-it (11.9 GB)'``). Storing the bare form would
    leave a saved graph carrying a value ComfyUI's own COMBO validation does not
    recognize, so resolve it to the live choice.

    Falls through UNCHANGED whenever the choice list is unavailable or nothing
    matches -- a remote handle (``openrouter:slot-a``), an uncurated id with no
    estimate, or an offline-schema gap must all keep working."""
    if widget not in _LLM_MODEL_WIDGETS or not isinstance(value, str) or not value:
        return value
    inp = ((schemas or {}).get(node_type) or {}).get("input") or {}
    spec = None
    for section in ("required", "optional"):
        spec = (inp.get(section) or {}).get(widget)
        if spec is not None:
            break
    type_def = spec[0] if isinstance(spec, (list, tuple)) and spec else None
    if not isinstance(type_def, (list, tuple)):
        return value
    if value in type_def:
        return value
    from ._otr_model_catalog import _strip_label_suffix
    bare = _strip_label_suffix(value)
    for choice in type_def:
        if _strip_label_suffix(str(choice)) == bare:
            return choice
    return value


def apply_profile(workflow: dict, profile, mapping: Optional[dict] = None,
                  schemas: Optional[dict] = None) -> dict:
    """PURE semantic widget patching: return a DEEP COPY of ``workflow`` with
    every profile-managed widget set to the profile's value.

    * ``profile`` may be a profile id (loaded + shape-validated) or a dict
      (shape-validated by the loader path the caller used).
    * Never stamps; never touches node-63's ``workflow_json_path`` (that is
      ``emit_snapshot``'s job, S3).
    * Nodes are found by TYPE (unique-match enforced); raw ids are banned.
    * Every widget write goes through the promoted patch-by-NAME machinery,
      so position drift / linked sockets / wrong value types all refuse loudly.

    One LOUD log line records the resolved profile + every override applied
    (the queue-start availability line's apply-side sibling).
    """
    if isinstance(profile, str):
        profile = load_profile(profile)
    mapping = mapping if mapping is not None else load_widget_mapping()
    schemas = schemas if schemas is not None else build_offline_schemas()

    out = copy.deepcopy(workflow)
    applied: list = []
    flat = _flatten_profile_values(profile)
    managed = mapping["managed"]
    for dotted in sorted(flat):
        entry = managed.get(dotted)
        if entry is None:
            raise ProfileError(
                f"apply_profile: profile key {dotted!r} has no widget-mapping "
                f"entry (typo'd key, or the mapping needs updating)"
            )
        for node_type, widget in entry["targets"]:
            node = _node_by_type(out, node_type)
            value = _director_option_value(node_type, widget, flat[dotted])
            value = _llm_option_value(node_type, widget, value, schemas)
            _patch_node_widget(node, widget, value, schemas)
            applied.append(f"{dotted} -> {node_type}.{widget} = {value!r}")
    log.info(
        "apply_profile: profile=%s (%s) applied %d widget writes:\n  %s",
        profile.get("id"), profile.get("display_name"), len(applied),
        "\n  ".join(applied) or "(none)",
    )
    return out


# ---------------------------------------------------------------------------
# Headless creative whitelist (S2 spec section 8; enforced, stateless)
# ---------------------------------------------------------------------------
CREATIVE_WHITELIST = frozenset({
    "num_characters", "act_count", "request_seed",
    "seed_mode",
    "episode_title", "custom_premise", "style_custom",
    "openrouter_slot_a_model", "openrouter_slot_b_model",
    "comfy_slot_a_model", "comfy_slot_b_model",
    "creative_writing_model", "technical_model",
    # creativity is a pure CREATIVE dial (temperature/top_p preset on node 1), not an
    # engine/feature widget -- safe to set directly from a soak (apply_profile never
    # manages it). Lets OTR_COMBO_CREATIVITY=maximum chaos reach the writer.
    "creativity",
    # refine_target_grade (v1 story-refine loop dropdown Off/C+/B/B+/A) is a pure
    # CREATIVE dial on node 1, not an engine/feature widget -- apply_profile never
    # manages it, so a headless soak may set the grade bar directly.
    "refine_target_grade",
    # source_bank (Stage 2C multi-modal story schema) is the story-path
    # selector on node 1 -- a pure creative/content dial; apply_profile never
    # manages it. Headless drivers may set it; a non-runnable pick still
    # fails LOUD at run() (require_runnable_bank, no fallback).
    "source_bank",
    # visual_style (Stage 3C) is the visual-style selector on node 1 -- a
    # pure creative dial (prompt-tail deltas only); apply_profile never
    # manages it. Unknown id fails LOUD at run() (resolve_visual_style).
    "visual_style",
    # source_ref (Source Banks v2) is an optional source reference string for
    # bank fetchers. Blank is inert; nonblank semantics belong to the selected
    # bank and must fail loud there.
    "source_ref",
    # Google BYO API concrete model pickers bind the virtual google_api:slot-a/b
    # LLM rows. They are content/model-selection widgets on the writer, not
    # managed audio/image/video engine widgets.
    "google_api_slot_a_model", "google_api_slot_b_model",
    # lemmy_cameo is a pure CREATIVE dial on node 1 -- it decides whether the
    # recurring cameo character joins the cast ("roll (~11% chance)" /
    # "always include" / "never include"). apply_profile never manages it.
    #
    # WHITELISTED SO A QUALIFICATION RUN CAN FORCE HIM DETERMINISTICALLY. An
    # acceptance render that has to wait for an 11% roll is not an acceptance
    # render, and the alternative -- hand-patching the graph -- is the ad-hoc
    # JSON edit this seam exists to prevent.
    #
    # It cannot be used to smuggle him into a source-faithful adaptation:
    # `_source_bank_excludes_lemmy` overrides `force_lemmy` unconditionally, so
    # "always include" on a shakespeare/public_domain bank still yields
    # `lemmy_policy = "source_fidelity_exclusion"`. This entry adds a creative
    # dial, never a route, engine, reference or other managed/runtime field.
    "lemmy_cameo",
})


def patch_creative(workflow: dict, node_id: int, widget_name: str, value: Any,
                   schemas: Optional[dict] = None) -> None:
    """The ONLY sanctioned direct-widget helper for headless scripts: patches a
    CREATIVE (whitelisted) widget by name. Engine/feature widgets must go
    through ``apply_profile`` -- this helper refuses them."""
    if widget_name not in CREATIVE_WHITELIST:
        raise ProfileError(
            f"patch_creative: widget {widget_name!r} is not on the creative "
            f"whitelist {sorted(CREATIVE_WHITELIST)!r}; managed widgets are "
            f"patched ONLY via apply_profile"
        )
    schemas = schemas if schemas is not None else build_offline_schemas()
    patch_widget_by_name(workflow, node_id, widget_name, value, schemas)
