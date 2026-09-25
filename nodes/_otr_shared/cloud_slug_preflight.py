"""Queue-time cloud slug check -- refuse before the first credit moves.

Wired from ``OTR_WorkflowValidator`` on every shipping graph (the node is
already in all ``workflows/**/*.json``). Reads the LIVE queued prompt, not
saved JSON. Schema reads and catalog list pings only -- never generate.

Cold-import-clean: stdlib only at module import. Partner class import, catalog
GETs, and registry lookups are lazy and injectable so tests run with no
Comfy tree and no socket.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from enum import Enum
from typing import Callable, NamedTuple, Optional
from urllib.parse import urlparse
from urllib.request import urlopen  # bare name clears the registry $http2 literal; still seen by the network-sites guard

log = logging.getLogger("OTR.cloud.slug")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_HOST = "openrouter.ai"
GOOGLE_HOST = "generativelanguage.googleapis.com"
COMFY_PAID_HOST = "api.comfy.org"

_WRITER = "OTR_LedgerScriptWriter"
_DIRECTOR = "OTR_VideoDirector"
_CASTLOCK = "OTR_CastLock"
_MUSIC = "OTR_StableAudioTheme"
_CAST_SLOTS = ("char_voice_engine", "announcer_voice_engine")

_COMFY_A = "comfy:slot-a"
_COMFY_B = "comfy:slot-b"
_OR_A = "openrouter:slot-a"
_OR_B = "openrouter:slot-b"
_GOOG_A = "google_api:slot-a"
_GOOG_B = "google_api:slot-b"

_CACHE: dict[str, list] = {}


class Finding(NamedTuple):
    engine: str
    node_key: str
    input_name: str
    value: str
    reason: str
    fix_hint: str
    severity: str  # "refuse" | "warn"


class CatalogResult(NamedTuple):
    ids: Optional[frozenset]
    error: str
    host: str


def default_partner_selectors(node_key: str) -> dict:
    """V3 rows expose the resolved ``model``; product rows expose no combo."""
    key = str(node_key or "").strip()
    if not key:
        return {}
    try:
        from .cloud_model_ids import V3_MODEL_IDS, resolve_model_id
    except ImportError:  # pragma: no cover -- flat import
        from cloud_model_ids import V3_MODEL_IDS, resolve_model_id
    if key in V3_MODEL_IDS:
        return {key: {"model": (resolve_model_id(key),)}}
    return {key: {}}


def _norm(value) -> str:
    return str(value).strip()


def _option_values(options) -> tuple:
    if options is None:
        return ()
    if isinstance(options, type) and issubclass(options, Enum):
        return tuple(_norm(v.value) for v in options)
    if isinstance(options, (list, tuple)):
        out = []
        for item in options:
            if hasattr(item, "key"):
                out.append(_norm(item.key))
            elif isinstance(item, Enum):
                out.append(_norm(item.value))
            else:
                out.append(_norm(item))
        return tuple(out)
    return ()


def options_from_class(cls) -> dict:
    """Combo + DynamicCombo option keys from ``define_schema`` or INPUT_TYPES."""
    if cls is None:
        return {}
    if callable(getattr(cls, "define_schema", None)):
        try:
            schema = cls.define_schema()
        except Exception:
            schema = None
        if schema is not None:
            extracted = _options_from_v3_schema(schema)
            if extracted:
                return extracted
    types_fn = getattr(cls, "INPUT_TYPES", None)
    if callable(types_fn):
        try:
            types = cls.INPUT_TYPES()
        except Exception:
            types = None
        if isinstance(types, dict):
            return _options_from_input_types(types)
    return {}


def _options_from_v3_schema(schema) -> dict:
    inputs = getattr(schema, "inputs", None)
    if inputs is None and isinstance(schema, dict):
        inputs = schema.get("inputs")
    if not inputs:
        return {}
    out = {}
    for inp in inputs:
        name = getattr(inp, "id", None) or getattr(inp, "name", None)
        if not name and isinstance(inp, dict):
            name = inp.get("id") or inp.get("name")
        if not name:
            continue
        options = getattr(inp, "options", None)
        if options is None and isinstance(inp, dict):
            options = inp.get("options")
        vals = _option_values(options)
        if vals:
            out[str(name)] = vals
    return out


def _options_from_input_types(types) -> dict:
    out = {}
    for group in ("required", "optional"):
        block = types.get(group) or {}
        if not isinstance(block, dict):
            continue
        for name, spec in block.items():
            if not isinstance(spec, (list, tuple)) or not spec:
                continue
            first = spec[0]
            if isinstance(first, (list, tuple)):
                vals = tuple(_norm(x) for x in first)
                if vals:
                    out[str(name)] = vals
    return out


def schema_options(node_key, *, resolve_class=None, rows=None) -> dict:
    """T1: live Combo / DynamicCombo options for a pinned partner row."""
    try:
        from . import cloud_media_invoke as cmi
    except ImportError:  # pragma: no cover
        import cloud_media_invoke as cmi  # type: ignore
    table = rows if rows is not None else cmi.partner_rows()
    row = table.get(node_key) if isinstance(table, dict) else None
    if not isinstance(row, dict):
        raise RuntimeError("unknown partner node_key %r" % node_key)
    resolver = resolve_class if resolve_class is not None else cmi._resolve_node_class
    cls = resolver(row)
    return options_from_class(cls)


def catalog_ids(authority, *, get_json=None) -> CatalogResult:
    """T2 list ping. ``ids is None`` means the catalog was not obtained."""
    auth = str(authority or "").strip().lower()
    if auth == "openrouter":
        getter = get_json or _live_openrouter_get_json
        try:
            payload = getter(OPENROUTER_MODELS_URL)
        except Exception as exc:  # noqa: BLE001 -- classified as transport
            return CatalogResult(None, "%s: %s" % (type(exc).__name__, exc),
                                 OPENROUTER_HOST)
        if not isinstance(payload, dict):
            return CatalogResult(None, "openrouter catalog is not an object",
                                 OPENROUTER_HOST)
        data = payload.get("data")
        if not isinstance(data, list):
            return CatalogResult(None, "openrouter catalog has no data list",
                                 OPENROUTER_HOST)
        ids = frozenset(_norm(row.get("id")) for row in data
                        if isinstance(row, dict) and row.get("id"))
        return CatalogResult(ids, "", OPENROUTER_HOST)
    if auth == "google":
        getter = get_json or _live_google_get_json
        try:
            from .google_slug_verifier import fetch_catalog
        except ImportError:  # pragma: no cover
            from google_slug_verifier import fetch_catalog
        fetched = fetch_catalog(getter)
        if not fetched.complete:
            return CatalogResult(
                None, fetched.error or "google models.list incomplete",
                GOOGLE_HOST)
        return CatalogResult(fetched.ids, "", GOOGLE_HOST)
    raise ValueError("unknown catalog authority %r" % authority)


def _live_openrouter_get_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "OTR-cloud-slug-preflight"})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _live_google_get_json(path: str) -> dict:
    from .._otr_google_api.client import get_json
    return get_json(path)


def transport_severity(catalog_host: str, paid_host: str) -> str:
    """Refuse when the catalog host is the host the paid call will hit."""
    cat = (urlparse(catalog_host).hostname or catalog_host or "").lower()
    paid = (urlparse(paid_host).hostname or paid_host or "").lower()
    return "refuse" if cat and cat == paid else "warn"


def _fix_hint(node_key: str, input_name: str, live) -> str:
    env_name = ""
    try:
        from .cloud_model_ids import V3_MODEL_IDS
    except ImportError:  # pragma: no cover
        from cloud_model_ids import V3_MODEL_IDS
    spec = V3_MODEL_IDS.get(node_key) or {}
    if input_name == "model" and spec.get("env"):
        env_name = spec["env"]
    extras = {
        ("cloud_elevenlabs_tts", "model"): "OTR_ELEVENLABS_MODEL_ID",
        ("cloud_luma_photon_flash", "model"): "OTR_CLOUD_LUMA_PHOTON_MODEL",
        ("cloud_ideogram_v4", "resolution"): "OTR_CLOUD_IDEOGRAM_RESOLUTION",
        ("cloud_vidu_q2_i2v", "movement_amplitude"): "OTR_CLOUD_VIDU_Q2_MOVEMENT",
    }
    env_name = env_name or extras.get((node_key, input_name), "")
    shown = ", ".join(list(live)[:16]) if live else "(none listed)"
    if env_name:
        return "set %s to one of: %s" % (env_name, shown)
    return "pick a live %s value: %s" % (input_name, shown)


def check_engine(engine, selectors, *, schema_options_fn, catalog_fn) -> list:
    """Compare one adapter's candidate sets to T1 schema or T2 catalog."""
    findings = []
    catalog = getattr(engine, "cloud_catalog", None)
    engine_id = str(getattr(engine, "name", "") or "")
    if catalog:
        result = catalog_fn(catalog)
        paid_host = GOOGLE_HOST if str(catalog) == "google" else str(catalog)
        if result.ids is None:
            sev = transport_severity(result.host, paid_host)
            findings.append(Finding(
                engine_id, engine_id, "catalog", "",
                "T2 catalog unavailable (%s)" % (result.error or "transport"),
                "catalog URL host=%s status logged; %s"
                % (result.host,
                   "run cannot verify a spend on this host"
                   if sev == "refuse" else
                   "paid host differs; check is skipped"),
                sev))
            return findings
        for node_key, fields in (selectors or {}).items():
            for input_name, candidates in (fields or {}).items():
                for raw in candidates:
                    # Same strip as writers: Google catalog ids are bare,
                    # adapters sometimes still carry a models/ prefix.
                    value = _normalize_checked_slug(catalog, raw)
                    if value and value not in result.ids:
                        findings.append(Finding(
                            engine_id, str(node_key), str(input_name), value,
                            "slug %r is not in the %s catalog"
                            % (value, catalog),
                            _fix_hint(str(node_key), str(input_name),
                                      sorted(result.ids)[:16]),
                            "refuse"))
        return findings

    for node_key, fields in (selectors or {}).items():
        try:
            opts = schema_options_fn(node_key)
        except Exception as exc:  # noqa: BLE001 -- import/schema is a finding
            findings.append(Finding(
                engine_id, str(node_key), "", "",
                "pinned partner class could not be imported: %s" % exc,
                "re-pin partner_nodes.yaml or install the Comfy partner node",
                "refuse"))
            continue
        for input_name, candidates in (fields or {}).items():
            live = opts.get(input_name)
            if live is None:
                continue
            live_norm = {_norm(x) for x in live}
            for raw in candidates:
                value = _norm(raw)
                if value and value not in live_norm:
                    findings.append(Finding(
                        engine_id, str(node_key), str(input_name), value,
                        "value %r is not a live option on %s.%s"
                        % (value, node_key, input_name),
                        _fix_hint(str(node_key), str(input_name), live),
                        "refuse"))
    return findings


def _widget_str(inputs, name, default=""):
    value = (inputs or {}).get(name, default)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            "cloud slug preflight cannot inspect linked/dynamic %s before "
            "execution" % name)
    return value.strip()


def _looks_paid(engine_id: str) -> bool:
    eid = str(engine_id or "")
    if eid.startswith("cloud_") or eid.startswith("google_"):
        return True
    return eid in {"sonilo", "ideo"}


def _default_resolve_engine(engine_id: str):
    eid = str(engine_id or "").strip()
    if not eid:
        return None
    try:
        from .._otr_video_engines import registry as vreg
        if vreg.is_registered(eid):
            return vreg.get_engine(eid)
    except Exception:  # noqa: BLE001
        pass
    try:
        from .._otr_image_engines import registry as ireg
        if ireg.is_registered(eid):
            return ireg.get_engine(eid)
    except Exception:  # noqa: BLE001
        pass
    try:
        from .._otr_audio_engines import registry as areg
        if areg.is_registered(eid):
            return areg.get_engine(eid)
    except Exception:  # noqa: BLE001
        pass
    return None


def _is_placeholder_slug(value: str) -> bool:
    text = _norm(value)
    if not text:
        return True
    return text.startswith("(") and text.endswith(")")


def _writer_handles(inputs) -> list:
    """(handle, widget_name, authority, paid_host) for selected remote slots."""
    creative = _widget_str(inputs, "creative_writing_model")
    technical = _widget_str(inputs, "technical_model")
    picked = {creative, technical}
    rows = []
    mapping = (
        (_COMFY_A, "comfy_slot_a_model", "openrouter", COMFY_PAID_HOST),
        (_COMFY_B, "comfy_slot_b_model", "openrouter", COMFY_PAID_HOST),
        (_OR_A, "openrouter_slot_a_model", "openrouter", OPENROUTER_HOST),
        (_OR_B, "openrouter_slot_b_model", "openrouter", OPENROUTER_HOST),
        (_GOOG_A, "google_api_slot_a_model", "google", GOOGLE_HOST),
        (_GOOG_B, "google_api_slot_b_model", "google", GOOGLE_HOST),
    )
    for handle, widget, authority, paid_host in mapping:
        if handle in picked:
            rows.append((handle, widget, authority, paid_host))
    return rows


def _slot_widget(inputs, name) -> str:
    value = (inputs or {}).get(name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            "cloud slug preflight cannot inspect linked/dynamic %s before "
            "execution" % name)
    return value


def _restore_slot_map(target, saved) -> None:
    target["A"] = saved.get("A")
    target["B"] = saved.get("B")


def _normalize_checked_slug(authority, slug) -> str:
    value = _norm(slug)
    if str(authority) != "google":
        return value
    try:
        from .google_slug_verifier import normalize_model_name
    except ImportError:  # pragma: no cover
        from google_slug_verifier import normalize_model_name
    return normalize_model_name(value) or value


def _writer_config_errors() -> tuple:
    """Exception classes a backend resolver raises for 'nothing will post'.

    Each backend spells its config error differently (``RuntimeError``
    subclasses, not ``ValueError``); a linked widget raises ``ValueError``
    from ``_slot_widget``. All of them are a refuse finding, never a crash.
    """
    errors = [ValueError]
    try:
        from .._otr_openrouter_backend import OpenRouterConfigError
        errors.append(OpenRouterConfigError)
    except ImportError:  # pragma: no cover -- no Comfy tree
        pass
    try:
        from .._otr_comfy_backend import ComfyCreditsConfigError
        errors.append(ComfyCreditsConfigError)
    except ImportError:  # pragma: no cover
        pass
    try:
        from .._otr_google_api.models import GoogleAPISlotBindingError
        errors.append(GoogleAPISlotBindingError)
    except ImportError:  # pragma: no cover
        pass
    return tuple(errors)


def _posted_writer_slug(handle, inputs):
    """The slug the selected backend will POST, or a refuse reason.

    Binds this writer's live slot widgets, then uses that backend's own
    resolver, then puts the process-global bindings back. Checking the
    widget text alone would miss an env / recommended fallback that still
    spends (OpenRouter, Comfy), or keep a stale binding from the previous
    episode. Google has no fallback: an unbound slot raises before any
    request, so the finding is a refuse with no spend behind it.
    """
    try:
        if handle in (_OR_A, _OR_B):
            from .. import _otr_openrouter_backend as orb
            saved = dict(orb._slot_bindings)
            try:
                orb.set_slot_bindings(
                    slot_a=_slot_widget(inputs, "openrouter_slot_a_model"),
                    slot_b=_slot_widget(inputs, "openrouter_slot_b_model"),
                )
                return orb.resolve_slug(handle), ""
            finally:
                _restore_slot_map(orb._slot_bindings, saved)
        if handle in (_COMFY_A, _COMFY_B):
            from .. import _otr_comfy_backend as occ
            saved = dict(occ._slot_bindings)
            try:
                occ.set_slot_bindings(
                    slot_a=_slot_widget(inputs, "comfy_slot_a_model"),
                    slot_b=_slot_widget(inputs, "comfy_slot_b_model"),
                )
                return occ.resolve_slug(handle), ""
            finally:
                _restore_slot_map(occ._slot_bindings, saved)
        if handle in (_GOOG_A, _GOOG_B):
            from .._otr_google_api import models as gai
            saved = dict(gai._slot_bindings)
            try:
                gai.set_slot_bindings(
                    slot_a=_slot_widget(inputs, "google_api_slot_a_model"),
                    slot_b=_slot_widget(inputs, "google_api_slot_b_model"),
                )
                return gai.resolve_model_for_slot(handle), ""
            finally:
                _restore_slot_map(gai._slot_bindings, saved)
    except _writer_config_errors() as exc:
        return "", str(exc)
    return "", "unknown writer handle %r" % handle


def director_engine_picks(inputs) -> tuple:
    """(video_engine_ids, image_engine_ids) one ``OTR_VideoDirector`` will run.

    Video picks go through the same custom-model / route-freeze resolution the
    render path uses. Image roles whose paired video lane is proven no-still
    are dropped, so a wallet is never charged for a still that never renders.
    Shared with the balance preflight so both gates read the same roles.
    """
    try:
        from .._otr_visual_assets import (
            _IMAGE_SLOTS, _VIDEO_SLOTS, _custom_models, _proven_no_still,
            _resolve_slot, _default_role_video_slots, _image_slot_for,
        )
        from .._otr_shared.public_engines import resolve_engine_id
        from .._otr_shared import route_freeze
    except ImportError:
        _resolve_slot = None

    videos_out = []
    images_out = []
    if _resolve_slot is None:
        for slot in ("announcer_video_model", "music_video_model",
                     "character_video_model"):
            picked = _widget_str(inputs, slot, "")
            if picked and not picked.startswith("+ Add Custom"):
                videos_out.append(picked)
        for slot in ("announcer_image_model", "music_image_model",
                     "character_image_model"):
            picked = _widget_str(inputs, slot, "")
            if picked and not picked.startswith("+ Add Custom"):
                images_out.append(picked)
        return videos_out, images_out

    custom = _custom_models(inputs)
    videos = {}
    for slot in _VIDEO_SLOTS:
        picked = _resolve_slot(inputs, slot, custom, "video")
        videos[slot] = resolve_engine_id(picked)
    effective = (route_freeze.freeze_role_engines(videos)
                 if route_freeze is not None else videos)
    videos_out.extend(resolve_engine_id(v) for v in effective.values() if v)
    pairing = _default_role_video_slots()
    no_still = {}
    for role, vslot in dict(pairing).items():
        lane = effective.get(role)
        lane = resolve_engine_id(lane) if lane else ""
        if _proven_no_still(lane, None):
            no_still[_image_slot_for(vslot)] = lane
    for slot in _IMAGE_SLOTS:
        if slot in no_still:
            continue
        picked = _resolve_slot(inputs, slot, custom, "image")
        if picked:
            images_out.append(picked)
    return videos_out, images_out


def _collect_scoped(prompt, unique_id, *, walk_fn=None):
    walker = walk_fn
    if walker is None:
        from .._otr_visual_assets import walk_gate_consumers
        walker = walk_gate_consumers
    return list(walker(prompt, unique_id) or [])


def collect_findings(
    prompt, unique_id, *,
    resolve_engine=None,
    schema_options_fn=None,
    catalog_fn=None,
    walk_fn=None,
    skip_writer=False,
) -> list:
    """All findings for one queued prompt. Does not raise."""
    findings = []
    if unique_id is None or not str(unique_id).strip():
        return findings
    if not isinstance(prompt, dict) or not prompt:
        findings.append(Finding(
            "", "", "", "",
            "cloud slug preflight requires the live queued prompt",
            "queue through OTR_WorkflowValidator",
            "refuse"))
        return findings

    scoped = _collect_scoped(prompt, unique_id, walk_fn=walk_fn)
    writers = [n for n in scoped if n.get("class_type") == _WRITER]
    replay_flags = []
    for node in writers:
        replay_from = _widget_str(node.get("inputs") or {}, "replay_from", "")
        replay_flags.append(bool(replay_from))
    replay = bool(replay_flags) and all(replay_flags)
    if replay_flags and any(replay_flags) and not all(replay_flags):
        findings.append(Finding(
            "", "", "replay_from", "",
            "mixed replay/live writers behind one validator",
            "use one replay_from state for every writer on this gate",
            "refuse"))
        return findings

    lookup = resolve_engine or _default_resolve_engine
    schema_fn = schema_options_fn or schema_options
    raw_catalog = catalog_fn or catalog_ids
    _catalog_memo = {}

    def cat_fn(authority):
        key = str(authority or "")
        if key not in _catalog_memo:
            _catalog_memo[key] = raw_catalog(authority)
        return _catalog_memo[key]

    engine_ids = []
    directors = [n for n in scoped if n.get("class_type") == _DIRECTOR]
    for node in directors:
        videos, images = director_engine_picks(node.get("inputs") or {})
        engine_ids.extend(videos)
        engine_ids.extend(images)

    for node in scoped:
        ctype = node.get("class_type")
        inputs = node.get("inputs") or {}
        if ctype == _MUSIC:
            picked = _widget_str(inputs, "engine", "")
            if picked and not picked.startswith("+ Add Custom"):
                engine_ids.append(picked)
        if ctype == _CASTLOCK:
            for slot in _CAST_SLOTS:
                picked = _widget_str(inputs, slot, "")
                if picked:
                    engine_ids.append(picked)

    seen = set()
    for engine_id in engine_ids:
        eid = str(engine_id or "").strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        eng = lookup(eid)
        if eng is None:
            if _looks_paid(eid):
                findings.append(Finding(
                    eid, eid, "engine", eid,
                    "cloud engine %r is not registered" % eid,
                    "pick a registered cloud engine from the dropdown",
                    "refuse"))
            continue
        selector_fn = getattr(eng, "cloud_selectors", None)
        if not callable(selector_fn):
            if _looks_paid(eid):
                findings.append(Finding(
                    eid, str(getattr(eng, "node_key", "") or eid),
                    "cloud_selectors", "",
                    "paid engine %r has no cloud_selectors()" % eid,
                    "implement cloud_selectors on the adapter",
                    "refuse"))
            continue
        try:
            selectors = selector_fn()
        except Exception as exc:  # noqa: BLE001 -- env/resolve is a finding
            findings.append(Finding(
                eid, str(getattr(eng, "node_key", "") or eid), "", "",
                "cloud_selectors() failed: %s" % exc,
                "fix the env override or pick another engine",
                "refuse"))
            continue
        findings.extend(check_engine(
            eng, selectors,
            schema_options_fn=schema_fn,
            catalog_fn=cat_fn))

    if skip_writer or replay:
        return findings

    for node in writers:
        inputs = node.get("inputs") or {}
        for handle, widget, authority, paid_host in _writer_handles(inputs):
            # The checked value is the one the backend will POST -- a
            # placeholder widget may still spend via an env / recommended
            # fallback (OpenRouter, Comfy), so the widget text is not enough.
            slug, reason = _posted_writer_slug(handle, inputs)
            if _is_placeholder_slug(slug):
                findings.append(Finding(
                    handle, handle, widget, slug,
                    "%s is selected but no slug will post (%s)"
                    % (handle, reason or "%s is unbound" % widget),
                    "bind a concrete slug on %s" % widget,
                    "refuse"))
                continue
            result = cat_fn(authority)
            if result.ids is None:
                sev = transport_severity(result.host, paid_host)
                findings.append(Finding(
                    handle, handle, widget, slug,
                    "T2 catalog unavailable (%s)" % (result.error or "transport"),
                    "catalog host=%s paid host=%s" % (result.host, paid_host),
                    sev))
                continue
            if _normalize_checked_slug(authority, slug) not in result.ids:
                findings.append(Finding(
                    handle, handle, widget, slug,
                    "writer slug %r is not in the %s catalog"
                    % (slug, authority),
                    "set %s to a live catalog id" % widget,
                    "refuse"))
    return findings


def format_refusal(findings) -> str:
    refuses = [f for f in findings if f.severity == "refuse"]
    parts = []
    for item in refuses:
        loc = ".".join(p for p in (item.engine, item.node_key, item.input_name)
                       if p)
        extra = (" value=%r" % item.value) if item.value else ""
        parts.append("%s%s -- %s (%s)" % (loc, extra, item.reason, item.fix_hint))
    return (
        "OTR_WorkflowValidator: cloud slug preflight -- %d issue(s) -- %s. "
        "A listed model is not active; fix the pick or env before the run "
        "spends a credit."
        % (len(refuses), "; ".join(parts))
    )


def ensure_prompt_cloud_slugs(prompt, unique_id, **kwargs) -> list:
    """Wired entry. Raises ValueError listing every refuse finding."""
    cache_key = str(unique_id or "").strip()
    if cache_key and cache_key in _CACHE and not kwargs:
        findings = _CACHE[cache_key]
    else:
        findings = collect_findings(prompt, unique_id, **kwargs)
        if cache_key and not kwargs:
            _CACHE[cache_key] = findings
    for item in findings:
        if item.severity == "warn":
            log.warning("[OTR.cloud.slug] %s %s %s -- %s",
                        item.engine, item.input_name, item.value, item.reason)
    refuses = [f for f in findings if f.severity == "refuse"]
    if refuses:
        raise ValueError(format_refusal(findings))
    return findings


__all__ = [
    "CatalogResult",
    "Finding",
    "catalog_ids",
    "check_engine",
    "collect_findings",
    "default_partner_selectors",
    "director_engine_picks",
    "ensure_prompt_cloud_slugs",
    "format_refusal",
    "options_from_class",
    "schema_options",
    "transport_severity",
]
