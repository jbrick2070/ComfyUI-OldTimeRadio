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
    #: The host rejected (401/403) or lacked the API key -- a verdict, unlike
    #: an outage: every paid call with that key fails too, so it refuses.
    key_rejected: bool = False


#: A list ping is a probe: keep it short. The Google client's own default is
#: 120 s with retries -- far too long to hold every queue in an outage.
CATALOG_TIMEOUT_S = 15


def _key_rejected(exc) -> bool:
    """Whether a Google catalog fetch failed on the KEY rather than the network.

    Read from the client's structured evidence, never the message text: a
    missing key (``GoogleAPIKeyMissingError``), ``failure_kind == "auth"`` or an
    HTTP 401/403 -- and HTTP 400 with reason ``API_KEY_INVALID``, which is how
    Google actually answers a malformed or revoked key (measured; the client
    files that one as ``request_shape``)."""
    if exc is None:
        return False
    if type(exc).__name__ == "GoogleAPIKeyMissingError":
        return True
    if getattr(exc, "failure_kind", None) == "auth":
        return True
    status = getattr(exc, "http_status", None)
    if status in (401, 403):
        return True
    body = getattr(exc, "response_json", None)
    err = body.get("error") if isinstance(body, dict) else None
    details = err.get("details") if isinstance(err, dict) else None
    return any(isinstance(d, dict) and d.get("reason") == "API_KEY_INVALID"
               for d in (details if isinstance(details, list) else ()))


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
    """T2 list ping. ``ids is None`` means the catalog was not obtained.

    An EMPTY list is "not obtained" too: no live provider lists zero models,
    so an empty answer is a degraded endpoint, not a verdict that every slug
    is gone. ``key_rejected`` marks a fetch that failed on the API key."""
    auth = str(authority or "").strip().lower()
    if auth == "openrouter":
        getter = get_json or _live_openrouter_get_json
        try:
            payload = getter(OPENROUTER_MODELS_URL)
        except Exception as exc:  # noqa: BLE001 -- classified as transport
            # Never key_rejected: this listing sends NO key, so a 401/403 here
            # is a proxy or the host, not the user's key.
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
        if not ids:
            return CatalogResult(None, "openrouter catalog listed no models",
                                 OPENROUTER_HOST)
        return CatalogResult(ids, "", OPENROUTER_HOST)
    if auth == "google":
        getter = get_json or _live_google_get_json
        try:
            from .google_slug_verifier import fetch_catalog
        except ImportError:  # pragma: no cover
            from google_slug_verifier import fetch_catalog
        last = {}

        def watched(path):
            try:
                return getter(path)
            except Exception as exc:  # noqa: BLE001 -- kept for classification
                last["exc"] = exc
                raise

        fetched = fetch_catalog(watched)
        if not fetched.complete:
            return CatalogResult(
                None, fetched.error or "google models.list incomplete",
                GOOGLE_HOST, _key_rejected(last.get("exc")))
        if not fetched.ids:
            return CatalogResult(None, "google models.list listed no models",
                                 GOOGLE_HOST)
        return CatalogResult(fetched.ids, "", GOOGLE_HOST)
    raise ValueError("unknown catalog authority %r" % authority)


def _live_openrouter_get_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "OTR-cloud-slug-preflight"})
    with urlopen(req, timeout=CATALOG_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _live_google_get_json(path: str) -> dict:
    """One plain GET with the short probe timeout. Deliberately NOT the
    client's ``get_json``: that wraps every call in ``_quota_retry``, which
    sleeps on a 429 (Retry-After, up to 60 s, several times) -- minutes of a
    queue held by a probe. ``fetch_catalog`` already retries a retryable
    failure a bounded number of times, without sleeping."""
    from .._otr_google_api import client
    body = client._get_bytes(path, api_key=client.resolve_api_key(),
                             timeout_s=CATALOG_TIMEOUT_S, accept="application/json")
    parsed = client._safe_json(body)
    if not isinstance(parsed, dict):
        raise client.GoogleAPIRequestShapeError(
            "Google models.list did not return a JSON object.")
    return parsed


def catalog_outage_hint(catalog_host: str, paid_host: str) -> str:
    """The note on a T2 catalog outage, which is a WARNING.

    An unreachable, failing or empty catalog is a failed probe, not a finding
    about the pick, and no probe may fail a workflow (operator rule,
    2026-10-14). When the paid call goes to the same host, a real outage makes
    that call fail loudly on its own; a blip that has cleared by then costs
    nothing. Two cases still refuse because they are verdicts: a catalog that
    ANSWERS without the slug, and a host that rejects the API key
    (``CatalogResult.key_rejected``) -- every paid call with it fails too."""
    cat = (urlparse(catalog_host).hostname or catalog_host or "").lower()
    paid = (urlparse(paid_host).hostname or paid_host or "").lower()
    if cat and cat == paid:
        return ("slug unverified; the paid call goes to this host and fails "
                "loudly if it is really down")
    return "paid host differs; check is skipped"


def _catalog_unavailable(result, paid_host, engine, node_key, input_name, value):
    """The finding for a catalog that was not obtained: a refusal when the key
    was rejected, otherwise a warning (see :func:`catalog_outage_hint`)."""
    if result.key_rejected:
        return Finding(
            engine, node_key, input_name, value,
            "the %s catalog rejected the API key (%s)"
            % (result.host, result.error or "auth"),
            "set a valid key for %s; every paid call with this one fails"
            % result.host,
            "refuse")
    return Finding(
        engine, node_key, input_name, value,
        "T2 catalog unavailable (%s)" % (result.error or "transport"),
        "catalog host=%s paid host=%s; %s"
        % (result.host, paid_host, catalog_outage_hint(result.host, paid_host)),
        "warn")


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
            findings.append(_catalog_unavailable(
                result, paid_host, engine_id, engine_id, "catalog", ""))
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
                "update ComfyUI and the pack so the partner pin table matches "
                "(in a git clone, re-pin partner_nodes.yaml with "
                "scripts/otr_pin_partner_nodes.py)",
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


def _retired_reason(engine_id: str) -> str:
    """The named "retired" text for a tombstoned id, or "" if it is not one.

    A retired lane is unregistered too, so without this a saved graph naming
    one is refused here as "not registered" -- the generic message the
    tombstone set exists to replace -- before the director can say why."""
    try:
        from .public_engines import (
            RETIRED_ENGINE_IDS, RetiredEngineError, resolve_engine_id)
    except ImportError:  # pragma: no cover -- flat test imports
        from public_engines import (  # type: ignore
            RETIRED_ENGINE_IDS, RetiredEngineError, resolve_engine_id)
    resolved = resolve_engine_id(engine_id)
    return str(RetiredEngineError(resolved)) if resolved in RETIRED_ENGINE_IDS else ""


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
    scoped = list(walker(prompt, unique_id) or [])
    return scoped + _castlocks_feeding(prompt, scoped)


def _castlocks_feeding(prompt, scoped) -> list:
    """Every ungated ``OTR_CastLock`` whose output a scoped node consumes.

    The shipped graphs feed CastLock the frozen script and leave its
    ``gate_in`` unlinked, so the gate walk never reached it, and its voice
    engines went unchecked and unpriced: ElevenLabs on five cloud graphs,
    Google TTS on ``otr_google_still`` (plan item 2, 2026-10-14). The voice
    nodes that consume its cast ARE behind this validator, so the cast belongs
    to this run. A CastLock whose ``gate_in`` is linked is left to the gate walk
    -- already in scope, or another validator's, which must not check or charge
    for it."""
    held = {id(node) for node in scoped}
    found = []
    for node in scoped:
        for value in (node.get("inputs") or {}).values():
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                continue
            src = prompt.get(str(value[0])) if isinstance(prompt, dict) else None
            if not isinstance(src, dict) or src.get("class_type") != _CASTLOCK:
                continue
            gate = (src.get("inputs") or {}).get("gate_in")
            if isinstance(gate, (list, tuple)) or id(src) in held:
                continue
            held.add(id(src))
            found.append(src)
    return found


def castlock_voice_picks(inputs) -> tuple:
    """``(engine_ids, wired_slots)`` one CastLock's voice widgets name.

    Nothing to check when the cast rolls: with ``roll_audio_engines`` on and
    ``cast_voice_policy`` ``auto_registry``, CastLock draws one LOCAL engine at
    run time (``_otr_lane_rolls``), so a saved paid pick never casts and must not
    be checked or charged. A voice slot wired to another node has no value
    before execution; it is named in ``wired_slots`` instead of raising."""
    inputs = inputs or {}
    roll = inputs.get("roll_audio_engines", False)
    policy = inputs.get("cast_voice_policy", "preserve_ledger")
    if not isinstance(roll, (list, tuple)) and policy == "auto_registry":
        if isinstance(roll, str):
            rolling = roll.strip().lower() in ("true", "1", "yes", "on")
        else:
            rolling = bool(roll)
        if rolling:
            return [], []
    try:
        from ..cast_lock import _VOICE_ENGINE_ALIASES as aliases
    except ImportError:  # pragma: no cover -- no Comfy tree
        aliases = {}
    picks, wired = [], []
    for slot in _CAST_SLOTS:
        value = inputs.get(slot, "")
        if isinstance(value, (list, tuple)):
            wired.append(slot)
        elif isinstance(value, str) and value.strip():
            # The id CastLock will actually cast with: a saved alias such as
            # "elevenlabs" becomes "cloud_elevenlabs" there, so check that.
            picks.append(aliases.get(value.strip(), value.strip()))
    return picks, wired


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
        if ctype == _CASTLOCK and not replay:
            # A replay passes the frozen cast through with no TTS at all, so
            # its voice widgets are neither checked nor priced.
            picks, wired = castlock_voice_picks(inputs)
            engine_ids.extend(picks)
            for slot in wired:
                findings.append(Finding(
                    _CASTLOCK, _CASTLOCK, slot, "",
                    "voice slot is wired to another node; its engine is "
                    "unknown until the run",
                    "unchecked here; a dead cloud pick fails when the voice "
                    "engine calls it", "warn"))

    seen = set()
    for engine_id in engine_ids:
        eid = str(engine_id or "").strip()
        if not eid or eid in seen:
            continue
        seen.add(eid)
        eng = lookup(eid)
        if eng is None:
            retired = _retired_reason(eid)
            if retired:
                findings.append(Finding(
                    eid, eid, "engine", eid, retired,
                    "pick another engine from the dropdown", "refuse"))
            elif _looks_paid(eid):
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
                findings.append(_catalog_unavailable(
                    result, paid_host, handle, handle, widget, slug))
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
    """Wired entry. Raises ValueError listing every refuse finding.

    Computed afresh for every queue. It used to memoize the findings under the
    validator's node id, which is the same id (63) on every queue of a shipped
    graph -- so a refusal outlived the repaired pick, and a cached empty result
    let a later bad cloud pick through. The per-call catalog memo inside
    collect_findings is all the caching this needs."""
    findings = collect_findings(prompt, unique_id, **kwargs)
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
    "catalog_outage_hint",
    "castlock_voice_picks",
    "check_engine",
    "collect_findings",
    "default_partner_selectors",
    "director_engine_picks",
    "ensure_prompt_cloud_slugs",
    "format_refusal",
    "options_from_class",
    "schema_options",
]
