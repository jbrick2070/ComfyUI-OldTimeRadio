"""nodes/_otr_lane_rolls.py -- the three model randomizers (operator, 2026-09-28).

Three Yes/No switches, off in every shipped workflow, local engines only, each
marked "Suggested for 16 GB+" in the app form:

* ``roll_video_lanes`` (OTR_VideoDirector): one local video lane for the whole
  episode, written into all three video roles.
* ``roll_still_models`` (OTR_VideoDirector): one local image model for the
  whole episode, written into all three still roles.
* ``roll_audio_engines`` (OTR_CastLock): one local voice engine for the whole
  cast, announcer included, and one local music engine for the theme
  (OTR_StableAudioTheme).

The video and still switches each have a checklist (``video_roll_pool``,
``still_roll_pool``; operator, 2026-09-28) drawn as clickable boxes at the foot
of the app form: tick models to narrow the draw, or tick none to draw from every
local model that can run here, exactly like the language, bank and style pools.

WHY SWITCHES, NOT ROLL ROWS. The story-bank, visual-style and language rolls
(``_otr_rolls``) are a roll row inside one dropdown. One lane for the whole
episode spans three dropdowns, and three roll rows could disagree, so each of
these is one switch that rolls every dropdown it owns.

WHERE EACH ROLL HAPPENS.

* Video, stills and music roll first thing at the queue-time gate
  (``_otr_workflow_validator._queue_time_readiness_gates``). The cloud
  preflights, the node-pack and boot checks and the weight downloads all read
  the queued prompt, and each must see the rolled engine, not the saved pick.
  So the roll writes the chosen id into that prompt's own node inputs. ComfyUI
  hands each node its inputs from the same prompt object
  (``DynamicPrompt.get_node`` returns ``original_prompt[id]``, and hidden PROMPT
  is ``get_original_prompt()``; read on core 0.37.4), and every consumer sits
  downstream of the validator, whose IS_CHANGED is NaN on a live run, so each
  one executes with the rolled value and never from a cache. Core's newer
  ``override_node`` would leave the submitted prompt untouched, but older
  cores do not have it.
* The voice rolls inside OTR_CastLock (``roll_voice_engine``), because only
  there is the episode's language known -- it may itself have been rolled by
  the writer -- and a non-English episode admits only the engines its language
  row lists. No gate download covers a voice engine, so nothing is lost.
* The language rolls in the writer, after the bank. The gate settles it first
  (``settle_prompt_language``): a language picked by hand that this run's
  voices cannot speak refuses there, and a rolled one loses those from its
  pool, written back into the prompt like the model picks above.

WHAT CAN BE ROLLED: a local engine this machine can run now, or whose weights
the gate fetches at the start of the run. Every check is an existing authority
asked early; this module decides nothing about fit on its own.

* Local: the adapter module is not ``eng_cloud*`` or ``eng_google*``, the
  naming every cloud adapter in the three registries follows.
* This machine: the shared capability rules
  (``capability_profiles.availability``) against core's own device and vendor,
  so a CUDA box started with ``--cpu`` reads as cpu.
* An engine that runs in its own install: that install is on disk
  (``install_gaps``, the check its ``load`` refuses on).
* Video: serves all three roles, is not redirected by the route freeze, has its
  node pack, matches this server's boot, and -- when it renders from a still --
  some local image model can make one here.
* Video and stills the gate does not fetch: the adapter's own ``assert_usable``
  passes now. Music: the engine's own check passes.
* Voices: the engine speaks both roles, so one engine voices everyone
  (IndexTTS2 voices characters only); a cloning engine has its reference clips
  on disk; a non-English episode's row admits it, and the engine can speak it
  on this machine (its ``language_gap``). With ``preserve_ledger`` the
  writer's voices are kept, so the voice is not rolled.

NO VRAM FILTER, on purpose: an OOM is recorded, never pre-empted, and the note
beside each switch says which cards they are meant for.

RECEIPTS. The gate leaves each receipt beside its node's inputs, under
``otr_lane_rolls``: inside the queued prompt, so it belongs to this run and no
other. CastLock, the first ledger rewrite after the gate, copies them into the
ledger meta beside the writer's ``bank_roll``, ``style_roll`` and
``language_roll``, and stamps its own voice receipt. A node whose switch is on
but which carries no gate receipt refuses to run, because nothing rolled it.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Callable, Mapping

try:
    from . import _otr_rolls as _ROLLS
except ImportError:  # pragma: no cover -- flat test imports
    import _otr_rolls as _ROLLS  # type: ignore

log = logging.getLogger(__name__)

VIDEO_SWITCH = "roll_video_lanes"
STILL_SWITCH = "roll_still_models"
AUDIO_SWITCH = "roll_audio_engines"

#: The checklists that narrow the video and still rolls (operator, 2026-09-28:
#: "make a video and still randomizer using the clickboxes"). Typed lists
#: underneath, like the language, bank and style pools; none ticked means every
#: local model that can run here.
VIDEO_POOL = "video_roll_pool"
STILL_POOL = "still_roll_pool"

VIDEO_SEED_ENV = "OTR_VIDEO_LANE_SEED"
STILL_SEED_ENV = "OTR_STILL_MODEL_SEED"
VOICE_SEED_ENV = "OTR_VOICE_ENGINE_SEED"
MUSIC_SEED_ENV = "OTR_MUSIC_ENGINE_SEED"

#: Where a node's gate receipts ride in the queued prompt, beside ``inputs``.
RECEIPT_KEY = "otr_lane_rolls"

#: surface -> (the switch that rolls it, its seed variable, its ledger key,
#: what it picks, in words)
SURFACES = {
    "video_lane": (VIDEO_SWITCH, VIDEO_SEED_ENV, "video_lane_roll", "video lane"),
    "still_model": (STILL_SWITCH, STILL_SEED_ENV, "still_model_roll", "still model"),
    "music_engine": (AUDIO_SWITCH, MUSIC_SEED_ENV, "music_engine_roll", "music engine"),
    "voice_engine": (AUDIO_SWITCH, VOICE_SEED_ENV, "voice_engine_roll", "voice engine"),
}
#: The surfaces rolled at the gate. The voice rolls in CastLock.
GATE_SURFACES = ("video_lane", "still_model", "music_engine")
#: Every ledger meta key a roll may stamp; CastLock persists exactly these.
LEDGER_KEYS = tuple(spec[2] for spec in SURFACES.values())

#: Every cloud and Google adapter module in the registries is named so.
CLOUD_MODULE_PREFIXES = ("eng_cloud", "eng_google")

_DIRECTOR = "OTR_VideoDirector"
_CAST_LOCK = "OTR_CastLock"
_THEME = "OTR_StableAudioTheme"
_WRITER = "OTR_LedgerScriptWriter"
_VIDEO_SLOTS = ("announcer_video_model", "music_video_model", "character_video_model")
_IMAGE_SLOTS = ("announcer_image_model", "music_image_model", "character_image_model")
_VOICE_ROLES = ("char_voice", "announcer_voice")
_AUTO_REGISTRY = "auto_registry"

#: The shared capability reason codes, as a person reads them.
_FIT_WORDS = {
    "requires_cuda": "cannot run on this machine's device",
    "impractical_on_cpu": "is too slow without a GPU",
    "requires_vendor": "needs a different GPU maker",
    "missing_toolchain": "needs a toolchain this machine lacks",
    "sidecars_disabled": "needs a separate install",
}


class LaneRollError(_ROLLS.RollError):
    """A switch is on and could not roll. Always loud, never a fallback."""


# ---------------------------------------------------------------------------
# What can be rolled on this machine
# ---------------------------------------------------------------------------

def _instance(engine):
    return engine() if isinstance(engine, type) else engine


def is_cloud_engine(engine) -> bool:
    """True when ``engine``'s adapter calls a remote service."""
    cls = engine if isinstance(engine, type) else type(engine)
    module = str(getattr(cls, "__module__", "") or "").rsplit(".", 1)[-1]
    return module.startswith(CLOUD_MODULE_PREFIXES)


def live_host() -> dict:
    """This machine, in the shape ``capability_profiles.availability`` reads.

    The device is core's own answer, through the resolver CastLock uses for
    ``default``. No engine declares a toolchain today, so none is claimed;
    separate installs are allowed and each one is checked on disk below.
    """
    try:
        from ._otr_shared import device_options as dev
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import device_options as dev  # type: ignore
    backend = str(dev.resolve_device(dev.DEFAULT_DEVICE_OPTION)).split(":", 1)[0]
    vendor = dev.vendor()
    return {"device_backend": backend,
            "gpu_vendor": vendor if vendor in ("nvidia", "amd", "apple") else "none",
            "toolchains": [], "allow_sidecars": True}


def _short(exc) -> str:
    text = (str(exc) or type(exc).__name__).strip().splitlines()[0]
    return text if len(text) <= 160 else text[:157] + "..."


def _first_reason(*checks):
    """The first check that leaves an engine out, in words, or None. A check
    that raises leaves it out with its own message."""
    for check in checks:
        try:
            reason = check()
        except Exception as exc:  # noqa: BLE001 -- a refusal is a reason
            reason = _short(exc)
        if reason:
            return reason
    return None


def _host_reason(capabilities, name, host):
    try:
        from ._otr_shared import capability_profiles as cp
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import capability_profiles as cp  # type: ignore
    declaration = capabilities.get(name)
    if not declaration:
        return "has no capability row"
    code = cp.availability(host, {name: declaration})[name]
    return None if code == cp.REASON_OK else _FIT_WORDS.get(code, code)


def _install_reason(capabilities, name, engine):
    """For an engine that runs in its own install: is that install on disk?"""
    if not (capabilities.get(name) or {}).get("requires_sidecar"):
        return None
    gaps = getattr(_instance(engine), "install_gaps", None)
    if not callable(gaps):
        return "runs in a separate install this check cannot see"
    missing = list(gaps())
    return ("its separate install has no %s" % missing[0][0]) if missing else None


def _adapter_reason(engine):
    """The adapter's own readiness check, asked now. Only for engines whose
    weights the gate does not fetch: for the rest that check would refuse a
    fresh machine for weights about to be downloaded."""
    check = getattr(_instance(engine), "assert_usable", None)
    if not callable(check):
        return None
    try:
        check(host_caps={}, profile={})
    except TypeError:
        check({}, {})
    return None


def _reference_clips_reason(name, engine):
    """A cloning engine needs its reference clips; every one the voice bank
    lists for it must be on disk, through the one resolver the workers use."""
    if not getattr(_instance(engine), "requires_voice_ref", False):
        return None
    try:
        from ._otr_voice_bank import load_voice_bank
        from ._otr_audio_engines.base import resolve_voice_ref_path
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_voice_bank import load_voice_bank  # type: ignore
        from _otr_audio_engines.base import resolve_voice_ref_path  # type: ignore
    bank, _sha = load_voice_bank()
    refs = [str(e.ref_path or "") for e in bank if e.engine == name]
    if not refs:
        return "has no reference clips in the voice bank"
    missing = [r for r in refs
               if not r or not os.path.isfile(resolve_voice_ref_path(r) or "")]
    if missing:
        return "is missing %d of its %d reference clips" % (len(missing), len(refs))
    return None


def video_pool(host=None) -> "tuple[tuple[str, ...], dict]":
    """``(eligible, left_out)``: the local video lanes a roll may draw here."""
    try:
        from . import _otr_video_engines  # noqa: F401 -- registers built-ins
        from ._otr_video_engines import registry as vreg
        from ._otr_video_engines import wrapper_bridge as bridge
        from ._otr_shared import boot_contracts, role_slots, route_freeze
        from . import _otr_visual_assets as assets
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_video_engines  # type: ignore  # noqa: F401
        from _otr_video_engines import registry as vreg  # type: ignore
        from _otr_video_engines import wrapper_bridge as bridge  # type: ignore
        from _otr_shared import boot_contracts, role_slots, route_freeze  # type: ignore
        import _otr_visual_assets as assets  # type: ignore
    host = live_host() if host is None else host
    roles = tuple(role_slots.ROLE_TO_VIDEO_SLOT)
    slots = tuple(role_slots.ROLE_TO_VIDEO_SLOT.values())
    # The freeze WITHOUT the operator's force map: a forced role renders its
    # forced engine whatever is picked, exactly as for a saved pick.
    snapshot = dict(route_freeze.routing_env_snapshot(), force_engine_map="")
    # Asked once. Off a server (the CPU suite, a script) there is no node
    # registry and no boot to read; the render-time checks still stand.
    packs_known = bool(bridge.node_class_mappings())
    boot = boot_contracts.running_server_boot_state()
    boot_known = bool(boot.get("available"))
    # A lane that renders from a still is only drawable where some local
    # image model can make one (none can on a CPU-only machine).
    stills_possible = bool(still_pool(host)[0])

    def still_reason(name):
        if stills_possible or assets._proven_no_still(name, None):
            return None
        return "needs a still, and no local image model runs on this machine"

    def serves_every_role(name):
        for role in roles:
            vreg.assert_usable(name, role)

    def not_redirected(name):
        effective = route_freeze.freeze_role_engines(
            {slot: name for slot in slots}, snapshot=snapshot)
        if any(effective.get(role) != name for role in roles):
            return "is redirected to another lane when picked for every role"
        return None

    def reason_for(name, engine):
        return _first_reason(
            lambda: _host_reason(vreg.CAPABILITIES, name, host),
            lambda: serves_every_role(name),
            lambda: not_redirected(name),
            lambda: packs_known and assets._refuse_missing_node_packs([name]),
            lambda: boot_known and assets._refuse_unmet_boot_contracts(
                [name], state=boot),
            lambda: None if name in assets._COVERED else _adapter_reason(engine),
            lambda: still_reason(name),
        )

    eligible, left_out = [], {}
    for name in vreg.all_engine_names():
        engine = vreg.get_engine(name)
        if is_cloud_engine(engine):
            continue
        reason = reason_for(name, engine)
        if reason:
            left_out[name] = reason
        else:
            eligible.append(name)
    return tuple(eligible), left_out


def still_pool(host=None) -> "tuple[tuple[str, ...], dict]":
    """``(eligible, left_out)``: the local image models a roll may draw here."""
    try:
        from . import _otr_image_engines  # noqa: F401 -- registers built-ins
        from ._otr_image_engines import registry as ireg
        from . import _otr_visual_assets as assets
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_image_engines  # type: ignore  # noqa: F401
        from _otr_image_engines import registry as ireg  # type: ignore
        import _otr_visual_assets as assets  # type: ignore
    host = live_host() if host is None else host

    def reason_for(name, engine):
        return _first_reason(
            lambda: _host_reason(ireg.CAPABILITIES, name, host),
            lambda: None if name in assets._COVERED else _adapter_reason(engine),
        )

    eligible, left_out = [], {}
    for name in ireg.all_engine_names():
        engine = ireg.get_engine(name)
        if is_cloud_engine(engine):
            continue
        reason = reason_for(name, engine)
        if reason:
            left_out[name] = reason
        else:
            eligible.append(name)
    return tuple(eligible), left_out


def _audio_pool(choices, roles, host, *, extra=None):
    try:
        from . import _otr_audio_engines  # noqa: F401 -- registers built-ins
        from ._otr_audio_engines import registry as areg
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_audio_engines  # type: ignore  # noqa: F401
        from _otr_audio_engines import registry as areg  # type: ignore

    def serves(name):
        for role in roles:
            areg.assert_usable(name, role)

    def own_check(engine):
        check = getattr(_instance(engine), "assert_usable", None)
        if callable(check):
            check(roles[0])

    def reason_for(name, engine):
        return _first_reason(
            lambda: _host_reason(areg.CAPABILITIES, name, host),
            lambda: serves(name),
            lambda: _install_reason(areg.CAPABILITIES, name, engine),
            lambda: own_check(engine),
            lambda: extra(name, engine) if extra else None,
        )

    eligible, left_out = [], {}
    for name in choices:
        if not areg.is_registered(name):
            continue
        engine = areg.get_engine(name)
        if is_cloud_engine(engine):
            continue
        reason = reason_for(name, engine)
        if reason:
            left_out[name] = reason
        else:
            eligible.append(name)
    return tuple(eligible), left_out


def music_pool(host=None) -> "tuple[tuple[str, ...], dict]":
    """``(eligible, left_out)``: the theme-music engines a roll may draw here,
    from the music node's own dropdown."""
    try:
        from .stable_audio_theme import StableAudioTheme
    except ImportError:  # pragma: no cover -- flat test imports
        from stable_audio_theme import StableAudioTheme  # type: ignore
    choices = StableAudioTheme.INPUT_TYPES()["required"]["engine"][0]
    return _audio_pool(choices, ("music",), live_host() if host is None else host)


def _engine_language_gap(name, row) -> "str | None":
    """Why engine ``name`` cannot speak ``row``'s language on this machine, in
    the engine's own words, or None. Only an engine with a ``language_gap``
    method knows of one (Kokoro on ONNX: the libraries under its copy of
    misaki's Mandarin and Japanese phonemizers)."""
    try:
        from . import _otr_audio_engines  # noqa: F401 -- registers built-ins
        from ._otr_audio_engines import registry as areg
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_audio_engines  # type: ignore  # noqa: F401
        from _otr_audio_engines import registry as areg  # type: ignore
    if not areg.is_registered(name):
        return None
    check = getattr(_instance(areg.get_engine(name)), "language_gap", None)
    return check((row.engines or {}).get(name) or {}) if callable(check) else None


def voice_pool(host=None, language=None) -> "tuple[tuple[str, ...], dict]":
    """``(eligible, left_out)``: the voice engines a roll may draw here.

    From CastLock's own dropdowns, and only an engine both of them offer,
    because one engine voices the characters and the announcer. ``language``
    is the episode's language row, or None for English: an engine the row
    does not admit, or one that cannot speak it on this machine, is left out.
    """
    try:
        from .cast_lock import CastLock
        from ._otr_episode_languages import ENGLISH_ISO
    except ImportError:  # pragma: no cover -- flat test imports
        from cast_lock import CastLock  # type: ignore
        from _otr_episode_languages import ENGLISH_ISO  # type: ignore
    optional = CastLock.INPUT_TYPES()["optional"]
    characters = [c for c in optional["char_voice_engine"][0] if c != "auto"]
    announcer = set(optional["announcer_voice_engine"][0])
    if getattr(language, "iso", None) == ENGLISH_ISO:
        # English does not depend on its row: every engine voices it, so the
        # English row's short engine list must never narrow this pool.
        language = None

    def language_reason(name, _engine):
        if language is None:
            return None
        if name not in (language.engines or {}):
            return "is not admitted for this episode's language"
        why = _engine_language_gap(name, language)
        return ("cannot speak %s here: %s" % (language.label, why)) if why else None

    eligible, left_out = _audio_pool(
        [c for c in characters if c in announcer], _VOICE_ROLES,
        live_host() if host is None else host,
        extra=lambda name, engine: (language_reason(name, engine)
                                    or _reference_clips_reason(name, engine)))
    for name in characters:
        if name not in announcer and name not in left_out:
            left_out[name] = "voices characters only, so it cannot voice the whole cast"
    return eligible, left_out


_POOLS = {"video_lane": video_pool, "still_model": still_pool,
          "music_engine": music_pool}


# ---------------------------------------------------------------------------
# What a person can tick
# ---------------------------------------------------------------------------

def video_roll_choices() -> "tuple[str, ...]":
    """The video lanes the checklist offers, named as the dropdown names them.

    Every registered local lane that can take all three roles. Whether it runs
    on THIS machine is decided at run time, so the list is the same on every
    machine. HuMo is not offered: it voices a face from speech, and the route
    freeze moves it off the announcer and music roles (the redirect keys on the
    same ``audio_driven_face`` family), so it could never be drawn.
    """
    try:
        from . import _otr_video_engines  # noqa: F401 -- registers built-ins
        from ._otr_video_engines import registry as vreg
        from ._otr_shared import role_slots
        from ._otr_shared.public_engines import _INTERNAL_TO_PUBLIC
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_video_engines  # type: ignore  # noqa: F401
        from _otr_video_engines import registry as vreg  # type: ignore
        from _otr_shared import role_slots  # type: ignore
        from _otr_shared.public_engines import _INTERNAL_TO_PUBLIC  # type: ignore
    out = []
    for name in vreg.all_engine_names():
        engine = vreg.get_engine(name)
        if is_cloud_engine(engine) or getattr(engine, "family", "") == "audio_driven_face":
            continue
        try:
            for role in role_slots.ROLE_TO_VIDEO_SLOT:
                vreg.assert_usable(name, role)
        except Exception:  # noqa: BLE001 -- a lane that cannot take every role
            continue
        out.append(_INTERNAL_TO_PUBLIC.get(name, name))
    return tuple(out)


def still_roll_choices() -> "tuple[str, ...]":
    """The image models the checklist offers: every registered local one."""
    try:
        from . import _otr_image_engines  # noqa: F401 -- registers built-ins
        from ._otr_image_engines import registry as ireg
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_image_engines  # type: ignore  # noqa: F401
        from _otr_image_engines import registry as ireg  # type: ignore
    return tuple(n for n in ireg.all_engine_names()
                 if not is_cloud_engine(ireg.get_engine(n)))


def _ticked(inputs, pool_name, where) -> "tuple[str, ...]":
    """The internal ids ticked in one checklist; ``()`` when none is ticked.
    Forgiving about case and separators, strict about names, like the other
    pools (``_otr_rolls.parse_roll_pool``)."""
    value = inputs.get(pool_name, "")
    if isinstance(value, (list, tuple)) and len(value) == 2 and not isinstance(value[1], str):
        raise LaneRollError("%s: %s is wired to another node; tick the "
                            "choices instead" % (where, pool_name))
    try:
        from ._otr_shared.public_engines import resolve_engine_id
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared.public_engines import resolve_engine_id  # type: ignore
    choices = video_roll_choices() if pool_name == VIDEO_POOL else still_roll_choices()
    picked = _ROLLS.parse_roll_pool(value, valid_ids=choices, surface=pool_name)
    return tuple(resolve_engine_id(p) for p in picked)


# ---------------------------------------------------------------------------
# One draw
# ---------------------------------------------------------------------------

def _left_out_words(left_out, limit=6) -> str:
    items = sorted(left_out.items())
    text = "; ".join("%s %s" % pair for pair in items[:limit])
    if len(items) > limit:
        text += "; and %d more" % (len(items) - limit)
    return text


def _draw(surface, pool, env, rng_factory, ticked=()) -> dict:
    """One draw. ``ticked`` narrows the pool to the checked models; a ticked
    model this machine cannot run stays out, with its reason."""
    switch, seed_env, _ledger_key, noun = SURFACES[surface]
    eligible, left_out = pool
    if ticked:
        left_out = {k: v for k, v in left_out.items() if k in ticked}
        eligible = [e for e in eligible if e in ticked]
    order = tuple(sorted(set(eligible)))
    if not order:
        if ticked:
            raise LaneRollError(
                "%s is on, but none of the ticked %ss can run on this machine%s. "
                "Tick others, or clear the list to roll among every local %s "
                "that can." % (
                    switch, noun,
                    (" (%s)" % _left_out_words(left_out)) if left_out else "", noun))
        raise LaneRollError(
            "%s is on, but no local %s can run on this machine%s. Turn the "
            "switch off and pick one by hand." % (
                switch, noun,
                (" (%s)" % _left_out_words(left_out)) if left_out else ""))
    seed, seed_source = _ROLLS.resolve_seed(seed_env, env)
    selected = _ROLLS.draw(order, seed, rng_factory)
    receipt = _ROLLS.RollReceipt(
        surface=surface, requested=switch, selected=selected, seed=seed,
        seed_source=seed_source, eligible_order=order).to_meta()
    if ticked:
        receipt["ticked"] = list(ticked)
    if left_out:
        receipt["left_out"] = dict(sorted(left_out.items()))
    log.info("[OTR.rolls] %s: %s, drawn from %d local %ss (seed %d, %s)%s",
             switch, selected, len(order), noun, seed, seed_source,
             ("; left out: " + _left_out_words(left_out)) if left_out else "")
    return receipt


# ---------------------------------------------------------------------------
# The gate rolls, in the queued prompt
# ---------------------------------------------------------------------------

def _where(node_id, node) -> str:
    return "%s #%s" % (node.get("class_type"), node_id)


def _switch_on(inputs, name, where) -> bool:
    value = inputs.get(name, False)
    if isinstance(value, (list, tuple)):
        raise LaneRollError(
            "%s: %s is wired to another node; the roll needs a plain Yes or No"
            % (where, name))
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _set(inputs, name, value, where) -> None:
    if isinstance(inputs.get(name), (list, tuple)):
        raise LaneRollError(
            "%s: %s is wired to another node, so the roll cannot set it"
            % (where, name))
    inputs[name] = value


def _downstream(prompt, root_id):
    """Every ``(node_id, node)`` reachable from ``root_id`` through any link;
    in a queued prompt a link is ``[source_id, output_index]``."""
    reached = {str(root_id)}
    found = []
    grew = True
    while grew:
        grew = False
        for node_id, node in prompt.items():
            if node_id in reached or not isinstance(node, dict):
                continue
            for value in (node.get("inputs") or {}).values():
                if (isinstance(value, (list, tuple)) and len(value) == 2
                        and str(value[0]) in reached):
                    reached.add(node_id)
                    found.append((node_id, node))
                    grew = True
                    break
    return found


def _video_label(lane) -> str:
    """The exact dropdown text for ``lane``, as a person's pick would save it."""
    try:
        from .otr_video_director import exact_menu_option_for
    except ImportError:  # pragma: no cover -- flat test imports
        from otr_video_director import exact_menu_option_for  # type: ignore
    return exact_menu_option_for(lane)


def uses_a_still(inputs) -> bool:
    """Does any role's video lane, as it will render, consume its still?"""
    try:
        from . import _otr_visual_assets as assets
        from ._otr_shared import route_freeze
        from ._otr_shared.public_engines import resolve_engine_id
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_visual_assets as assets  # type: ignore
        from _otr_shared import route_freeze  # type: ignore
        from _otr_shared.public_engines import resolve_engine_id  # type: ignore
    custom = assets._custom_models(inputs)
    videos = {slot: resolve_engine_id(assets._resolve_slot(inputs, slot, custom, "video"))
              for slot in _VIDEO_SLOTS}
    effective = route_freeze.freeze_role_engines(videos)
    return any(not assets._proven_no_still(resolve_engine_id(lane), None)
               for lane in effective.values() if lane)


def roll_prompt_lanes(prompt, unique_id, *, env: "Mapping[str, str] | None" = None,
                      pools: "Mapping[str, Any] | None" = None,
                      still_check: "Callable[[dict], bool] | None" = None,
                      rng_factory: Callable[[int], Any] = random.Random) -> dict:
    """Roll the video, still and music switches that are on, writing the picks
    into the queued prompt. Only nodes downstream of this validator are read
    or written. Returns ``{node_id: {surface: receipt}}``.

    ``pools`` (``{surface: (eligible, left_out)}``) and ``still_check`` replace
    the live registry reads, for tests.
    """
    if unique_id is None or not str(unique_id).strip() or not isinstance(prompt, dict):
        return {}
    nodes = _downstream(prompt, unique_id)
    # A receipt a submitted prompt already carries -- re-queued from history --
    # is not this run's.
    for _node_id, node in nodes:
        node.pop(RECEIPT_KEY, None)
    by_type: dict = {}
    for node_id, node in nodes:
        by_type.setdefault(node.get("class_type"), []).append((node_id, node))

    wanted: dict = {}
    for node_id, node in by_type.get(_DIRECTOR, []):
        inputs = node.get("inputs") or {}
        for surface in ("video_lane", "still_model"):
            if _switch_on(inputs, SURFACES[surface][0], _where(node_id, node)):
                wanted.setdefault(node_id, []).append(surface)
    for node_id, node in by_type.get(_CAST_LOCK, []):
        if _switch_on(node.get("inputs") or {}, AUDIO_SWITCH, _where(node_id, node)):
            wanted.setdefault(node_id, []).append("music_engine")
    if not wanted:
        return {}

    receipts = {node_id: {} for node_id in wanted}
    replay = False
    for _writer_id, writer in by_type.get(_WRITER, []):
        replay_from = (writer.get("inputs") or {}).get("replay_from")
        replay = replay or (isinstance(replay_from, str) and bool(replay_from.strip()))
    if replay:
        for node_id, surfaces in wanted.items():
            for surface in surfaces:
                receipts[node_id][surface] = {
                    "skipped": "a replay keeps the engines its frozen bundle recorded"}
        _attach(prompt, receipts)
        return receipts

    live_pools: dict = {}
    drawn: dict = {}

    def draw(surface, ticked=()):
        # One draw per surface and checklist for the whole run, shared by every
        # node that asked for it: one lane, one still model, one composer.
        if surface not in live_pools:
            live_pools[surface] = (pools or {}).get(surface) or _POOLS[surface]()
        key = (surface, tuple(ticked))
        if key not in drawn:
            drawn[key] = _draw(surface, live_pools[surface], env, rng_factory, ticked)
        return drawn[key]

    for node_id, node in by_type.get(_DIRECTOR, []):
        surfaces = wanted.get(node_id, ())
        inputs = node.setdefault("inputs", {})
        where = _where(node_id, node)
        if "video_lane" in surfaces:
            receipt = draw("video_lane", _ticked(inputs, VIDEO_POOL, where))
            label = _video_label(receipt["selected"])
            for slot in _VIDEO_SLOTS:
                _set(inputs, slot, label, where)
            receipts[node_id]["video_lane"] = receipt
        if "still_model" in surfaces:
            if not (still_check or uses_a_still)(inputs):
                receipts[node_id]["still_model"] = {
                    "skipped": "no video lane in this episode uses a still"}
            else:
                receipt = draw("still_model", _ticked(inputs, STILL_POOL, where))
                for slot in _IMAGE_SLOTS:
                    _set(inputs, slot, receipt["selected"], where)
                receipts[node_id]["still_model"] = receipt

    for node_id, _node in by_type.get(_CAST_LOCK, []):
        if node_id not in wanted:
            continue
        themes = by_type.get(_THEME, [])
        if not themes:
            receipts[node_id]["music_engine"] = {
                "skipped": "this workflow has no theme-music node"}
            continue
        music = draw("music_engine")
        for theme_id, theme in themes:
            _set(theme.setdefault("inputs", {}), "engine", music["selected"],
                 _where(theme_id, theme))
        receipts[node_id]["music_engine"] = music

    _attach(prompt, receipts)
    return receipts


def _attach(prompt, receipts) -> None:
    for node_id, got in receipts.items():
        if got:
            prompt[node_id][RECEIPT_KEY] = got


# ---------------------------------------------------------------------------
# The voice, rolled in CastLock
# ---------------------------------------------------------------------------

def roll_voice_engine(meta, cast_voice_policy, *,
                      env: "Mapping[str, str] | None" = None, host=None,
                      pool=None, rng_factory: Callable[[int], Any] = random.Random
                      ) -> dict:
    """The voice roll's receipt: ``selected`` names the engine for the whole
    cast, or ``skipped`` says why none was rolled.

    ``meta`` is the ledger meta, which carries the episode's language. ``pool``
    (``(eligible, left_out)``) replaces the live read, for tests.
    """
    if cast_voice_policy != _AUTO_REGISTRY:
        return {"skipped": "%s keeps the voices the writer assigned, so the "
                           "voice engine is not rolled" % cast_voice_policy}
    if pool is None:
        try:
            from . import _otr_episode_languages as langs
        except ImportError:  # pragma: no cover -- flat test imports
            import _otr_episode_languages as langs  # type: ignore
        meta = meta if isinstance(meta, dict) else {}
        language = None
        if langs.iso_from_meta(meta) != langs.ENGLISH_ISO:
            language = langs.row_from_meta(meta)
        pool = voice_pool(host, language)
    return _draw("voice_engine", pool, env, rng_factory)


# ---------------------------------------------------------------------------
# The episode language, voiced by this run (2026-09-28)
# ---------------------------------------------------------------------------
# Three of the operator's randomized Desktop runs wrote their scripts and died
# at the first Spanish line, two minutes in. The ONNX Kokoro now speaks the
# five espeak rows, but a language can still be out of this run's reach:
# a row whose phonemizer libraries or readiness extra are missing, or a
# voice engine the row does not admit.
# CastLock and the voice engine refuse those only after the writer has run, so
# the gate asks the same questions before it.

def language_voice_gap(row, engines=None, *, host=None) -> "str | None":
    """Why a cast voiced by ``engines`` cannot speak ``row``'s language on this
    machine, or None when it can.

    ``engines`` is ``(char, announcer)`` as CastLock resolves them
    (``cast_lock.cast_voice_engines``), or None when CastLock rolls the voice.
    These are the refusals CastLock and the engine would make after the
    writer: an engine the row does not admit, an engine that cannot speak it
    here, and -- when Kokoro voices part of the cast -- a readiness extra that
    does not import. English is never a gap.
    """
    try:
        from . import _otr_episode_languages as langs
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_episode_languages as langs  # type: ignore
    if row is None or row.iso == langs.ENGLISH_ISO:
        return None
    admitted = row.engines or {}
    if engines is None:
        eligible, left_out = voice_pool(host, row)
        if not eligible:
            own = {k: v for k, v in left_out.items() if k in admitted} or left_out
            return "the voice roll has no local engine that speaks %s here (%s)" % (
                row.label, _left_out_words(own))
        voiced_by = set(eligible)
    else:
        for name in dict.fromkeys(engines):
            if name not in admitted:
                return "the %s voice does not speak %s (that language admits %s)" % (
                    name, row.label, ", ".join(sorted(admitted)))
            why = _engine_language_gap(name, row)
            if why:
                return "%s cannot speak %s here: %s" % (name, row.label, why)
        voiced_by = set(engines)
    if "kokoro" in voiced_by:        # the extras are Kokoro's phonemizers
        try:
            langs.assert_readiness_extras(row)
        except langs.EpisodeLanguageError as exc:
            return str(exc)
    return None


def _cast_voices(node_id, node):
    """What voices this CastLock's cast: ``(char, announcer)``, None when it
    rolls the voice, or ``()`` when a wired input makes that unknowable here.
    A missing key reads as ``lock()``'s own default, which is what it gets."""
    inputs = node.get("inputs") or {}
    policy = inputs.get("cast_voice_policy", "preserve_ledger")
    engines = (inputs.get("char_voice_engine", "auto"),
               inputs.get("announcer_voice_engine", "auto"))
    if any(isinstance(v, (list, tuple)) for v in (policy,) + engines):
        return ()
    if _switch_on(inputs, AUDIO_SWITCH, _where(node_id, node)) and policy == _AUTO_REGISTRY:
        return None
    try:
        from .cast_lock import cast_voice_engines
    except ImportError:  # pragma: no cover -- flat test imports
        from cast_lock import cast_voice_engines  # type: ignore
    return cast_voice_engines(*engines)


def _run_voice_gap(row, voices, host):
    for engines in voices:
        if engines == ():
            continue
        gap = language_voice_gap(row, engines, host=host)
        if gap:
            return gap
    return None


#: The form's own names for the rows a refusal points at (config/app_mode.json).
_LANGUAGE_ROW = "Language"
_POOL_ROW = "Languages to roll"
_VOICE_ROWS = "'Characters - voices' and 'Announcer - voice'"
_AUDIO_ROLL_ROW = "Randomize audio (non-cloud)"


def _language_advice(row, voices, host) -> str:
    """What besides another language would let this run speak ``row``: turning
    the audio roll off when it is on, or the engines that can speak it here."""
    if any(engines is None for engines in voices):
        return ("or turn '%s' off and set %s by hand (the roll draws only local "
                "engines)" % (_AUDIO_ROLL_ROW, _VOICE_ROWS))
    able = [name for name in sorted(row.engines or {})
            if language_voice_gap(row, (name, name), host=host) is None]
    if not able:
        return "no voice engine can speak it on this machine"
    # Local engines first, and the cloud note on the cloud engine itself:
    # after "google_tts or kokoro" it read as if Kokoro needed a Google key.
    able.sort(key=lambda name: (name.startswith("google"), name))
    words = " or ".join(
        name + (" (a cloud voice: it needs your Google API key)"
                if name.startswith("google") else "")
        for name in able)
    return "or set %s to %s" % (_VOICE_ROWS, words)


def settle_prompt_language(prompt, unique_id, *, host=None) -> dict:
    """Keep each writer's episode language to one this run's voices can speak.

    A language picked by hand that they cannot speak refuses now, before the
    writer runs, and says what would work. A rolled language loses those from
    its pool, each logged with its reason, and refuses only when nothing is
    left. The narrowed pool is written into the queued prompt, as the model
    rolls write theirs, and the writer's roll draws from it: with two or more
    left, the ledger's ``language_roll`` names exactly those; with one left the
    writer takes it as a pick, as it takes a one-item pool, and the log line
    here is the record. Returns ``{writer_id: {label: reason}}`` for what was
    left out. A replay, English or Off, a wired language input, and a graph
    with no CastLock downstream of this validator are left alone.
    """
    if unique_id is None or not str(unique_id).strip() or not isinstance(prompt, dict):
        return {}
    try:
        from . import _otr_episode_languages as langs
    except ImportError:  # pragma: no cover -- flat test imports
        import _otr_episode_languages as langs  # type: ignore
    nodes = _downstream(prompt, unique_id)
    voices = [_cast_voices(node_id, node) for node_id, node in nodes
              if node.get("class_type") == _CAST_LOCK]
    if not voices:
        return {}
    left: dict = {}
    for writer_id, writer in nodes:
        if writer.get("class_type") != _WRITER:
            continue
        inputs = writer.setdefault("inputs", {})
        where = _where(writer_id, writer)
        replay_from = inputs.get("replay_from")
        requested = inputs.get("episode_language", "")
        if (isinstance(replay_from, str) and replay_from.strip()) \
                or not isinstance(requested, str):
            continue
        if not _ROLLS.is_language_sentinel(requested):
            try:
                row = langs.resolve_label(requested).row
            except langs.EpisodeLanguageError:
                continue              # the writer names an unknown label itself
            gap = _run_voice_gap(row, voices, host)
            if gap:
                raise langs.EpisodeLanguageError(
                    "%s: pick another %s, %s. This run cannot voice %s: %s."
                    % (where, _LANGUAGE_ROW, _language_advice(row, voices, host),
                       row.label, gap))
            continue
        pool_value = inputs.get("language_roll_pool", "")
        if isinstance(pool_value, (list, tuple)) and len(pool_value) == 2 \
                and not isinstance(pool_value[1], str):
            continue                  # wired: the writer reads it at run time
        labels = tuple(c for c in langs.dropdown_choices() if c != langs.OFF_LABEL)
        try:
            chosen = _ROLLS.parse_roll_pool(pool_value, valid_ids=labels,
                                            surface="episode_language",
                                            refused=(langs.OFF_LABEL,))
        except _ROLLS.RollError:
            continue                  # the writer names the bad entry itself
        pool = chosen or labels
        gaps = {}
        for label in pool:
            gap = _run_voice_gap(langs.row_by_label(label), voices, host)
            if gap:
                gaps[label] = gap
        if not gaps:
            continue
        words = "; ".join("%s (%s)" % pair for pair in gaps.items())
        remaining = [label for label in pool if label not in gaps]
        if not remaining:
            raise langs.EpisodeLanguageError(
                "%s: tick English or another language this run's voices speak "
                "in %s. None of the ones it would roll can be voiced: %s."
                % (where, _POOL_ROW, words))
        _set(inputs, "language_roll_pool", ", ".join(remaining), where)
        log.info("[OTR.rolls] episode_language roll: left out %s; %s", words,
                 ("only %s is left, so that is the episode's language" % remaining[0])
                 if len(remaining) == 1
                 else "rolling among " + ", ".join(remaining))
        left[writer_id] = gaps
    return left


# ---------------------------------------------------------------------------
# Reading the gate receipts back
# ---------------------------------------------------------------------------

def receipts_on_node(prompt, node_id) -> dict:
    """The receipts the gate left on one node of this run's prompt."""
    if not isinstance(prompt, dict) or node_id is None:
        return {}
    node = prompt.get(str(node_id))
    got = node.get(RECEIPT_KEY) if isinstance(node, dict) else None
    return dict(got) if isinstance(got, dict) else {}


def assert_rolled(prompt, node_id, surfaces, node_label) -> dict:
    """This node's gate receipts for ``surfaces`` (the ones whose switch is on).

    Refuses when a switch is on and nothing rolled it: the gate rolls only the
    nodes downstream of the validator, and any other node would render its
    saved picks while its switch says they were rolled.
    """
    surfaces = tuple(surfaces)
    if not surfaces:
        return {}
    got = receipts_on_node(prompt, node_id)
    missing = [s for s in surfaces if s not in got]
    if missing:
        switches = " and ".join(sorted({SURFACES[s][0] for s in missing}))
        raise LaneRollError(
            "%s: %s is on, but nothing rolled it. The roll happens in the OTR "
            "Workflow Validator before the run starts, and this node is not "
            "downstream of one. Turn the switch off, or wire the validator "
            "back in." % (node_label, switches))
    return {s: got[s] for s in surfaces}


def ledger_meta(prompt) -> dict:
    """Every gate receipt in this run's prompt, under its ledger key."""
    out: dict = {}
    if not isinstance(prompt, dict):
        return out
    for node in prompt.values():
        got = node.get(RECEIPT_KEY) if isinstance(node, dict) else None
        if not isinstance(got, dict):
            continue
        for surface, receipt in got.items():
            if surface in GATE_SURFACES and isinstance(receipt, dict):
                out[SURFACES[surface][2]] = dict(receipt)
    return out
