"""The WAN render-recipe freeze -- ONE mechanism, each adapter's own data.

WHY THIS MODULE EXISTS. ``PBUG-20260723-02``: a production episode is submitted
to an ALREADY-BOOTED ComfyUI server, so a knob exported at launch cannot bind
the work. Code binds on every leg; an environment cannot. B6 (2026-07-27) closed
that hole for ``ltx_8gb`` by freezing its render recipe in code behind an
explicit consent act. Both WAN adapters had the whole pre-B6 defect, unfrozen --
``eng_wan_ti2v`` read sampler, scheduler, steps, cfg, shift, negative and four
VAE-tile vars from the environment, and ``eng_wan_i2v`` read six of its own
INLINE in ``_build_graph`` with bare ``int()`` / ``float()``: no range check, no
named refusal. Neither emitted a ``recipe`` receipt at all, so a WAN clip
stamped ``recipe: None`` and there was not even a wrong receipt to catch the
drift with.

WHAT IS SHARED HERE AND WHAT IS NOT. This module owns the MECHANISM: what the
consent act is, how a knob is parsed and range-checked, how a demoted knob is
named without being parsed, and what a measurement run stamps on its artifacts.
Each adapter owns its own DATA -- its recipe dict, its env names, its receipt
string and its consent env var -- because the two tiers are different models
with different measured values, and a single central recipe would be exactly the
central-authority shape this build exists to kill.

ONE RULE, ONE IMPLEMENTATION. The B6 panel found the same range check written
twice with OPPOSITE failure modes: one raised MALFORMED_CONFIG, the other
swallowed the bad value and substituted the default, so a sweep could mistype
the value it was measuring, render at something else, and stamp a receipt
saying it had measured it. Two adapters copying a freeze is how that recurs,
which is why the mechanism lives in one place and is called twice.

UTF-8, no BOM, ASCII-only. Cold-import clean: stdlib + the dep-free registry
exceptions only.
"""
from __future__ import annotations

import os

from .registry import EngineUnusable, EngineUsabilityReason

#: The explicit YES spellings for a consent act or a yes/no knob.
_TRUTHY = ("1", "true", "yes", "on")
#: The explicit NO spellings. A knob must be recognisably yes or recognisably
#: no -- "anything not yes means no" is how a typo becomes a silently different
#: measurement.
_FALSY = ("0", "false", "no", "off")

#: What a MEASUREMENT run stamps onto its clips instead of the frozen name.
#: The receipt rides the manifest row into ``stamp_durable(meta.render_engines)``
#: -- a durable ledger a published episode carries -- so a sweep artifact must
#: never be mistakable for a production one in the record that outlives the run.
PREQUALIFICATION_RECIPE_SUFFIX = "+prequalification"

#: ``VAEDecodeTiled`` input floors from the LIVE ``/object_info`` capture of
#: 2026-07-20 (docs/2026-07-20-OTR-video-tiers/ltx_8gb_discovery.json): tile_size
#: min 64, overlap min 0, temporal_size min 8, temporal_overlap min 4. A value
#: under the NODE'S OWN floor is a render that dies inside ComfyUI, so it is
#: refused here by name rather than handed over to fail late.
#:
#: This is a NODE-SCHEMA fact, not an adapter opinion, and ``eng_ltx_8gb`` holds
#: the same numbers in its own ``_VAE_TILE_BOUNDS``. The two are deliberately not
#: collapsed in this chunk: ltx_8gb is landed, green and out of this lane. Fold
#: them together whenever a general video-config seam next opens.
VAE_TILE_BOUNDS = {
    "vae_tile": (64, 4096),
    "vae_overlap": (0, 4096),
    "vae_temporal": (8, 4096),
    "vae_temporal_overlap": (4, 4096),
}


def prequalification_active(env_name):
    """True when the operator has explicitly opened one adapter's recipe knobs.

    Deliberately NOT inferred from the absence of a ledger or from any other
    ambient condition: a signal you can arrive at by accident is one a
    production leg can arrive at by accident. Present-but-falsy (``0``, empty,
    ``no``) is a production leg.

    The env var is passed IN, never assumed, because each adapter owns its own
    consent act. One shared switch would open both tiers at once and stamp a
    ``+prequalification`` receipt on a clip that had rendered with the frozen
    recipe -- a receipt that lies in the safer direction is still a receipt
    that lies."""
    return (os.environ.get(env_name, "") or "").strip().lower() in _TRUTHY


def ignored_override_keys(recipe_env_keys):
    """Which frozen knobs the environment is trying to set, by NAME ONLY.

    PRESENCE, NEVER PARSING, and that is load-bearing rather than tidy. If this
    parsed, a stale malformed ``OTR_WAN_TI2V_STEPS=not-a-number`` left in a
    long-booted server's environment would raise MALFORMED_CONFIG and kill a
    production leg over a knob that has NO EFFECT on that leg -- the precise
    shape ``PBUG-20260723-02`` says must not happen. Outside the consent act
    these values cannot bind, so they cannot be wrong; they can only be
    ignored, and the operator is told."""
    return sorted(env for env in recipe_env_keys.values()
                  if os.environ.get(env) not in (None, ""))


def recipe_receipt(frozen_receipt, prequalification_env):
    """The recipe string a rendered clip is STAMPED with.

    NOT simply the frozen name, and the difference is a ledger-integrity one.
    Under the consent act the knobs genuinely bind, so the clip on disk may
    share none of the frozen values -- while the receipt rides the manifest row
    into ``stamp_durable(meta.render_engines)``. Stamping the frozen name onto a
    sweep artifact would make a measurement indistinguishable from production in
    the one record that outlives the run."""
    if prequalification_active(prequalification_env):
        return frozen_receipt + PREQUALIFICATION_RECIPE_SUFFIX
    return frozen_receipt


def config_number(engine, env, dflt, lo, hi, cast):
    """Parse + RANGE-CHECK one numeric knob, or fail CLOSED by name.

    THE ONE implementation, deliberately -- see this module's docstring. Called
    only under the consent act; outside it the frozen value is returned without
    this function being reached, which is what keeps a stale malformed value
    from killing a leg it cannot influence."""
    raw = os.environ.get(env)
    if raw is None or raw == "":
        return cast(dflt)
    try:
        val = cast(raw)
    except (TypeError, ValueError):
        raise EngineUnusable(
            engine.name, engine.family, EngineUsabilityReason.MALFORMED_CONFIG,
            "%s=%r is not a valid number" % (env, raw), kind="video")
    if not (lo <= val <= hi):
        raise EngineUnusable(
            engine.name, engine.family, EngineUsabilityReason.MALFORMED_CONFIG,
            "%s=%s out of range [%s, %s]" % (env, val, lo, hi), kind="video")
    return val


def config_flag(engine, env, frozen):
    """A yes/no knob under the consent act, or fail CLOSED by name.

    ``or dflt``, not ``get(name, dflt)``: an exported-but-EMPTY var would
    otherwise read as ``""`` -- not truthy -- and force the knob OFF against a
    frozen default of ON. Every accessor in this build treats empty as unset.

    Unrecognised values RAISE rather than collapsing to False, because inside
    the consent act this is a MEASUREMENT INPUT: a sweep that mistypes the knob
    it is varying would otherwise decode the other way and stamp a receipt
    saying it had measured the one it named."""
    dflt = "1" if frozen else "0"
    raw = (os.environ.get(env) or dflt).strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    raise EngineUnusable(
        engine.name, engine.family, EngineUsabilityReason.MALFORMED_CONFIG,
        "%s=%r is not a yes/no value (yes: %s; no: %s)"
        % (env, os.environ.get(env), ", ".join(_TRUTHY), ", ".join(_FALSY)),
        kind="video")


def config_text(engine, env, frozen, allowed=None, strip=True, hint=""):
    """A text knob under the consent act, optionally whitelisted.

    Empty means unset (``or frozen``), matching every other accessor. When
    ``allowed`` is given, a value outside it fails CLOSED by name rather than
    reaching ComfyUI -- and ``hint`` carries the adapter's own reason so the
    refusal still says WHY that whitelist exists.

    ``strip=False`` is for free prose such as the negative prompt, where the
    exact text is the render input and trimming it would silently change what
    was measured."""
    val = os.environ.get(env) or frozen
    if strip:
        val = val.strip()
    if allowed is not None and val not in allowed:
        raise EngineUnusable(
            engine.name, engine.family, EngineUsabilityReason.MALFORMED_CONFIG,
            "%s=%r is not in the floor whitelist %s%s"
            % (env, val, sorted(allowed), hint), kind="video")
    return val


#: THE DIRECTION MARKERS. Both warnings below name the same knobs, interpolate
#: the same recipe, and contain the substring "PREQUALIFICATION" -- it is inside
#: the consent env var's own name. The B6 panel proved that a test asserting
#: those shared parts stays GREEN when the two bodies are SWAPPED, so each
#: branch carries a marker the other must not, and the tests pin the marker
#: rather than a copy of the prose.
FROZEN_MARKER = "FROZEN"
MEASUREMENT_MARKER = "measurement run"


def warn_ignored(logger, engine_name, receipt, ignored, prequalification_env):
    """Outside the consent act: NAME what is being ignored (never parse it).

    Named because a knob that silently does nothing is a complaint this build
    has already collected; ignored because a stale malformed value must not be
    able to kill a leg it cannot influence."""
    if not ignored:
        return
    logger.warning(
        "[OTR video] %s recipe %s is %s in code; ignoring %s from the "
        "environment. These knobs bind only under %s=1 -- a production leg is "
        "submitted to an already-booted server, so its recipe may not depend "
        "on how that server started (PBUG-20260723-02).",
        engine_name, receipt, FROZEN_MARKER, ", ".join(ignored),
        prequalification_env)


def warn_honoured(logger, engine_name, receipt, honoured, prequalification_env):
    """Under the consent act: say what was actually honoured, so a sweep's log
    records what it measured rather than what it intended to measure."""
    if not honoured:
        return
    logger.warning(
        "[OTR video] %s PREQUALIFICATION (%s is set): honouring %s from the "
        "environment in place of recipe %s. This is a %s, not a production "
        "contract.",
        engine_name, prequalification_env, ", ".join(honoured), receipt,
        MEASUREMENT_MARKER)


__all__ = [
    "PREQUALIFICATION_RECIPE_SUFFIX", "VAE_TILE_BOUNDS",
    "FROZEN_MARKER", "MEASUREMENT_MARKER",
    "prequalification_active", "ignored_override_keys", "recipe_receipt",
    "config_number", "config_flag", "config_text",
    "warn_ignored", "warn_honoured",
]
