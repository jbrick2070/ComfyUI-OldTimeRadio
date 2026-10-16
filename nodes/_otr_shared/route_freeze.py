"""THE ONE ROUTE-FREEZE AUTHORITY: role -> EFFECTIVE video engine, once.

One operator mutation decides which engine actually renders a beat:
``OTR_FORCE_ENGINE_MAP``, the all-one-engine / per-role experiment knob.

There is NO role-based engine rewrite (2026-10-15, operator: "if you choose
ltx25 as the music or announcer that's fine but it should not auto populate").
A HuMo pick on announcer_visual / music_visual stays HuMo and renders the
brief-driven radio-with-a-face host (``radio_host_portrait``); the old
radio-is-host redirect to the LTX 2.5 audio-in lane is gone, so a HuMo run
succeeds or fails on HuMo alone.

WHY THIS MODULE EXISTS (2026-07-25, multi-clip coverage chunk 1a; kibitz r4
codex ``gpt-5.6-sol`` high + agy ``Gemini 3.6 Flash (High)``, then a six-way
grounded Sonnet fan-out, judge concurring). Before it, the routing rules were
implemented FOUR times: the real authority in ``render_driver``
(``apply_engine_override`` + the since-removed bookend redirect) and THREE independent
mirrors -- ``otr_meta_brief_image_prompt._effective_prompt_engine_for_role``,
``otr_image_gen_dispatcher._effective_video_engine_for_role`` /
``_effective_engine_after_force_map``, and ``otr_shot_lock``'s nested
``_effective_cast_time_engine``. TWO of the mirrors hard-coded the redirect
target as a bare literal, and TWO of them SWALLOWED a
malformed force map that the render path treats as terminal. Four copies of one
routing rule, disagreeing about both the target and the failure policy, is the
defect class the whole coverage build exists to close -- so the fix is to
delete the copies, not to relocate them.

FAIL-CLOSED, EVERYWHERE (2026-07-25). A malformed ``OTR_FORCE_ENGINE_MAP`` is
operator misconfiguration and is terminal at EVERY consumer, not only at render
(``render_driver.apply_engine_override``, landed ``57f4983a``). The mirrors
used to warn and continue on the UNFORCED plan, which meant the image phase
could spend ~10 minutes minting stills for engines that were never going to
render, and only then die. Failing at the first reader is strictly cheaper and
strictly honester. The old "failsafe" contract WAS the bug.

LAYERING. Module scope imports STDLIB ONLY, so the cheap policy nodes
(``OTR_VideoDirector``) can import this without dragging in the 3,400-line
render driver and without tripping the cold-import guard
(``tests/test_video_platform_aseam.py:71-82``: no torch / transformers /
diffusers at import time). The pure core takes its engine predicates INJECTED;
the convenience wrappers lazily import ``render_driver`` only when actually
called, which is always inside a node's execution, never at import.
"""

from __future__ import annotations

try:
    from . import env as otr_env
except ImportError:  # pragma: no cover -- loaded flat
    try:
        from _otr_shared import env as otr_env  # type: ignore  # nodes/ on sys.path
    except ImportError:
        import env as otr_env  # type: ignore  # _otr_shared/ on sys.path

#: The environment variable that decides effective routing. Any new routing
#: env var MUST be added here AND to the snapshot, or it escapes the freeze
#: (the r4 panel's standing warning: an env read outside the snapshot is a hole
#: in the ledger).
FORCE_ENGINE_MAP_ENV = "OTR_FORCE_ENGINE_MAP"


class RouteFreezeError(ValueError):
    """A malformed routing environment. Terminal at every consumer."""


def routing_env_snapshot(env=None) -> dict:
    """Capture the routing environment ONCE, normalized. Pure w.r.t. ``env``.

    Returns a plain JSON-serializable dict so it can ride the policy payload
    and the ledger unchanged. ``force_engine_map`` is the RAW spec string
    (stripped) -- kept raw, not pre-parsed, so the snapshot stays serializable
    and a consumer can re-verify the exact operator input it was frozen from.
    """
    # snapshot() is a COPY, and that is the right shape here: both reads
    # happen on the next two lines, so a copy and the live mapping cannot
    # disagree -- and a copy cannot be mutated back into this process by a
    # consumer of the returned dict.
    src = otr_env.snapshot() if env is None else env
    return {
        "force_engine_map": str(src.get(FORCE_ENGINE_MAP_ENV, "") or "").strip(),
    }


def snapshot_fingerprint(snapshot) -> str:
    """A short stable string for ``IS_CHANGED``. Pure.

    ComfyUI caches a node's output unless ``IS_CHANGED`` returns something new,
    so every node that FREEZES routing must fingerprint the captured state or
    the graph will happily serve a lock taken under a different environment.
    """
    snap = snapshot or {}
    return "force=%s" % str(snap.get("force_engine_map", "") or "")


def snapshots_agree(frozen, current) -> bool:
    """True iff two snapshots describe the same routing environment. Pure."""
    a, b = frozen or {}, current or {}
    return (str(a.get("force_engine_map", "") or "")
            == str(b.get("force_engine_map", "") or ""))


def forced_engine_for_role(role: str, mapping) -> str:
    """The force-map target for ``role`` (``*`` wins as the catch-all), or "".

    Pure. ``mapping`` is the already-parsed ``{role: engine}`` dict.
    """
    m = mapping or {}
    return str(m.get(str(role or "")) or m.get("*") or "")


def resolve_effective_engine(role, engine_id, *, force_mapping) -> str:
    """THE PURE CORE: one role's FINAL effective engine. No env, no registry.

    ``force_mapping`` is the parsed force map (``{}`` when unset). The force
    map is the ONLY rewrite: a role's picked engine is what renders it unless
    the operator forced another. Pure, so it is exhaustively testable on CPU.
    """
    forced = forced_engine_for_role(str(role or ""), force_mapping)
    return forced or str(engine_id or "")


def _render_driver():
    """Lazy import of the render-side authority. Never at module scope."""
    try:
        from .._otr_video_engines import render_driver as _rd  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import render_driver as _rd  # type: ignore
    return _rd


def parse_force_map(spec, *, driver=None) -> dict:
    """Parse a force-map spec, FAIL-CLOSED, via the render-side grammar.

    Raises :class:`RouteFreezeError` on a malformed spec. There is no
    warn-and-continue path: a silently unforced episode is indistinguishable
    from a working one, which is exactly how the old mirrors hid the defect.
    """
    text = str(spec or "").strip()
    if not text:
        return {}
    rd = driver if driver is not None else _render_driver()
    try:
        return dict(rd.parse_engine_override(text))
    except RouteFreezeError:
        raise
    except Exception as exc:  # noqa: BLE001 -- normalize to ONE terminal type
        raise RouteFreezeError(
            "%s is malformed and routing cannot be frozen: %s -- fix the map "
            "or unset it. NO FALLBACK to the unforced plan (a silently "
            "unforced render is indistinguishable from a working one)."
            % (FORCE_ENGINE_MAP_ENV, exc)) from exc


def effective_engine_for_role(role, engine_id, *, snapshot=None, driver=None) -> str:
    """One role's effective engine, resolving the render-side predicates itself.

    The convenience wrapper every former mirror now calls. Fail-closed on a
    malformed force map.
    """
    snap = routing_env_snapshot() if snapshot is None else snapshot
    mapping = parse_force_map(snap.get("force_engine_map", ""), driver=driver)
    return resolve_effective_engine(role, engine_id, force_mapping=mapping)


def freeze_role_engines(video_models, *, snapshot=None, driver=None,
                        roles=None) -> dict:
    """THE FROZEN MAP: ``{role: effective_engine}`` for every routed role.

    ``video_models`` is the director's per-role video slot map. Roles whose
    slot is empty are OMITTED (an empty pick has no effective engine, and
    inventing one would let a blank slot silently acquire a route).
    """
    snap = routing_env_snapshot() if snapshot is None else snapshot
    mapping = parse_force_map(snap.get("force_engine_map", ""), driver=driver)
    try:
        from . import role_slots as _rs  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        import role_slots as _rs  # type: ignore
    role_ids = tuple(_rs.ROLE_TO_VIDEO_SLOT) if roles is None else tuple(roles)
    out = {}
    for role in role_ids:
        picked = str(_rs.engine_id_for_role(video_models or {}, role) or "")
        if not picked:
            continue
        out[str(role)] = resolve_effective_engine(role, picked,
                                                  force_mapping=mapping)
    return out
