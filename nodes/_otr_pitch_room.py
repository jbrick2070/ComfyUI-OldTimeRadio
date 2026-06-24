"""nodes/_otr_pitch_room.py -- T1 pitch room + greenlight (2026-06-23).

THE primary story-architecture lever: instead of letting the outliner collapse
every episode into the same "console standoff", the pitch room generates THREE
forcibly-divergent premises (each seeded from a different genre + protagonist
archetype + a concrete conflict from the real DOMAIN_PALETTE), taste-selects one
via a greenlight rubric, and hands the winner's distilled brief to the outliner
through ``dataclasses.replace(outline_req, script_brief=...)``.

Ships DARK: gated behind ``OTR_ENABLE_PITCH_ROOM`` (default OFF). When OFF the
writer never calls ``run_pitch_room`` -- byte-identical, no ``meta.story_quality
.pitch`` key. Local greenlight is the DEFAULT; a frontier greenlight upgrade is
opt-in (``OTR_ENABLE_FRONTIER_GREENLIGHT`` + ``OTR_GREENLIGHT_MODEL``) and
fail-CLOSED to local.

Determinism: the (genre, archetype, conflict) seeds are sha256-keyed off the
episode cast_seed, so the same episode always pitches the same slate. The LLM
calls are best-effort; any failure falls back to the original brief and stamps
``pitch.status = failed_fallback`` -- the pitch room NEVER breaks the writer.

Dependency note: stdlib + the stdlib-leaf ``_otr_story_quality_l12`` at module
load. ``structured_call`` / pydantic are imported LOCALLY inside the call (the
writer forbids heavy module-level imports; mirrors grade_story).

UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# Stdlib-leaf L1/L2 helpers (no torch, no _otr_outline cycle).
try:
    from ._otr_story_quality_l12 import DOMAIN_PALETTE, conflict_palette, select_domain
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_story_quality_l12 import (  # type: ignore
        DOMAIN_PALETTE,
        conflict_palette,
        select_domain,
    )

log = logging.getLogger("OTR")

# Forced-divergence seed pools. Three pitches draw THREE DISTINCT pairs so the
# slate cannot collapse to one shape.
_GENRES: Tuple[str, ...] = ("thriller", "drama", "sci-fi", "noir")
_ARCHETYPES: Tuple[str, ...] = (
    "reluctant hero",
    "anti-hero",
    "naive idealist",
    "jaded veteran",
)
_N_PITCHES = 3
_MAX_PITCH_REGEN = 2          # <=2 regenerations then fall back (SPEC).
_BRIEF_MAX_WORDS = 200        # hard-truncate the handoff brief (~200 tokens).


# ---------------------------------------------------------------------------
# Flag gate
# ---------------------------------------------------------------------------
def _env_truthy(val: Any) -> bool:
    return str(val if val is not None else "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def pitch_room_enabled(env=None) -> bool:
    """True when OTR_ENABLE_PITCH_ROOM is truthy. Default OFF => the writer skips
    the pitch room entirely (byte-identical)."""
    import os
    if env is None:
        env = os.environ
    return _env_truthy(env.get("OTR_ENABLE_PITCH_ROOM", ""))


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
@dataclass
class PitchMeta:
    """Outcome of the pitch room. ``status`` is one of: ``greenlit`` (a winner
    was selected + the brief replaced), ``failed_fallback`` (kept the original
    brief). JSON-friendly via ``to_dict``."""

    status: str
    seed_context: Any = None
    domain: str = ""
    genres: List[str] = field(default_factory=list)
    archetypes: List[str] = field(default_factory=list)
    selected_id: Optional[int] = None
    ranking: List[int] = field(default_factory=list)
    greenlight_source: str = ""
    rationale: str = ""
    brief: str = ""
    candidates: List[dict] = field(default_factory=list)
    fail_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "seed_context": self.seed_context,
            "domain": self.domain,
            "genres": list(self.genres),
            "archetypes": list(self.archetypes),
            "selected_id": self.selected_id,
            "ranking": list(self.ranking),
            "greenlight_source": self.greenlight_source,
            "rationale": self.rationale,
            "brief": self.brief,
            "candidates": list(self.candidates),
            "fail_reason": self.fail_reason,
        }


# ---------------------------------------------------------------------------
# Deterministic divergence seeds
# ---------------------------------------------------------------------------
def _seed_int(seed_context: Any, tag: str) -> int:
    return int(hashlib.sha256(f"{seed_context}:{tag}".encode("utf-8")).hexdigest(), 16)


def _distinct_pick(pool: Tuple[str, ...], base: int, count: int) -> List[str]:
    """Pick ``count`` DISTINCT items from ``pool`` deterministically from ``base``.

    A seeded shuffle (NOT a modular step -- a step sharing a factor with
    len(pool) only visits a subset and could never reach ``count`` distinct
    items) guarantees ``min(count, len(pool))`` distinct picks while still
    varying the slate per episode. Pads by cycling only when count > len(pool)
    (never for the size-4 genre/archetype pools)."""
    import random
    items = list(pool)
    if not items:
        return []
    rng = random.Random(base)
    rng.shuffle(items)
    out = items[:count]
    i = 0
    while len(out) < count:        # only if count > len(pool)
        out.append(items[i % len(items)])
        i += 1
    return out


def build_pitch_seeds(seed_context: Any, domain: str) -> List[dict]:
    """Build the 3 (genre, archetype, conflict_object, conflict_type) seeds.
    Deterministic. The conflict materials come from the REAL DOMAIN_PALETTE."""
    palette = conflict_palette(domain) or {}
    objs = list(palette.get("conflict_objects") or ()) or ["the contested thing"]
    types = list(palette.get("conflict_types") or ()) or ["a fight over what is right"]
    g_base = _seed_int(seed_context, "pitch:genres")
    a_base = _seed_int(seed_context, "pitch:archetypes")
    genres = _distinct_pick(_GENRES, g_base, _N_PITCHES)
    archetypes = _distinct_pick(_ARCHETYPES, a_base, _N_PITCHES)
    seeds: List[dict] = []
    for i in range(_N_PITCHES):
        o_idx = _seed_int(seed_context, f"pitch:obj:{i}") % len(objs)
        t_idx = _seed_int(seed_context, f"pitch:type:{i}") % len(types)
        seeds.append({
            "id": i + 1,
            "genre": genres[i],
            "archetype": archetypes[i],
            "conflict_object": objs[o_idx],
            "conflict_type": types[t_idx],
        })
    return seeds


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
def _source_material(outline_req: Any) -> str:
    brief = str(getattr(outline_req, "script_brief", "") or "").strip()
    if brief:
        return brief
    return str(getattr(outline_req, "news_seed", "") or "").strip()


def build_pitch_prompt(outline_req: Any, seeds: List[dict]) -> Tuple[str, str]:
    """Return (system, user) for the divergent-pitch generation call."""
    system = (
        "You are a sharp showrunner breaking three COMPLETELY DIFFERENT story "
        "pitches for a short science-fiction audio drama from the SAME source "
        "material. The three pitches MUST diverge -- different protagonists, "
        "different central conflicts, different emotional cores. AVOID the "
        "'console standoff' cliche (people shouting over a lever/console while a "
        "gauge climbs and a countdown runs, with the climax happening off-stage). "
        "Make the decisive moment happen ON-STAGE between characters. Return ONE "
        "JSON object {\"candidates\": [ ... 3 items ... ]}; each item has the "
        "fields exactly as specified, no prose, no fences."
    )
    lines = [
        f"Source material:\n{_source_material(outline_req)}",
        "",
        f"Style: {getattr(outline_req, 'style', '')}",
        "",
        "Break exactly 3 pitches, one per seed below. Each pitch must take its "
        "assigned genre and protagonist archetype, and build its central "
        "conflict from the assigned concrete conflict material:",
    ]
    for s in seeds:
        lines.append(
            f"  Pitch {s['id']}: genre={s['genre']}; protagonist archetype="
            f"{s['archetype']}; conflict material={s['conflict_object']}; "
            f"conflict shape={s['conflict_type']}."
        )
    lines.append("")
    lines.append(
        "For each pitch return: id (1/2/3, matching the seed), logline, "
        "protagonist, antagonist_or_pressure, genre_mode, emotional_core, "
        "theme_sentence, final_20_seconds (what the LAST 20 seconds sound like "
        "-- the on-stage climax, NOT an announcer summary), conflict_type, "
        "setting_class, surprise (1-5), human_want (1-5), stageability (1-5), "
        "console_standoff_risk (1-5, LOWER is better), why_different (one line "
        "on how this pitch differs from the other two)."
    )
    return system, "\n".join(lines)


def build_greenlight_prompt(candidates: List[dict], style: str) -> Tuple[str, str]:
    """Return (system, user) for the taste/greenlight selection call."""
    system = (
        "You are a discerning greenlight executive. Pick the ONE pitch that "
        "would make the best short audio drama: a clear human want, a conflict "
        "that plays ON-STAGE between characters, real surprise, and the LOWEST "
        "'console standoff' risk. Return ONE JSON object {\"selected_id\": int, "
        "\"ranking\": [ids best-to-worst], \"rationale\": str}. No prose, no fences."
    )
    parts = [f"Style: {style}", "", "Pitches:"]
    for c in candidates:
        parts.append(
            f"- id {c.get('id')}: {c.get('logline', '')} "
            f"[protagonist: {c.get('protagonist', '')}; core: "
            f"{c.get('emotional_core', '')}; final 20s: "
            f"{c.get('final_20_seconds', '')}; console_standoff_risk="
            f"{c.get('console_standoff_risk', '?')}]"
        )
    parts.append("")
    parts.append("Choose the strongest. Return only the JSON object.")
    return system, "\n".join(parts)


# ---------------------------------------------------------------------------
# Generation + greenlight
# ---------------------------------------------------------------------------
def _schemas():
    """Build the pydantic schemas lazily (no pydantic at module load)."""
    from pydantic import BaseModel, Field

    class _PitchCandidate(BaseModel):
        id: int = Field(..., ge=1, le=3)
        logline: str = Field("", max_length=400)
        protagonist: str = Field("", max_length=200)
        antagonist_or_pressure: str = Field("", max_length=200)
        genre_mode: str = Field("", max_length=80)
        emotional_core: str = Field("", max_length=200)
        theme_sentence: str = Field("", max_length=300)
        final_20_seconds: str = Field("", max_length=400)
        conflict_type: str = Field("", max_length=200)
        setting_class: str = Field("", max_length=120)
        surprise: int = Field(3, ge=0, le=10)
        human_want: int = Field(3, ge=0, le=10)
        stageability: int = Field(3, ge=0, le=10)
        console_standoff_risk: int = Field(3, ge=0, le=10)
        why_different: str = Field("", max_length=300)

    class _PitchSlate(BaseModel):
        candidates: List[_PitchCandidate] = Field(default_factory=list)

    class _GreenlightDecision(BaseModel):
        selected_id: int
        ranking: List[int] = Field(default_factory=list)
        rationale: str = Field("", max_length=400)

    return _PitchCandidate, _PitchSlate, _GreenlightDecision


def _valid_candidates(slate: Any) -> List[dict]:
    """Pull candidates with UNIQUE ids in {1,2,3} from a parsed slate."""
    out: List[dict] = []
    seen: set = set()
    for c in getattr(slate, "candidates", []) or []:
        cid = int(getattr(c, "id", 0))
        if cid in (1, 2, 3) and cid not in seen:
            seen.add(cid)
            out.append(c.model_dump() if hasattr(c, "model_dump") else dict(c))
    return out


def generate_pitches(outline_req: Any, seeds: List[dict], *, generate_fn) -> List[dict]:
    """Generate the divergent pitch slate. Up to ``_MAX_PITCH_REGEN`` extra
    attempts to reach >= 3 valid unique-id candidates. Returns a list of dicts
    (possibly < 3 -- the caller decides fallback). Never raises."""
    try:
        from ._otr_structured_call import structured_call
        from ._otr_repair_prompts import make_dispatching_repair_factory
    except ImportError:  # pragma: no cover - standalone / test load
        from _otr_structured_call import structured_call  # type: ignore
        from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore

    _PC, _PitchSlate, _GD = _schemas()
    system, user = build_pitch_prompt(outline_req, seeds)

    def _post(slate) -> Optional[str]:
        valid = _valid_candidates(slate)
        if len(valid) < _N_PITCHES:
            return (
                f"need {_N_PITCHES} pitches with unique ids 1,2,3; got "
                f"{len(valid)}"
            )
        return None

    best: List[dict] = []
    for attempt in range(_MAX_PITCH_REGEN + 1):
        try:
            # LLM slot: creative -- the pitch room reuses the writer's creative
            # generate_fn (the model that writes the story).
            slate = structured_call(
                prompt=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                schema=_PitchSlate,
                slot_fn=generate_fn,
                base_temperature=0.8,
                structural_retry_temperature=0.4,
                repair_prompt_factory=make_dispatching_repair_factory(),
                post_validator=_post,
                max_new_tokens=1400,
                max_attempts=2,
                helper_name="OTR_PitchRoom",
            )
            valid = _valid_candidates(slate)
            if len(valid) > len(best):
                best = valid
            if len(valid) >= _N_PITCHES:
                return valid
        except Exception as exc:  # noqa: BLE001 -- never break the writer
            log.warning(
                "[pitch_room] pitch generation attempt %d failed (%s)",
                attempt, type(exc).__name__,
            )
    return best


def _tie_break(candidates: List[dict]) -> int:
    """Deterministic taste fallback: lowest console_standoff_risk, then lowest
    id. Returns the selected id."""
    def _key(c: dict) -> Tuple[int, int]:
        return (int(c.get("console_standoff_risk", 99)), int(c.get("id", 99)))
    return int(min(candidates, key=_key)["id"])


def _resolve_greenlight_fn(local_fn, frontier_cfg, env) -> Tuple[Any, str]:
    """Resolve the greenlight generate_fn. Frontier upgrade ONLY when
    OTR_ENABLE_FRONTIER_GREENLIGHT + OTR_GREENLIGHT_MODEL are set AND a frontier
    callable is supplied in ``frontier_cfg``; else local (the default).
    Fail-CLOSED: any missing piece => local."""
    if not _env_truthy(env.get("OTR_ENABLE_FRONTIER_GREENLIGHT", "")):
        return local_fn, "local"
    if not str(env.get("OTR_GREENLIGHT_MODEL", "") or "").strip():
        return local_fn, "local"
    fn = None
    if isinstance(frontier_cfg, dict):
        fn = frontier_cfg.get("generate_fn")
    elif callable(frontier_cfg):
        fn = frontier_cfg
    if callable(fn):
        return fn, "frontier"
    log.warning(
        "[pitch_room] frontier greenlight requested but no frontier generate_fn "
        "wired; falling back to local greenlight",
    )
    return local_fn, "local"


def greenlight(candidates: List[dict], *, generate_fn, style: str) -> Tuple[int, List[int], str, str]:
    """Taste-select a winner. Returns (selected_id, ranking, rationale, source).
    On any LLM/parse failure or invalid selection, falls back to the
    deterministic tie-break with source='tie_break'."""
    ids = [int(c["id"]) for c in candidates]
    try:
        from ._otr_structured_call import structured_call
        from ._otr_repair_prompts import make_dispatching_repair_factory
    except ImportError:  # pragma: no cover - standalone / test load
        from _otr_structured_call import structured_call  # type: ignore
        from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore
    _PC, _PitchSlate, _GD = _schemas()
    system, user = build_greenlight_prompt(candidates, style)

    def _post(dec) -> Optional[str]:
        if int(getattr(dec, "selected_id", -1)) not in ids:
            return f"selected_id must be one of {ids}"
        rk = [int(x) for x in (getattr(dec, "ranking", []) or [])]
        if sorted(rk) != sorted(ids):
            return f"ranking must be a permutation of {ids}"
        return None

    try:
        # LLM slot: creative -- greenlight taste call runs on the creative
        # generate_fn by default (frontier override is opt-in + fail-closed).
        dec = structured_call(
            prompt=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            schema=_GD,
            slot_fn=generate_fn,
            base_temperature=0.2,
            structural_retry_temperature=0.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_post,
            max_new_tokens=256,
            max_attempts=2,
            helper_name="OTR_PitchGreenlight",
        )
        sid = int(dec.selected_id)
        ranking = [int(x) for x in (dec.ranking or [])]
        rationale = str(getattr(dec, "rationale", "") or "").strip()
        return sid, ranking, rationale, "llm"
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[pitch_room] greenlight call failed (%s); deterministic tie-break",
            type(exc).__name__,
        )
        sid = _tie_break(candidates)
        ranking = sorted(
            ids, key=lambda i: (
                int(next(c for c in candidates if int(c["id"]) == i)
                    .get("console_standoff_risk", 99)),
                i,
            ),
        )
        return sid, ranking, "deterministic tie-break (greenlight unavailable)", "tie_break"


# ---------------------------------------------------------------------------
# Handoff
# ---------------------------------------------------------------------------
def winner_to_brief(candidate: dict) -> str:
    """Build the concise handoff ``script_brief`` from the winning pitch via the
    fixed template; hard-truncate to ~200 tokens (words)."""
    brief = (
        f"{str(candidate.get('logline', '') or '').strip()} "
        f"Protagonist: {str(candidate.get('protagonist', '') or '').strip()}. "
        f"Conflict: {str(candidate.get('conflict_type', '') or '').strip()}, "
        f"{str(candidate.get('setting_class', '') or '').strip()}. "
        f"Emotional core: {str(candidate.get('emotional_core', '') or '').strip()}. "
        f"Final 20s: {str(candidate.get('final_20_seconds', '') or '').strip()}."
    )
    brief = " ".join(brief.split())     # collapse whitespace
    words = brief.split(" ")
    if len(words) > _BRIEF_MAX_WORDS:
        brief = " ".join(words[:_BRIEF_MAX_WORDS]).rstrip(",;") + "."
    return brief


def _stamp(meta: Any, pm: PitchMeta) -> None:
    if isinstance(meta, dict):
        sq = meta.setdefault("story_quality", {})
        if isinstance(sq, dict):
            sq["pitch"] = pm.to_dict()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_pitch_room(outline_req: Any, *, generate_fn, local_model="",
                   frontier_cfg=None, seed_context=None, meta=None, env=None):
    """Generate 3 divergent pitches, taste-select one, and return
    ``(new_outline_req, PitchMeta)``. On any failure returns
    ``(outline_req, PitchMeta(status='failed_fallback'))`` (the original brief is
    kept). Always stamps ``meta.story_quality.pitch``. NEVER raises."""
    import os
    if env is None:
        env = os.environ

    domain = ""
    try:
        premise_hint = _source_material(outline_req)
        domain = select_domain(meta if isinstance(meta, dict) else {}, premise_hint)
        seeds = build_pitch_seeds(seed_context, domain)
        genres = [s["genre"] for s in seeds]
        archetypes = [s["archetype"] for s in seeds]

        candidates = generate_pitches(outline_req, seeds, generate_fn=generate_fn)
        if len(candidates) < _N_PITCHES:
            pm = PitchMeta(
                status="failed_fallback", seed_context=seed_context,
                domain=domain, genres=genres, archetypes=archetypes,
                candidates=candidates,
                fail_reason=f"only {len(candidates)} valid pitch(es) generated",
            )
            _stamp(meta, pm)
            log.warning(
                "[pitch_room] fell back to the original brief (%s)", pm.fail_reason,
            )
            return outline_req, pm

        gl_fn, gl_source = _resolve_greenlight_fn(generate_fn, frontier_cfg, env)
        sid, ranking, rationale, taste_source = greenlight(
            candidates, generate_fn=gl_fn,
            style=str(getattr(outline_req, "style", "") or ""),
        )
        winner = next(c for c in candidates if int(c["id"]) == sid)
        brief = winner_to_brief(winner)
        new_req = dataclasses.replace(outline_req, script_brief=brief)
        pm = PitchMeta(
            status="greenlit", seed_context=seed_context, domain=domain,
            genres=genres, archetypes=archetypes, selected_id=sid,
            ranking=ranking, greenlight_source=f"{gl_source}:{taste_source}",
            rationale=rationale, brief=brief, candidates=candidates,
        )
        _stamp(meta, pm)
        log.info(
            "[pitch_room] greenlit pitch %d/%d (domain=%s, source=%s): %s",
            sid, len(candidates), domain, pm.greenlight_source,
            str(winner.get("logline", ""))[:80],
        )
        return new_req, pm
    except Exception as exc:  # noqa: BLE001 -- the pitch room must never break the writer
        pm = PitchMeta(
            status="failed_fallback", seed_context=seed_context, domain=domain,
            fail_reason=f"{type(exc).__name__}: {exc}",
        )
        _stamp(meta, pm)
        log.warning("[pitch_room] unexpected failure (%s); kept original brief", exc)
        return outline_req, pm
