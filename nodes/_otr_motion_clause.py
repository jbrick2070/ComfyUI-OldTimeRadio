r"""Per-beat LTX motion_clause -- generation, validation, and the read-side resolver.

Spec: docs/2026-06-16-ltx-motion/MOTION_CLAUSE_SPEC.md (v2, roundtable-hardened).

Two layers:
  * A SEPARATE post-brief BATCH pass (:func:`generate_motion_clauses`) fills
    ``ledger['video']['shots'][i]['motion_clause']`` for EVERY shot -- a real clause for
    character/dialogue beats, a fallback object (static role text) otherwise -- so the
    render path stays READ-ONLY and deterministic.
  * The render path calls :func:`resolve_motion_clause_text` (read-only, guarded) to get
    the validated motion text, or ``None`` to fall back to the static role map.

Opt-in: default OFF (env ``OTR_LTX_MOTION_CLAUSE`` != "1") makes the batch pass write
static fallbacks only, so the composed prompt is byte-identical to today.

Self-contained (stdlib only) so it imports under bare pytest with no torch / no ComfyUI.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable, Optional

SCHEMA_VERSION = 1
FLAG_ENV = "OTR_LTX_MOTION_CLAUSE"
CLAUSE_MAX_CHARS = 70

# Allowed SUBTLE motion phrases (substring match -- the clause must contain >=1).
ALLOWED_PHRASES = (
    "talks", "speaks", "gestures", "leans", "nods", "glances", "tilts",
    "breathes", "shifts", "blinks", "small hand", "looks", "smiles", "listens",
)
# BANNED big-action / scene-cut phrases (the clause must contain NONE).
BANNED_PHRASES = (
    "stands up", "stands", "walks", "runs", "jumps", "turns around", "enters",
    "exits", "dramatic", "full-body", "full body", "camera cut", "cut to",
    "flies", "dances", "spins",
)
GENERIC_SUBJECTS = ("a person", "the person", "someone", "a man", "a woman", "a figure")

GENERATED_MODEL_UNSET = "static-role-map"


def enabled() -> bool:
    """True when the operator opted in via the env flag."""
    return str(os.environ.get(FLAG_ENV, "")).strip() == "1"


def _norm(text: str) -> str:
    """Lower-case + collapse whitespace (for hashing + matching)."""
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def compute_source_hash(char_id: str, beat_id: str, dialogue: str,
                        schema_version: int = SCHEMA_VERSION) -> str:
    """Stable hash over the inputs that should INVALIDATE a stored clause.

    Dialogue is included so a changed line regenerates the clause (operator
    directive: the line drives the motion)."""
    canon = json.dumps(
        {"c": str(char_id or ""), "b": str(beat_id or ""),
         "d": _norm(dialogue), "v": int(schema_version)},
        sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def validate_motion_clause(text: str, subject_name: str) -> tuple[bool, str]:
    """Return ``(ok, reason)``. The clause must be non-empty, within budget, name the
    subject, contain >=1 allowed phrase, and contain NO banned phrase / generic subject."""
    t = str(text or "").strip()
    if not t:
        return False, "empty"
    if len(t) > CLAUSE_MAX_CHARS:
        return False, "over_budget"
    low = t.lower()
    if any(b in low for b in BANNED_PHRASES):
        return False, "banned_phrase"
    if any(g in low for g in GENERIC_SUBJECTS):
        return False, "generic_subject"
    subj = _norm(subject_name)
    if subj and subj.split()[0] not in low:
        return False, "subject_missing"
    if not any(a in low for a in ALLOWED_PHRASES):
        return False, "no_allowed_phrase"
    return True, "ok"


def build_clause_messages(subject_name: str, dialogue: str, scene_ctx: str) -> list:
    """Few-shot chat messages for one beat (constrained, dialogue-driven)."""
    system = (
        "You write ONE short present-tense clause describing SUBTLE motion for a "
        "character in a 1940s radio-drama shot. Rules: name the character; <= 60 "
        "characters; use only small movements (talks, gestures slightly, leans, nods, "
        "glances, tilts head, breathes, shifts, blinks); NEVER big actions (stands up, "
        "walks, runs, turns around) or camera cuts; do NOT describe the scene. Match the "
        "delivery of the line -- livelier for an urgent line, smaller for a quiet one. "
        "Output ONLY the clause.")
    user = (
        "Character: %s\nLine: \"%s\"\nScene: %s\nClause:"
        % (str(subject_name or "the speaker"), str(dialogue or "").strip(),
           str(scene_ctx or "").strip()[:120]))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _clause_obj(text: str, *, model: str, fallback: bool,
                char_id: str, beat_id: str, dialogue: str) -> dict:
    return {
        "text": str(text or "").strip(),
        "model": str(model or ""),
        "fallback": bool(fallback),
        "schema_version": SCHEMA_VERSION,
        "source_hash": compute_source_hash(char_id, beat_id, dialogue),
    }


def resolve_motion_clause_text(shot: Any) -> Optional[str]:
    """Read-side guard for the render path. Returns the stored motion text when the shot
    carries a structurally-valid clause, else ``None`` (caller uses the static role map).
    Never trusts ledger text blindly: must be a dict with a non-empty, in-bounds string."""
    if not isinstance(shot, dict):
        return None
    mc = shot.get("motion_clause")
    if not isinstance(mc, dict):
        return None
    if mc.get("fallback"):
        # A fallback object means "no real clause" -> caller keeps its existing
        # (static role map / brief-composed) prompt, so render is byte-identical.
        return None
    text = mc.get("text")
    if not isinstance(text, str):
        return None
    t = text.strip()
    # generation already validated real clauses; allow fallback (static) text through up to
    # the static-map budget. Guard against absurd/garbage lengths.
    if not t or len(t) > 260:
        return None
    return t


def _first_clause(raw: str) -> str:
    """Take the first non-empty line/sentence from an LLM response."""
    s = str(raw or "").strip().strip('"').strip()
    for line in s.splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line
    return s


def _line_text_index(ledger: Any) -> dict:
    """``{line_id: text}`` from ``ledger['lines']`` (line_id == beat_id)."""
    out = {}
    for ln in (ledger or {}).get("lines") or []:
        if isinstance(ln, dict):
            lid = str(ln.get("line_id") or "")
            if lid:
                out.setdefault(lid, str(ln.get("text") or ""))
    return out


def _beat_id_for_shot(shot: Any) -> str:
    """A shot's beat id == its first source line id (mirrors render_driver)."""
    if not isinstance(shot, dict):
        return ""
    sids = shot.get("source_line_ids") or []
    if isinstance(sids, (list, tuple)) and sids:
        return str(sids[0] or "")
    return str(shot.get("shot_id") or "")


def generate_motion_clauses(ledger: Any, *,
                            generate_fn: Optional[Callable] = None,
                            name_resolver: Optional[Callable] = None,
                            scene_ctx: str = "") -> dict:
    """Post-brief BATCH pass: fill ``shots[i]['motion_clause']`` for EVERY shot.

    * ``generate_fn(messages, *, temperature, max_new_tokens, stop=None) -> str`` -- the
      writer-slot LLM closure. When ``None`` (or the flag is OFF), every shot gets a
      ``fallback`` object; :func:`resolve_motion_clause_text` returns ``None`` for those,
      so the render path keeps its existing prompt (byte-identical).
    * ``name_resolver(char_id) -> str`` -- display name for the subject; falls back to char_id.

    Returns disposition counts. Idempotent: a stored clause whose ``source_hash`` still
    matches is REUSED (deterministic re-render)."""
    video = (ledger or {}).get("video") if isinstance(ledger, dict) else None
    shots = (video or {}).get("shots") if isinstance(video, dict) else None
    counts = {"generated": 0, "reused": 0, "fallback": 0, "invalid": 0}
    if not isinstance(shots, list):
        return counts

    on = enabled() and callable(generate_fn)
    line_text = _line_text_index(ledger)

    def _name(cid: str) -> str:
        if callable(name_resolver):
            try:
                return str(name_resolver(cid) or cid or "")
            except Exception:  # noqa: BLE001 -- resolver must never break the pass
                return str(cid or "")
        return str(cid or "")

    for shot in shots:
        if not isinstance(shot, dict):
            continue
        beat_id = _beat_id_for_shot(shot)
        char_id = str(shot.get("char_id") or "")
        dialogue = line_text.get(beat_id, "")
        subject = _name(char_id)
        want_hash = compute_source_hash(char_id, beat_id, dialogue)

        prev = shot.get("motion_clause")
        if (isinstance(prev, dict) and not prev.get("fallback")
                and prev.get("source_hash") == want_hash
                and resolve_motion_clause_text(shot)):
            counts["reused"] += 1
            continue

        # Console / no-dialogue / flag-off -> static fallback (byte-identical path).
        if not on or not dialogue.strip() or not char_id:
            shot["motion_clause"] = _clause_obj(
                "", model=GENERATED_MODEL_UNSET, fallback=True,
                char_id=char_id, beat_id=beat_id, dialogue=dialogue)
            counts["fallback"] += 1
            continue

        try:
            # LLM slot: creative -- subtle per-beat motion clause (writer slot)
            raw = generate_fn(
                build_clause_messages(subject, dialogue, scene_ctx),
                temperature=0.4, max_new_tokens=40, stop=None)
            text = _first_clause(raw)
        except Exception:  # noqa: BLE001 -- LLM failure must never abort the episode
            text = ""

        ok, _reason = validate_motion_clause(text, subject)
        if ok:
            shot["motion_clause"] = _clause_obj(
                text, model="writer", fallback=False,
                char_id=char_id, beat_id=beat_id, dialogue=dialogue)
            counts["generated"] += 1
        else:
            if text:
                counts["invalid"] += 1
            shot["motion_clause"] = _clause_obj(
                "", model=GENERATED_MODEL_UNSET, fallback=True,
                char_id=char_id, beat_id=beat_id, dialogue=dialogue)
            counts["fallback"] += 1

    return counts


def make_name_resolver(ledger: Any) -> Callable:
    """``char_id -> display name`` from ``ledger['cast']`` (falls back to char_id)."""
    by_id = {}
    for c in (ledger or {}).get("cast") or []:
        if isinstance(c, dict):
            cid = str(c.get("char_id") or "")
            if cid:
                by_id[cid] = str(c.get("name") or cid)

    def _resolve(char_id):
        return by_id.get(str(char_id or ""), str(char_id or ""))

    return _resolve


def make_writer_generate_fn(model_id: Optional[str] = None):
    """Build a creative-slot writer ``generate_fn`` at RUNTIME (lazy; torch/model stack).

    Returns ``None`` on ANY failure so the caller falls back to static (never breaks the
    render). Reuses the proven script-writer primitives -- request_slot +
    _build_truncating_generate_fn -- which handle the HF / HTTP-backed lanes."""
    try:
        try:
            from . import _otr_model_loader as _ml  # type: ignore
            from . import _otr_model_catalog as _cat  # type: ignore
            from .OTR_LedgerScriptWriter import (  # type: ignore
                _build_truncating_generate_fn)
        except ImportError:  # pragma: no cover -- flat imports
            import _otr_model_loader as _ml  # type: ignore
            import _otr_model_catalog as _cat  # type: ignore
            from OTR_LedgerScriptWriter import (  # type: ignore
                _build_truncating_generate_fn)
        mid = str(model_id or getattr(_cat, "DEFAULT_LLM", "") or "")
        if not mid:
            return None
        entry = _ml.request_slot("creative", mid)  # LLM slot: creative
        return _build_truncating_generate_fn(entry)
    except Exception:  # noqa: BLE001 -- runtime model stack optional; fall back to static
        return None
