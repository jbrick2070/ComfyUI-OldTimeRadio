"""OTR_ShotLock -- the video-phase lock authority (A-S1/W1).

The video analogue of ``OTR_CastLock``: it runs AFTER audio timing freezes
(gated on ``OTR_EpisodeAssembler``'s ``audio_done``, out3), reads the frozen
ledger, and stamps ONE ``ledger['video']`` section -- the audio-derived clip
budget, the DAG-validated ordered ``execution_groups``, and the per-shot rows
with their creative directives. It mirrors ``OTR_CastLock``'s I/O exactly,
including the load-bearing ``done`` STRING gate that downstream ordering needs.

It OWNS prompt generation (it supersedes ``OTR_VideoPlan``): for character-
bearing beats it runs the M4 per-beat derivation -- one batched LLM call on the
writer's slot (V-11, NO new model_id widget), mirroring the Meta-brief protocol
of ``nodes/_otr_music_prompt.py`` -- deriving ``{expression, motion, camera}``
and composing a rich ``text_prompt`` + a structured ``creative`` sidecar into
``ledger['video'].shots[].creative``. The derivation is fail-soft: empty /
unparseable / truncated model output reseeds (max 2) then falls back to a
DETERMINISTIC template (``{appearance}, {setting}, {beat_text}``); a consistency
check that the prompt carries the cast's core traits + the brief setting WARNs +
falls back on a miss. It NEVER aborts the episode or touches the frozen audio
(invariant V-1). Cheap families (abstract / still / station-card) get NO creative
LLM call.

Determinism (V-7): per-shot ``request_hash`` mixes brief + cast content hashes +
beat_id + char_id; the prompt hash is taken AFTER the call. 3D ``expression`` is
a DRIVER-channel directive, never part of any mesh/cache key.

Import-time is side-effect-free; module scope imports only stdlib + the dep-free
shared resolver/registry/schemas. The LLM is resolved lazily (and is injectable
for tests). UTF-8, no BOM, ASCII source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

log = logging.getLogger("OTR")

from ._otr_shared import resolver as _resolver
from ._otr_shared.role_compat import Role
from ._otr_shared import role_slots as _role_slots

# ---------------------------------------------------------------------------
# Role mapping + which roles are "character-bearing" (get the rich derivation)
# ---------------------------------------------------------------------------

#: ledger ``speaker_role`` -> video role.
#: BUG 1 (2026-06-20): ``"character"`` is the CANONICAL writer speaker_role for a
#: dialogue line (set in OTR_LedgerScriptWriter / _otr_outline, compared in
#: _otr_anti_loop / _otr_ledger_reviewer). "char_voice"/"dialogue" stay as aliases.
#: rip-sfx-broll (2026-07-01): the "sfx" entry + the _DEFAULT_VIDEO_ROLE
#: fallback (background_abstract) were REMOVED with their roles -- an unmapped
#: speaker_role now FAILS LOUD in :func:`_video_role_for_line` (NO FALLBACKS).
SPEAKER_TO_VIDEO_ROLE = {
    "announcer": Role.ANNOUNCER_VISUAL.value,
    "music": Role.MUSIC_VISUAL.value,
    "music_open": Role.MUSIC_VISUAL.value,
    "music_close": Role.MUSIC_VISUAL.value,
    "music_inter": Role.MUSIC_VISUAL.value,
    "character": Role.CHARACTER_VIDEO.value,
    "char_voice": Role.CHARACTER_VIDEO.value,
    "dialogue": Role.CHARACTER_VIDEO.value,
}

#: Only these roles receive the M4 creative LLM derivation. Everything else is
#: a cheap family (radio floor / abstract) and gets NO creative LLM call.
CHARACTER_BEARING_ROLES = frozenset({Role.CHARACTER_VIDEO.value})

_FALLBACK_SETTING = "a vintage radio studio"


def _video_role_for_line(line: dict) -> str:
    role = str((line or {}).get("speaker_role") or "").strip().lower()
    mapped = SPEAKER_TO_VIDEO_ROLE.get(role)
    if mapped is None:
        raise ValueError(
            f"OTR_ShotLock: line "
            f"{str((line or {}).get('line_id') or '?')!r} carries unmapped "
            f"speaker_role {role!r} (known: {tuple(SPEAKER_TO_VIDEO_ROLE)}). "
            f"The 'sfx' role + the background_abstract default were removed "
            f"2026-07-01 (rip-sfx-broll) -- NO FALLBACKS; regenerate the "
            f"episode with the current writer."
        )
    return mapped


# ---------------------------------------------------------------------------
# Brief / cast readers (Meta-brief protocol, never crash on absent brief)
# ---------------------------------------------------------------------------


def _read_setting(meta: dict) -> str:
    """Setting string from the Meta brief, via the brief-reader protocol when
    available; tolerant fallback otherwise."""
    terms = (meta or {}).get("story_brief_terms") or {}
    setting = []
    if isinstance(terms, dict):
        raw = terms.get("setting") or []
        if isinstance(raw, list):
            setting = [str(t).strip() for t in raw if str(t).strip()]
    if not setting:
        try:
            from ._otr_brief_reader import _read_brief_field

            raw = _read_brief_field(meta, "setting", default=[])
            if isinstance(raw, list):
                setting = [str(t).strip() for t in raw if str(t).strip()]
            elif isinstance(raw, str) and raw.strip():
                setting = [raw.strip()]
        except Exception:  # noqa: BLE001
            pass
    return ", ".join(setting[:2]) if setting else _FALLBACK_SETTING


def _appearance_for_char(ledger: dict, char_id: str) -> str:
    """Appearance LOOKUP by char_id (alias-safe), never by display name."""
    if not char_id:
        return ""
    try:
        from . import _otr_ledger_consumers as _OTRLC

        entry = _OTRLC.cast_lookup(ledger, char_id)
    except Exception:  # noqa: BLE001
        entry = {}
        for c in (ledger or {}).get("cast") or []:
            if isinstance(c, dict) and str(c.get("char_id") or "") == str(char_id):
                entry = c
                break
    # character_description added 2026-06-10 (operator look-QA): the writer's
    # RICH per-character physical description lives under that key on the
    # cast row; without it the M4 prompts lost the character grounding.
    base = ""
    for key in ("portrait_prompt", "appearance", "description",
                "character_description"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            base = val.strip()
            break
    if not base:
        name = entry.get("name")
        base = str(name) if name else ""
    # Outfit LOCK (opt-in OTR_OUTFIT_LOCK=1; default OFF -> base unchanged). The writer's
    # description carries no clothing, so FLUX drifts the outfit per beat. When ON, the
    # wardrobe is LLM-generated from the character + story brief and locked per character.
    # Import is guarded (a missing module never breaks appearance), but a WardrobeError is
    # allowed to propagate LOUD by design (operator: no silent generic fallback).
    try:
        from ._otr_wardrobe import apply_wardrobe
    except Exception:  # noqa: BLE001 -- module absent -> no wardrobe, appearance unchanged
        return base
    base = apply_wardrobe(base, char_id, ledger)
    return base


def _content_hash(obj) -> str:
    try:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:  # noqa: BLE001
        blob = repr(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Beat extraction + audio-derived clip budget (Smart Generation Limit)
# ---------------------------------------------------------------------------


def overlay_audio_timing(ledger: dict) -> dict:
    """When the (pre-audio frozen) input ledger's lines carry NO audio timing,
    overlay per-line ``dur_s``/``start_s``/``samples``/``sample_rate`` (+ any
    ``*_wav_path``) from the NEWEST on-disk OTR ledger. The audio path persists
    the real timing there while the ledger is still ``pending_*`` (the same disk
    contract SceneSequencer/AudioEnhance/EpisodeAssembler use). ShotLock is gated
    on ``audio_done`` so the timing exists by the time this runs. Fail-soft +
    test-mode-skipped: no disk ledger -> the input ledger is returned unchanged.
    Without this the audio-derived clip budget is all-zeros (the frozen
    ``script_json`` from the freeze cascade is pre-audio)."""
    import os
    if os.environ.get("OTR_TEST_MODE") == "1":
        return ledger                       # CPU tests never read disk state
    lines = ledger.get("lines") or []
    if not lines:
        return ledger
    if any(isinstance(ln, dict) and (ln.get("dur_s") or ln.get("duration_s")
           or ln.get("samples") or ln.get("audio_samples")) for ln in lines):
        return ledger                       # input already carries timing
    try:
        import json
        from pathlib import Path
        from . import _otr_ledger as _OL
        roots = []
        try:
            from . import _otr_paths as _OP
            roots.append(Path(_OP.otr_episodes_root()))
        except Exception:                    # noqa: BLE001
            base = os.environ.get("OTR_OUTPUT_DIR") or "."
            roots.append(Path(base) / "otr" / "episodes")
        p = _OL.find_most_recent_ledger(roots)
        if not p:
            return ledger
        disk = json.loads(Path(p).read_text(encoding="utf-8"))
        dmap = {str(dl.get("line_id")): dl for dl in (disk.get("lines") or [])
                if isinstance(dl, dict) and dl.get("line_id")}
        tkeys = ("dur_s", "duration_s", "start_s", "samples", "audio_samples", "sample_rate")
        for ln in lines:
            if not isinstance(ln, dict):
                continue
            d = dmap.get(str(ln.get("line_id") or ""))
            if not d:
                continue
            for k in tkeys:
                if ln.get(k) in (None, "") and d.get(k) not in (None, ""):
                    ln[k] = d[k]
            for k, v in d.items():
                if str(k).endswith("wav_path") and v and not ln.get(k):
                    ln[k] = v
        log.info("[OTR_ShotLock] audio-timing overlay from %s", p.name)
    except Exception as exc:                 # noqa: BLE001 - never block the lock
        log.warning("[OTR_ShotLock] audio-timing overlay skipped: %s", exc)
    return ledger


def extract_beats(ledger: dict) -> list:
    """Ordered, non-skipped beats from the frozen ledger.

    One beat per ledger line. Each beat carries its video role, char_id, text,
    and whatever audio timing the frozen ledger stamped (samples + sample_rate
    preferred; ``dur_s`` seconds as a fallback). Never raises on a sparse line.
    """
    try:
        from . import _otr_ledger_consumers as _OTRLC

        lines = list(_OTRLC.iter_lines(ledger))
    except Exception:  # noqa: BLE001
        lines = [
            ln for ln in (ledger or {}).get("lines") or []
            if isinstance(ln, dict) and not ln.get("skip")
        ]
    # Round 5 F5: lines may carry a NAME-scheme char_id (the announcer lines
    # stamp 'announcer') while the cast table + portrait index key by row id
    # (c01..). Normalize at the JOIN -- on the BEAT row only (frozen line rows
    # are never touched): an unknown char_id that case-insensitively matches a
    # cast row's name resolves to that row's char_id.
    cast_rows = [c for c in (ledger or {}).get("cast") or []
                 if isinstance(c, dict)]
    cast_ids = {str(c.get("char_id") or "") for c in cast_rows}
    name_to_id = {str(c.get("name") or "").strip().lower():
                  str(c.get("char_id") or "")
                  for c in cast_rows if c.get("name") and c.get("char_id")}

    def _normalize_char_id(cid: str) -> str:
        if not cid or cid in cast_ids:
            return cid
        return name_to_id.get(cid.strip().lower(), cid)

    id_to_first = {str(c.get("char_id") or ""):
                   (str(c.get("name") or "").split() or [""])[0]
                   for c in cast_rows}
    beats = []
    for i, ln in enumerate(lines):
        if not isinstance(ln, dict):
            continue
        cid = _normalize_char_id(str(ln.get("char_id") or ""))
        text = str(ln.get("text") or "").strip()
        # Round 5 F4 backstop (warn-only -- the ledger is FROZEN here): a
        # talking-head line that still opens with its own speaker's name as
        # a vocative means the writer's attribution repair missed it; the
        # beat renders with the stamped face, so make the miss LOUD.
        first = id_to_first.get(cid, "")
        if (len(first) > 1 and text.lower().startswith(first.lower())
                and re.match(r"^\s*" + re.escape(first) + r"\s*[,!?:;-]",
                             text, flags=re.IGNORECASE)):
            log.warning(
                "[OTR_ShotLock] line %s text opens with its OWN speaker's "
                "name (%s) -- probable mis-attribution shipped from the "
                "writer; the beat renders with char_id=%s's face",
                ln.get("line_id"), first, cid)
        beats.append({
            "beat_id": str(ln.get("line_id") or f"beat_{i:04d}"),
            "role": _video_role_for_line(ln),
            "char_id": cid,
            "text": text,
            "samples": ln.get("samples", ln.get("audio_samples")),
            "sample_rate": ln.get("sample_rate"),
            "dur_s": ln.get("dur_s", ln.get("duration_s")),
        })
    return beats


#: The synthetic OPENING-MUSIC beat id (operator look-QA 2026-06-10): the
#: opening theme plays over the episode head (audio starts at first-line
#: start_s, typically ~8-10s in) but no ledger LINE covers that span, so the
#: head fell to the procgen floor. A synthetic music_visual beat gives the
#: open a REAL rendered scene on the music engine (ltx_video in production).
OPENING_MUSIC_BEAT_ID = "b000_music_open"
_OPENING_MIN_S = 2.0


def derive_opening_music_beat(ledger: dict, fps: int):
    """``(beat, frames)`` for the head-gap opening-music scene, or ``(None, 0)``.

    Reads the FIRST non-skipped line's ``start_s`` from the frozen ledger
    (read-only); a head gap >= 2s earns the synthetic beat. Pure."""
    lines = [ln for ln in (ledger or {}).get("lines") or []
             if isinstance(ln, dict) and not ln.get("skip")]
    if not lines:
        return None, 0
    try:
        first_start = float(lines[0].get("start_s") or 0.0)
    except (TypeError, ValueError):
        return None, 0
    if first_start < _OPENING_MIN_S:
        return None, 0
    frames = int(round(first_start * int(fps or 25)))
    beat = {
        "beat_id": OPENING_MUSIC_BEAT_ID,
        "role": Role.MUSIC_VISUAL.value,
        "char_id": "",
        "text": "",
        "samples": None,
        "sample_rate": None,
        "dur_s": first_start,
        "_synthetic_open": True,
        "_start_s": 0.0,
    }
    return beat, frames


def compute_clip_budget(beats: list, policy: dict, fps: int) -> dict:
    """Audio-derived per-beat ``target_frame_count``.

    Frame counts come from CUMULATIVE audio SAMPLES -- ``frame_at(pos) =
    (pos*fps)//sample_rate`` -- so adjacent beats meet exactly (no double-count,
    no gap). When a beat carries only ``dur_s`` (no samples) it degrades to
    ``round(dur_s*fps)``. Returns ``{per_beat:{beat_id:frames}, total_frames,
    warnings}``. Pure; gated by the caller on ``audio_done``.

    rip-sfx-broll (2026-07-01): the other-beats POOLING budget
    (clip_mode / pool_n / other_beats_render_count) was removed with the
    scene_broll / background_abstract roles -- every beat renders per-beat.
    """
    fps = int(fps) if fps else 25
    warnings: list = []
    sample_rate = 0
    for b in beats:
        sr = b.get("sample_rate")
        if sr:
            sample_rate = int(sr)
            break

    per_beat: dict = {}
    if sample_rate and all(b.get("samples") is not None for b in beats):
        cum = 0
        prev_frame = 0
        for b in beats:
            cum += int(b.get("samples") or 0)
            frame_at = (cum * fps) // sample_rate
            per_beat[b["beat_id"]] = max(0, frame_at - prev_frame)
            prev_frame = frame_at
    else:
        for b in beats:
            dur = b.get("dur_s")
            per_beat[b["beat_id"]] = int(round(float(dur) * fps)) if dur else 0

    total_frames = sum(per_beat.values())

    return {
        "per_beat": per_beat,
        "total_frames": total_frames,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# M4 per-beat creative derivation (LLM -> deterministic-template fallback)
# ---------------------------------------------------------------------------

_DIRECTIVE_KEYS = ("expression", "motion", "camera")


def _deterministic_template(appearance: str, setting: str, beat_text: str) -> str:
    """The collapse-guard fallback prompt (BUG-046: never an empty/generic
    prompt into a render). Deterministic in its inputs."""
    parts = [p for p in (appearance, setting, beat_text) if p]
    return ", ".join(parts) if parts else setting


def _prompt_is_consistent(text_prompt: str, appearance: str, setting: str) -> bool:
    """Schema-level consistency: the prompt must carry the cast's core trait
    token and the brief setting (v1 gate; LLM-as-judge is v2).

    SCOPE (round 5): only ever called for CHARACTER_BEARING_ROLES beats (the
    char_beats loop in derive_creative_directives) -- object-only b-roll
    prompts never pass through here, so the person checks must not be reused
    for non-character roles."""
    if not text_prompt:
        return False
    low = text_prompt.lower()
    appearance_ok = (not appearance) or any(
        tok in low for tok in _core_tokens(appearance)
    )
    setting_ok = (not setting) or any(
        tok in low for tok in _core_tokens(setting)
    )
    return appearance_ok and setting_ok


#: Person-anchor vocabulary (round 5 F3): a talking-head prompt must show the
#: character. Checked on the UNANCHORED candidate (the LLM's contribution),
#: within a bounded head, so the anchor itself can never satisfy the guard.
_PERSON_TOKENS = ("face", "portrait", "speaking", "camera", "close-up",
                  "mid-shot")
_PERSON_GUARD_HEAD = 160


def _person_anchor_ok(text_prompt: str, appearance: str) -> bool:
    """True when the prompt's HEAD (first ``_PERSON_GUARD_HEAD`` chars) carries
    a core appearance token AND a person/framing token -- the b002 lesson: a
    prompt that describes props/scenery without the character lets HuMo walk
    away from the init portrait. Pure; character-bearing beats only."""
    if not text_prompt:
        return False
    head = text_prompt[:_PERSON_GUARD_HEAD].lower()
    appearance_ok = (not appearance) or any(
        tok in head for tok in _core_tokens(appearance)
    )
    person_ok = any(tok in head for tok in _PERSON_TOKENS)
    return appearance_ok and person_ok


def _subject_anchor(appearance: str) -> str:
    """The leading subject clause prepended to EVERY talking-head prompt path
    (round 5 F3): face/framing tokens lead (engines weigh leading tokens
    hardest), the appearance (bounded) follows. Pure."""
    base = "face visible, speaking to camera"
    app = (appearance or "").strip().rstrip(",.;: ")
    return f"{base}, {app[:120].rstrip(', ')}" if app else base


def _core_tokens(text: str) -> list:
    toks = [t.strip(",.;:").lower() for t in str(text).split() if len(t) > 3]
    return toks[:6]


def _parse_directives(raw: str, expected_ids: list) -> dict:
    """Parse a batch LLM reply into ``{beat_id:{expression,motion,camera}}``.

    Returns ``{}`` on empty / unparseable / truncated output (the collapse
    guard's trigger). Accepts a JSON list or object; tolerant of extra keys.
    """
    if not raw or not str(raw).strip():
        return {}
    txt = str(raw).strip()
    # tolerate ```json fences / leading prose: slice to the first bracket.
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = txt.find(opener), txt.rfind(closer)
        if 0 <= i < j:
            try:
                data = json.loads(txt[i:j + 1])
                break
            except (ValueError, TypeError):
                continue
    else:
        return {}
    rows = data if isinstance(data, list) else data.get("beats") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}
    out: dict = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        bid = str(row.get("beat_id") or "")
        if bid and bid in expected_ids:
            parsed = {k: str(row.get(k) or "").strip() for k in _DIRECTIVE_KEYS}
            # An adapter MAY author the full rich prompt; captured here and
            # passed through the consistency gate (a hallucinated text_prompt
            # that drops the cast/setting falls back to the template).
            parsed["text_prompt"] = str(row.get("text_prompt") or "").strip()
            out[bid] = parsed
    return out


def _build_batch_prompt(batch: list, meta: dict, ledger: dict, setting: str) -> str:
    """Compose ONE batched derivation prompt (mirrors the Meta-brief protocol:
    derive from brief + beat + cast; instrumental wording kept model-agnostic)."""
    lines = [
        "You are a film director. For EACH beat below, give a concise "
        "expression, motion, and camera direction that fits the character and "
        "the setting. Reply ONLY with a JSON list of objects "
        '{"beat_id","expression","motion","camera"}.',
        # Gap-audit F3 (2026-06-10): the era/style tails are APPENDED later
        # by the prompt finisher -- the model must not duplicate them.
        "Do not include film-stock, film-grain, or lighting-style terms; "
        "they are appended automatically later.",
        # Round 5 F3 (the b002 no-person catch): any authored text_prompt must
        # keep the character on screen -- props/scenery alone lose the face.
        "If you author a text_prompt, describe the named character as the "
        "VISIBLE subject (face-forward, mid-shot or closer); never describe "
        "scenery or props without the character.",
        f"Setting: {setting}",
        "Beats:",
    ]
    for b in batch:
        appearance = _appearance_for_char(ledger, b["char_id"])
        lines.append(
            json.dumps({
                "beat_id": b["beat_id"],
                "character": appearance[:160],
                "line": b["text"][:240],
            }, ensure_ascii=True)
        )
    return "\n".join(lines)


def derive_creative_directives(
    beats: list,
    meta: dict,
    ledger: dict,
    *,
    llm_fn=None,
    batch_size: int = 15,
    max_reseed: int = 2,
    consistency_gate_warn_only: bool = False,
):
    """Derive per-beat creative directives for character-bearing beats.

    Returns ``(creative_by_beat, warnings)`` where ``creative_by_beat[beat_id]``
    is ``{expression, motion, camera, text_prompt, source, prompt_hash}``.
    Cheap-family beats are skipped entirely (NO llm call). ``llm_fn`` is a
    ``callable(prompt:str) -> str`` (injectable for tests); when None it is
    resolved lazily from the writer's slot and, if unavailable, the deterministic
    template carries every beat. Collapse guard: empty/unparseable/truncated ->
    reseed up to ``max_reseed`` -> template. Never raises; never touches audio.
    """
    warnings: list = []
    char_beats = [b for b in beats if b["role"] in CHARACTER_BEARING_ROLES]
    if not char_beats:
        return {}, warnings

    if llm_fn is None:
        llm_fn = _resolve_writer_llm(meta, warnings)

    setting = _read_setting(meta)
    brief_hash = _content_hash(meta.get("story_brief_terms") or meta.get("story_brief") or {})
    cast_hash = _content_hash(ledger.get("cast") or [])

    creative: dict = {}
    for start in range(0, len(char_beats), max(1, int(batch_size))):
        batch = char_beats[start:start + max(1, int(batch_size))]
        expected = [b["beat_id"] for b in batch]
        directives: dict = {}
        if llm_fn is not None:
            prompt = _build_batch_prompt(batch, meta, ledger, setting)
            for attempt in range(max_reseed + 1):
                try:
                    raw = llm_fn(prompt)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"derivation llm_fn raised ({exc}); reseed {attempt}")
                    raw = ""
                directives = _parse_directives(raw, expected)
                if directives:
                    break
                if attempt < max_reseed:
                    warnings.append(
                        f"empty/unparseable derivation for batch "
                        f"{expected[:1]}..; reseed {attempt + 1}/{max_reseed}"
                    )
        for b in batch:
            appearance = _appearance_for_char(ledger, b["char_id"])
            d = directives.get(b["beat_id"]) or {}
            llm_text = d.get("text_prompt", "")
            has_directives = any(d.get(k) for k in _DIRECTIVE_KEYS)
            if llm_text:
                text_prompt = llm_text
                source = "llm"
            elif has_directives:
                text_prompt = ", ".join(
                    p for p in (
                        appearance, setting, b["text"],
                        d.get("expression"), d.get("motion"), d.get("camera"),
                    ) if p
                )
                source = "llm"
            else:
                text_prompt = _deterministic_template(appearance, setting, b["text"])
                source = "template"
                d = {k: "" for k in _DIRECTIVE_KEYS}
            # Round 5 F3: BOTH gates run on the UNANCHORED candidate (the
            # anchor would otherwise satisfy them tautologically -- pass03
            # panel catch). The person gate binds the AUTHORED freeform
            # text_prompt path only (the b002 lesson: an action/prop-only
            # authored prompt lets HuMo abandon the init portrait). The
            # composed-directives path leads with the appearance by
            # construction, and the deterministic template IS the sanctioned
            # fallback -- both get the anchor regardless.
            _person_ok = (not llm_text
                          or _person_anchor_ok(text_prompt, appearance))
            if not (_prompt_is_consistent(text_prompt, appearance, setting)
                    and _person_ok):
                level = "WARN" if consistency_gate_warn_only else "FAIL-CLOSED"
                warnings.append(
                    f"consistency gate {level} for beat {b['beat_id']}: prompt "
                    f"missing cast/setting trait or person anchor; using "
                    f"template fallback"
                )
                text_prompt = _deterministic_template(appearance, setting, b["text"])
                source = "template_consistency"
            # The subject anchor leads EVERY talking-head prompt path (llm,
            # composed, template): face/framing tokens first, bounded
            # appearance after -- prepended AFTER the gates, BEFORE finishing.
            text_prompt = f"{_subject_anchor(appearance)}, {text_prompt}"
            # FINISH the prompt (gap-audit F3, 2026-06-10): era tail (brief
            # atmosphere/palette/lighting) + the film style tail, restored
            # from the deleted legacy composer. MUST run after the
            # consistency gate and BEFORE prompt_hash so the stored hash
            # matches the rendered prompt. Fail-soft.
            try:
                try:
                    from ._otr_story_brief_helpers import (  # type: ignore
                        finish_visual_prompt)
                except ImportError:  # pragma: no cover -- flat test imports
                    from _otr_story_brief_helpers import (  # type: ignore
                        finish_visual_prompt)
                text_prompt = finish_visual_prompt(meta, text_prompt)
            except Exception:  # noqa: BLE001
                pass
            creative[b["beat_id"]] = {
                "expression": d.get("expression", ""),
                "motion": d.get("motion", ""),
                "camera": d.get("camera", ""),
                "text_prompt": text_prompt,
                "source": source,
                "request_hash": _content_hash(
                    [brief_hash, cast_hash, b["beat_id"], b["char_id"]]
                ),
                "prompt_hash": _content_hash(text_prompt),
            }
    return creative, warnings


def _resolve_writer_llm(meta: dict, warnings: list):
    """Best-effort writer-slot LLM resolver (V-11: no new model_id widget --
    the model name comes from the ledger meta the writer stamped). Returns a
    ``callable(prompt)->str`` or None. Fails soft to None in headless/test mode
    so the deterministic template carries the episode (the live wiring lands
    with the M4 GPU gate before CW-6)."""
    import os

    if os.environ.get("OTR_TEST_MODE") == "1":
        return None
    model_id = ""
    if isinstance(meta, dict):
        model_id = str(
            meta.get("technical_model") or meta.get("creative_writing_model") or ""
        )
    if not model_id:
        warnings.append("no writer model in meta; creative derivation uses template")
        return None
    try:  # lazy: never import the loader at module scope (V-12)
        # FIXED 2026-06-10 (operator look-QA root cause): this called
        # make_generate_fn(model_id, slot=...) -- a signature that never
        # existed -- so the LLM path failed on EVERY live run and the
        # deterministic template silently carried all creative/image
        # derivation. The real seam is request_slot(slot, model_id) ->
        # cache entry (a same-model call is a cache HIT, no reload) ->
        # make_generate_fn(entry) -> gen(messages, ...).
        from ._otr_model_loader import make_generate_fn, request_slot  # type: ignore

        entry = request_slot("technical", model_id)  # LLM slot: technical
        gen = make_generate_fn(entry)

        def _call(prompt: str) -> str:
            # 0.1 not 0.0: the local HF lane hardcodes do_sample=True and
            # transformers rejects a non-positive temperature (live 30w4
            # catch); 0.1 is near-greedy for short derivation prompts.
            return gen([{"role": "user", "content": str(prompt)}],
                       temperature=0.1, max_new_tokens=300)

        return _call
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"writer LLM unavailable ({exc}); derivation uses template")
        return None


# ---------------------------------------------------------------------------
# Execution plan (groups + shots) -> ledger['video']
# ---------------------------------------------------------------------------


def build_execution_plan(beats, budget, creative, policy):
    """Build DAG-validated ``execution_groups`` + per-shot rows.

    CW-1 emits one consumer group per role that has beats (no base-clip
    providers yet -> no edges). Each shot carries its engine_id (from the
    policy), audio-derived ``target_frame_count``, the creative sidecar, and
    cache_keys that deliberately EXCLUDE ``expression`` (3D expression is a
    driver-channel directive, never a cache/mesh key). Returns ``(groups,
    shots)`` after ``resolver.validate_execution_groups``.
    """
    video_models = (policy or {}).get("video_models") or {}

    def engine_for(role):
        # Route-A: per-role video slot with the legacy other_beats fallback
        # (ONE shared map; nodes/_otr_shared/role_slots.py).
        return _role_slots.engine_id_for_role(video_models, role)

    roles_present = []
    for b in beats:
        if b["role"] not in roles_present:
            roles_present.append(b["role"])

    groups = [{
        "group_id": f"grp_{role}",
        "kind": "consumer",
        "engine_id": engine_for(role),
        "profile_id": "",
        "depends_on": [],
        "produces_base_for": [],
    } for role in roles_present]
    groups = _resolver.validate_execution_groups(groups)

    # rip-sfx-broll (2026-07-01): the pool_n_loop still/clip POOLING died with
    # the scene_broll / background_abstract roles -- every beat renders
    # per-beat with its own scene still (no still_pool_key stamping).
    shots = []
    for b in beats:
        cre = creative.get(b["beat_id"], {})
        _timing = ({"start_s": b.get("_start_s", 0.0), "dur_s": b.get("dur_s")}
                   if b.get("_synthetic_open") else None)
        shots.append({
            "shot_id": f"shot_{b['beat_id']}",
            # Synthetic beats have no ledger LINE, so the shot row itself
            # carries the timeline position (the render driver falls back to
            # it when the line lookup is empty).
            **({"start_s": _timing["start_s"], "dur_s": _timing["dur_s"]}
               if _timing else {}),
            "source_line_ids": [] if b.get("_synthetic_open")
            else [b["beat_id"]],
            "group_id": f"grp_{b['role']}",
            # The shot's video ROLE, stamped explicitly (2026-06-10): the
            # render driver's role-scoped behaviors (the LTX radio-open
            # prompt) read it; before this only group_id embedded the role.
            "role": b["role"],
            # Round 5 F5: the NORMALIZED char_id rides the shot row so the
            # render driver's portrait join never depends on the raw line
            # scheme (the announcer's 'announcer' -> cast row id case).
            "char_id": b.get("char_id", ""),
            "engine_id": engine_for(b["role"]),
            "profile_id": "",
            "family": "",
            # Schema-stable constant post-pooling-rip (every beat is per-beat).
            "strategy": {"mode": "unique_per_beat"},
            "request_seed": 0,
            "target_frame_count": int(budget["per_beat"].get(b["beat_id"], 0)),
            "render_request_hash": cre.get("request_hash", ""),
            "binding_hash": "",
            # cache_keys EXCLUDE expression on purpose (V-7 / PASS-M4): the
            # expression is a driver-channel directive, not part of identity.
            "cache_keys": {
                "prompt_hash": cre.get("prompt_hash", ""),
                "request_hash": cre.get("request_hash", ""),
            },
            "degradation_trail": [],
            "creative": {k: v for k, v in cre.items() if k != "request_hash"},
        })
    return groups, shots


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


class OTRShotLock:
    """Registered as ``OTR_ShotLock``. Single ``ledger['video']`` authority."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "lock"
    # episode_id output is ADDITIVE (still-spine ST-6 / DS-3): ShotLock holds
    # the audio-overlaid ledger, so it is the in-graph episode_id authority;
    # the saved json wires it into OTR_ImageGenDispatcher.episode_id so every
    # still lands in episodes/<ep>/stills/. Existing slot indexes unchanged.
    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("patched_ledger_json", "video_revision", "shot_report",
                    "done", "episode_id")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "forceInput": True,
                    "tooltip": (
                        "Frozen ledger JSON (OTR_LedgerFreezeCascade out1 "
                        "script_json). ShotLock stamps a video section into it."
                    ),
                }),
                "audio_done": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Audio-done gate (OTR_EpisodeAssembler out3). Wiring it "
                        "orders ShotLock AFTER audio timing freezes so the clip "
                        "budget is bound against the real timeline. Opaque STRING."
                    ),
                }),
                "video_policy_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "forceInput": True,
                    "tooltip": "Per-role selection policy from OTR_VideoDirector.",
                }),
            },
            "optional": {
                "image_done": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Image-done gate (mirrors audio_done). Declared to "
                        "freeze the image-before-video contract; NON-BLOCKING "
                        "in v1 (Flux gen-1 runs in-process, nothing emits it "
                        "yet; C1 wires the emitter). Opaque STRING."
                    ),
                }),
                "consistency_gate_warn_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "M4 story-consistency gate: warn-only vs fail-closed on "
                        "a missing cast/setting trait. Either way the episode "
                        "still renders (template fallback); never aborts."
                    ),
                }),
                "gate_in": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    # ------------------------------------------------------------------ #
    def lock(self, script_json, audio_done="", video_policy_json="{}",
             image_done="", consistency_gate_warn_only=False, gate_in=""):
        from . import _otr_ledger_consumers as _OTRLC

        led = _OTRLC.load_ledger(script_json)
        led = overlay_audio_timing(led)     # fill per-line timing from the post-audio disk ledger
        meta = led.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            led["meta"] = meta
        try:
            policy = json.loads(video_policy_json or "{}")
            if not isinstance(policy, dict):
                policy = {}
        except (ValueError, TypeError):
            policy = {}

        canvas = (policy.get("canvas") or {})
        fps = int(canvas.get("fps") or 25)
        report: list = []
        warnings: list = []

        # Brief disposition, ONCE per run (gap-audit G4 restore).
        try:
            try:
                from ._otr_story_brief_helpers import (  # type: ignore
                    log_story_brief_disposition)
            except ImportError:  # pragma: no cover
                from _otr_story_brief_helpers import (  # type: ignore
                    log_story_brief_disposition)
            log_story_brief_disposition(meta, "shotlock_m4", log)
        except Exception:  # noqa: BLE001
            pass

        beats = extract_beats(led)
        budget = compute_clip_budget(beats, policy, fps)
        warnings.extend(budget.get("warnings", []))

        # The OPENING-MUSIC scene (operator look-QA 2026-06-10): injected
        # AFTER the budget so the real beats keep their exact cumulative-
        # samples frame math; the synthetic head beat adds its own frames.
        _open_beat, _open_frames = derive_opening_music_beat(led, fps)
        if _open_beat is not None and _open_frames > 0:
            beats.insert(0, _open_beat)
            budget["per_beat"][OPENING_MUSIC_BEAT_ID] = _open_frames
            budget["total_frames"] = int(budget.get("total_frames") or 0) \
                + _open_frames
            report.append(
                "opening-music scene injected: %d frames (head 0..%.2fs) on "
                "the music_visual engine" % (_open_frames,
                                             _open_frames / max(1, fps)))

        creative, cre_warn = derive_creative_directives(
            beats, meta, led,
            consistency_gate_warn_only=bool(consistency_gate_warn_only),
        )
        warnings.extend(cre_warn)

        groups, shots = build_execution_plan(beats, budget, creative, policy)

        revision = int(meta.get("video_revision") or 0) + 1
        video_section = {
            "video_revision": revision,
            "canonical_canvas": {
                "w": int(canvas.get("w") or 832),
                "h": int(canvas.get("h") or 480),
            },
            "fps": fps,
            "locked_against_audio_rev": str(meta.get("audio_revision") or ""),
            "execution_groups": groups,
            "roles": policy.get("video_models") or {},
            "shots": shots,
            "clip_budget": {
                "total_frames": budget["total_frames"],
            },
            "warnings": warnings,
        }
        led["video"] = video_section
        meta["video_revision"] = revision

        report.append(f"shot_lock_revision={revision} beats={len(beats)} shots={len(shots)}")
        report.append(
            f"clip_budget: total_frames={budget['total_frames']}"
        )
        report.append(f"execution_groups={[g['group_id'] for g in groups]}")
        for w in warnings:
            report.append(f"WARN: {w}")
            log.warning("[OTR_ShotLock] %s", w)

        patched = json.dumps(led, ensure_ascii=True, separators=(",", ":"))
        done = f"shot_lock:done:rev={revision}"
        episode_id = str(led.get("episode_id")
                         or (led.get("meta") or {}).get("episode_id") or "")
        return (patched, int(revision), "\n".join(report), done, episode_id)
