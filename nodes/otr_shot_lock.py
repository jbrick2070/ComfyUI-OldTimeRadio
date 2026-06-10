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

log = logging.getLogger("OTR")

from ._otr_shared import resolver as _resolver
from ._otr_shared.role_compat import Role

# ---------------------------------------------------------------------------
# Role mapping + which roles are "character-bearing" (get the rich derivation)
# ---------------------------------------------------------------------------

#: ledger ``speaker_role`` -> video role.
SPEAKER_TO_VIDEO_ROLE = {
    "announcer": Role.ANNOUNCER_VISUAL.value,
    "music": Role.MUSIC_VISUAL.value,
    "music_open": Role.MUSIC_VISUAL.value,
    "music_close": Role.MUSIC_VISUAL.value,
    "music_inter": Role.MUSIC_VISUAL.value,
    "char_voice": Role.CHARACTER_VIDEO.value,
    "dialogue": Role.CHARACTER_VIDEO.value,
}
_DEFAULT_VIDEO_ROLE = Role.BACKGROUND_ABSTRACT.value

#: Only these roles receive the M4 creative LLM derivation. Everything else is
#: a cheap family (radio floor / abstract) and gets NO creative LLM call.
CHARACTER_BEARING_ROLES = frozenset({Role.CHARACTER_VIDEO.value})

_FALLBACK_SETTING = "a vintage radio studio"


def _video_role_for_line(line: dict) -> str:
    role = str((line or {}).get("speaker_role") or "").strip().lower()
    return SPEAKER_TO_VIDEO_ROLE.get(role, _DEFAULT_VIDEO_ROLE)


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
    for key in ("portrait_prompt", "appearance", "description",
                "character_description"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    name = entry.get("name")
    return str(name) if name else ""


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
    beats = []
    for i, ln in enumerate(lines):
        if not isinstance(ln, dict):
            continue
        beats.append({
            "beat_id": str(ln.get("line_id") or f"beat_{i:04d}"),
            "role": _video_role_for_line(ln),
            "char_id": str(ln.get("char_id") or ""),
            "text": str(ln.get("text") or "").strip(),
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
    """Audio-derived per-beat ``target_frame_count`` + the Other-Beats budget.

    Frame counts come from CUMULATIVE audio SAMPLES -- ``frame_at(pos) =
    (pos*fps)//sample_rate`` -- so adjacent beats meet exactly (no double-count,
    no gap). When a beat carries only ``dur_s`` (no samples) it degrades to
    ``round(dur_s*fps)``. Other-beats ``pool_n_loop`` clamps N to the number of
    other-beats and WARNs. Returns ``{per_beat:{beat_id:frames}, total_frames,
    other_beats_render_count, clip_mode, warnings}``. Pure; gated by the caller
    on ``audio_done``.
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

    other = [
        b for b in beats
        if b["role"] in (Role.BACKGROUND_ABSTRACT.value, Role.SCENE_BROLL.value)
    ]
    clip_mode = ((policy or {}).get("other_beats") or {}).get("clip_mode") \
        or "unique_per_beat"
    pool_n = int(((policy or {}).get("other_beats") or {}).get("pool_n") or 0)
    if clip_mode == "pool_n_loop":
        render_count = min(pool_n, len(other)) if other else 0
        if pool_n > len(other):
            warnings.append(
                f"pool_n={pool_n} exceeds other-beats count {len(other)}; "
                f"clamped to {render_count} (no over-generation)"
            )
    else:
        render_count = len(other)

    return {
        "per_beat": per_beat,
        "total_frames": total_frames,
        "other_beats_render_count": render_count,
        "clip_mode": clip_mode,
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
    token and the brief setting (v1 gate; LLM-as-judge is v2)."""
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
            if not _prompt_is_consistent(text_prompt, appearance, setting):
                level = "WARN" if consistency_gate_warn_only else "FAIL-CLOSED"
                warnings.append(
                    f"consistency gate {level} for beat {b['beat_id']}: prompt "
                    f"missing cast/setting trait; using template fallback"
                )
                text_prompt = _deterministic_template(appearance, setting, b["text"])
                source = "template_consistency"
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
        from ._otr_model_loader import make_generate_fn  # type: ignore

        gen = make_generate_fn(model_id, slot="technical")

        def _call(prompt: str) -> str:
            return gen(prompt, temperature=0.0)

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
    role_to_slot = {
        Role.ANNOUNCER_VISUAL.value: "announcer_video_model",
        Role.MUSIC_VISUAL.value: "music_video_model",
        Role.CHARACTER_VIDEO.value: "other_beats_video_model",
        Role.SCENE_BROLL.value: "other_beats_video_model",
        Role.BACKGROUND_ABSTRACT.value: "other_beats_video_model",
    }

    def engine_for(role):
        slot = role_to_slot.get(role, "other_beats_video_model")
        entry = video_models.get(slot)
        if isinstance(entry, dict):
            return entry.get("engine_id") or ""
        return str(entry or "")

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
            "engine_id": engine_for(b["role"]),
            "profile_id": "",
            "family": "",
            "strategy": {"mode": budget.get("clip_mode", "unique_per_beat")},
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
    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("patched_ledger_json", "video_revision", "shot_report", "done")
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
                "other_beats_render_count": budget["other_beats_render_count"],
                "clip_mode": budget["clip_mode"],
            },
            "warnings": warnings,
        }
        led["video"] = video_section
        meta["video_revision"] = revision

        report.append(f"shot_lock_revision={revision} beats={len(beats)} shots={len(shots)}")
        report.append(
            f"clip_budget: total_frames={budget['total_frames']} "
            f"other_beats_render={budget['other_beats_render_count']} "
            f"mode={budget['clip_mode']}"
        )
        report.append(f"execution_groups={[g['group_id'] for g in groups]}")
        for w in warnings:
            report.append(f"WARN: {w}")
            log.warning("[OTR_ShotLock] %s", w)

        patched = json.dumps(led, ensure_ascii=True, separators=(",", ":"))
        done = f"shot_lock:done:rev={revision}"
        return (patched, int(revision), "\n".join(report), done)
