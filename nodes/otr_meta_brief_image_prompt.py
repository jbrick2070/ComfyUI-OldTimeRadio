"""OTR_MetaBriefImagePromptGen -- LLM image prompts from the Meta brief (C1).

The image-side mirror of ``OTR_ShotLock``'s per-beat creative derivation and of
``nodes/_otr_music_prompt.py``: every portrait/scene prompt is composed from the
propagating Meta brief (setting / period / mood) + the character's appearance,
optionally refined by ONE LLM call on the writer's slot (V-11, no new model_id
widget) at ``temperature=0`` with the ``prompt_hash`` taken AFTER the call.

Collapse guard (PASS-IMG SHOULD-FIX + BUG-099): empty / unparseable LLM output
-> reseed up to ``max_reseed`` -> a DETERMINISTIC brief-composed template that is
NEVER empty (a generic portrait must never ship a generic mesh silently, so we
WARN, but we never abort the episode or emit an empty prompt). Appearance is
looked up by ``char_id`` (BUG-098), never the display name.

Story-consistency gate (v1 = a SCHEMA assertion, not a 2nd LLM call): the final
prompt MUST carry the character's appearance token + the brief setting; a
hallucinated / missing trait -> WARN + fall back to the template
(``consistency_gate_warn_only`` toggles fail-closed vs warn on the hard case).

PURE core (``compose_image_prompt_fallback`` / ``derive_image_prompts``): no I/O,
no GPU, no engine imports -- the LLM is injected (tests) or resolved lazily from
the writer slot. Cold-import clean. UTF-8 no BOM, ASCII-only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re

log = logging.getLogger("OTR")

#: Stopwords ignored when checking brief-grounding (so "a"/"the" never count).
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "and", "with", "in", "on", "at", "to", "for",
    "from", "into", "setting", "portrait", "style", "studio",
})


def _significant_words(s: str) -> set:
    """Significant (len>=4, non-stopword) lowercase words in ``s``."""
    return {w for w in re.findall(r"[a-z]{4,}", (s or "").lower())
            if w not in _STOPWORDS}


def _content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_setting(meta: dict) -> str:
    """Brief setting string (mirrors the music-prompt brief read; fail-soft)."""
    terms = (meta or {}).get("story_brief_terms")
    if not isinstance(terms, dict):
        terms = {}
    setting_raw = terms.get("setting") or []
    if not isinstance(setting_raw, list):
        setting_raw = []
    setting = [str(t).strip() for t in setting_raw if str(t).strip()]
    return ", ".join(setting[:2])


def _appearance_for_char(cast: list, char_id: str) -> str:
    """Appearance text looked up by char_id ONLY (BUG-098), never display name.

    Key chain includes ``character_description`` (operator look-QA
    2026-06-10): the writer stamps the RICH per-character physical
    description on the cast row as ``character_description`` (the
    ``portrait_prompt`` mirror lives under meta.visual_plan keyed by NAME,
    not on the row), so without this key every character fell to the same
    generic setting+anchor fallback -> ONE shared portrait for the whole
    cast, styled as an actor in a radio booth."""
    cid = str(char_id or "")
    for c in cast or []:
        if isinstance(c, dict) and str(c.get("char_id") or "") == cid:
            return str(c.get("portrait_prompt") or c.get("appearance")
                       or c.get("character_description") or "").strip()
    return ""


#: Shared portrait style anchor. Reworded 2026-06-10 (operator look-QA): the
#: old "studio portrait, neutral lighting" framing read as an ACTOR in a
#: recording booth; portraits must show the CHARACTER in character, in the
#: story's world -- never a voice actor at a microphone.
# Round 5 operator notes (2026-06-10): the wider framing is a KEEPER ("this
# week's portraits show more body -- better"), so it is now intentional
# (three-quarter, not head-and-shoulders). The old "no microphone, not a
# recording studio" NEGATIONS are gone -- negative phrasing PLANTS the tokens
# in the image embedding (the c01 giant-mic catch); gear words are instead
# scrubbed from the OUTPUT (see _GEAR_WORDS) and banned in the instruction.
STYLE_ANCHOR = ("in-character cinematic three-quarter portrait, face clearly "
                "visible, period-accurate costume and environment, dramatic "
                "film lighting")

#: The station ANNOUNCER is a synthetic, non-cast portrait subject (CastLock
#: owns ``ledger['cast']``; the announcer is the station voice, never a cast
#: row). Announcer beats are talking beats, so HuMo needs an ``init_image``
#: for them exactly like character beats -- without one the intro/outro
#: starve to the still floor (the b001/b005 keystone gap). The pseudo-id
#: matches the ``char_id`` the writer stamps on announcer lines.
ANNOUNCER_CHAR_ID = "announcer"

#: Radio-style announcer portrait anchor (operator-directed 2026-06-09:
#: "announcer should get a 'radio' style image"). A human face stays in
#: frame so the audio_driven_face family can drive the mouth; the styling
#: reads unmistakably as period radio.
ANNOUNCER_PORTRAIT_ANCHOR = (
    "vintage 1940s radio announcer at a large chrome ribbon microphone, "
    "suit and tie, art deco broadcast studio backdrop, ON AIR sign glow, "
    "warm tube lighting"
)


#: Portrait canvas (the proven FLUX portrait dims; unchanged by the spine).
PORTRAIT_W = 832
PORTRAIT_H = 1216


def _landscape_still_dims():
    """(w, h) for SCENE stills: the landscape composite canvas, each dim
    snapped DOWN to /32 (the latent-grid contract). Env-overridable via
    OTR_VIDEO_LANDSCAPE_CANVAS (the same knob the render driver reads)."""
    import os
    raw = os.environ.get("OTR_VIDEO_LANDSCAPE_CANVAS", "1472x832")
    try:
        w, h = (int(x) for x in raw.lower().split("x", 1))
    except (ValueError, AttributeError):
        w, h = 1472, 832
    return max(32, (w // 32) * 32), max(32, (h // 32) * 32)


def _iter_beat_lines(lines):
    """(beat_id, line) pairs mirroring OTR_ShotLock's beat-id scheme exactly
    (line_id or beat_%04d over the NON-SKIPPED lines) so a still minted here
    joins the ShotLock shot rows downstream. Pure."""
    live = [ln for ln in (lines or [])
            if isinstance(ln, dict) and not ln.get("skip")]
    for i, ln in enumerate(live):
        yield str(ln.get("line_id") or f"beat_{i:04d}"), ln


def derive_scene_still_targets(lines, fps: int = 25):
    """Still-spine ST-2: the v1 SCENE-STILL targets -- open + announcer +
    outro ONLY (panel cut: not every beat) -- derived from the LINES via pure
    helpers, never from ``video.shots`` (graph order: image gen runs BEFORE
    ShotLock). Returns ``(targets, warnings)``; each target is
    ``{beat_id, kind, role, source}``.

    The OPEN comes from the same pure helper ShotLock uses
    (``derive_opening_music_beat``). That helper needs the first line's
    ``start_s`` -- which the audio path persists to the DISK ledger, not to
    this node's pre-audio ``script_json``. When timing is UNKNOWN (first
    line carries no ``start_s``) the open target is still emitted
    (``source="scene_pretiming"``, warned LOUD): production always opens on
    the music head gap, and an unused still costs one render while a
    MISSING open still costs the 6/5 look.
    """
    warnings: list = []
    targets: list = []
    seen: set = set()

    def _add(beat_id, kind, role, source):
        if beat_id and beat_id not in seen:
            seen.add(beat_id)
            targets.append({"beat_id": beat_id, "kind": kind,
                            "role": role, "source": source})

    try:  # lazy: one source of truth for the open beat + the role map
        from .otr_shot_lock import (
            OPENING_MUSIC_BEAT_ID, SPEAKER_TO_VIDEO_ROLE,
            derive_opening_music_beat)
    except ImportError:  # pragma: no cover -- flat test imports
        from otr_shot_lock import (  # type: ignore
            OPENING_MUSIC_BEAT_ID, SPEAKER_TO_VIDEO_ROLE,
            derive_opening_music_beat)

    live = [ln for _bid, ln in _iter_beat_lines(lines)]
    beat, _frames = derive_opening_music_beat({"lines": list(lines or [])},
                                              int(fps or 25))
    if beat is not None:
        _add(str(beat.get("beat_id") or OPENING_MUSIC_BEAT_ID),
             "scene_open", str(beat.get("role") or "music_visual"),
             "scene_timed")
    elif live and live[0].get("start_s") is None:
        warnings.append(
            "scene_open b000: line timing absent (pre-audio ledger); "
            "emitting the open still target OPTIMISTICALLY -- production "
            "opens on the music head gap; an unused still is cheap, a "
            "missing open still loses the 6/5 look")
        _add(OPENING_MUSIC_BEAT_ID, "scene_open", "music_visual",
             "scene_pretiming")

    scene_roles = ("announcer_visual", "music_visual")
    # EVERY i2v beat must carry its OWN scene still (operator 2026-06-12: "my
    # stills ARE the look; every i2v beat must have its still"). The v1 panel
    # cut (open + FIRST announcer + LAST scene only) left MIDDLE announcer/music
    # beats (e.g. b003) with NO still -> a silent text-only LTX i2v fallback,
    # which is exactly the flat/unanchored beat the operator caught. Cover ALL
    # announcer/music beats now (the open b000 is already added above; _add
    # dedupes via `seen`). The per-beat visual variety the motion-prompt design
    # relies on comes from these per-beat stills.
    for bid, ln in _iter_beat_lines(lines):
        role = SPEAKER_TO_VIDEO_ROLE.get(
            str(ln.get("speaker_role") or "").strip().lower(), "")
        if role in scene_roles:
            _add(bid, "scene_beat", role, "scene_role_map")
    return targets, warnings


def objects_by_id(payload) -> dict:
    """``{object_id: object}`` accessor over the versioned ``{"objects":[...]}``
    payload (portrait object_ids are the char_ids). Pure, tolerant."""
    out: dict = {}
    for obj in (payload or {}).get("objects") or []:
        if isinstance(obj, dict) and obj.get("object_id"):
            out.setdefault(str(obj["object_id"]), obj)
    return out


def announcer_line_char_ids(lines) -> list:
    """Distinct ``char_id``s of ledger lines spoken by the ANNOUNCER role, in
    first-appearance order (normally just ``["announcer"]``). The video render
    path resolves ``init_image`` by the LINE's char_id, so prompts are keyed
    the same way. Pure; tolerates malformed rows."""
    out: list = []
    for ln in lines or []:
        if not isinstance(ln, dict):
            continue
        if str(ln.get("speaker_role") or "") != "announcer":
            continue
        cid = str(ln.get("char_id") or "") or ANNOUNCER_CHAR_ID
        if cid not in out:
            out.append(cid)
    return out


def compose_image_prompt_fallback(meta: dict, char: dict) -> str:
    """Deterministic brief-composed portrait prompt -- NEVER empty.

    ``"{appearance}, {setting} setting, {style anchor}"`` with empty parts
    dropped; degrades to the style anchor alone if the brief + cast are bare.
    """
    # Same key chain as _appearance_for_char incl. character_description
    # (2026-06-10): this fallback is what actually runs whenever the LLM is
    # unavailable, and it read only the two empty keys -- every character got
    # the identical setting+anchor prompt -> ONE shared portrait.
    appearance = str(
        (char or {}).get("portrait_prompt") or (char or {}).get("appearance")
        or (char or {}).get("character_description") or ""
    ).strip()
    setting = _read_setting(meta)
    parts = []
    if appearance:
        parts.append(appearance)
    if setting:
        parts.append(f"{setting} setting")
    parts.append(STYLE_ANCHOR)
    return ", ".join(parts)


def _build_char_prompt_request(char: dict, meta: dict, setting: str) -> str:
    """The instruction handed to the writer LLM (temp=0) for one character."""
    appearance = _appearance_for_char([char], str(char.get("char_id") or ""))
    return (
        "Write ONE vivid still-image portrait prompt (a single comma-separated "
        "line, no preamble) for this character. The image MUST depict the "
        "CHARACTER THEMSELVES -- a person with a clearly visible face, "
        "three-quarter framing showing head and upper body -- IN CHARACTER "
        "inside the story's world. NEVER an empty room, an object, or scenery "
        "alone. Ground it in the appearance and the story setting; keep it "
        "photographic and period-consistent.\n"
        f"character_appearance: {appearance or '(unspecified)'}\n"
        f"story_setting: {setting or '(unspecified)'}\n"
        f"style_anchor: {STYLE_ANCHOR}\n"
        "Do not include film-stock, film-grain, or lighting-style terms; "
        "they are appended automatically later.\n"
        "Do not mention radios, microphones, studios, or any broadcasting "
        "equipment anywhere in the prompt -- the character is a person in "
        "the STORY's world, not a performer at a station.\n"
        "Return only the prompt line."
    )


#: Person-evidence vocabulary for the portrait guard: an accepted prompt that
#: matches NONE of these almost certainly depicts scenery/objects (the
#: "microphone, no person" live catch, look-QA round 4) -> template fallback.
_PERSON_WORDS = re.compile(
    r"\b(face|faces|person|man|woman|portrait|eyes|hair|head|gentleman|lady|"
    r"his|her|he|she|year-old|years old|beard|jaw|brow|cheek|smile|"
    r"expression|wearing|suit|uniform|coat|engineer|worker|officer|host|"
    r"announcer|operator|controller|captain|doctor|detective|pilot|"
    r"scientist|reporter|narrator|figure)\b",
    re.IGNORECASE)


def _depicts_person(prompt: str) -> bool:
    """True when the prompt carries any person-evidence token."""
    return bool(_PERSON_WORDS.search(prompt or ""))


#: Broadcast-gear vocabulary (round 5 operator directive 2026-06-10): CHARACTER
#: portrait prompts must not mention radio/mic/studio gear -- the tokens drag
#: FLUX toward microphones and consoles (the c01 giant-mic catch). The
#: ANNOUNCER is exempt (his portrait is radio-styled BY DESIGN).
_GEAR_WORDS = re.compile(
    r"\s*\b(?:radios?|microphones?|mics?|broadcasts?|broadcasters?|"
    r"broadcasting|recording\s+studios?|radio\s+(?:station|studio|set|"
    r"booth)s?|studios?|on[- ]air(?:\s+sign)?)\b[,;]?",
    re.IGNORECASE)


def _scrub_gear_words(prompt: str) -> str:
    """Remove broadcast-gear tokens from a CHARACTER portrait prompt, tidying
    the leftover separators. Pure; '' stays ''."""
    out = _GEAR_WORDS.sub("", prompt or "")
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"(,\s*)+,", ", ", out)
    out = re.sub(r"\s+,", ",", out)
    return out.strip(" ,;").strip()


def _clean_llm_prompt(raw: str) -> str:
    """First non-empty line of the LLM output, trimmed; '' if unusable."""
    if not raw:
        return ""
    for line in str(raw).splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line
    return ""


def _passes_consistency(prompt: str, appearance: str, setting: str) -> bool:
    """v1 schema gate: the prompt must be GROUNDED in the brief -- it shares at
    least one significant word with the character appearance or the story
    setting. Cheap word-overlap check, not a 2nd LLM call. When neither
    appearance nor setting is known, nothing to assert -> passes."""
    want = _significant_words(appearance) | _significant_words(setting)
    if not want:
        return True
    return bool(want & _significant_words(prompt))


def derive_image_prompts(cast: list, meta: dict, *, llm_fn=None, max_reseed: int = 2,
                         consistency_gate_warn_only: bool = False, lines=None,
                         fps: int = 25):
    """ONE versioned image-object payload: ``{"version": 1, "objects": [...]}``
    (still-spine ST-2 / pass-02 item 1: portraits MIGRATED to the object
    schema in the same patch; no dual-schema shims).

    Each object carries ``object_id`` / ``kind`` / ``role`` / ``w`` / ``h`` /
    ``prompt`` / ``prompt_hash`` / ``source`` plus ``char_id`` (portraits;
    object_id == char_id) or ``beat_id`` (scene stills). Guards branch by
    KIND before running: the person guard + gear scrub run ONLY on
    kind=portrait; scene stills get the no-text clause (inside
    ``compose_still_prompt``) and skip the person guard entirely.

    Portrait path: LLM (temp=0, injected or lazily resolved) refines each;
    empty/unparseable -> reseed -> deterministic fallback. ``prompt_hash`` is
    taken AFTER the call. Never raises; never emits an empty prompt.
    Returns ``(payload, warnings)``.

    ``lines`` (optional, the frozen ledger lines, READ-ONLY): announcer
    portrait minting (as before) PLUS the v1 scene-still targets
    (open/announcer/outro via :func:`derive_scene_still_targets`).
    """
    warnings: list = []
    setting = _read_setting(meta)
    out: dict = {}
    roster = list(cast or [])
    cast_ids = {str(c.get("char_id") or "") for c in roster if isinstance(c, dict)}
    for cid in announcer_line_char_ids(lines):
        if cid in cast_ids:
            continue                      # a real cast row already covers it
        roster.append({
            "char_id": cid,
            "portrait_prompt": ANNOUNCER_PORTRAIT_ANCHOR,
            "_synthetic_announcer": True,
        })
    for char in roster:
        if not isinstance(char, dict):
            continue
        cid = str(char.get("char_id") or "")
        if not cid:
            continue
        appearance = _appearance_for_char([char], cid)
        prompt = ""
        source = "template"
        if llm_fn is not None:
            req = _build_char_prompt_request(char, meta, setting)
            for attempt in range(max_reseed + 1):
                try:
                    raw = llm_fn(req)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"image prompt llm_fn raised for {cid} ({exc}); reseed {attempt}")
                    raw = ""
                cand = _clean_llm_prompt(raw)
                if cand:
                    prompt = cand
                    source = "llm"
                    break
                if attempt < max_reseed:
                    warnings.append(f"empty image prompt for {cid}; reseed {attempt + 1}/{max_reseed}")
        if not prompt:
            prompt = compose_image_prompt_fallback(meta, char)
            source = "template"
        # Story-consistency gate (schema assertion, v1). The synthetic
        # ANNOUNCER grounds on APPEARANCE ONLY (the radio anchor): an LLM
        # line that drops the radio styling for pure story-setting flavor
        # fails the gate and falls back to the radio template (operator
        # directive 2026-06-09: the announcer gets a RADIO-style image).
        gate_setting = "" if char.get("_synthetic_announcer") else setting
        if not _passes_consistency(prompt, appearance, gate_setting):
            msg = f"image prompt for {cid} missing appearance/setting trait"
            if consistency_gate_warn_only:
                warnings.append(msg + " (warn-only; kept)")
            else:
                warnings.append(msg + "; fell back to template")
                prompt = compose_image_prompt_fallback(meta, char)
                source = "template_consistency"
        # PERSON GUARD (look-QA round 4, 2026-06-10): a portrait prompt that
        # depicts no person (the live "microphone under a lamp" catch for a
        # cast character) falls back to the template, which LEADS with the
        # writer's physical character description. Always enforced -- a
        # face-less init_image also starves the audio-driven-face engine.
        if not _depicts_person(prompt):
            warnings.append(
                f"image prompt for {cid} depicts no PERSON; fell back to "
                f"the appearance template")
            prompt = compose_image_prompt_fallback(meta, char)
            source = "template_person_guard"
        # GEAR SCRUB (round 5 operator directive): character portraits never
        # mention radio/mic/studio gear -- the tokens pull FLUX toward
        # equipment (the c01 giant-mic catch). The ANNOUNCER keeps his radio
        # styling by design (radio-grounding gate): synthetic announcer rows
        # AND a cast row literally named ANNOUNCER are exempt.
        _is_announcer_row = bool(
            char.get("_synthetic_announcer")
            or str(char.get("name") or "").strip().upper() == "ANNOUNCER")
        if not _is_announcer_row:
            _scrubbed = _scrub_gear_words(prompt)
            if _scrubbed != prompt:
                warnings.append(
                    f"image prompt for {cid}: broadcast-gear tokens scrubbed")
                prompt = _scrubbed or compose_image_prompt_fallback(meta, char)
        if char.get("_synthetic_announcer"):
            source = "announcer_" + source   # traceable in reports/ledger
        # FINISH the prompt (gap-audit F3, 2026-06-10): era tail + film style
        # tail, restored from the deleted legacy composer. Runs AFTER the
        # consistency + person guards (finishing never re-triggers them) and
        # BEFORE the hash so the stamped hash matches the rendered prompt.
        try:
            try:
                from ._otr_story_brief_helpers import (  # type: ignore
                    finish_visual_prompt)
            except ImportError:  # pragma: no cover -- flat test imports
                from _otr_story_brief_helpers import (  # type: ignore
                    finish_visual_prompt)
            # era_profile="portrait": never bleeds the episode's ambient
            # colour palette into character faces (sci-fi = blue wash,
            # period drama = red wash). Only the atmosphere mood line is
            # safe; full palette is explicitly excluded (BUG-LOCAL-113).
            prompt = finish_visual_prompt(meta, prompt,
                                          era_profile="portrait")
        except Exception:  # noqa: BLE001
            pass
        out[cid] = {
            "prompt": prompt,
            "prompt_hash": _content_hash(prompt),   # hash AFTER the call
            "source": source,
            "_role": ("announcer_visual" if _is_announcer_row
                      else "character_video"),
        }

    # ---- assemble the ONE versioned object payload (pass-02 item 1) ----
    objects: list = []
    for cid, pinfo in out.items():
        objects.append({
            "object_id": cid,                 # portrait object_id == char_id
            "kind": "portrait",
            "role": pinfo.pop("_role", "character_video"),
            "char_id": cid,
            "w": PORTRAIT_W, "h": PORTRAIT_H,
            "prompt": pinfo["prompt"],
            "prompt_hash": pinfo["prompt_hash"],
            "source": pinfo["source"],
        })

    # SCENE-STILL objects (ST-2): open/announcer/outro from pure helpers on
    # the LINES -- never video.shots (image gen runs BEFORE ShotLock). The
    # prompt comes from the shared 5-layer composer (subject parity with the
    # driver's text prompts is locked in tests); no LLM call, no person
    # guard, no gear scrub -- guards branch by kind BEFORE running.
    scene_targets, scene_warns = ([], [])
    if lines:
        try:
            scene_targets, scene_warns = derive_scene_still_targets(
                lines, fps=fps)
        except Exception as exc:  # noqa: BLE001 -- stills never kill prompts
            warnings.append(f"scene-still derivation failed ({exc}); "
                            "episode renders without scene stills (LOUD)")
    warnings.extend(scene_warns)
    if scene_targets:
        try:
            from ._otr_story_brief_helpers import (  # type: ignore
                compose_still_prompt)
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_story_brief_helpers import (  # type: ignore
                compose_still_prompt)
        sw, sh = _landscape_still_dims()
        for tgt in scene_targets:
            sprompt = compose_still_prompt(
                meta, kind=tgt["kind"], role=tgt["role"],
                beat_id=tgt["beat_id"])
            objects.append({
                "object_id": f"still_{tgt['beat_id']}",
                "kind": tgt["kind"],
                "role": tgt["role"],
                "beat_id": tgt["beat_id"],
                "w": sw, "h": sh,
                "prompt": sprompt,
                "prompt_hash": _content_hash(sprompt),
                "source": tgt["source"],
            })
    return {"version": 1, "objects": objects}, warnings


def _resolve_writer_llm(meta, warnings):
    """Lazily resolve the writer's slot LLM as a callable(prompt)->str (temp=0).
    Returns None if unavailable -> the deterministic template carries every
    character. Mirrors OTR_ShotLock._resolve_writer_llm; never raises."""
    try:
        from . import otr_shot_lock as _sl
        return _sl._resolve_writer_llm(meta, warnings)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"writer LLM unavailable ({exc}); using template prompts")
        return None


class OTRMetaBriefImagePromptGen:
    """Registered as ``OTR_MetaBriefImagePromptGen``. Brief -> per-character image prompts."""

    CATEGORY = "OldTimeRadio/v2/image"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompts_json", "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "Frozen ledger JSON (cast + meta brief). Image prompts derive from here.",
                }),
            },
            "optional": {
                "image_policy_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "OTR_ImageDirector policy (granularity/seed); opaque to prompt text.",
                }),
                "consistency_gate_warn_only": ("BOOLEAN", {"default": False}),
                "gate_in": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def generate(self, script_json, image_policy_json="{}",
                 consistency_gate_warn_only=False, gate_in=""):
        try:
            led = json.loads(script_json or "{}")
            if not isinstance(led, dict):
                led = {}
        except (ValueError, TypeError):
            led = {}
        meta = led.get("meta") if isinstance(led.get("meta"), dict) else {}
        cast = led.get("cast") if isinstance(led.get("cast"), list) else []
        lines = led.get("lines") if isinstance(led.get("lines"), list) else []

        warnings: list = []
        # Brief disposition, ONCE per run (gap-audit G4 restore).
        try:
            try:
                from ._otr_story_brief_helpers import (  # type: ignore
                    log_story_brief_disposition)
            except ImportError:  # pragma: no cover -- flat test imports
                from _otr_story_brief_helpers import (  # type: ignore
                    log_story_brief_disposition)
            log_story_brief_disposition(meta, "flux_portrait", log)
        except Exception:  # noqa: BLE001
            pass
        llm_fn = _resolve_writer_llm(meta, warnings)
        payload, warn2 = derive_image_prompts(
            cast, meta, llm_fn=llm_fn,
            consistency_gate_warn_only=bool(consistency_gate_warn_only),
            lines=lines,
        )
        warnings.extend(warn2)

        objs = payload.get("objects") or []
        report = [f"image_prompts v{payload.get('version')}: "
                  f"{len(objs)} objects"]
        for obj in objs:
            ident = obj.get("char_id") or obj.get("beat_id") or ""
            report.append(
                f"  {obj['object_id']}: kind={obj['kind']} role={obj['role']}"
                f" {obj['w']}x{obj['h']} id={ident}"
                f" source={obj['source']} hash={obj['prompt_hash'][:8]}")
        for w in warnings:
            report.append(f"WARN: {w}")
            log.warning("[OTR_MetaBriefImagePromptGen] %s", w)

        return (
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            "\n".join(report),
        )
