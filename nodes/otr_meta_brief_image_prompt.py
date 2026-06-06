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
    """Appearance text looked up by char_id ONLY (BUG-098), never display name."""
    cid = str(char_id or "")
    for c in cast or []:
        if isinstance(c, dict) and str(c.get("char_id") or "") == cid:
            return str(c.get("portrait_prompt") or c.get("appearance") or "").strip()
    return ""


#: Shared portrait style anchor (mirrors the shipped batch_flux_portrait_render
#: default so gen-1 Flux output stays consistent across the cast).
STYLE_ANCHOR = "head-and-shoulders studio portrait, neutral lighting, cinematic"


def compose_image_prompt_fallback(meta: dict, char: dict) -> str:
    """Deterministic brief-composed portrait prompt -- NEVER empty.

    ``"{appearance}, {setting} setting, {style anchor}"`` with empty parts
    dropped; degrades to the style anchor alone if the brief + cast are bare.
    """
    appearance = str(
        (char or {}).get("portrait_prompt") or (char or {}).get("appearance") or ""
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
        "line, no preamble) for this character. Ground it in the appearance and "
        "the story setting; keep it photographic and period-consistent.\n"
        f"character_appearance: {appearance or '(unspecified)'}\n"
        f"story_setting: {setting or '(unspecified)'}\n"
        f"style_anchor: {STYLE_ANCHOR}\n"
        "Return only the prompt line."
    )


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
                         consistency_gate_warn_only: bool = False):
    """Per-character image prompts: ``{char_id: {prompt, prompt_hash, source}}``.

    LLM (temp=0, injected or lazily resolved) refines each; empty/unparseable ->
    reseed -> deterministic fallback. ``prompt_hash`` is taken AFTER the call.
    Never raises; never emits an empty prompt. Returns ``(prompts, warnings)``.
    """
    warnings: list = []
    setting = _read_setting(meta)
    out: dict = {}
    for char in cast or []:
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
        # Story-consistency gate (schema assertion, v1).
        if not _passes_consistency(prompt, appearance, setting):
            msg = f"image prompt for {cid} missing appearance/setting trait"
            if consistency_gate_warn_only:
                warnings.append(msg + " (warn-only; kept)")
            else:
                warnings.append(msg + "; fell back to template")
                prompt = compose_image_prompt_fallback(meta, char)
                source = "template_consistency"
        out[cid] = {
            "prompt": prompt,
            "prompt_hash": _content_hash(prompt),   # hash AFTER the call
            "source": source,
        }
    return out, warnings


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

        warnings: list = []
        llm_fn = _resolve_writer_llm(meta, warnings)
        prompts, warn2 = derive_image_prompts(
            cast, meta, llm_fn=llm_fn,
            consistency_gate_warn_only=bool(consistency_gate_warn_only),
        )
        warnings.extend(warn2)

        report = [f"image_prompts: {len(prompts)} characters"]
        for cid, p in prompts.items():
            report.append(f"  {cid}: source={p['source']} hash={p['prompt_hash'][:8]}")
        for w in warnings:
            report.append(f"WARN: {w}")
            log.warning("[OTR_MetaBriefImagePromptGen] %s", w)

        return (
            json.dumps(prompts, ensure_ascii=True, separators=(",", ":")),
            "\n".join(report),
        )
