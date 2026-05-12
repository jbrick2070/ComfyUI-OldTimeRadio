"""nodes/_otr_lfc_context.py — Story Bible context injection (ADR 6.13).

Two-tier memory architecture used by LFC LLM phases (4, 5, 6). The
LLM never sees the whole script -- it sees a tightly curated context
block sourced from the ledger:

  Macro-Bible -- persistent, episode-level context (ADR 6.13)
    Built once at cascade entry. Cached for every LLM call in the
    same run. Contains the writer-stamped episode identity:

      episode_logline    meta.episode_title + meta.style summary
      cast_voice_cards   per-cast char_id / name / traits /
                          voice_preset / catchphrases
      allowed_entities   meta.news.key_terms + meta.allowed_people +
                          meta.allowed_things
      arc_skeleton       outline beats reduced to one-line intents
      world_rules        meta.style constraints

  Micro-Recap -- per-call, scoped context (ADR 6.13)
    Built once per LLM call. The scope determines the shape:

      scene   previous scene's last 3 lines + this scene's beat
              intents + active speakers' voice cards only
      speaker all lines from this speaker + speaker's voice card
              (other speakers omitted)
      episode one-paragraph synopsis per scene (LLM-generated once at
              cascade entry, cached in meta.scene_synopses)

  Combined ceiling: roughly 4-8K tokens per LLM call. Inside the
  Mistral-Nemo FP8 sweet spot; well under the 60K context-cliff
  benchmark (ADR section 16 references).

The Macro-Bible / Micro-Recap distinction is structural; both
return plain strings ready to splice into a system or user prompt.
Phase 4 / 5 / 6 prompts call build_macro_bible once and
build_micro_recap per scope unit.

Token estimation uses a rough heuristic (1 token per 4 chars) so the
helpers can self-cap before passing strings into a tokenizer. The
heuristic is conservative (Mistral-Nemo's tokenizer averages closer
to 1 / 3.5 chars per token in English narrative), so the actual
runtime token count comes in slightly UNDER the declared ceiling --
which is the safer direction for VRAM budgeting (ADR section 11).

Status: LFC sprint commit 7 of 14 (2026-05-11).
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Literal, Optional

log = logging.getLogger("OTR.lfc_context")


__all__ = [
    "MACRO_BIBLE_TOKEN_CAP",
    "MICRO_RECAP_TOKEN_CAP",
    "build_macro_bible",
    "build_micro_recap",
    "estimate_tokens",
    "scenes_from_lines",
    "Scope",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Soft caps. ADR section 6.13 calls for 4-8K active context; we
# split the budget roughly 1/3 macro, 2/3 micro because the macro
# block is mostly identity / static facts while the micro block
# carries the dynamic per-call detail. Either cap can be overridden
# per call.
MACRO_BIBLE_TOKEN_CAP: int = 2500
MICRO_RECAP_TOKEN_CAP: int = 4500


# Per-cast catchphrase budget. Reads cast[i].catchphrases when
# present; tolerates absence (most v2.0-alpha cast rows do not yet
# stamp catchphrases).
_MAX_CATCHPHRASES_PER_CAST: int = 3


Scope = Literal["scene", "speaker", "episode"]


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Conservative -- 1 tok / 4 chars.

    Real Mistral-Nemo tokenization for English narrative is
    typically closer to 1 / 3.5 chars/tok. Using 1/4 here means the
    real runtime token count will be slightly UNDER the declared
    ceiling, which is the safer direction for VRAM. The helpers
    don't try to be precise -- they cap on this estimate and rely
    on the loader's `context_cap` truncation as the second-line
    defence per ADR section 11.
    """
    return max(1, (len(text) + 3) // 4)


def _truncate_to_token_cap(text: str, cap: int) -> str:
    """Hard-truncate `text` to fit under `cap` estimated tokens.

    Cuts at the nearest line boundary to avoid mid-sentence breaks.
    Appends "..." when truncation happens. Returns `text` unchanged
    when it already fits.
    """
    if estimate_tokens(text) <= cap:
        return text
    char_budget = cap * 4 - 4  # leave room for the truncation marker
    if char_budget <= 0:
        return "..."
    cut = text[:char_budget]
    # Snap to nearest line boundary (newline-or-period) to avoid an
    # in-the-middle break that confuses the LLM.
    last_nl = cut.rfind("\n")
    last_dot = cut.rfind(". ")
    snap = max(last_nl, last_dot)
    if snap > char_budget // 2:
        cut = cut[:snap + 1]
    return cut.rstrip() + "\n..."


# ---------------------------------------------------------------------------
# Macro-Bible
# ---------------------------------------------------------------------------


def _cast_voice_card_line(row: dict) -> str:
    char_id = row.get("char_id", "")
    name = row.get("name", "")
    traits = (
        row.get("traits")
        or row.get("character_description")
        or ""
    )
    voice = (
        row.get("voice_preset")
        or row.get("tts_model")
        or row.get("speaker_role")
        or ""
    )
    catch = row.get("catchphrases")
    parts = [f"- {char_id} / {name}"]
    if traits:
        if isinstance(traits, list):
            traits = ", ".join(traits)
        parts.append(f"traits: {traits}")
    if voice:
        parts.append(f"voice: {voice}")
    if catch:
        # Cap catchphrases per cast member.
        if isinstance(catch, str):
            phrases = [catch]
        elif isinstance(catch, (list, tuple)):
            phrases = list(catch)[:_MAX_CATCHPHRASES_PER_CAST]
        else:
            phrases = []
        if phrases:
            parts.append(f"catchphrases: {phrases}")
    return "  " + " | ".join(parts)


def _outline_intents(ledger_data: dict) -> list[str]:
    """One-line intent per outline beat, in order.

    Reads meta.outline.beats when present; falls back to top-level
    beats[] when meta.outline is absent. Skips beats without an
    intent field.
    """
    meta = ledger_data.get("meta") or {}
    outline = meta.get("outline") or {}
    beats = outline.get("beats") if isinstance(outline, dict) else None
    if not isinstance(beats, list):
        beats = ledger_data.get("beats") or []
    out: list[str] = []
    for b in beats:
        if not isinstance(b, dict):
            continue
        intent = b.get("intent") or b.get("beat_intent") or ""
        if not isinstance(intent, str) or not intent.strip():
            continue
        bid = b.get("beat_id") or ""
        out.append(f"  {bid}: {intent.strip()}")
    return out


def build_macro_bible(
    led_or_data,
    *,
    token_cap: int = MACRO_BIBLE_TOKEN_CAP,
) -> str:
    """Build the persistent Macro-Bible context block.

    ADR section 6.13. Read-only -- never mutates the ledger. The
    returned string is ready to splice into a system or user prompt.

    Capped to `token_cap` estimated tokens via _truncate_to_token_cap.
    Sections drop in priority order if the cap is tight:
        episode_logline > cast_voice_cards > allowed_entities >
        arc_skeleton > world_rules
    """
    ledger_data = (
        led_or_data if isinstance(led_or_data, dict)
        else led_or_data.data
    )
    meta = ledger_data.get("meta") or {}

    sections: list[str] = []

    # 1. Episode logline
    title = (meta.get("episode_title") or "").strip()
    style = (meta.get("style") or "").strip()
    if title or style:
        bits = []
        if title:
            bits.append(f"title: {title}")
        if style:
            bits.append(f"style: {style}")
        sections.append("EPISODE:\n  " + " | ".join(bits))

    # 2. Cast voice cards
    cast = ledger_data.get("cast") or []
    if cast:
        cast_lines = [_cast_voice_card_line(row) for row in cast
                      if isinstance(row, dict)]
        if cast_lines:
            sections.append("CAST:\n" + "\n".join(cast_lines))

    # 3. Allowed entities
    news = meta.get("news") or {}
    key_terms = news.get("key_terms") if isinstance(news, dict) else []
    allowed_people = meta.get("allowed_people") or []
    allowed_things = meta.get("allowed_things") or []
    ent_lines: list[str] = []
    if key_terms:
        ent_lines.append(f"  key_terms: {list(key_terms)}")
    if allowed_people:
        ent_lines.append(f"  people: {list(allowed_people)}")
    if allowed_things:
        ent_lines.append(f"  things: {list(allowed_things)}")
    if ent_lines:
        sections.append("ALLOWED ENTITIES:\n" + "\n".join(ent_lines))

    # 4. Arc skeleton
    intents = _outline_intents(ledger_data)
    if intents:
        sections.append("ARC SKELETON:\n" + "\n".join(intents))

    # 5. World rules (style-derived; light touch).
    if style:
        sections.append(
            "WORLD RULES:\n"
            f"  style: {style}. Maintain the scenario's "
            f"era / texture / acoustic palette."
        )

    full = "\n\n".join(sections)
    return _truncate_to_token_cap(full, token_cap)


# ---------------------------------------------------------------------------
# Micro-Recap helpers
# ---------------------------------------------------------------------------


def scenes_from_lines(ledger_data: dict) -> list[dict]:
    """Partition lines[] into scenes based on music markers (ADR Q5).

    A scene starts at music_open (or the start of lines[]) and ends
    just before the next music marker (or the end of lines[]). The
    music marker line itself is included as the FIRST entry of the
    scene it opens (or LAST entry of the scene it closes -- the
    convention matches the existing reviewer / scene_sequencer's
    "music_inter is the divider, not the content" behavior).

    Returns a list of dicts:
        [{"scene_idx": 0, "lines": [...], "start_idx": 0,
          "end_idx": 4}, ...]

    Fallback per ADR Q5: when no music_inter markers exist, the
    whole body becomes one scene.
    """
    lines = ledger_data.get("lines") or []
    if not lines:
        return []

    # Find scene boundaries at music_inter markers (music_open and
    # music_close are episode bookends; the inter marker is the
    # in-body divider per ADR Q5).
    boundary_indices: list[int] = [0]
    for idx, ln in enumerate(lines):
        if not isinstance(ln, dict):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role == "music_inter":
            boundary_indices.append(idx)
    boundary_indices.append(len(lines))

    scenes: list[dict] = []
    for i in range(len(boundary_indices) - 1):
        start = boundary_indices[i]
        end = boundary_indices[i + 1]
        if start >= end:
            continue
        scenes.append({
            "scene_idx": len(scenes),
            "lines": lines[start:end],
            "start_idx": start,
            "end_idx": end,
        })
    return scenes


def _voice_card_for(cast_by_id: dict, char_id: str) -> str:
    row = cast_by_id.get(char_id) or {}
    if not row:
        return f"{char_id} (unknown speaker)"
    name = row.get("name") or char_id
    traits = row.get("traits") or row.get("character_description") or ""
    if isinstance(traits, list):
        traits = ", ".join(traits)
    if traits:
        return f"{name} ({traits})"
    return name


def _render_line_row(ln: dict) -> str:
    line_id = ln.get("line_id", "")
    cid = ln.get("char_id", "")
    text = (ln.get("text") or "").strip()
    return f"  {line_id} | {cid}: {text}"


def build_micro_recap(
    led_or_data,
    *,
    scope: Scope,
    target: Any,
    token_cap: int = MICRO_RECAP_TOKEN_CAP,
) -> str:
    """Build the per-call Micro-Recap context block.

    ADR section 6.13. Read-only.

    `scope` selects the shape; `target` carries the scope-specific
    arg:

      scope="scene"    target is the scene_idx (int) OR the scene
                       dict returned by scenes_from_lines.
      scope="speaker"  target is the char_id (str).
      scope="episode"  target is unused -- the recap shows the
                       cached meta.scene_synopses if present, else
                       a brief overview synthesized from the outline.

    Returns the formatted string capped to `token_cap` tokens.
    """
    ledger_data = (
        led_or_data if isinstance(led_or_data, dict)
        else led_or_data.data
    )
    cast = ledger_data.get("cast") or []
    cast_by_id = {
        r.get("char_id", ""): r for r in cast
        if isinstance(r, dict)
    }

    if scope == "scene":
        scenes = scenes_from_lines(ledger_data)
        if isinstance(target, int):
            if not 0 <= target < len(scenes):
                return f"(scene {target} out of range)"
            scene = scenes[target]
        elif isinstance(target, dict):
            scene = target
        else:
            return "(invalid scene target)"
        scene_idx = scene.get("scene_idx", 0)
        scene_lines = scene.get("lines") or []
        # Active speakers in this scene
        active_cids = set()
        for ln in scene_lines:
            if isinstance(ln, dict):
                cid = ln.get("char_id") or ""
                if cid:
                    active_cids.add(cid)
        # Previous scene's last 3 lines (ADR 6.13).
        prev_tail: list[str] = []
        if scene_idx > 0 and scene_idx - 1 < len(scenes):
            prev_lines = scenes[scene_idx - 1].get("lines") or []
            voiced_prev = [
                ln for ln in prev_lines
                if isinstance(ln, dict)
                and (ln.get("speaker_role") or "")
                .strip().lower() in ("character", "announcer")
            ]
            for ln in voiced_prev[-3:]:
                prev_tail.append(_render_line_row(ln))

        sections: list[str] = [f"SCENE {scene_idx}:"]
        if prev_tail:
            sections.append(
                "  prior context (last 3 lines of scene "
                f"{scene_idx - 1}):\n" + "\n".join(prev_tail)
            )
        if active_cids:
            cards = [_voice_card_for(cast_by_id, c) for c in active_cids]
            sections.append("  active speakers:\n  " + "\n  ".join(cards))
        body = []
        for ln in scene_lines:
            if isinstance(ln, dict):
                body.append(_render_line_row(ln))
        if body:
            sections.append("  lines:\n" + "\n".join(body))
        return _truncate_to_token_cap("\n".join(sections), token_cap)

    if scope == "speaker":
        if not isinstance(target, str) or not target:
            return "(invalid speaker target)"
        speaker_lines = [
            ln for ln in (ledger_data.get("lines") or [])
            if isinstance(ln, dict)
            and ln.get("char_id") == target
            and not ln.get("skip")
        ]
        voice_card = _voice_card_for(cast_by_id, target)
        sections = [
            f"SPEAKER: {voice_card}",
            "  lines spoken in this episode:",
        ]
        for ln in speaker_lines:
            sections.append(_render_line_row(ln))
        return _truncate_to_token_cap("\n".join(sections), token_cap)

    if scope == "episode":
        meta = ledger_data.get("meta") or {}
        synopses = meta.get("scene_synopses") or []
        sections = ["EPISODE OVERVIEW:"]
        if synopses:
            for i, s in enumerate(synopses):
                sections.append(f"  scene {i}: {s}")
        else:
            # Fall back to outline-intent overview.
            for line in _outline_intents(ledger_data):
                sections.append(line)
        return _truncate_to_token_cap("\n".join(sections), token_cap)

    return f"(unknown scope: {scope!r})"
