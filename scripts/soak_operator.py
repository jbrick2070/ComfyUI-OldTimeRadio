"""
soak_operator.py -- LEGACY SHIM (BUG-LOCAL-002 fix, 2026-05-02).

This module used to be a 1500-line soak runner with stale `WV_*` positional
widget indices that no longer matched the live OTR_LedgerScriptWriter node
(`episode_title` and `num_characters` were added between v1.5 and v2.0,
shifting every downstream index off by 1-2 slots). Anything calling
`patch_workflow()` from supersoaker would write to the wrong slots --
e.g. `creativity` value would land in `style_custom`, `optimization_profile`
in `arc_enhancer`. supersoaker.py has been deleted.

The current canonical surface for talking to ComfyUI's HTTP API lives in
``scripts/otr_api.py`` -- it patches widgets by NAME (not position) using
the live `/object_info` schemas, so widget reorders in the future cannot
silently corrupt a run.

The only function still exposed here is `scan_treatment` (read-only file
scanner used by ``tests/test_treatment_scanner_unicode.py``). It is kept
in this module to avoid churning the test import; for any new code, prefer
adding to ``scripts/treatment_scanner.py``.

DO NOT add new helpers here. Add them to ``scripts/otr_api.py`` (HTTP API)
or ``scripts/treatment_scanner.py`` (file scanning).
"""

from __future__ import annotations

import re


def scan_treatment(path: str) -> list[str]:
    """Read-only scan of a treatment file. Returns list of flag strings.

    Never modifies files. Never tries to fix anything. Carries the
    BUG-LOCAL-033 fix: regex classes accept both ASCII `-` and the
    U+2500 box-drawing horizontal that real treatments contain, plus
    `->`/`-->` AND U+2192 cast arrows.
    """
    flags: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as exc:
        return [f"READ_FAIL: Could not read treatment: {exc}"]

    # 1. Missing LLM closing
    if "__NEEDS_LLM_CLOSING__" in text:
        flags.append("NEEDS_LLM_CLOSING: Announcer sign-off never generated")

    # 2. Parse cast
    cast: dict[str, dict[str, str]] = {}
    cast_section = re.search(
        r"CAST & VOICES\n[-─]+\n(.*?)(?:\n\n|\nSCENE ARC)", text, re.DOTALL
    )
    if cast_section:
        for line in cast_section.group(1).strip().split("\n"):
            m = re.match(
                r"\s*(\S+(?:\s+\S+)*?)\s+(?:->|-->|→)\s+(\S+)\s+(.*)",
                line,
            )
            if m:
                cast[m.group(1).strip()] = {
                    "preset": m.group(2).strip(),
                    "traits": m.group(3).strip(),
                }
    if not cast:
        flags.append("EMPTY_CAST: No characters found in CAST section")

    # 3. Duplicate voice presets
    presets_seen: dict[str, str] = {}
    for name, info in cast.items():
        p = info["preset"]
        if p in presets_seen:
            flags.append(f"DUPLICATE_VOICE: {name} and {presets_seen[p]} share {p}")
        presets_seen[p] = name

    # 4. Gender mismatch
    for name, info in cast.items():
        traits = info["traits"].lower()
        has_male = "male" in traits and "female" not in traits
        has_female = "female" in traits
        if has_male and has_female:
            flags.append(f"GENDER_CONFLICT: {name} traits say both male and female")

    # 5. Scene arc dialogue counts
    scene_arc = re.search(
        r"SCENE ARC\n[-─]+\n(.*?)(?:\nFULL SCRIPT\b)", text, re.DOTALL
    )
    total_dialogue = 0
    if scene_arc:
        for m in re.finditer(
            r"Scene\s+(\d+).*?(\d+)\s+dialogue lines", scene_arc.group(1)
        ):
            dl = int(m.group(2))
            total_dialogue += dl
            if dl == 0:
                flags.append(f"ZERO_DIALOGUE: Scene {m.group(1)} has 0 lines")
    else:
        flags.append("NO_SCENE_ARC: Scene arc section missing or unparseable")

    # 6. Full script checks
    script_section = re.search(
        r"FULL SCRIPT.*?\n[-─]+\n(.*?)(?:\nPRODUCTION)", text, re.DOTALL
    )
    script_body = script_section.group(1).strip() if script_section else ""

    if not script_body:
        flags.append("EMPTY_SCRIPT: Full script section is blank")

    sfx_count = len(re.findall(r"\[SFX\]", script_body))
    if sfx_count == 0 and script_body:
        flags.append("NO_SFX: Script has zero [SFX] cues")

    # 7. Production stats
    m = re.search(r"Duration\s*:\s*([\d.]+)\s*min\s*\(([\d.]+)\s*s\)", text)
    duration_s = float(m.group(2)) if m else 0
    if 0 < duration_s < 30:
        flags.append(f"SHORT_DURATION: Only {duration_s}s")

    m = re.search(r"Size\s*:\s*([\d.]+)\s*MB", text)
    size_mb = float(m.group(1)) if m else 0
    if 0 < size_mb < 5:
        flags.append(f"TINY_FILE: Only {size_mb}MB -- possible empty render")
    if size_mb > 500:
        flags.append(f"HUGE_FILE: {size_mb}MB -- possible runaway render")

    # 8. VRAM (telemetry only -- the OOM budget is owned by the external
    # per-hardware tier JSON, not this scanner; record the peak, never gate).
    m = re.search(r"PEAKED AT ([\d.]+)GB", text)
    vram = float(m.group(1)) if m else 0
    if vram > 0:
        flags.append(f"VRAM_PEAK: {vram}GB (telemetry)")

    # 9. Character name drift (cast vs script body)
    char_line_counts: dict[str, int] = {}
    if cast and script_body:
        cast_upper = set()
        for name in cast:
            cast_upper.add(name.upper())
            cast_upper.add(name.upper().replace(" ", ""))
        script_chars = set()
        for line in script_body.split("\n"):
            m = re.match(r"^\s+([A-Z][A-Z\s]+?)(?:\s+\[.*\])?\s*$", line)
            if m:
                cname = m.group(1).strip()
                if cname not in ("SFX", "PAUSE", "BEAT", "SCENE"):
                    script_chars.add(cname)
        script_chars.discard("ANNOUNCER")
        cast_upper.discard("ANNOUNCER")
        for sc in script_chars:
            compressed = sc.replace(" ", "")
            if sc not in cast_upper and compressed not in cast_upper:
                flags.append(f"NAME_DRIFT: '{sc}' in script but not in cast list")

    # 10. All-same-gender cast
    if cast:
        genders = set()
        for name, info in cast.items():
            if name.upper() == "ANNOUNCER":
                continue
            traits = info["traits"].lower()
            if "female" in traits:
                genders.add("female")
            elif "male" in traits:
                genders.add("male")
        if len(genders) == 1 and len(cast) > 2:
            flags.append(
                f"ALL_SAME_GENDER: All non-announcer cast is {genders.pop()}"
            )

    # 11. Title stuck on default
    m_title = re.search(r'Title\s*:\s*"([^"]+)"', text)
    if m_title:
        title_text = m_title.group(1)
        if title_text.lower() in ("the last frequency", "untitled", "episode"):
            flags.append(
                f"TITLE_STUCK: Title is '{title_text}' -- may be default/stuck"
            )

    # 12. Dialogue count mismatch
    if script_body and total_dialogue > 0:
        actual_lines = 0
        for line in script_body.split("\n"):
            stripped = line.strip()
            m_char = re.match(
                r"^([A-Z][A-Z\s]+?)(?:\s+\[.*\])?\s*$", stripped
            )
            if m_char and m_char.group(1).strip() not in (
                "SFX", "PAUSE", "BEAT", "SCENE"
            ):
                actual_lines += 1
        if abs(actual_lines - total_dialogue) > 2:
            flags.append(
                f"DIALOGUE_MISMATCH: Arc says {total_dialogue} lines but "
                f"script has ~{actual_lines}"
            )

    # 13. Name squish
    if cast:
        for name in cast:
            if name.upper() == "ANNOUNCER":
                continue
            if len(name) > 10 and " " not in name and name.upper() == name:
                flags.append(
                    f"NAME_SQUISH: '{name}' looks like a compressed name "
                    f"(missing spaces)"
                )

    # 14. Single-line character
    if script_body and cast:
        for line in script_body.split("\n"):
            stripped = line.strip()
            m_char = re.match(
                r"^([A-Z][A-Z\s]+?)(?:\s+\[.*\])?\s*$", stripped
            )
            if m_char:
                cname = m_char.group(1).strip()
                if cname not in (
                    "SFX", "PAUSE", "BEAT", "SCENE", "ANNOUNCER"
                ):
                    char_line_counts[cname] = char_line_counts.get(cname, 0) + 1
        for cname, count in char_line_counts.items():
            if count == 1 and len(char_line_counts) > 1:
                flags.append(
                    f"SINGLE_LINE_CHAR: {cname} only speaks once "
                    f"(wasted voice load)"
                )

    # 15. Announcer dominates
    if script_body:
        ann_count = 0
        for line in script_body.split("\n"):
            stripped = line.strip()
            if re.match(r"^ANNOUNCER\b", stripped):
                ann_count += 1
        max_char_count = (
            max(char_line_counts.values()) if char_line_counts else 0
        )
        if ann_count > max_char_count and ann_count > 2:
            flags.append(
                f"ANNOUNCER_DOMINATES: Announcer has {ann_count} lines "
                f"vs max character {max_char_count}"
            )

    # 16. News seed missing or blank
    news_m = re.search(
        r"NEWS SEED\n[-─]+\n(.*?)(?:\n\n|\nCAST)", text, re.DOTALL
    )
    if not news_m or not news_m.group(1).strip():
        flags.append(
            "NEWS_SEED_MISSING: No news seed found -- "
            "RSS feed may not be injecting"
        )

    # 17. Duplicate dialogue
    if script_body:
        dialogue_lines: list[str] = []
        capture_next = False
        for line in script_body.split("\n"):
            stripped = line.strip()
            if capture_next and stripped and not stripped.startswith("["):
                dialogue_lines.append(stripped)
                capture_next = False
            m_char = re.match(
                r"^([A-Z][A-Z\s]+?)(?:\s+\[.*\])?\s*$", stripped
            )
            if m_char and m_char.group(1).strip() not in (
                "SFX", "PAUSE", "BEAT", "SCENE"
            ):
                capture_next = True
        seen_lines: set[str] = set()
        for dl in dialogue_lines:
            if dl in seen_lines and len(dl) > 20:
                flags.append(
                    f"DUPLICATE_DIALOGUE: Line repeated: '{dl[:60]}...'"
                )
                break
            seen_lines.add(dl)

    # 18. Mojibake
    mojibake_patterns = ["a]]]", "A(c)", "a]!", 'a]"', "A<<", "A>>"]
    for pattern in mojibake_patterns:
        if pattern in text:
            flags.append(
                f"MOJIBAKE: Encoding corruption detected ('{pattern}' found)"
            )
            break

    # 19. Scene count mismatch
    if scene_arc:
        arc_scene_count = len(re.findall(r"Scene\s+\d+", scene_arc.group(1)))
        script_scene_count = len(
            re.findall(r"SCENE\s+\d+", script_body, re.IGNORECASE)
        )
        if arc_scene_count > 0 and script_scene_count > 0:
            if abs(arc_scene_count - script_scene_count) > 1:
                flags.append(
                    f"SCENE_MISMATCH: Arc has {arc_scene_count} scenes but "
                    f"script has {script_scene_count}"
                )

    return flags


__all__ = ["scan_treatment"]
