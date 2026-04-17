# Untested Changes Report — 2026-04-15 Session 2

All items below are code-complete and pass the offline regression suite (50 dropdown guardrails, 82 core tests, 7 audio contract tests). None have been validated on a live ComfyUI run yet.

---

## 1. P0 #1: `min_line_count_per_character` (story_orchestrator.py)

**What it does:** Prevents the self-critique pass from deleting characters. The revision prompt now says "Do NOT reduce any character below 2 dialogue lines." After the LLM revises, a post-critique validator counts lines per character in draft vs revision. If any character drops below the floor, the revision is rejected and the original draft is kept.

**What could go wrong:**
- The `_count_character_lines()` regex might miscount on edge-case formatting (e.g. bare `NAME:` with tabs instead of spaces, or names with digits).
- If the LLM legitimately needs to merge two minor characters, the floor blocks it. Default of 2 is conservative but may need tuning.
- False rejection on scripts where ANNOUNCER has exactly 1 line in the draft — ANNOUNCER is NOT excluded from the count. If this causes spurious rejections, we can add ANNOUNCER to the exclusion set.

**What to look for in logs:**
- `CRITIQUE: Character line counts - draft={...} revised={...}` — shows the comparison.
- `CRITIQUE: CRITIQUE_REJECTED` — means the validator fired. Check if the rejection was correct or spurious.

---

## 2. P0 #2: Director JSON Schema Validator (story_orchestrator.py)

**What it does:** Validates the Director's parsed JSON against `_DIRECTOR_SCHEMA` before it reaches downstream nodes. Repairs missing keys (adds empty defaults), validates voice_preset strings, filters broken SFX entries, synthesizes missing music cues, clamps invalid duration_sec values.

**What could go wrong:**
- If the LLM produces a valid but unconventional structure (e.g. nested voice_assignments), the validator might add redundant defaults on top.
- The fallback voice preset `"v2/en_speaker_0"` is male — if assigned to a female character, voice gender will mismatch until `_procedurally_override_voice_assignments` fixes it downstream.
- Synthesized music cue prompts are generic ("1940s old time radio..."). If a cue is missing, the generated music may not match the episode's tone.

**What to look for in logs:**
- `DIRECTOR_SCHEMA: repaired ...` — any of these lines means the LLM output was incomplete. One or two repairs per run is normal; many repairs suggests the Director prompt needs tuning.

---

## 3. P1 #5: Length-Sorted Bark Batching (batch_bark_generator.py)

**What it does:** Sorts dialogue lines by text length within each voice preset group before generation. Shorter lines batch together, reducing zero-padding waste. Script order is restored at assembly via `results[script_idx]`.

**What could go wrong:**
- Bark's internal state may depend on generation order within a preset (e.g. the first-line hallucination guard relies on `_presets_started`). The sort means the first-generated line per preset is now the SHORTEST line, not the first in script order. The `[clears throat]` prefix + temp floor still apply to the first line regardless of length — this should be fine but watch for weird Bark output on short opening lines.
- If the sort introduces a subtle timing difference in the final assembly, check that crossfades and scene boundaries still align.

**What to look for in logs:**
- `[BatchBark] Length-sorted N preset groups for reduced padding waste` — confirms the sort ran.

---

## 4. P1 #7: VRAM-Sentinel Decorator (_vram_log.py + batch_bark_generator.py)

**What it does:** Decorator `@vram_sentinel("bark_batch", max_entry_gb=6.0)` on `generate_batch()`. Checks VRAM at entry; if above 6 GB, logs a warning and calls `force_vram_offload()` before proceeding. Snapshots at entry and exit for telemetry.

**What could go wrong:**
- The 6.0 GB threshold may be too aggressive. If Bark's own model load exceeds 6 GB before generation starts, the sentinel will fire on every run. The threshold should be below the LLM model size (~8-10 GB) but above Bark's idle state.
- Double-offload: `generate_batch()` already calls `force_vram_offload()` at line 511. With the sentinel, there's now a potential double-offload on the first call. Harmless but adds ~1-2s startup.
- The sentinel calls `vram_snapshot` at entry, which adds 2 extra log lines per batch run. This is intentional telemetry, not noise.

**What to look for in logs:**
- `VRAM_SENTINEL_TRIGGERED phase=bark_batch` — means LLM weights were still loaded when Bark started. This is a real bug if it fires, but the sentinel recovers from it.
- `VRAM_SNAPSHOT phase=bark_batch_entry/exit` — normal telemetry.

---

## 5. P1 #9: High-Creativity Soak Profile (soak_operator.py)

**What it does:** Re-adds `"maximum chaos"` to the CREATIVITIES pool with weighted selection (~10% of runs instead of the old 25%). Catches temperature-sensitive regressions.

**What could go wrong:**
- The original removal was due to ghost runs (EMPTY_CAST, NO_SCENE_ARC, EMPTY_SCRIPT, TITLE_STUCK). At 10% frequency, the same failures will occur — they just won't dominate the soak. If the fatal-streak detector (FATAL_STREAK_LIMIT=3) fires during a cluster of maximum-chaos runs, it will halt the soak.
- The watcher override `force_creativity` still accepts `"maximum chaos"` and can pin it for targeted testing.

**What to look for in soak logs:**
- Runs with `creativity=maximum chaos` that produce valid episodes — that's the win.
- Runs with `creativity=maximum chaos` that produce EMPTY_CAST or TITLE_STUCK — expected at this temperature, but should not exceed ~30% of max-chaos runs.

---

## 6. P2 #12: Per-LLM-Call VRAM Snapshots (story_orchestrator.py)

**What it does:** Added `vram_snapshot("llm_generate_entry")` and `vram_snapshot("llm_generate_exit")` inside `_generate_with_llm()`. Logs token count and inference time. Every LLM call now emits VRAM telemetry.

**What could go wrong:**
- Very minor: adds 2 VRAM_SNAPSHOT lines per LLM call to `otr_runtime.log`. A typical episode has 4-8 LLM calls, so this adds 8-16 extra lines. The log already handles this volume fine.
- The token count estimate is based on `new_tokens_cpu.shape[0]` which may differ from the actual output token count if the streamer truncates.

**What to look for in logs:**
- `VRAM_SNAPSHOT phase=llm_generate_entry/exit` with `current_gb` and `peak_gb` — these are the new telemetry lines.
- `llm_generate_exit: generated N tokens in M.Ms` — correlate with wall-clock timeouts.

---

## Files Modified

| File | Lines changed | Items |
|---|---|---|
| `nodes/story_orchestrator.py` | +247 | P0 #1, P0 #2, P2 #12 |
| `nodes/_vram_log.py` | +55 | P1 #7 |
| `nodes/batch_bark_generator.py` | +3 | P1 #5, P1 #7 |
| `scripts/soak_operator.py` | +17 | P1 #9 |
| `ROADMAP.md` | updated | Marked items shipped |

## How to Test

Queue one end-to-end run in ComfyUI Desktop (any model, default settings). Paste the full console output back. I'll check for:
1. `CRITIQUE: Character line counts` — P0 #1 validator active
2. `DIRECTOR_SCHEMA:` — P0 #2 validator active (may show 0 repairs if LLM output was clean)
3. `[BatchBark] Length-sorted` — P1 #5 active
4. `VRAM_SNAPSHOT phase=bark_batch_entry` — P1 #7 sentinel active
5. `VRAM_SNAPSHOT phase=llm_generate_entry/exit` — P2 #12 active
6. Audio output byte-identical to v1.5 baseline (C7 constraint)
