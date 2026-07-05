R2 JUDGMENT + CONVERGENCE (Claude, sole judge) -- STAGE1_SUBPLAN.md v2 -> v3

Panel: codex + antigravity (r2 coding focus). Both grounded; I verified each
load-bearing claim against the real files. The panel has CONVERGED -- r2 items are
implementability cleanups, not new structural defects, and both agents independently
recommend the same refined shape.

ACCEPTED (folded into v3):
1. [CONFIRMED] `outline_system` = the LEGACY sentinel `_otr_outline._SYSTEM_PROMPT`
   (:532). Grounded: comments :1045/:1826 ("the new per-stage prompts replace the
   legacy _SYSTEM_PROMPT"); the model is actually sent `_MACRO/_PHASE/_BEAT_SYSTEM_PROMPT`
   (:1868/:1996/:2101); :1847 uses `resolved is _SYSTEM_PROMPT` as a modern-profile
   sentinel. FOLD: do NOT author `outline_system`; author `outline_macro_system` /
   `outline_phase_system` / `outline_beat_system` from the real sent constants
   (:1102/:1115/:1130). The sentinel stays Python.
2. [CONFIRMED] Byte-identity via RUNTIME-IMPORT equality (import the node module,
   compare the real constant to `pack[seam]`), not module-level AST -- robust to
   implicit concat / `+`-joins, and the suite already imports node modules.
3. [CONFIRMED] coda: examples are appended UNCONDITIONALLY (:3407
   `_NEWS_CODA_SYSTEM + _NEWS_CODA_SYSTEM_V2_EXAMPLES`). FOLD: author ONE pre-joined
   `coda_system` seam (no split). Conditional composites (outro resolved tail :3517,
   an inline function-body literal) are NOT authored -- they stay Python like
   line_grounding. Author only clean module-level, statically-sent constants.
4. [CONFIRMED] Loader hardening: `read_text(encoding="utf-8")`; wrap OSError /
   UnicodeDecodeError / JSONDecodeError in typed errors; `object_pairs_hook` dup-key
   rejection (fires on nested `prompt_stages` too); reject whitespace-only values.
5. [ACCEPTED] StoryPack dataclass: explicit defaults for all optional/inert fields
   (avoid `StoryPack(**data)` TypeError); `REQUIRED_TOP_LEVEL = {source_bank_id,
   story_model_id, story_pipeline_id, schema_version, prompt_stages}`; exact types.
6. [ACCEPTED] Exception hierarchy: `StoryPackError` base + `StoryPackNotFoundError`,
   `StoryPackParseError`, `StoryPackValidationError`, `UnknownSeamError`.
7. [ACCEPTED] Allowlist: `PRODUCTION_SEAM_ALLOWLIST` = EXACTLY the Stage 1 authored
   seams (cut reserved future names -- they let unpinned seams pass); add an
   exact-key-set test asserting the science pack's seam keys == the authored set.
8. [ACCEPTED] `_PACK_CACHE` (load+parse each pack at most once) -- matters at the
   Stage 1b consumer; harmless while dormant.
9. [ACCEPTED] Dormancy guard TEST: assert no production file imports/calls
   `load_pack`/`get_pack_prompt*` in Stage 1 (proves "no behavior change").
10.[CONFIRMED] Stage 1b identity cleanup must fix ALL modern-prompt `is` checks:
   `test_creative_prompt_router.py:62` AND `:103`, `test_audio_c7_clamp_counter.py:52`,
   and `_otr_outline.py:1847`. Pilot seam = `line_composer_system` (sent directly,
   no identity logic) -- codex + antigravity agree.
11.[ACCEPTED] Workflow no-diff = a chunk VERIFICATION step (`git diff --quiet --
   workflows/otr_scifi_16gb_full.json`), not a brittle pinned-sha pytest.

REJECTED / CORRECTED:
- antigravity "refactor the outro tail to a module constant to author it" -> instead
  DON'T author the conditional tail in Stage 1 (simpler, no production edit). Correct
  per the "author only clean static constants" rule.
- pydantic "unavailable" rationale (codex): pydantic IS imported directly elsewhere
  (`_otr_casting.py:67`, `_otr_dramatic_state.py:38`). Keep stdlib validator on its
  real merit (tiny, zero-dep, v1/v2-agnostic, trivially auditable), not availability.

CONVERGENCE CALL: r1 + r2 (both agents, all claims grounded) converged. The residual
risk is the LIVE WIRING, which is quarantined into Stage 1b and will get its OWN
kibitz + Fable gate at build time. Stopping the arc here per operator "stop at
convergence" -- r3 (wiring) / r4 would review a wiring scope this plan deliberately
defers. Next: Fable structural gate on the Stage 1 (Chunk 1) plan, then code.

Agent calls this arc: 4 (codex + antigravity x r1, r2).
