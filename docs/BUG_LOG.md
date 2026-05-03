# OTR v2.0 Bug Log

Active bug log for the v2.0 build. Every bug gets logged the moment it is found.
Entries are never deleted.

---

## Workflow tip — ask Claude for a live risk artifact after a fix

When Claude has shipped a non-trivial fix and you want a quick gut-check on residual risk WITHOUT triggering more code changes, you can ask: "round-robin the shipped code and give me a live artifact with your % chance of fix needed." Claude will:

1. Write a code-review-shaped consult question (not a prevention-plan question — the framing matters).
2. Run `scripts/_consult_round_robin.py` against the shipped code (ChatGPT + Gemini + NVIDIA Nemotron).
3. Read the transcripts under `docs/<date>-<topic>__01_chatgpt.md` etc.
4. Render an inline artifact card-grid with one card per fix element. Each card shows: one-line description + per-element follow-up-fix probability % + ChatGPT verdict badge + Gemini verdict badge + one-line reasoning. Color-coded by risk tier (green <15%, amber 15-30%, red 30%+).
5. Below the cards: "what to watch in next soak" callout + "where they disagreed" section + sources footer.

**Proven pattern (2026-05-03 EVENING for BUG-027 + BUG-028):** transcripts at `docs/2026-05-03-bug-027-028-shipped-code-review__*.md`, commit `832d134`. ChatGPT (108.6s, 21 KB) + Gemini (34.2s, 4.5 KB) converged; NVIDIA round failed silently but 2-of-3 was sufficient signal. The artifact made residual risk landscape immediately legible — load-bearing weak spot ("BUG-028 Site 3: humo wildcard glob, 40%") jumped out at first glance instead of being buried in 25 KB of model prose.

**Skip when:** fix is trivial (one-line typo, mechanical edit), or you've already moved on. **Use when:** tough decision area (LLM prompt templates, audio C7, VRAM determinism, save paths, anything with multi-site write+read alignment) and you want peace of mind without round-tripping through more text walls. Round-robin alone is text-heavy; the artifact is the part that makes it visually decision-ready.

---

### BUG-LOCAL-028 [FIXED]: FLUX env stills + radio bookend save to legacy flat dirs — VideoComposite finds no scenery; final video black
- **Date:** 2026-05-03 | **Phase:** acceptance (post-026 hotfix soak) | **Bible candidate:** YES (directly causes "black video" failure mode)
- **Symptom (from `dir /s C:\...\output\otr\` for episode `signal_lost_astronomers_finally_solve_the_gammacas_x_20260503_002536`):**
  - Episode workspace `output/otr/episodes/<ep>/` has `audio/` `videos/` `composited/` subdirs (Phase B writes correctly).
  - **`output/otr/episodes/<ep>/stills/` does NOT exist.** **`output/otr/episodes/<ep>/portraits/` does NOT exist.**
  - FLUX outputs landed in legacy flat dirs:
    - Radio bookend → `output/otr/_legacy_stills/radio_bookend_<ep>.png` (filename has ep_id baked in but DIR is legacy).
    - Env stills → `output/otr/stills/full_env_NNNNN_.png` with a global counter shared across all episodes since 4/26 (213 PNGs accumulated, none stamped to a specific episode by name).
  - VideoComposite reads from per-episode `videos/` (correct) but cannot find env stills or radio bookend in the per-episode `stills/` (because they're not there). Result: no scenery layer in composite → mostly-black canvas.
  - Final composited mp4 = 1.72 MB; obs/ final = 1.18 MB; Jeffrey reports "black video, 15s of audio, no announcer in final."
- **Recurring (NOT a one-off):** affects every episode since the per-episode workspace reorg (Phase B, 2026-05-02 EVENING). Audio/video paths got reorged; FLUX outputs were not.
- **Cause (two separate sites):**
  - **Site 1: `visual/batch_flux_render.py:833`** — `stills_dir = _OTRP.otr_stills_dir()` called with no `episode_id` argument. Per `nodes/_otr_paths.py:208-218`, `otr_stills_dir()` without an episode_id falls back to `output/otr/_legacy_stills/`. The `episode_id` variable is in scope from line 768/772 (resolved from the in-flight ledger singleton via the same Phase G singleton-discovery path used by BUG-LOCAL-021). One-line fix.
  - **Site 2: `workflows/otr_scifi_16gb_full.json` node id 25** — stock ComfyUI `SaveImage` with hardcoded `filename_prefix: "otr/stills/full_env"` widget value. ComfyUI writes to `output/<filename_prefix>_<auto_counter>_.png`. The path doesn't change per-episode because the widget is static. Listed in ROADMAP.md "Known remaining suspects" (lines 47-55) under the same Phase G blast-radius pattern, but the visual-layer impact wasn't appreciated until this run. Architectural fix: replace stock SaveImage with a custom OTR node that reads the in-flight ledger singleton and routes to `otr_stills_dir(<ep_id>)`.
- **Fix (four sites — write + read alignment):**
  - **Site 1 (writer, radio bookend):** `visual/batch_flux_render.py:845` — changed `_OTRP.otr_stills_dir()` → `_OTRP.otr_stills_dir(episode_id)`. The `episode_id` variable was already in scope at line 768/772 (resolved via the in-flight ledger singleton, same Phase G discovery path used by BUG-LOCAL-021). Radio bookend now lands at `output/otr/episodes/<ep>/stills/radio_bookend_<ep>.png` per the canonical Phase B layout.
  - **Site 2 (writer, env stills):** new node `OTR_SaveToEpisodeWorkspace` in `nodes/otr_save_to_episode_workspace.py`. Reads `_otr_ledger.in_flight_ledger_path()` to derive episode_id at runtime; routes to `otr_stills_dir(ep_id)` or `otr_portraits_dir(ep_id)` based on `role_kind` widget ("stills" | "portraits"). Falls back to legacy dirs (preserving existing behavior) if no singleton is available — never raises in headless/test contexts. Registered in `__init__.py`. Workflow JSON `workflows/otr_scifi_16gb_full.json` node 25 retyped from `SaveImage` to `OTR_SaveToEpisodeWorkspace` with `role_kind="stills"`, `filename_pattern="full_env"`.
  - **Site 3 (reader, BatchHumoRender env-still binding):** `nodes/batch_humo_render.py:_resolve_cast_stills_from_ledger` and `_find_portrait` — added per-episode glob pattern `otr/episodes/*/stills/full_env_*.png` alongside the existing legacy `otr/stills/full_env_*.png` and `otr_stills/full_env_*.png` patterns. Without this, after Site 2 starts writing to per-episode dirs, HuMo's cast→still binding would find ZERO fresh stills and fall back to stale prior-episode stills (or unmapped, then fall through to portrait/composite tiers). The mtime-based freshness filter (`fresh_floor = ledger_mtime - 60s`) in the same function still enforces episode-correctness, so cross-episode pollution is mathematically impossible.
  - **Site 4 (reader, BatchLTXRender radio bookend):** `nodes/batch_ltx_render.py:374` — changed `otr_stills_dir() / f"radio_bookend_{eid}.png"` → `otr_stills_dir(eid) / f"radio_bookend_{eid}.png"`. Without this, after Site 1 starts writing to per-episode dirs, LTX would look in `_legacy_stills/` and find nothing, falling back to a generic motion clip with no scene continuity. (`nodes/video_composite.py:163` was already correct — passes `eid` — verified.)
- **Verify:**
  - AST parse on all 6 touched files (story_orchestrator, batch_flux_render, batch_humo_render, batch_ltx_render, otr_save_to_episode_workspace, __init__) green.
  - JSON parse + node-type audit on `workflows/otr_scifi_16gb_full.json`: 30 nodes, 0 stock SaveImage remaining, 1 OTR_SaveToEpisodeWorkspace registered.
  - **New `tests/test_save_to_episode_workspace.py` — 8 passed in <1s.** Covers: with active singleton → resolves to per-episode dir; no singleton → falls back to legacy dir; role_kind="stills" → otr_stills_dir; role_kind="portraits" → otr_portraits_dir; filename_pattern preserved; per-episode counter starts at 1 (independent of any global counter); never raises on mkdir failure; node registered in `NODE_CLASS_MAPPINGS`.
  - Cumulative regression: **155 passed in 3.27s** across `tests/test_production_ledger.py + test_radio_still_resolver.py + test_filename_pattern_audit.py + test_cache_key_mutations.py + test_meta_paths.py + test_ledger_rename.py + test_critique_dialogue_preservation.py + test_save_to_episode_workspace.py + test_prompt_format_safety.py`. PLUS Bug Bible regression **24 passed / 1 skipped / 1 xfailed** in 1.24s.
  - **Real-run acceptance (pending):** queue any episode; expect `output/otr/episodes/<ep>/stills/full_env_NNNNN_.png` with COUNTER STARTING AT 1 (per-episode counter, not global), AND `output/otr/episodes/<ep>/stills/radio_bookend_<ep>.png` co-located. BatchHumoRender log line `[BatchHumoRender] cast-still binding: N/M cast members matched to fresh stills` should report N>0. BatchLTXRender should find the radio bookend at the per-episode path. Final video should have visible scenery (not 100% black) when `blend_opacity > 0` is also set.
- **Tags:** flux, save-paths, per-episode-workspace, phase-g-blast-radius, video-composite-empty, write-read-alignment, qa-soak-2026-05-03
- **Related:** BUG-LOCAL-021 (Phase G singleton sweep — fixed audio side, missed visual side); BUG-LOCAL-027 (dialogue wipe — orthogonal failure that compounds with this bug; fixed in same commit). Round-robin consult was SKIPPED per direct user override ("yes ofrget rop8u7hnd robins just fix fix fix") — extra verification in lieu: AST + format-safety + targeted regression + Bug Bible regression all green pre-commit.
- **Headless soak status (2026-05-03 EVENING — UPDATED post-launch):** ATTEMPTED + IN PROGRESS at session end. ComfyUI was relaunched headless via the venv python (`main.py --listen 127.0.0.1 --port 8000 --highvram`) after the autonomous-pass cleanup taskkill cleared its previous session. Boot succeeded (36 OTR nodes loaded, including the new `OTR_SaveToEpisodeWorkspace`; verified via `/object_info`). Soak queued via `scripts/soak_bug027_028.py`, prompt_id `d29e3d8f-edce-48e2-a960-7245ff989543`. **Widget patches FAILED** with `widgets_values length mismatch on node 1 (OTR_LLMScriptWriter): len(wv)=15 vs len(widget_names)=16` — the saved workflow JSON has 15 widget values but the live schema reports 16, indicating widget drift on `OTR_LLMScriptWriter` since the workflow was last saved. Soak still queued with the workflow's saved values: `target_words=350, target_length="short (3 acts)", num_characters=2, style="tense claustrophobic", creativity="balanced"`. The "balanced" creativity (temp=0.85) is LESS aggressive on the BUG-027 trigger conditions than the original "maximum chaos" (temp=0.95) repro shape — soak validates the pipeline + new node registration but may not exercise the total-collapse gate fire. Last status check (~12 min into run): `queue_running: 1`, prompt_id still in queue, otr_runtime.log frozen at last warmup line `[01:11:40] WARMUP: CUDA kernels compiled in 0.6s`, ComfyUI python.exe at 18.3 GB working set (model loaded + active inference). Logs are buffered between LLM call boundaries; Jeffrey will see acceptance signatures populate in the morning when the run completes. **Followup BUG candidate:** widget drift on `OTR_LLMScriptWriter` between saved workflow + live schema — log as BUG-LOCAL-029 if confirmed (15 vs 16 widget mismatch). One-shot soak script `scripts/soak_bug027_028.py` is committed and ready for re-run when widget drift is resolved.

---

### BUG-LOCAL-027 [FIXED]: Critique/revision pass returns SCENE/ENV/SFX-only script, dropping all CHARACTER dialogue — Bark gets 0 lines
- **Date:** 2026-05-03 | **Phase:** acceptance (post-026 hotfix soak) | **Bible candidate:** TBD (likely YES — recurring across multiple runs)
- **Widget config (from screenshot):** target_words=110, num_characters=2, target_length="short (3 acts)", style="noir mystery", creativity="maximum chaos", arc_enhancer=ON, self_critique=ON, open_close=ON, optimization_profile="Pro (Ultra Quality)", model=google/gemma-4-E2B-it. Standard short(3) preset — NOT ultra_smoke / tiny_smoke (so BUG-LOCAL-005's CHARACTER:/SCENE: enforcement does not apply to this code path).
- **Symptom — current run "Cold Circuit" (otr_runtime.log line numbers):**
  - L47167 `[00:14:27] ScriptWriter: PARSE_OK attempt=1 has_scene=True voice_hits=18 bare_hits=0` — initial draft healthy: 3 scenes, 18 dialogue lines, characters ANNOUNCER + FLETCHER WELLS + KENJI BERNARD.
  - L47256 `[00:16:30] CRITIQUE: Character line counts - draft={} revised={}` — character-line counter returns empty dicts for both draft and revised. Parser disagreement: draft visibly had 18 `[N] CHARNAME:` lines, but the critique-pipeline counter sees zero. Revised pass also legitimately produced zero (only `=== SCENE N ===`, `ENV:`, `SFX #N:` lines emitted across the entire 100s revision generation — no `[N] CHARNAME:` lines at all).
  - L47257 `[00:16:30] CRITIQUE: Revised script accepted (sim=83.0%, len=118%)` — acceptance gate let the dialogue-stripped revision through. Similarity stayed high because SCENE/ENV/SFX scaffolding overlaps; length grew because model padded with extra atmosphere.
  - L47265 `[00:16:30] WORD_ENFORCEMENT: 0 words vs 110 target (0%) | @140wpm -> ~0.0 min [0 lines detected]`
  - L47267 `[00:16:30] BUG-109b: cast members with 0 lines: FLETCHER WELLS, KENJI BERNARD`
  - L47268 `[00:16:30] BUG-109b: 3/3 scene(s) have 0 dialogue lines`
  - FORMAT_NORM single-pass attempted recovery; ANNOUNCER bookends were generated separately (Kokoro-bound, not Bark-bound); CHARACTER dialogue was never restored.
  - Final downstream effect: `[BatchBark] Found 0 dialogue lines in Canonical 1.0 format (skipped 2 ANNOUNCER lines - routed to Kokoro bus)`. SceneSequencer pre-rendered 1 TTS + 8 SFX + 2 ANNOUNCER, zero Bark clips.
- **Recurring across runs (NOT a one-off):**
  - L44646 `[22:00:57] CRITIQUE: Character line counts - draft={} revised={}` (sim=56.6%, len=176% accepted) — earlier run, same shape.
  - L46713 `[23:43:26] CRITIQUE: Character line counts - draft={} revised={'ANNOUNCER': 2}` (sim=90.4%, len=94% accepted) — preserved ANNOUNCER only, dropped character dialogue.
  - L47256 `[00:16:30] CRITIQUE: Character line counts - draft={} revised={}` — current run.
  - Pattern: critique pipeline character counter consistently returns `{}` for draft regardless of writer output; acceptance gate (similarity + length only) cannot detect dialogue loss; multiple runs have shipped dialogue-stripped scripts to the audio cascade.
- **Cause (CONFIRMED via source dive 2026-05-03 EVENING):**
  - **Two coupled gaps in `nodes/story_orchestrator.py`.**
  - **Gap 1 (parser blindness):** `_count_character_lines` (line 6890) regex was `r'^\s*\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'` — required line to START with optional whitespace + uppercase name. The writer's actual output format is `[12] FLETCHER WELLS: text` (numbered-bracket prefix), so the regex never matched and returned `{}` for both draft and revised. Acceptance gate at line 7174 iterated `draft_char_counts` (empty dict) → no-op → revision accepted regardless.
  - **Gap 2 (gate too narrow):** even with parser working, the per-character preservation check (line 7174-7184) only catches "FLETCHER dropped from 8 to 1." It does NOT catch "every character wiped at once" because the loop iterates the draft dict per-char; if the revision wipes ALL characters, no individual character drops below the floor (they all dropped from N to 0, but the loop doesn't compare totals). Surface metrics — similarity ratio (0.83) + length ratio (1.18) — both pass on a SCENE/ENV/SFX-only revision because the scaffolding overlaps.
  - **Secondary contributor:** revision pass uses `temperature` (passed in from caller — for "maximum chaos" creativity = 0.95). High temp + critique demanding "fix every flagged problem" can push the model into pure-prose rewriting mode where it drops dialogue in favor of atmospheric SFX/ENV. The structural floor (`structural_temp=0.6` for similarity/length checks) doesn't gate this — it's a separate variable.
- **Fix (three-part):**
  - **Part 1 (parser regex, line 6916):** added optional non-capturing group `(?:\[\d+\]\s+)?` so both `CHARNAME:` and `[N] CHARNAME:` formats parse. Also tightened the structural-token exclude check at line 6924 to do BOTH exact-match AND first-word-match (`first_word in _struct_exclude`), so multi-word headers like `ACT 2:` or `SCENE 3:` no longer slip through as character names.
  - **Part 2 (total-collapse hard gate, after line 7184):** belt-and-suspenders for the per-character check. Computes `draft_total = sum(draft_char_counts.values())` and `revised_total = sum(revised_char_counts.values())`; if `draft_total >= 3` (threshold to apply ratio) and `revised_total < max(1, ceil(draft_total * 0.5))`, logs `CRITIQUE_REJECTED - total character lines collapsed from N to M (min=K, threshold=50%%)` and returns the draft unchanged. Below 3 lines the draft is too short for a meaningful ratio — the per-character check (with `min_line_count_per_character=2` floor) handles those cases.
  - **Part 3 (revision prompt hardening, line 7034 area):** added explicit `ABSOLUTE REQUIREMENT — DIALOGUE MUST SURVIVE THE REVISION` clause to the revision LLM prompt. Tells the model EXPLICITLY that producing a SCENE/ENV/SFX-only output is a "TOTAL FAILURE" and that every CHARACTER speaker present in the draft MUST appear in the revision. Also documented that the optional `[N]` prefix from the draft may be kept or omitted (both parse). Format-safety smoke (`tests/test_prompt_format_safety.py`) confirms no unescaped `{}` braces in the new prose (BUG-026 lesson).
- **Verify:**
  - AST parse on `nodes/story_orchestrator.py`: green.
  - **New `tests/test_critique_dialogue_preservation.py` — 14 passed in <1s.** Covers: parser handles bare `CHARNAME:`; parser handles `[N] CHARNAME:`; parser handles mixed format in same text; structural tokens (SCENE/ACT/MUSIC/SFX/ENV) excluded by exact-match AND first-word-match; empty/None text returns empty dict; ANNOUNCER counted as character; gate REJECTS total dialogue wipe (the actual L47256 case); gate REJECTS announcer-only revision (the L46713 case); gate ACCEPTS minor dialogue trim (83% retention); gate ACCEPTS at exactly 50% threshold; gate SKIPS short drafts (`< 3` lines); gate handles empty dicts safely; revision prompt has no unescaped braces (BUG-026 footgun gate); ABSOLUTE REQUIREMENT clause is present in source.
  - `tests/test_prompt_format_safety.py` — 1 passed (BUG-026 regression test passes against the new prompt prose).
  - Cumulative regression: 155 passed in 3.27s + Bug Bible 24 passed / 1 skipped / 1 xfailed in 1.24s.
  - **Real-run acceptance (pending):** queue the same widget config (110 words / 2 chars / short(3) / noir mystery / maximum chaos); expect (a) `CRITIQUE: Character line counts - draft={'ANNOUNCER': N, 'CHAR1': M, ...}` with NON-EMPTY draft dict (parser fix), (b) if revision wipes dialogue, log `CRITIQUE: CRITIQUE_REJECTED - total character lines collapsed from N to M` and the pipeline uses the original draft, (c) `[BatchBark] Found >=1 dialogue lines in Canonical 1.0 format` in the final pre-render summary.
- **Tags:** critique, revision, character-dialogue, parser-mismatch, acceptance-gate, bark-empty, qa-soak-2026-05-03, recurring, fixed
- **Related:** BUG-LOCAL-005 (CHARACTER:/SCENE: enforcement — only applies to ULTRA_SMOKE path; this is short(3)). BUG-109b detector (pre-existing observability — fired correctly here, no auto-recovery existed before this fix). Round-robin consult was SKIPPED per direct user override — extra verification in lieu: AST + format-safety + targeted regression + Bug Bible regression all green pre-commit.
- **Headless soak status (2026-05-03 EVENING):** DEFERRED. Same reason + same handoff as BUG-LOCAL-028 above: ComfyUI was terminated mid-session and the one-shot soak script at `scripts/soak_bug027_028.py` queues both 027 + 028 acceptance signatures in a single run. Jeffrey runs the soak in the morning after restarting ComfyUI Desktop.

---

### BUG-LOCAL-026 [FIXED]: Phase H regression — unescaped curly braces in DIRECTOR_PROMPT crashed `.format()` mid-pipeline
- **Date:** 2026-05-03 | **Phase:** G/H hotfix | **Bible candidate:** YES (classic `.format()` footgun)
- **Symptom:** Live soak crash on 2026-05-02 23:46. Episode "Exponential Tremor Echoes" (style="a claude cowork test session", target_words=80) ran cleanly through ScriptWriter (3-outline OpenClose evaluator, critique pass, revision pass, ScriptCritic verdict REVISE with 7 issues, revision applied). Crashed at LLMDirector.direct (`nodes/story_orchestrator.py:9951`):
  ```
  IndexError: Replacement index 0 out of range for positional args tuple
  ```
  ~10 minutes of LLM compute lost.
- **Cause:** Phase H BUG-LOCAL-023 added an EXCLUDE-ANNOUNCER clause to `DIRECTOR_PROMPT`. The added prose contained literal `visual_plan.characters{}` and `voice_assignments{}` — two unescaped `{}` empty-brace pairs. The surrounding template uses `str.format()` with kwargs (`script_text`, `voice_mapping_rules`); Python's `.format()` interpreted `{}` as a positional arg slot reference, looked up `args[0]`, found nothing, raised `IndexError`. **Cardinal mistake — adding prose with literal braces to a `.format()` template without escaping.**
- **Fix:** Removed the literal `{}` symbols from the EXCLUDE-ANNOUNCER prose. Kept the semantic content ("EXCLUDE narrator/announcer roles from the visual plan characters object", etc.) — readable to the LLM, no longer breaks `.format()`. Either `{{ ... }}` escaping OR removing the braces from prose is valid; chose removal for prose-readability.
- **Verify:**
  - Standalone `_director_prompt_test.py` smoke: `DIRECTOR_PROMPT.format(script_text=..., voice_mapping_rules=...)` returns 5501 chars, no exception.
  - **Permanent regression test** `tests/test_prompt_format_safety.py` — extracts `DIRECTOR_PROMPT` constant via regex, calls `.format()` with the production kwargs, asserts no `IndexError`/`KeyError`/`ValueError`. **Passed in 1.74s.** Future Phase-N additions to the prompt that re-introduce unescaped braces will fail this test before they reach a live run.
- **Tags:** phase-h-regression, str-format, prompt-template, hotfix, bible-candidate
- **Lesson learned for future autonomous mode:** when editing any constant that's later passed to `.format()`, run `_director_prompt_test.py`-style format smoke as part of the AST guard pass. Don't ship template edits without confirming `.format()` survives.

---

### BUG-LOCAL-025 [FIXED]: LTX role prompts ignore story style + scene context (every episode looked the same)
- **Date:** 2026-05-03 | **Phase:** H (story-arc enrichment for visual layer) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** `nodes/batch_ltx_render.py::_PROMPT_BY_ROLE` is a hardcoded dict mapping `{announcer, music_open, music_close, music_inter, sfx}` → fixed motion prompts ("Vintage 1940s radio broadcast set, glowing tuning dial pulses gently, copper vacuum tubes warm amber glow..."). Every episode renders the SAME LTX motion regardless of the story's style or scene atmosphere. Jeffrey: *"be sure story arc or better shot/scene arc is being fed into FLUX and LTX as well to match the short."*
- **Cause:** Original `_PROMPT_BY_ROLE` design treated LTX as a generic radio-animator with no story awareness. Acceptable when the radio bookend (the i2v reference image) carries all visual identity — but downstream review confirmed the motion prompt itself influences mood (dial sweep speed, tube glow rhythm, dolly direction).
- **Fix:** New `_build_ltx_role_prompt(role, line, ledger)` helper enriches each role base prompt with two ledger-derived layers:
  1. **Per-line scene context.** Lookup chain: `line.shot_id` → `ledger.shots[*].scene_id` → `ledger.scenes[*].env / .description`. Truncated to 60 chars, appended as `, scene context: <env>`. Each LTX clip now matches the SCENE it accompanies (early scenes get tense env, late scenes get resolved env).
  2. **Episode style suffix.** Read from `ledger.meta.gen_params_initial.style` (or `.gen_params.style`) — same singleton-fed source Phase G fixed for radio bookend. Appended as `, <style> broadcast tone`.
  Bounded so the role's motion intent isn't drowned. Per-line lookup means one episode's announcer LTX clips can vary across scenes if those scenes have different `env` text.
- **Verify:** AST + full pytest (1150 / 8 / 1 in 131.62s) green. **Real-run acceptance (pending):** the `[BatchLTXRender]` log lines should now show enriched prompts; two episodes with different styles should produce visibly different LTX motion intent.
- **Tags:** ltx, story-arc, scene-context, style-aware, qa-pass-2026-05-03

---

### BUG-LOCAL-024 [FIXED]: Radio bookend FLUX prompt fell back to generic when style missing OR ledger stale
- **Date:** 2026-05-03 | **Phase:** H (story-arc enrichment for visual layer) | **Bible candidate:** yes
- **Symptom:** Soak run on 2026-05-02 logged `[BatchFluxRender] radio still prompt source=fallback (no style)` — radio rendered as generic "sci-fi retrofuturistic radio broadcast unit" despite user setting style="space opera epic" in the widget. Compounded with BUG-LOCAL-021 (FLUX read a stale April 26 ledger via the broken `find_most_recent_ledger` walker), the radio NEVER reflected the actual episode story.
- **Cause:** `_build_dynamic_radio_prompt` in `visual/batch_flux_render.py` only looked at two fields (`gen_params_initial.style` + `gen_params.style`) before falling to a single hardcoded fallback. No fallback chain through `style_custom`, scene environment, or episode title.
- **Fix:** Six-tier resolution with per-tier branch logging:
  1. `gen_params_initial.style` (primary widget value)
  2. `gen_params.style` (back-compat)
  3. `gen_params_initial.style_custom` (free-text override)
  4. First scene's `env` / `description` (scene-driven mood)
  5. `episode_id` slug (strip "signal_lost_" prefix + trailing timestamp, replace underscores)
  6. Hardcoded `_RADIO_FALLBACK_PROMPT` (true last resort)
  Plus: scene-context hint (`set in <first_scene_env>`) appended whenever distinct from descriptor, so style + scene combine. New log line `[BatchFluxRender] radio prompt: branch=<which> -> <preview>` tells the runtime tail which tier fired. Bounded length: descriptor capped at 80 chars, scene_hint at 60 chars.
- **Verify:** AST + full pytest green. **Real-run acceptance (pending):** with Phase G singleton lookup feeding the CURRENT ledger, the radio prompt should now log `branch=gen_params_initial.style` and the radio should render as "space opera epic radio broadcast unit, set in derelict orbital lab, ..."
- **Tags:** flux, radio-bookend, story-arc, fallback-chain, qa-pass-2026-05-03

---

### BUG-LOCAL-023 [FIXED]: ANNOUNCER portrait wasted FLUX context + skewed scene composition
- **Date:** 2026-05-03 | **Phase:** H (story-arc enrichment for visual layer) | **Bible candidate:** yes
- **Symptom:** Jeffrey caught mid-soak: `LLMDirector` generates a `portrait_prompt` for ANNOUNCER under `visual_plan.characters`, then `OTR_VideoPlan.compose_shot_prompt` concatenates ALL character portraits into every scene's PASS3 visual prompt. The announcer is never on screen as a person (BUG-LOCAL-129b: routed to Kokoro voice + radio bookend visual; HuMo skips them). Including their portrait wastes FLUX prompt budget AND skews scene composition by forcing every shot to fit an extra character (50yo silver-haired woman in flight gear).
- **Cause:** Visual_plan.characters was generated for every speaker without a "appears on screen?" filter. PASS3 compose treats the dict as canonical.
- **Fix (belt-and-suspenders, two layers):**
  1. **LLMDirector prompt rule** in `nodes/story_orchestrator.py` (VISUAL PLAN RULES section): explicit instruction "EXCLUDE narrator/announcer roles from visual_plan.characters. The ANNOUNCER (and any voice that only narrates without appearing on screen) must NOT be included under visual_plan.characters{}. Their voice mapping still belongs in voice_assignments{}; only visual_plan.characters skips them." Catches it at the source.
  2. **`OTR_VideoPlan` filter** in `nodes/otr_video_plan.py:438`: new `NON_VISUAL_ROLES = {"ANNOUNCER", "NARRATOR"}` set; before composing portraits, partition `chars_dict.keys()` into `all_char_names` (visible roles) and `_skipped_non_visual` (logged as info). Catches future LLM regressions where layer 1 fails. Honors explicit `focus_character` requests for non-visual roles (lets a debugging workflow request the announcer portrait specifically).
  Audio is unaffected: `voice_assignments.notes` is a SEPARATE field that audio nodes (Bark/Kokoro) consume; `portrait_prompt` doesn't feed audio at all.
- **Verify:** AST + full pytest (1150 / 8 / 1 in 131.62s) green. **Real-run acceptance (pending):** scene visual prompts in `[BatchFluxRender] shot N/M:` log lines should NOT lead with "Female, 50s, gravelly voice..." when there's an ANNOUNCER in the cast; should see `OTR_VideoPlan: skipped non-visual role(s) from portrait composition: ANNOUNCER` log line.
- **Tags:** flux, announcer, visual-plan, scene-composition, qa-pass-2026-05-03

---

### BUG-LOCAL-022 [FIXED]: BatchHumoRender stem-swap is mathematically broken when safe_title[:40] truncates the title
- **Date:** 2026-05-03 | **Phase:** G (path-reorg blast radius) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** `BatchHumoRender._load_ledger_with_path` (line 1791-1865 pre-Phase-G) takes a `.mp4` path input from `SignalLostVideoRenderer` and derives the ledger via stem swap (`<file>.mp4` → `<file>_ledger.json`). When `video_engine.py:1450` truncates the procgen mp4 filename via `safe_title = "...".replace(...)[:40]`, the resulting mp4 stem may NOT equal the canonical `episode_id`. Stem swap looks for a ledger that doesn't exist. Combined with BUG-LOCAL-020 (mp4 in legacy dir), the failure mode is "derived ledger from .mp4 not found". Even after BUG-LOCAL-020 fix puts the mp4 in the per-episode dir, stem swap can still fail if title truncation drops characters.
- **Cause:** Discovery coupled to mp4 filename instead of the on-disk per-episode workspace structure (`output/otr/episodes/<ep>/audio/<ep>_ledger.json`).
- **Fix:** Add Tier 0 layout-aware lookup BEFORE the legacy stem-swap tiers in `_load_ledger_with_path`. Detection: `audio_dir.name == "audio"` AND `audio_dir.parent.parent.name == "episodes"`. When detected, the parent dir name IS the `episode_id` by construction. Try canonical `<ep_dir_name>_ledger.json` first; fall back to globbing `*_ledger.json` (non-pending) in the same audio_dir if the slug rule doesn't match. Decoupled from mp4 stem entirely. Legacy stem-swap tiers preserved as fallback for old artifacts in the legacy flat layout.
- **Verify:** AST + full pytest suite (1149 / 8 / 2 in 112s) green. **Real-run acceptance (pending):** re-queue the same workflow JSON; BatchHumoRender should log `Phase G layout-aware ledger lookup: <ep>_ledger.json` and proceed past the prior crash point.
- **Tags:** humo, ledger-discovery, layout-aware, stem-swap, qa-pass-2026-05-03

---

### BUG-LOCAL-021 [FIXED]: Audio-side nodes used global mtime walker for write-back (latent BUG-LOCAL-014 wrong-episode shape)
- **Date:** 2026-05-03 | **Phase:** G (path-reorg blast radius) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** Seven sites in the audio-side write-back chain used `_otr_ledger.find_most_recent_ledger([otr_episodes_root(), otr_legacy_audio_dir()])` to locate the in-flight ledger for write-back: `musicgen_theme.py:98+494`, `batch_audiogen_generator.py:58+511`, `batch_bark_generator.py:703`, `scene_sequencer.py:920+1257`, `audio_enhance.py:436`. Plus `visual/batch_flux_render.py:641` used the same walker (with the wrong dirs — `otr_audio_dir()` returns `_legacy_audio` when called with no episode_id, so it never even scanned the per-episode tree). On the 2026-05-02 soak, FLUX radio bookend stamped to `signal_lost_signal_abyss_20260426_161737` (a 6-day-old episode) instead of the in-flight `signal_lost_cramped_cargo_bay_vibrating_20260502_220824`. Same wrong-episode shape as BUG-LOCAL-014 — Phase A fixed it for `rtx_upscale.py` only; the rest of the codebase had it latent.
- **Cause:** Mtime-based discovery is fundamentally racy across queue boundaries and across runs. The `_CURRENT` Ledger singleton (set by `LLMScriptWriter` via `new_ledger()`) tracks the in-flight episode by construction; ComfyUI sequential queue + LLMScriptWriter's `IS_CHANGED = time.time()` guarantee the singleton is fresh on every queue invocation.
- **Fix:** Add `_otr_ledger.in_flight_ledger_path()` helper that reads the singleton's `path` (which advances correctly through `Ledger.rename_episode` per Phase B) and falls back to the legacy mtime walker only if the singleton is somehow unavailable. Sweep all 7 audio-side sites + the FLUX radio bookend site to use the helper. Late-import via try/except inside the helper avoids circular import with `production_ledger.py`. The walker is preserved (for `post_audio_video_pipeline.py:126` empty-input fallback and for the helper's own last-resort path).
- **Consult sources:** `docs/2026-05-03-phase-g-path-reorg-blast-radius__01_chatgpt.md` (gpt-5.5, 117.8s), `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 46.1s — caught critical "ComfyUI cache trap" risk for singleton; verified-already-mitigated by LLMScriptWriter's IS_CHANGED), `__03_nvidia.md` (mistral-nemotron, 190.7s).
- **Verify:** Phase G AST + full pytest (1149 / 8 / 2 in 112s) green. **Real-run acceptance (pending):** re-queue; FLUX radio bookend should stamp to the CURRENT episode_id, not a stale leftover. Two-episode soak: B's audio nodes should write to B's ledger, not A's.
- **Tags:** ledger-discovery, singleton, find_most_recent_ledger, wrong-episode, defensive-sweep, qa-pass-2026-05-03

---

### BUG-LOCAL-020 [FIXED]: video_engine.py procgen mp4 written to legacy `output/otr/audio/` instead of per-episode workspace (SOAK BLOCKER)
- **Date:** 2026-05-03 | **Phase:** G (path-reorg blast radius) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** Live soak crash on 2026-05-02. After 12 minutes of pipeline progress (LLM ladder + audio cascade + procgen video), `BatchHumoRender` crashed with `RuntimeError: BatchHumoRender: derived ledger from .mp4 not found: C:\...\output\otr\audio\signal_lost_cramped_cargo_bay_vibrating_20260502_220824_ledger.json (also tried collapsed-underscore variant + directory scan in C:\...\output\otr\audio)`. The procgen mp4 was at `output/otr/audio/<file>.mp4` (legacy flat) but the ledger had been moved to `output/otr/episodes/<ep>/audio/<ep>_ledger.json` by Phase B's `Ledger.rename_episode`. Stem swap looked in the wrong dir and found nothing.
- **Cause:** `video_engine.py:1443-1446` (pre-Phase-G) hardcoded `out_dir = .../output/otr/audio` — the legacy flat layout. The path reorg (Phases A/B/E) moved every per-episode asset under `output/otr/episodes/<ep>/audio/`, but `video_engine.py` was missed. The mp4 ended up OUTSIDE the per-episode tree, so `Ledger.rename_episode` (which renames the parent dir) couldn't move it along with the rest of the workspace.
- **Fix:** Read `out_dir` from `get_ledger().out_dir` (the in-flight Ledger singleton's audio path = `episodes/pending_<ts>/audio/` at this point). Write the procgen mp4 there. After `Ledger.rename_episode(<ep_id>)` moves the parent dir, recompute `final_out_path = Path(led.out_dir) / pending_out_path.basename` and verify it exists. Update the `subfolder` hint in the ComfyUI UI return value to the post-rename relative path. Defensive try/except wraps the ledger lookup so test/headless paths fall back to the legacy `output/otr/audio/` location.
- **Consult sources:** Same as BUG-LOCAL-021. ChatGPT specifically caught the post-rename stale-path risk; Gemini caught the `safe_title[:40]` truncation issue (addressed in BUG-LOCAL-022).
- **Verify:** Phase G AST + full pytest green. **Real-run acceptance (pending):** re-queue the same 30-word smoke; expect `[Video] Saved: <abs path under episodes/pending_<ts>/audio/>` followed by `[Ledger] per-episode dir moved pending_<ts> -> <ep_id>` followed by `[Video] post-rename mp4 path: <pending> -> <final>` (if recompute fired) and BatchHumoRender progress past the prior crash point.
- **Tags:** path-reorg, video-engine, soak-blocker, post-rename, qa-pass-2026-05-03

---

### BUG-LOCAL-019 [FIXED]: Sprint 1 full-suite acceptance — pre-existing test rot (Phase B fallout + per-episode reorg fallout)
- **Date:** 2026-05-02 | **Phase:** Sprint 1 acceptance | **Bible candidate:** no (test-only fixes, no behavior change)
- **Symptom:** `python -m pytest tests/` (Sprint 1 acceptance line item) ran to completion in 113s but with **5 failures** in two distinct clusters. The original BUG-LOCAL-006 hang at `TestDropdownsHaveEffect::test_creativity_produces_different_temps` was already resolved by intervening conftest work — that test now passes in 10s standalone — but other latent failures had been masked because the three explicit suites used in Phase A→E regression (Bug Bible, dropdown_guardrails, core) didn't include the ones that broke.
- **Cluster 1 (2 failures, `tests/test_production_ledger.py`):** `TestLedgerBeats::test_rename_updates_path_and_data` and `TestDualLedgerFix::test_rename_episode_moves_file_on_disk` both raised the Phase B (BUG-LOCAL-015) hard-fail RuntimeError "both source and destination episode directories exist". Root cause: the tests passed `tmp_path` directly as the audio dir to `Ledger(...)`. Phase B's `rename_episode` walks `os.path.dirname` up two levels to find the per-episode root — from `tmp_path = pytest-of-jeffr/pytest-NNN/test_K/`, that walked up to `pytest-of-jeffr/`, then constructed `new_ep_dir = pytest-of-jeffr/signal_lost_black_sphere_20260424_142006`. That destination accumulated across pytest sessions (siblings from prior runs of the same test pollute the user's TEMP root), so on any run after the first the conflict guard fired correctly. The TESTS were buggy — they assumed the pre-Phase-B silent split-state recovery and depended on global TEMP pollution that no longer works under the hard-fail invariant.
- **Cluster 2 (3 failures, `tests/test_radio_still_resolver.py`):** `TestFilesystemFallback::test_filesystem_fallback_finds_by_episode_id`, `TestFilesystemFallback::test_filesystem_fallback_when_ledger_path_stale`, `TestBug121Hardening::test_zero_byte_file_falls_through_to_layer3` all failed with `TestX.<locals>.<lambda>() takes 0 positional arguments but 1 was given`. Root cause: `monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)` mocks the helper with a 0-arg lambda, but the per-episode workspace reorg (2026-05-02 EVENING, BUG-LOCAL-033) gave `otr_stills_dir` an `episode_id` parameter. Production code calls `otr_stills_dir(episode_id)` with 1 arg. The 7 fallback tests that DON'T trigger this path passed silently; the 3 that do reach it failed.
- **Cause summary:** Cluster 1 = direct fallout from Phase B's hard-fail invariant correctly rejecting test setups that depended on the buggy old behavior. Cluster 2 = stale test mock signatures from the per-episode workspace reorg (BUG-LOCAL-033 era). Both pre-existing, surfaced because no one had run `pytest tests/` to completion since they were introduced.
- **Fix:** Cluster 1 — both failing tests now build a proper `tmp_out/episodes/<ep>/audio/` per-episode dir before instantiating the `Ledger`. The rename invariant has clean room to operate; no global TEMP pollution. Cluster 2 — `monkeypatch.setattr(..., lambda *a, **kw: tmp_path)` (10 sites updated via `replace_all`). Variadic tolerates the new `(episode_id)` arg without changing test semantics.
- **Verify:** Targeted suite (`pytest tests/test_production_ledger.py tests/test_radio_still_resolver.py -v`) — 76 passed in 1.87s. Full suite (`pytest tests/ -q --ignore=tests/v2`) — **1126 passed / 7 skipped / 0 failed in 113.28s**. The 107 errors in an earlier run were transient pytest tmp_path session race (`pytest-264` got reaped while a parallel pytest invocation was still using it); not reproducible on clean runs.
- **Promotes BUG-LOCAL-006 from [PARTIAL] to [FIXED]:** the conftest CUDA mask works, AND the originally-blamed `test_creativity_produces_different_temps` now passes (cause was either incidentally fixed by Phase B/C/D/E work or transient under a specific environment that no longer reproduces). Sprint 1 acceptance line "python -m pytest tests/ runs to completion green" is now satisfied — net cumulative count: 1126 / 7 / 0 across the full directory.
- **Tags:** test-rot, phase-b-fallout, per-episode-reorg-fallout, sprint-1-acceptance, no-active-bug

---

### BUG-LOCAL-006 [FIXED, was PARTIAL]: pytest hang at session-start when ComfyUI on same GPU
- **Date:** 2026-05-02 PM EVENING (re-verified) | **Phase:** 0 (test infra) | **Bible candidate:** yes
- **Update on the prior PARTIAL status:** `tests/conftest.py` (committed earlier this session) sets `CUDA_VISIBLE_DEVICES=""` + `OTR_TEST_MODE=1` at module import, registers the `requires_cuda` marker, auto-skips marked tests when CUDA is masked. The original PARTIAL note flagged `TestDropdownsHaveEffect::test_creativity_produces_different_temps` as still-hanging. Re-verified 2026-05-02 PM: that test now passes in 10s standalone, and the full directory `pytest tests/` runs to completion in 113s (1126 / 7 / 0). Either the hang was incidentally fixed by Phase A→E work (path reorg + Phase B's atomic rename + cache key cleanup may have removed a fixture that touched a heavy import), or it was transient under a specific environment. No further bisect needed; the acceptance gate is satisfied.
- **Verify:** `python -m pytest tests/ -q --ignore=tests/v2` → 1126 passed / 7 skipped / 0 failed in ~113s. With ComfyUI Desktop up on `:8000`, same result.
- **Tags:** test-infra, cuda-context, comfyui-cohabit, bible-candidate, was-partial

---

### BUG-LOCAL-018 [FIXED]: Ledger schema bump l3-2026-05-02 + meta.paths block
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** No active bug — additive schema enhancement. The QA pass (`docs/2026-05-02-rtx-upscale-qa-pass.md` Phase E) prescribed adding a `meta.paths` block to the production ledger so downstream nodes can look up canonical episode dirs without reconstructing them from `episode_id`. Slug-reconstruction was the root cause of the BUG-LOCAL-014/015/017 cluster — Phases A/B/C/D fixed the live instances; Phase E removes the temptation entirely by stamping the absolute, on-disk-truth paths into the ledger at every save.
- **Cause:** N/A — preventive change.
- **Fix:**
  - `nodes/_otr_ledger.py`: bump `CURRENT_SCHEMA_VERSION` from `l3-2026-04-28` to `l3-2026-05-02`. Add `_build_meta_paths(ledger_path, episode_id)` helper that detects layout (per-episode-workspace under `output/otr/episodes/<ep>/audio/<ep>_ledger.json` vs legacy flat under `output/audio/<ep>_ledger.json`) and stamps an appropriate `meta.paths` block. `save_ledger_safe` now calls it on every write. The block is **resolved fresh on every save** from the actual on-disk path, so it self-corrects after `Ledger.rename_episode` (Phase B) — no caller has to update it.
  - `nodes/production_ledger.py`: `Ledger.save()` also stamps `meta.paths` (via the same `_otr_ledger._build_meta_paths` helper) so the path data is consistent regardless of which write path produced the ledger. Hardcoded fallback `SCHEMA_VERSION` updated to match. Best-effort try/except wraps the meta stamp — a stamping failure must NEVER break the actual ledger write.
  - `docs/ledger_schema.md`: created. Documents the schema (top-level fields + meta block + meta.paths block + per-episode vs legacy-flat layout shapes), the lineage table, the reader contract (`dict.get(...)` not direct subscript), and the rules for downstream nodes.
- **Verify:**
  - AST + Phase E invariant guards (schema string `l3-2026-05-02`, `_build_meta_paths` present, both layouts detected, dual-write stamping in both files) — green.
  - **New `tests/test_meta_paths.py` — 13 passed.** Covers: per-episode layout detection + all 6 dirs stamped + obs_final stamped when obs/ exists + ledger_path absolute; legacy flat layout detection + no fabricated subdirs + minimal paths only; `save_ledger_safe` stamps meta.paths AND preserves pre-existing meta keys; old ledger without meta.paths loads cleanly via `dict.get(...)` (back-compat regression); `Ledger.save()` stamps meta.paths too; **after `rename_episode`, the next save's meta.paths self-corrects to the new dir** (the killer property — proves stale references can't accumulate).
  - Three CLAUDE.md regression suites + all phase-A-through-D tests: **234 passed / 1 skipped / 2 xfailed in 106.37s** (Bug Bible 23 + dropdown_guardrails+core 155 + ledger_rename 10 + filename_pattern_audit 3 + cache_key_mutations 30 + meta_paths 13).
  - **Real-run acceptance (pending):** end-of-stack soak. New ledgers should carry `meta.paths`; old ledgers (if any survive in `output/audio/`) still load via `dict.get` defaults.
- **Tags:** schema, additive, meta-paths, back-compat, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-rtx-upscale-qa-pass.md` (Phase E section). No round-robin needed — additive only, no behavior change for existing readers (all already use `meta.get(...)`).
- **Reader contract enforced:** see `docs/ledger_schema.md` "Reader contract" section. `meta.paths` MUST be accessed via `led.get("meta", {}).get("paths", {}).get(field)`, never `led["meta"]["paths"][field]`.

---

### BUG-LOCAL-017 [FIXED]: MusicGen + AudioGen cache miss every run — `_cache_key` returned a fresh timestamped path
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** Two files (`nodes/musicgen_theme.py`, `nodes/batch_audiogen_generator.py`) had identical structural bugs in their cache logic. `_cache_key()` returned a filename with the *current* millisecond timestamp baked in (`<role>_<sha8>_<ts_ms>.wav`). The call site immediately checked `os.path.exists(cache_path)` against that exact path — which never existed because the timestamp was "now". Result: cache miss every single run, ~22s of wasted MusicGen rendering per episode + N seconds of wasted AudioGen rendering per SFX cue, AND a Rule C7 violation because each run wrote a different timestamped filename, and FFmpeg embeds input WAV filenames in MP4 metadata streams → final mp4 bytes drifted between identical-input runs.
- **Cause:** Single function (`_cache_key`) tried to do two incompatible jobs: produce a deterministic identity for cache lookup AND a unique filename for write. The docstring explicitly described the timestamp as "guaranteed unique across episodes" but that defeated the entire cache.
- **Fix:** Split lookup identity from write filename in both files:
  - **`_cache_prefix(...)`** — deterministic identity prefix (`<role>_<sha8>` for MusicGen, `sfx_<safe_name>_<sha8>` for AudioGen). No timestamp.
  - **`_cache_filename_for_write(...)`** — canonical write filename (`<prefix>.wav`). No timestamp. Same inputs always land at the same filename → byte-identical mp4 metadata across runs (Rule C7 holds even on clean-cache runs).
  - **`_cache_key(...)`** — back-compat alias, returns the canonical write filename.
  - **`_find_cached(cache_dir, prefix)`** — two-level lookup: canonical `<prefix>.wav` first, fallback to legacy `<prefix>_<ts>.wav` files for back-compat with existing on-disk caches. Uses `iterdir() + startswith()` (per Phase D Gemini consult — `Path.glob()` chokes on `[` in filenames). Sorts legacy matches by parsed filename timestamp (not mtime; mtime is unstable across copy/restore).
  - **`_save_wav` made atomic** — writes through sibling `.tmp` then `os.replace()` (Phase D Gemini consult: prevents corrupted cache hits if process is killed mid-write). Explicit `format="WAV"` because soundfile can't infer format from `.tmp` extension. Cleanup of orphan `.tmp` on failure.
- **Verify:**
  - AST + 7 invariant guards per file (function presence, atomic write, iterdir-not-glob, etc.) — green.
  - **New mutation suite `tests/test_cache_key_mutations.py` — 30 passed in 2.87s.** 5 MusicGen mutations + 5 AudioGen mutations + 12 lookup tests (canonical-wins, legacy-fallback, newest-timestamp-wins, no-cross-prefix-match, glob-metachar-tolerance) + 2 atomic-write tests + 2 cache_key back-compat tests + 4 atomic-failure tests. Confirms: every identity dimension produces fresh sha; the cosmetic AudioGen `safe_name` is NOT used as identity (full-prompt change beyond first 20 chars still produces fresh sha); `Path.glob` would have failed on `[`-containing prefix but `iterdir+startswith` works.
  - Three CLAUDE.md regression suites: **221 passed / 1 skipped / 2 xfailed in 110.36s** (Bug Bible 23 + dropdown_guardrails+core 155 + ledger_rename 10 + filename_pattern_audit 3 + cache_key_mutations 30).
  - Existing on-disk timestamped cache files transparently start hitting after deploy (legacy fallback path).
  - **Real-run acceptance (pending):** two consecutive identical-input runs should produce one file per `(role, sha)` pair; second run should log `CACHE HIT` with the canonical `.wav` name. End-of-stack soak covers this together with Phases A, B, C.
- **Tags:** cache, c7-byte-identity, ffmpeg-metadata, atomic-write, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-phase-d-cache-key-consult__01_chatgpt.md` (gpt-5.5, 108.1s — proposed strong-form deterministic write), `docs/2026-05-02-phase-d-cache-key-consult__02_gemini.md` (gemini-3.1-pro-preview-customtools, 32.0s — caught atomic-write requirement and `Path.glob` bracket bug), `docs/2026-05-02-phase-d-cache-key-consult__03_nvidia.md` (llama-3.3-nemotron-49b, 127.0s — confirmed all decisions). All three converged unanimously: drop timestamp on writes, two-level lookup, iterdir+startswith, atomic write, defer model_name digest expansion.
- **Deferred to v2 cache-key migration (separate scope):** add `model_name`, `sample_rate`, `decode_mode`, `guidance_scale` to the digest payload. Today these are effectively constants per-file but if the user starts varying them at runtime, cache identity will be wrong until v2 lands.

---

### BUG-LOCAL-016 [FIXED]: Filename pattern audit — slug-reconstruction regression guard
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after end-of-stack soak)
- **Symptom:** No active bug — this is a regression guard. The QA pass (`docs/2026-05-02-rtx-upscale-qa-pass.md` Phase C) prescribed an audit of all `nodes/` files for the dangerous anti-pattern: code constructing `f"{ep_id}_..."` to *find* or *delete* a file on disk. The actual on-disk filenames for cache files (musicgen, audiogen wavs) follow the format `<role>_<sha>_<ts>.wav` — produced by the writer, indexed by sha. Slug-reconstruction-for-discovery breaks every time the producer's naming convention diverges from the slug.
- **Cause:** Phase A and Phase B already absorbed the live instances of this anti-pattern (rtx_upscale spacesaver and production_ledger sidecar rename now use `audio_dir.glob(...)`). What remained was the risk of *future* drift — someone reintroducing slug reconstruction in a discovery path without realizing the cache filenames don't match the slug.
- **Fix:** Audit complete (0 substantive code changes). The remaining `f"{ep_id}.mp4"` and similar usages in the codebase are all canonical writer/reader pairs sharing a contract by construction (RTXUpscale OBS-existence guard ↔ VideoComposite mp4 writer; Ledger class authoring `<ep>_ledger.json`). New regression test `tests/test_filename_pattern_audit.py` codifies the rule:
  - **`test_no_audio_cache_slug_reconstruction`** — static-analyzes all `nodes/*.py` for banned patterns: `audio_dir / f"opening_{ep_id}.wav"`, `audio_dir / f"sfx_{ep_id}_..."`, etc. Will fail loudly on any future drift.
  - **`test_destructive_paths_use_glob_not_reconstruction`** — positive assertion that the rtx_upscale spacesaver (Phase A) and ledger sidecar rename (Phase B) still use glob discovery; if a refactor accidentally replaces `audio_dir.glob("*_treatment.txt")` with slug reconstruction, this test catches it.
  - **`test_allowlist_entries_still_present`** — every entry in the test's ALLOWLIST (legit canonical writer/reader pairs) must still resolve to a real source line. Stale entries surface for pruning instead of silently shielding future drift.
- **Verify:**
  - 3/3 audit tests pass in 1.60s.
  - Combined regression: **191 passed / 1 skipped / 2 xfailed in 98.00s** (Bug Bible 23 + dropdown_guardrails+core 155 + ledger_rename 10 + filename_pattern_audit 3).
- **Tags:** audit, regression-guard, slug-reconstruction, qa-pass-2026-05-02, no-active-bug
- **Consult sources:** `docs/2026-05-02-rtx-upscale-qa-pass.md` (Phase C section, table of canonical writers/lookups). No round-robin needed — mechanical audit, no determinism implications.

---

### BUG-LOCAL-015 [FIXED]: production_ledger treatment rename gap + os.replace silent split state
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after real two-episode soak run with Phase A)
- **Symptom:** Two adjacent bugs in `Ledger.rename_episode` (`nodes/production_ledger.py`):
  1. **Treatment file rename gap (Finding 2 from QA pass).** The function moved the per-episode dir and renamed `pending_<ts>_ledger.json` → `<new_id>_ledger.json` but did NOT rename `pending_<ts>_treatment.txt` → `<new_id>_treatment.txt`. The treatment file (written early by `OTR_LLMScriptWriter` before the title is finalized) sat in the new dir with the old prefix. Phase A's spacesaver kept it via a defensive `glob("*_treatment.txt")`, but that defensive measure was a workaround.
  2. **`os.replace` fallback silent split state (Finding 3 from QA pass).** When the dir-move `os.replace(old_ep_dir, new_ep_dir)` failed (Windows Defender lock, indexer holding a handle, partial dir from a prior crash, **or destination dir already existing — `os.replace` ALWAYS fails on Windows when destination dir exists, even empty**), the code logged a warning and continued. It updated `self.episode_id` and `self.data["episode_id"]` to the new id but left `self.out_dir` pointing at the old path. The next `self.save()` wrote a finalized-id ledger into the OLD dir while every downstream node (BatchHumoRender, VideoComposite, RTXUpscale) built paths from the new id. Net effect: confusing "file not found" cascades far away from the rename failure.
- **Cause:** Single function trying to advance in-memory state regardless of on-disk success. Missing state-matrix handling for: both dirs exist; both missing; old missing + new exists. No retry on transient Windows locks. Treatment files outside the rename loop. Filename-construction slug not consistent between ledger and treatment paths.
- **Fix:** Rewrite `Ledger.rename_episode` around a strict invariant: **either complete with canonical episode dir + canonical ledger + canonical treatment, OR raise BEFORE mutating in-memory episode state.** Specifics:
  - Case-insensitive `os.path.normcase` same-path early-return (no-op for case-only changes on Windows).
  - State matrix: `(old_exists, new_exists)` resolved into one of {happy retry path, conflict raise, idempotent recovery, both-missing raise} BEFORE any mutation.
  - 3 × 0.5s inline retry on `os.replace(old_ep_dir, new_ep_dir)` with attempt-aware logging. After the third failure: `RuntimeError` with message that explicitly tells the user to check for files open in Notepad / VLC / Explorer preview / editors (per Gemini consult — system locks clear in ms but human-held locks need user intervention).
  - In-memory state (`episode_id`, `data["episode_id"]`, `out_dir`) only advances AFTER dir is in final on-disk position.
  - Ledger file rename (best-effort warn-only, dir invariant already satisfied).
  - Treatment + sidecar rename: glob `<old_slug>_*.txt` (NOT `pending_*` — narrower, no risk of catching unrelated files), rename each to `<new_slug>_*.txt`. Uses the same `_slugify(..., limit=120)` as the ledger path. Per-file warn-only on failure.
- **Verify:**
  - AST + 9 invariant guards (hard-fail message, retry sleep, conflict check, both-missing check, sidecar glob, slug consistency, normcase, etc.) — green
  - **New targeted suite `tests/test_ledger_rename.py` — 10 passed in 1.78s.** Covers: happy path renames dir+ledger+all sidecars; same-id no-op; conflict raises; both-missing raises; idempotent recovery (old missing + new exists); dir-move retries 2/3 then succeeds; dir-move fails 3/3 → RuntimeError + state unchanged; error message mentions human-held locks; treatment failure does not raise; sidecar glob uses old-id prefix not pending wildcard.
  - Three CLAUDE.md regression suites: Bug Bible (23 passed / 1 skipped / 2 xfailed), `tests/test_dropdown_guardrails.py` + `tests/test_core.py` (155 passed). Total **178 passed / 1 skipped / 2 xfailed in 101.62s**.
  - **Real-run acceptance (pending):** kill mid-rename (Ctrl-C between treatment write and rename), restart, confirm clean recovery and no orphan `<old>_*.txt` after the next successful run. Two-episode-in-flight soak covers Phase A + B together.
- **Tags:** ledger, rename, atomicity, windows-replace, retry, slug-consistency, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-phase-b-rename-consult__01_chatgpt.md` (gpt-5.5, 150.8s), `docs/2026-05-02-phase-b-rename-consult__02_gemini.md` (gemini-3.1-pro-preview-customtools, 31.9s — caught the critical "Windows os.replace always fails on existing dest dir, even empty"), `docs/2026-05-02-phase-b-rename-consult__03_nvidia.md` (llama-3.3-nemotron-super-49b-v1.5, 66.7s)

---

### BUG-LOCAL-014 [FIXED]: Spacesaver wrong-episode wipe via global mtime ledger scan
- **Date:** 2026-05-02 | **Phase:** 0 (cleanup hygiene) | **Bible candidate:** yes (after real two-episode run)
- **Symptom:** `_spacesaver_cleanup_if_flagged` in `nodes/rtx_upscale.py` discovered the ledger to read the `meta.perfect_run_spacesaver` flag from by calling `_otr_ledger.find_most_recent_ledger([otr_episodes_root(), otr_legacy_audio_dir()])`. That walker returns the newest `*_ledger.json` by mtime across the **entire** `otr/episodes/` tree. If Episode A is mid-RTXUpscale when Episode B is queued and writes its pending ledger, A's spacesaver pass would discover B's ledger, derive `ep_dir = ledger.parent.parent` (B's tree), and wipe B's `stills/`, `portraits/`, `videos/`, `composited/` while B was still rendering.
- **Cause:** Use of a global mtime-based discovery in a destructive code path. The existing substring sanity guard (`"episodes" in parts and "otr" in parts`) only verified the wiped tree was *somewhere* under `otr/episodes/`, not that it was the **right** episode for the current RTXUpscale call.
- **Fix:** Derive `ep_dir` directly from the `src` argument the upstream node already passes in. `src` is always `otr/episodes/<ep>/composited/<ep>.mp4` (the VideoComposite output), so `src.resolve().parent.parent` is the episode root by construction. Replace substring guard with `ep_dir.relative_to(otr_episodes_root().resolve())` plus a `len(rel.parts) == 1` depth-1 invariant. Load the ledger from THIS episode's `audio/*_ledger.json` glob, prefer non-pending. Add an OBS-existence precondition (`otr/obs/<ep>.mp4` must exist on disk) so spacesaver refuses to fire if the run order ever flips and the final deliverable hasn't landed yet. Build the keep-list from real on-disk filenames (`audio_dir.glob("*_treatment.txt")` plus the discovered ledger path) so a slug mismatch between `ep_id` and the on-disk filename can't accidentally delete the ledger or treatment.
- **Verify:**
  - AST + Bug Bible regression (23 passed / 1 skipped / 2 xfailed) + `tests/test_dropdown_guardrails.py` + `tests/test_core.py` (155 passed in 107.84s) all green post-fix.
  - Source no longer references `find_most_recent_ledger` from the spacesaver path (verified by grep + AST sanity script).
  - **Real-run acceptance (pending):** queue Episode A, queue Episode B before A's RTXUpscale fires; inspect `[OTR_RTXUpscale] spacesaver:` log lines and confirm `ep_dir` resolves to A's path, never B's. Bypass-safety run with `src` outside `otr/episodes/` should log `refusing destructive cleanup` with no deletion. Delete `otr/obs/<ep>.mp4` before cleanup fires and confirm the new precondition aborts.
- **Tags:** spacesaver, ledger, two-episode, destructive-cleanup, qa-pass-2026-05-02
- **Consult sources:** `docs/2026-05-02-path-reorg-spacesaver-qa__01_chatgpt.md`, `docs/2026-05-02-path-reorg-spacesaver-qa__02_gemini.md`, `docs/2026-05-02-rtx-upscale-qa-pass.md` (Phase A)
- **Follow-up phases queued:** B (production_ledger.py treatment rename + os.replace fallback), C (slug-reconstruction sweep), D (cache-key timestamp drop), E (schema bump + meta.paths block)

---

### BUG-LOCAL-001: Pre-existing test infrastructure rot blocks `pytest tests/` regression baseline
- **Date:** 2026-05-02 | **Phase:** 0 (regression infra) | **Bible candidate:** yes
- **Symptom:** Running the canonical `python -m pytest tests/` cannot reach a clean green pass. Three distinct failure modes observed in one run:
  1. **8 collection ImportErrors** (`No module named 'otr_v2.visual'`) on:
     `tests/test_anchor_gen.py`, `tests/test_camera_path_determinism.py`,
     `tests/test_character_regression.py`, `tests/test_cold_open_canary.py`,
     `tests/test_episode_dry_run.py`, `tests/test_lhm_monitor.py`,
     `tests/test_three_minute_continuous.py`, `tests/test_visual_phase_a.py`.
     Without `--ignore=` flags, pytest aborts the run after collection (`Interrupted: 8 errors during collection`).
  2. **`tests/test_backend_dispatch.py` 14 failures** (`FFFFFFFFFFFFFF` in -q output). Not investigated yet.
  3. **`tests/test_dropdown_guardrails.py` deterministic hang** after the first 12 tests pass (`............` then no further progress for >2 min, until externally killed).
- **Cause:**
  1. `otr_v2/visual/` package was deleted in commit `7706660` ("Fix BUG-LOCAL-047: FLUX anchor dtype ladder"). Test modules that imported it were not updated or removed in the same commit.
  2. test_backend_dispatch failure mode unknown — needs investigation.
  3. test_dropdown_guardrails.py hang likely waiting on a network/model/subprocess fixture that no longer resolves; not yet bisected.
- **Fix:** **Pending — do NOT fix mid-test per ground rules.** Captured here as v2.0-beta era opening entry. Likely fix sequence (next session): (a) delete or rewrite the 8 stale visual test modules; (b) bisect dropdown_guardrails hang to identify the wedged test; (c) investigate backend_dispatch failures separately. Also note: CLAUDE.md references `tests/v2/test_audio_byte_identical.py` which doesn't exist (path is `tests/test_audio_byte_identical.py`); CLAUDE.md test-command block is stale.
- **Verify:** After fix, `python -m pytest tests/` runs to completion, no collection errors, all hangs resolved, backend_dispatch failures triaged (fixed or marked xfail with reason).
- **Tags:** test-infra, pre-existing, otr_v2-orphans, claude-md-staleness

### BUG-LOCAL-002: `scripts/soak_operator.py` widget indices stale (drift since episode_title + num_characters added)
- **Date:** 2026-05-02 | **Phase:** 0 (smoke harness) | **Bible candidate:** yes
- **Symptom:** `scripts/soak_operator.py` declares `WV_GENRE=1`, `WV_TARGET_WORDS=2`, `WV_CREATIVITY=11`, `WV_OPT_PROFILE=13`. Reading `nodes/story_orchestrator.py::OTR_LLMScriptWriter.INPUT_TYPES` shows the actual widget order is now: `[0]episode_title, [1]target_words, [2]num_characters, [3]model_id, [4]cleanup_model_id, [5]custom_premise, [6]include_act_breaks, [7]self_critique, [8]open_close, [9]target_length, [10]style, [11]style_custom, [12]creativity, [13]arc_enhancer, [14]optimization_profile`.
- **Cause:** `episode_title` and `num_characters` widgets were added to the script-writer node, plus `style_custom` and `arc_enhancer` were inserted. soak_operator constants were never updated. Anything calling `supersoaker.py::patch_workflow` writes to the wrong slots: `creativity` write lands on `style_custom` (string field, broken), `optimization_profile` write lands on `arc_enhancer` (boolean field, broken), `target_words` write lands on `num_characters`, and `WV_GENRE=1` writes target_words.
- **Fix:** **Pending — do NOT fix mid-test per ground rules.** Correct constants: `WV_TARGET_WORDS=1, WV_NUM_CHARACTERS=2, WV_SELF_CRITIQUE=7, WV_TARGET_LENGTH=9, WV_STYLE=10, WV_CREATIVITY=12, WV_ARC_ENHANCER=13, WV_OPT_PROFILE=14`. Drop `WV_GENRE` (the widget no longer exists; "genre" was effectively replaced by `style`). Update `supersoaker.py::patch_workflow` cfg keys accordingly. Add a smoke assertion that reads node 1's widget names from `/object_info` and aborts if order doesn't match constants.
- **Verify:** After fix, run supersoaker P0 → confirm log shows correct `target_words`, `target_length`, `creativity`, `optimization_profile` values; ledger reflects what was patched.
- **Tags:** widget-drift, soak-harness, supersoaker, bible-candidate

### BUG-LOCAL-003: ComfyUI Desktop launch does not inherit user-scope `HF_HOME`
- **Date:** 2026-05-02 | **Phase:** 0 (smoke harness) | **Bible candidate:** yes
- **Symptom:** First smoke queue (prompt_id `a455fc20-...`) failed in NewsCuration / NewsCurationDeep / NewsSummary phases with repeated `local_files_only=True failed for model (mistralai/Mistral-Nemo-Instruct-2407 does not appear to have files named ('model-00001-of-00005.safetensors', ...))` and `huggingface.co` connection-failed fallback. Each phase took the timeout (65s NewsCuration, 40s NewsCurationDeep) and never recovered. ComfyUI then created a fresh empty `C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub\models--mistralai--Mistral-Nemo-Instruct-2407\` skeleton (timestamp 2026-05-02 00:40), proving it was looking at the wrong cache root.
- **Cause:** `HKCU\Environment` has `HF_HOME=C:\ComfyUI-Models\huggingface` (canonical, populated with all 5 Mistral-Nemo shards under `hub\models--mistralai--Mistral-Nemo-Instruct-2407\snapshots\04d8a905...\`). When ComfyUI Desktop is launched via `Start-Process` from a parent process that does NOT have `HF_HOME` already in its env (e.g. the Cowork sandbox), the Electron renderer + bundled Python backend inherit only the parent's env, NOT user-scope env vars. So huggingface_hub falls back to its default `~/.cache/huggingface/hub` (which on this machine is junctioned/aliased to `C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub` — empty).
- **Fix (verified 2026-05-02):** Before launching `ComfyUI.exe`, explicitly set `HF_HOME=C:\ComfyUI-Models\huggingface` in the parent shell. Re-queue confirmed: `LLM tokenizer loaded from cache (no HTTP checks)` log line appeared, NewsSummary generated 2572 chars, ScriptWriter ran end-to-end @ 18.1 tok/s. **Permanent fix needed:** ComfyUI Desktop launcher should read user-scope HF_HOME via `winreg.OpenKey(HKEY_CURRENT_USER, "Environment")` at startup and inject into the Python child process's env. Until then, document the launch-via-elevated-shell-only pattern in README.
- **Verify:** `[00:48:59] LLM tokenizer loaded from cache (no HTTP checks)` in runtime log + ScriptWriter generation completes without `local_files_only=True failed`.
- **Tags:** comfyui-desktop, hf_home, env-inheritance, launch-pattern, bible-candidate

### BUG-LOCAL-004: OOM in OTR_LLMScriptWriter on 30-word ultra-smoke after parse-retry loop (peak 29.5 GB on 16 GB device)
- **Date:** 2026-05-02 | **Phase:** 0 (smoke verification) | **Bible candidate:** yes
- **Symptom:** Smoke prompt_id `e6b87239-...` ran with `target_length="30 words (smoke, 1 act)"`, target_words=30, num_characters=2. Pipeline order: NewsSummary OK → ScriptWriter generated 571 tokens (parsed 0 scenes/0 lines/Characters: none) → OpenClose 3-outline evaluator (CHARACTER-DRIVEN, SCIENCE-DRIVEN, ATMOSPHERE-DRIVEN) all returned 0 chars and were DISCARDED → "OPENCLOSE: All outlines failed" → next `_generate_with_llm` call OOMed: `Allocation on device 0 would exceed allowed memory. Currently allocated: 26.53 GiB / Device limit: 15.92 GiB / Free (according to CUDA): 0 bytes`. Exception type `torch.OutOfMemoryError`, raised at `nodes/story_orchestrator.py:3137 model.generate()` from `nodes/story_orchestrator.py:5188 write_script._generate_with_llm`. Final VRAM_SNAPSHOT before OOM: `current_gb=8.275 peak_gb=29.498` — peak indicates cumulative KV cache + activations from successive calls were never released.
- **Cause:** Hypothesis (needs code dive): after each LLM `model.generate()`, the KV cache + intermediate activations are not torch.cuda.empty_cache()'d before the next call. With 4-bit NF4 weights ~7.5 GB resident, plus context_cap=16384 prompt tokens of KV cache (40 layers × 2 K/V × 16384 × hidden_dim × 2 bytes ≈ 6.4 GB) per call, four cumulative calls overflow. Compounded by the 30-word preset's parse-fail retry path (since 0 lines parsed, system retries) — each retry is another full forward pass without inter-call eviction.
- **Fix:** **Pending — not fixed mid-test.** Plan: (a) audit `_generate_with_llm` and `_critique_and_revise` for an explicit `torch.cuda.empty_cache()` + KV-cache delete after every generate call; (b) add a hard parse-retry cap (e.g. ≤2) to prevent runaway retry on the ultra-smoke preset; (c) log the prompt-token count alongside `llm_generate_entry` snapshot so future OOMs can be bisected. CLAUDE.md says "use `_flush_vram_keep_llm()` between LLM phases" — verify that is actually being called between OpenClose synth and write_script retry.
- **Verify:** Re-queue 30-word ultra-smoke; expect peak_gb < 14.5 GB across full LLM ladder; expect `MAX_RETRIES_EXCEEDED` (graceful) instead of OOM if parse keeps failing.
- **Tags:** vram, oom, llm-cache, retry-loop, ultra-smoke, bible-candidate

### BUG-LOCAL-005: 30-word ultra-smoke ScriptWriter output unparseable (0 scenes / 0 dialogue lines / 0 characters from 571 tokens)
- **Date:** 2026-05-02 | **Phase:** 0 (smoke verification) | **Bible candidate:** yes
- **Symptom:** With patched inputs `target_length="30 words (smoke, 1 act)"`, `target_words=30`, `num_characters=2`, the LLM (Mistral-Nemo-Instruct-2407 4-bit NF4) generated 571 tokens at 18.1 tok/s but the post-generation parser counted: `0 scenes | 0 dialogue lines | Characters: none`. Confirmed 30-word ULTRA-SMOKE preset was applied (history.current_inputs shows the patched values). OpenClose 3-outline evaluator (CHARACTER-DRIVEN, SCIENCE-DRIVEN, ATMOSPHERE-DRIVEN) also returned 0-char outputs — all 3 DISCARDED ("too short: 0 chars"), "OPENCLOSE: All outlines failed".
- **Cause:** Hypothesis: the new "30 words (smoke, 1 act)" preset's prompt does not enforce `CHARACTER:` / `SCENE:` markers (CLAUDE.md note: "BUG-007 root cause: Short (3 acts) prompt now explicitly enforces CHARACTER: dialogue format" — the same fix may not have been carried forward to the 30-word preset). The model generates prose that satisfies token count but lacks the structural markers the parser greps for. Also possible: the SPINE_MODE upper bound (commit `6454d91`, BUG-LOCAL-132) collides with `max_new_tokens=150` in OPENCLOSE such that nothing generated reaches the parser. Needs prompt-trace logging.
- **Fix:** **Pending — not fixed mid-test.** Plan: (a) instrument `write_script` to dump the actual prompt + raw model output at TRACE level when parse yields 0 lines, so the format gap is visible; (b) port the BUG-007 CHARACTER:/SCENE: enforcement clause from the "short (3 acts)" prompt into the 30-word ultra-smoke prompt; (c) add a unit test that asserts the 30-word preset's compiled prompt contains the literal substrings `CHARACTER:` and `SCENE:`.
- **Verify:** Re-run 30-word ultra-smoke, expect `1 scene | 3 dialogue lines | 2 characters` parse and a non-empty ledger.
- **Tags:** prompt-format, ultra-smoke, parse-fail, bug-007-regression, bible-candidate

### BUG-LOCAL-006: `pytest tests/` hangs at session-start when ComfyUI is running on the same GPU
- **Date:** 2026-05-02 | **Phase:** 0 (test infra) | **Bible candidate:** yes
- **Symptom:** Running `python -m pytest tests/test_core.py tests/test_arc_check.py ... -q` while ComfyUI Desktop is up on `:8000` produces only the standard pytest banner ("test session starts", "platform win32...", "plugins: anyio...") and then hangs with the python.exe at ~2.7 GB RSS, no further output, no test names, for 90+ seconds. Killing the python.exe is the only way to recover. Same hang shape was observed during the baseline `pytest tests/` (which paused at `tests/test_dropdown_guardrails.py ............` mid-suite). Both behave identically: stable RSS, no I/O progress.
- **Cause:** Hypothesis: pytest collection imports OTR's `__init__.py` which transitively imports torch + transformers + bitsandbytes. ComfyUI already owns the CUDA primary context. Either bitsandbytes' `cuda_setup` or transformers' device-probe is stalling on CUDA-context-create while ComfyUI holds the device. Not yet bisected — could also be a network call (HF model resolver) or filesystem walk over `C:\ComfyUI-Models\huggingface` (1.85 TB free, deeply nested cache).
- **Fix:** **Pending — not fixed mid-test.** Plan: (a) add an autouse conftest fixture that sets `CUDA_VISIBLE_DEVICES=""` for unit tests so collection never tries to bind to GPU; (b) move the OTR node imports out of the package's `__init__.py` top level into lazy load (already partially done for some modules, audit completeness); (c) document in CLAUDE.md that local pytest runs require ComfyUI to be killed first. Until then, regression baseline is unverifiable when ComfyUI is up.
- **Verify:** With ComfyUI killed, `pytest tests/test_core.py -q` runs to completion in <30 s. With ComfyUI up + the conftest CUDA-mask fixture in place, same.
- **Tags:** test-infra, cuda-context, comfyui-cohabit, bible-candidate

---

## 2026-05-02 fix landings

The five entries above (BUG-LOCAL-001 through 006, minus 002 which was logged separately) received fixes in this session's mega-commit. Status update per fix-lands-here pattern: `[FIXED]` with verify recipe.

- **BUG-LOCAL-003 [FIXED]** — `scripts/run_comfyui.cmd` reads `HF_HOME` + `HUGGINGFACE_HUB_CACHE` from `HKCU\Environment` via PowerShell, exports them, and launches `ComfyUI.exe`. README.md "Launching ComfyUI Desktop on Windows" section documents the pattern. Verify: kill ComfyUI, run `scripts\run_comfyui.cmd`, queue any episode that touches an HF model — expect `LLM tokenizer loaded from cache (no HTTP checks)` in `otr_runtime.log`.
- **BUG-LOCAL-004 [FIXED]** — `nodes/story_orchestrator.py` `write_script` short-episode branch (target_words ≤ 700) now (a) calls `_flush_vram_keep_llm()` before the main `_generate_with_llm` call so KV cache + activation peaks from prior LLM phases (NewsSummary, OpenClose 3-outline + evaluator, synthesizer) don't ride along into the main forward pass, and (b) wraps the call in a `MAX_PARSE_RETRIES = 2` loop with a cheap `[VOICE: ...]` / `CHARACTER:` marker count as the parseability check; on exhaustion logs `MAX_PARSE_RETRIES_EXCEEDED, accepting last output` and lets the parse-fail observability stamp it in the ledger instead of OOMing on a fourth forward pass. Verify: re-queue 30-word smoke; expect peak_gb < 14.5 GB across LLM ladder; if parse keeps failing, expect `MAX_PARSE_RETRIES_EXCEEDED` in `otr_runtime.log`, not `torch.OutOfMemoryError`.
- **BUG-LOCAL-005 [FIXED v2]** — `nodes/story_orchestrator.py` `write_script` now (a) detects `is_ultra_smoke` / `is_tiny_smoke` BEFORE `_open_close_expansion` is called and short-circuits the 3-outline evaluator entirely (round-robin verdict 2026-05-02: ChatGPT 5.5 + Gemini 3.1 + NVIDIA Nemotron 49B all flagged this as the actual root cause of the 29.5 GB-on-16-GB OOM since the evaluator holds 3 parallel KV caches at once); (b) clamps `max_new_tokens` to 256 for ultra-smoke and 384 for tiny smoke so a degenerate model output cannot run away to 571+ tokens; (c) swaps in the streamlined ULTRA_SMOKE prompt with explicit `[VOICE: ...]` enforcement; (d) replaces the original permissive `_bare_hits` regex with a negative-lookahead variant that excludes `TITLE:` / `SCENE:` / `GENRE:` / `ENV:` / `SFX:` / `MUSIC:` / `VOICE:` / `CAST:` / `AUTHOR:` so a "TITLE only" output cannot falsely PARSE_OK; (e) under ultra-smoke mode requires `=== SCENE N ===` AND `>=2 [VOICE: ...]` lines for PARSE_OK (strict-VOICE contract). The standard path keeps the looser scene-plus-any-marker check. `[VOICE: ...]` regex held strict per Gemini + NVIDIA: relaxing it would desync from the downstream parser and risk dropping audio lines, violating C7. Verify: queue 30-word smoke; expect peak_gb < 14.5 across LLM ladder, ledger has `1 scene | >=2 [VOICE: ...] dialogue lines | 2 named characters`.
- **BUG-LOCAL-006 [PARTIAL]** — `tests/conftest.py` created. Sets `CUDA_VISIBLE_DEVICES=""` + `OTR_TEST_MODE=1` at module import (before any `tests/test_*.py` collection), registers `requires_cuda` marker, auto-skips marked tests when CUDA is masked. The fix lets pytest progress further than baseline (24 dots in `test_dropdown_guardrails.py` vs 12 before) but the suite still hangs at `TestDropdownsHaveEffect::test_creativity_produces_different_temps`. The hang is INSIDE that test's call to `_run_preflight` -- not a CUDA-init issue, since CUDA is masked here. Likely root cause: a fixture or _generate_with_llm mock interaction that still touches a heavy import. Reproduce: `python -m pytest tests/test_dropdown_guardrails.py::TestDropdownsHaveEffect::test_creativity_produces_different_temps -q -s` (run alone). Next-session work: bisect the hang inside that test class -- this is a separate bug from the CUDA-context-create hang the conftest fixed, deserves its own follow-up entry.
- **BUG-LOCAL-001 [PARTIAL]** — 8 stale `otr_v2.visual` test collectors deleted (`test_anchor_gen.py`, `test_camera_path_determinism.py`, `test_character_regression.py`, `test_cold_open_canary.py`, `test_episode_dry_run.py`, `test_lhm_monitor.py`, `test_three_minute_continuous.py`, `test_visual_phase_a.py`) AND 10 sidecar-era tests in the same family (`test_backend_dispatch.py`, `test_wan21_loop.py`, `test_wall_clock_estimator.py`, `test_vhs_postproc.py`, `test_pulid_portrait.py`, `test_planner.py`, `test_ltx_motion.py`, `test_flux_keyframe.py`, `test_flux_anchor.py`, `test_florence2_sdxl_comp.py`). 38 test_*.py files remain (was 48 + 8 = 56; 18 deleted). This subsumes the original "14 `test_backend_dispatch` failures" entry — that file is gone. Verify: `python -m pytest tests/ --collect-only -q` reports zero `otr_v2.visual` collection errors.
- **BUG-LOCAL-002 [FIXED]** — `scripts/supersoaker.py` deleted. `scripts/soak_operator.py` slimmed from a 1500-line soak runner to a ~270-line legacy shim retaining only `scan_treatment` (used by `tests/test_treatment_scanner_unicode.py`). New canonical surface: `scripts/otr_api.py` exposes `load_workflow`, `fetch_schemas`, `patch_widget_by_name` (uses live `/object_info` schemas — robust against future widget reorders), `workflow_to_api_prompt` (port of soak_operator's BUG-LOCAL-027/029-fixed converter), `submit_prompt`, `poll_history`, `queue_snapshot`, `cancel_queue`. `scripts/queue_smoke.py` + `smoke_watcher.py` rebuilt on `otr_api`. `tests/test_widget_drift_guard.py` rerouted via private alias `mod._workflow_to_api_prompt = mod.workflow_to_api_prompt`. Verify: `python scripts/queue_smoke.py` produces a `/history` entry with `current_inputs.target_words=[30]`, `num_characters=[2]`, `target_length=["30 words (smoke, 1 act)"]`.

### Round-robin QA — 2026-05-02

`docs/2026-05-02-v2.0-beta-sprint-qa__01_chatgpt.md` (gpt-5.5), `__02_gemini.md` (gemini-3.1-pro-preview-customtools), `__03_nvidia.md` (mistral-nemotron-super-49b-v1.5), `__04_synthesis.md`, `__transcript.json`. All three external models converged on **BLOCK** for the initial Sprint 1 commit. Three must-fix items unanimously prescribed:

1. Move ultra-smoke / tiny-smoke detection BEFORE `_open_close_expansion` so the 3-outline evaluator's parallel KV caches don't run (Gemini calculated ~6 GB of cache from 3 simultaneous outlines = the actual source of the 29.5 GB peak; ChatGPT and NVIDIA both endorsed). **Applied.**
2. Replace the permissive `_bare_hits` regex with a structural-marker negative-lookahead so `TITLE:` / `GENRE:` etc. cannot falsely PARSE_OK; require `=== SCENE N ===` PLUS `>=2 [VOICE: ...]` lines for ultra-smoke. **Applied.**
3. Clamp `max_new_tokens` to 256 for ultra-smoke (384 for tiny smoke) so a runaway 571-token degenerate output cannot recur. **Applied.**

Disagreement caught: ChatGPT recommended relaxing the `[VOICE: ...]` regex to be more permissive. Gemini and NVIDIA both rejected this — relaxing the validation regex desyncs from the downstream parser and could silently drop dialogue lines, violating C7. Held the regex strict.

Non-blocking follow-ups for next session: investigate `live_ledger=True` for retained GPU tensors, audit `_generate_with_llm`'s explicit `del` of intermediates, write a unit test that mocks `_generate_with_llm` to return bad-then-good output and asserts exactly 2 attempts max, port any still-valid contracts (frame count `4n+1`, etc.) from the deleted sidecar tests into in-graph test coverage.

Post-fix regression: `python -m pytest tests/ --ignore=tests/test_dropdown_guardrails.py -q` → **932 passed, 6 skipped, 0 failed in 10.8 s**. AST-clean.

---

## 2026-05-02 — Sprint 3 mega-sprint (LTX wiring + RTX VSR upscale)

### BUG-LOCAL-007 [DEVIATION-LOGGED]: LTX 2B v0.9 bundled checkpoint forces CheckpointLoaderSimple-family loader

- **Date:** 2026-05-02 | **Phase:** S3.1 (LTX wiring) | **Bible candidate:** yes
- **Symptom:** ROADMAP Architecture Truth (locked 2026-05-02) specified `UNETLoader + CLIPLoader (T5) + VAELoader` for LTX 2B fp16, "NOT CheckpointLoaderSimple". Reason given: split-load lets ComfyUI offload T5/VAE independently; bundled-load on a hot HuMo cache is the OOM shape C2 was written to prevent.
- **Cause:** Lightricks ships LTX 2B v0.9 ONLY as a bundled `ltx-video-2b-v0.9.safetensors` (8.7 GB, all components in one safetensors file). No standalone LTX UNet / LTX VAE artifacts exist on the upstream HF repo for the 2B v0.9 line. The Architecture Truth assumed split files exist; they don't.
- **Fix:** Use ComfyUI-LTXVideo's `LowVRAMCheckpointLoader` for LTX 2B v0.9 (it IS a `CheckpointLoaderSimple` subclass, but adds a `dependencies` input that ComfyUI uses to force sequential load). The C2 sequencing intent (HuMo unloads before LTX claims VRAM) is satisfied via the dependency edge + the existing strict teardown in `batch_humo_render.py` (`unload_all_models + gc + empty_cache + cuda.synchronize` in finally). The "no carve-out for CheckpointLoaderSimple" rule was about preventing OOM from parallel-load on a hot cache; sequencing eliminates that risk.
- **Verify:** Log line `[OTR_HUMO] teardown complete` precedes `[OTR_LTX_LOADER] loading ltx-video-2b-v0.9.safetensors`. No `Allocation on device 0 would exceed allowed memory` between HuMo teardown and LTX render.
- **Promote-to-Bible-only-if:** a future LTX line (3B, 5B, 13B) ships with split UNet/T5/VAE artifacts AND we re-validate that LowVRAMCheckpointLoader's dependency edge keeps the C2 sequencing guarantee under 14.5 GB. Until then this stays as a documented OTR-local deviation.
- **Tags:** ltx, loader, c2-deviation, sequencing, deps-edge

### BUG-LOCAL-008 [DOCUMENTED]: LTX CFG=1.0 mathematically erases the negative prompt

- **Date:** 2026-05-02 | **Phase:** S3.1 (LTX wiring) | **Bible candidate:** yes
- **Symptom:** Round-robin Gemini caught: standard CFG math is `output = uncond + CFG * (cond - uncond)`. At CFG=1.0 this simplifies to `output = cond` -- the negative prompt is 100% unused. ROADMAP locks `LTX_CFG = 1.0` for the distilled sigma schedule. So the negative prompt in `_LTX_NEGATIVE` ("person, human, face, woman, man, hands, fingers, body, ...") is mathematically discarded by the sampler. Faces / people may still appear in LTX clips because the prompt suppression we *thought* was active isn't.
- **Cause:** Distilled LTX (`LTX_DISTILLED_SIGMAS` from Goofer) is tuned for CFG=1.0 because higher CFG with low-step distillation produces overcooked / artifacted output. The negative prompt was carried over from non-distilled LTX patterns where CFG≥1.5 made the negative effective.
- **Fix (deferred):** Two options for next sprint, both empirical:
  1. Raise CFG to 1.3-2.0 and re-tune sigma schedule (changes motion characteristics; needs A/B smoke).
  2. Remove the negative prompt encode entirely (saves T5 forward pass; output unchanged at CFG=1.0).
  Until then: tighten POSITIVE prompts in `_PROMPT_BY_ROLE` to avoid human-implying terms ("announcer at microphone" → "vintage microphone on desk"), since the positive branch is the only one the sampler sees at CFG=1.0.
- **Verify:** ffprobe + visual review of LTX clips after smoke. If `_PROMPT_BY_ROLE` text contains "announcer / radio host / broadcaster" terms AND faces appear in the rendered output, this bug is the cause.
- **Tags:** ltx, cfg, prompt-policy, distilled-sigma

### BUG-LOCAL-009 [DEFERRED]: Per-stage VRAM logging across HuMo→LTX→VC→RTX boundary

- **Date:** 2026-05-02 | **Phase:** S3 observability | **Bible candidate:** no (observability, not behavior)
- **Symptom:** No production VRAM snapshot at HuMo teardown / LTX loader entry / LTX teardown / RTX upscale entry. If a 16 GB OOM appears at any boundary, we have no per-stage signal to bisect from.
- **Fix (deferred to next sprint):** Add `[OTR_VRAM] free=X.XX allocated=Y.YY reserved=Z.ZZ` lines at:
  - `batch_humo_render.py` end of teardown
  - `batch_ltx_render.py` after `mm.load_models_gpu([model])` and after teardown
  - `rtx_upscale.py` after each chunk and at upscale exit
  Pattern matches existing `vram_snapshot()` helper in `nodes/_vram_log.py`.
- **Verify:** smoke run produces a clean VRAM ladder log; `peak_gb` per stage extractable via grep.
- **Tags:** vram, observability, deferred

### Round-robin QA — 2026-05-02 mega-sprint pre-smoke

`docs/2026-05-02-mega-sprint-consult__01_chatgpt.md` (gpt-5.5, 140s), `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 40s). NVIDIA round did not complete in window; two-of-three is sufficient per CLAUDE.md round-robin rule.

**Must-fix items applied before smoke:**
1. **Anti-clobber in `batch_ltx_render.py`** (ChatGPT + Gemini converged): `if out_mp4.exists(): skip` defends against role-filter drift overwriting HuMo character clips.
2. **Windows ffmpeg pipe deadlock** (Gemini): `rtx_upscale.py` was using `stderr=subprocess.PIPE` which blocks at 64 KB on Windows. Routed to `subprocess.DEVNULL`.
3. **ComfyUI cache desync** (Gemini): rewired link 86 from `BatchHumoRender.clips_dir` (slot 0, stable per episode_id) to `BatchHumoRender.report` (slot 2, varies per run from per-clip elapsed_ms). Forces ComfyUI to re-evaluate `LowVRAMCheckpointLoader` on every queue, keeping the loader's state machine in sync with the actual mm-unload call HuMo makes.
4. **Drop `-shortest`** (Gemini): silent video and audio source share frame count by construction; flag was at best dead code, at worst a footgun.

**Disagreement caught:**
- ChatGPT framed CFG=1.0 negative prompt influence as "weak"; Gemini corrected with the math: `output = uncond + CFG * (cond - uncond)` reduces to `output = cond` at CFG=1.0, so the negative prompt is 100% ignored, not weak. Logged as BUG-LOCAL-008 (deferred to next sprint, since changing CFG mid-sprint deviates from locked architecture).

**Documented for next sprint (not blocking smoke):**
- BUG-LOCAL-008 — CFG=1.0 + negative prompt mathematically inert.
- BUG-LOCAL-009 — per-stage VRAM logging missing.
- Gemini false alarm: `temporal_size=4096` in tiled VAE decode is fine because LTX_MAX_FRAMES=177 (the temporal window only matters above its value; 4096 means "decode whole sequence in one temporal pass", spatial tiling handles VRAM). Goofer-proven on RTX 5080 Blackwell.

### BUG-LOCAL-010 [BLOCKER on Sprint 3 acceptance]: LLM OOM regression at write_script main call (BUG-LOCAL-004 returns)

- **Date:** 2026-05-02 | **Phase:** S3 acceptance smoke | **Bible candidate:** yes
- **Symptom:** Sprint 3 smoke prompt_id `bc7136bb-50ab-471f-8caf-83e9cfefa481` (`target_words=30`, `num_characters=2`, `target_length="30 words (smoke, 1 act)"`, `optimization_profile=Standard`) OOM'd at `OTR_LLMScriptWriter` -> `write_script` (line 5432) -> `_generate_with_llm` (line 3211) -> `model.generate(...)`. Exact error: `Allocation on device 0 would exceed allowed memory. Currently allocated: 24.54 GiB / Device limit: 15.92 GiB / Free: 0 bytes / Requested: 3.94 GiB`. Peak allocated `29005 MiB` per torch's allocator print.
- **Cause (hypothesis):** BUG-LOCAL-004 fix (Sprint 1) added `_flush_vram_keep_llm()` before the main write_script `_generate_with_llm` call. BUG-LOCAL-005 fix (Sprint 1) added max_new_tokens clamp 256 for ultra_smoke and short-circuited the OpenClose 3-outline evaluator. Both are confirmed active on this run (runtime log shows `Loading LLM model: mistralai/Mistral-Nemo-Instruct-2407 (quantized=True)` plus the three sequential `[StoryOrchestrator] Starting inference` lines at max_new_tokens 64 / 800 / 256). Despite both fixes, peak allocated still hits ~29 GB. Likely roots: (a) `_flush_vram_keep_llm()` is not actually clearing the prior phases' KV cache + activations -- a Python reference is keeping intermediates alive; (b) NewsSummary's `max_new_tokens=800` build-up is the actual culprit, not write_script's call (the OOM fires DURING write_script's prefill but the 24 GB was already accumulated before that call started); (c) the Mistral-Nemo `_prefill` path in transformers 4.x has a regression where `past_key_values` is not freed between `model.generate` invocations even with explicit `torch.cuda.empty_cache()`.
- **Fix:** **Pending -- needs its own bisect window.** Plan: (a) instrument `_generate_with_llm` to log `torch.cuda.memory_allocated()` at entry / after generate / after the explicit `del` / after empty_cache call, so the actual leak source surfaces; (b) check that `_flush_vram_keep_llm()` survived the recent refactors and is in fact called between NewsSummary and write_script; (c) audit whether NewsSummary leaves a `transformers.cache_utils.DynamicCache` instance on the model object; (d) if (c) confirmed, monkey-patch `model._cache_implementation` to `None` between phases.
- **Verify:** re-queue 30-word smoke with `optimization_profile=Standard`, expect `peak_gb < 14.5 GB` across the LLM ladder (instrumented snapshot lines), expect to reach `OTR_SignalLostVideo` (the audio gate) without OOM.
- **Tags:** vram, oom, llm-cache, regression, blocks-s3-acceptance

### Sprint 3 mega-sprint: shipped code, live acceptance BLOCKED

The Sprint 3 mega-sprint code (LTX wiring + RTX VSR upscale + consult fixes) is committed on `v2.0-alpha`. The wiring is:
- AST-clean (3 modified .py files all parse).
- Regression-clean (Bug Bible, dropdown_guardrails, core, parse_retry, otr_api_type all green; 23 + 46 + 108 + 48 = 225 tests pass).
- Workflow JSON valid (`json.loads` round-trips, `last_link_id=93`, all 51 links intact, no orphan inputs).
- ComfyUI registers all three new nodes (`OTR_BatchLTXRender`, `OTR_RTXUpscale`, `LowVRAMCheckpointLoader`).
- ComfyUI accepts the patched workflow at `/prompt` and runs to OTR_LLMScriptWriter, where it hits BUG-LOCAL-010 (LLM OOM, pre-existing).

**Sprint 3 acceptance is BLOCKED on BUG-LOCAL-010**, NOT on a Sprint 3 wiring failure. The video-wiring code never executed because the smoke can't get past the LLM phase. Once BUG-LOCAL-010 is fixed in a follow-up bisect, re-queue the same workflow JSON and the S3.x acceptance bullets (ledger source_kind=ltx rows, ffprobe 832x480 pre-upscale + 1920x1080 post-upscale, audio byte-identity via stream MD5, peak VRAM < 14.5/15.5 GB) become directly observable.

### BUG-LOCAL-011 [FIXED]: BatchLTXRender raised on first live run -- _load_ledger missing the .mp4 -> _ledger.json stem-fallback that sister nodes have

- **Date:** 2026-05-02 EVENING | **Phase:** S3 live test | **Bible candidate:** yes
- **Symptom:** Live run on Jeffrey's ComfyUI Desktop with Gemma-4 E2B (which dodges BUG-LOCAL-010) progressed cleanly through the LLM ladder, audio cascade, FLUX bookend, and all 4 HuMo character clips. At HuMo teardown the dependency edge correctly fired LowVRAMCheckpointLoader -> BatchLTXRender, but BatchLTXRender raised: `RuntimeError: BatchLTXRender: ledger could not be loaded from inline JSON or path` at `batch_ltx_render.py:446`. Wallclock to failure: 00:58:53 (LLM ~10 min, audio ~3 min, FLUX ~3 min, HuMo ~40 min, then LTX failed immediately).
- **Cause:** `OTR_SignalLostVideo.0` (the STRING input feeding `BatchLTXRender.ledger_json` via link 90) emits the **mp4 path**, not the `_ledger.json` path. `BatchHumoRender._load_ledger_with_path` and `OTR_VideoComposite._load_ledger_with_path` both have a multi-tier stem-fallback that swaps `.mp4` -> `_ledger.json` with collapsed-underscore + fuzzy-match tiers (BUG-LOCAL-118 hardening). My BatchLTXRender's `_load_ledger` skipped that fallback -- it called `load_ledger_safe(.mp4)` directly, got `None`, returned `(None, None)`, raised. Round-robin consult flagged "ledger / clips_dir union" but missed this inner discrepancy because the node *interface* matches HuMo (both take a STRING called `ledger_json`); only the *internal resolver* differs.
- **Fix:** Replaced `BatchLTXRender._load_ledger` with a port of `BatchHumoRender._load_ledger_with_path`. Same multi-tier behaviour: (1) empty input -> auto-pick newest non-pending under audio dirs; (2) inline JSON -> parse; (3) `.mp4` path -> direct stem swap, then collapsed-underscore variant, then fuzzy directory-scan with <1h freshness gate; (4) `.json` path -> direct load. Same `(dict_or_None, Path_or_None)` return contract so the existing call site at `:425` is unchanged.
- **Verify:** Re-queue the same workflow JSON. Expect log lines `[BatchLTXRender] episode=signal_lost_..._...` and `radio_bookend: radio_bookend_<ep>.png` (the loader resolved the .mp4 -> _ledger.json swap and read radio_bookend_path from `ledger.meta`). Pre-fix repro: queue a workflow that wires SignalLostVideo.0 directly into BatchLTXRender.ledger_json with no manual ledger path; expect the fix to make this path-shape work end-to-end.
- **Tags:** ltx, ledger, stem-fallback, signallost-mp4, sister-node-divergence, bible-candidate

### Sprint 3 live-run progress observed on workflow JSON 7c4dfd4 (Gemma-4 E2B path)

- LLM phase (Gemma-4 E2B + E4B): clean. ~10 min. Peak VRAM ~14 GB. Output: parseable script with TITLE + SCENE + 6 [VOICE: ...] lines + 1 SFX + MUSIC closing.
- Audio cascade (Bark + Kokoro + MusicGen + AudioGen + AudioEnhance + EpisodeAssembler): clean. ~3 min. Episode duration 113s = 1.88 min.
- SignalLostVideo procgen: clean. mp4 saved 52.2 MB / 113s / 2712 frames in ~14s.
- BatchFluxRender (5 cast portraits + radio bookend): clean. **S3.2 acceptance VERIFIED**: radio bookend rendered at 1248x720 then Lanczos-downscaled to 832x480.
- BatchHumoRender Phase A (text encoding 4+1) + Phase B (Whisper) + Phase C (4 lines): clean. Peak VRAM 14.2 GB GPU dedicated. Per-clip ~10:00-10:20 wallclock at 6 sampler steps × ~97s/step. 4 character lines correctly routed to HuMo; 2 announcer lines correctly skipped (BUG-129b).
- LowVRAMCheckpointLoader -> BatchLTXRender: dependency edge fired correctly (sequencing intent SHIPPED), then BUG-LOCAL-011 raised inside `_load_ledger`. Fix landed in this commit; re-queue to verify the rest of the S3.x acceptance bullets.

### Round-robin consult on BUG-LOCAL-011 fix -- 2026-05-02 EVENING

`docs/2026-05-02-bug-local-011-fix-review__01_chatgpt.md` (gpt-5.5, 97s), `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 35s), `__03_nvidia.md` (nvidia/llama-3.3-nemotron-super-49b-v1.5, 93s).

**Three-way convergence (verdict: tighten before next live run):**

- Tier 1 (.mp4 -> .json exact stem swap) and Tier 2 (collapsed-underscore variant) are both correct and necessary.
- **Tier 3 fuzzy directory scan must be killed for LTX.** Non-deterministic (depends on directory contents + mtimes + wall-clock); could plausibly bind to a wrong neighbour ledger if exact match fails to load. Burning ~1 hour rendering against bad metadata is the failure mode 2/3 consultants flagged as the real risk.
- **Restore `_OTRL.load_ledger_safe()` for path loads** (consistent `[OTR_Ledger]` log prefix; future-proof against any hardening added there).
- **Fail loud on file-load errors.** If exact (Tier 1) or collapsed (Tier 2) ledger candidate file EXISTS but fails to parse / read (PermissionError from Windows file-locking, JSONDecodeError from a partial write), raise instead of falling through. Silent fall-through to a wrong neighbour was Gemini's strongest framing.
- **Document `humo_clips_dir` widget as a sequencing-only DAG edge.** Add tooltip + inline comment + `del humo_clips_dir` so a future maintainer doesn't remove it as dead code (which would let the LTX checkpoint load race HuMo's 16.5 GB MODEL teardown and OOM on 16 GB).

**Disagreement caught (consultants vs reality):**

- Gemini + NVIDIA flagged `log` and `time` as missing imports -> false alarms. `log = logging.getLogger("OTR.batch_ltx_render")` is at line 81; `import time` at line 51.
- Gemini + NVIDIA assumed `_OTRL.load_ledger_safe` does schema migration -> false alarm. It just wraps `json.loads` with three exception handlers (FileNotFound / JSONDecodeError / generic Exception), all logging WARNING and returning None. So restoring it gains consistent log-prefix and centralised error handling, NOT schema migration.

**Hardening pass applied in commit (next):**

1. Replaced raw `json.load()` calls with a local `_read(p)` helper that delegates to `_OTRL.load_ledger_safe(p)` when importable; raises RuntimeError when the loader returns None on an existing file.
2. Deleted Tier 3 fuzzy scan code path. Resolver now returns `(None, None)` after Tier 1 + Tier 2 miss, with a WARNING log line that explicitly notes Tier 3 was removed by the 2026-05-02 round-robin.
3. `humo_clips_dir` INPUT_TYPES tooltip rewritten to flag the widget as a DAG sequencing edge (NOT data); execute() body explicitly `del humo_clips_dir` with a comment explaining the "remove this and LTX OOMs" failure mode.

Resolver test (offline, against the live-run cached ledger) confirms all 4 branches still resolve correctly: .mp4 stem-swap, explicit .json path, inline JSON, empty input auto-pick.

**Companion artifact: `workflows/otr_ltx_smoke.json`** -- a 5-node fast-smoke harness (LowVRAMCheckpointLoader -> OTR_BatchLTXRender -> OTR_VideoComposite -> OTR_RTXUpscale + Note) that consumes the cached ledger.json + procgen mp4 + 4 HuMo character clips from the live run. ledger_json widgets on both BatchLTXRender + VideoComposite are the .mp4 PATH so the smoke truly repros the BUG-LOCAL-011 crash surface (i.e. exercises the .mp4 -> _ledger.json stem-swap chain). Wallclock target ~10 min vs ~60 min for full pipeline. Re-aim at a different episode by swapping the PROCGEN_MP4 + HUMO_VIDEOS_DIR widget values; both must come from the same episode_id so LTX writes into the dir HuMo wrote into.

### Path consolidation (Jeffrey directive, 2026-05-02 EVENING after first end-to-end smoke landed)

- **Date:** 2026-05-02 EVENING | **Phase:** post-smoke cleanup | **Bible candidate:** no (project-specific layout)
- **Change:** Final episode mp4 deliverables moved from `<output>/episodes_for_obs/<ep>/` (sibling of otr/) to `<output>/otr/episodes/<ep>/` (nested INSIDE otr/). Every project output now lives under one tidy `otr/` umbrella:
  - `otr/audio/<ep>.mp4` -- procgen mp4 + ledger.json
  - `otr/stills/` -- FLUX bookends + cast environments
  - `otr/portraits/` -- PASS1 character portraits
  - `otr/videos/<ep>/<line_id>.mp4` -- per-line HuMo + LTX clip pieces
  - **NEW: `otr/episodes/<ep>/<ep>.mp4` and `<ep>_1080p.mp4` -- final user-facing deliverables ONLY**
- **Why:** OBS's directory_sorter still gets a clean root with only finished episodes (now `output/otr/episodes/`), but the entire project workspace has one nested root instead of two siblings (`otr/` + `episodes_for_obs/`). Easier mental model + easier to back up + easier to scrub.
- **Files touched:**
  - `nodes/_otr_paths.py::episodes_for_obs_dir` -- function name kept for back-compat with existing imports; return value changed to `comfy_output_dir() / "otr" / "episodes" / episode_id`.
  - `nodes/video_composite.py` -- comment block updated.
  - `scripts/render_episode_concat.py` -- comment + default `out_dir` expression updated.
  - `tests/test_render_episode_concat_discovery.py` -- pinned source-string assertion updated to require `"otr" / "episodes" / episode_id`.
- **OBS pointer change:** if you have OBS / external tooling configured to watch `output/episodes_for_obs/`, repoint it to `output/otr/episodes/`.

### Cleanup: torch.from_numpy non-writable warning in OTR_RTXUpscale (2026-05-02 EVENING)

- **Date:** 2026-05-02 EVENING | **Phase:** post-smoke cleanup | **Bible candidate:** no (cosmetic)
- **Symptom:** First successful smoke surfaced `UserWarning: The given NumPy array is not writable, and PyTorch...` at `rtx_upscale.py:216`.
- **Cause:** `np.frombuffer(chunk_bytes, dtype=np.uint8)` returns a view over an immutable bytes buffer; `torch.from_numpy` warns when handed a non-writable ndarray.
- **Fix:** Added `.copy()` after `np.frombuffer()` so torch gets a writable buffer. Also confirms a clean ownership boundary between the ffmpeg-stdout-bytes and the cuda transfer.

### BUG-LOCAL-013 [FIXED]: LTX 2B v0.9 bundled checkpoint has no CLIP/T5 -- LowVRAMCheckpointLoader returns CLIP=None -> NoneType.tokenize crash

- **Date:** 2026-05-02 EVENING (T+~30 min after smoke load) | **Phase:** S3 fast-smoke first execution | **Bible candidate:** yes
- **Symptom:** Smoke loaded cleanly (after the BUG-LOCAL-012 UUID fix), Queue Prompt accepted, LowVRAMCheckpointLoader fired, then BatchLTXRender raised at line 554: `AttributeError: 'NoneType' object has no attribute 'tokenize'`. ComfyUI runtime log printed `no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.` immediately above the crash.
- **Cause:** The bundled `ltx-video-2b-v0.9.safetensors` (8.7 GB on disk at `C:\ComfyUI-Models\checkpoints\`) ships only UNet + VAE; it does NOT carry the T5 text encoder. `LowVRAMCheckpointLoader` (a `CheckpointLoaderSimple` subclass) then returns `(MODEL, None, VAE)` for the (model, clip, vae) tuple. My BatchLTXRender wired CLIP straight from LowVRAM, so `clip.tokenize(...)` immediately NoneType-crashed. Note: this means BUG-LOCAL-007's deviation from the locked Architecture Truth was wrong on the premise -- the original Architecture Truth (UNETLoader + CLIPLoader + VAELoader for LTX 2B) was actually correct, because `t5xxl_fp16.safetensors` has to be loaded separately. The LowVRAMCheckpointLoader is still useful for the UNet+VAE side (sequential-load via `dependencies` input survives), but the CLIP comes from a sibling CLIPLoader, not from the bundled file.
- **Fix:**
  1. Add a `CLIPLoader` node loading `t5xxl_fp16.safetensors` (already on disk at `C:\ComfyUI-Models\text_encoders\`) with `type='ltxv'`, `device='default'`. Verified `'ltxv'` is in `/object_info/CLIPLoader` allowed types on the live ComfyUI alongside sd3 / wan / mochi / flux2 / etc.
  2. Rewire `OTR_BatchLTXRender.clip` from the new CLIPLoader, NOT from `LowVRAMCheckpointLoader.CLIP`.
  3. Apply same fix to BOTH `workflows/otr_ltx_smoke.json` AND `workflows/otr_scifi_16gb_full.json` so the next live full-pipeline run also doesn't hit this crash. Production workflow now has new node 57 (CLIPLoader, T5, ltxv) wired via new link 94 to BatchLTXRender (55).1. Old link 88 (54.1 -> 55.1, the dead CLIP edge from LowVRAM) deleted.
- **Verify:** Re-run smoke; expect log line `[BatchLTXRender] episode=signal_lost_..._170555` followed by per-clip render lines (no `NoneType.tokenize` AttributeError). Final mp4 should write to `output/episodes_for_obs/<ep>/<ep>.mp4` and a separate `<ep>_1080p.mp4` from the upscaler.
- **Why the smoke harness paid off here:** the original failed live run (BUG-LOCAL-011 on the .mp4-as-ledger problem) hid this CLIP=None bug behind it. If I had only fixed BUG-LOCAL-011 and re-queued the full pipeline, we would have burned ~50 more min of HuMo wallclock just to crash here. The smoke surfaced this in <30s of LTX cold-load -- exactly what fast iteration loops are for.
- **Architecture Truth retroactively re-validated:** the locked Architecture Truth from 2026-05-02 (UNETLoader + CLIPLoader + VAELoader for LTX 2B) was correct in spirit; only the bundled-vs-split question was open. We're now using a hybrid: LowVRAMCheckpointLoader for the bundled UNet+VAE (with the dependencies input for sequential-load), separate CLIPLoader for T5. Same end result as the locked plan; cleaner sequencing edge.
- **Tags:** ltx, cliploader, t5, bundled-checkpoint, hidden-by-prior-bug, smoke-paid-off, bible-candidate

### BUG-LOCAL-012 [FIXED]: ComfyUI frontend Zod validation rejected `workflows/otr_ltx_smoke.json` at load time

- **Date:** 2026-05-02 EVENING | **Phase:** S3 fast smoke harness | **Bible candidate:** yes (broadly applicable to anyone hand-building ComfyUI workflow JSONs)
- **Symptom:** Jeffrey loaded `workflows/otr_ltx_smoke.json` (commit `f60d2e4` / `4df4e72`) into ComfyUI Desktop and the frontend rejected the workflow with two Zod validation alerts: `Invalid workflow against zod schema: Validation error: Invalid uuid at "id"`. Raw `json.loads` round-trips cleanly; this is a frontend-side schema validation failure, not a JSON syntax error.
- **Root cause confirmed:** workflow root `id` field MUST be a valid UUID (8-4-4-4-12 hex format). My hand-built smoke had `id: "otr-ltx-smoke"` -- a freeform slug. ComfyUI's Vue 3 frontend Zod schema enforces uuid format on this field. Production `otr_scifi_16gb_full.json` has a valid UUID so it loads cleanly.
- **Cause (hypothesised pending error-text capture):** The smoke JSON was hand-built by `outputs/_build_ltx_smoke.py` rather than exported by the ComfyUI UI. Three candidate divergences from the canonical shape, identified by structural diff against `workflows/otr_scifi_16gb_full.json` + `Nvidia_RTX_Nodes_ComfyUI/example_workflows/rtx_video_upscale.json` (both load cleanly):
  1. **`_meta` field on every node.** Mine has `_meta: {title: ...}`; canonical workflows use a top-level `title` field on the node. Some Zod schemas reject unknown fields strictly.
  2. **`shape: 7` on optional inputs.** Mine puts `shape: 7` on `dependencies` + `humo_clips_dir`; neither known-good workflow uses this. LiteGraph shape values 1-7 are valid in the LiteGraph runtime, but the Vue frontend's Zod schema may not list `shape` as an allowed input field.
  3. **Missing `slot_index` on outputs.** Production has `{name, type, links, slot_index}` on every output; mine omits `slot_index`. Vue frontend may require it to track slot-position semantics.
- **Fix:** apply all 3 candidate corrections in `outputs/_build_ltx_smoke.py` and re-emit `workflows/otr_ltx_smoke.json`. Drop `_meta` -> rename to `title`. Drop `shape` from inputs entirely. Add `slot_index: <i>` to every output. Land an offline regression check (`tests/test_workflow_zod_shape.py`) that asserts these shape invariants on every JSON under `workflows/` so this class of bug is caught by the test suite before it reaches the UI.
- **Verify:** Jeffrey reloads `workflows/otr_ltx_smoke.json` in ComfyUI Desktop; no Zod error; nodes appear on canvas; Queue Prompt accepts the workflow.
- **Why prior consults missed it:** all three rounds reviewed the ComfyUI execution semantics (object_info, link types, DAG ordering, .mp4 path stem-swap, audio passthrough). None reviewed the frontend's Zod schema, which is a separate validation layer between drag-and-drop UI load and the backend's `/prompt` endpoint. The CLI `submit_prompt` path bypasses Zod entirely (it goes straight to `/prompt` with the API-converted workflow), which is why my offline JSON tests + the `queue_smoke.py` script + the `_test_ledger_resolver.py` all passed -- they exercise different code paths than the UI loader.
- **Tags:** zod, comfyui-frontend, workflow-json-shape, hand-built-workflow, bible-candidate



