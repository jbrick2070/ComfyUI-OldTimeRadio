# `meta.story_brief` v2 — go-forward plan

**Branch:** `v2.0-alpha`
**Format:** one commit = one unit of work. Each commit runs the five-step cycle: **research → code → wired → regression → commit**. No commit closes until all five steps pass.

---

## The five-step unit (applies to every C-numbered commit below)

| Step | Definition | Skip rule |
|---|---|---|
| **research** | Round-robin or targeted ADR if non-trivial. ChatGPT + Gemini per CLAUDE.md round-robin section. | Skip for mechanical edits, typo fixes, one-line bug fixes with known cause. |
| **code** | Edit the source files. UTF-8 no BOM. No `dummy` (use `placeholder`/`stub`). No profanity. | Never skipped. |
| **wired** | Update the workflow JSON to match the new node/widget surface. CLAUDE.md prime directive 3 — a code change isn't done until it's wired. | Skip only if the change has no node/widget/socket surface (pure helper module, internal refactor). |
| **regression** | AST parse + Bug Bible regression (`python -m pytest C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py -v`) + any per-file self-tests + audio byte-identity (`tests/v2/test_audio_byte_identical.py`) when the audio path is touched. Baseline 23/1/2 must hold. | Never skipped. |
| **commit** | Write `.git\COMMIT_EDITMSG` via the file tool (never inline through cmd for multi-line). `git commit -F .git\COMMIT_EDITMSG` via Desktop Commander cmd. Verify `git log -1 --format=%H%n%s%n%n%b` before push. | Never skipped. |

**Standing rules (locked across this plan):**
- UTF-8 no BOM. Always.
- Never use the word "dummy". Use `placeholder` / `stub` / descriptive name.
- 14.5 GB VRAM ceiling. Never violated.
- Audio output stays byte-identical to the baseline at every gate.
- Cmd shell only for git. Never PowerShell.
- One git push attempt max per commit; if it fails, hand a PowerShell block.

**Canonical docs this plan executes against:**
- `docs/2026-05-12-story-brief-v2-problem-statement.md` (Jeffrey)
- `docs/2026-05-12-story-brief-v2-research.md` (R1, Cowork)
- `docs/2026-05-12-story-brief-v2-design-refinements.md` (locked design surface)

---

## Decision gate — before any code lands

**Open round-robin question (refinement §12, item 6.2):**

> Reflection-pass call-site position — inside `OTR_LedgerScriptWriter.execute()` (new section after K.5, before return) vs new `OTR_StoryBriefReflection` node between writer and FreezeCascade.

**Run this round-robin BEFORE C5.** Output: one paragraph in `docs/2026-05-1?-story-brief-call-site-decision.md`. Decision affects C5 + C6 + workflow JSON wiring.

Default lean if no round-robin: **inside writer** (cohesion with the cast-lock + visual_plan stamp that already lives in section K.5; no new workflow JSON node; ledger stamped before writer returns).

---

## Phase 1 — Pre-flight cleanbreaks (4 commits, mechanical)

These ship in order. Each closes Bug Bible green before the next starts.

### C1 — era literals delete

**Why:** brief testing pollution risk. If a hardcoded "1940s" lives in a FLUX prompt, soak-test visual drift could come from the brief or from the literal fighting it.

| Step | Action |
|---|---|
| research | Skip — mechanical replacement, two files. |
| code | `visual/batch_flux_portrait_render.py:107` — replace `style_anchor` default `"1940s noir radio drama style"` with era-neutral text (e.g. `"head-and-shoulders studio portrait, neutral lighting"`). `nodes/otr_video_plan.py:79` — drop `"1980s broadcast aesthetic"` from `_DEFAULT_STYLE_TAIL`, keep cinematic-grammar parts. |
| wired | No workflow JSON change — defaults live in code, no widget surface. |
| regression | AST parse + Bug Bible (23/1/2). Spot-check: grep `"1940s\|1980s"` returns zero hits in `visual/` and `nodes/`. |
| commit | Subject: `cleanbreak: delete hardcoded era literals from FLUX portrait + video plan defaults` |

### C2 — `_GENRE_BY_STYLE` deletion + `meta.ltx_style_brief` retirement

**Why:** R1 research §1.4 — dead-code categorical projection of `style`. No FLUX/LTX/HuMo/MusicGen consumer; only HUD + treatment-text display reads it, and those fall through `style or genre or "..."` which never fires when `style` is present. Same shape: `meta.ltx_style_brief` is an orphaned field stamped by deleted code paths — `_generate_ltx_style_brief` in legacy `story_orchestrator.py` is no longer called by the LPL writer (R1 research §4), and the LTX node dropped consumption in BUG-LOCAL-112. Two orphaned-field retirements with the same forensic profile — bundle as one cleanbreak.

| Step | Action |
|---|---|
| research | Skip — R1 §1.3 already confirmed zero live consumers via full grep. Belt-and-suspenders: re-grep `_GENRE_BY_STYLE\|_resolve_genre\|_preview_genre\|visual_plan\[.genre.\]\|_generate_ltx_style_brief\|ltx_style_brief` on current HEAD before editing. |
| code | (a) **Genre delete.** `nodes/OTR_LedgerScriptWriter.py:236-301` — delete the table, both resolvers, the block comment. `nodes/OTR_LedgerScriptWriter.py:2400` — drop the `"genre": _resolve_genre(...)` key from the `visual_plan` stamp. `nodes/otr_video_plan.py:306` — drop `"genre": visual_plan.get("genre") or ""` from `_visual_plan_from_script_json` projection. `nodes/video_engine.py:711, 836, 1075` — collapse `style or genre or "..."` to `style or "..."` (the `or "sci-fi"`/`or "audio drama"` tails stay). `tests/test_musicgen_style_palette.py:229-331` — delete the genre-drift guards and `_resolve_genre`/`_preview_genre` tests. (b) **`ltx_style_brief` retirement.** `nodes/story_orchestrator.py:3398-3472` — delete `_LTX_STYLE_BRIEF_PROMPT` constant and `_generate_ltx_style_brief` helper. `nodes/batch_ltx_render.py:396` — update the stale comment block claiming `ledger.meta.ltx_style_brief` is "still stamped" — replace with one-line forensic note referencing this commit. |
| wired | No workflow JSON change — neither `visual_plan.genre` nor `meta.ltx_style_brief` was widget-bound. |
| regression | AST parse + Bug Bible (23/1/2). Render one episode end-to-end through the HUD + treatment surfaces; confirm HUD `STYLE` row and treatment `Style:` line render correctly from `meta.style` alone. Final grep gate: `_GENRE_BY_STYLE\|_resolve_genre\|_preview_genre\|_generate_ltx_style_brief\|ltx_style_brief` returns zero hits in `nodes/` and `visual/` (forensic comments referencing this commit by hash are the only allowed exception). |
| commit | Subject: `cleanbreak: delete _GENRE_BY_STYLE + meta.visual_plan.genre + retire meta.ltx_style_brief` |

### C3 — VRAM budget (default model + threshold + context cap)

**Why:** refinement §11.1-§11.3. The reflection pass adds a second LLM invocation per episode. Mistral-Nemo at ~7.5 GB NF4 leaves no co-residency room for Bark/MusicGen/FLUX/LTX/HuMo. Gemma-4-E4B at ~3 GB makes the second call free in VRAM terms.

| Step | Action |
|---|---|
| research | Skip — refinement §11 is the locked decision. Pre-flight: smoke-test `google/gemma-4-E4B-it` load on the 5080 Laptop, confirm `<3.5 GB` resident, confirm 1-token warmup pass works. |
| code | (a) `nodes/story_orchestrator.py` — flip default `model_id` constant from `mistralai/Mistral-Nemo-Instruct-2407` to `google/gemma-4-E4B-it`. (b) `_MODEL_CONTEXT_CAPS["google/gemma-4-E4B-it"] = 8192` (was 16384). (c) `story_orchestrator.py:~580` — flagship threshold `15.0 → 14.5` for the `total_vram >= 15.0` gate, plus any sibling sites. (d) `nodes/OTR_LedgerScriptWriter.py:_MODEL_CHOICES` — keep dropdown order but confirm Gemma-4-E4B is first option. |
| wired | `workflows/otr_scifi_16gb_full.json` — Node 1 (`OTR_LedgerScriptWriter`) `widgets_values` default flips Mistral-Nemo → Gemma-4-E4B-it. **Position-pinned binding — confirm widget index matches the saved workflow before editing.** |
| regression | AST parse + Bug Bible (23/1/2). Run one full episode through ComfyUI Desktop on Gemma-4-E4B. Confirm: VRAM peak ≤14.5 GB, audio byte-identical to a previous baseline (or document the new baseline if model swap legitimately changes audio path identity). |
| commit | Subject: `vram: default model gemma-4-E4B-it, threshold 14.5, gemma ctx cap 8192` |

### C4 — orphan-thread hard sync barrier on timeout

**Why:** refinement §11.4. Current `_run_with_timeout` does `executor.shutdown(wait=False)` — abandoned worker continues GPU work, next visual node OOMs. Reflection pass adds a second LLM call (+ optional third repair pass) — three orphan-thread opportunities. **Non-negotiable** before the reflection pass ships.

| Step | Action |
|---|---|
| research | Required — round-robin (ChatGPT + Gemini) on the hard-sync sequence. Confirm `future.cancel() + torch.cuda.synchronize() + torch.cuda.empty_cache() + gc.collect() + mem_get_info() headroom check` is the right order, and that `REQUIRED_HEADROOM_GB` constant should be derived from the next-phase model footprint (FLUX ≈4.5 GB, LTX ≈6 GB, HuMo ≈8 GB) — pick `4.0 GB` as a conservative floor. |
| code | `nodes/story_orchestrator.py::_run_with_timeout` — replace the `executor.shutdown(wait=False)` body with the §11.4 reference implementation. Add `REQUIRED_HEADROOM_GB = 4.0` module constant. Audit every caller: the new behavior raises `_LLMTimeout` whether the orphan reclaims or not, but callers should not assume VRAM is reclaimed — they should treat `_LLMTimeout` as fatal for the current phase. |
| wired | No workflow JSON change. |
| regression | AST parse + Bug Bible (23/1/2). New test: `tests/test_run_with_timeout_orphan_sync.py` — mock a hanging LLM call, assert (a) `_LLMTimeout` raises, (b) `torch.cuda.empty_cache` was called, (c) `mem_get_info()` was queried, (d) headroom-fail path raises a second `_LLMTimeout` with the diagnostic message. Soak run: writer composition pass followed by deliberate timeout, followed by FLUX render — confirm FLUX does not OOM. |
| commit | Subject: `vram: hard sync barrier on _LLMTimeout — cancel + sync + empty_cache + headroom check` |

---

## Phase 2 — Build sprint (5 commits)

### C5 — reflection-pass module + storage schema

**Why:** refinement §2-§4 + §12 decision 6.2. Heart of the feature.

| Step | Action |
|---|---|
| research | Required — round-robin on the reflection prompt template. Start from the legacy `_LTX_STYLE_BRIEF_PROMPT` (R1 research §4 — note: the constant itself is being deleted in C2, so capture it from `git show v2.0-alpha^^...` or the R1 paper before that commit lands), rewrite for the new input shape (`lines[]` + `cast[]`) and broader scope (full scene flavor, not just radio room). ChatGPT + Gemini critique two iterations before locking. **Also: 6.2 call-site decision must close per the Decision Gate section above — required before this commit starts.** |
| code | New module `nodes/_otr_story_brief.py` containing: (a) `_build_reflection_input(led) -> str` — capped input builder per refinement §2, hard caps on opening/closing line counts. (b) `_REFLECTION_PROMPT` constant — strict-JSON schema, no-period-no-location rule per §3.3, ≤250 token prompt body. (c) `_validate_brief(parsed, cast_names) -> tuple[bool, list[str]]` — the §3.4 gate. (d) `_REPAIR_PROMPT` constant + repair-and-revalidate flow per §3.5. (e) `generate_story_brief(led, generate_fn) -> dict` — orchestrator returning the full stamping dict per §4. Empty-string-with-status on terminal failure per §4.1. Call-site wiring per the 6.2 decision (inside `OTR_LedgerScriptWriter.execute()` after K.5, OR new `OTR_StoryBriefReflection` node — pick before this commit starts). |
| wired | If new-node decision: add node to workflow JSON between writer and FreezeCascade. If writer-inline: no workflow JSON change. Either way: confirm `meta.story_brief` and the seven sidecar fields land in the saved ledger. |
| regression | AST parse + Bug Bible (23/1/2). New self-test in the module: 6 cases — happy path, named-character rejection, dialogue-verb rejection, invented-period rejection, JSON-malformed rejection, repair-pass success. Mock `generate_fn` per the news_interpreter pattern. Audio byte-identity must hold (the reflection pass does not touch the audio path). |
| commit | Subject: `feat(story_brief): reflection pass + validation + repair + empty-with-status storage` |

### C6 — central consumer helpers

**Why:** refinement §5. Five helpers, one shared module. Stops N consumer files from each re-parsing the brief.

| Step | Action |
|---|---|
| research | Skip — refinement §5 spec is unambiguous. |
| code | Extend `nodes/_otr_story_brief.py` (or sibling `nodes/_otr_story_brief_helpers.py`) with: `get_story_brief_full(meta) -> str`, `get_story_brief_ltx(meta, max_chars=90) -> str` (sentence/clause boundary trim, never mid-word), `get_story_brief_lighting(meta) -> str` (lighting + atmosphere terms joined — **join order: `lighting_terms` first, `atmosphere_terms` second**; concrete cues lead in portrait prompts, e.g. `"swinging bare bulb, harsh shadows, tense"` reads more naturally than the reverse), `get_story_brief_music_mood(meta) -> list[str]` (intersect with MusicGen's existing `_MOOD_TAGS` vocabulary), `get_story_brief_status(meta) -> {'ok','failed','absent'}`. Each helper returns empty/empty-list on absent/failed status — consumers fall through cleanly. |
| wired | No workflow JSON change. |
| regression | AST parse + Bug Bible (23/1/2). New self-test: 5 cases per helper, parameterized over status `ok` / `failed` / `absent`. Boundary-trim test for `get_story_brief_ltx` — confirm no mid-word truncation across 20 random brief samples. Music-mood test — confirm intersection only returns known MusicGen mood vocabulary. |
| commit | Subject: `feat(story_brief): central consumer helpers (full/ltx/lighting/music_mood/status)` |

### C7 — FLUX integration wave (env + radio bookend + portraits)

**Why:** refinement §6. Three FLUX surfaces. Highest leverage — every FLUX render consumes one of them.

| Step | Action |
|---|---|
| research | Required for portraits — round-robin on whether `get_story_brief_lighting` helps or hurts head-and-shoulders composition. Skip for env + radio bookend (refinement §6 placement is unambiguous). |
| code | (a) `visual/batch_flux_render.py::_parse_env_prompts` — insert `get_story_brief_full(meta)` between env description and style_suffix tail. (b) `visual/batch_flux_render.py::_build_dynamic_radio_prompt` — replace tier 4 (`scenes[0].env`) and tier 5 (`episode_id`) with a single `get_story_brief_full(meta)` tier, sandwiched between tier 3 (`style_custom`) and the hardcoded `_RADIO_FALLBACK_PROMPT` tier. (c) `visual/batch_flux_portrait_render.py::_build_portrait_prompt` — append `get_story_brief_lighting(meta)` after `appearance`. All three sites: add `meta` parameter to the function signature, thread the ledger meta dict in from each call site (one site per file). |
| wired | No workflow JSON change — these are internal prompt builders called by existing nodes. |
| regression | AST parse + Bug Bible (23/1/2). Render-side smoke: one episode through each FLUX consumer (env still, radio bookend, portraits). Visual eyeball — confirm brief content shows up in renders without overwhelming subject/composition. Audio byte-identity must hold. |
| commit | Subject: `feat(story_brief): FLUX integration — env + radio bookend + portraits` |

### C8 — LTX + HuMo + MusicGen + `otr_video_plan` integration wave

**Why:** refinement §6. Four downstream consumers. Per refinement §6.1 + §11.5, LTX gets the tightest budget.

| Step | Action |
|---|---|
| research | Required for HuMo — round-robin on clean-append vs replace `_DEFAULT_POS_SUFFIX`. Refinement §6 spec is "append before `_DEFAULT_POS_SUFFIX`", which is the conservative choice. Confirm and lock. Required for MusicGen — confirm the keyword-merge with existing `_mood_suffix(script_brief)` doesn't double-tag the same mood. |
| code | (a) `nodes/batch_ltx_render.py::_build_ltx_role_prompt` — append `get_story_brief_ltx(meta, max_chars=90)` after the role template; drop the brief fragment if combined length would push motion verbs past char 140. Hard-cap total prompt at 220-240 chars. (b) `nodes/batch_humo_render.py::_build_pos_prompt` — insert `get_story_brief_lighting(meta)` between `speaker_desc` and `_DEFAULT_POS_SUFFIX`. (c) `nodes/musicgen_theme.py::_mood_suffix` — accept second optional source arg `story_brief_mood_kws: list[str] = None`; merge with `script_brief` keyword scan, dedupe. Update `_resolve_cue_from_style` and `MusicGenTheme.render` to pass `get_story_brief_music_mood(meta)` in. (d) `nodes/otr_video_plan.py::compose_shot_prompt` — replace the `scene_visual` slot (currently always empty per OTR_LedgerScriptWriter.py:2398) with `get_story_brief_full(meta)`. Thread `meta` through the three `build_*` helpers. |
| wired | No workflow JSON change. |
| regression | AST parse + Bug Bible (23/1/2). Render-side smoke: one episode end-to-end through all four consumers. Three checks: (1) LTX MAD between consecutive frames stays in motion range (≥6.0; BUG-LOCAL-112 measured 1.86-5.92 as the failure mode). (2) HuMo lip-sync visual quality preserved — eyeball one character-line clip. (3) MusicGen cue produces expected mood when brief carries a recognized atmosphere term. Audio byte-identity must hold for non-MusicGen audio; MusicGen output legitimately changes — document the new baseline if it does. |
| commit | Subject: `feat(story_brief): LTX + HuMo + MusicGen + otr_video_plan integration` |

### C9 — three-ugly-ledger fixtures + canaries + soak

**Why:** refinement §9. Normal-case fixtures will pass anything. Brief earns its keep on adversarial cases.

| Step | Action |
|---|---|
| research | Skip — refinement §9 spec is unambiguous. |
| code | (a) **Fixtures + tests.** `tests/fixtures/story_brief/` directory with three ledger fixtures: noir slug + space-colony script, detective script with no clear setting, long script (15+ min) with three distinct locations. New `tests/test_story_brief_three_ugly_ledgers.py` — for each fixture: run reflection pass, assert brief follows script (not slug), assert no invented periods/locations, assert dominant scene picked for the multi-location case. (b) **Downstream-consumer canary.** `tests/test_downstream_prompt_contract.py::test_story_brief_lands_in_all_eight_consumers` — render each downstream prompt with a known-content brief, regex-match brief fragments in the assembled prompts. This is the only check that catches "forgot to thread `meta` into one of the consumer call sites" silent failure. (c) **VRAM instrumentation (in-process).** Add `_log_vram_peak(label)` helper to `nodes/_vram_log.py` — wraps `torch.cuda.memory_allocated()` + `torch.cuda.max_memory_allocated()` and logs `[VRAM] {label}: peak={x} GB current={y} GB`. Insert calls at six phase boundaries: writer-start (entry to `OTR_LedgerScriptWriter.execute`), writer-end (before return), reflection-start (entry to `generate_story_brief`), reflection-end (before stamp), FLUX-start (entry to first FLUX render node), LTX-start (entry to `BatchLtxRender.execute`). ~30 LOC total. Phase attribution is the point — knowing peak hit 14.4 GB during reflection vs LTX changes the fix. |
| wired | No workflow JSON change. |
| regression | AST parse + Bug Bible (23/1/2). Run the new three-ugly-ledger suite plus the canary. Run a full episode soak with the new VRAM instrumentation active per refinement §11.6 — capture peak VRAM at each of the six phase boundaries for: writer composition pass (baseline), reflection pass, reflection + repair + LTX worst case. Expected ceiling ≤14.5 GB at every phase. If any phase exceeds, the failure attribution is concrete and the fix lands per-phase. |
| commit | Subject: `test(story_brief): three ugly ledgers + consumer canary + per-phase VRAM instrumentation` |

---

## Round-robin closes after C9

After C9 ships, story_brief is end-to-end live. Run an A/B sanity check (per the news_interpreter sprint pattern, ROADMAP §128):

- 10 episodes through pre-C5 path (saved baseline) vs 10 episodes through post-C9 path, same seeds.
- Eyeball: visual specificity (does the brief actually change what FLUX/LTX/HuMo render?), audio quality (any regressions on MusicGen mood?), failure rate (does the reflection pass + repair land inside the timeout window on every episode?).
- ~30 min subjective scoring. Catches the regression class unit tests miss.

If A/B passes: tag the post-C9 commit as the new soak baseline. Promote any Bug Bible candidates from `BUG_LOG.md` to the survival guide per the standing batch-promote rule.

If A/B fails: log to `BUG_LOG.md`, do not promote, fix and re-A/B.

---

## Commit dependency graph

```
C1 ─┐
C2 ─┤
C3 ─┼─► C4 ──► C5 ──► C6 ──► C7 ──► C8 ──► C9 ──► A/B sanity check
    │
6.2 round-robin (closes before C5)
```

- C1, C2, C3 can ship in parallel or any order (pre-flight cleanbreaks are independent).
- C4 must close before C5 (orphan-sync is infrastructure for the reflection pass).
- 6.2 round-robin must close before C5 (call-site decision affects the C5 commit shape).
- C5 → C6 → C7 → C8 → C9 are strictly sequential (each depends on the previous helper or surface).

---

## Definition of done

The plan closes when:

1. All 9 commits land on `v2.0-alpha` with Bug Bible 23/1/2 holding after each.
2. A/B sanity check passes.
3. `meta.story_brief` is populated on every produced ledger; `story_brief_status` is `ok` on ≥95% of episodes; `failed` on <5%; reflection-pass timeout rate is <1%.
4. The 8 consumer prompt-assembly sites in refinement §6 all consume the brief through one of the §5 helpers — confirmed by the C9 canary.
5. `meta.ltx_style_brief` is fully retired (zero grep hits in `nodes/` and `visual/` outside the BUG_LOG forensic entry).
6. ROADMAP.md `NEXT SPRINT — meta.story_brief v2` section marked COMPLETE; bullet update copied into ROADMAP_HISTORY.md.

---

End of plan.
