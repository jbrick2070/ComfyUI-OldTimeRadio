# Sprint C plan v1 — Cowork review findings (first pass, with code access)

**Reviewer:** Cowork (code-access pass — first of three planned passes).
**Subject:** `docs/2026-05-15-sprint-c-story-brief-v2-plan.md`.
**Branch state at review:** `s34-p0-p1-hotfix @ f758f02` (S34 closed). Sprint C branch NOT cut.
**Output:** this findings doc + `docs/2026-05-15-sprint-c-story-brief-v2-plan-v2.md`.

The v1 plan was authored 2026-05-12 (pre-S30). Five sprints have shipped between authorship and now (S30, S31, S31.5, S32, S33, S34). Three of v1's nine staleness-audit targets have ALREADY been delivered by intervening sprints. Two more are off in line-number space and need correction. One pre-flight item (the `_run_with_timeout` "hard sync barrier") describes a behavior the current code DELIBERATELY rejects on CUDA-safety grounds.

The plan also under-scopes the era-literal cleanbreak (C2): the locked-plan claim points at one site per file; the actual code has 3 sites in `batch_flux_portrait_render.py` plus a workflow JSON occurrence.

This is a first pass. Round-robin passes (Gemini, ChatGPT) without code access will catch different things. Expect 2-3 cycles before v2 stabilises.

---

## Severity legend

- **HIGH** — actively breaks the sprint, ships a regression, or invalidates a locked decision.
- **MEDIUM** — wastes a commit, hides a hidden assumption, or under-scopes a target.
- **LOW** — clerical / readability / forensic accuracy.

Items that survived review unchanged are listed in the final section so you can see what was checked.

---

## Findings

### F-01 [HIGH] — C4 staleness: VRAM threshold `15.0 → 14.5` is ALREADY DONE

**Plan v1 claim:** C4 pre-flight cleanbreak 3 includes `Flagship VRAM threshold 15.0 → 14.5`.

**Code state:** No `15.0` VRAM threshold exists in the codebase. Every threshold reference is already `14.5`:
- `nodes/_otr_model_catalog.py:510` — `DEFAULT_VRAM_CEILING_GB = 14.5`
- `nodes/_otr_model_loader.py:396` — `if total_vram >= 14.5:`
- `nodes/_vram_log.py:45` — `VRAM_CEILING_GB: float = 14.5`
- `visual/lhm_monitor.py:47` — `VRAM_CEILING_GB: float = 14.5`
- ~24 additional comment / docstring mentions, all `14.5`.

**Authoritative ROADMAP entry:** the move from 15.0 → 14.5 shipped in **S21.1** (commit `6d08f63`), per `ROADMAP.md:546`:
> `S21.1+2 | Flagship VRAM threshold 15.0 -> 14.5 + Gemma 4 E-series context cap 16K -> 8K`

**Proposed v2 change:** drop the threshold item from C4. Reference S21.1 as the authority. C4 shrinks to just the model-default and sync-barrier items — both of which also need rethinking (see F-02 and F-04).

---

### F-02 [HIGH] — C4 staleness: Gemma-4-E4B context cap `16384 → 8192` is ALREADY DONE

**Plan v1 claim:** C4 includes `Gemma-4-E4B context cap 16384 → 8192`.

**Code state:** Already at 8192.
- `nodes/_otr_model_catalog.py:417` — `_hard_vram_context_limit()` returns 8192 (with env-var override).
- `nodes/_otr_model_catalog.py:420` — `HARD_VRAM_CONTEXT_LIMIT = _hard_vram_context_limit()` → 8192.
- `nodes/_otr_model_catalog.py:429-436` — `CURATED_CONTEXT_OVERRIDES` pins gemma-4-E4B-it explicitly at 8192.

**Authoritative ROADMAP entry:** same S21.1 commit `6d08f63` (`ROADMAP.md:546`). The override constants also landed in S30 B1b (`d307348`).

**Proposed v2 change:** drop the context-cap item from C4. Reference S21.1 + S30 B1b. If the design refinement §11 still wants something different from the current 8192, that needs to be stated explicitly — but the v1 plan reads as if the move from 16K to 8K still needs doing, and it does not.

---

### F-03 [HIGH] — C4 staleness + AUDIO RISK: changing `DEFAULT_LLM` from Mistral-Nemo to Gemma-4-E4B threatens audio C7 byte-identity

**Plan v1 claim:** C4 includes `Default model Mistral-Nemo → google/gemma-4-E4B-it`.

**Code state:** `nodes/_otr_model_catalog.py:32` — `DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"`. Catalog `notes` field on this entry: **"Audio C7 regression baseline -- soak-tested. Default for both slots."** Both writer slots (`creative_writing_model`, `technical_model`) default to this id (`OTR_LedgerScriptWriter.py:1044-1045`).

**Why this is HIGH:** Sprint C plan's Hard Rule 1 says "Audio C7 byte-identical pytest proxy must hold on default-config happy path at every commit boundary." Changing `DEFAULT_LLM` changes the model that produces the default-config happy path. The catalog literally calls Mistral-Nemo the **audio C7 byte-identical baseline**. A different model will produce different tokens, which produces different line text, which produces different audio. C7 will NOT hold.

This is not theoretical. The S30 acceptance gate explicitly preserved Mistral-Nemo as Slot 2 default so audio C7 byte-identity held across the two-model split (ROADMAP scoping doc §10 decision 6).

**Open question this raises** (do NOT resolve here — needs Jeffrey's call):
- If the goal is "gemma-4-E4B is small enough to leave VRAM headroom for the reflection pass," is the cost of re-establishing an audio C7 baseline acceptable? Re-soak + new audio fixtures + likely BUG_LOG entries.
- Or: keep Mistral-Nemo as `creative_writing_model` default; only the reflection pass (technical slot? or its own slot?) runs Gemma-4-E4B. The two-model selector already supports this — no DEFAULT_LLM change needed.
- Or: the reflection-pass VRAM envelope holds at Mistral-Nemo + reflection-on-same-model; no default change needed at all.

**Proposed v2 change:** REMOVE the `DEFAULT_LLM` change from C4. Replace with an explicit decision item: "How does the reflection pass acquire its model id?" with three options scoped (keep default, run on technical_model, dedicate a new slot). Surface to round-robin alongside §6.2 call-site question. Until resolved, the C4 commit cannot delete `DEFAULT_LLM = "mistralai/..."`.

---

### F-04 [HIGH] — C4 staleness + REGRESSION RISK: "hard sync barrier on _LLMTimeout" describes a behavior current code DELIBERATELY rejects

**Plan v1 claim:** C4 includes `_run_with_timeout orphan-thread hard sync barrier on _LLMTimeout`. Refinement §11.4 is referenced as the spec.

**Code state:** `nodes/story_orchestrator.py:301-397`. The current implementation is `executor.shutdown(wait=False)` — explicitly NON-blocking. The orphan worker thread is left to drain on its own. After timeout, the code calls `invalidate_cache_no_gpu_teardown()` to invalidate cache dict references **without touching the GPU**, then raises `_LLMTimeoutWorkflowPause` to halt the workflow.

The 30-line comment block at lines 336-372 explains why a hard sync barrier was DELIBERATELY NOT IMPLEMENTED:

> When FuturesTimeout fires, the worker thread is still running an LLM forward pass on GPU. Python cannot safely terminate threads, and `executor.shutdown(wait=False)` does NOT kill the worker — it keeps churning until its forward pass completes naturally (could be 30-60+ more seconds for a 16K prompt). Result: the GPU has in-flight kernels the main thread doesn't control. The cached model instance thinks it's idle but the orphan is still mutating its tensors. The NEXT phase that calls `model.cpu()` / any CUDA op collides with the orphan's stale ops and Python aborts with `cudaErrorIllegalAddress`.

This is **BUG-LOCAL-228**, fixed in S31 B4 (`a4fe67a`). A hard sync barrier — making the timeout wait for the orphan worker — was the bug, not the fix. The fix is the **opposite** of a sync barrier: do not block, do not touch GPU, raise `_LLMTimeoutWorkflowPause` to halt the queue.

**Why this is HIGH:** the v1 plan's C4 item, if executed literally per §11.4 "hard sync barrier" wording, would reintroduce BUG-LOCAL-228. The C4 test `test_run_with_timeout_hard_sync_barrier` would test for the wrong behavior. C4 would ship a CUDA-race regression.

**Possible benign interpretation:** the refinement doc may use "hard sync barrier" to mean "make the timeout cleanly recoverable from the next phase's perspective" — which is what `invalidate_cache_no_gpu_teardown` + `_LLMTimeoutWorkflowPause` already do. If so, the item is ALREADY DONE (S31 B4) and C4 should drop it with a reference to BUG-LOCAL-228.

**Proposed v2 change:** drop the sync-barrier item from C4 entirely. Add a finding to C1 staleness audit: refinement §11.4 must be re-read against the BUG-LOCAL-228 fix BEFORE the round-robin pass touches the reflection-pass plumbing. If §11.4 really wants the orphan thread joined synchronously, that needs to be argued against BUG-LOCAL-228, not assumed.

---

### F-05 [MEDIUM] — C2 under-scopes era-literal removal: 3 sites in `batch_flux_portrait_render.py`, not 1; workflow JSON also affected

**Plan v1 claim:** C2 removes the era literal at `visual/batch_flux_portrait_render.py:107` (`"1940s noir radio drama style"`).

**Code state:** the literal "1940s noir radio drama style" appears in `batch_flux_portrait_render.py` at THREE sites:
- **Line 109** — `style_anchor = (style_anchor or "1940s noir radio drama style").strip()` — fallback inside `_build_portrait_prompt`. (Plan says 107, actually 109.)
- **Line 170** — `"default": "1940s noir radio drama style, cinematic"` — ComfyUI widget default for the optional `style_anchor` input on `BatchFluxPortraitRender.INPUT_TYPES`.
- **Line 234** — `style_anchor: str = "1940s noir radio drama style, cinematic"` — function signature default on `BatchFluxPortraitRender.execute()`.

Plus a workflow JSON occurrence:
- `workflows/otr_scifi_16gb_full.json:1915` — `"1940s noir radio drama style, cinematic"` (the widget value persisted in the canonical workflow JSON, separate from the Python widget default).

A test fixture also carries the string:
- `tests/test_musicgen_cache_keys.py:23` — `"1940s noir radio drama style, brass + strings"` as a cache-key payload. This is fixture data exercising the cache hash, not a production read; the forbidden-sweep marker C3 introduces must NOT trip on test fixtures.

**Same applies to `nodes/otr_video_plan.py`:** the `_DEFAULT_STYLE_TAIL` literal starts at line 78-81 (multi-line), with `"1980s broadcast aesthetic"` on line 79 (the plan says line 79, correctly — but the constant itself is named at line 78). And `workflows/otr_scifi_16gb_full.json:706` ALSO carries that string as a widget value.

**Why this is MEDIUM:** if C2 only edits the Python fallback at line 109, the widget default at line 170 + the function-signature default at line 234 + the persisted workflow JSON value at 1915 all still emit "1940s noir radio drama style" at runtime. The cleanbreak ships without actually cleaning the break.

**Proposed v2 change:** rewrite C2 scope to enumerate all four sites for the noir literal plus both sites (Python + workflow JSON) for the 1980s literal. Add a per-prompt-renderer wiring check: workflow JSON re-saved with the era-neutral default before C2 closes (Prime Directive 3: "code change isn't done until it's wired").

---

### F-06 [MEDIUM] — C3 line-number drift: `_GENRE_BY_STYLE` is at line 254, not 246-301; `meta.visual_plan.genre` stamp is at line 2801, not 2400

**Plan v1 claim:** `_GENRE_BY_STYLE` at `nodes/OTR_LedgerScriptWriter.py:246-301`; `meta.visual_plan.genre` stamp at line 2400.

**Code state:**
- `_GENRE_BY_STYLE` definition starts at **line 254** (table runs to line 265). Helpers `_resolve_genre` at line 268, `_preview_genre` at line 296. Whole block is roughly lines 244-309. Off by ~8 lines from the v1 plan's stated 246-301 range.
- `meta.visual_plan.genre` stamp is at **line 2801** (inside section K.5 of `execute()`). The v1 plan claim of "line 2400" is off by ~400 lines.
- Forensic comment reference at line 2764 (inside section K.5, referencing `_GENRE_BY_STYLE` by name).

The video_engine fall-through line numbers (**711, 836, 1075**) in `nodes/video_engine.py` are **correct** in the v1 plan.

**Why this is MEDIUM:** C1's job is to fix these line refs before C3 executes. Worth flagging now so C1 doesn't think `meta.visual_plan.genre` was deleted by an intervening sprint just because the line number is way off.

**Proposed v2 change:** correct the line numbers in the C3 "Locked-scope targets" section and the staleness-audit table. Note in C1 that the video_engine fall-throughs are correct (rare bit of good news for a five-sprint-old plan).

---

### F-07 [MEDIUM] — Sprint A overlap with C5 needs explicit handoff contract, not just "C closes first"

**Plan v1 claim:** Sprint A (downstream verification) opens after Sprint C closes. Sprint A is "queued."

**ROADMAP state (`ROADMAP.md:725-756`):** Sprint A's scope is verifying FLUX env / FLUX portraits / LTX motion / HuMo lip-sync against the post-C contract. Sprint A's own gating says:
> Audio C7 byte-identity must hold against the **post-C3 baseline** (the Gemma-4-E4B-it audio output, documented in the C3 commit).

This is the same baseline-redefinition problem flagged in F-03. Sprint A is GATED on a post-C3 baseline change that, per F-03, hasn't actually been re-validated and may not be safe to ship.

Also, Sprint A's scope item 2 says: "Confirm motion verbs lead, brief fragment lands at the §6.7 budget (220-240 chars total, 80-100 chars brief), **no `meta.ltx_style_brief` fallback fires**." Sprint A is checking for the absence of a thing Sprint C is supposed to delete. Order matters but the contract is currently informal ("C closes first").

**Why this is MEDIUM:** Sprint A's verification target is whatever Sprint C ships. If Sprint C ships a partial brief or leaves `meta.ltx_style_brief` around as a comment / stale doc reference, Sprint A's check passes silently when it shouldn't.

**Proposed v2 change:** add to C-final acceptance an explicit handoff bullet — "C-final defines the post-C contract Sprint A must verify against. Specifically: (a) `meta.story_brief` present on every non-failed run; (b) `meta.story_brief_status` present on every run; (c) zero reads of `meta.ltx_style_brief` in `nodes/` + `visual/` excluding forensic comments; (d) audio C7 baseline identifier (model id + seed + style) documented in the C-final commit message so Sprint A's audio gate is anchored." This protects Sprint A from a soft handoff.

---

### F-08 [MEDIUM] — K.5 → return placement: the L + M sections between K.5 and return need explicit call-site contract

**Plan v1 claim (§6.2 open question):** Reflection-pass call site is "inside `OTR_LedgerScriptWriter.execute()` (after K.5, before return)" vs separate node.

**Code state of `OTR_LedgerScriptWriter.execute()` lines 2660-2845:**
- Section K (lines 2659-2755) — stamp meta block (gen_params, model ids, slot scheduler stats, title).
- Section K.5 (lines 2757-2803) — stamp `meta.visual_plan` + `meta.style`.
- Section L (lines 2805-2824) — assemble return values: `script_text = _PL.assemble_script_text_from_ledger(led.data)`, `script_json`, `news_json`, word counts.
- Section M (lines 2826-2845) — `saved_path = led.save()`, log, RETURN tuple.

"After K.5, before return" is ambiguous between two materially-different placements:
1. **K.5.5 (between K.5 and L)** — reflection writes to `meta.story_brief` BEFORE script_text assembly. Safe if reflection writes only to meta (the plan promises this). `script_text = _PL.assemble_script_text_from_ledger` reads `led.data["lines"]` only, so meta writes don't disturb it.
2. **L.5 (between L and M)** — reflection writes to `meta.story_brief` AFTER script_text assembly but BEFORE `led.save()`. Also safe for audio (script_text already computed). Slightly cleaner separation.
3. **NOT VALID: after M** — reflection writes after `led.save()`. The persisted ledger on disk would NOT contain `meta.story_brief`. Downstream consumers reading the saved ledger from path would miss the brief.

S31-S34 did NOT add anything between K.5 and the return that interferes — sections L and M are unchanged in shape since the writer's K.5 introduction. **But the audio C7 byte-identical guarantee requires that the reflection pass NEVER touch `led.data["lines"]`** — if it did, `assemble_script_text_from_ledger` would return different output, audio path consumes script_text indirectly via line text. Tests must enforce this.

**Why this is MEDIUM:** the v1 plan's "after K.5, before return" wording is correct in intent but loose enough to permit a save-order bug. The round-robin question (§6.2 inside-writer vs separate-node) doesn't surface this nuance — both placements work, but the inside-writer one needs an explicit pre-save constraint.

**Proposed v2 change:** Do NOT resolve §6.2 in v2 — that's explicitly out of scope. But ADD an annotation to §6.2: "If inside-writer is chosen, the call site must land BEFORE `led.save()` at section M (line 2827). After-save is incorrect. Section L (return-value assembly) is the latest safe placement." Add a C5 unit test: `test_story_brief_persists_in_saved_ledger` that loads the ledger from disk post-save and asserts `meta.story_brief` is present.

---

### F-09 [MEDIUM] — `_run_with_timeout` siblings: the S34 fail-loud pattern applies to the reflection pass; plan should cite it

**Plan v1 claim:** §6.3 failure mode is empty string with `story_brief_status` field (observable failure, not silent, not raise).

**Code state:** the reviewer module (`_otr_ledger_reviewer.py`) has two sibling LLM-call wrappers that were hardened to fail-loud at S34 B1:
- `audit_cast_contract` (lines 420-455): three `except` arms all return `_audit_failed_sentinel(...)` — a sentinel that downstream verifies-on.
- `run_script_doctor` (lines 815-854): three `except` arms all return `ScriptDoctorReport(overall_verdict="needs_full_rerun")`.

Both patterns: catch the failure, log warning, return a value that downstream can DETECT as "the LLM call failed, take the failure path." This is exactly the §6.3 contract for `meta.story_brief = ""` + `meta.story_brief_status = "failed"`.

**Why this is MEDIUM:** v1 plan references §6.3 abstractly. v2 should cite the S34 P0 fix as the production precedent and require C5's reflection pass to follow the same three-arm pattern (`except Exception`, `except json.JSONDecodeError`, `except ValidationError`). Without naming the precedent, a reviewer might implement a different failure shape.

The S34 P0 P1 hotfix forbidden-pattern marker is `return\s+ScriptDoctorReport\s*\(\s*\)` (locks bare-default-construction reintroduction). C5's reflection pass should ship a parallel marker locking against a fail-soft `return ""` or `return None` from the reflection function.

**Proposed v2 change:** in C5 Review section, add: "Reflection-pass exception-handling matches the S34 B1 fail-loud sentinel pattern from `_otr_ledger_reviewer.run_script_doctor` (three explicit `except` arms returning the empty-brief-with-status sentinel, log.warning on each). Forbidden-sweep marker for `return\s+\(?\s*['\"]['\"]?\s*,\s*['\"]['\"]?\s*\)?` from the reflection function (or whatever shape the new sentinel takes) added in lockstep."

---

### F-10 [LOW] — `meta.ltx_style_brief` retirement: zero LIVE consumers in Python; only one stale COMMENT remains

**Plan v1 claim:** C5 retires `meta.ltx_style_brief` across all consumers; no alias, no shim (§6.6).

**Code state:** repo-wide grep for `ltx_style_brief` in `.py` files returns:
- `nodes/batch_ltx_render.py:396` — **a stale comment** in a documentation block (not a live read). Comment text: "ledger.meta.ltx_style_brief: still stamped by OTR_LedgerScriptWriter, used by BUG-LOCAL-111 (FLUX bookend integration, future commit)." This comment is stale — the function it documents (`_build_ltx_role_prompt`) doesn't read `meta.ltx_style_brief` (it returns a fixed role-prompt dict).
- `tests/test_fetch_science_news_no_legacy_wrapper.py:164-179` — **tests that assert deletion** of `_generate_ltx_style_brief` (the writer helper) and `_LTX_STYLE_BRIEF_PROMPT` constant. Per S31 B3 these are already deleted.

So `meta.ltx_style_brief` is **already effectively retired in the production code path**. C5's job here is mostly:
1. Delete the stale comment block in `batch_ltx_render.py:395-403`.
2. Add a forbidden-sweep marker `\bmeta\.ltx_style_brief\b` to lock the field name out (plan v1 already calls this out).
3. Verify no OTHER subsystem (a workflow JSON, an external script, a docs example) refers to it.

**Why this is LOW:** the heavy lifting of retiring `_generate_ltx_style_brief` already happened in S31 B3. C5's "retire across consumers" is more like "scrub the remnants and lock against reintroduction." This is GOOD NEWS — C5 has less work than the plan suggests.

**Proposed v2 change:** in C5 Part C, change the framing from "retire ltx_style_brief references across consumers" to "remove stale `ltx_style_brief` comment in `batch_ltx_render.py:395-403`; add forbidden-sweep marker; confirm zero remaining references in `nodes/`, `visual/`, `workflows/`, `scripts/`." Less work to do, more honest about what was already done.

---

### F-11 [LOW] — C0 staleness-audit: plan v1 says "9 items" but the table actually has 9 rows AND a 10th implicit item

**Plan v1 staleness-audit table** (§"Pre-S34 state check") has these rows:
1. `_GENRE_BY_STYLE` table
2. `meta.visual_plan.genre` stamp at line 2400
3. Three `video_engine` fall-throughs at 711, 836, 1075
4. `_DEFAULT_STYLE_TAIL` at `otr_video_plan.py:79`
5. `"1940s noir radio drama style"` at `batch_flux_portrait_render.py:107`
6. Default model Mistral-Nemo (target gemma-4-E4B-it)
7. VRAM threshold 15.0 (target 14.5)
8. Gemma-4-E4B context cap (target 8192)
9. `_run_with_timeout` orphan-thread sync barrier

Implicit additional item (per F-05): the **widget defaults / function-sig defaults / workflow JSON copies** of the era literals. These are separate sites from the Python fallback at line 109. Plan v1's "1 item per file" framing misses them.

**Proposed v2 change:** expand the staleness-audit table to 10 explicit rows (split row 5 into "Python fallback default" + "widget default + function-sig default + workflow JSON"). Same expansion for row 4 (Python module-level constant + workflow JSON occurrence). C1 produces a more honest baseline.

---

### F-12 [HIGH, upgraded from LOW post-verification] — Plan v1's canonical-doc list references a phantom file that never existed in git history

**Plan v1 claim:** Three canonical docs lock the design:
- `docs/2026-05-12-story-brief-v2-problem-statement.md` ← phantom
- `docs/2026-05-12-story-brief-v2-research.md`
- `docs/2026-05-12-story-brief-v2-design-refinements.md`

**Verified state (2026-05-15, via `git log --all` against the sandbox checkout):**

```
git log --all --diff-filter=A -- 'docs/2026-05-12-story-brief-v2-problem-statement.md'  → empty
git log --all --diff-filter=D -- 'docs/2026-05-12-story-brief-v2-problem-statement.md'  → empty
git log --all              -- 'docs/2026-05-12-story-brief-v2-problem-statement.md'  → empty
```

The file **never existed** in git history. No stash carries it either (one unrelated `v2.0-visual-engine` WIP stash; nothing relevant). Fuzzy `*problem-statement*` and `*story-brief*` searches confirmed: only `2026-05-12-story-brief-v2-research.md`, `2026-05-12-story-brief-v2-design-refinements.md`, and `2026-05-13-story-brief-v2-go-forward-plan.md` exist for this surface.

**Why this is now HIGH, not LOW:** Cowork's initial pass cleared this row by confirming the files "appear in grep results." That only proved the OTHER two existed. The problem-statement file was never separately verified. Six live references to the phantom path existed across four files (v1 plan, v2 plan, ROADMAP at two sites, the 2026-05-13 go-forward plan, and this findings doc itself). Round-robin reviewers attaching the v1 plan's listed artifacts to Gemini / ChatGPT were either attaching nothing under that slot, or pulling the wrong file under that name. Their critique of `§3.6` / `§6.3` / `§11.4` is still valid (those sections live in the design-refinements doc, which DOES exist). But any claim that depends on the missing file's exact wording must be discounted.

**Verdict:** the phantom file's supposed scope ("the original ask: schema, scope, deliverable shape") is fully covered by the research doc §0 framing + design-refinements doc §1-2. No content gap.

**Action landed in v2 plan (this revision, 2026-05-15):**

1. Phantom row removed from v2 plan's "Canonical design artifacts" table.
2. Phantom row removed from `ROADMAP.md:71` (sprint-canonical-docs list) and `ROADMAP.md:590` (Sprint C canonical artifacts table).
3. Phantom bullet removed from `docs/2026-05-13-story-brief-v2-go-forward-plan.md:27`.
4. Phantom entry removed from this findings doc's Sources section.
5. Forensic note added in the v2 plan and in ROADMAP Sprint C section.
6. Round-robin prompt block updated (chat ephemera — future reviewers receive a 5-file attachment list, not 6).

Canonical design surface restated: **research inventory + design refinements** (the locked spec). The 2026-05-13 go-forward plan is historical input only, superseded by the Sprint C plan-v2 except where explicitly cited.

---

### F-13 [LOW] — C5 part C "FLUX env" consumer: the plan says read brief, but the v2 plan should name the file path

**Plan v1 claim:** C5 Part C lists six consumers: FLUX env, FLUX radio bookend, FLUX portraits, LTX motion, HuMo lip-sync, MusicGen mood.

**Code state:** the file inventory in the repo shows:
- FLUX env → `visual/batch_flux_env_render.py` (per ROADMAP Sprint A scope; not confirmed by direct read in this pass).
- FLUX radio bookend → no file with that exact name visible; possibly in `visual/` or covered by `batch_flux_env_render.py` via a render mode.
- FLUX portraits → `visual/batch_flux_portrait_render.py` (confirmed).
- LTX motion → `nodes/batch_ltx_render.py` + `_build_ltx_role_prompt()` (confirmed).
- HuMo lip-sync → `nodes/batch_humo_render.py` (confirmed exists).
- MusicGen mood → `nodes/musicgen_theme.py` (confirmed exists).

**Why this is LOW:** v2 should name each consumer file path explicitly so C5 has unambiguous file targets and the C-final scrub-test knows what to grep. The "FLUX radio bookend" lack of a clear file path is the only fuzzy entry.

**Proposed v2 change:** in C5 Part C, add a file-path column. Resolve "FLUX radio bookend" — either it's a sub-mode of `batch_flux_env_render.py` (cite the function), or it has its own module (cite that), or it doesn't exist yet and C5 needs to create it (HIGH change in scope, escalate).

---

### F-14 [LOW] — Forbidden-sweep marker for `_GENRE_BY_STYLE` needs an explicit forensic-comment carve-out

**Plan v1 claim:** C3 adds forbidden-sweep marker `\b_GENRE_BY_STYLE\b` to lock against reintroduction.

**Code state:** the existing forbidden-sweep tool (`docs/_s28_forbidden_sweep.py`, per CLAUDE.md and ROADMAP) uses tokenize classification to suppress hits inside docstrings / comments. But forensic comments citing deleted symbols by name are a recurring pattern in this codebase (standing directive 11 — "Deleted symbols don't survive as words in active code. Forensic comments cite commit hashes, not symbol names"). The S33 markers passed this gate; the S34 P0 marker `return\s+ScriptDoctorReport\s*\(\s*\)` passes this gate.

**Why this is LOW:** the existing tooling already handles this. v2 should explicitly state the marker is `\b_GENRE_BY_STYLE\b` running through `_s28_forbidden_sweep.py` so the tokenize-classified docstring suppression applies. Otherwise a reviewer might add a comment-block carve-out exception that isn't needed.

**Proposed v2 change:** in C3 commit-gate clarify "marker runs through `docs/_s28_forbidden_sweep.py` (tokenize-classified docstring/comment suppression already handles forensic mentions)."

---

## Items checked and unchanged (so the second pass sees what was reviewed)

The following were checked against current code and survived without changes:

- **video_engine.py fall-through line numbers (711, 836, 1075)** — confirmed at all three sites. `style or genre or "sci-fi"` / `("STYLE", self.data.get("style", self.data.get("genre", "?")))` / `style = style or genre or "audio drama"`. Plan v1 line refs are correct.
- **Projection in `nodes/otr_video_plan.py:306`** — confirmed. `_visual_plan_from_script_json` extracts `genre` from `visual_plan` and projects it into the video plan dict at line 311. (Plan said line 306; actual is the function starting around 280, with `genre` key at 311 — close enough, no correction needed.)
- **`tests/test_musicgen_style_palette.py` genre-table guards** — confirmed at lines 229-331 (drift guards + `_resolve_genre` semantics tests). C3's "Updated; existing genre-table tests removed" is correctly scoped.
- **Forbidden-sweep tool exists** — `docs/_s28_forbidden_sweep.py` referenced in CLAUDE.md. Tokenize-based gate script. C3 + C5 markers can integrate cleanly.
- **`_otr_ledger_reviewer.py` LLM-call wrappers** — fail-loud sentinel pattern present at both `audit_cast_contract` (lines 420-455) and `run_script_doctor` (lines 815-854). No fail-soft regressions detected in the writer's other `except` arms (the writer's `except` arms either raise loudly or fall back to documented-safe paths).
- **Audio C7 byte-identical pytest proxy** — assumed to exist at `tests/test_audio_byte_identical.py` per S30 deviation #2 in ROADMAP (the original path `tests/v2/test_audio_byte_identical.py` was renamed). v2 should cite the actual path.
- **Bug Bible regression target 23/1/2xf** — confirmed as the current baseline from S34 close (ROADMAP `## CURRENT WORK -- S34` section).
- **Sprint B (Two-Model Selector) closed status** — confirmed via S30/S32 close entries in ROADMAP. The two-model selector is shipped; widget surface is `creative_writing_model` + `technical_model`. Sprint C plan's "no new widgets" rule (no.4, no.7) is consistent.
- **Sprint G placement** — confirmed as queued after Sprint A (ROADMAP line 114). Sprint G's audit items overlap with Sprint C surface only in the "comment/docstring drift" item; the rest are non-overlapping. The plan v1's claim of "Sprint G has audit items in this surface" is accurate but mild — Sprint G is mostly outside Sprint C's blast radius. Plan v1's "no Sprint G surface fixed prematurely" hard rule (no.10) is the right framing.
- **Plan v1 acceptance table rows 1-5, 14-21** — these don't depend on staleness-audit findings; they're sprint mechanics (canonical pytest, Bug Bible, forbidden sweep, audio C7, brief length, etc.). All survive.
- **Locked design decisions §6.1, §6.3, §6.4, §6.5, §6.6, §6.7** — none of these decisions are contradicted by current code. They remain locked. (§6.6 retire-`meta.ltx_style_brief` is the only one with notable state change: most of the retire happened already in S31 B3. See F-10.)
- **Plan v1's commit structure (7 commits: C0 + C1 + C2 + C3 + C4 + C5 + C-final)** — the shape is fine. The fixes are in the per-commit scopes (C4 shrinks substantially, C2 widens slightly), not in the commit count.

---

## What the round-robin passes should look at (not for v2 to resolve)

These are flagged for the Gemini + ChatGPT passes that follow this Cowork pass. They are NOT being resolved here.

1. **§6.2 inside-writer vs separate-node** — the open round-robin question Jeffrey already scheduled. F-08 adds a save-order constraint but does not resolve.
2. **F-03 model-slot decision for the reflection pass** — given Mistral-Nemo is the audio C7 baseline, where does Gemma-4-E4B (or whatever lightweight model the refinement wants) plug in? Round-robin should weigh: (a) keep Mistral-Nemo on both slots, run reflection on `technical_model`; (b) introduce a dedicated reflection-model slot; (c) accept C7 baseline reset and re-soak.
3. **F-04 §11.4 re-read against BUG-LOCAL-228** — does refinement §11.4 really want a sync barrier, or does it want "the orphan thread doesn't poison the next phase"? If the latter, S31 B4 already shipped it.
4. **VRAM envelope for the reflection pass** — independently of model choice, the worst-case path (composition LLM call → reflection LLM call → repair LLM call) is THREE LLM calls in series on the 14.5 GB ceiling. Refinement §11 needs explicit measurement evidence that this fits. Round-robin reviewers without code access can press on whether the refinement doc's evidence is empirical or assumed.

---

## Summary

- 4 HIGH severity findings (F-01 through F-04) — all in C4 pre-flight cleanbreak 3. C4 needs substantial rewrite or near-deletion.
- 5 MEDIUM severity findings (F-05 through F-09) — scope widening in C2, line-ref fixes in C3, handoff contract in Sprint A/C boundary, save-order constraint on §6.2, failure-mode precedent pointer in C5.
- 5 LOW severity findings (F-10 through F-14) — clerical / docs / forensic.
- ~13 items checked and unchanged — sprint mechanics, locked design decisions, and most of C2/C3/C5's actual code surface.

v2 plan is in `docs/2026-05-15-sprint-c-story-brief-v2-plan-v2.md`. It reflects the proposed v2 changes from F-01..F-14. Round-robin passes will follow.
