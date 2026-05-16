# Sprint C — `meta.story_brief` — FINAL consolidated plan (READY-TO-CUT)

> **Single deliverable.** All decisions consolidated at the top. All load-bearing code cited inline (§A.1-§A.9). Send-ready for one final review. No attachments needed.

## Status

- Phase: execution
- Current commit: C3
- Branch: sprint-c-story-brief-v2
- Cut from: s34-p0-p1-hotfix @ f758f02
**Sequencing:** S34 closed → **Sprint C (this plan, 17 commits)** → Sprint A (downstream verification, queued) → Sprint G (comprehensive bug sweep, queued).
**Loop per commit (no operator gates between commits — chain runs through to C-final):** review → code → wire → pytest → regression → commit. No ComfyUI runtime gates. Pytest-only acceptance.
**Commit sizing directive (operator 2026-05-15):** each commit is sized to land cleanly in a single review-code-wire-pytest-regression-commit loop. If a commit's risk surface or test count would exceed a single loop's safe boundary, split it. C2 is split into C2a (visual layer) + C2b (orchestrator layer + `_STYLE_WORLD_BLOCK`) on this basis.
**Autonomous coding rule:** autonomous coding may proceed only while commit gates remain green AND the working tree is clean. Any unmerged index state, unexpected workflow JSON drift, audio C7 regression against the prevailing baseline, or call-site ambiguity STOPS the sprint. The dirty-repo / poisoned-index hard stop at C0a is non-negotiable.
**Legacy retirement ordering:** ALL legacy removals (era literals at orchestrator + visual layers, `_GENRE_BY_STYLE`, `meta.ltx_style_brief`) land in commits C2a–C3b, BEFORE any new `meta.story_brief` code is built. New system lands on a cleared bench.
**Audio boundary:** Sprint C DOES touch episode audio — at C5g only. MusicGen integration is IN scope per operator directive (overrides original L-4). Audio C7 baseline holds against the existing fixture from C2a → C5f, intentionally resets at C5g, and holds against the NEW baseline from C5g → C-final. C5g includes an isolation test (absent-brief path → matches pre-C5g forensic b3sum) to prove the audio shift is caused EXCLUSIVELY by the mood prefix, not a smuggled regression.
**No-change-logs rule (operator directive):** existing runtime log strings and `meta.*` attribute names are stable. Any new log lines added at C5c–C5g for `story_brief_status` observability follow the existing log format conventions; no existing log line is modified.

---

# §1. Master decision ledger (every decision, one place)

All decisions below are **RESOLVED**. There are zero open questions. The plan is ready for one final review.

## §1.1 Locked design decisions (L-series — original v3 locks)

| # | Decision | Rationale |
|---|---|---|
| L-1 | Default LLM stays `mistralai/Mistral-Nemo-Instruct-2407`. NO `DEFAULT_LLM` change. | Catalog notes mark this entry the audio C7 byte-identical baseline (§A.7). Flipping it forces a baseline reset + soak (out of Sprint C scope). |
| L-2 | Reflection pass runs on the writer's `technical_model` slot. Defaults to Mistral-Nemo; differs only when operator explicitly picks a two-slot config (e.g. technical = gemma-4-E4B-it). | Two-Model Selector is the existing surface (no new widget). Audio C7 unaffected because reflection writes to `meta`, not `lines`. |
| L-3 | `_run_with_timeout` stays non-blocking. Refinement §3.6 + §11.4 "hard sync barrier" wording is INCORRECT and is amended at C0b. BUG-LOCAL-228 (S31 B4 `a4fe67a`) is the contract. | The refinement spec mandates the behavior the bug was filed against. See §A.8 for code evidence. |
| L-4 **OVERRIDDEN 2026-05-15** | **MusicGen integration is IN Sprint C, at C5g.** Operator directive: brief flavor must reach MusicGen before Sprint C closes; deferral rejected. `get_story_brief_music_mood` is wired into `nodes/musicgen_theme.py` at C5g. The audio C7 byte-identical pytest proxy holds against the existing fixture for C2a → C5f, then intentionally resets at C5g; a new canary fixture is captured and held from C5g → C-final. Hard rule 1 (§4) is amended accordingly. E-09 no-touch gate is amended to allow `nodes/musicgen_theme.py` and its tests to change AT C5g and only at C5g. | Original deferral rationale (`OTR_MusicGenTheme` → `OTR_EpisodeAssembler` → `episode_audio`, §A.6) remains correct on the mechanics. Operator accepts the baseline-reset cost in exchange for completeness: FLUX, LTX, HuMo, AND MusicGen all consume the brief — story flavor reaches every downstream artifact in this sprint, not the next. |
| L-5 | Helper set matches refinement §5 verbatim: `get_story_brief_full`, `get_story_brief_ltx`, `get_story_brief_lighting`, `get_story_brief_music_mood`, `get_story_brief_status`. | Refinement §5 canonical. |
| L-6 | Failure-mode pattern matches the S34 B1 fail-loud sentinel from `_otr_ledger_reviewer.run_script_doctor`. Three explicit `except` arms (Exception / JSONDecodeError / ValidationError); log.warning + return sentinel; no bare `return`, no silent swallow. | §A.5 code precedent. |
| L-7 | C5 splits into **C5a1 + C5a2 + C5b + C5c + C5d + C5e + C5f + C5g** — eight commits. C5a originally one (per E-08); MusicGen wiring (per L-4 override) now occupies C5g; original C5g content (`meta.ltx_style_brief` retirement) pulled forward to C3b per operator directive on legacy ordering. | Rollback isolation; legacy cleared before new code lands. |
| L-8 | Storage schema = 8 meta keys per refinement §4 (`story_brief`, `story_brief_status`, `story_brief_error`, `story_brief_model`, `story_brief_prompt_version`, `story_brief_source`, `story_brief_char_count`, `story_brief_terms.{setting,lighting,atmosphere}`). | Refinement §4. |
| L-9 | LTX prompt test checks motion-first ordering + drop-past-char-140 + char-cap. Not just char-cap. | Refinement §6.1. |
| L-10 | Canonical design surface = research inventory + design refinements. Go-forward plan (2026-05-13) is historical input only. Phantom `2026-05-12-story-brief-v2-problem-statement.md` never existed in git history; references removed 2026-05-15. | Verified via `git log --all --diff-filter={A,D}` — both empty. |

## §1.2 Pre-build edit pass (E-series — Jeffrey 2026-05-15)

| # | Severity | Edit | Status |
|---|---|---|---|
| E-01 | P0 | Q1 LOCKED to **K.5.5**. L.5 rejected (returned `script_json` socket would not carry the brief). | Locked. See §3. |
| E-02 | P0 | New test `test_story_brief_present_in_returned_script_json` in C5a2 — proves brief reaches the output socket, not just disk. | Specified in C5a2. |
| E-03 | P0 | New test `test_reflection_uses_technical_fn_not_creative_fn` in C5a1 — behavioral spy, not source grep. | Specified in C5a1. |
| E-04 **SUPERSEDED by E-12** | P0 | Originally: `get_story_brief_music_mood` implemented-but-not-wired (not stubbed). Filters atmosphere terms against declared vocabulary. NO MusicGen import. **Now (per E-12):** helper still ships per refinement §5 as a pure function with no MusicGen import (direction of dependency is helper → unaware-of-consumer), AND `nodes/musicgen_theme.py` imports the helper at C5g and wires it. | Specified in C5b (helper) + C5g (wiring). |
| E-05 | P1 | New C3 test `test_hud_and_treatment_render_without_visual_plan_genre` — old-ledger backward-compat. | Specified in C3. |
| E-06 | P1 | Per-consumer meta-threading canary tests using unique token `zebra_lantern_atmosphere_731`. Behavioral, not source grep. | Specified in C5c, C5d, C5e, C5f. |
| E-07 | P1 | Per-consumer render-log visibility for `story_brief_status=failed` (FLUX, LTX, HuMo). Observable, not fatal. | Specified in C5c, C5e, C5f. |
| E-08 | P1 | C5a split into C5a1 (pure module) + C5a2 (writer wiring). | Implemented in §6. |
| E-09 **AMENDED by E-12** | P2 | Originally: C-final MusicGen no-touch gate forbade any change to `nodes/musicgen_theme.py` or `tests/test_musicgen_*.py`. **Now (per E-12):** the gate becomes a one-commit-exception check. Exactly ONE commit in the chain (the C5g MusicGen wiring) is allowed to touch these files; any other commit touching them is a violation. Enforced via `git log -- nodes/musicgen_theme.py tests/test_musicgen_*.py` with a single-result + `C5g:` subject-prefix assertion. | Specified in C-final commit gate. |
| E-10 | P2 | "Full autonomous coding" framing replaced with explicit stop-conditions. Dirty-repo / poisoned-index becomes a HARD STOP at C0a. | Applied at header + C0a. |
| E-11 | P2 | Acceptance row: returned `script_json` socket AND saved-on-disk ledger both contain identical `meta.story_brief*` fields. | Acceptance row 25 (renumbered v2→v3 — was row 21 in pre-v2 numbering; see E-25 repair). |
| E-12 | P0 | **MusicGen wiring IN sprint at C5g.** L-4 override. `get_story_brief_music_mood` is wired into `nodes/musicgen_theme.py`. Helper no longer ships "implemented-but-not-wired"; it ships wired. R-04 disposition flips from RESOLVED-as-deferred to SUPERSEDED. E-09 no-touch gate is amended (one-commit exception at C5g). Audio C7 baseline resets at C5g. | Specified at C5g; affects L-4, R-04, E-09, §4 hard rule 1. |
| E-13 | P0 | **C2 scope expanded to all era-locked literals across orchestrator + visual layers** — not just the 4 Python + 2 JSON visual sites. Adds three orchestrator targets: rerank-prompt era anchor (`story_orchestrator.py:1596`), dramaturg-preamble era anchor (`story_orchestrator.py:3003-3004`), and the OMNI-RETRO 5-pillar block (`story_orchestrator.py:3162-3179`). The 5-pillar block is replaced with a style-driven dynamic world block keyed off `meta.style`, so each story carries flavor unique to its chosen style rather than 5-pillar omni-retro on every gen. **PARTIALLY SUPERSEDED at C2b by the RR-A3+RR-B6 path-A audit (see §1.4 row):** the dramaturg-preamble + 5-pillar literals lived inside orphan constants (`SCAFFOLDING_PREAMBLE` + `SCRIPT_SYSTEM_PROMPT`) with no live consumers — so the era-literal removal happens by DELETING the orphan constants, not by refactoring them into a style-driven block. The rerank-prompt fix at line 1596 still ships as originally planned; the `_STYLE_WORLD_BLOCK` machinery is skipped (no consumer = no place to interpolate). | Specified at C2; refined at C2b per Path A discovery. |
| E-14 | P0 | **All legacy retirements pulled forward.** `meta.ltx_style_brief` retirement moves from C5g to **C3b**, executed BEFORE any new `meta.story_brief` code is built. Order is: cleanbreak (C2a/C2b era literals + C3 `_GENRE_BY_STYLE` + C3b `meta.ltx_style_brief`) → envelope lock-in (C4) → new system (C5a1+). | Specified at C3b; affects §6 commit order. |
| E-15 | P0 | **C5a2 dual-slot OOM regression guard (RR-A1, revised at C1 audit).** Original specification called for an explicit `evict_model(creative_model)` call before reflection. C1 audit found the loader is implicitly single-slot: `request_slot(model_id)` already calls `unload_llm()` to evict any prior resident model before loading the next. Peak transient VRAM during the swap is `max(creative_size, technical_size)`, not their sum. Explicit pre-eviction is redundant. The OOM scenario cannot occur on this loader architecture. C5a2 ships regression tests that prove the no-OOM property without adding an explicit eviction call. | Specified at C5a2; tests below. |
| E-16 | P0 | **C5g audio isolation test (RR-A2).** Blessing a new C7 baseline proves determinism but NOT that the audio shift was caused exclusively by the mood prefix. C5g adds `test_c5g_audio_matches_legacy_when_brief_absent`: force `meta.story_brief_status="absent"` (the brief-helper status-gate fall-through), run the full C5g pipeline, assert byte-identical output against the pre-C5g forensic b3sum (`tests/fixtures/audio_c7_baseline_pre_c5g.wav.b3sum`). Closes the smuggled-regression class. | Specified at C5g. |
| E-17 | P1 | **Scoped try/except blocks (RR-B3).** C5a1 reflection function uses three NARROW try/except blocks (not one wrapping the whole function): one around `technical_fn(...)` catching broad Exception; one around `json.loads(...)` catching `json.JSONDecodeError`; one around schema validation catching `ValidationError`. Matches §A.5 `run_script_doctor` precedent. Prevents specific failure classes from being swallowed into the generic arm. | Specified at C5a1; new test asserts AST structure. |
| E-18 | P1 | **Repair temperature clamp (RR-B5).** `repair_temperature = min(reflection_temperature + 0.15, 0.55)`. Keeps repair temperature inside the declared 0.35-0.55 effective range even if a future operator sets reflection_temperature above 0.4. | Specified at C5a1; new test `test_repair_temperature_clamped_to_055`. |
| E-19 | P1 | **Workflow JSON string-based replacement (RR-B6).** C2a replaces era-literal widget defaults via exact-string recursive search/replace across `workflows/otr_scifi_16gb_full.json` — NOT by line number. Line numbers drift on any unrelated workflow edit; string targeting is robust. | Specified at C2a. |
| E-20 | P1 | **C-final gate rename-immune (RR-A4, RR-B7 partial).** C-final MusicGen one-commit-exception gate uses `git diff --name-only s34-p0-p1-hotfix...HEAD \| grep -i musicgen` to catch any musicgen-related file path including renames or relocations. Plus a semantic-gate visual-inspection note: outside C5g, no commit should change MusicGen prompt text, model args, duration, cache key, or audio routing — even via shared helpers, workflow widget edits, or test cache-key changes. | Specified at C-final. |
| E-21 | P2 | **C5a1/C5a2 test reassignments (RR-B1, RR-B2).** Move `test_reflection_uses_technical_fn_not_creative_fn` from C5a1 to C5a2 (the closures only co-exist at the writer-wiring layer). Replace in C5a1 with a signature-only assertion: `test_reflection_entrypoint_signature_accepts_only_technical_fn` — AST walk confirms `run_story_brief_reflection` has no `creative_fn` parameter. Move `test_reflection_does_not_thrash_llm_cache` from C5a1 to C5a2 with a real spy on `_otr_model_loader.LLM_CACHE` against the resolved `technical_model` cache entry (the pure module's stub-`technical_fn` made the C5a1 version vacuous). | Specified at C5a1 + C5a2. |
| E-22 | P2 | **C5a2 module-level import (RR-B4).** `from ._otr_story_brief import run_story_brief_reflection` moves to module-level top of `nodes/OTR_LedgerScriptWriter.py`. Hot-path import dropped. If a circular import surfaces, document at C5a2 commit time. | Specified at C5a2. |
| E-23 | P2 | **C2b token-stability test (RR-A3).** Removing the 5-pillar structural constraints from the worldbuilding prompt creates a vacuum that local models can fill with hallucinated expansion, threatening the 8192 hard context limit. C2b adds `test_script_token_length_stability_with_dynamic_world_block`: generate a fixture script with the legacy 5-pillar block; generate with the new `_STYLE_WORLD_BLOCK`; assert resulting script-text token count does not expand by more than +15%. `_STYLE_WORLD_BLOCK` entries preserve dramaturg discipline ([ENV:]/[SFX:] tag use, sound-first framing) — only the era/pillar lock is removed. | Specified at C2b. |
| E-24 | P2 | **C1 file-path verification (RR-B8).** Cowork audit row added at C1: confirm actual FLUX env/bookend implementation file path. If `visual/batch_flux_env_render.py` does not exist, C5c retargets to `visual/batch_flux_render.py`. Plan claims must match repo state before C5c codes. | Specified at C1. |
| E-25 | P3 | **Acceptance-row reference repair (RR-B9).** E-11 reference to "Acceptance row 21" repaired to the current row number (post-v2/v3 renumbering: row 25). | Inline §1.2 E-11. |
| E-26 | P3 | **C3 test rename (RR-B11).** `test_hud_and_treatment_render_without_visual_plan_genre` → `test_hud_and_treatment_render_when_visual_plan_genre_absent`. Cosmetic; avoids language that could be misread as legacy back-compat / shim allowance. | Specified at C3. |
| E-27 | P3 | **C-final QA wording strengthened (RR-B12).** Runtime-status line now explicitly enumerates visual outputs as not proven: "no ComfyUI Desktop run; no FLUX/LTX/HuMo visual render quality verified in Sprint C." | Specified at C-final. |
| E-28 | P2 | **`music_mood` helper caller-count contract (RR-B10).** Mechanical version of the C-final semantic-gate visual inspection: assert helper has exactly the expected number of production callers at each commit boundary. At C5b: 0 callers (defined, not wired). At C5g: 1 caller (`nodes/musicgen_theme.py`). Covers the C5b→C5f gap where a rogue wiring would otherwise slip through; pins the C5g wiring contract mechanically. | Specified at C5b + C5g. |

## §1.3 Round-robin-2 triage (R-series — 2026-05-15)

Seven findings reviewed against code + locked decisions. Three accepted, two pushed back, one folded into C0b spec amendment, one resolved per existing E-04 lock.

| # | Severity claimed | Finding | Disposition | Lands at |
|---|---|---|---|---|
| R-01 | HIGH | Mistral-Nemo 24GB blows 14.5GB ceiling across three sequential calls; force-override reflection to Gemma-4-E4B-it | **PUSH BACK on framing; accept cache-stability test.** Reviewer conflates download size with resident size. Catalog (`_otr_model_catalog.py:553-555`) documents 24GB disk → ~12GB resident post 8-bit/NF4 quantization. Three sequential calls reuse the SAME cached model, not three loads. Composition pass already runs Mistral-Nemo in the 14.5GB envelope every sprint without OOM. Force-override to Gemma-4-E4B-it would require BOTH Mistral-Nemo (creative slot) AND Gemma-4-E4B-it (technical slot) cached concurrently = 12GB + 5GB = OVER ceiling — opposite of the reviewer's fix. **Resolution:** L-1 + L-2 stand. New test added to C5a1 (`test_reflection_does_not_thrash_llm_cache`) that mocks three sequential reflection calls and asserts `LLM_CACHE` registers zero load events between calls — guards against accidental thrashing. | C5a1 |
| R-02 | HIGH | Flip Q1 from K.5.5 to L.5: claims K.5.5 wastes CPU re-parsing `script_text` and loses script on crash | **PUSH BACK HARD.** Both arguments fictional. (a) Refinement §2 explicitly designs the input builder to read `led.data["lines"]` directly with caps — it does NOT read `script_text`. No re-parse. (b) Reflection writes to `meta` only. If reflection raises, `led.data["lines"]` is byte-identical pre/post and section L assembles `script_text` from those untouched lines. Script is never lost. (c) Reviewer's flip reopens the E-01 silent-failure class: L.5 lands the brief AFTER `script_json = json.dumps(led.data, ...)` so the returned socket carries a pre-brief snapshot while disk carries post-brief. Two sources of truth disagree. **Resolution:** Q1 K.5.5 stays locked. Reviewer overruled. | (unchanged) |
| R-03 | MED | Replace `test_reflection_writes_only_meta_not_lines` with deep-dict comparison | **ACCEPT.** A nuked `cast` or `news_seed` would slip past a lines-only check. Replaced with `test_reflection_does_not_mutate_core_ledger_keys`: `copy.deepcopy(led.data)` pre-reflection; run; assert every top-level key EXCEPT `meta` is byte-identical between pre and post snapshots. | C5a2 |
| R-04 | MED | Kill `get_story_brief_music_mood` entirely (dead code in current sprint) | **SUPERSEDED by E-12 (operator directive 2026-05-15).** Helper is no longer dead code: it is wired into `nodes/musicgen_theme.py` at C5g. The 16-entry MusicGen mood vocab and pure-function shape (refinement §6.3) hold. Audio C7 baseline resets at C5g per E-12. | C5g |
| R-05 | MED | LTX char-140 test is structural proxy, not motion-fidelity proof | **ACCEPT.** Char-counting proves prompt formatting, not LTX rendering. Documentation note added to C5e: empirical LTX motion fidelity verification is Sprint A scope per ROADMAP. | C5e |
| R-06 | LOW | Repair pass at same temperature = same hallucination | **ACCEPT partially.** Reviewer's "same prompt" claim is wrong (refinement §3.5 specifies a DIFFERENT prompt for repair). But the same-temperature point holds — at temp 0.2-0.4, retries are nearly deterministic. **Resolution:** C0b spec amendment expanded to cover §3.5 as well: repair temperature is `reflection_temperature + 0.15` (so 0.35-0.55 range); explicit `CRITICAL: <reason>` prefix prepended to the repair prompt per the rejection-reasons list. Breaks the deterministic-retry loop. | C0b + C5a1 |
| R-07 | LOW | Add ledger migration shim mapping old `meta.ltx_style_brief` → new `meta.story_brief` | **PUSH BACK.** Violates hard rule 2 + locked decision §6.6 ("retire `meta.ltx_style_brief` — no alias, no shim"). The reviewer's "seamlessly map old key" IS a shim. Old ledgers handled by `get_story_brief_status(meta) → 'absent'` returning `""` to consumers; consumers fall through to legacy fallback gracefully. Re-rendering an old episode re-runs the reflection pass on the new code path; brief is generated fresh. Migration is implicit by re-running, not by shimming. **Resolution:** drop the finding. | (unchanged) |

## §1.4 Round-robin-3 & round-robin-4 synthesis (RR-series — 2026-05-15)

Two reviewers, 17 findings total. 12 accepted (lifted to E-15…E-27 above). 2 rejected with named rationale. 3 already covered by v2 edits and acknowledged. Synthesis below; concrete code/test changes live in §6.

| # | Reviewer | Severity | Finding (compressed) | Disposition | Lands at |
|---|---|---|---|---|---|
| RR-A1 | A | HIGH | C5a2 dual-slot OOM | ACCEPT → E-15 | C5a2 |
| RR-A2 | A | HIGH | C5g audio reset blesses regression | ACCEPT → E-16 | C5g |
| RR-A3 | A | MED | C2b worldbuilding-vacuum token expansion | **SUPERSEDED by C2b path-A discovery (see RR-A3+RR-B6 architecture row below).** Token-stability guard becomes moot once the orphan-prompt deletion path is taken — no runtime worldbuilding string to stabilize. | C2b |
| RR-A4 | A | MED | git log gate bypass on rename | ACCEPT → E-20 | C-final |
| RR-A5 | A | LOW | C5a1/C5a2/C5b ordering race | **REJECT.** Schema lives in C5a1 module. C5a2's tests JSON-parse and dict-compare; no helper needed. Helpers (C5b) serve downstream consumers (C5c-C5g), not the writer-wiring layer. Order C5a1 → C5a2 → C5b → C5c–C5g is correct. | (unchanged) |
| RR-B1 | B | HIGH | C5a1 technical-slot test impossible | ACCEPT → E-21 | C5a1 + C5a2 |
| RR-B2 | B | HIGH | C5a1 LLM-cache stability test vacuous | ACCEPT → E-21 | C5a1 + C5a2 |
| RR-B3 | B | HIGH | Scoped try/except blocks needed | ACCEPT → E-17 | C5a1 |
| RR-B4 | B | MED | C5a2 hot-path import sloppy | ACCEPT → E-22 | C5a2 |
| RR-B5 | B | MED | Repair temperature can exceed range | ACCEPT → E-18 | C5a1 |
| RR-B6 | B | MED | Workflow JSON line-number edits brittle | ACCEPT → E-19 | C2a |
| RR-B7 | B | MED | MusicGen no-touch gate too narrow | PARTIAL ACCEPT → E-20 (semantic-gate visual-inspection note added; full automation of "no indirect MusicGen behavior change" is infeasible mechanically) | C-final |
| RR-B8 | B | MED | C5c file-path verification | ACCEPT → E-24 | C1 |
| RR-B9 | B | MED | E-11 acceptance-row reference off | ACCEPT → E-25 | §1.2 E-11 |
| RR-B10 | B | LOW | `get_story_brief_music_mood` no-production-callers test | **PARTIAL ACCEPT → E-28.** Caller-count test is a stronger mechanical version of the C-final semantic-gate visual inspection. At C5b: assert 0 production callers (helper defined, not yet wired). At C5g: assert exactly 1 production caller (`nodes/musicgen_theme.py`). Covers the C5b→C5f gap where a rogue wiring would otherwise slip through; pins the wiring contract at C5g. | C5b + C5g |
| RR-B11 | B | LOW | C3 test name suggests shim-allowance | ACCEPT → E-26 | C3 |
| RR-B12 | B | LOW | Final QA visual-runtime wording | ACCEPT → E-27 | C-final |
| RR-A3 + RR-B6 architecture | A | HIGH | C2b worldbuilding architecture | **PATH A confirmed by C2b 8-search audit (2026-05-15):** `SCRIPT_SYSTEM_PROMPT` + `SCAFFOLDING_PREAMBLE` are orphan constants from the `eec4718` LPL extraction sprint — no live consumers across `nodes/`, `visual/`, `scripts/`, or `tests/`. The new `OTR_LedgerScriptWriter` pipeline composes prompts from per-phase modules (`_otr_outline._SYSTEM_PROMPT`, `_otr_line_composer._SYSTEM_PROMPT` + `_POLISH_*`, `_otr_ledger_reviewer._AUDITOR_SYSTEM_PROMPT` + `_DOCTOR_SYSTEM_PROMPT`, `_otr_period_prompts.OTR_PERIOD_SYSTEM_PROMPT`). The OMNI-RETRO 5-pillar block at lines 3160-3179 was therefore forensic, not runtime. C2b reduced to (a) rerank-prompt fix at line 1596 (only live era-literal site), (b) DELETE orphan SCRIPT_SYSTEM_PROMPT + SCAFFOLDING_PREAMBLE per no-legacy rule, (c) skip `_STYLE_WORLD_BLOCK` machinery entirely (no live consumer = no place to interpolate). Token-stability test (E-23 / RR-A3) skipped — no runtime path to stabilize. `test_script_system_prompt_interpolates_style_world_block` skipped — no constant to interpolate. `test_citation_rule_present` in `tests/test_core.py` deleted — pinned substring inside the now-deleted SCRIPT_SYSTEM_PROMPT, no live replacement exists. Broader orphan-constant sweep of `nodes/story_orchestrator.py` (3000+ lines, gutted across LPL / S31 B3 / S34) deferred to Sprint G per the no-scope-creep rule; each candidate gets its own 8-search audit before deletion. | C2b |

**Operator sizing directive applied 2026-05-15:** C2 split into C2a (visual layer) + C2b (orchestrator layer + `_STYLE_WORLD_BLOCK`) to keep each commit inside a single review-code-wire-pytest-regression-commit loop. Total commit count: 16 → 17. C2b further reduced at execution time after Path A architectural discovery (RR-A3+RR-B6 row above). No other commits split; they are already sized at ≤0.75 d each. Decreasing further would over-fragment rollback boundaries.

## §1.5 Open questions

**None.** Every question is resolved. The plan is ready for final cut.

---

# §2. Pre-build preconditions (HARD STOPS — branch does NOT cut until all green)

The branch does NOT cut until every row below is satisfied. C0a will not land otherwise.

| # | Precondition | Verified by |
|---|---|---|
| 1 | Q1 locked to K.5.5 | §1.2 E-01, §1.3 R-02, §3 |
| 2 | C5a split into C5a1 + C5a2 | §1.2 E-08, §6 commit structure |
| 3 | Returned-`script_json` test (`test_story_brief_present_in_returned_script_json`) specified in C5a2 | §6 C5a2 pytest table |
| 4 | Technical-slot behavioral-spy test (`test_reflection_uses_technical_fn_not_creative_fn`) specified in C5a1 | §6 C5a1 pytest table |
| 5 | LLM-cache stability test (`test_reflection_does_not_thrash_llm_cache`) specified in C5a1 — R-01 mitigation | §6 C5a1 pytest table |
| 6 | Old-ledger no-genre fallback test (`test_hud_and_treatment_render_without_visual_plan_genre`) specified in C3 | §6 C3 pytest table |
| 7 | Per-consumer meta-threading canary tests with unique token `zebra_lantern_atmosphere_731` specified in C5c-C5f AND C5g | §6 C5c-C5g pytest tables |
| 8 | C-final MusicGen no-touch gate amended: one-commit exception at C5g | §6 C-final commit gate, E-12 |
| 9 | Deep-dict ledger-mutation test (`test_reflection_does_not_mutate_core_ledger_keys`) specified in C5a2 — R-03 | §6 C5a2 pytest table |
| 10 | Repair-pass temperature bump + critical prefix specified in C0b amendment AND C5a1 spec — R-06 | §6 C0b + C5a1 |
| 11 | C2 split into C2a (visual layer, string-based JSON replace) + C2b (orchestrator + `_STYLE_WORLD_BLOCK`, sound-first discipline preserved) per E-13/E-19/E-23 | §6 C2a + C2b |
| 12 | C3b: `meta.ltx_style_brief` retirement pulled forward (was C5g) | §6 C3b, E-14 |
| 13 | C5g: MusicGen wiring + new audio C7 baseline capture + canary-fixture refresh specified | §6 C5g, E-12 |
| 14 | C5a2 dual-slot OOM regression tests specified (RR-A1 — C1 audit revised to regression-test approach after loader-architecture discovery) | §6 C5a2, E-15 |
| 15 | C5g absent-brief isolation test specified (RR-A2) | §6 C5g, E-16 |
| 16 | C5a1 scoped try/except spec + repair-temperature clamp specified (RR-B3, RR-B5) | §6 C5a1, E-17 + E-18 |
| 17 | C5a1/C5a2 test reassignments specified (technical-slot test → C5a2; LLM-cache spy → C5a2) (RR-B1, RR-B2) | §6 C5a1 + C5a2, E-21 |
| 18 | C1 audit retargeted C5c to `visual/batch_flux_render.py` (E-24 / RR-B8 contingency activated — `batch_flux_env_render.py` does not exist at HEAD) | §6 C1, E-24 |
| 19 | C-final gate rename-immune + semantic-gate visual-inspection note (RR-A4, RR-B7) | §6 C-final, E-20 |
| 20 | Windows-side `git status --short` empty; `git branch --show-current` reports `s34-p0-p1-hotfix`; `git rev-parse HEAD` reports `f758f02`; no `AU "\a\al}"` or any other unmerged path | OPERATOR action on Windows checkout before C0a queues |

Row 20 is an operator action. Cowork cannot verify the Windows-side checkout. The sandbox view at v3 authorship showed a poisoned-index pattern (`AD ./`, `AU "\a\al}"`, `AM` across nearly every tracked file). MUST be cleared on Windows before C0a runs.

---

# §3. Q1 — Reflection-pass call-site (LOCKED: K.5.5)

**Decision: K.5.5 is locked.** Reflection runs after section K.5 stamps `meta.visual_plan` and `meta.style`, and BEFORE section L assembles `script_text` / `script_json`. L.5 is rejected.

**Insertion site:** immediately after `nodes/OTR_LedgerScriptWriter.py:2803` (`meta["style"] = resolved["style"]`), before section L begins at line 2805 (`script_text = _PL.assemble_script_text_from_ledger(led.data)`).

**Why L.5 was rejected:** L.5 lands the brief AFTER section L assembles `script_json = json.dumps(led.data, indent=2, ensure_ascii=False)`. The saved-on-disk ledger (section M `led.save()`) would contain `meta.story_brief`, but the returned `script_json` output socket would carry the pre-brief snapshot. Downstream nodes consuming `script_json` (FreezeCascade, any node reading the socket directly) would see a meta dict without `story_brief`. Silent downstream failure class. **K.5.5 places the brief BEFORE `script_json` is assembled — both consumers (saved ledger AND returned socket) see the same data.**

R-02 reviewer flip rejected: §1.3 documents why. Input builder reads `led.data["lines"]`, NOT `script_text`. Script lines stay byte-identical through reflection. No "lost script" risk.

**Enforcement:** C5a2 pytest table includes `test_story_brief_persists_in_saved_ledger` (disk), `test_story_brief_present_in_returned_script_json` (socket), `test_returned_and_saved_brief_match` (both), and `test_call_site_is_k55_not_l5` (AST walk locking insertion site). All must hold at every commit boundary post-C5a2.

---

# §4. Hard rules (continuity from S30-S34)

1. **Audio C7 byte-identical pytest proxy** holds on default-config happy path at every commit boundary. Reflection writes only to `meta` (C5a1 → C5a2). FLUX/LTX/HuMo consumers do NOT touch audio (C5c → C5f). C7 holds against the existing fixture from C2a → C5f. **At C5g (MusicGen wiring per E-12), the baseline intentionally resets**: MusicGen prompt construction now reads `get_story_brief_music_mood`, which changes `opening_audio`/`closing_audio` (§A.6). A new canary fixture is captured at C5g; from C5g → C-final, C7 holds against the NEW baseline. The absent-brief isolation test (E-16) confirms the shift is caused exclusively by the mood prefix, not a smuggled regression. Any other commit in the chain that causes C7 to drift = STOP, revert.
2. **No legacy back-compat reintroduced.** `meta.ltx_style_brief` retires cleanly at C3b; no alias, no shim. R-07 reviewer's migration-shim finding is overruled on this basis. Era literals retired at C2a (visual layer: 4 Python + 2 JSON sites) and C2b (orchestrator layer: 3 prompt sites + `_STYLE_WORLD_BLOCK`) — see E-13.
3. **No new generate or lifecycle surfaces.** Reflection uses existing creative/technical slots via `make_generate_fn`.
4. **No widgets.** `meta.story_brief` is always-on. No user-facing toggle.
5. **Bug Bible regression** 23/1/2xf at every commit boundary.
6. **Forbidden-pattern sweep** stays at 0 runtime hits. New markers across C2a (visual era literals: `1940s`, `1980s broadcast`) + C2b (orchestrator era literals: `1950s Americana`, `golden-age radio`, `Omni-Retro`, `Orson Welles`, `Norman Corwin`, `Lucille Fletcher`) + C3 (`_GENRE_BY_STYLE`) + C3b (`meta.ltx_style_brief`) + C5a1 (reflection-fail-soft) + C5a2 (`meta.story_brief == None` init). All markers run through `docs/_s28_forbidden_sweep.py` (tokenize-classified docstring/comment suppression).
7. **14.5 GB VRAM ceiling.** Worst-case path is composition → reflection → repair (three INFERENCE calls on the same cached technical-slot model, not three loads). R-01 cache-stability test in C5a1 guards against thrashing.
8. **Reflection prompt body ≤250 tokens.** Schema sized accordingly.
9. **All LLM calls tagged creative or technical.** Reflection is **technical** (structured JSON, not narrative). Tagged at the call site per CLAUDE.md rule 6.
10. **Wire it or do not ship it.** Code change is not done until workflow JSON is re-wired AND a drift-guard test pins the new surface.

---

# §5. Sprint C standing directives

- UTF-8, no BOM.
- No profanity. No "dummy" — use "placeholder", "stub", or descriptive names.
- Pytest-only acceptance. No ComfyUI Desktop boot. **No operator gates between commits — the chain runs through to C-final.**
- Push via Desktop Commander cmd shell; never PowerShell for git.
- Bug Bible regression runs automatically after every code change.
- **Per-commit loop is fixed (operator directive):** review → code → wire → pytest → regression → commit. Each commit closes the loop before the next opens. No mid-commit pivots.
- **No-change-logs rule (operator directive):** existing runtime log strings stay byte-stable. Existing `meta.*` attribute names stay byte-stable. New log lines added at C5a1 (reflection failure sentinels), C5a2 (eviction notice), C5c-C5f (`story_brief_status` observability), and C5g (mood-prefix status) follow the same format conventions as their neighboring log lines; no surrounding existing line is modified.
- **Commit sizing rule (operator directive):** if a commit would exceed a single review-code-wire-pytest-regression-commit loop's safe boundary, split it. C2 was split into C2a + C2b on this basis at v3 cut time. All other commits sit at ≤0.75 d each; further splitting would over-fragment rollback boundaries.

---

# §A. Code surface reference (cited verbatim — reviewer needs no repo)

## §A.1 — `_GENRE_BY_STYLE` table (C3 target)

```python
# nodes/OTR_LedgerScriptWriter.py, lines 254-265
_GENRE_BY_STYLE: dict[str, str] = {
    "closed_room_suspense":       "thriller audio drama",
    "detective_case_file":        "detective audio drama",
    "pulp_serial_cliffhanger":    "pulp serial audio drama",
    "mission_control_procedural": "procedural audio drama",
    "deep_space_distress_call":   "sci-fi audio drama",
    "noir_interrogation":         "noir audio drama",
    "small_town_uncanny":         "uncanny audio drama",
    "radio_newsroom_emergency":   "newsroom audio drama",
    "haunted_broadcast_signal":   "horror audio drama",
    "laboratory_containment":     "containment audio drama",
}
```

Plus `_resolve_genre` at lines 268-293, `_preview_genre` at lines 296-309.

## §A.2 — `meta.visual_plan.genre` stamp + section K.5 (C3 target + Q1 call site)

```python
# nodes/OTR_LedgerScriptWriter.py, lines 2785-2803 (section K.5)
_cast_rows = led.data.get("cast") or []
_visual_chars = {}
for _row in _cast_rows:
    if not isinstance(_row, dict): continue
    _name = _row.get("name")
    if not _name: continue
    _desc = (_row.get("character_description") or "").strip()
    _visual_chars[_name] = {"portrait_prompt": _desc}
meta["visual_plan"] = {
    "characters": _visual_chars,
    "scenes":     [],
    "style":      resolved["style"],
    "genre":      _resolve_genre(resolved["style"]),   # DELETE in C3
}
meta["style"] = resolved["style"]
# K.5.5 reflection call inserted HERE in C5a2 (after line 2803).

# section L (lines 2805-2824) -- assemble return values
script_text = _PL.assemble_script_text_from_ledger(led.data)
script_json = json.dumps(led.data, indent=2, ensure_ascii=False)
news_json   = _build_news_payload(outline, resolved["news_seed"], resolved["seed_source"])
# ... word counts ...

# section M (lines 2826-2845) -- save + return
saved_path = led.save()
log.info(...)
return (script_text, script_json, news_json, est_minutes,
        resolved["creative_writing_model"], resolved["technical_model"])
```

## §A.3 — `video_engine.py` genre fall-throughs (C3 target)

```python
# nodes/video_engine.py, line 711
"style":      style or genre or "sci-fi",

# nodes/video_engine.py, lines 834-836
("STYLE", self.data.get("style", self.data.get("genre", "?"))),

# nodes/video_engine.py, line 1075
style = style or genre or "audio drama"
```

Plus projection at `nodes/otr_video_plan.py:311`:

```python
"genre":             visual_plan.get("genre") or "",
```

## §A.4 — Era literals (C2a target — all 4+2 visual-layer sites)

```python
# visual/batch_flux_portrait_render.py, line 109 -- Python fallback
style_anchor = (style_anchor or "1940s noir radio drama style").strip()
# line 170 -- ComfyUI widget default in INPUT_TYPES
"default": "1940s noir radio drama style, cinematic",
# line 234 -- function-signature default on execute()
style_anchor: str = "1940s noir radio drama style, cinematic",
```

```python
# nodes/otr_video_plan.py, lines 78-81
_DEFAULT_STYLE_TAIL = (
    "cinematic, 35mm film look, 1980s broadcast aesthetic, "
    "subtle film grain, volumetric lighting"
)
```

Plus persisted widget values in `workflows/otr_scifi_16gb_full.json`:
- Line 706: `"cinematic, 35mm film look, 1980s broadcast aesthetic, subtle film grain, volumetric lighting"`
- Line 1915: `"1940s noir radio drama style, cinematic"`

Test fixture at `tests/test_musicgen_cache_keys.py:23` retains its noir literal (cache-key payload). C2/C3 forbidden-sweep markers must NOT trip on test fixtures.

## §A.5 — Fail-loud sentinel pattern (C5a1 contract — L-6)

```python
# nodes/_otr_ledger_reviewer.py, lines 815-854 (run_script_doctor)
try:
    raw = generate_fn(messages, temperature=_DOCTOR_TEMPERATURE,
                      max_new_tokens=_DOCTOR_MAX_NEW_TOKENS)
except Exception as exc:  # noqa: BLE001
    log.warning("[OTR_LedgerReviewer:doctor] generate_fn raised: %s; "
                "returning needs_full_rerun report", exc)
    return ScriptDoctorReport(overall_verdict="needs_full_rerun")
json_str = _extract_json_block(raw or "")
try:
    data = json.loads(json_str)
except json.JSONDecodeError as exc:
    log.warning("[OTR_LedgerReviewer:doctor] JSON parse failed (%s); "
                "raw=%r; returning needs_full_rerun report",
                exc, (raw or "")[:200])
    return ScriptDoctorReport(overall_verdict="needs_full_rerun")
try:
    report = ScriptDoctorReport.model_validate(data)
except ValidationError as exc:
    log.warning("[OTR_LedgerReviewer:doctor] schema validation failed (%s); "
                "returning needs_full_rerun report", exc)
    return ScriptDoctorReport(overall_verdict="needs_full_rerun")
return report
```

C5a1 reflection pass mirrors this exactly: three explicit `except` arms; each logs + returns the empty-brief-with-status sentinel; no bare `return`.

## §A.6 — MusicGen audio coupling (L-4 evidence)

```json
// workflows/otr_scifi_16gb_full.json -- node 7 OTR_EpisodeAssembler inputs
{"name": "scene_audio",          "type": "AUDIO", "link": 6},
{"name": "opening_theme_audio",  "type": "AUDIO", "link": 22},   // from MusicGen
{"name": "closing_theme_audio",  "type": "AUDIO", "link": 23}    // from MusicGen

// node 7 outputs
{"name": "episode_audio", "type": "AUDIO", "links": [15, 78]}    // final episode audio

// node 35 OTR_MusicGenTheme outputs
{"name": "opening_audio", "type": "AUDIO", "links": [22]},
{"name": "closing_audio", "type": "AUDIO", "links": [23]}
```

Changing MusicGen's mood prompt changes `opening_audio` + `closing_audio` → `OTR_EpisodeAssembler` → `episode_audio`. Per E-12 (operator directive), this audio change is accepted and contained to C5g: the audio C7 byte-identical pytest proxy holds against the existing baseline through C5f, intentionally resets at C5g with a new canonical b3sum captured, and holds against the new baseline from C5g forward.

## §A.7 — `DEFAULT_LLM` + audio-baseline catalog note (L-1 evidence + R-01 evidence)

```python
# nodes/_otr_model_catalog.py, lines 32-76
DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"
"""Default for both writer slots. Audio C7 byte-identical baseline."""

CURATED_LLM_MODELS = (
    CuratedModel(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        requires_auth=True,
        loader_backend="transformers_safetensors",
        vram_fit_tier="PASS",
        approx_safetensors_gb=24.0,
        notes="Audio C7 regression baseline -- soak-tested. Default for both slots.",
    ),
    # ... 5 more entries; gemma-4-E4B-it present but NOT marked default
)

# nodes/_otr_model_catalog.py, lines 553-555
# OTR's loader uses 8-bit / NF4-style quantization by default, so peak
# resident is roughly half the BF16 safetensors download size.
# Documented numbers per plan section 6:
#     Mistral-Nemo:  ~24 GB disk -> ~12 GB resident.
#     Gemma-4-E2B:    ~6 GB disk -> ~3 GB resident.
```

R-01 reviewer claimed 24GB blows the 14.5GB ceiling. The 24GB is disk-size BF16. Resident is ~12GB post-quantization. Composition pass already lives in this envelope every sprint. Plus per-call activation memory (~1-2GB), peak ~14GB. Under ceiling.

## §A.8 — `_run_with_timeout` BUG-LOCAL-228 contract (L-3 evidence)

```python
# nodes/story_orchestrator.py, lines 327-397 (abbreviated)
except FuturesTimeout:
    # orphan worker still running GPU forward pass; cannot kill
    # invalidate cache dict references WITHOUT touching GPU
    _otr_loader_mod.invalidate_cache_no_gpu_teardown()
    # raise workflow-pause subclass so ComfyUI halts the queue
    raise _LLMTimeoutWorkflowPause(...)
finally:
    # Do not wait for the orphaned worker -- let it drain in the background.
    executor.shutdown(wait=False)
```

Refinement §3.6 says "cancel the future, force GPU sync, empty cache, verify VRAM headroom before yielding." That is the anti-pattern BUG-LOCAL-228 was filed against. C0b amends the refinement doc with the correct contract.

## §A.9 — Already-shipped envelope items (C4 lock-in targets)

```python
# nodes/_otr_model_catalog.py
DEFAULT_VRAM_CEILING_GB = 14.5                                     # line 510 -- S21.1
HARD_VRAM_CONTEXT_LIMIT = _hard_vram_context_limit()  # -> 8192    # line 420 -- S21.1 + S30 B1b
CURATED_CONTEXT_OVERRIDES = {
    "mistralai/Mistral-Nemo-Instruct-2407": 8192,
    "google/gemma-4-E2B-it":                8192,
    "google/gemma-4-E4B-it":                8192,                  # line 432
    # ...
}
```

Plus `_otr_model_loader.py:396` (`if total_vram >= 14.5:`), `_vram_log.py:45`, `visual/lhm_monitor.py:47`. NO `15.0` threshold exists anywhere. C4 ships regression-guard tests only, no production changes.

## §A.10 — Orchestrator era-anchor sites (E-13 / C2b target — orchestrator-layer sites)

Three runtime LLM-prompt sites currently bake era flavor into every script generation regardless of `meta.style`:

```python
# nodes/story_orchestrator.py, lines 1591-1603 (_llm_rerank_with_bodies prompt)
prompt = (
    f"You are picking ONE news story to seed a {genre_human} radio "
    f"drama. You have already shortlisted {len(candidates_with_body)} "
    f"candidates by headline. Now you can read each article body. "
    f"Choose the SINGLE story with the strongest narrative bones "
    f"for a 1940s-style radio drama: specific human stakes, "      # <-- C2b target
    f"mystery, scientific breakthrough, or vivid scene potential. "
    ...
)
```

```python
# nodes/story_orchestrator.py, lines 2999-3006 (SCAFFOLDING_PREAMBLE dramaturg craft anchor)
SCAFFOLDING_PREAMBLE = """<system_role>
You are a MASTER DRAMATURG for the audio drama anthology "SIGNAL LOST". Not a
novelist. Not a writer. A DRAMATURG. Your job is to produce AUDITORY BLUEPRINTS
- precise, timed, sound-first specifications that a director, a voice cast, and
a Foley artist could record tonight. You think like the golden age of radio
drama: Orson Welles, Norman Corwin, Lucille Fletcher. The page is NEVER prose.    # <-- C2b target
The page is a recording score.
</system_role>
```

```text
# nodes/story_orchestrator.py, lines 3160-3179 (worldbuilding rules section, OMNI-RETRO 5-pillar block)
=== [EMOJI] WORLDBUILDING, RHYTHM, & SONIC ARCHITECTURE RULES ===

1. OMNI-RETRO CULTURAL COLLISION:
This world is a massive, colliding melting pot of five distinct aesthetics: 1950s Americana Noir, Afrofuturism, Neo-Tokyo Cyberpunk, Thai Street Density, and Russian Dieselpunk. When writing the story, casually mix these cultures. A 1950s detective might argue with an Afrofuturist engineer in a Neo-Tokyo noodle bar during a Thai monsoon.

2. TEXTURAL SOUND DESIGN ([ENV:] and [SFX:]):
Make the world sound like a collision of these cultures. Use [ENV:] and [SFX:] to paint the setting BEFORE anyone speaks. Mix at least TWO cultural soundscapes per scene.
- 1950s Americana: crackling radio static, humming neon, theremin swells, revolver clicks.
- Neo-Tokyo: high-pitch digital buzzing, mag-lev trains, synthetic rain, holographic ad jingles.
- Thai: monsoon rain on tin roofs, distant temple gongs, sizzling street woks, sputtering tuk-tuks.
- Russian Dieselpunk: brutalist echoes, heavy diesel machinery, hydraulic hisses.
- Afrofuturist: analog synth swells, polyrhythmic drum-circle static, deep bass hums.

WRONG [ENV:]: [ENV: a futuristic city street]
RIGHT [ENV:]: [ENV: heavy Thai monsoon on tin roofs, Neo-Tokyo mag-lev train screams overhead, deep dieselpunk engine idling]

3. RHYTHM & PACING (CRITICAL FOR TTS):
...
- Keep golden-age radio pacing: short, punchy, visceral dialogue.   # <-- C2b target
```

This entire `OMNI-RETRO CULTURAL COLLISION` paragraph + its 5 textural-soundscape bullets + the `golden-age radio pacing` line are replaced at C2b with a single `{style_world_block}` placeholder, interpolated at `.format()` time from `_resolve_style_world_block(resolved["style"])`. Per-style world tone replaces fixed 5-pillar omni-retro. Every script becomes unique to its chosen style. The dramaturg discipline (sound-first, scaffolding) is preserved; only the era flavor is removed and replaced with dynamic style-driven flavor.

**Sites NOT targeted at C2b (mixed-pool randomized data, kept):**
- `nodes/story_orchestrator.py:806-820` — `_MINCED_OATHS` array (golden-age radio + pulp adventure + sci-fi space-opera categories; randomized per-replacement so no single era dominates the output script).
- `nodes/story_orchestrator.py:876-916` — `_FIRST_NAMES` array (mixed 5-pillar + classic-fiction + actor-name pools; era tags in comments only).
- `nodes/story_orchestrator.py:918+` — `_LAST_NAMES` array (same pattern).

Comment-only era references at lines 804 and 874-877 are cosmetically updated at C2b (optional, no runtime effect).

---

# §B. C1 staleness audit (2026-05-15)

Audit completed at C1 against `f758f02` HEAD on branch `s34-p0-p1-hotfix` (parent of `sprint-c-story-brief-v2`). Verification commands documented inline. Two contingencies activated; both have in-spec resolution paths and are encoded above (E-15 row, E-24 file-path retarget, §6 C5a2 + §6 C5c spec edits).

## §B.1 Audit table — verified state

| Locked-plan claim | Verification command (cmd, run from repo root) | C1-verified state |
|---|---|---|
| `_GENRE_BY_STYLE` table at `OTR_LedgerScriptWriter.py:254-265` | `findstr /n /c:"_GENRE_BY_STYLE: dict" nodes\OTR_LedgerScriptWriter.py` | Confirmed line 254. §A.1 holds. |
| `_resolve_genre` + `_preview_genre` helpers | `findstr /n /c:"def _resolve_genre" /c:"def _preview_genre" nodes\OTR_LedgerScriptWriter.py` | Confirmed lines 268, 296. §A.1 holds. |
| `DEFAULT_LLM = "mistralai/Mistral-Nemo-Instruct-2407"` (audio C7 baseline) | `findstr /n /c:"DEFAULT_LLM = " nodes\_otr_model_catalog.py` | Confirmed line 32. §A.7 holds. L-1 holds. |
| `meta.visual_plan.genre` stamp at K.5 (claim: line 2801) | (line-number drift tolerated; constant referenced via §A.2) | Spot-checked — `_resolve_genre` is called from K.5; stamp present. §A.2 holds within line-number drift. |
| Three `video_engine.py` genre fall-throughs | (claim: lines 711, 836, 1075 per §A.3) | Spot-checked via §A.3 quoted excerpts; expressions present. §A.3 holds. |
| `_DEFAULT_STYLE_TAIL` at `otr_video_plan.py:78-81` | (claim per §A.4) | Spot-checked; constant present. §A.4 holds. |
| `"1940s noir radio drama style"` at `batch_flux_portrait_render.py:109/170/234` + 2 workflow JSON sites | (claim per §A.4) | Spot-checked; literals present. §A.4 holds. |
| `DEFAULT_VRAM_CEILING_GB == 14.5` | (claim per §A.9) | Confirmed via S21.1 history; §A.9 holds. |
| `HARD_VRAM_CONTEXT_LIMIT == 8192` + Gemma-4-E4B override == 8192 | (claim per §A.9) | Confirmed via S21.1 + S30 B1b history; §A.9 holds. |
| `_run_with_timeout` non-blocking + `invalidate_cache_no_gpu_teardown` recovery | `findstr /n /i /c:"invalidate_cache_no_gpu_teardown" nodes\_otr_model_loader.py` | Confirmed at line 623; non-blocking contract present. §A.8 holds. L-3 holds. |
| MusicGen → EpisodeAssembler → `episode_audio` routing | (claim per §A.6) | Workflow JSON routing present. §A.6 holds. L-4 override at C5g remains valid. |
| FLUX env/bookend implementation file path (E-24): `visual/batch_flux_env_render.py` OR `visual/batch_flux_render.py` | `dir /b visual\batch*.py` returned `batch_flux_portrait_render.py`, `batch_flux_render.py` | **CONTINGENCY ACTIVATED.** `batch_flux_env_render.py` does not exist. `batch_flux_render.py` is the actual env/bookend implementation. C5c retargeted (§6 C5c, §2 row 18, E-24). |
| `evict_model` symbol in `_otr_model_loader.py` (precondition for E-15 explicit eviction) | `findstr /n /c:"def evict_model" nodes\_otr_model_loader.py` | Symbol absent. **DEEPER FINDING:** `_otr_model_loader.py` exposes `unload_llm()` (line 555), `invalidate_cache_no_gpu_teardown()` (line 623), and `request_slot(model_id)` (load-or-reuse with auto-unload at lines 732-739). The loader is single-slot by architecture — only one LLM is resident at a time. `request_slot` already handles eviction implicitly: if a different model is currently resident, it calls `unload_llm()` before loading the next. Peak transient VRAM during a model swap is `max(creative_size, technical_size)`, NOT their sum. The OOM scenario E-15 / RR-A1 originally guarded against (creative + technical concurrently resident summing to ~15 GB) cannot occur on this loader architecture. **Resolution per operator directive 2026-05-15:** delete the explicit `evict_model` call; replace with regression tests that prove the no-OOM property and verify `request_slot`'s swap order. E-15 row revised, §6 C5a2 spec/code/tests/commit-subject revised, §2 row 14 revised, §7 row 28 revised. |

## §B.2 Findings summary

**Finding F1 — FLUX env/bookend file path mismatch (E-24 / RR-B8 contingency activated).**

- v3 plan claim: `visual/batch_flux_env_render.py` is the FLUX env/bookend implementation file.
- HEAD reality: `visual/batch_flux_render.py` is the actual file. `batch_flux_env_render.py` does not exist.
- Resolution: C5c spec retargeted to `visual/batch_flux_render.py` per the C5c spec contingency. No deviation from plan intent — the contingency was authored for exactly this case.

**Finding F2 — `evict_model` symbol absent; loader is implicitly single-slot.**

- v3 plan claim: `_otr_model_loader.evict_model(model_id)` exists; C5a2 calls it explicitly to evict the creative model before reflection.
- HEAD reality: no `evict_model` symbol. The loader is single-slot by architecture: `request_slot(model_id)` calls `unload_llm()` to evict any prior resident model BEFORE loading the next.
- Implication: peak transient VRAM during a model swap = `max(creative_size, technical_size)`, not their sum. The OOM scenario RR-A1 was concerned about (both models concurrently resident → ~15 GB > 14.5 GB ceiling) cannot occur on this loader.
- Resolution per operator directive: delete the explicit eviction call. Replace with regression tests (a) proving total resident VRAM never exceeds 14.5 GB during composition→reflection in dual-slot config, (b) verifying `request_slot`'s swap order calls `unload_llm(creative)` BEFORE the technical-model load, (c) confirming single-slot config emits no spurious unload.
- E-15 row revised; §6 C5a2 imports (`evict_model` removed), code (eviction block removed), wire paragraph (contingency closed), pytest table (3 new regression tests), and commit subject all revised. §2 precondition row 14 and §7 acceptance row 28 also revised.

## §B.3 Disposition authority

Both findings disposed by operator directive in this conversation, 2026-05-15. F1 invokes the v3-plan contingency for E-24 / RR-B8 verbatim. F2 deletes the explicit-eviction approach in favor of regression tests, on the grounds that the original OOM concern was framed for a hypothetical multi-slot loader and does not apply to this codebase's single-slot architecture.

---

# §6. Commit structure (17 commits: C0a + C0b + C1 + C2a + C2b + C3 + C3b + C4 + C5a1 + C5a2 + C5b + C5c + C5d + C5e + C5f + C5g + C-final)

**Cleanbreak block (C2a → C3b):** all legacy retirements land before any new `meta.story_brief` code is built — era literals (C2a visual, C2b orchestrator), `_GENRE_BY_STYLE` (C3), `meta.ltx_style_brief` (C3b). Order: visual era literals → orchestrator era literals + style-world-block → genre table → ltx_style_brief.

**Envelope lock-in (C4):** baseline regression-guard tests; no production change.

**New system (C5a1 → C5g):** reflection module → writer wiring + dual-slot eviction → helpers → FLUX env/bookend → FLUX portraits → LTX → HuMo → MusicGen. The C5g MusicGen wiring is the boundary where audio C7 baseline resets per E-12; absent-brief isolation test per E-16 proves the shift is exclusively mood-prefix.

**Per-commit loop (operator directive):** review → code → wire → pytest → regression → commit. No operator gates between commits; chain runs through to C-final. Bug Bible regression runs automatically after every code change.

## C0a — Branch cut + plan landing (~0.1 d) — HARD STOP if Windows checkout is dirty

**Hard stop (E-10):** before any other C0a step, operator confirms on Windows:

```
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git status --short
git branch --show-current
git rev-parse HEAD
git diff --name-status
git diff --cached --name-status
```

`git status --short` MUST be empty. Branch MUST be `s34-p0-p1-hotfix`. HEAD MUST be `f758f02`. The sandbox view at v3 authorship showed `AD ./` + `AU "\a\al}"` + `AM` across nearly every tracked file — that pattern is a poisoned index. If anything like it appears on Windows, **the sprint does not start.** Operator clears (`git reset`, `git checkout -- .`, resolve the unmerged garbage-filename path) and re-verifies clean.

**Review.** Confirm parent + clean tree per the hard stop.
**Code.** Cut `sprint-c-story-brief-v2`. Land this final plan at `docs/2026-05-15-sprint-c-plan-final.md`. Mark v3-roundrobin and v2 plans as superseded with one-line preambles.
**Wire / Pytest.** None.
**Commit gate.** Plan present. Branch cut clean. Hard stop satisfied.
**Commit subject.** `C0a: branch cut + Sprint C final consolidated plan landing`

## C0b — Refinement spec amendment (~0.25 d, doc-only)

**Review.** Confirm refinement §3.5, §3.6, §11.4 wording against L-3 (sync barrier) + R-06 (repair pass).
**Code.** Edit `docs/2026-05-12-story-brief-v2-design-refinements.md`:
- **§3.5 (R-06):** replace the existing repair-pass spec with: "On validation failure, run ONE repair pass at temperature `reflection_temperature + 0.15` (effective range 0.35-0.55). The repair prompt prepends an explicit `CRITICAL: You previously failed validation because: <rejection_reasons_list>.` directive before the refinement-§3.5 rewrite instruction. This breaks the deterministic-retry failure loop characteristic of low-temperature local-model JSON output."
- **§3.6 (L-3):** replace "Reflection pass MUST clear before the workflow advances" + "cancel the future, force GPU sync, empty cache" with: "On `_LLMTimeout` the reflection pass MUST follow the BUG-LOCAL-228 contract: do not block, invalidate cache dict references via `invalidate_cache_no_gpu_teardown`, raise `_LLMTimeoutWorkflowPause` to halt the queue. The orphan worker drains naturally; the next request_slot forces a fresh load."
- **§11.4 (L-3):** align same.
- Add a forensic footnote citing BUG-LOCAL-228, S31 B4 fix `a4fe67a`, and `story_orchestrator.py:336-372`.

**Wire / Pytest.** None.
**Commit gate.** Refinement doc internally consistent with L-3 + R-06. Diff is doc-only.
**Commit subject.** `C0b: refinement §3.5 (repair temp+critical) + §3.6 + §11.4 (sync-barrier per BUG-LOCAL-228) amended`

## C1 — Staleness audit addendum (~0.25 d, no code changes)

**Review.** Walk audit table below against `f758f02` HEAD.

| Locked-plan claim | C1-verified state |
|---|---|
| `_GENRE_BY_STYLE` table at `OTR_LedgerScriptWriter.py:254-265` | Confirmed (§A.1). |
| `meta.visual_plan.genre` stamp at line 2801 | Confirmed (§A.2). |
| Three `video_engine.py` fall-throughs at 711, 836, 1075 | Confirmed (§A.3). |
| `_DEFAULT_STYLE_TAIL` at `otr_video_plan.py:78-81` | Confirmed (§A.4). |
| `"1940s noir radio drama style"` at `batch_flux_portrait_render.py:109` + 170 + 234 | Confirmed all 3 sites + 2 workflow JSON sites (§A.4). |
| `DEFAULT_LLM = mistralai/Mistral-Nemo-Instruct-2407` (audio C7 baseline) | Confirmed (§A.7). |
| `DEFAULT_VRAM_CEILING_GB == 14.5` | Confirmed — S21.1 `6d08f63` (§A.9). |
| `HARD_VRAM_CONTEXT_LIMIT == 8192` + Gemma-4-E4B override == 8192 | Confirmed — S21.1 + S30 B1b (§A.9). |
| `_run_with_timeout` non-blocking + `invalidate_cache_no_gpu_teardown` recovery | Confirmed — S31 B4 `a4fe67a` (§A.8). |
| MusicGen → EpisodeAssembler → `episode_audio` routing | Confirmed (§A.6). L-4 override at C5g validated. |
| FLUX env/bookend implementation file path (E-24): `visual/batch_flux_env_render.py` OR `visual/batch_flux_render.py` | Verified at C1 against `f758f02` HEAD. If `batch_flux_env_render.py` does not exist, C5c spec retargets to `batch_flux_render.py` before C5c codes. |

**Code.** `docs/2026-05-15-sprint-c-C1-staleness-audit.md` mirroring this table + verification commands.
**Wire / Pytest.** None.
**Commit gate.** Audit document complete.
**Commit subject.** `C1: staleness audit -- Sprint C final scope vs s34-p0-p1-hotfix HEAD`

## C2a — Era literal cleanbreak — visual layer (~0.4 d, split per operator sizing directive)

**Review.** Targets enumerated in §A.4 (visual layer). C2 was split into C2a + C2b at v3 cut time to keep each commit inside one review-code-wire-pytest-regression-commit loop. C2a is the lower-risk half: visual-prompt literals only; no script-gen prompt change; audio C7 expected to hold without drama.

**Code.** Replace all 4 Python sites + 2 workflow JSON values with era-neutral text. **Workflow JSON edits use exact-string recursive replacement (E-19 / RR-B6), NOT line numbers** — line numbers drift on any unrelated workflow edit.

- `visual/batch_flux_portrait_render.py:109, 170, 234`: replace `"1940s noir radio drama style, cinematic"` → `"head-and-shoulders studio portrait, neutral lighting, cinematic"`.
- `nodes/otr_video_plan.py:78-81` `_DEFAULT_STYLE_TAIL`: `"cinematic, 35mm film look, subtle film grain, volumetric lighting"` (drop the `"1980s broadcast aesthetic"` clause).
- `workflows/otr_scifi_16gb_full.json`: recursive walk, replace every occurrence of these exact string values:
  - `"cinematic, 35mm film look, 1980s broadcast aesthetic, subtle film grain, volumetric lighting"` → `"cinematic, 35mm film look, subtle film grain, volumetric lighting"`
  - `"1940s noir radio drama style, cinematic"` → `"head-and-shoulders studio portrait, neutral lighting, cinematic"`

**Wire.** Workflow JSON re-saved; link integrity validator (`tools/validate_workflow_links.py`) confirms 0 violations. No graph-level wiring change.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_no_1940s_literal_in_visual_layer` | `tests/test_era_literals_c2a.py` (new) | Repo scan: `visual/` + `workflows/`: zero `"1940s"` outside tokenize-suppressed forensic comments and outside `tests/test_musicgen_cache_keys.py:23` fixture. |
| `test_no_1980s_broadcast_literal_in_video_plan` | same | `nodes/otr_video_plan.py` + `nodes/video_engine.py`: zero `"1980s broadcast"` outside tokenize-suppressed forensic comments. |
| `test_workflow_json_string_based_replace_completed` | same | `workflows/otr_scifi_16gb_full.json` recursive value-walk: zero occurrences of either pre-replacement string. The expected post-replacement strings ARE present. Locks the E-19 string-based approach. |
| `test_workflow_json_no_era_literal_widget_default` | same | Targeted check: widget-default values for the FLUX portrait nodes contain era-neutral text. |
| `test_audio_c7_byte_identical_c2a` | existing canary | Audio holds against existing baseline. C2a touches only visual-prompt strings — script-gen prompts and audio pipeline are untouched at this commit. Expected to hold cleanly. |

**Commit gate.** All tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean. Workflow link validator clean. **New marker:** `\b1940s\b` (tokenize suppression handles forensic mentions; test fixture at `test_musicgen_cache_keys.py:23` retains its noir-literal cache-key payload — fixture allowed).

**Commit subject.** `C2a: era literal cleanbreak -- visual layer (FLUX portrait fallback/widget/sig + _DEFAULT_STYLE_TAIL + workflow JSON via string-based replace)`

## C2b — Era literal cleanbreak — orchestrator (REDUCED SCOPE per C2b path-A audit) (~0.2 d)

**Review.** Path-A architectural discovery at C2b execution time (see §1.4 RR-A3+RR-B6 row): of the three orchestrator-layer targets originally enumerated in §A.10, only one is runtime-live. The dramaturg-preamble + OMNI-RETRO 5-pillar literals lived inside `SCAFFOLDING_PREAMBLE` and `SCRIPT_SYSTEM_PROMPT` — both orphan constants from the `eec4718` LPL extraction sprint with zero live consumers across `nodes/`, `visual/`, `scripts/`, and `tests/`. The new `OTR_LedgerScriptWriter` pipeline composes prompts from per-phase modules. The 8-search audit (getattr / glob-import / substring-content / alt-name / `git log -S` / writer-callsite / `__all__` / tests) returned no hits for either constant outside the definition site and forensic comments. Reduced C2b ships the runtime fix + deletes the orphan constants per the no-legacy-back-compat standing directive; the `_STYLE_WORLD_BLOCK` machinery is skipped (no live consumer = no place to interpolate). The broader orphan-constant sweep of `nodes/story_orchestrator.py` (3000+ lines, gutted across LPL / S31 B3 / S34) deferred to Sprint G per the no-scope-creep rule.

**Code.** Three changes in `nodes/story_orchestrator.py`:

1. **`story_orchestrator.py:1596`** — `_llm_rerank_with_bodies` body-rerank prompt. The ONLY runtime-live era-literal site at the orchestrator layer. Replace:
   ```
   "for a 1940s-style radio drama: specific human stakes, "
   ```
   with:
   ```
   "for an audio drama: specific human stakes, "
   ```
   Style-neutral. The `{genre_human}` interpolation at line 1592 already carries the chosen style into the prompt.

2. **DELETE `SCAFFOLDING_PREAMBLE` constant** (was lines 2999-3046 plus its header comment at 2994-2998). Orphan from `eec4718` LPL extraction; no live consumer. No shim, no alias, no migration. Per `from .story_orchestrator import SCAFFOLDING_PREAMBLE` becomes `AttributeError` — intentional, so dead wirings fail loud.

3. **DELETE `SCRIPT_SYSTEM_PROMPT` constant** (was lines 3049-3478). Same rationale. The 5-pillar block + dramaturg name-drops + golden-age-radio pacing lines lived inside this constant; deleting the constant deletes the era literals.

Both constants are replaced with a single forensic comment block at the deletion site explaining the cleanup, the 8-search audit pointer, and the Sprint G handoff. The pre-existing `_load_canon_for_writer` function (also orphan-pending-Sprint-G-audit) has its docstring updated to note SCAFFOLDING_PREAMBLE is now gone.

**Skipped — `_STYLE_WORLD_BLOCK` machinery.** The dict + helper + interpolation wiring described in the v3 plan are not built. Building them would ship dead code to replace dead code. If a future commit wires a per-style world block into one of the live per-phase prompt modules (`_otr_outline._SYSTEM_PROMPT`, `_otr_line_composer._SYSTEM_PROMPT`, `_otr_ledger_reviewer`, `_otr_period_prompts.OTR_PERIOD_SYSTEM_PROMPT`), the dict + helper can be revived from this plan at that time.

**Skipped — comment cleanup at lines 804 / 874-877.** Out of scope per the lock-it-tight directive. Sprint G's broader orphan/era sweep will handle data-array comment cleanup as part of its single audit pass.

**Out of scope at C2b.** `_MINCED_OATHS` / `_FIRST_NAMES` / `_LAST_NAMES` data arrays (mixed-pool, randomized — kept). All other orphan candidates in `story_orchestrator.py` (deferred to Sprint G).

**Wire.** None. No graph surface, no widget, no workflow JSON change. The deletion eliminates an import surface; no new one is created.

**Test fallout — `tests/test_core.py::test_citation_rule_present` DELETED.** The test pinned the substring `"CITATION RULE"` or `"cite ONLY"` inside the orchestrator+loader source union; both lived ONLY inside the now-deleted SCRIPT_SYSTEM_PROMPT. Per the no-legacy-back-compat rule (delete the old function AND every test that pinned the old contract), the test is removed and a forensic comment is added to the existing test-deletion comment block in `TestStoryOrchestratorCodePatterns` documenting why. No live per-phase prompt currently carries an equivalent citation-rule string; if the LPL pipeline needs one in a future sprint, the replacement test should live alongside whichever per-phase prompt module gets the rule, not as a union-of-files substring scan.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_no_1940s_literal_in_rerank_prompt` | `tests/test_era_literals_c2b.py` (new) | The rerank prompt site no longer contains `"1940s-style radio drama"`. |
| `test_rerank_prompt_audio_drama_anchor_present` | same | The expected replacement text `'"for an audio drama: specific human stakes, "'` is present at the rerank prompt site. Locks against silent reversion. |
| `test_scaffolding_preamble_constant_deleted` | same | `nodes.story_orchestrator.SCAFFOLDING_PREAMBLE` does NOT exist (`hasattr` is False). Runtime assertion, not source scan — so forensic attribution comments in tests / docstrings remain allowed. |
| `test_script_system_prompt_constant_deleted` | same | `nodes.story_orchestrator.SCRIPT_SYSTEM_PROMPT` does NOT exist. Same pattern. |
| `test_literal_not_in_runtime_strings` (parametrized over 10 era literals) | same | For each of `1950s Americana`, `Afrofuturism`, `Neo-Tokyo`, `Thai Street Density`, `Russian Dieselpunk`, `Orson Welles`, `Norman Corwin`, `Lucille Fletcher`, `golden age of radio`, `OMNI-RETRO CULTURAL COLLISION`: zero hits in code-or-string-context of `nodes/story_orchestrator.py` (tokenize-classified comment hits suppressed; the C2b deletion forensic note itself is "comment" so filtered). |
| `test_audio_c7_byte_identical_c2b` | existing canary | Audio holds. C2b changes only the rerank prompt at line 1596 (used by news rerank, not script gen) and deletes orphan constants. Script-gen prompts are not touched — they live in per-phase modules. Expected to hold cleanly. |

**Commit gate.** All tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean. **New markers** (armed prospectively; no current hits since the only source instances were deleted with the constants): `1950s Americana`, `golden.age radio`, `\bOmni.Retro\b`, `\bOrson Welles\b`, `\bNorman Corwin\b`, `\bLucille Fletcher\b`. Tokenize suppression handles forensic mentions in comments and docstrings.

**Commit subject.** `C2b: era literal cleanbreak -- orchestrator (rerank prompt line 1596) + deleted orphan SCRIPT_SYSTEM_PROMPT and SCAFFOLDING_PREAMBLE constants (C1 audit confirmed no live consumers; _STYLE_WORLD_BLOCK machinery skipped as moot)`

## C3 — `_GENRE_BY_STYLE` deletion (~1 d)

**Review.** §A.1 + §A.2 + §A.3 targets.
**Code.** Delete the table + helpers + stamp + 3 fall-throughs + projection + forensic-name mentions (convert to hash citations) + `tests/test_musicgen_style_palette.py:229-331` genre-table tests.
**Wire.** None graph-level. (HUD + treatment-text display readers fall back to `style` slug.)

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_no_genre_by_style_constant` | `tests/test_no_genre_by_style_c3.py` (new) | `_GENRE_BY_STYLE` not importable from `nodes.OTR_LedgerScriptWriter`. |
| `test_no_resolve_genre_helper` | same | `_resolve_genre` not importable. |
| `test_no_preview_genre_helper` | same | `_preview_genre` not importable. |
| `test_no_meta_visual_plan_genre_stamp` | same | Run writer; `meta.visual_plan` keys: `{characters, scenes, style}` — no `genre`. |
| `test_video_engine_no_genre_fallthroughs` | same | AST walk `nodes/video_engine.py`; no `genre` token in the three previously-cited expressions. |
| `test_hud_and_treatment_render_when_visual_plan_genre_absent` (E-05; renamed per E-26 / RR-B11) | same | Fixture ledger with `meta.style = "noir_interrogation"`, no `meta.visual_plan.genre`. HUD frame-builder `("STYLE", ...)` row renders; treatment-text `Style:` line renders. Neither crashes; neither emits a missing-genre warning. Locks refinement §8.2 graceful-fallback promise before `_GENRE_BY_STYLE` is pulled. (Renamed from `_without_visual_plan_genre` to avoid language that could be misread as legacy back-compat / shim allowance.) |
| `test_audio_c7_byte_identical_c3` | existing canary | Audio holds. |

**Commit gate.** New tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean. **New marker:** `\b_GENRE_BY_STYLE\b` (tokenize suppression handles forensic mentions). Test fixture at `test_musicgen_cache_keys.py:23` retains its noir-literal cache-key payload (allowed — fixture data).
**Commit subject.** `C3: pre-flight cleanbreak -- _GENRE_BY_STYLE deleted (table + helpers + stamp + 3 fall-throughs)`

## C3b — `meta.ltx_style_brief` retirement (~0.25 d, pulled forward per E-14)

**Review.** Per Cowork audit F-10: most retire happened in S31 B3 (`_generate_ltx_style_brief` + `_LTX_STYLE_BRIEF_PROMPT` deleted). Remaining: stale comment block at `nodes/batch_ltx_render.py:395-403` + forbidden-sweep marker. Pulled forward from former-C5g position per operator directive on legacy ordering — old key retires BEFORE the new `meta.story_brief` system is built.

**Code.** Delete stale comment block. Confirm zero remaining references in `nodes/`, `visual/`, `workflows/`, `scripts/`.
**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_no_meta_ltx_style_brief_references` | `tests/test_no_ltx_style_brief_c3b.py` (new) | Repo-wide scan of `nodes/` + `visual/` + `workflows/` + `scripts/`: zero references to `meta.ltx_style_brief` outside tokenize-suppressed forensic comments. |
| `test_no_generate_ltx_style_brief_symbol` | same | `_generate_ltx_style_brief` not importable from `nodes.story_orchestrator`. |
| `test_no_ltx_style_brief_prompt_constant` | same | `_LTX_STYLE_BRIEF_PROMPT` not importable. |
| `test_audio_c7_byte_identical_c3b` | existing canary | Audio holds. |

**Commit gate.** Tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean. **New marker:** `\bmeta\.ltx_style_brief\b` (tokenize suppression handles forensic mentions).
**Commit subject.** `C3b: meta.ltx_style_brief retirement -- stale comment removed, forbidden marker armed (pulled forward from former-C5g position)`

## C4 — VRAM envelope lock-in (~0.25 d, regression-guard tests only)

**Review.** §A.7 + §A.8 + §A.9 confirm all four items already shipped. Tests only.
**Code / Wire.** None production. Tests only.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_default_llm_is_mistral_nemo` | `tests/test_vram_envelope_c4.py` (new) | `_otr_model_catalog.DEFAULT_LLM == "mistralai/Mistral-Nemo-Instruct-2407"`. Locks L-1. |
| `test_default_vram_ceiling_14_5` | same | 14.5 across `_otr_model_catalog`, `_vram_log`, `visual.lhm_monitor`. |
| `test_hard_context_limit_8192` | same | `HARD_VRAM_CONTEXT_LIMIT == 8192`. |
| `test_gemma_4_e4b_context_override_8192` | same | `CURATED_CONTEXT_OVERRIDES["google/gemma-4-E4B-it"] == 8192`. |
| `test_run_with_timeout_non_blocking` | same | AST walk: `executor.shutdown` called with `wait=False`. Locks BUG-LOCAL-228 fix. |
| `test_run_with_timeout_invalidates_cache` | same | AST walk: `invalidate_cache_no_gpu_teardown` referenced in the timeout branch. |
| `test_audio_c7_byte_identical_c4` | existing canary | Audio holds. |

**Commit gate.** Tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.
**Commit subject.** `C4: VRAM envelope lock-in -- S21.1 + S30 B1b + S31 B4 baseline regression-guarded`

## C5a1 — Reflection pure module (~0.75 d, NO writer wiring)

**Review.** Refinement §2 + §3 (post-C0b amendment) + §4 + §6.3 + L-6 fail-loud pattern. E-17 (scoped try/except per RR-B3). E-18 (repair-temperature clamp per RR-B5). E-21 (technical-slot + cache tests moved to C5a2 per RR-B1/RR-B2).

**Code.** New module `nodes/_otr_story_brief.py`:
- `_build_reflection_input(led: Ledger) -> str` — capped input builder per §2.
- `_REFLECTION_PROMPT` constant — strict-JSON, ≤250 tokens, no-period rule per §3.3.
- `_validate_brief(brief: str, ledger: dict) -> list[str]` — rejection reasons per §3.4.
- `_repair_pass(failed_output, errors, technical_fn, reflection_temperature) -> str` — ONE repair attempt with **temperature clamp per E-18**:
  ```python
  repair_temperature = min(reflection_temperature + 0.15, 0.55)
  ```
  Explicit `CRITICAL: <reasons>` prefix (RR-B6 / C0b §3.5 amendment).
- `run_story_brief_reflection(led, technical_fn) -> dict` — main entry; returns the 8-key meta delta (L-8). Tagged `# LLM slot: technical`. **Signature accepts ONLY `technical_fn`** — no `creative_fn` parameter (E-21 / RR-B1).
- **Scoped try/except per E-17 / RR-B3** — three NARROW blocks, no broad function-level wrapper:
  ```python
  # Block 1 -- LLM call only.
  try:
      raw = technical_fn(messages, temperature=_REFLECTION_TEMPERATURE,
                        max_new_tokens=_REFLECTION_MAX_NEW_TOKENS)
  except Exception as exc:  # noqa: BLE001 -- narrow: only the LLM call line
      log.warning("[OTR_StoryBrief] technical_fn raised: %s; "
                  "returning failed-status sentinel", exc)
      return _failure_sentinel()

  json_str = _extract_json_block(raw or "")

  # Block 2 -- JSON parse only.
  try:
      data = json.loads(json_str)
  except json.JSONDecodeError as exc:
      log.warning("[OTR_StoryBrief] JSON parse failed (%s); raw=%r; "
                  "returning failed-status sentinel",
                  exc, (raw or "")[:200])
      return _failure_sentinel()

  # Block 3 -- schema validation only.
  try:
      brief_model = StoryBriefModel.model_validate(data)
  except ValidationError as exc:
      # Try one repair pass before giving up (§3.5 / E-18).
      repaired = _repair_pass(data, exc.errors(), technical_fn,
                              reflection_temperature=_REFLECTION_TEMPERATURE)
      try:
          brief_model = StoryBriefModel.model_validate(json.loads(repaired))
      except (json.JSONDecodeError, ValidationError) as exc2:
          log.warning("[OTR_StoryBrief] schema validation failed after repair "
                      "(%s); returning failed-status sentinel", exc2)
          return _failure_sentinel()

  return _success_delta(brief_model, ledger=led.data)
  ```
  Matches §A.5 `run_script_doctor` pattern. Each except arm covers EXACTLY one operation. Specific failure classes cannot be swallowed into the generic arm.

**Wire.** None. C5a1 ships module only.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_reflection_input_builder_caps_long_episode` | `tests/test_story_brief_c5a1.py` (new) | Input for a 15-min ledger ≤ ~1500 tokens. |
| `test_reflection_input_builder_includes_required_fields` | same | Output contains title, style, cast, scene headers, opening, closing, non-dialogue rows. |
| `test_reflection_input_builder_reads_lines_not_script_text` | same | AST walk: `_build_reflection_input` references `led.data["lines"]`, NOT a `script_text` parameter. Locks R-02 push-back. |
| `test_reflection_prompt_under_250_tokens` | same | Prompt body ≤ 250 tokens. |
| `test_validation_rejects_named_character` | same | Brief mentioning a cast name fails validation. |
| `test_validation_rejects_dialogue_verb` | same | Brief with "speaking" / "arguing" fails. |
| `test_validation_rejects_decade_literal` | same | Brief with "1940s" fails. |
| `test_validation_rejects_over_300_chars` | same | 301-char brief fails. |
| `test_repair_pass_runs_once` | same | Repair invoked iff initial validation fails; not invoked otherwise. |
| `test_repair_pass_uses_higher_temperature` (R-06) | same | Spy on `technical_fn` calls; assert repair-pass call's `temperature` arg is `reflection_temperature + 0.15` ± epsilon. |
| `test_repair_temperature_clamped_to_055` (E-18 / RR-B5) | same | Call `_repair_pass` with `reflection_temperature=0.5`. Spy on `technical_fn`; assert observed temperature == 0.55, NOT 0.65. Clamp upper bound enforced. |
| `test_repair_pass_prepends_critical_prefix` (R-06) | same | Spy on the repair messages; assert the user message starts with `CRITICAL: You previously failed validation because:`. |
| `test_reflection_exception_sentinel_pattern` | same | AST walk: `run_story_brief_reflection` has 3 distinct `except` arms; each returns the failure sentinel. |
| `test_reflection_exception_arms_are_scoped` (E-17 / RR-B3) | same | AST walk: each `try` block contains EXACTLY ONE statement (the LLM call, OR the `json.loads`, OR the schema validation). NO broad `try: ... except Exception` wraps the whole function body. |
| `test_reflection_stamps_8_meta_keys` | same | After successful run, returned dict contains all 8 meta keys per L-8. |
| `test_reflection_failure_stamps_status_failed` | same | Mock technical_fn to raise; returned dict has `story_brief == ""`, `story_brief_status == "failed"`. |
| `test_reflection_entrypoint_signature_accepts_only_technical_fn` (E-21 / RR-B1) | same | AST walk: `run_story_brief_reflection`'s parameter list contains `led` and `technical_fn`. Does NOT contain `creative_fn`. Replaces the misplaced behavioral-spy test (now lands at C5a2). |
| `test_reflection_slot_tag_technical` | same | Grep `nodes/_otr_story_brief.py` for `# LLM slot: technical`. |
| `test_audio_c7_byte_identical_c5a1` | existing canary | Audio holds (module is unused at this commit). |

(Tests `test_reflection_uses_technical_fn_not_creative_fn` and `test_reflection_does_not_thrash_llm_cache` MOVED to C5a2 per E-21 / RR-B1+RR-B2.)

**Commit gate.** All tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean. **New marker:** `def\s+run_story_brief_reflection.*\n\s+return\s+["']{2}` (catches accidental fail-soft empty-string return without the sentinel dict).
**Commit subject.** `C5a1: reflection pure module -- input builder + strict-JSON prompt + validation + scoped try/except (3 narrow arms) + repair (temp+0.15 clamped to 0.55 / CRITICAL prefix) + 8-key sentinel`

## C5a2 — Writer wiring at K.5.5 (~0.5 d)

**Review.** C5a1 module green. Q1 K.5.5 lock. E-15 revised at C1 audit (loader is single-slot — no explicit eviction call needed; regression-test the no-OOM property instead). E-22 (module-level import per RR-B4). E-21 (technical-slot behavioral spy + LLM-cache spy land here, not at C5a1, per RR-B1/RR-B2).

**Code — module-level import (E-22 / RR-B4).** Add to module-level imports at the top of `nodes/OTR_LedgerScriptWriter.py`:

```python
from ._otr_story_brief import run_story_brief_reflection
```

If a circular import surfaces at commit time, document at C5a2 commit message and fall back to hot-path import with a forensic note. Expected: no circular import — `_otr_story_brief.py` does not import `OTR_LedgerScriptWriter`.

**Code — K.5.5 insertion in `execute()`.** Between section K.5 line 2803 (`meta["style"] = resolved["style"]`) and section L line 2805 (`script_text = ...`):

```python
# K.5.5 -- meta.story_brief reflection pass. Per Sprint C final plan Q1 lock.
# Writes to meta only; lines untouched; runs on technical_model slot
# (creative untouched by L-2). Failure path stamps story_brief_status
# per L-6 fail-loud sentinel pattern.
#
# Dual-slot OOM analysis (E-15 revised at C1 audit per RR-A1):
# The loader is single-slot by architecture. request_slot(technical_model)
# inside technical_fn calls unload_llm() to evict any prior resident model
# BEFORE loading the next. Peak transient VRAM during the swap is
# max(creative_size, technical_size), not their sum. No explicit pre-eviction
# call is needed here; regression tests in C5a2 prove the no-OOM property.
# LLM slot: technical
_brief_delta = run_story_brief_reflection(led, technical_fn=technical_fn)
meta.update(_brief_delta)
```

**Wire.** No new graph surface. No new widget. `technical_fn` is the existing closure from the Two-Model Selector. Workflow JSON unchanged. The C1 audit's evict_model contingency is resolved by deletion: the loader's existing `request_slot`/`unload_llm` swap handles eviction implicitly; no wrapper is needed.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_story_brief_persists_in_saved_ledger` | `tests/test_story_brief_c5a2.py` (new) | Load saved ledger; `meta.story_brief` present on success; `meta.story_brief_status` present on every path. Disk-side proof. |
| `test_story_brief_present_in_returned_script_json` (E-02) | same | Run `execute()`; parse returned `script_json` via `json.loads`. Assert `data["meta"]["story_brief_status"]` exists. Success path: `data["meta"]["story_brief"]` non-empty. Failure path: `story_brief == ""` and `story_brief_status == "failed"`. Socket-side proof — closes K.5.5/L.5 silent-failure class. |
| `test_returned_and_saved_brief_match` (E-11) | same | Run `execute()`; parse returned `script_json` and load saved ledger from disk. Assert identical `meta.story_brief*` fields across both sources. |
| `test_call_site_is_k55_not_l5` | same | AST walk: `run_story_brief_reflection` call appears AFTER `meta["style"] = resolved["style"]` AND BEFORE `script_text = _PL.assemble_script_text_from_ledger(led.data)`. Locks Q1 against future refactor. |
| `test_call_site_before_led_save` | same | AST walk: call appears BEFORE `led.save()` invocation. Belt-and-suspenders against post-save placement. |
| `test_reflection_does_not_mutate_core_ledger_keys` (R-03) | same | `copy.deepcopy(led.data)` pre-reflection; run reflection; assert every top-level key EXCEPT `meta` is byte-identical between pre and post snapshots. Catches accidental nukes of `cast`, `lines`, `news_seed`, `title`, etc. |
| `test_module_level_import_of_reflection_entrypoint` (E-22 / RR-B4) | same | AST walk: `from ._otr_story_brief import run_story_brief_reflection` appears at module-level in `OTR_LedgerScriptWriter.py`. Does NOT appear inside `execute()` body. |
| `test_c5a2_dual_slot_does_not_oom` (E-15 / RR-A1 revised) | same | Fixture: dual-slot config (`creative=Mistral-Nemo`, `technical=Gemma-4-E4B-it`). Instrument `_otr_model_loader` with a VRAM-accounting spy that tracks the running sum of resident model sizes. Run `execute()` through K.5.5+reflection. Assert that at no point during the composition→reflection sequence does the spy report total resident size > 14.5 GB. Proves the no-OOM property without relying on an explicit eviction call. |
| `test_c5a2_dual_slot_evicts_creative_before_technical_loads` (E-15 / RR-A1 revised) | same | Fixture: dual-slot config. Spy on `unload_llm()` and on the technical-model load operation inside `request_slot`. Run `execute()` through K.5.5+reflection. Assert call order: `unload_llm()` invoked with the creative model id BEFORE the technical-model load begins. Verifies `request_slot`'s swap order — which is the actual safety mechanism, given the loader is single-slot. |
| `test_c5a2_single_slot_no_unload` | same | Fixture: single-slot config (`creative == technical == Mistral-Nemo`). Spy on `unload_llm()`. Assert `unload_llm` is NOT called during the composition→reflection sequence — same cache entry, no swap. Single-slot path is the audio C7 baseline path; must not be perturbed by spurious unloads. |
| `test_reflection_uses_technical_fn_not_creative_fn` (E-03, MOVED from C5a1 per E-21 / RR-B1) | same | Behavioral spy at the writer-wiring layer where BOTH closures co-exist: `creative_fn` raises if called, `technical_fn` returns valid JSON. Run `execute()`. Reflection succeeds. `creative_fn` call count during reflection phase == 0. Locks L-2 at the realistic call site. |
| `test_reflection_reuses_technical_model_cache_entry` (R-01, MOVED from C5a1 per E-21 / RR-B2) | same | Behavioral spy on `_otr_model_loader.LLM_CACHE`: clear cache; run `execute()` through composition + reflection; assert the `technical_model` cache entry registers EXACTLY ONE load event across the composition→reflection sequence in single-slot config, OR EXACTLY ONE load event for `technical_model` after creative eviction in dual-slot config. Replaces the vacuous stub-`technical_fn` version that lived at C5a1. |
| `test_audio_c7_byte_identical_c5a2` | existing canary | Audio holds. Single-slot path (default config = audio C7 baseline) must not be perturbed by the dual-slot-only eviction branch. |

**Commit gate.** All tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean. **New marker:** `meta\.story_brief\s*=\s*None` (catches accidental `None`-init that would mask the sentinel-vs-success branch).
**Commit subject.** `C5a2: writer wires reflection at K.5.5 + module-level import (E-22); dual-slot OOM regression-guarded (E-15 revised at C1 -- loader is single-slot; no explicit eviction needed); deep-dict mutation guard; technical-slot + cache-stability tests moved here (E-21)`

## C5b — Central helpers module (~0.25 d)

**Review.** Refinement §5: 5 helpers. L-5 lock + E-04 (superseded by E-12 — helper still ships pure at C5b; consumer wires at C5g).
**Code.** New module `nodes/_otr_story_brief_helpers.py`:

```python
def get_story_brief_full(meta: dict) -> str: ...
def get_story_brief_ltx(meta: dict, max_chars: int = 90) -> str: ...
def get_story_brief_lighting(meta: dict) -> str: ...
def get_story_brief_music_mood(meta: dict) -> list[str]:
    # Pure function. Reads meta.story_brief_terms.atmosphere; intersects with
    # the declared MusicGen mood vocabulary in _MUSIC_MOOD_VOCAB. NO MusicGen
    # import here -- dependency direction is consumer (nodes/musicgen_theme.py)
    # imports this helper at C5g per E-12 / refinement §6.3.
    ...
_MUSIC_MOOD_VOCAB: frozenset[str] = frozenset({
    "tense", "ominous", "melancholic", "hopeful", "urgent", "calm",
    "eerie", "sombre", "playful", "menacing", "wistful", "frantic",
    "reverent", "uneasy", "stoic", "yearning",
})
def get_story_brief_status(meta: dict) -> str: ...  # 'ok' / 'failed' / 'absent'
```

**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_helper_signatures_match_refinement_section_5` | `tests/test_story_brief_helpers_c5b.py` (new) | All 5 helpers exist with documented signatures. |
| `test_get_full_empty_when_absent_or_failed` | same | Returns "" when status != "ok". |
| `test_get_ltx_respects_max_chars_word_boundary` | same | Output ≤ max_chars; trimmed at sentence/clause boundary; never mid-word. |
| `test_get_lighting_joins_lighting_plus_atmosphere` | same | Output is `lighting_terms + atmosphere_terms`, comma-joined. |
| `test_get_music_mood_returns_list_intersected_with_vocab` | same | List output. Given `atmosphere = ["tense", "smoke-filtered", "claustrophobic"]`, returns `["tense"]`. Empty intersection returns `[]`, never raises. |
| `test_get_music_mood_no_musicgen_import` (E-04, dependency direction) | same | AST walk: helper module `_otr_story_brief_helpers.py` does NOT `import nodes.musicgen_theme`. Dependency direction is consumer → helper (musicgen_theme imports the helper at C5g), never the reverse. Guards against accidental circular import when C5g wires the call site. |
| `test_get_status_ok_failed_absent` | same | Returns 'ok' / 'failed' / 'absent' for the three meta states. |
| `test_get_music_mood_has_zero_production_callers_at_c5b` (E-28 / RR-B10) | same | AST-grep across `nodes/`, `visual/`, `scripts/` (excluding `tests/`): zero imports or call-sites of `get_story_brief_music_mood`. Helper is defined at C5b but not yet wired. C5g flips this assertion to "exactly one production caller in `nodes/musicgen_theme.py`." Mechanical guard against rogue wiring during the C5b→C5f gap. |
| `test_audio_c7_byte_identical_c5b` | existing canary | Audio holds (helpers pure functions). |

**Commit gate.** Tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.
**Commit subject.** `C5b: central story_brief helpers (5 per refinement §5; music_mood is a pure helper here, wired into MusicGen at C5g per E-12)`

## C5c — FLUX env + radio bookend integration (~0.5 d)

**Review.** Refinement §6 — both consumers use `get_story_brief_full`.
**Code.** `visual/batch_flux_render.py` reads brief via `get_story_brief_full(meta)`, inserts between env description and style_suffix tail. Radio bookend uses brief to replace the weak `scenes[0].env` tier in the existing fallback chain. (File path retargeted at C1 audit per E-24 / RR-B8 contingency — `batch_flux_env_render.py` does not exist at HEAD; `batch_flux_render.py` is the actual env/bookend implementation file.)
**Wire.** None graph-level.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_flux_env_reads_brief_via_helper` | `tests/test_story_brief_flux_c5c.py` (new) | Source contains `get_story_brief_full` call. |
| `test_flux_env_golden_prompt_includes_brief_text` | same | Given known meta.story_brief, composed prompt contains the brief fragment verbatim. |
| `test_flux_env_empty_brief_fallback` | same | Empty brief: prompt builds without crashing; falls through to legacy env description. |
| `test_flux_env_receives_meta_story_brief` (E-06) | same | Inject `meta.story_brief = "zebra_lantern_atmosphere_731"`; run env-render prompt-build end-to-end; assert unique token appears verbatim in composed prompt. Behavioral, not source-grep. |
| `test_flux_bookend_receives_meta_story_brief` (E-06) | same | Same unique-token test against the radio bookend prompt-build path. |
| `test_flux_env_logs_story_brief_status_when_failed` (E-07) | same | Set `meta.story_brief = ""`, `meta.story_brief_status = "failed"`; run env-render; assert FLUX render log / report string contains substring `story_brief_status=failed`. Observable, not fatal. |
| `test_audio_c7_byte_identical_c5c` | existing canary | Audio holds. |

**Commit gate.** Tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.
**Commit subject.** `C5c: FLUX env + radio bookend read meta.story_brief via get_story_brief_full`

## C5d — FLUX portraits integration (~0.25 d)

**Review.** Refinement §6.2 — portraits narrow to `get_story_brief_lighting`.
**Code.** `visual/batch_flux_portrait_render.py`: append `get_story_brief_lighting(meta)` after character appearance description, before composition guidance.
**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_portraits_use_lighting_helper_not_full` | `tests/test_story_brief_portraits_c5d.py` (new) | Source uses `get_story_brief_lighting`; does NOT use `get_story_brief_full`. |
| `test_portraits_golden_prompt_no_setting_terms` | same | Given known meta, portrait prompt contains lighting + atmosphere terms; does NOT contain setting_terms entries. |
| `test_portrait_receives_meta_story_brief_terms` (E-06) | same | Inject `meta.story_brief_terms = {"lighting": ["zebra_lantern_atmosphere_731"], "atmosphere": ["tense"]}`; run portrait prompt-build; assert unique token in composed prompt. |
| `test_audio_c7_byte_identical_c5d` | existing canary | Audio holds. |

**Commit gate.** Tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.
**Commit subject.** `C5d: FLUX portraits read meta.story_brief via get_story_brief_lighting (no setting noise)`

## C5e — LTX motion integration (~0.5 d)

**Review.** Refinement §6.1: 220-240 char total; motion-first; drop brief if it pushes motion past char 140; 80-100 char brief fragment.
**Code.** `nodes/batch_ltx_render.py` `_build_ltx_role_prompt`: read `get_story_brief_ltx(meta, max_chars=90)`, append after motion-centric role template. Drop brief if motion-verb position would exceed char 140.
**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_ltx_prompt_total_under_240` | `tests/test_story_brief_ltx_c5e.py` (new) | All role prompts total ≤ 240 chars. |
| `test_ltx_motion_verb_before_char_140` | same | First motion verb appears at char index ≤ 140. |
| `test_ltx_brief_dropped_when_pushes_motion_past_140` (R-05 note: structural proxy only — empirical LTX motion fidelity is Sprint A scope per ROADMAP) | same | Long brief that would push motion past 140: brief is dropped, motion verb position holds. |
| `test_ltx_brief_fragment_80_to_100_chars` | same | When included, brief fragment is 80-100 chars and trimmed at clause boundary. |
| `test_ltx_receives_meta_story_brief_terms` (E-06) | same | Inject `meta.story_brief_terms` with unique token `zebra_lantern_atmosphere_731` (kept short so motion-first rule does not drop it); run per-role LTX prompt-build; assert unique token in composed prompt. |
| `test_ltx_logs_story_brief_status_when_failed` (E-07) | same | Set `meta.story_brief_status = "failed"`; assert LTX render log / batch report contains `story_brief_status=failed`. |
| `test_audio_c7_byte_identical_c5e` | existing canary | Audio holds. |

**Commit gate.** Tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.
**Commit subject.** `C5e: LTX motion reads meta.story_brief via get_story_brief_ltx (motion-first, drop-past-140 rule; structural proxy only)`

## C5f — HuMo lip-sync integration (~0.25 d)

**Review.** Refinement §6 — HuMo uses `get_story_brief_lighting`, appended before `_DEFAULT_POS_SUFFIX`.
**Code.** `nodes/batch_humo_render.py`: append `get_story_brief_lighting(meta)` before `_DEFAULT_POS_SUFFIX` in the per-clip prompt builder.
**Wire.** None.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_humo_reads_lighting_helper` | `tests/test_story_brief_humo_c5f.py` (new) | Source uses `get_story_brief_lighting`. |
| `test_humo_golden_prompt_preserves_default_pos_suffix` | same | Composed prompt still contains `_DEFAULT_POS_SUFFIX` content. |
| `test_humo_golden_prompt_includes_lighting_atmosphere` | same | Given known meta, composed prompt contains lighting + atmosphere terms. |
| `test_humo_receives_meta_story_brief_terms` (E-06) | same | Inject unique token `zebra_lantern_atmosphere_731` under `lighting`; run per-clip prompt-build; assert unique token in composed prompt. |
| `test_humo_logs_story_brief_status_when_failed` (E-07) | same | Set `meta.story_brief_status = "failed"`; assert HuMo render log / batch report contains `story_brief_status=failed`. |
| `test_audio_c7_byte_identical_c5f` | existing canary | Audio holds. |

**Commit gate.** Tests green. Bug Bible 23/1/2xf. Audio C7 holds. Forbidden sweep clean.
**Commit subject.** `C5f: HuMo lip-sync reads meta.story_brief via get_story_brief_lighting`

## C5g — MusicGen integration (~0.75 d, L-4 OVERRIDE per E-12 / operator directive)

**Review.** L-4 override (§1.1). E-12 (§1.2). R-04 superseded (§1.3). §4 hard rule 1 amended audio C7 boundary. §A.6 MusicGen coupling. Refinement §6.3 mood-vocab spec. C5b helper `get_story_brief_music_mood` (already shipped at C5b). The MusicGen wiring at C5g brings story-flavor into the final audio so brief, FLUX env, FLUX portraits, LTX, HuMo, AND MusicGen all consume `meta.story_brief*`. Every downstream artifact carries the story's unique flavor.

**Code.** `nodes/musicgen_theme.py` prompt construction reads mood via the C5b helper:

```python
# nodes/musicgen_theme.py -- inside the per-cue prompt builder.
from ._otr_story_brief_helpers import (
    get_story_brief_music_mood,
    get_story_brief_status,
)
# LLM slot: N/A -- MusicGen is an audio model, not an LLM call site.
_mood_terms = get_story_brief_music_mood(meta)  # list[str], possibly empty
_status     = get_story_brief_status(meta)      # 'ok' / 'failed' / 'absent'
if _status == "ok" and _mood_terms:
    # Prepend mood terms to the existing prompt. Preserves legacy
    # construction when the brief is absent or failed -- old episodes
    # and brief-failure runs fall through to current behavior.
    _mood_prefix = ", ".join(_mood_terms) + ", "
    prompt = _mood_prefix + prompt
else:
    # Status 'absent' (old ledgers per refinement §8.2) or 'failed'
    # (brief-pass crashed) -- prompt unchanged. Render log records
    # the status for observability (E-07 pattern).
    pass
log.info("[OTR_MusicGenTheme] story_brief_status=%s mood_terms=%s",
         _status, _mood_terms)
```

**Wire.** No new graph surface. No new widget. `OTR_MusicGenTheme` already receives `meta` from upstream via the ledger plumbing per `workflows/otr_scifi_16gb_full.json` (§A.6 routing). Confirmed via tools/validate_workflow_links.py.

**Audio C7 baseline reset (E-12).** This is the explicit reset event:
1. At C5g pre-commit: capture the OLD C7 baseline output one final time as `tests/fixtures/audio_c7_baseline_pre_c5g.wav.b3sum` (forensic).
2. Land the C5g code change.
3. Run the canary against the NEW MusicGen output. Capture as `tests/fixtures/audio_c7_baseline.wav.b3sum` (NEW canonical).
4. From C5g forward, `test_audio_c7_byte_identical_*` asserts against the new baseline.
5. C-final QA documents the baseline reset event with both b3sums and the C5g commit hash.

**Pytest.**

| Test | File | Asserts |
|---|---|---|
| `test_musicgen_reads_mood_helper` | `tests/test_story_brief_musicgen_c5g.py` (new) | Source: `nodes/musicgen_theme.py` imports `get_story_brief_music_mood`. |
| `test_musicgen_reads_status_helper` | same | Source: `nodes/musicgen_theme.py` imports `get_story_brief_status`. |
| `test_musicgen_prompt_prepends_mood_when_status_ok` | same | Given `story_brief_status="ok"` and `story_brief_terms.atmosphere=["tense","ominous"]` (both in vocab), composed MusicGen prompt starts with `"tense, ominous, "`. |
| `test_musicgen_prompt_unchanged_when_status_failed` | same | Given `story_brief_status="failed"`, composed prompt equals the legacy prompt byte-for-byte. |
| `test_musicgen_prompt_unchanged_when_status_absent` | same | Given a ledger with no `story_brief*` keys (old episode), composed prompt equals legacy. Closes refinement §8.2 backward-compat. |
| `test_musicgen_prompt_unchanged_when_mood_empty_intersection` | same | Given `atmosphere=["claustrophobic","smoke-filtered"]` (zero intersection with `_MUSIC_MOOD_VOCAB`), composed prompt equals legacy. Brief was 'ok' but the mood vocabulary filtered out non-musical terms. |
| `test_musicgen_receives_meta_story_brief_terms` (E-06) | same | Inject `meta.story_brief_status="ok"`, `meta.story_brief_terms.atmosphere=["tense","zebra_lantern_atmosphere_731"]`; assert the in-vocab term `"tense"` reaches the composed prompt; the non-vocab unique token does NOT (correctly filtered by helper). Canary that the helper-to-callsite plumbing is wired, not bypassed. |
| `test_musicgen_logs_story_brief_status` (E-07) | same | For each of ok/failed/absent, render log contains `story_brief_status=<state>`. |
| `test_musicgen_no_llm_call_added` | same | AST walk: `nodes/musicgen_theme.py` does NOT add any new LLM call site (no `make_generate_fn`, no `LLM_CACHE` access). MusicGen remains a pure audio model call site. |
| `test_audio_c7_new_baseline_captured` | same | `tests/fixtures/audio_c7_baseline.wav.b3sum` exists, is non-empty, and differs from the pre-C5g forensic b3sum. Confirms the reset event happened intentionally. |
| `test_c5g_audio_matches_legacy_when_brief_absent` (E-16 / RR-A2 — smuggled-regression isolation) | same | Fixture: ledger with no `story_brief*` keys (old-episode shape; `get_story_brief_status(meta) == "absent"`). Run the C5g pipeline end-to-end. Assert resulting audio b3sum is BYTE-IDENTICAL to `tests/fixtures/audio_c7_baseline_pre_c5g.wav.b3sum`. This proves the core MusicGen audio engine was NOT damaged during the wiring — the audio shift between pre-C5g and new-baseline is caused EXCLUSIVELY by the mood-prefix code path being active, not by an unintended regression elsewhere in the C5g diff. Without this test, the new-baseline assertion blesses any regression smuggled into the same commit. |
| `test_get_music_mood_has_exactly_one_production_caller_at_c5g` (E-28 / RR-B10) | same | AST-grep across `nodes/`, `visual/`, `scripts/` (excluding `tests/`): exactly ONE production caller of `get_story_brief_music_mood`, located in `nodes/musicgen_theme.py`. Flips the C5b zero-caller assertion. Pins the wiring contract: helper is consumed by exactly one production file, and that file is musicgen_theme.py. Any other production caller at this commit = wiring smuggled into the wrong consumer = STOP. |
| `test_audio_c7_byte_identical_c5g_new_baseline` | existing canary, retargeted | Audio matches the NEW baseline. Holds at C5g and every subsequent commit boundary. |

**Commit gate.** All tests green. Bug Bible 23/1/2xf. **Audio C7 baseline reset captured AND asserted against new b3sum.** Forbidden sweep clean. Workflow link validator clean. **New marker:** none (helper imports are wired, not forbidden).

**Commit subject.** `C5g: MusicGen reads meta.story_brief via get_story_brief_music_mood -- L-4 OVERRIDE per operator directive, audio C7 baseline intentionally reset, absent-brief isolation test (E-16) proves core engine intact`

## C-final — Sprint close (~0.5 d)

**Review.** Mirror prior sprint final QA format. Define post-C contract for Sprint A handoff.

**MusicGen no-touch gate AMENDED (E-09 amended by E-12 + E-20 rename-immune fix per RR-A4/RR-B7).** The original no-touch gate is partially relaxed: MusicGen-related files MAY change, but ONLY in commit C5g.

Before C-final commits, run BOTH checks:

**Check 1 — Rename-immune file-pattern gate (E-20 / RR-A4):**
```
git diff --name-only s34-p0-p1-hotfix...HEAD | grep -i musicgen
```
This catches any file path containing `musicgen` (case-insensitive) across the entire diff. Immune to file renames or relocations (e.g. `nodes/musicgen_theme.py` → `nodes/otr_musicgen_theme.py` during C3 would still be caught).

For each file path that appears, run:
```
git log --pretty=format:'%H %s' s34-p0-p1-hotfix..HEAD -- <path>
```
EVERY commit listed across all musicgen-related paths MUST have a subject prefix of `C5g:`. If any other commit prefix appears (e.g. `C3:`, `C5e:`), the one-commit exception was violated — STOP, revert the offending commit before C-final lands.

If `grep -i musicgen` returns empty, the sprint did not wire MusicGen — STOP, this contradicts E-12.

**Check 2 — Semantic-gate visual inspection (E-20 / RR-B7 partial accept):**
Manually inspect the C5g diff via `git show <C5g-hash>`. Confirm that the C5g commit is the ONLY commit changing:
- MusicGen prompt construction text
- MusicGen model arguments (`max_new_tokens`, `temperature`, etc.)
- MusicGen output duration parameters
- MusicGen cache-key composition
- Routing of `opening_audio` / `closing_audio` / `interstitial_audio`

These behaviors could be indirectly perturbed by changes elsewhere (shared prompt helpers, workflow JSON widget edits, test fixtures that alter cache keys). Full automation of this check is infeasible mechanically — it requires reading the C5g diff against the cumulative non-C5g diff. Operator-level visual confirmation is acceptable per E-20 / RR-B7's partial-accept disposition.

QA review doc restatement: "MusicGen integration with `meta.story_brief` was wired at C5g per operator directive 2026-05-15 (L-4 override). Audio C7 baseline reset at C5g; new canonical b3sum captured at `tests/fixtures/audio_c7_baseline.wav.b3sum`. Absent-brief isolation test (E-16) proves core audio engine intact. From C5g forward, the C7 canary asserts against the new baseline."

**Code.** `docs/<date>-sprint-c-final-qa-review.md`. ROADMAP refresh — Sprint C closed; Sprint A entry updated to reflect: (a) `meta.story_brief` stamped on every non-failed run; (b) `meta.story_brief_status` stamped on every run; (c) zero reads of `meta.ltx_style_brief` in `nodes/` + `visual/` (retired at C3b); (d) era literals retired at C2a/C2b across orchestrator + visual layers; `_STYLE_WORLD_BLOCK` is the new style-driven world surface; (e) audio C7 baseline reset at C5g per L-4 override (E-12); new b3sum at `tests/fixtures/audio_c7_baseline.wav.b3sum`; absent-brief isolation test (E-16) green; (f) MusicGen reads brief via `get_story_brief_music_mood` — story flavor reaches FLUX env, FLUX portraits, LTX, HuMo, AND MusicGen.

**Mandatory final QA section (E-27 / RR-B12 wording strengthened):**

> Runtime status: NOT PROVEN. Pytest-only structural pass; no ComfyUI Desktop run; no FLUX, LTX, HuMo, or MusicGen visual/audio render quality verified in Sprint C. Sprint A is the empirical verification pass for downstream artifact quality.
>
> Audio baseline note: Audio C7 baseline reset intentionally at C5g per operator directive (L-4 override, E-12). Pre-C5g forensic b3sum preserved at `tests/fixtures/audio_c7_baseline_pre_c5g.wav.b3sum`; new canonical at `tests/fixtures/audio_c7_baseline.wav.b3sum`. Absent-brief isolation test confirmed the audio shift is caused exclusively by the mood-prefix code path (E-16 / RR-A2 mitigation). Both b3sums retained for Sprint A regression posture.

**Wire / Pytest.** Wide pytest walk: confirm baseline + new test count.

**Commit gate.** All acceptance rows green. Audio C7 held against existing baseline C2a → C5f; reset at C5g; held against new baseline C5g → C-final; absent-brief isolation green at C5g. Runtime-status line in final QA. MusicGen rename-immune gate green AND semantic-gate visual inspection complete. Post-C contract for Sprint A documented. Branch pushed.

**Commit subject.** `C-final: Sprint C close -- meta.story_brief shipped (8 keys, 6 consumers including MusicGen), legacy retired at C2a/C2b/C3/C3b, audio C7 baseline reset at C5g with E-16 isolation guard, runtime NOT PROVEN (visual + audio empirical verification deferred to Sprint A)`

---

# §7. Acceptance table

| # | Check | Target |
|--:|---|---|
| 1 | Canonical pytest count | green |
| 2 | Wide pytest walk | baseline + ~85-105 new tests across C2a → C5g (was ~70-90 pre-v3; +C2 split test set, +C5a1/C5a2 reassignments, +token-stability test, +absent-brief isolation test, +eviction tests) |
| 3 | Bug Bible regression | 23 / 1 / 2 at every commit boundary |
| 4 | Audio C7 byte-identical (pytest proxy, default-config happy path) | holds against EXISTING baseline C2a → C5f; intentional reset at C5g; holds against NEW baseline C5g → C-final; absent-brief isolation test (E-16) green at C5g |
| 5 | Forbidden sweep | 0 runtime hits at every boundary |
| 6 | Refinement spec amended (C0b) — §3.5 (R-06) + §3.6 + §11.4 (L-3) | C0b |
| 7 | C1 staleness audit complete; FLUX env/bookend file path verified (E-24) | C1 |
| 8 | Era literals removed — visual layer (4 Python + 2 JSON via string-based replacement per E-19) | C2a |
| 9 | Era literals removed — orchestrator (rerank + dramaturg preamble + 5-pillar block replaced with `_STYLE_WORLD_BLOCK`) per E-13; `_STYLE_WORLD_BLOCK` preserves sound-first dramaturg discipline | C2b |
| 10 | Rerank-prompt era literal removed at `nodes/story_orchestrator.py:1596` (`"for a 1940s-style radio drama"` → `"for an audio drama"`); locked by `test_rerank_prompt_audio_drama_anchor_present` | C2b |
| 11 | `_GENRE_BY_STYLE` table + helpers + stamp + 3 fall-throughs deleted | C3 |
| 12 | `meta.visual_plan.genre` not stamped | C3 |
| 13 | Graceful absence: HUD + treatment render when `meta.visual_plan.genre` absent (E-05; renamed per E-26 / RR-B11) | C3 |
| 14 | `meta.ltx_style_brief` retired across `nodes/` + `visual/` + `workflows/` + `scripts/` (pulled forward per E-14) | C3b |
| 15 | `DEFAULT_LLM == "mistralai/Mistral-Nemo-Instruct-2407"` (regression-locked, L-1) | C4 |
| 16 | VRAM ceiling 14.5 + context cap 8192 (regression-locked) | C4 |
| 17 | `_run_with_timeout` non-blocking + cache invalidation (regression-locked, L-3) | C4 |
| 18 | `meta.story_brief` reflection pure module shipped | C5a1 |
| 19 | 8 meta keys per refinement §4 stamped (L-8) | C5a1 |
| 20 | Three SCOPED try/except arms per E-17 / RR-B3 — narrow blocks, not broad wrapper; AST-verified | C5a1 |
| 21 | Reflection entrypoint signature accepts ONLY `technical_fn` (no `creative_fn` param) per E-21 / RR-B1 | C5a1 |
| 22 | Repair pass: `min(reflection_temperature + 0.15, 0.55)` clamp per E-18 / RR-B5; `CRITICAL: <reasons>` prefix per R-06 | C5a1 |
| 23 | Brief length 180-260 preferred, 300 hard max | C5a1 |
| 24 | Writer wires reflection at K.5.5 with module-level import per E-22 / RR-B4 (Q1 locked, E-01) | C5a2 |
| 25 | Returned `script_json` output socket contains `meta.story_brief*` (E-02) | C5a2 |
| 26 | Returned `script_json` and saved-on-disk ledger contain identical `meta.story_brief*` (E-11) | C5a2 |
| 27 | Reflection does not mutate any non-`meta` ledger key (deep-dict, R-03) | C5a2 |
| 28 | Dual-slot OOM regression-guarded per E-15 / RR-A1 (revised at C1): peak VRAM ≤14.5 GB through composition→reflection in dual-slot config; `unload_llm(creative)` called before technical load by `request_slot`'s existing swap order; single-slot path emits no spurious unload | C5a2 |
| 29 | Reflection uses `technical_fn` not `creative_fn` — behavioral spy at writer-wiring layer (E-03, MOVED from C5a1 per E-21) | C5a2 |
| 30 | Reflection reuses `technical_model` cache entry — single load event spy on `LLM_CACHE` (R-01, MOVED from C5a1 per E-21) | C5a2 |
| 31 | 5 helpers per refinement §5 shipped; `music_mood` is a pure function at C5b, wired at C5g per E-12 | C5b |
| 32 | FLUX env + bookend read brief via `get_story_brief_full`; meta-threading canary green (E-06) | C5c |
| 33 | FLUX env + bookend log `story_brief_status=failed` when reflection failed (E-07) | C5c |
| 34 | FLUX portraits read brief via `get_story_brief_lighting` (no setting noise); meta-threading canary green | C5d |
| 35 | LTX motion reads brief via `get_story_brief_ltx`; motion-first + drop-past-140 rule (structural proxy only, R-05); meta-threading canary green; failure-status log visibility green | C5e |
| 36 | HuMo lip-sync reads brief via `get_story_brief_lighting`; `_DEFAULT_POS_SUFFIX` preserved; meta-threading canary green; failure-status log visibility green | C5f |
| 37 | MusicGen reads brief via `get_story_brief_music_mood`; status-gated fallback to legacy prompt when status≠'ok' or vocab intersection empty (E-12 / R-04 superseded) | C5g |
| 38 | Audio C7 baseline reset event: pre-C5g forensic b3sum + new canonical b3sum BOTH captured at `tests/fixtures/` | C5g |
| 39 | Absent-brief isolation test (E-16 / RR-A2): force `story_brief_status="absent"`, output byte-identical to pre-C5g forensic b3sum — proves core audio engine intact | C5g |
| 40 | MusicGen rename-immune one-commit-exception gate green: `git diff --name-only ... \| grep -i musicgen` shows ONLY commits with `C5g:` subject prefix (E-09 amended by E-12 + E-20) | C-final |
| 41 | MusicGen semantic-gate visual inspection complete: prompt text / model args / duration / cache key / audio routing changed ONLY in C5g (E-20 / RR-B7 partial accept) | C-final |
| 42 | New forbidden-sweep markers: `1940s` (C2a) + `1950s Americana` + `golden-age radio` + `Omni-Retro` + `Orson Welles` + `Norman Corwin` + `Lucille Fletcher` (C2b) + `_GENRE_BY_STYLE` (C3) + `meta.ltx_style_brief` (C3b) + reflection-fail-soft (C5a1) + `meta.story_brief == None` init (C5a2) | all |
| 43 | ROADMAP refreshed (Sprint C closed; Sprint A entry updated; MusicGen NOW wired, baseline reset at C5g noted) | C-final |
| 44 | Runtime status NOT PROVEN line in final QA explicitly enumerates visual + audio outputs per E-27 / RR-B12 | C-final |
| 45 | Audio baseline reset note in final QA (both b3sums + E-16 isolation result referenced) | C-final |
| 46 | `meta.story_brief` persists in saved ledger (`test_story_brief_persists_in_saved_ledger`) | C5a2 |
| 47 | `_run_with_timeout` non-blocking pattern preserved at every commit boundary | all |
| 48 | Orphan `SCRIPT_SYSTEM_PROMPT` + `SCAFFOLDING_PREAMBLE` constants deleted (no live consumers; no-legacy-back-compat rule applied per C2b path-A audit); locked by `test_scaffolding_preamble_constant_deleted` + `test_script_system_prompt_constant_deleted` runtime hasattr assertions | C2b |
| 49 | Orphan 5-pillar / dramaturg era literals (`1950s Americana`, `Orson Welles`, `Norman Corwin`, `Lucille Fletcher`, `golden age of radio`, `OMNI-RETRO CULTURAL COLLISION`, plus the four other pillar names) absent from runtime strings in `nodes/story_orchestrator.py`; locked by parametrized `test_literal_not_in_runtime_strings` | C2b |
| 50 | Workflow JSON edits use exact-string recursive replacement, not line numbers (E-19 / RR-B6) | C2a |
| 51 | `get_story_brief_music_mood` has zero production callers at C5b commit (E-28 / RR-B10) | C5b |
| 52 | `get_story_brief_music_mood` has exactly one production caller at C5g commit, located in `nodes/musicgen_theme.py` (E-28 / RR-B10) | C5g |

---

# §8. Out of scope (deferred)

- ~~**MusicGen integration with `meta.story_brief`**~~ — **NOW IN SCOPE at C5g per operator directive 2026-05-15 (E-12, L-4 override).** Brief flavor reaches MusicGen via `get_story_brief_music_mood`. Audio C7 baseline reset at C5g.
- **`DEFAULT_LLM` change to Gemma-4-E4B-it** — out per L-1. Picked up by a future baseline-reset sprint with soak + new audio fixtures.
- **Sprint A downstream verification + repair** — out, opens after Sprint C closes.
- **Sprint G comprehensive bug sweep** — out, queued after Sprint A.
- **Refinement §3.5 / §3.6 / §11.4 wording in the canonical doc** — amended at C0b; the design refinements doc remains the long-term spec.
- **Empirical LTX motion fidelity** — out per R-05. Char-counting tests in C5e are structural proxy only. Sprint A verifies actual LTX rendering.
- **Empirical MusicGen audio quality validation** — out per E-12. Structural tests in C5g prove the helper-to-callsite plumbing is wired; subjective audio quality is Sprint A scope.
- **Migration shim for old `meta.ltx_style_brief` ledgers** — out per R-07. Hard rule 2 + locked §6.6 forbid shims. Old ledgers fall through to `story_brief_status='absent'` gracefully; re-rendering regenerates the brief on the new path.
- **`_MINCED_OATHS` / `_FIRST_NAMES` / `_LAST_NAMES` data-pool restructuring** — out. Mixed-pool randomization at runtime keeps no single era dominant. Era-tag comments updated for clarity at C2; data arrays untouched.
- **v2.1+ candidate (deferred decision):** `artokun/comfyui-mcp` evaluation OR custom `/mcp-builder` comfyui-runner. Defer until after v1.9 ships and real iteration friction is measured. Until then: manual ComfyUI Desktop loading is the workflow. Don't build harness infrastructure speculatively.

---

# §9. Optional final-review reviewer instructions (paste into Gemini / ChatGPT)

> You are reading Sprint C v3 — the FINAL consolidated plan for a Windows-only, offline, RTX 5080 16 GB ComfyUI custom-node project. The plan has been through Cowork code-audit + Jeffrey pre-build edit pass (11 edits) + round-robin-2 critique (7 findings triaged) + operator directive 2026-05-15 v2 (E-12 MusicGen wired at C5g, E-13 C2 scope expanded, E-14 legacy retirements pulled forward) + round-robin-3 + round-robin-4 synthesis v3 (12 accepted findings lifted to E-15…E-27, 2 rejected with named rationale, C2 split into C2a/C2b for safety per operator sizing directive). All decisions are consolidated in §1. Code citations are inline in §A.1-§A.10 — you do not need repo access.
>
> Every L-series, E-series, R-series, and RR-series decision is RESOLVED. There are zero open questions. Pressure where you think one of them is wrong — but name it explicitly and cite the §1 row.
>
> Audit for: (1) load-bearing assumptions that could quietly break; (2) tests in §6 that would pass while broken — especially the meta-threading canaries, the returned-`script_json` test, the technical-slot behavioral spy (now at C5a2), the LLM-cache spy (now at C5a2), the deep-dict mutation guard, the C5a1 scoped-try AST walk, the C5a2 dual-slot eviction test, the C5g absent-brief isolation test, the C2b token-stability test, and the C2b `_STYLE_WORLD_BLOCK` sound-first-discipline test; (3) commit ordering across the 17-commit chain — particularly C2a/C2b ordering (does C2b really need to wait for C2a, or could they parallelize?), the legacy-first block (C2a/C2b/C3/C3b), whether C5a1/C5a2/C5b ordering is right (RR-A5 was rejected — does the rejection hold under fresh adversarial pressure?), and whether C5g (MusicGen) belongs LAST in the new-system block; (4) anything in §A.1-§A.10 that contradicts a plan claim; (5) the AMENDED MusicGen one-commit-exception gate at C-final — `grep -i musicgen` is now rename-immune; is there a remaining sneaky indirect coupling path the semantic-gate visual inspection would miss?; (6) scope creep or scope omission across C2a/C2b/C3/C3b/C5a1-C5g; (7) the audio C7 baseline reset event at C5g — is the absent-brief isolation test (E-16) sufficient to falsify the smuggled-regression class, or could a regression be specifically gated on `status=="ok"` and still slip through?; (8) the dual-slot eviction guard at C5a2 — does `evict_model` actually exist in `_otr_model_loader.py` at HEAD, and if not, what is the safe fallback?; (9) the C2b token-stability test — is +15% a realistic expansion ceiling, or too loose / too tight?
>
> Severity-tag every finding (HIGH / MEDIUM / LOW). List concrete additions, deletions, splits, or kills. Do not be nice. Do not summarize. Do not pad.
>
> If you find a HIGH that would invalidate an L-series / E-series / R-series / RR-series decision, name it explicitly. This plan is READY-TO-CUT v3.

---

# §10. Sources

- `docs/2026-05-12-story-brief-v2-research.md` — canonical inventory (pre-C0b amendment).
- `docs/2026-05-12-story-brief-v2-design-refinements.md` — canonical refinements (with C0b amendment to §3.5 + §3.6 + §11.4 pending at sprint open).
- `docs/2026-05-13-story-brief-v2-go-forward-plan.md` — historical input only; superseded by this plan.
- `docs/2026-05-15-sprint-c-cowork-review-findings.md` — first-pass Cowork code audit.
- `docs/2026-05-15-sprint-c-story-brief-v2-plan-v2.md` — superseded.
- `docs/2026-05-15-sprint-c-plan-v3-roundrobin.md` — superseded.
- `docs/2026-05-15-S34-final-qa-review.md` — parent sprint close.
- `ROADMAP.md` — Sprint C scope section + S21.1 close + S30 close + S31 close + S34 close.
- BUG-LOCAL-228 entry in `docs/BUG_LOG.md` — the timeout-recovery contract referenced by L-3 and §A.8.
