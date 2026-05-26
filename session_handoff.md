# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25 (post Sprint 7C close + brief audit)

## Core goal

Wire the post-script `meta.story_brief` reflection into every downstream creative prompt so it becomes a load-bearing artifact instead of decoration. The 2026-05-25 live run proved the music path was reading nothing useful from the brief; the audit landed this session classifies every consumer and lays out the per-commit wiring order. The v4 sprint plan is now COMPLETE; the brief consumer wiring is the next sprint ("Sprint 8.x" by audit convention -- rename if Jeffrey prefers another label).

## Tech stack & constraints

OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md + BUG_LOG.md + the two new docs at repo root (`downstream_brief_consumer_followup.md` + `downstream_brief_consumer_audit.md`) auto-load -- do not repeat their contents here. Operational notes:

- **HEAD `2c36f1e` on `v2.0-alpha`, pushed.** Clean working tree except the parked `docs/s28_diff_tmp.txt` (NEVER commit it).
- **Cowork Linux sandbox `Bash` mount is STALE for this repo** -- its `git status` shows phantom modified files. Use Desktop Commander (`shell: cmd.exe`) for git + tests. Read/Write/Edit file tools write the real Windows file correctly.
- Venv python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
- Full OTR suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~32 s, 2856 tests). Bug Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`. LLM-slot sweep: `<venv python> docs\_s28_llm_slot_sweep.py` (must exit 0 with `23 tagged, 0 untagged, 0 parse failures`).
- Redirect long runs to a log file (`> %TEMP%\x.log 2>&1`) and read the file -- a single long `interact_with_process` call drops the cmd session at the MCP timeout.
- `cmd` gotchas: commit messages via the file tool to `.git\COMMIT_EDITMSG`, then `git commit -F`. Never `git commit -m`.
- LLM backend is HF Transformers 5.5.0 (`model.generate()`). No llama.cpp, no GBNF.
- Run Bug Bible + full OTR suite + LLM-slot sweep after every code change, unprompted.
- Live-run is healthy. Do NOT pause for validation between items inside a sprint -- build, commit, push, move on. Eyes-on review is needed only for workflow JSON changes (PD3) or non-obvious root cause / diff decisions.

## What's done & decided

- **Sprint 7C COMPLETE (`91007e7`, pushed).** BUG-LOCAL-275 fixed. Added `payload_null_repair` typed-prompt factory in `nodes/_otr_repair_prompts.py`; `make_dispatching_repair_factory` now detects pydantic `ValidationError` reprs containing BOTH `payload` (field path) AND `input_value=none` (annotation) and routes them BEFORE falling through to the generic `schema_field_repair`. Two detection-token constants (`_PAYLOAD_NULL_FIELD_TOKEN` / `_PAYLOAD_NULL_VALUE_TOKEN`) document the signal so a future pydantic version reshape degrades to schema_field_repair rather than crashing. Schema stays strict (`ReviewerEdit.payload: str`) -- option (a) widen-to-Optional was rejected because it would mask intentional doctor mistakes. Regression: `tests/test_repair_prompts.py` 18 -> 21 (+3: builder test, dispatcher routing, specificity guard); full OTR 2856 / 21 / 0; Bug Bible 16 / 7 / 3 xfailed / 0; LLM-slot sweep 23/23.
- **v4 plan COMPLETE.** Sprints 0, 1, 2A-2E, 3A-3G, 4, 5A-5C, 6, 7A, 7B, 7C all closed. Status Board and the stale Sprint 7A/7B/7C sub-section checkboxes were updated in `story_pipeline_sprint_plan_v4_audited.md` in the same commit. The only carried gate is the operator live-run on HEAD `2c36f1e` (Prime Directive 1, operator-owned).
- **Downstream brief consumer audit DONE (`2c36f1e`, pushed).** `downstream_brief_consumer_audit.md` at repo root classifies every consumer: **1 C** (MusicGenTheme -- reads `story_brief_terms.atmosphere` directly in `_compose_music_prompt`, but the field carries visually-tuned values not music-tuned; `get_story_brief_music_mood` over-filters via a 16-word vocab and reports empty), **6 A** (LTX prose, FLUX env prose, FLUX portrait lighting, FLUX radio prose, HuMo lip-sync lighting, OTR_VideoPlan era tail), **0 B**, **D-class candidates declined** (title path has Sprint 3E rich grounding; style picker + upstream creative passes are causal blocks -- they run pre-brief).
- **Canonical "big plan" doc landed at repo root.** `downstream_brief_consumer_followup.md` (same commit `2c36f1e`) holds the wiring sprint plan with a live "State of the work" status table, the original nested-object schema proposal preserved verbatim, and 5 open decisions (A-E) listed at the bottom. **Update protocol pinned:** append `## Update <date>` sections; status table at top is the only authoritative live view; the audit doc is a snapshot pinned at HEAD `91007e7` and never edited in place. Re-snapshot at the next live-run anchor.
- **Decisions explicitly made + rejected this session (do NOT reopen):**
  - **Sprint 7C option (a)** (widen `ReviewerEdit.payload` to `Optional[str]`) -- REJECTED. Schema stays strict; typed repair recovers.
  - **Title scratchpad as D-class** -- DECLINED. Sprint 3E excerpt grounding already covers the title path; brief would be redundant.
  - **Schema shape A2** (nested `meta.story_brief.<field>` object replacing the prose-as-string) -- AUDIT RECOMMENDED AGAINST. Flagged as open decision A; flat additive (A1) is the audit-recommended path because it keeps every v1 A-class consumer working with zero rename. Final call is Jeffrey's.
- **No new bug ids opened this session.**

## State of the art

- **HEAD `2c36f1e`** == `origin/v2.0-alpha`. Two commits this session: `91007e7` (Sprint 7C code + tests + BUG_LOG + sprint plan housekeeping), `2c36f1e` (the two new docs at repo root).
- **Working tree clean** except the intentionally parked `docs/s28_diff_tmp.txt`.
- **Regression baseline at HEAD `2c36f1e`:** full OTR suite **2856 passed / 21 skipped / 0 failed**; Bug Bible 16 passed / 7 skipped / 3 xfailed / 0 failed; LLM-slot sweep 23/23 tagged, 0 parse failures.
- **Files touched this session (code):** `nodes/_otr_repair_prompts.py` (+~90 lines: new `payload_null_repair` factory, dispatcher predicate `_is_payload_null_validation_error`, two named detection-token constants, updated module docstring and `__all__`); `tests/test_repair_prompts.py` (+~90 lines: new `_payload_null_validation_error` fixture, three new tests, extended `test_every_builder_returns_single_user_message` to seven builders, coverage-map docstring update 18 -> 21 tests).
- **Files touched this session (docs):** `BUG_LOG.md` (BUG-LOCAL-275 [LOGGED] -> [FIXED]; header Last/Prior entries reshuffled), `story_pipeline_sprint_plan_v4_audited.md` (Sprint 7C row -> COMPLETE; stale 7A/7B/7C checkboxes flipped with completion pointers; new Build Progress Log entry; Sprint 7 completion gate 4/5 ticked), `downstream_brief_consumer_followup.md` (NEW at repo root -- canonical big-plan doc), `downstream_brief_consumer_audit.md` (NEW at repo root -- classification snapshot).
- **Three Prime Directive 6 invariants verified post-commit:** LLM-slot sweep stays at 23/23 tagged (no LLM call added or removed by 7C or the docs); the workflow JSON is untouched (PD3 N/A); the brief consumer wiring sprint is designed to keep all four invariants holding (additive schema, no new LLM call, no node surface change).

## Immediate next steps

1. **Resolve open decisions A-E in `downstream_brief_consumer_followup.md`.** Jeffrey-gated. Decision A is the load-bearing one (schema shape -- audit recommends A1 flat additive). Decisions B (helper API dotted-path vs flat-key), C (sequencing -- helper+commit1 together vs helper-first), D (atmosphere naming under A1 -- recommend `meta.atmosphere_line`), E (Bark/TTS deep audit scope -- before commit 1 or after visual wiring) follow.
2. **Sprint 8.1 -- producer v2 + reader helper + MusicGenTheme rewire (commit 1).** Per audit recommendation: producer v2 schema add in `nodes/_otr_story_brief.py` (`StoryBriefModel` gains the new fields with safe defaults; reflection prompt body extended to ask for them; `_success_delta` / `_failure_sentinel` extended; `_PROMPT_VERSION` bumped v1 -> v2); new pure module `nodes/_otr_brief_reader.py` exposing `_read_brief_field(meta, field_name, default)`; `nodes/musicgen_theme.py` `_compose_music_prompt` rewires to read `music_mood_terms` first with the existing atmosphere-as-mood path as the v1 fallback; per-commit tests in `tests/test_brief_reader_musicgen.py` (or similar) + extend `tests/test_story_brief_c5a1.py` for the v2 producer fields.
3. **Sprint 8.2-8.7 -- one commit per A-class consumer.** Audit-recommended order: FLUX env (`_parse_env_prompts` -- `visual_palette` + `key_objects`), FLUX portrait (`_build_portrait_prompt` -- `visual_palette` + `atmosphere_line`), FLUX radio bookend (`_build_dynamic_radio_prompt` -- `visual_palette`), LTX motion (`_build_motion_prompt` -- `tempo_hint`), HuMo lip-sync (`_build_pos_prompt` -- `atmosphere_line`), OTR_VideoPlan (`_resolve_era_tail` -- `visual_palette` + `atmosphere_line`). Each commit: rewire consumer through `_read_brief_field`, add a regression test, run Bug Bible + LLM-slot sweep + full OTR suite.
4. **Sprint 8.8 -- Bark / TTS deep audit (carry-forward from this session).** Confirm or classify whether the Bark voice-render path consumes anything from the brief; if A-class, no commit; if B/C, add a wiring commit in the same per-consumer pattern.
5. **Operator live-run reversion gate (PD1).** Operator-owned. One ComfyUI episode on the post-Sprint-8.1 HEAD must show `[OTR_MusicGenTheme] mood_terms=[<non-empty list>]` in the console; later sprints ride opportunistic live runs.

## Open questions

Five open decisions for Jeffrey, all recorded in `downstream_brief_consumer_followup.md` section "Open decisions":

- **A. Schema shape.** A1 (flat additive top-level `meta.*` fields, audit-recommended, zero v1 breakage) vs A2 (nested `meta.story_brief.<field>` object, original draft, forces a v1 prose-key rename).
- **B. Reader helper API.** Dotted-path (`"story_brief.music_mood_terms"`) vs flat-key (`"music_mood_terms"`). Cosmetic under A1, load-bearing under A2.
- **C. Sequencing.** Commit 1 = producer v2 + reader + MusicGenTheme (C1, larger diff, reader's first real caller proves it works) vs reader-first as a standalone commit (C2, smaller diffs, reader sits unused for one commit).
- **D. `atmosphere_line` naming under A1.** Recommend `meta.atmosphere_line` (or `meta.story_brief_atmosphere_line`) to avoid colliding with v1's `story_brief_terms.atmosphere` list.
- **E. Bark / TTS audit timing.** Before Sprint 8.1 (audit completion gate) or after the visual wiring is done.

No other blockers. The Sprint 7C live-run signal (the doctor recovers from `payload: null` via the typed repair) and the brief-consumer wiring signal (`mood_terms=[<non-empty>]`) are both observable on the SAME operator live-run if Jeffrey chooses to bundle them.

---

## Resume instructions

Open a fresh window, attach this file, and say:

"Read this handoff file and prepare to execute the immediate next steps. Acknowledge when you're ready to start."
