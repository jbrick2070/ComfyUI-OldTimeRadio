# LFC sprint — Multi-Turn Polish QA Handoff

**Branch:** `v2.0-alpha`
**HEAD at handoff:** `80b4244`
**Sprint start:** `c3f5e3b` (composer Tier 3 forward-plan)
**Date:** 2026-05-11
**ADR:** `docs/2026-05-11-multi-turn-polish-adr.md`
**Problem statement:** `docs/2026-05-11-multi-turn-polish-problem-statement.md`

---

## 0. What landed vs. what's deferred

The ADR sprint plan called for 14 commits. **10 commits landed** on
`v2.0-alpha`; **4 commits are deferred to a fresh session** with
detailed scaffolding in place.

| Commit | Description | Status |
|--------|-------------|--------|
| 1 | Phase 0 + Phase 10 scaffolding (gap audit + freeze + null-rejection) | **LANDED** `64a818f` |
| 2 | Reviewer rename + freeze-cascade orchestrator skeleton | **LANDED** `ba4d7c5` |
| 3 | Code fixes 6.1-6.4 (announcer guard + refusal detector + polish context + separate generate_fn) | **LANDED** `edc2086` |
| 4 | Phase 3 per-line polish wired into the cascade | **LANDED** `5f6ceb1` |
| 5 | Phase 7 audio readiness + Phase 8 video readiness | **LANDED** `b6278e4` |
| 6 | Two-step gen + JSON repair loop helpers (ADR 6.11 / 6.12) | **LANDED** `7e8ea6f` |
| 7 | Macro-Bible + Micro-Recap context injection (ADR 6.13) | **LANDED** `50ad53e` |
| 8 | Phase 4 per-scene coherence (LLM) | **DEFERRED** |
| 9 | Phase 5 per-speaker voice drift (LLM) | **DEFERRED** |
| 10 | Phase 6 episode arc (LLM) | **DEFERRED** |
| 11 | Phase 4.5 Smart Suggestion (deterministic SFX/music synthesis) | **LANDED** `b07a6b6` |
| 12 | Code fixes 6.5-6.10 (VRAM watchdog + voice-drift stats + per-phase widgets + workflow JSON wiring) | **LANDED** `80b4244` |
| 13 | Acceptance test for script_parse_json retirement | **LANDED** `03bbcf1` |
| 14 | Soak test harness | **DEFERRED** |

Why deferred: commits 8 / 9 / 10 are the heavy LLM phases. Each
needs careful prompt design, scene-boundary edge cases, edit-cap
tuning, and 20-30 test fixtures. Shipping them rushed would be worse
than shipping them carefully. The orchestrator stubs from commit 2
are still in place (`_phase_4_per_scene_coherence_stub`,
`_phase_5_voice_drift_stub`, `_phase_6_episode_arc_stub` in
`nodes/_otr_freeze_cascade.py`); commits 8 / 9 / 10 replace each stub
in place and wire the corresponding widget toggle. The LLM-side
infrastructure (two-step gen + JSON repair + Macro-Bible /
Micro-Recap) is ready and tested.

The Bug Bible regression baseline held across the entire sprint
(23 passed / 1 skipped / 2 xfailed at the most recent run, against
the user's stated 16/7/3xf reference). No new Bug Bible entries
required.

---

## 1. Per-commit summary (what each commit did)

### Commit 1 — `64a818f` — Phase 0 + Phase 10 scaffolding

- **New module:** `nodes/_otr_ledger_freeze.py`
- Phase 0 (`phase_0_gap_audit_pre`) is warn-mode, never raises.
  Stamps `meta.gap_audit_pre`.
- Phase 10 (`phase_10_gap_audit_post_and_freeze`) hard-asserts on
  critical gaps. Stamps `meta.cleanup_locked = True` +
  `meta.freeze_timestamp` + `meta.freeze_verdict`.
- ADR §7 invariants: per-line / per-cast / meta / null-rejection.
  §6.16 null-rejection adapted to live L3 schema (`sfx_cues` /
  `music_cues` ADR mentions were speculative; reality has scalar
  `sfx_cue` on beats + top-level `sfx` / `music` lists).
- **Tests:** `tests/test_lfc_phase_0_10_gap_audit.py` — 59 cases.

### Commit 2 — `ba4d7c5` — Reviewer rename + orchestrator skeleton

- **Renamed node:** `OTR_LedgerScriptReviewer` → `OTR_LedgerFreezeCascade`.
- **Back-compat:** old class name preserved via NODE_CLASS_MAPPINGS
  rename alias in `__init__.py`; old module file
  (`OTR_LedgerScriptReviewer.py`) rewritten as a re-export shim.
- **5-slot output contract preserved.** 5th slot renamed
  `reviewer_verdict` → `freeze_verdict`.
- **Orchestrator** (`nodes/_otr_freeze_cascade.py`) wires Phase 0 →
  existing 3-pass reviewer → Phase 10. Phases 3/4/4.5/5/6/7/8 are
  no-op stubs in this commit.
- **Workflow JSON updated:** node id 62 type renamed; title
  "1b. Ledger Freeze Cascade (Phase 0..10)"; output socket 4 renamed.
- **Tests:** `tests/test_lfc_freeze_cascade_orchestrator.py` — 17 cases.

### Commit 3 — `edc2086` — Code fixes 6.1-6.4 (polish path)

- **§6.1:** `_POLISH_SYSTEM_PROMPT_ANNOUNCER` for announcer-aware
  polish. `polish_line` takes `speaker_role` keyword.
  `LineRequest.speaker_role` defaults to `"character"`.
- **§6.2:** `_REFUSAL_PATTERNS` + `is_polish_refusal()`. `polish_line`
  returns pre-polish text on refusal-regex hit. Verb whitelist
  (rewrite/help/comply/produce/etc.) anchors the match so
  in-character "I cannot believe…" doesn't false-trigger.
- **§6.3:** `polish_line` takes `beat_intent` + `previous_lines`
  keywords. Renders BEAT INTENT + PREVIOUS LINES blocks in the user
  prompt. previous_lines capped at last 2.
- **§6.4:** `_otr_model_loader.make_polish_generate_fn` factory.
  Separate closure off cache_entry with polish-conservative sampling
  (top_p=0.9, no min_p, no repetition_penalty). `polish_line`
  accepts an optional `polish_generate_fn` keyword.
- **Tests:** `tests/test_lfc_polish_fixes.py` — 38 cases.

### Commit 4 — `5f6ceb1` — Phase 3 per-line polish in cascade

- `_phase_3_per_line_polish` replaces the commit-2 stub.
- Scans the post-writer ledger; polishes any line that trips
  `needs_polish`. Uses commit-3 polish_line surface (announcer
  guard + refusal detector + context + dedicated generate_fn).
- **Default OFF** (`enable_phase_3_polish=False`). Cascade-side
  polish stays gated until soak validates interaction with the
  composer's inline polish path.
- Maintains a rolling 2-line PREVIOUS LINES window of dialogue-only.
- Rejection paths counted on `Phase3PolishReport`: refusal /
  still-leaky / word-cap miss.
- **Tests:** `tests/test_lfc_phase_3_polish_in_cascade.py` — 20 cases.

### Commit 5 — `b6278e4` — Phase 7 + Phase 8 readiness

- **New module:** `nodes/_otr_readiness.py`.
- **Phase 7 (audio readiness):** ABBREV_EXPANSIONS (Dr. → Doctor,
  etc.), SYMBOL_REPLACEMENTS (& → and, etc.), num2words integer
  expansion. Recomputes word/char counts in lockstep.
- **Phase 8 (video readiness):** pure audit — checks each cast row
  for portrait_path / portrait_image / portrait / image_path keys.
  Stamps `meta.video_readiness` with missing-portrait list.
- **Both default ON** — deterministic, cheap.
- **Tests:** `tests/test_lfc_phase_7_8_readiness.py` — 24 cases.

### Commit 6 — `7e8ea6f` — Two-step gen + JSON repair helpers

- **New module:** `nodes/_otr_lfc_llm_helpers.py`.
- `two_step_generate(reasoning_fn, formatter_fn, ...)`: ADR §6.11
  free-reasoning → structured-formatting decoupling. Formatter fn
  optional — falls back to reasoning fn with a telemetry log tag
  (`FORMATTER_REASONING_REUSE_LOG`).
- `parse_with_repair(raw, schema, generate_fn)`: ADR §6.12 regex
  fast-path (markdown fences, trailing prose, trailing commas,
  smart quotes) + LLM repair loop (up to 3 attempts) on
  ValidationError.
- **Schemas:** `ReviewerEdit`, `ReviewerOutput`, `EditorNote`,
  `EditorNotesOutput`. Pydantic shapes used by Phase 4/5/6.
- **Tests:** `tests/test_lfc_llm_helpers.py` — 23 cases.

### Commit 7 — `50ad53e` — Macro-Bible + Micro-Recap

- **New module:** `nodes/_otr_lfc_context.py`.
- `build_macro_bible(led, token_cap=2500)`: episode logline + cast
  voice cards + allowed entities + arc skeleton + world rules.
- `build_micro_recap(led, scope, target, token_cap=4500)`: per-call
  context for `scope="scene"` / `"speaker"` / `"episode"`.
- `scenes_from_lines(ledger_data)`: ADR Q5 scene partitioning
  (music_inter dividers, single-scene fallback).
- `estimate_tokens(text)`: conservative 1/4 chars-per-token
  heuristic. Truncation snaps to nearest line/sentence boundary.
- **Tests:** `tests/test_lfc_context_helpers.py` — 29 cases.

### Commit 11 — `b07a6b6` — Phase 4.5 Smart Suggestion

- **New module:** `nodes/_otr_lfc_smart_suggestion.py`.
- SFX_VERB_PATTERNS (16 regex → tag pairs) + MUSIC_VERB_PATTERNS
  (4 regex → MusicGen-shape tag pairs).
- Synthesizes new beats with `auto_generated=True`. Mutation
  policy: only this LFC phase adds lines; existing line.text /
  char_id / speaker_role / line_id untouched.
- Per-scene SFX dedupe + ledger-wide music_inter dedupe.
- **Default OFF** per ADR §6.17. Promote to mandatory once soak
  validates the allow-list.
- **Tests:** `tests/test_lfc_phase_4_5_smart_suggestion.py` — 14 cases.

### Commit 12 — `80b4244` — Code fixes 6.5-6.10 + widget surface

- **New module:** `nodes/_otr_lfc_watchdog.py`.
- `compute_speaker_stats(lines) -> dict[char_id, SpeakerStats]`:
  ADR §6.5 voice-drift detection. mean_line_length +
  vocab_diversity (Type-Token Ratio).
- `flag_line_drift(text, stats)`: returns (drifted, reason) per
  ADR 40%/60% thresholds.
- `vram_over_ceiling(ceiling_gb=14.0)`: ADR §6.8 watchdog. ALARM
  PLUMBING ONLY (per Jeffrey's no-VRAM-dragons memory) — no
  quantization/streaming/FA chasing.
- **Cascade widget surface (ADR §6.9):**
  - `enable_phase_3_polish` BOOLEAN default False
  - `polish_announcer_beats` BOOLEAN default False
  - `enable_phase_4_5_smart_suggestion` BOOLEAN default False
  - `enable_phase_7_audio_readiness` BOOLEAN default True
  - `enable_phase_8_video_readiness` BOOLEAN default True
  - `vram_ceiling_gb` FLOAT default 14.0
- **Workflow JSON updated:** widgets_values extended from 1 → 7
  positional slots. polish_generate_fn now built off cache_entry
  via `make_polish_generate_fn` (ADR §6.4 wiring).
- **Tests:** `tests/test_lfc_watchdog_and_widgets.py` — 24 cases.

### Commit 13 — `03bbcf1` — script_parse_json retirement test

- **New test file:** `tests/test_legacy_contract_retired.py`.
- ADR §10 acceptance test. Greps `nodes/`, `tests/`, `otr_v2/`,
  `visual/` for code references to `script_parse_json`. Comments +
  docstrings tolerated. Allow-list: acceptance test itself + ADR +
  problem statement.
- **Tests:** 3 cases.

---

## 2. New widget surface — positions in `widgets_values`

`workflows/otr_scifi_16gb_full.json` → node id 62
(`OTR_LedgerFreezeCascade`) → `widgets_values`:

| Index | Widget name | Type | Default | Notes |
|-------|-------------|------|---------|-------|
| 0 | `model_id` | STRING | `"mistralai/Mistral-Nemo-Instruct-2407"` | Reused from old reviewer |
| 1 | `enable_phase_3_polish` | BOOLEAN | `false` | LFC Phase 3 |
| 2 | `polish_announcer_beats` | BOOLEAN | `false` | LFC Phase 3 sub-toggle |
| 3 | `enable_phase_4_5_smart_suggestion` | BOOLEAN | `false` | LFC Phase 4.5 |
| 4 | `enable_phase_7_audio_readiness` | BOOLEAN | `true` | LFC Phase 7 |
| 5 | `enable_phase_8_video_readiness` | BOOLEAN | `true` | LFC Phase 8 |
| 6 | `vram_ceiling_gb` | FLOAT | `14.0` | ADR §6.8 |

`RETURN_NAMES` (output sockets, unchanged shape; slot 4 renamed):
`("script_text", "script_json", "news_used", "estimated_minutes", "freeze_verdict")`

---

## 3. Pitfalls flagged for QA

1. **Bash sandbox vs Windows file view.** The sandbox shows
   `_otr_line_composer.py` at 934 lines when Windows has it at
   1614+. Trust the **Windows-canonical Read tool** for file
   contents; use **Desktop Commander cmd shell** for git operations.
   The bash sandbox lies about both. (Caught at session start;
   memory `feedback_bash_sandbox_can_truncate_files.md` confirmed.)

2. **Phase 4 / 5 / 6 LLM phases are STUBS.** The orchestrator calls
   `_phase_4_per_scene_coherence_stub`,
   `_phase_5_voice_drift_stub`, `_phase_6_episode_arc_stub` — all
   no-ops with a debug log. Run a live episode and inspect
   `meta.cleanup_passes` — you'll see Phase 0, 3, 4.5, 7, 8, 10
   real records + the 1+2+9 reviewer composite. Phase 4/5/6 are
   NOT in the meta record because they're stubs (intentional).

3. **The polish closure-leak fix is opt-in.** The cascade node
   builds `polish_generate_fn` via `make_polish_generate_fn` if
   available, then threads it through `run_freeze_cascade`. But
   the composer's inline polish path (when `enable_polish_pass=True`
   on the writer node) still uses the writer's main generate_fn
   with the closure leak. The proper fix is to route the writer's
   inline polish through `make_polish_generate_fn` too — that's a
   separate writer-side change not in this sprint.

4. **Phase 7 num2words is an optional dep.** If `num2words` isn't
   installed, the numeric-expansion step is skipped with a warning
   stamped on `meta.audio_readiness.warnings`. The other Phase 7
   expansions (abbreviations + symbols) still fire. `pip install
   num2words --break-system-packages` if you want the numeric path.

5. **Workflow JSON `widgets_values` is positional.** If you re-save
   the workflow in the ComfyUI Desktop GUI, double-check the order
   of values is `[model_id, enable_phase_3_polish,
   polish_announcer_beats, enable_phase_4_5_smart_suggestion,
   enable_phase_7_audio_readiness, enable_phase_8_video_readiness,
   vram_ceiling_gb]`. Any reorder breaks the cascade node's
   positional mapping.

6. **Phase 4.5 Smart Suggestion mutates `lines` array length.**
   This is the ONLY LFC phase that adds entries. Synthesized SFX /
   music beats land at the END of the scene they originated in,
   tagged `auto_generated=True`. Downstream consumers that assume
   strict input-line-count parity with the writer's output will
   see drift when Phase 4.5 is enabled.

7. **The Bug Bible baseline in the prompt was 16/7/3xf.** Actual
   current state is 23/1/2xf (more entries shipped between the
   ADR and this sprint). No regressions — but the prompt baseline
   is stale.

---

## 4. 5-step manual QA recipe in ComfyUI Desktop

1. **Pull + restart ComfyUI.** Pull `v2.0-alpha` to HEAD `80b4244`
   on Windows. Stop ComfyUI Desktop, start fresh:
   `localhost:8000`. Confirm the boot log has
   `[OldTimeRadio] OK - All N nodes loaded successfully` with no
   "Skipped" warnings.
2. **Load `workflows/otr_scifi_16gb_full.json`.** Confirm node
   id 62 is "1b. Ledger Freeze Cascade (Phase 0..10)". The node
   should show 7 widgets in this order: `model_id`,
   `enable_phase_3_polish`, `polish_announcer_beats`,
   `enable_phase_4_5_smart_suggestion`,
   `enable_phase_7_audio_readiness`,
   `enable_phase_8_video_readiness`, `vram_ceiling_gb`.
3. **Default-config smoke run.** Queue a writer run with the default
   widget values (all new toggles at their defaults — Phase 3 / 4.5
   OFF; Phase 7 / 8 ON; VRAM ceiling 14.0). The cascade should
   complete with `freeze_verdict` of `frozen_clean` or
   `frozen_with_warns`. Inspect `meta.cleanup_passes` in the
   ledger — expect entries for `phase_0_gap_audit_pre`,
   `phase_3_per_line_polish` (no-op record),
   `phase_1_2_9_reviewer_composite`,
   `phase_4_5_smart_suggestion` (no-op record),
   `phase_7_audio_readiness`, `phase_8_video_readiness`,
   `phase_10_gap_audit_post_and_freeze`.
4. **Phase 7 audio normalization check.** Inspect
   `meta.audio_readiness` on the ledger. `lines_scanned` should
   match the voiced-line count. `lines_normalized` ≥ 0 (typically
   1-3 lines per episode get abbreviation / symbol / number
   expansion). Spot-check a couple of voiced lines — any "Dr."
   should now read "Doctor", any "&" should read " and ", etc.
5. **Phase 3 polish opt-in check.** Queue a second run with
   `enable_phase_3_polish=True`. The cascade should make additional
   polish LLM calls on any leaky line; inspect
   `meta.phase_3_polish_record` for `edits_applied` counts. If 0
   edits despite leaky lines, check the polish refusal log — the
   model may be refusing on the corpus you fed it (Phase 3 returns
   pre-polish text on refusal, which is the correct fail-soft
   behavior).

---

## 5. File list — round-robin reading guide

For the next round-robin (ChatGPT + Gemini + Claude), feed these
files:

**Core cascade infrastructure (ship-blocking) —**
- `nodes/_otr_ledger_freeze.py` (Phase 0 + Phase 10)
- `nodes/_otr_freeze_cascade.py` (orchestrator)
- `nodes/OTR_LedgerFreezeCascade.py` (ComfyUI node wrapper)
- `nodes/_otr_lfc_llm_helpers.py` (two-step gen + JSON repair)
- `nodes/_otr_lfc_context.py` (Macro-Bible + Micro-Recap)
- `nodes/_otr_lfc_watchdog.py` (VRAM watchdog + voice-drift stats)
- `nodes/_otr_readiness.py` (Phase 7 + Phase 8)
- `nodes/_otr_lfc_smart_suggestion.py` (Phase 4.5)

**Polish-path fixes (sit on the composer side) —**
- `nodes/_otr_line_composer.py` (LineRequest + polish_line + needs_polish + refusal detector)
- `nodes/_otr_model_loader.py` (make_polish_generate_fn factory)

**Workflow JSON wiring —**
- `workflows/otr_scifi_16gb_full.json` (node id 62 + widgets_values)

**Test suites —**
- `tests/test_lfc_phase_0_10_gap_audit.py` (59)
- `tests/test_lfc_freeze_cascade_orchestrator.py` (17)
- `tests/test_lfc_polish_fixes.py` (38)
- `tests/test_lfc_phase_3_polish_in_cascade.py` (20)
- `tests/test_lfc_phase_7_8_readiness.py` (24)
- `tests/test_lfc_llm_helpers.py` (23)
- `tests/test_lfc_context_helpers.py` (29)
- `tests/test_lfc_phase_4_5_smart_suggestion.py` (14)
- `tests/test_lfc_watchdog_and_widgets.py` (24)
- `tests/test_legacy_contract_retired.py` (3)

**Reference docs —**
- `docs/2026-05-11-multi-turn-polish-adr.md` (the ADR)
- `docs/2026-05-11-multi-turn-polish-problem-statement.md`
- this handoff

---

## 6. 4-5 round-robin review prompts

Paste these into ChatGPT + Gemini for the post-commit review.

### Prompt A — Architecture review

> Review the LFC sprint as a whole. Files attached: ADR, problem
> statement, this handoff, `nodes/_otr_freeze_cascade.py`,
> `nodes/_otr_ledger_freeze.py`, `nodes/_otr_lfc_llm_helpers.py`,
> `nodes/_otr_lfc_context.py`, `nodes/_otr_readiness.py`,
> `nodes/_otr_lfc_smart_suggestion.py`, `nodes/_otr_lfc_watchdog.py`,
> `nodes/OTR_LedgerFreezeCascade.py`. Assess:
> 1. Does the cascade's phase-chain ordering match the ADR's
>    `[Writer] → Phase 0 → Phases 1+2+9 → Phase 3..8 → Phase 10`
>    contract? Highlight any deviation.
> 2. Is the mutation contract (ADR §5) respected by every
>    implemented phase? Specifically: does any phase add/remove
>    lines other than Phase 4.5?
> 3. Are the no-op stubs for Phase 4 / 5 / 6 wired in a way that a
>    follow-up commit can replace each stub in place without
>    rewiring the orchestrator?
> 4. Flag any place where the cascade could silently corrupt
>    `meta.cleanup_locked` (e.g. stamp `cleanup_locked=True` on a
>    ledger that didn't actually pass Phase 10).

### Prompt B — Phase 4 design pre-flight (commits 8 deferred)

> The Phase 4 (per-scene coherence) commit is deferred. Based on
> ADR §6.13 / §6.14 / §6.15 and the helpers we've shipped
> (`build_macro_bible`, `build_micro_recap`, `two_step_generate`,
> `parse_with_repair`, `scenes_from_lines`, `ReviewerOutput`), draft
> the implementation:
> 1. What's the recommended reasoning-prompt body for Phase 4?
>    Show how the Narrator-Character-Editor scaffold (§6.14) +
>    audio-first directives (§6.15) compose with the Macro-Bible /
>    Micro-Recap context.
> 2. How should the per-scene edit cap
>    (`min(3, scene_lines // 2)`) interact with the LLM's
>    proposed-edits count? Sketch the apply-or-drop logic.
> 3. What's the failure-cascade policy when the formatter pass
>    returns garbage on a SINGLE scene? Drop just that scene's
>    edits and advance to next scene, or fail the whole Phase 4?

### Prompt C — Phase 5 voice drift design (commit 9 deferred)

> The Phase 5 (voice drift) commit is deferred. We've shipped
> `compute_speaker_stats` + `flag_line_drift` in
> `nodes/_otr_lfc_watchdog.py`. Draft Phase 5:
> 1. Should Phase 5 batch ALL flagged lines into a single LLM call
>    (cheap, ~1 call), or do per-speaker batches (~N calls), or
>    per-line (~K calls)? ADR §6.5 says "one batched LLM call";
>    confirm or push back.
> 2. The drift thresholds (40% word-count deviation, 60% vocab
>    diversity collapse) are tunable constants in `_otr_lfc_watchdog.py`.
>    Are they reasonable defaults, or should soak data drive them?
> 3. How should Phase 5 interact with Phase 3 polish? If a line
>    drifted AND tripped needs_polish, does it get polished or
>    voice-drifted first?

### Prompt D — Phase 6 episode arc design (commit 10 deferred)

> The Phase 6 (episode arc) commit is deferred. Draft Phase 6:
> 1. The Editor-note scaffold (§6.14): how should the orchestrator
>    translate Editor notes into edits? Show the prompt + the
>    formatter pass that produces `EditorNotesOutput`.
> 2. Phase 6's edit_cap is `min(8, max(3, voiced_beats // 3))`
>    matching the existing reviewer. Should it use the SAME cap as
>    the reviewer (risking double-spend) or a separate cap?
> 3. `meta.scene_synopses` is supposed to be cached by Phase 4 (one
>    LLM call per scene at cascade entry). Phase 6 then reads
>    cached synopses to keep its own context small. What's the
>    fallback when Phase 4 didn't fire (disabled / failed)?

### Prompt E — VRAM watchdog enforcement (follow-up)

> The VRAM watchdog (`nodes/_otr_lfc_watchdog.py:vram_over_ceiling`)
> is built but NOT yet wired into every LLM call site in the
> cascade. Per ADR §6.8 it should fire BEFORE each LLM call. Draft
> the integration: where in `run_freeze_cascade` should the
> watchdog check fire? What's the right behavior on
> `over_ceiling=True` — skip the whole phase, skip the LLM portion
> of a phase, or fail the cascade? Constraint: no
> quantization/streaming/FA chasing per Jeffrey's
> no-VRAM-dragons memory (alarm plumbing only).

---

## 7. Standing context for the next session

- **Branch:** `v2.0-alpha`. Do NOT touch `main`.
- **Workflow JSON:** `workflows/otr_scifi_16gb_full.json` is the
  canonical graph. Node id 62 is the cascade.
- **Tests:** Run from repo root with
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/test_lfc_*.py`.
  Full LFC suite is ~250+ cases and runs in ~3 seconds.
- **Bug Bible regression:**
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest "C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\tests\bug_bible_regression.py"`.
  Baseline at HEAD `80b4244`: 23 passed / 1 skipped / 2 xfailed.
- **Sandbox notes:** bash sandbox truncates view of repo files.
  Use **Read** tool (Windows-canonical) for file inspection and
  **Desktop Commander cmd shell** for git operations.
- **Cascade widget surface:** all 6 new widgets default to their
  ADR-correct values. Phase 3 / 4.5 OFF (need soak); Phase 7 / 8
  ON (deterministic + cheap); VRAM ceiling 14.0 GB.

---

**End of handoff.**
