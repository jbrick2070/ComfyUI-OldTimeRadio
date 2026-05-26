# Follow-up: meta.story_brief as Single Source of Truth for Downstream Creative

- **Status:** **WIRING SPRINT COMPLETE.** All 7 commits landed (Sprint 8.1 producer v2 + reader + MusicGenTheme, Sprints 8.2-8.7 A-class consumers, Sprint 8.8 Bark/TTS audit -> D-class declination). PD1 live-run gate is the only carry-forward (operator-owned).
- **Origin:** 2026-05-25 live-run log showed `[OTR_MusicGenTheme] story_brief_status=ok mood_terms=[] style_slug_diag=sanctioned_trade_battle` -- brief landed successfully, music node consumed nothing useful from it. Confirmed: brief was decoration for the music path.
- **Current HEAD at last update:** see `Update 2026-05-25 (closeout)` section below.
- **Last refreshed:** 2026-05-25 (post-8.8 closeout).

---

## State of the work

| Phase | Status | Pointer |
|---|---|---|
| Audit -- classify every consumer A/B/C/D against the v1 schema | **DONE** 2026-05-25 | `downstream_brief_consumer_audit.md` at repo root |
| Schema-shape decision (flat additive vs nested object) | **DONE** -- A1 flat additive chosen 2026-05-25 | Update 2026-05-25 (commit 1) section below |
| `_otr_brief_reader.py` shared read helper | **DONE** -- shipped in Sprint 8.1 commit 1 | `nodes/_otr_brief_reader.py` |
| Producer v2 schema add (`_otr_story_brief.py`) | **DONE** -- shipped in Sprint 8.1 commit 1, `_PROMPT_VERSION` bumped v1 -> v2 | `nodes/_otr_story_brief.py` |
| Consumer wiring -- MusicGenTheme (Sprint 8.1, C-class) | **DONE** -- `3296b12` | `nodes/musicgen_theme.py` |
| Consumer wiring -- FLUX env (Sprint 8.2, A-class) | **DONE** -- `c36adc0` | `visual/batch_flux_render.py` |
| Consumer wiring -- FLUX portrait (Sprint 8.3, A-class) | **DONE** -- `0ed24ea` | `visual/batch_flux_portrait_render.py` |
| Consumer wiring -- FLUX radio bookend (Sprint 8.4, A-class) | **DONE** -- `9ae15f4` | `visual/batch_flux_render.py` |
| Consumer wiring -- LTX motion (Sprint 8.5, A-class) | **DONE** -- `e4448cf` | `nodes/batch_ltx_render.py` |
| Consumer wiring -- HuMo lip-sync (Sprint 8.6, A-class) | **DONE** -- `9fe95cd` | `nodes/batch_humo_render.py` |
| Consumer wiring -- OTR_VideoPlan era tail (Sprint 8.7, A-class) | **DONE** -- `b8972a6` | `nodes/otr_video_plan.py` |
| Bark / TTS deep audit (Sprint 8.8, carry-forward) | **DONE** -- D-class declination, see Update 2026-05-25 (closeout) | -- (no code change) |
| PD1 live-run gate (operator-owned) | **PENDING** -- one ComfyUI episode on post-8.7 HEAD must show v2 brief readers firing | operator only |

The audit's headline read: **one C-class consumer (MusicGenTheme)**, **six A-class consumers** (LTX, FLUX env, FLUX portrait, FLUX radio bookend, HuMo lip-sync, OTR_VideoPlan), **zero B-class**, and **the only D-class candidate (title scratchpad) declines** because Sprint 3E already grounds the title path on a rich excerpt set. Full classification table with file:line evidence lives in `downstream_brief_consumer_audit.md`.

---

## Decision (carried forward, unchanged)

`meta.story_brief` is the single source of truth for all downstream creative prompts. Brief schema upgrades to carry what downstream needs. Downstream nodes become deterministic consumers. Zero new LLM calls.

This matches the v4 architectural pattern: ledger writer is authoritative, downstream nodes are deterministic consumers. The brief earns its keep across all readers from one LLM pass. Cross-modal consistency becomes structural, not coincidental.

---

## Brief Schema v2

**Open decision A (below)** has to land before code -- the original draft proposed a nested object (`meta.story_brief.premise`, `.mood`, ...). The audit found a naming collision: v1 emits `meta.story_brief` as a **string** (the prose) plus a sidecar `meta.story_brief_terms` dict. Moving to a nested object replaces the prose-as-string contract every existing consumer reads. Flat additive (A1 below) avoids that.

### Original proposal (nested object -- flagged for review)

```json
"story_brief": {
  "premise": "...",
  "mood": "...",                  // one phrase, narrative register
  "music_mood_terms": [...],      // 3-5 music-vocabulary tags
  "visual_palette": [...],        // 3-5 color/texture tags
  "atmosphere": "...",            // one phrase for portraits/scenes
  "style_descriptor": "...",      // matches the locked style_slug
  "tempo_hint": "...",            // pacing / energy register
  "key_objects": [...]            // anchors for visual / portrait
}
```

### Audit-recommended alternative (flat additive, A1 in decision A)

Keep v1 keys intact. Add v2 keys at the top-level meta:

```text
meta.story_brief                  str           # v1 (the prose) -- unchanged
meta.story_brief_status           str           # v1 -- unchanged
meta.story_brief_terms            dict          # v1 (setting/lighting/atmosphere arrays) -- unchanged
meta.story_brief_terms.setting    list[str]
meta.story_brief_terms.lighting   list[str]
meta.story_brief_terms.atmosphere list[str]
# v2 additive (all optional, safe defaults):
meta.music_mood_terms             list[str]     # NEW -- music-tuned mood vocabulary
meta.visual_palette               list[str]     # NEW -- color / texture / material descriptors
meta.atmosphere_line              str           # NEW -- one-sentence atmosphere line (naming distinct from terms.atmosphere)
meta.tempo_hint                   str           # NEW -- "slow" / "moderate" / "driving"
meta.key_objects                  list[str]     # NEW -- named props the brief promised
```

Zero v1 breakage; the C5b helpers keep working; the new consumers read via the shared `_read_brief_field` contract with dotted keys (`"music_mood_terms"`, `"terms.atmosphere"`, etc.).

Implementation notes (unchanged regardless of A1 vs A2):
- One LLM call still -- the existing brief pass. Schema widens; call count stays at +0.
- `StoryBriefModel` (Pydantic) gets the new fields. Reject lists already in place stay as the safety net.
- Sprint 3G's pre-sanitization (`HAYES VANCE` -> `character_a`, etc.) still applies; the model never sees raw names.
- Goes through `structured_call` (Sprint 2A) -- 3 attempts + repair, same as today.
- Backward compat: old ledgers without v2 fields -> downstream consumers fall back to today's behavior (the same fallback pattern Sprint 6 uses for `meta.render_plan`).

---

## Downstream Sweep -- DONE

Full classification table lives in `downstream_brief_consumer_audit.md`. Headline:

| Node | Class | Action |
|---|---|---|
| `OTR_MusicGenTheme` | **C** | Reads atmosphere directly + helper-filtered mood. v2 `music_mood_terms` is the dedicated fix. **Commit 1 of the wiring sprint.** |
| `OTR_BatchLTXRender` | A | Reads brief prose via `get_story_brief_ltx`. v2 `tempo_hint` is opportunistic enrichment. |
| `OTR_BatchFluxPortraitRender` | A | Reads lighting+atmosphere. v2 `visual_palette` + `atmosphere_line` enrichments. |
| `OTR_BatchFluxRender` (env stills) | A | Reads brief prose -- leads the env prompt. v2 `visual_palette` + `key_objects` enrichments. |
| `OTR_BatchFluxRender` (radio bookend) | A | Reads brief prose -- primary radio descriptor. v2 `visual_palette` enrichment. |
| `OTR_VideoPlan` (`_resolve_era_tail`) | A | Reads lighting+atmosphere. v2 `visual_palette` + `atmosphere_line` enrichments. |
| `OTR_BatchHumoRender` (`_build_pos_prompt`) | A | Reads lighting+atmosphere. v2 `atmosphere_line` enrichment. |

D-class candidates checked: title scratchpad (declines -- Sprint 3E grounding sufficient), style picker (causal block -- runs pre-script), upstream creative passes (same causal block).

Music / audio: Bark / TTS voice-render path was NOT individually probed in this audit pass -- voice cards already exist, but the audit did not deep-dive whether brief tempo / timbre hints flow into Bark prompt construction. **Carry-forward task:** confirm Bark consumes nothing from the brief, or classify and add to the wiring order.

Cascade / freeze: `OTR_LedgerFreezeCascade` is meta-level (no creative inference); confirmed A-by-design (no need). Phase 7 / 8 / 10 are normalization / freeze passes -- no creative inference.

---

## Wiring Pattern for Consumers (unchanged)

Every downstream consumer follows the same shape so the pattern is uniform:

```python
# nodes/_otr_brief_reader.py -- shared read contract
def _read_brief_field(meta: dict, field: str, fallback):
    """Single helper. Reads meta.story_brief.<field> with backward-compat fallback.
    Brief absent or field absent -> fallback (today's behavior).
    Never raises. Logs which path was taken at INFO level."""
    ...
```

Open decision B below decides whether `field` is a dotted path under a nested object (e.g. `"story_brief.music_mood_terms"`) or a flat top-level key (e.g. `"music_mood_terms"`); both work, A1 favours flat.

```python
# In MusicGenTheme:
mood_terms = _read_brief_field(meta, "music_mood_terms", fallback=[])
style_slug = _read_brief_field(meta, "style_descriptor", fallback=meta.get("style_slug", ""))

# In portrait node:
atmosphere = _read_brief_field(meta, "atmosphere_line", fallback="")
key_objects = _read_brief_field(meta, "key_objects", fallback=[])

# In HuMo prompt builder:
visual_palette = _read_brief_field(meta, "visual_palette", fallback=[])
```

The helper lives in `nodes/_otr_brief_reader.py` (new module, pure stdlib, no sibling imports -- same pattern as `_otr_json.py`). Tested independently. Reused by every downstream node so the read contract is one place, not N.

---

## Prime Directives Check (carried forward)

- **PD1 (never raises):** `_read_brief_field` never raises; consumers degrade to today's behavior on any failure. The downstream creative pipeline continues to produce output even with a malformed or absent brief.
- **PD3 (no workflow JSON re-wire):** schema v2 is additive on the meta dict. No node `INPUT_TYPES` changes, no widgets added. Workflow JSON untouched.
- **PD6 (no new LLM calls):** brief pass already exists; schema widens. Call count delta: **0.** LLM-slot sweep stays at 23/23 tagged.
- **VRAM:** no change. No new model residency.

---

## Implementation Order

1. **Audit first, fix second.** -- **DONE 2026-05-25.** `downstream_brief_consumer_audit.md` at repo root.
2. **Resolve open decisions A / B / C below.** -- **OPEN.** Jeffrey-gated.
3. **Land `_otr_brief_reader.py` helper.** Tested. No consumers yet -- just the read contract. (Commit gate per decision C.)
4. **Upgrade `_otr_story_brief.py` to schema v2.** New fields stamped; reject-list safety net carried forward; backward-compat verified.
5. **Wire consumers one at a time.** Each B/C classification from the audit becomes its own small commit. **Commit order (audit recommendation):** MusicGenTheme (C) -> FLUX env (A enrichment) -> FLUX portrait -> FLUX radio bookend -> LTX tempo -> HuMo atmosphere -> VideoPlan tail. Each consumer wires via the helper; each commit ships with its own regression test.
6. **Bark / TTS audit -- carry-forward.** Confirm or classify the voice-render path's brief consumption.
7. **Final regression sweep.** Live-run a clean episode and verify `mood_terms` is non-empty, palette flows to visual, atmosphere flows to portrait.

---

## Success Criteria (unchanged)

The follow-up is done when:

- Every node identified in the sweep carries an audit classification (A/B/C/D). **DONE for 7 nodes; Bark TTS is carry-forward.**
- Every B and C is fixed and tested.
- A live-run episode shows `[OTR_MusicGenTheme] mood_terms=[<non-empty list>]` and equivalent non-empty reads in visual / portrait paths.
- `meta.story_brief` is a load-bearing artifact, not decoration. Removing it from a ledger would visibly degrade downstream creative output.

---

## Open decisions

A. **Schema shape -- flat additive (A1) vs nested object (A2).**

   * **A1 (recommended by audit).** v2 fields live at top-level `meta.music_mood_terms` etc. v1 keys untouched. Zero breakage for the six A-class consumers. The `_read_brief_field` helper takes a flat field name.
   * **A2 (original proposal).** v2 fields live under `meta.story_brief.<field>`, i.e. `meta.story_brief` becomes a dict instead of a string. **Naming collision** -- v1 emits `meta.story_brief` as a string (the prose); every A-class consumer reads it that way through `get_story_brief_full`. Moving to a dict forces a same-commit rename of the v1 prose field (e.g. `meta.story_brief.prose` or `meta.story_brief_text`).

B. **`_otr_brief_reader.py` API -- dotted-path read vs flat-key read.** Cosmetic if A1 wins; load-bearing if A2 wins. Dotted-path (`"story_brief.music_mood_terms"`) reads through either shape; flat-key (`"music_mood_terms"`) reads only A1. Recommend dotted-path for forward-compat regardless of A choice.

C. **Sequencing -- ship `_otr_brief_reader.py` with commit 1, or as its own preceding commit?**

   * **C1.** Commit 1 carries: producer v2 fields + reader module + MusicGenTheme rewire + tests. Larger commit; the reader's first real caller proves it works.
   * **C2.** Commit 1 ships only the reader module + its unit tests (no consumers yet). Commit 2 wires MusicGenTheme. Smaller diffs; the reader sits unused for one commit.

D. **`atmosphere` naming under A1.** v1 has `meta.story_brief_terms.atmosphere` as a list. v2's single-sentence atmosphere field needs a non-colliding name. Audit recommends `meta.atmosphere_line` (or `meta.story_brief_atmosphere_line`). Decide before commit 1.

E. **Bark / TTS deep audit.** Not done in this pass. Decide if it lands before commit 1 (audit completion) or as a Bark-specific follow-up after the visual wiring is done.

---

## Out of Scope (Don't Drift) -- unchanged

- No brief-schema breakage. v2 is additive (A1) or carries an explicit v1 rename (A2); v1 readers must still work either way.
- No node surface changes (PD3).
- No work on Sprint 5C reroll, Sprint 6 render plan, or any v2 follow-up from the earlier menus until this audit is done. **Audit is now done** -- wiring sprint is clear to start once decisions A-D land.

---

## Update protocol

This file is the canonical "big plan" doc. The discipline:

- **Append, don't rewrite.** Add `## Update <date>` sections at the bottom recording state changes (decisions resolved, commits landed, sprint state transitions). Earlier section bodies stay frozen so the history is auditable.
- **Status table at the top is the only authoritative live view.** Update its cells in place when state changes; everything else is append-only.
- **`downstream_brief_consumer_audit.md` is a snapshot.** It pins what the audit saw at HEAD `91007e7`. Re-audit and re-snapshot at the next live-run anchor; do not edit the snapshot in place.

---

## Update 2026-05-25 -- file moved to repo root, audit folded in

- File first lived in Jeffrey's upload area. Moved to `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\downstream_brief_consumer_followup.md` (repo root) so the session_handoff item 1's "check first if it exists" lookup hits.
- Audit completion folded into "State of the work" + "Downstream Sweep -- DONE" sections; classification table pulled from `downstream_brief_consumer_audit.md`.
- Schema-shape collision flagged as open decision A (flat additive recommended by audit; original nested-object proposal preserved verbatim above).
- Carry-forward: Bark / TTS deep audit (decision E).
- Sprint state: v4 plan COMPLETE (Sprints 0, 1, 2A-2E, 3A-3G, 4, 5A-5C, 6, 7A-7C) at HEAD `91007e7`. Wiring sprint = "Sprint 8.x" by audit convention (rename if a different label is preferred).

---

## Update 2026-05-25 (commit 1) -- Sprint 8.1 producer v2 + reader + MusicGenTheme rewire

- **Open decisions resolved by Jeffrey:**
  - **A1 flat additive** -- v2 fields land as top-level meta keys alongside the v1 8-key contract. Zero rename of `meta.story_brief` prose-string. Six A-class consumers continue to read v1 fields unchanged.
  - **B dotted-path** -- `_read_brief_field(meta, "story_brief_terms.atmosphere", default=[])` forward-compat regardless of A.
  - **C1 bundled** -- producer v2 + reader + MusicGenTheme rewire in a single commit so the reader has a real first caller and the PD1 live-run signal (`mood_terms=[<non-empty>]`) is observable from one push.
  - **D `meta.atmosphere_line` (bare)** -- consistent with the rest of the v2 bare-key group; v1 `story_brief_terms.atmosphere` list stays untouched, no collision.
  - **E Bark / TTS audit deferred** -- after Sprint 8.7 visual rewires; Bark is a hygiene check, not a flagged miss.
- **Producer changes (`nodes/_otr_story_brief.py`):**
  - `StoryBriefModel` grew five v2 fields with safe defaults: `music_mood_terms: list[str]`, `visual_palette: list[str]`, `key_objects: list[str]`, `tempo_hint: str` (cap 80), `atmosphere_line: str` (cap 200). A v1-era LLM response missing these keys still validates -- field falls to its default and the consumer drops through to v1.
  - `_REFLECTION_PROMPT` extended to ask for the five new fields. Prompt body grew from ~243 -> ~312 approx-tokens (linear scaling for 4 -> 9 fields would have predicted ~540). Test cap bumped 250 -> 320 with an inline explanation.
  - `_success_delta` + `_failure_sentinel` stamp all five v2 fields as top-level meta entries (A1). On the failure path the v2 fields land safe-empty (`[]` / `""`) so downstream readers can call `_read_brief_field` unconditionally.
  - `_PROMPT_VERSION` bumped `v1` -> `v2`.
- **New reader (`nodes/_otr_brief_reader.py`):**
  - Single public function `_read_brief_field(meta, dotted_path, default)`. Pure module, no GPU / I/O / ComfyUI imports. Accepts either a brief-shaped meta dict OR a parent dict carrying a `meta` sub-key (mirrors `_otr_story_brief_helpers._meta`).
  - Dotted-path navigator returns `default` on missing segments, non-dict intermediates, empty meta, and terminal `None`. Raises `ValueError` on empty / whitespace / empty-segment paths -- catches typos before they degrade to a silent default.
- **MusicGenTheme rewire (`nodes/musicgen_theme.py`):**
  - `_compose_music_prompt` now reads `music_mood_terms` first (top 3, matches v1 atmosphere[:3] slice). On empty falls through to the existing `story_brief_terms.atmosphere` path, then to keyword-mining of `news.script_brief`, then to the neutral `atmospheric` default. The v1 paths are unchanged so legacy ledgers and the failure sentinel continue to work.
  - PD1 live-run log line restructured: `mood_terms=` now reports the resolved list (v2 if present, v1 fallback otherwise) and a new `mood_source=v2_music_mood_terms|v1_atmosphere_vocab` annotation tells operators which path won.
  - Class C resolved -- the audit's only C-class consumer is now reading music-tuned mood signal from the brief.
- **Test additions (39 new passing tests):**
  - `tests/test_brief_reader.py` (20 tests): every helper contract -- flat-key read, dotted-path traversal of v1 nested objects, default fallback paths, parent-dict normalization, None-terminal safety, typo-guard ValueErrors.
  - `tests/test_musicgen_brief_rewire.py` (12 tests): v2 path preferred, top-3 slice, v1 atmosphere fallback, keyword-mining fallback, neutral default, composition spine still intact (setting clause + cue character + prompt tail), malformed v2 fields degrade gracefully.
  - `tests/test_story_brief_c5a1.py` (+7 tests in `TestV2ProducerFields`): `_PROMPT_VERSION == "v2"`, v2 fields stamped on success delta, v1-era LLM response fills v2 safe defaults, failure sentinel stamps v2 safe defaults on both raise + parse-failure paths, schema length caps fire on over-long tempo_hint / atmosphere_line, prompt body lists all nine field names.
- **Regression at HEAD post-push:**
  - Full OTR suite: 2895 passed / 21 skipped / 0 failed (was 2856/21/0; +39 new tests).
  - Bug Bible: 16 passed / 7 skipped / 3 xfailed / 0 failed (unchanged).
  - LLM-slot sweep: 23/23 tagged, 0 untagged, 0 parse failures -- no LLM call added or removed.
- **Out-of-scope confirmations (PD1, PD3, PD6):**
  - **PD1 (audio is king).** Audio resolution path unchanged -- only the mood prefix on MusicGen prompts gained a new source. Failure sentinel keeps all v1 audio outputs working.
  - **PD3 (workflow JSON).** N/A -- no node surface change, no widget rename, no input/output socket added.
  - **PD6 (LLM-slot tagging).** Sprint 8.1 added zero LLM calls. Slot sweep stayed at 23/23.
- **PD1 live-run gate signal (operator-owned):** one ComfyUI episode on the post-commit HEAD must show `[OTR_MusicGenTheme] story_brief_status=ok mood_terms=[<non-empty>] mood_source=v2_music_mood_terms` in the console for the v2 path to be confirmed live. v1-fallback observation (`mood_source=v1_atmosphere_vocab`) is also acceptable -- proves the rewire's resolution order kicked in -- but the v2 source confirms the producer v2 schema reached the consumer end-to-end.
- **Next:** Sprint 8.2 -- FLUX env consumer (`nodes/...` -- `_parse_env_prompts`) reads `visual_palette` + `key_objects` via `_read_brief_field`. One commit per A-class consumer, audit order, until Sprint 8.7 closes the visual sweep.

---

## Update 2026-05-25 (closeout) -- Sprints 8.2-8.8 landed, A-class queue closed

**Headline:** every post-script creative-pass consumer the audit classified is now reading the meta brief through the same `_read_brief_field` helper, same signature, same default-on-missing fallback. The 5-tuple (`music_mood_terms`, `visual_palette`, `key_objects`, `tempo_hint`, `atmosphere_line`) served all 7 wired consumers with **no field extensions and no new access patterns** -- "one stamp, many readers" landed as designed.

### Sprints 8.2-8.7 -- A-class consumer wiring (all pushed)

| Sprint | Consumer | File | Function | Fields read | Commit |
|---|---|---|---|---|---|
| 8.2 | FLUX env | `visual/batch_flux_render.py` | `_parse_env_prompts` | `visual_palette` + `key_objects` | `c36adc0` |
| 8.3 | FLUX portrait | `visual/batch_flux_portrait_render.py` | `_build_portrait_prompt` (via caller) | `visual_palette` + `atmosphere_line` | `0ed24ea` |
| 8.4 | FLUX radio bookend | `visual/batch_flux_render.py` | `_build_dynamic_radio_prompt` | `visual_palette` | `9ae15f4` |
| 8.5 | LTX motion | `nodes/batch_ltx_render.py` | `_build_ltx_role_prompt` | `tempo_hint` | `e4448cf` |
| 8.6 | HuMo lip-sync | `nodes/batch_humo_render.py` | `_build_pos_prompt` | `atmosphere_line` | `9fe95cd` |
| 8.7 | OTR_VideoPlan era tail | `nodes/otr_video_plan.py` | `_resolve_era_tail` | `visual_palette` + `atmosphere_line` | `b8972a6` |

**Pattern consistency across all six A-class consumers:**
- Same helper: `_read_brief_field(meta, "<field>", default=<v1-fallback-shape>)`.
- Same fallback discipline: v1-era ledgers and the v2 failure sentinel produce byte-identical renders to the pre-rewire path.
- Same defensive guards: non-list / non-string / whitespace / None terminal values all degrade to safe-empty.
- Same source-level test lock: every commit has a `TestReaderHelperUsed` class that pins `_read_brief_field` + the exact field key as the canonical access path (catches typo / rename drift).
- Same top-3 slice convention on list fields (matches MusicGen v1 atmosphere[:3] slice from Sprint 8.1).

**Audit naming-drift caught:** the 2026-05-25 audit named the LTX motion builder `_build_motion_prompt`; the actual function name is `_build_ltx_role_prompt`. The function does the job the audit described; the audit doc should be updated in a follow-up housekeeping pass.

### Sprint 8.8 -- Bark / TTS deep audit: D-CLASS DECLINATION

**Classification: D-class (causal block). No code change.**

**Why Bark does not read the brief:**
- Bark consumes script dialogue text directly from `ledger.lines[].text`, authored by the writer during `OTR_LedgerScriptWriter` (Sprint 3E).
- Voice preset selection is entirely cast-driven (Gate 3, voice-path-cleanbreak 2026-05-12): Bark reads `cast.voice_preset` and hard-raises if it's missing or doesn't start with `"v2/"`. No brief read, no fallback.
- Pipeline order: writer → freeze → Bark TTS (pre-compute) → SceneSequencer → composition → reflection (produces brief). **Bark executes BEFORE the brief exists.**
- The only meta field Bark reads is `meta.freeze_unload_ok` -- an infrastructure signal for defensive VRAM recovery, not a creative control field.

**Why wiring it would break things:**
- (a) Re-running Bark post-brief breaks PD1 (audio is king -- byte-identical narrative audio is load-bearing; a second TTS pass loses reproducibility and adds latency).
- (b) Inverting the pipeline (moving the reflection pass earlier to feed Bark) breaks post-script reflection semantics; the brief is generated FROM the locked script, not before it.

The 5-tuple's creative latitude (mood / visual palette / objects / tempo / atmosphere) is for consumers that have rendering freedom AFTER the narrative is locked. TTS does not -- it executes against authored dialogue lines. Bark is correctly positioned outside the brief-consumer sweep.

**Bark audit confirmed; close decision E.**

### Regression baseline at closeout HEAD `b8972a6`

- Full OTR suite: **2984 passed / 21 skipped / 0 failed** (was 2856/21/0 at Sprint 8.1 land; +128 new tests across 8.1-8.7).
- Bug Bible: **16 passed / 7 skipped / 3 xfailed / 0 failed** (unchanged across the whole sprint).
- LLM-slot sweep: **23/23 tagged, 0 untagged, 0 parse failures** (unchanged -- zero LLM calls added across 8.1-8.8).

### PD invariants held across the whole sprint

- **PD1 (audio is king).** Every commit preserved byte-identical narrative audio output. Sprints 8.1 (MusicGen) and 8.6 (HuMo) touched audio-adjacent paths -- both kept the v1 fallback behavior structurally identical when the v2 producer landed empty values. Failure sentinels are safe-empty across all five v2 fields.
- **PD3 (workflow JSON).** Zero node-surface, widget, or socket changes across the sprint. The workflow JSON is untouched at HEAD.
- **PD6 (LLM-slot tagging).** Zero LLM call sites added or removed. Slot sweep stayed at 23/23 across all 7 commits.

### Open carry-forward (operator-owned)

**PD1 live-run gate.** One ComfyUI episode on post-8.7 HEAD `b8972a6` should surface:

- `[OTR_MusicGenTheme] story_brief_status=ok mood_terms=[<non-empty>] mood_source=v2_music_mood_terms` -- confirms v2 reached MusicGen.
- `[BatchFluxRender] queued N env prompt(s) ... palette=[...] key_objects=[...]` -- confirms v2 reached FLUX env.
- `OTR_VideoPlan: era tail story_brief_status=ok (atmosphere_line_chars=N palette_terms=M v1_chars=K total_chars=T)` -- confirms v2 reached VideoPlan.
- Bark / TTS console output is **unchanged** (no v2 read by design, per the 8.8 D-class declination).

The wiring sprint is now closed end-to-end on Claude's side; the live-run is operator territory.
