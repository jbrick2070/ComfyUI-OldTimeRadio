# Follow-up: meta.story_brief as Single Source of Truth for Downstream Creative

- **Status:** Audit phase. Sprint 7 is closed (v4 plan landed pending operator live-run). This document is the canonical plan for the brief consumer wiring sprint -- update it in place as the work lands.
- **Origin:** 2026-05-25 live-run log showed `[OTR_MusicGenTheme] story_brief_status=ok mood_terms=[] style_slug_diag=sanctioned_trade_battle` -- brief landed successfully, music node consumed nothing useful from it. Confirmed: brief was decoration for the music path.
- **Current HEAD at last update:** `91007e7` (Sprint 7C close -- payload_null typed repair).
- **Last refreshed:** 2026-05-25 (post audit -- pre wiring).

---

## State of the work

| Phase | Status | Pointer |
|---|---|---|
| Audit -- classify every consumer A/B/C/D against the v1 schema | **DONE** 2026-05-25 | `downstream_brief_consumer_audit.md` at repo root |
| Schema-shape decision (flat additive vs nested object) | **OPEN** -- decision A below | -- |
| `_otr_brief_reader.py` shared read helper | **PENDING** (commit 1 of the wiring sprint, or its own preceding commit -- decision C below) | new module |
| Producer v2 schema add (`_otr_story_brief.py`) | **PENDING** (folded into commit 1 -- the consumer needs the producer fields to read) | `nodes/_otr_story_brief.py` |
| Consumer wiring -- one commit per consumer, order below | **PENDING** -- starts with MusicGenTheme | per-consumer files |

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
