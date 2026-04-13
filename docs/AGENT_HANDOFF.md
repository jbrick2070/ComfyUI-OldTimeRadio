# OTR v2.0 Agent Handoff Document

**Last updated:** 2026-04-12 (guardrail sweep)  
**Active branch:** `v2.0-alpha`  
**Last confirmed remote commit:** `30ce8b3` (guardrail changes uncommitted)

---

## Where We Are: Phase 0 — COMPLETE

### Completed
- All v2.0 Phase 0 code written and pushed to `v2.0-alpha`
- Bug Bible regression: **109 passed, 25 skipped, 2 xfailed** — clean
- `tests/test_core.py`: 83 passed — clean
- 10 bugs found and fixed (see BUG_LOG.md — BUG-LOCAL-001 through 010)
- **Pre-flight guardrail sweep (BUG-009/010):** runtime presets auto-clamp target_length, dialogue line minimums scale dynamically with target_minutes, character counts clamped per episode length, Obsidian profile capped at 10 min, outline temp capped at model max, dead widgets (news_headlines, temperature) marked DEPRECATED
- 1-min test preset removed, minimum runtime = 3 minutes
- `otr_v2/` package scaffolded (subprocess runner + visual plan schema)
- `tests/v2/test_audio_byte_identical.py` — Phase 0 regression gate
- `tests/v2/_run_baseline.py` — ComfyUI HTTP API integration
- `tests/v2/_extract_from_history.py` — audio extraction utility
- Workflow `otr_scifi_16gb_full.json` — lean, API-clean, 5-min episodes
- **Baseline fixtures committed and pushed** — confirmed on GitHub at `30ce8b3`

### Baseline Captured
**SHA-256: `68654b607b33e7cce7e157e82b737d996d8d766c21383f177c9a5841e355e490`**

- `tests/v2/fixtures/baseline_v1.5.wav` — 12.9 MB, 117s episode
- `tests/v2/fixtures/baseline_v1.5.sha256` — contains the hash above
- Episode: "The Last Frequency" | hard_sci_fi | [FAST] quick (5 min) | medium (5 acts)
- Characters: CHIDI (f), REN (m), RUFUS (m), DMITRI (m)
- Node 15 widget drift workaround applied (BUG-LOCAL-008)
- Captured 2026-04-12 via automated scheduled task (seed=42)

---

## Next: Phase 1

Read `docs/superpowers/specs/2026-04-12-otr-v2-visual-sidecar-design.md` for full spec.

**Phase 1 Summary**: Teach `OTR_Gemma4Director` to emit a `visual_plan` block inside its existing `production_plan_json` output. No IO changes to the node — only the system prompt and post-processing change. On schema validation failure, emit empty `visual_plan` (graceful degradation — audio still ships).

---

## Critical Paths

### Python
```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
```
System `python` is NOT on PATH. Always use full path or activate venv first.

### Key Files
| File | Purpose |
|------|---------|
| `workflows/otr_scifi_16gb_full.json` | Single source of truth workflow (5-min episode) |
| `tests/v2/test_audio_byte_identical.py` | Phase 0 gate + CLI entry point |
| `tests/v2/_run_baseline.py` | ComfyUI HTTP API integration |
| `tests/v2/_extract_from_history.py` | Audio extraction by prompt_id |
| `tests/v2/fixtures/baseline_v1.5.wav` | Baseline WAV (12.9 MB) |
| `tests/v2/fixtures/baseline_v1.5.sha256` | SHA-256 gate hash |
| `BUG_LOG.md` | Live bug log — 10 entries, all fixed |
| `nodes/v2_preview.py` | 4 placeholder nodes (CharacterForge, ScenePainter, VisualCompositor, ProductionBus) |
| `otr_v2/schema/visual_plan.schema.json` | JSON Schema for Phase 1 visual plan |
| `docs/superpowers/specs/2026-04-12-otr-v2-visual-sidecar-design.md` | Full 6-phase design spec |

### ComfyUI Desktop
- Port: `8000` (not the default 8188)
- API base: `http://127.0.0.1:8000`
- History evicts completed prompts — use `_extract_from_history.py <prompt_id>` immediately after completion

### v2.0 Hard Constraints (from CLAUDE.md)
- **C1**: No new inputs on any v1.5 node — causes widget drift
- **C2**: No CheckpointLoaderSimple in main graph — OOM
- **C3**: All visual generation in subprocesses (`multiprocessing.get_context("spawn")`)
- **C4**: LTX-2.3 clips max 10-12s
- **C7**: Audio output byte-identical to v1.5 baseline at every gate
- Only legal v1.5 modification: `OTR_SignalLostVideo` gets one optional `visual_overlay` input (last slot)

### Episode Length Settings
- `runtime_preset`: `[FAST] quick (5 min)` — 5-min target (workflow default)
- `target_length`: `medium (5 acts)` — DO NOT use `short (3 acts)` — causes PARSE_FATAL (BUG-LOCAL-007)
- Changing `runtime_preset` alone is safe; changing `target_length` to short breaks the parser

### Git Push Protocol
- One attempt max from AI side
- PowerShell blocks must use plain ASCII, start with `cd`, be copy-paste ready
- Never use PowerShell for Python file writes (BOM/encoding corruption)
- Lockstep verify after every push

### Known Open Issues
- **BUG-LOCAL-008 (WORKAROUND)**: `OTR_BatchAudioGenGenerator` widget drift — schema ordering mismatch. `_fix_known_widget_drift()` in `_run_baseline.py` hardcodes correct values. `debug_audiogen_schema.json` dumped on each run for root cause investigation.

---

## Bug Log Summary
See `BUG_LOG.md` for full entries. Bible candidates: BUG-LOCAL-001, 003, 005, 006, 007, 008.

| ID | Summary | Status |
|----|---------|--------|
| BUG-LOCAL-001 | v2_preview.py OUTPUT_NODE flag on data-flow nodes | FIXED |
| BUG-LOCAL-002 | Stale TestWorkflowJSONLite referencing deleted workflow | FIXED |
| BUG-LOCAL-003 | Widget-value drift in workflow-to-API conversion | FIXED |
| BUG-LOCAL-004 | v2 placeholder nodes cause API 400 | FIXED |
| BUG-LOCAL-005 | Emoji vs [EMOJI] mismatch in dropdown values | FIXED |
| BUG-LOCAL-006 | Converted widget alignment in widgets_values | FIXED |
| BUG-LOCAL-007 | PARSE_FATAL with short (3 acts) + 5-min preset | FIXED |
| BUG-LOCAL-008 | Node 15 BatchAudioGenGenerator widget drift recurrence | WORKAROUND |

---

## Phase Roadmap
| Phase | Name | Status |
|-------|------|--------|
| 0 | Audio Regression Baseline | **COMPLETE** — SHA-256: `68654b607b33e7cce7e157e82b737d996d8d766c21383f177c9a5841e355e490` |
| 1 | Visual Plan Schema Extension | Not started |
| 2 | VisualGateway Sink Node | Not started |
| 3 | VisualSidecar (Diffusion B-roll) | Not started |
| 4 | Composite into SignalLostVideo | Not started |
| 5 | Blender + Rhubarb NG | Not started |
| 6 | Ship | Not started |
