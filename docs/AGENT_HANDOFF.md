# OTR v2.0 Agent Handoff Document

**Last updated:** 2026-04-12  
**Active branch:** `v2.0-alpha`  
**Last confirmed commit:** `dc7e865` (matches origin HEAD — verified)

---

## Where We Are: Phase 0 — Audio Regression Baseline

### Completed
- All v2.0 Phase 0 code written and pushed to `v2.0-alpha`
- Bug Bible regression: **109 passed, 25 skipped, 2 xfailed** — clean
- `tests/test_core.py`: 83 passed — clean
- 6 bugs found and fixed (see BUG_LOG.md — BUG-LOCAL-001 through 006)
- `otr_v2/` package scaffolded (subprocess runner + visual plan schema)
- `tests/v2/test_audio_byte_identical.py` — Phase 0 regression gate
- `tests/v2/_run_baseline.py` — ComfyUI HTTP API integration
- Workflow `otr_scifi_16gb_full.json` is the single source of truth (lean, API-clean)

### In Progress RIGHT NOW
**Baseline capture is actively running in ComfyUI Desktop.**

- Prompt ID: `612f9a65-3cdd-4f36-b214-681cf7db995b`
- ComfyUI Desktop running on `http://127.0.0.1:8000`
- Command that was run:
  ```powershell
  cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
  & C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe tests/v2/test_audio_byte_identical.py --capture-baseline
  ```
- Script polls /history every 5s, 30min timeout
- On success: writes `tests/v2/fixtures/baseline_v1.5.wav` and `tests/v2/fixtures/baseline_v1.5.sha256`

### Pending (after baseline capture completes)
1. **Commit baseline fixtures** to `v2.0-alpha`:
   ```powershell
   cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
   git add tests/v2/fixtures/baseline_v1.5.wav tests/v2/fixtures/baseline_v1.5.sha256
   git commit -m "Phase 0: capture audio regression baseline (seed=42)"
   git push origin v2.0-alpha
   ```
2. **Lockstep verify** push (local HEAD == origin HEAD, no zero-byte files)
3. **Promote Bible candidates** — BUG-LOCAL-001, 003, 005, 006 are Bible candidates needing Three-File Contract (YAML + README + regression test in `comfyui-custom-node-survival-guide`)
4. **Phase 1** — teach `OTR_Gemma4Director` to emit `visual_plan` block (see design spec)

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
| `workflows/otr_scifi_16gb_full.json` | Single source of truth workflow |
| `tests/v2/test_audio_byte_identical.py` | Phase 0 gate + CLI entry point |
| `tests/v2/_run_baseline.py` | ComfyUI HTTP API integration |
| `tests/v2/fixtures/` | Baseline WAV + SHA-256 (empty until capture completes) |
| `BUG_LOG.md` | Live bug log — all 6 entries fixed |
| `nodes/v2_preview.py` | 4 placeholder nodes (CharacterForge, ScenePainter, VisualCompositor, ProductionBus) |
| `otr_v2/schema/visual_plan.schema.json` | JSON Schema for Phase 1 visual plan |
| `docs/superpowers/specs/2026-04-12-otr-v2-visual-sidecar-design.md` | Full 6-phase design spec |

### ComfyUI Desktop
- Port: `8000` (not the default 8188)
- API base: `http://127.0.0.1:8000`

### v2.0 Hard Constraints (from CLAUDE.md)
- **C1**: No new inputs on any v1.5 node — causes widget drift
- **C2**: No CheckpointLoaderSimple in main graph — OOM
- **C3**: All visual generation in subprocesses (`multiprocessing.get_context("spawn")`)
- **C4**: LTX-2.3 clips max 10-12s
- **C7**: Audio output byte-identical to v1.5 baseline at every gate
- Only legal v1.5 modification: `OTR_SignalLostVideo` gets one optional `visual_overlay` input (last slot)

### Git Push Protocol
- One attempt max from AI side
- PowerShell blocks must use plain ASCII, start with `cd`, be copy-paste ready
- Never use PowerShell for Python file writes (BOM/encoding corruption)
- Lockstep verify after every push

---

## Bug Log Summary
See `BUG_LOG.md` for full entries. Bible candidates: BUG-LOCAL-001, 003, 005, 006.

| ID | Summary | Status |
|----|---------|--------|
| BUG-LOCAL-001 | v2_preview.py OUTPUT_NODE flag on data-flow nodes | FIXED |
| BUG-LOCAL-002 | Stale TestWorkflowJSONLite referencing deleted workflow | FIXED |
| BUG-LOCAL-003 | Widget-value drift in workflow-to-API conversion | FIXED |
| BUG-LOCAL-004 | v2 placeholder nodes cause API 400 | FIXED |
| BUG-LOCAL-005 | Emoji vs [EMOJI] mismatch in dropdown values | FIXED |
| BUG-LOCAL-006 | Converted widget alignment in widgets_values | FIXED |

---

## Phase Roadmap
| Phase | Name | Status |
|-------|------|--------|
| 0 | Audio Regression Baseline | IN PROGRESS (baseline capture running) |
| 1 | Visual Plan Schema Extension | Not started |
| 2 | VisualGateway Sink Node | Not started |
| 3 | VisualSidecar (Diffusion B-roll) | Not started |
| 4 | Composite into SignalLostVideo | Not started |
| 5 | Blender + Rhubarb NG | Not started |
| 6 | Ship | Not started |
