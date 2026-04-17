# OTR Roadmap

**Last updated:** 2026-04-15 (session 2)
**Branch:** `v2.0-alpha`
**Owner:** Jeffrey A. Brick

Consolidated going-forward plan. Two horizons: **v1.7** (audio pipeline, ship now) and **v2.0** (visual drama engine, gated on HyWorld). Sources: docs/BUG_LOG.md, external review triage (2026-04-15), HyWorld Integration Plan v2.5, soak lessons. Everything shipped or discarded stays in the source docs — this file is open items only.

---

## Platform Pins

Lock these. Any work item that contradicts this list is wrong.

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA.
- Flash Attention 2/3: NOT AVAILABLE. Do not chase.
- 100% local, offline-first, open source, no API keys.
- VRAM ceiling: 14.5 GB real-world target.
- Audio is king. Full narrative output must never break, shorten, or degrade.

---

## P0 — Shipped or blocked

### 1. `min_line_count_per_character` structural constraint — SHIPPED (awaiting live test)
Injected `min_line_count_per_character=2` into `_critique_and_revise()`. Constraint added to the revision prompt. Post-critique validator counts dialogue lines per character in draft vs revision; rejects revision if any character drops below floor. Helper method `_count_character_lines()` added. Falls back to pre-critique draft on rejection.

### 2. Director JSON schema + validator — SHIPPED (awaiting live test)
`_DIRECTOR_SCHEMA` constant + `_validate_director_plan()` method in LLMDirector. Validates required keys (voice_assignments, sfx_plan, music_plan), repairs missing entries, validates voice_preset strings, filters broken sfx entries, synthesizes missing music cues, clamps duration_sec. Wired into `direct()` between `_extract_json()` and `_randomize_character_names()`.

### 3. Scene-Geometry-Vault *(v2.0 — blocked on Gate 0)*
Series-scale visual continuity bet. Without a persistent geometry vault, Act 3's bridge will not match Act 1's bridge across episodes. Design only after Gate 0 empirical measurements exist.

Source: review triage Keep #6.

### 4. Style-Anchor cache (World Seed + Lighting/Mood split) *(v2.0 — blocked on Gate 0)*
Turns the vault (#3) into a reuse engine. Same geometry, N relight passes (Day vs Night, Tense vs Calm). `style_anchor_hash` field in the Director schema keys the split.

Source: review triage Keep #2.

---

## P1 — Shipped or blocked

### 5. Length-sorted Bark batching — SHIPPED (awaiting live test)
Added `.sort(key=lambda item: len(item["line"]))` within each preset group in `batch_bark_generator.py`. Script order restored at assembly via `results[script_idx]`. Pure throughput win, zero quality risk.

### 6. Head-Start async pre-bake (Phase B.5) *(v2.0 — blocked on Gate 0)*
While ScriptWriter and Director run, kick off HyworldBridge on `outline_json` so geometry generation overlaps the LLM passes. Wall-clock win. Blocked on Gate 0 and vault stability.

Source: review triage Keep #1.

### 7. VRAM-Sentinel decorator on bark utils — SHIPPED (awaiting live test)
Added `vram_sentinel(phase_label, max_entry_gb)` decorator to `_vram_log.py`. Checks VRAM at entry, logs warning and force-offloads if above threshold. Applied to `BatchBarkGenerator.generate_batch()` at 6.0 GB ceiling. CUDA-absent safe.

### 8. ASCII sanitizer in prompt_compiler *(v2.0 — blocked on Gate 0)*
Strip non-ASCII (smart quotes, em-dashes) before Tencent text encoders. Preserve case. Collapse whitespace. Reference implementation in `2026-04-15-hyworld-integration-plan-review.md` (Rework §B).

Source: review triage Keep #4.

### 9. High-creativity soak profile — SHIPPED (awaiting live test)
Re-added `"maximum chaos"` to CREATIVITIES pool in `soak_operator.py` with weighted selection (~10% of runs). Catches temperature-sensitive TITLE/WORD_ENFORCEMENT regressions.

---

## P2 — Experiments and cleanup

### 10. `torch.compile` on Bark sub-models
`torch.compile(mode="reduce-overhead")` on semantic, coarse, and fine acoustic models. Needs isolated A/B timing — variable-length loops may fight the compiler. Benefit is batch runs with many lines per preset.

Source: Bark optimization triage B2.

### 11. Skip/shorten Bark fine acoustic pass
The fine pass adds high-frequency detail that AudioEnhance then destroys via tape emulation, LPF, Haas delay. If A/B blind test confirms quality loss is inaudible post-processing, truncating the fine pass cuts per-line wall time. Needs listening test, not spectrogram.

Source: Bark optimization triage B3.

### 12. Per-LLM-call VRAM snapshots — SHIPPED (awaiting live test)
Added `vram_snapshot("llm_generate_entry")` and `vram_snapshot("llm_generate_exit")` inside `_generate_with_llm()` in `story_orchestrator.py`. Logs token count and inference time. Every LLM call now emits VRAM telemetry.

### 13. `episode_title` socket input on OTR_SignalLostVideo
Replace implicit `script_json` title-token read with explicit socket input wired from ScriptWriter. v2.1 cleanup, not alpha-blocking.

Source: review triage K4.

---

## HyWorld Gate 0 — The gating empirical step

All visual-pipeline items (3, 4, 6, 8) are blocked on Gate 0. Do not write integration code until Gate 0 produces `gate0_results.md`.

Full protocol in `ComfyUI_Hyworld_Narrative_Integration_Plan_v2_5.md` §5. Summary:

1. Install HY-World 2.0 in separate conda env (`hyworld2`, Python 3.10).
2. Smoke test WorldMirror 2.0 (Gradio demo + CLI). Record peak VRAM, wall time.
3. Smoke test HunyuanWorld 1.0-lite (one panorama from Director prompt).
4. Smoke test WorldStereo 1.x install. Pass/fail.
5. Smoke test WorldPlay-5B (one short clip, CPU offload). Record VRAM, time.

Gate decision:
- All pass → Path A (full hybrid interim). Begin integration nodes.
- WorldStereo/WorldPlay fail → Path B (Mirror-Only + Blender/Three.js camera).
- WorldMirror fails → Path C (audio-only, revisit later).
- If 2.0 does not fit 16 GB → stay on 1.5, wait for upstream quantization.

---

## Recently shipped (for context, not action)

| Item | Summary | Status |
|---|---|---|
| v1.7 | Tagged and merged to `main` (`0aa6d6e`) | Shipped |
| BUG-LOCAL-034–040 | Parser resilience, title fixes, JSON repair | Shipped with v1.7 |
| P0 #1 | `min_line_count_per_character` self-critique guard | Code complete, needs live test |
| P0 #2 | Director JSON schema validator | Code complete, needs live test |
| P1 #5 | Length-sorted Bark batching | Code complete, needs live test |
| P1 #7 | VRAM-Sentinel decorator | Code complete, needs live test |
| P1 #9 | High-creativity soak profile | Code complete, needs live test |
| P2 #12 | Per-LLM-call VRAM snapshots | Code complete, needs live test |

---

## Discarded ideas (do not revisit)

- Flash Attention 2/3 on this platform
- Pinning torch>=2.5.1 (stale by 5 minor versions)
- Weight streaming from system RAM via ComfyUI-Manager
- Speculating on unreleased HyWorld unified latent space
- Asynchronous weight streamer as a fallback for 16 GB OOM
- "Shift Bark to HuggingFace implementation" (already on it)

Full discard reasoning in `2026-04-15-hyworld-integration-plan-review.md`.

---

## References

- docs/BUG_LOG.md — live bug tracking
- `docs/2026-04-15-hyworld-integration-plan-review.md` — full external review triage
- `ComfyUI_Hyworld_Narrative_Integration_Plan_v2_5.md` — HyWorld integration master plan
- `docs/2026-04-12-otr-v2-visual-sidecar-design.md` — v2 design spec
- `docs/2026-04-14-otr-v2.1-spec.md` — v2.1 spec
- `docs/2026-04-14-green-zone-guardrail-decision.md` — guardrail decision
- CLAUDE.md — platform pins, standing rules
