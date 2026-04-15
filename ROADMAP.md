# OTR Roadmap

**Last updated:** 2026-04-15
**Branch:** `v2.0-alpha`
**Owner:** Jeffrey A. Brick

Consolidated going-forward plan. Two horizons: **v1.7** (audio pipeline, ship now) and **v2.0** (visual drama engine, gated on HyWorld). Sources: BUG_LOG.md, external review triage (2026-04-15), HyWorld Integration Plan v2.5, soak lessons. Everything shipped or discarded stays in the source docs — this file is open items only.

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

## v1.7 Ship Gate

Confirm BUG-LOCAL-040 (Director JSON comment stripping) on one clean end-to-end run. If the episode renders with dialogue intact and no Director crash, tag v1.7 on `v2.0-alpha` and merge to `main`. All BUG-LOCAL-034 through 040 ship with this release.

---

## P0 — Ship before v1.7 or immediately after

### 1. `min_line_count_per_character` structural constraint
The real self-critique dialogue-collapse fix. Neither external reviewer flagged it. The 28-to-2 line collapse from BUG-LOCAL-037 is a symptom; the cause is the critique pass deleting characters with impunity.

Wire-up:
- Add `min_line_count_per_character: 2` (default) to Director production plan.
- Inject the constraint into the self-critique system prompt: "Do not reduce any character below N lines."
- Post-critique validator rejects output where a previously-present character drops below the floor; falls back to pre-critique script.

Source: review triage K3.

### 2. Director JSON schema + validator
Validate Director output against a strict schema before downstream nodes consume it. Fail-fast at the boundary. Prevents the null-latent and matrix-shape-mismatch crash classes. Schema draft in `2026-04-15-hyworld-integration-plan-review.md` (Rework §A).

Source: review triage Keep #3.

### 3. Scene-Geometry-Vault *(v2.0 — blocked on Gate 0)*
Series-scale visual continuity bet. Without a persistent geometry vault, Act 3's bridge will not match Act 1's bridge across episodes. Design only after Gate 0 empirical measurements exist.

Source: review triage Keep #6.

### 4. Style-Anchor cache (World Seed + Lighting/Mood split) *(v2.0 — blocked on Gate 0)*
Turns the vault (#3) into a reuse engine. Same geometry, N relight passes (Day vs Night, Tense vs Calm). `style_anchor_hash` field in the Director schema keys the split.

Source: review triage Keep #2.

---

## P1 — Ship once P0 is stable

### 5. Length-sorted Bark batching
Bark pads to the longest sequence in the batch. Sort by token length within each voice-preset group, generate, re-sort by scene position afterward. Pure throughput win, no quality risk. `BatchBark` already groups by preset — add one sort step.

Source: Bark optimization triage B1.

### 6. Head-Start async pre-bake (Phase B.5) *(v2.0 — blocked on Gate 0)*
While ScriptWriter and Director run, kick off HyworldBridge on `outline_json` so geometry generation overlaps the LLM passes. Wall-clock win. Blocked on Gate 0 and vault stability.

Source: review triage Keep #1.

### 7. VRAM-Sentinel decorator on bark utils
Formalize `_flush_vram_keep_llm()` as a decorator that asserts "Director weights cleared before I run" on every Bark call. Cheap defensive depth. Codifies existing handoff discipline.

Source: review triage Keep #5.

### 8. ASCII sanitizer in prompt_compiler *(v2.0 — blocked on Gate 0)*
Strip non-ASCII (smart quotes, em-dashes) before Tencent text encoders. Preserve case. Collapse whitespace. Reference implementation in `2026-04-15-hyworld-integration-plan-review.md` (Rework §B).

Source: review triage Keep #4.

### 9. High-creativity soak profile
Add one soak variant at `creativity="maximum"` to verify TITLE resolution and WORD_ENFORCEMENT hold under high LLM temperature. Low effort; catches temperature-sensitive regressions.

Source: review triage K1.

---

## P2 — Experiments and cleanup

### 10. `torch.compile` on Bark sub-models
`torch.compile(mode="reduce-overhead")` on semantic, coarse, and fine acoustic models. Needs isolated A/B timing — variable-length loops may fight the compiler. Benefit is batch runs with many lines per preset.

Source: Bark optimization triage B2.

### 11. Skip/shorten Bark fine acoustic pass
The fine pass adds high-frequency detail that AudioEnhance then destroys via tape emulation, LPF, Haas delay. If A/B blind test confirms quality loss is inaudible post-processing, truncating the fine pass cuts per-line wall time. Needs listening test, not spectrogram.

Source: Bark optimization triage B3.

### 12. Per-LLM-call VRAM snapshots in soak_operator
Extend `VRAM_SNAPSHOT phase=director_exit` to every LLM call. Reuses `_vram_log`. Cheap defensive observability.

Source: review triage K2.

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

| Bug | Summary | Commit |
|---|---|---|
| BUG-LOCAL-034 | Streak auto-halt 5→3 | `ee67d9c` |
| BUG-LOCAL-035 | TITLE_STUCK fix | `ee67d9c` |
| BUG-LOCAL-036 | WordExtend NameError | `ee67d9c` |
| BUG-LOCAL-037 | TITLE-as-character regression | `36d13aa` |
| BUG-LOCAL-038 | Bare NAME: dialogue parser (4-part) | `27e41a4` |
| BUG-LOCAL-039 | Markdown bold leak in title | `7ff8fa7` |
| BUG-LOCAL-040 | JS-style comments in Director JSON | `2828338` |

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

- BUG_LOG.md — live bug tracking
- `docs/superpowers/specs/2026-04-15-hyworld-integration-plan-review.md` — full external review triage
- `ComfyUI_Hyworld_Narrative_Integration_Plan_v2_5.md` — HyWorld integration master plan
- `docs/superpowers/specs/2026-04-12-otr-v2-visual-sidecar-design.md` — v2 design spec
- `docs/superpowers/specs/2026-04-14-otr-v2.1-spec.md` — v2.1 spec
- `docs/superpowers/specs/2026-04-14-green-zone-guardrail-decision.md` — guardrail decision
- CLAUDE.md — platform pins, standing rules
