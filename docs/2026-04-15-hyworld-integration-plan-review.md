# HyWorld Integration Plan - Review & Revisions

**Date:** 2026-04-15
**Reviewer:** Claude (Cowork mode)
**Source:** external review of the HyWorld Integration Plan v2
**Verdict:** six keepers, five throwaways, two need rework. Reviewer has good instincts on caching and async, but hallucinates the hardware stack in several places. Below is the sorted, revised version.

---

## Platform Pins (ground truth)

Before triaging recommendations, lock what is true about this stack. Any advice that contradicts this list should be discarded on sight.

- Hardware: RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Stack: Windows, Python 3.12, torch **2.10.0**, CUDA 13.0, SageAttention + SDPA.
- **Flash Attention 2 and 3 are NOT available on this platform.** No prebuilt wheel exists for torch 2.10 + CUDA 13 + sm_120 + Windows. Blackwell FA support is delayed upstream. Do not chase it.
- 100% local, offline-first, open source, no API keys, no paid services.
- VRAM ceiling: 14.5 GB real-world target.

---

## Keep (6 ideas worth shipping)

### 1. Head-Start async pre-bake
While ScriptWriter and Director run, kick off `HyworldBridge` on `outline_json` so geometry generation overlaps the LLM passes. On a 16 GB GPU, wall-clock is the scarce resource, not VRAM. Worth formalizing as its own phase contract in the pipeline (Phase B.5: async geometry warm-up).

### 2. Style-Anchor persistent layer
Split the cache into two layers:
- **World Seed** - the underlying 3DGS / mesh geometry (expensive, rarely changes within an episode)
- **Lighting/Mood Pass** - relight + palette applied on top (cheap, varies per scene)

Same bridge interior can be reused Day vs. Night, Tense vs. Calm, with one geometry bake and N relight passes. Single best idea in the external review for a series-scale product. Maps cleanly to the existing ProjectState continuity work.

### 3. Fail-fast JSON schema validation at the orchestrator boundary
Validate Director output against a strict schema before `prompt_compiler.py` consumes it. Same philosophy as the BUG-LOCAL-038 fix - validate at the boundary, not three nodes downstream when an `UnboundLocalError` fires cryptically. Low-cost, high-signal. Prevents the null-latent and matrix-shape-mismatch classes of crash.

### 4. ASCII-sanitize text inputs to HyWorld's encoder
Tencent text encoders are known to choke on smart quotes, em-dashes, and other non-ASCII. Stripping those before the encoder consumes them is cheap defensive hygiene. **Drop the `.lower()` step** from the reviewer's version - case is usually fine and lowercasing loses proper-noun cues (e.g. `COMMANDER` vs. `commander` as a speaker hint).

### 5. VRAM-Sentinel decorator on bark utils
Formalize the existing `_flush_vram_keep_llm()` pattern as a decorator that asserts "Director weights cleared before I run" on every Bark call. Catches regressions before they OOM. Aligns with the current VRAM handoff discipline.

### 6. Scene-Geometry-Vault as P0
Correct prioritization for multi-episode visual continuity. Without a persistent geometry vault, Act 3's bridge will not match Act 1's bridge across episodes. This is the structural-continuity bet; underinvesting here kills series coherence.

---

## Discard (5 ideas wrong for this stack)

### 1. Flash Attention 3
No FA2 on this box means no FA3. Wheel does not exist for torch 2.10 + CUDA 13 + sm_120 + Windows. Documented in `CLAUDE.md` for a reason. Ignore any reviewer recommending FA on this hardware.

### 2. "Pin `torch>=2.5.1`"
This stack is on torch 2.10 - the recommendation is stale by two minor versions. The general principle (keep torch modern on Blackwell) is fine; the specific pin is noise.

### 3. "Stream Weights from system RAM via ComfyUI-Manager"
ComfyUI-Manager does not do this. HuggingFace `accelerate` with `device_map="auto"` does CPU offload, but the perf hit on a single 16 GB GPU usually turns a 30-second render into 5 minutes. Not the rescue plan the reviewer implies.

### 4. "If 2.0 has Unified Latent Space for Geometry and Video, skip Blender/Rhubarb"
Speculating about unreleased features. Plan around shipped capabilities, not rumored ones. Keep an escape hatch but do not rearrange the roadmap around a phrase that might appear in a tech report.

### 5. Asynchronous Weight Streamer as P1
Solution in search of a problem. If HyWorld 2.0 genuinely does not fit 16 GB, the correct fallback is to stay on 1.5 until a real fit is validated. Complex weight-streamers ship their own bugs; do not add that surface area on spec.

---

## Rework

### A. Scene schema (trimmed)

The reviewer's schema was both over- and under-specified: `scene_id` locked the format to a concatenated SHA prefix, the string-length caps were arbitrary, and the required-field set did not match what the OTR Director actually emits. Leaner version:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "OTR_Director_Scene_Schema",
  "type": "object",
  "required": ["scene_id", "episode_fingerprint", "setting", "mood", "time_of_day"],
  "properties": {
    "scene_id":            { "type": "string", "pattern": "^scene_[0-9]{2,3}$" },
    "episode_fingerprint": { "type": "string", "pattern": "^[a-f0-9]{12}$" },
    "setting":             { "type": "string", "minLength": 1, "maxLength": 250 },
    "mood":                { "type": "string", "minLength": 1, "maxLength": 150 },
    "time_of_day":         { "type": "string", "minLength": 1, "maxLength": 100 },
    "genre":               { "type": "string", "maxLength": 100 },
    "tone":                { "type": "string", "maxLength": 150 },
    "pacing":              { "type": "string", "maxLength": 100 },
    "production_style":    { "type": "string", "maxLength": 150 },
    "style_anchor_hash":   {
      "type": "string",
      "pattern": "^[a-f0-9]{12}$",
      "description": "Keys the geometry bake so Day/Night reuses the World Seed."
    }
  },
  "additionalProperties": false
}
```

Key changes vs. the reviewer's version:
- `scene_id` and `episode_fingerprint` are split. Scene id stays human-readable; fingerprint lives once per scene, not concatenated into the id.
- Required fields pruned to what actually gates render (`setting`, `mood`, `time_of_day`). `genre` / `tone` / `pacing` / `production_style` are optional because Director does not always emit them.
- New `style_anchor_hash` field makes keeper #2 (Style-Anchor cache) first-class in the schema.
- `additionalProperties: false` retained - fail fast on drift.

Save as `schema_director_scene.json` and wire into whatever JSON-validation node sits between Director and `prompt_compiler.py`.

### B. Sanitizer (revised)

```python
import re

def sanitize_for_hyworld(prompt_dict):
    """Strip non-ASCII and collapse whitespace. Preserves case for proper nouns.

    Tencent text encoders choke on smart quotes, em-dashes, and control
    characters. Removing those before the encoder sees them prevents
    token-count inflation and null-latent errors. Lowercasing is NOT
    applied - case carries speaker/proper-noun signal downstream.
    """
    clean = {}
    for k, v in prompt_dict.items():
        if isinstance(v, str):
            s = re.sub(r'[^\x00-\x7F]+', '', v)   # drop non-ASCII
            s = re.sub(r'\s+', ' ', s).strip()     # collapse whitespace
            clean[k] = s
        else:
            clean[k] = v
    return clean
```

Changes vs. the reviewer's version:
- Dropped the `.lower()` call.
- Collapsed whitespace runs (catches stray `\r\n` and double-spaces).
- Preserved non-string values so the function is safe to call on the whole dict.

---

## Updated "To-Build" Priority Matrix

| Component | Priority | Reason |
|---|---|---|
| Scene-Geometry-Vault | **P0** | Series-scale continuity bet. Without it, Act 3 will not look like Act 1. |
| Style-Anchor cache key + relight pass | **P0** | Turns the vault into a reuse engine instead of a one-shot cache. |
| Director JSON schema + validator | **P0** | Fail-fast at the boundary. Blocks the null-latent and matrix-shape-mismatch crash classes. |
| Head-Start async pre-bake (Phase B.5) | **P1** | Wall-clock win. Worth shipping once the vault is stable. |
| VRAM-Sentinel decorator on bark utils | **P1** | Cheap defensive depth. Codifies existing handoff discipline. |
| ASCII sanitizer in `prompt_compiler.py` | **P1** | Low-cost hygiene. Prevents encoder token inflation. |
| Asynchronous Weight Streamer | **DROPPED** | Not needed. If 2.0 does not fit 16 GB, stay on 1.5. |
| Flash Attention 3 integration | **DROPPED** | Not available on this platform. Do not chase. |

---

## Next Concrete Artifact

When HyWorld 2.0 drops:

1. Run the 16 GB fit test against a single-scene outline.
2. If it fits - land the Director JSON schema + validator FIRST. This is the gate every downstream node will assume.
3. Then land the Scene-Geometry-Vault with the Style-Anchor split (World Seed vs. Lighting/Mood). Without the split, the vault is just a disk-full trap.
4. Only after #2 and #3 are green: start on Head-Start async pre-bake.

If 2.0 does not fit - stay on 1.5, log the OOM signature, and wait for the upstream quantization / distillation pass that will eventually arrive. Do not try to force-fit via weight streaming.

---

## References

- OTR v2 Design Spec: `docs/2026-04-12-otr-v2-visual-sidecar-design.md`
- OTR v2.1 Spec: `docs/2026-04-14-otr-v2.1-spec.md`
- Green Zone Guardrail Decision: `docs/2026-04-14-green-zone-guardrail-decision.md`
- Platform pins: `../../CLAUDE.md` (`Platform Reality` section)

---

# Addendum: v2.0-alpha Ship-Readiness Review

**Date:** 2026-04-15
**Source:** second external review (unnamed) focused on v2.0-alpha ship criteria
**Verdict:** ~30-40% useful. Most of the "critical fixes" are already shipped in BUG-LOCAL-034/035/036/037/038. The "v2.0.1 schema-native branch" section invents repo structure that does not exist. Kept below is only what is actionable and not already done.

## Already shipped (discarded from review)

- **Streak auto-halt 5 → 3 runs.** Already at 3 (`FATAL_STREAK_LIMIT=3` in BUG-LOCAL-034).
- **WordExtend short-circuit / no-crash.** Already fixed - BUG-LOCAL-036 resolved the NameError; extension pass runs cleanly.
- **Workflow JSON "The Last Frequency" removal.** All three defaults cleared in BUG-LOCAL-035.
- **Tok/s in ScriptWriter heartbeat.** Already present (`ScriptWriter DONE: N tokens in Xs (Y tok/s)...`).

## Hallucinated infrastructure (discarded)

None of the following exist in the repo; the review presumed them:
- "C4 skeleton writer" node with GBNF grammar (you are on HF Transformers, not llama.cpp).
- "C5 refiner" node with per-character token budget.
- "C7 voice mapper" with `gender_ratio_target` / `recency_penalty` widgets.
- "C8 fallback rate limit" (predicated on the skeleton writer that does not exist).
- "v2.0.1 schema-native branch" (no such branch).
- "RuntimeError with exact widget line numbers 23, 301, 432" (suspiciously specific - `TITLE_RESOLVE_FAIL` already fires loud with real context).

## Kept (worth doing)

### K1. High-creativity soak profile
Add one soak variant that runs at `creativity="maximum"` (highest LLM temperature profile) to verify TITLE resolution and WORD_ENFORCEMENT hold up under high-randomness. Low effort; catches temperature-sensitive regressions. **Priority: P1.**

### K2. Per-LLM-call VRAM snapshots in soak_operator
Extend the existing `VRAM_SNAPSHOT phase=director_exit` pattern to cover every LLM call, not just director exit. Reuses existing `_vram_log`. Cheap defensive logging that would have caught a lot of historical OOM regressions earlier. **Priority: P2.**

### K3. `min_line_count_per_character` structural constraint
Episode-level constraint wired into the self-critique prompt. Directly addresses the self-critique dialogue-collapse failure mode (the 28 → 2 line collapse seen in BUG-LOCAL-037 symptom and not yet fully solved). Makes SINGLE_LINE_CHAR structurally harder for the critique pass to produce.

Proposed wire-up:
- Add `min_line_count_per_character: 2` (default) to the Director production plan.
- Inject the constraint into the self-critique system prompt so the model is told explicitly: "Do not reduce any character below N lines."
- Post-critique validator rejects any output where a previously-present character drops below the floor, and falls back to the pre-critique script.

**Priority: P0** - this is the real dialogue-collapse fix, and neither external reviewer flagged it by name.

### K4. `episode_title` socket input on `OTR_SignalLostVideo`
Today the video node reads the title from `script_json`'s title token (BUG-LOCAL-035 design). A cleaner v2.1 design is an explicit socket input wired from ScriptWriter's resolved title. Removes the implicit coupling on `script_json` shape and lets the video node fail-loud if the wire is missing. Delete the widget default entirely at the same time. **Priority: P2, v2.1 work (not alpha-blocking).**

## Rejected with caveats (half-good ideas)

- **B3 TITLE_TRACE as JSON.** Structural improvement, but current `TITLE_TRACE | source=... | resolved='...'` is already one-grep parseable. Switching to JSON breaks `_qa_grep.py` key-matching and forces every downstream consumer to import `json`. Not worth it unless something external needs to ingest the log programmatically.
- **Stop condition 5 → 10 consecutive clean runs.** Fine in principle, but base it on observed soak failure-rate data, not a round number. Defer until the soak has enough history to justify the bar.
- **30-second tok/s benchmark node at run start.** Already effectively covered by `ScriptWriter DONE`'s tok/s output + existing VRAM_SNAPSHOT. A dedicated node is overkill; a `tok/s` line added to the first heartbeat would be a strictly smaller change.
- **Schema validation pass rate ≥ 98% as a stop condition.** Presumes the schema-native branch exists. Can't gate on something that isn't built.

## Updated v2.0-alpha ship-readiness matrix

| Item | Priority | Status | Notes |
|---|---|---|---|
| BUG-LOCAL-034 streak auto-halt | - | DONE | Shipped in `ee67d9c`. |
| BUG-LOCAL-035 TITLE_STUCK | - | DONE | Shipped in `ee67d9c`. |
| BUG-LOCAL-036 WordExtend NameError | - | DONE | Shipped in `ee67d9c`. |
| BUG-LOCAL-037 TITLE-as-character regression | - | DONE | Shipped in `36d13aa`. |
| BUG-LOCAL-038 bare NAME: dialogue parser | - | DONE | Shipped in `27e41a4`. |
| K3 `min_line_count_per_character` constraint | **P0** | OPEN | Self-critique dialogue-collapse fix. Needs ticket. |
| K1 creativity="maximum" soak profile | **P1** | OPEN | Low effort regression catcher. |
| K2 per-LLM-call VRAM snapshots | **P2** | OPEN | Defensive observability. |
| K4 `episode_title` socket on video node | **P2** | OPEN | v2.1 cleanup, not alpha-blocking. |

## Pattern note across both reviews

Both external reviews on this date (HyWorld integration + v2.0-alpha ship-readiness) share the same tell: confident-sounding roadmaps that invent repo structure (skeleton writer, refiner, weight streamer, unified latent space). The useful content is always about (a) schema validation at boundaries, (b) cache-layer splits, (c) async overlap. Everything else is generic LLM-pipeline advice that may or may not fit this stack.

Future external reviews worth engaging with will name specific files, specific commits, or specific BUG-LOCAL-NNN entries. Reviews that do not will be triaged at roughly this same hit rate.

---

# Addendum 2: Bark Batch-TTS Optimization Triage

**Date:** 2026-04-15
**Source:** third external review, focused on Bark generation throughput on RTX 5080
**Verdict:** ~50% useful. Better hit rate than the first two reviews, but still invents the platform (FA, torch version) and does not know OTR is already on the HuggingFace `transformers` Bark implementation.

## Discarded

### "Shift to the HuggingFace implementation"
OTR is already on it. Logs show `BarkModel on cuda` — that is the `transformers` class, not the Suno repo. The reviewer assumed we were wrapping the legacy Suno code. The FlashAttention sub-recommendation (`attn_implementation="flash_attention_2"`) is the same dead horse: FA2/FA3 does not exist for torch 2.10 + CUDA 13 + sm_120 + Windows. **Discard.**

## Kept

### B1. Length-sorted batching within voice-preset groups
**Priority: P1.** Bark pads to the longest sequence in the batch. Mixing a 3-word line with a 30-word line wastes GPU cycles on padding the short one. `BatchBark` already groups lines by voice preset — adding a token-length sort within each group is a small, self-contained change. Generate in length-sorted order, then re-sort by scene position using existing metadata before handoff to `SceneSequencer`. No quality impact, pure throughput win.

### B2. `torch.compile` on Bark sub-models
**Priority: P2 (experiment).** Reviewer cited "PyTorch 2.4.0" — we are on **2.10.0**, so the version is wrong, but `torch.compile(mode="reduce-overhead")` on the semantic, coarse, and fine acoustic models is a valid idea in principle. Caveat: Bark's generation uses variable-length loops internally which can fight the graph compiler. Needs isolated A/B timing before wiring into the pipeline. First-pass cost is high; benefit is only on batch runs with many lines per preset.

### B3. Skip or shorten the fine acoustic pass for OTR lo-fi output
**Priority: P2 (experiment).** Bark generates in three stages: semantic → coarse → fine. The fine pass adds high-frequency detail that OTR's `AudioEnhance` node then destroys via tape emulation, LPF, Haas delay, and bass warmth. If A/B testing confirms the quality loss is inaudible after post-processing, truncating or skipping the fine pass could cut per-line wall time significantly. Needs blind listening test, not just spectrogram comparison.

## Updated Bark optimization matrix

| Item | Priority | Status | Notes |
|---|---|---|---|
| B1 Length-sorted batching | **P1** | OPEN | Pure throughput win, no quality risk. |
| B2 `torch.compile` on sub-models | **P2** | OPEN | Needs isolated timing. Variable-length loops may fight compiler. |
| B3 Skip/shorten fine acoustic pass | **P2** | OPEN | Needs blind A/B test post-AudioEnhance. |

## Reference

- HyWorld Integration Plan v2.5: `ComfyUI_Hyworld_Narrative_Integration_Plan_v2_5.md`
- OTR v2 Design Spec: `docs/2026-04-12-otr-v2-visual-sidecar-design.md`
