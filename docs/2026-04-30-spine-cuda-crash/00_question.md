# OTR v2.0-alpha — CUDA crash diagnosis after genre_flavor → style consolidation

**Branch:** `v2.0-alpha`
**Hardware:** RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, Windows
**Stack:** Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA. 100% local.
**Internal QA only** — no cloud LLMs in the shipped workflow.

---

## The crash

ComfyUI Desktop console (the user reported, not captured in otr_runtime.log):

```
[OpenClose] SPINE SCIENCE-DRIVEN failed: CUDA error: unknown error
Search for `cudaErrorUnknown` in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

**Key user statement:** *"the news never crashed before"* — this run pattern was working before today's code changes.

---

## What ran before the crash (otr_runtime.log slice)

```
[20:32:03] PARAMS open_close=True self_critique=True custom_premise=(empty)
           target_words=100 chars=2 length=short (3 acts)
           style=noir mystery creativity=maximum chaos arc_enhancer=True
[20:32:04] VRAM_RESET phase=NewsCuration
[20:32:07] Quantizing: 4-bit NF4 for google/gemma-4-E4B-it
[20:32:23] LLM loaded: google/gemma-4-E4B-it (quantized=True, budget=Pro (Ultra Quality))
[20:32:23] WARMUP: 1-token CUDA kernel warmup OK in 0.5s
[20:32:25] ScriptWriter DONE: 14 tokens in 1.4s (NewsCuration LLM call OK)
[20:32:25] ScriptWriter DONE: 2 tokens in 0.3s (NewsCurationDeep OK)
[20:32:25] NEWS_SUMMARY: Summarizing 'Study: Infrasound...' (1510 chars) via LLM
[20:33:00] ScriptWriter DONE: 317 tokens in 35.0s (NewsSummary OK)
[20:33:00] OPENCLOSE: Generating 3 competing outlines
[20:33:00] LLM cache mismatch (Context Cap: 16384) -- reloading to enforce budget
           [fields drifted: budget_profile: 'Pro (Ultra Quality)' -> 'Standard']
[20:33:04] [StoryOrchestrator] Zero-Prime: ComfyUI Models Evicted.
[20:33:06] Quantizing: 4-bit NF4 for google/gemma-4-E4B-it (RELOAD)
[20:33:16] LLM loaded: google/gemma-4-E4B-it (quantized=True, budget=Standard)
[20:33:16] WARMUP: 1-token CUDA kernel warmup OK in 0.1s
[20:33:50] VRAM_RESET phase=OpenClose-SPINE-SCIENCE-DRIVEN
           ^-- crash on the next LLM call (SPINE 2 of 3, SCIENCE-DRIVEN)
```

**Pattern:**
- 4 successful LLM generations on Gemma-4-E4B (NewsCuration / NewsCurationDeep / NewsSummary / SPINE-CHARACTER-DRIVEN)
- Between SPINE 1 and SPINE 2, code calls `VRAM_RESET phase=OpenClose-SPINE-SCIENCE-DRIVEN` (probably eviction + state reset)
- SPINE 2 then attempts to generate and the LLM call raises CUDA error: unknown error

---

## What I changed today (could one of these have caused this?)

13 commits today on `v2.0-alpha`. The relevant ones for the LLM stack:

### Commit `4e30d82` — genre_flavor + style_variant consolidation (LARGEST CHANGE)

Touched 17 files; in `nodes/story_orchestrator.py`:
- Removed `genre_flavor` widget from INPUT_TYPES dropdown
- Removed `genre_flavor` parameter from `write_script()` signature (was positional arg #2)
- **Bulk renamed every `genre_flavor` → `style` in the file** (function vars, dict keys, log strings, format placeholders in ~12 LLM prompt templates)
- Added new `style_custom` widget + parameter (free-text override; if non-empty, replaces dropdown preset before downstream use)
- Added bridge code at top of write_script: `if style_custom and style_custom.strip(): style = style_custom.strip()`

Workflow JSON (`otr_scifi_16gb_full.json`) widget values restructured:
- Dropped `"hard_sci_fi"` at slot [1]
- Inserted `""` for `style_custom` at slot [11]
- Net same count (15 entries), positions shifted

The bridge code runs BEFORE `target_words = int(target_words)` and BEFORE any LLM operation.

### Commit `f0421c2` — P1 hardening (2 files outside LLM stack: scene_sequencer + video_composite)
### Commit `9c55773` — Step 6 VideoComposite (no story_orchestrator touch)
### Commit `088177c` — speaker_role taxonomy (no story_orchestrator touch in significant way)

---

## The exact bridge code I added (write_script lines ~4230)

```python
def write_script(self, episode_title,
                 target_words, num_characters, model_id="...",
                 cleanup_model_id="...",
                 custom_premise="", news_headlines=3, temperature=0.8,
                 include_act_breaks=True, self_critique=True,
                 open_close=True,
                 target_length="medium (5 acts)",
                 style="tense claustrophobic",
                 style_custom="",
                 creativity="balanced",
                 arc_enhancer=True,
                 project_state=None,
                 optimization_profile="Standard"):
    force_lemmy = False

    # 2026-04-30 STYLE OVERRIDE: when style_custom is non-empty,
    # it replaces the dropdown preset for ALL downstream use.
    if isinstance(style_custom, str) and style_custom.strip():
        _override_tone = style_custom.strip()
        _runtime_log(
            f"ScriptWriter: STYLE_OVERRIDE preset {style!r} replaced "
            f"by style_custom={_override_tone!r}"
        )
        style = _override_tone

    target_words = int(target_words)
    ...
```

---

## My questions

1. **Could the `genre_flavor → style` bulk rename have introduced a subtle Python-level bug** (KeyError, NameError, late binding) that fires in SPINE 2 specifically (after a successful SPINE 1) and surfaces as cudaErrorUnknown on the next CUDA call? Bulk replace_all on `genre_flavor → style` was applied to story_orchestrator.py, script_critic.py, video_engine.py, otr_video_plan.py, batch_flux_render.py, and several scripts.

2. **The widget-order shift in workflow JSON** — `widgets_values` array slots shifted by 1 in some positions (genre_flavor removed at [1], style_custom inserted at [11]). ComfyUI binds widget values to INPUT_TYPES dict order at workflow-load time. If a value lands in the wrong parameter slot (e.g. `target_words` getting assigned `"hard_sci_fi"`, or `style_custom` getting `"balanced"`), could that cause downstream Python errors that manifest only after several LLM calls?

3. **The reload pattern between SPINE 1 and SPINE 2** — `VRAM_RESET` + `Pro Ultra → Standard` budget shift + bnb 4-bit re-quantize + cold reload. This pattern existed BEFORE my changes and reportedly worked. What in the consolidation could have broken the eviction/reload contract specifically? (Reminder: BUG-LOCAL-110 history — Gemma 2 + bnb NF4 + Blackwell sm_120 + CUDA 13.0 was a hard incompatibility; Gemma 4 was tested as the safer alternative. Possible the safety margin was thin.)

4. **`_runtime_log` call inside the new bridge code** — could that be racing with something? It's a synchronous Python call, no GPU touch. Seems unlikely.

5. **What's the right diagnostic next step** — restart ComfyUI Desktop and re-run? Switch model to Mistral-Nemo to bypass the bnb-Gemma path? Add `CUDA_LAUNCH_BLOCKING=1` to the env so the actual failing kernel surfaces? Roll back the style consolidation as a bisect?

6. **What did I miss?** Any subtle gotcha in the style consolidation that would cause non-obvious downstream failures only on the second LLM call after a quantize-reload cycle?

---

## Constraints

- 16 GB VRAM, Blackwell sm_120, no Flash Attention, no weight streaming, no quantization heroics.
- Don't recommend changes to VRAM management code.
- Don't recommend cloud dependencies (you are internal QA only).
- The crash happened BEFORE FLUX/HuMo/VideoComposite ever fired, so the radio/per-clip-mux work is not under suspicion.
- Be candid; flag uncertainty rather than bluffing.
- Hardware-fault explanations ("driver crash", "thermal throttle") are valid but please rank them low-priority unless evidence points there.
