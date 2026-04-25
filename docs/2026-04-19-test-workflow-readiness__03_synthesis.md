# Synthesis -- 2026-04-19

**Question:** # Consult: Is `workflows/otr_scifi_16gb_TEST.json` ready to fire?

## Context

OTR v2.0 ComfyUI custom node pack, branch `v2.0-alpha`. Jeffrey wants to run the TEST workflow against a live ComfyUI instance on Windows (RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, torch 2.10, CUDA 13). Audio is king and must not degrade.

## Current audit + regression state (fresh pass, 2026-04-19)

- AST parse of 102 .py files: **0 violations**
- UTF-8 BOM check: **0 violations**
- Node INPUT/OUTPUT contract audit (INPUT_TYPES ↔ RETURN_TYPES ↔ FUNCTION ↔ CATEGORY): **0 violations**, 21 node classes scanned
- Workflow ↔ NODE_CLASS_MAPPINGS cross-check: **0 violations**
- Regression suite (Bug Bible + dropdown guardrails + core + audio byte-identical): **183 passed, 2 skipped, 2 xfailed** (138s)

## Workflow shape

`workflows/otr_scifi_16gb_TEST.json` — 6 nodes, UI format. Uses 5 `OTR_Visual*` classes:
- `OTR_VisualBridge`
- `OTR_VisualLLMSelector`
- `OTR_VisualPoll`
- `OTR_VisualPromptCoercion`
- `OTR_VisualRenderer`

All 5 are registered in `__init__.py`.

## The open red flag

Task #72 is still open: `visual/backends/ltx_motion.py` line 226 calls:

```python
LTXImageToVideoPipeline.from_pretrained(
    _LTX_PATH,                         # points at models/diffusers/LTX-Video/
    torch_dtype=torch.float8_e4m3fn,
    local_files_only=True,
)
```

That folder is a partial HF snapshot **missing `model_index.json`**. We already confirmed (via `scripts/verify_ltx_hybrid.py`) that the working loader is the hybrid approach:
1. Build T5Config from `models/huggingface/hub/.../LTX-Video/.../text_encoder`
2. Instantiate empty T5EncoderModel + load weights from existing local `models/text_encoders/t5xxl_fp16.safetensors`
3. Build T5Tokenizer from the snapshot `tokenizer/` subfolder
4. `LTXImageToVideoPipeline.from_single_file("models/checkpoints/ltx-video-2b-v0.9.safetensors", text_encoder=t5, tokenizer=tok, torch_dtype=bfloat16)`

The hybrid verify script loads cleanly end-to-end on this machine. The production backend hasn't been patched to match.

When ltx_motion fails to load, it falls back silently to stub-mode (still frames + procgen overlay). That silent fallback is BUG-LOCAL-046, also still open.

## Unknowns I can't resolve without running

- Whether `OTR_VisualRenderer` actually dispatches into `ltx_motion` for this TEST workflow, or whether it short-circuits to `flux_anchor`-only / still-renderer given the TEST offline-asset injection (Task #77 rewrote the TEST JSON to sever LLM + inject offline assets + hardcode telemetry).
- Whether FLUX anchor weights are still present at the expected path.

## The question

Given:
- All static gates are green (audit 0/0/0/0, regression 183 pass).
- The workflow JSON is valid and every `OTR_*` class is registered.
- The ltx_motion loader is known-broken and will fall back to stubs **silently**.
- Audio-path tests are all passing and the audio spine has never been touched by the video work.

**Is it reasonable to fire this workflow as-is, or should we patch Task #72 first?**

Specifically:
1. What's the realistic chance of a clean end-to-end run without the ltx_motion patch?
2. What's the worst-case failure mode if we fire without patching? (silent stub fallback, hang, OOM, corrupted audio?)
3. Is there a cheap pre-flight check that would tell us which backend path `OTR_VisualRenderer` takes in this specific TEST workflow before we commit to firing?
4. If we do fire and get stubs back, is the fix still just the `from_single_file` swap, or should we also close BUG-LOCAL-046 (surface the silent fallback) at the same time?

Please give a grounded recommendation. Disagree freely if you think the regression-green signal is enough to justify firing.

---

## ChatGPT (gpt-5.4)

Short answer: **I would not treat `workflows/otr_scifi_16gb_TEST.json` as “ready to fire” for a meaningful visual validation until Task #72 is patched or you first prove this TEST workflow never touches `ltx_motion`.**  
If your goal is only “does ComfyUI execute the graph without crashing and does audio remain intact,” then **yes, you can fire it now** with low risk to audio.  
If your goal is “does the TEST workflow validate the intended video path,” then **no, not yet**.

## My recommendation

**Best smallest-change path:**

1. **Do one cheap pre-flight to determine whether `OTR_VisualRenderer` can dispatch into `ltx_motion` for this exact TEST JSON.**
2. If **no**, go ahead and fire as-is.
3. If **yes or unclear**, **patch Task #72 first**.
4. While touching that code, I would **also close BUG-LOCAL-046** or at least convert the silent fallback into an unmistakable warning/error. That is a very high-payoff, low-risk change.

That is the smallest change with the largest payoff.

---

## Why I’m not comfortable calling it “ready”

Your green static/regression state is strong, but it does **not** cover the known production loader defect:

- `visual/backends/ltx_motion.py:226` uses:
  ```python
  LTXImageToVideoPipeline.from_pretrained(...)
  ```
- You already know the local asset layout is incompatible because the snapshot is partial and **missing `model_index.json`**.
- You already have a verified working load path via `from_single_file(...)` plus injected text encoder/tokenizer.
- Production backend is **not** using that verified path yet.
- Failure currently **degrades silently into stub mode**.

That means the workflow can appear to “work” while actually bypassing the intended motion backend. For a TEST workflow, that’s exactly the kind of false green you want to avoid.

---

## 1. Realistic chance of a clean end-to-end run without the `ltx_motion` patch

I’d split this into two meanings of “clean”:

### A. Clean = graph executes, no crash
**Moderate to high chance**, probably **70–90%**, assuming:
- ComfyUI sees all 5 `OTR_Visual*` nodes,
- offline assets injected by Task #77 are valid,
- FLUX/still path assets exist if used,
- no unrelated pathing issue.

Why that high:
- Static audits are clean.
- Regressions are green.
- The known broken path appears to fail into a stub fallback rather than hard-crash.

### B. Clean = intended visual backend runs correctly
**Low confidence unless you prove `ltx_motion` is not selected.**

If this TEST workflow can route into `ltx_motion`, then the chance of a **true** clean run through that backend is, candidly, **poor**. I’d estimate **<20%**, and honestly maybe near **0%** if the code path definitely reaches `from_pretrained()` against that incomplete snapshot.

The only way I’d raise that estimate is if:
- `OTR_VisualRenderer` never selects `ltx_motion` for this TEST JSON, or
- there is some guard/short-circuit already forcing still mode.

So the answer depends almost entirely on dispatch.

---

## 2. Worst-case failure mode if you fire without patching

### Most likely worst case
**Silent stub fallback**:
- still frames,
- procgen overlay,
- no real LTX motion generation,
- run looks “successful” unless you inspect logs/output carefully.

Given BUG-LOCAL-046, this is the main operational risk.

### Less likely but plausible
**Backend load exception handled badly enough to waste time or confuse state**
- long startup delay while trying to initialize the broken loader,
- noisy logs,
- partial render artifacts,
- maybe a failed visual segment but completed workflow.

### OOM / VRAM risk
Possible, but **not my primary concern from the information you gave**.

Why:
- The known failure is a loader mismatch, not a “barely fits in 16 GB” issue.
- Your verified hybrid path uses `bfloat16`, which is more realistic on Blackwell than weird experimental `float8_e4m3fn` loading in this context.
- The current broken code may fail before heavy inference even starts.

That said, once you patch to the real LTX path, **VRAM may become the next issue**. On a 16 GB laptop GPU with a 14.5 GB ceiling, LTX video is exactly the sort of thing that can get tight. But that is a second-order problem after fixing the loader.

### Corrupted audio?
**Very unlikely based on your evidence.**

You said:
- audio byte-identical tests pass,
- audio spine untouched by video work,
- rule C7 is already guarded by regression.

So I do **not** see a credible path from this visual backend defect to audio corruption, unless there is some bizarre shared temp-file or muxing bug you haven’t mentioned. Based on what you provided, **audio risk is low**.

### Hang?
Possible but not the base case.  
If the loader retries, blocks on local file resolution, or some exception path is swallowed poorly, you could get an apparent stall. But the stronger signal you already have is “fails and falls back silently,” not “hangs forever.”

So my ranking of likely bad outcomes is:

1. **Silent stub fallback**  
2. **Misleading successful run with degraded visuals**  
3. **Visual-stage exception / partial failure**  
4. **Hang or OOM**  
5. **Audio corruption** — least likely

---

## 3. Cheap pre-flight check to tell which backend path `OTR_VisualRenderer` takes

Yes. This is what I would do before any run.

## Cheapest useful pre-flight: inspect renderer dispatch logic against the TEST JSON

You want to answer:

- What input fields in `workflows/otr_scifi_16gb_TEST.json` feed `OTR_VisualRenderer`?
- Which of those fields determine backend selection?
- Does that combination force still/FLUX, or can it choose `ltx_motion`?

### What to inspect
Search for the `OTR_VisualRenderer` class implementation and its execution function.

You already know the workflow is UI-format JSON with 6 nodes. So:

1. Open:
   - `workflows/otr_scifi_16gb_TEST.json`
2. Find the node whose `"type"` is:
   - `"OTR_VisualRenderer"`
3. Note its widget values / inputs / linked upstream nodes.
4. Open the renderer implementation and inspect:
   - `INPUT_TYPES`
   - `FUNCTION`
   - the method body that resolves backend selection
   - any helper it calls for backend choice

You didn’t provide the file path for `OTR_VisualRenderer`, so I won’t bluff one. But it should be straightforward to grep:

```powershell
rg -n "class OTR_VisualRenderer|OTR_VisualRenderer" .
```

Then inspect the method that actually runs.

### What you’re looking for
Any branch like:
- `if mode == "motion": use ltx_motion`
- `if backend == "ltx": ...`
- `if offline_assets_present: short_circuit_to_still`
- `if test_mode: force_flux_anchor`
- `if missing_llm: use injected still path`

Also grep for the backend import/use sites:

```powershell
rg -n "ltx_motion|flux_anchor|still|stub" visual . 
```

### Fastest possible answer without running
If the renderer has a deterministic branch based solely on node inputs and TEST JSON values, you can know the answer statically.

---

## Even cheaper: add one log line, then queue once

If static inspection is messy, the next-cheapest pre-flight is **instrumentation**, not a full patch.

Add a single explicit log line at the point where `OTR_VisualRenderer` selects the backend, e.g.:

```python
logger.warning("OTR_VisualRenderer selected backend=%s test_mode=%s offline_assets=%s", backend, test_mode, bool(offline_assets))
```

Then queue the workflow once.

This is low risk, high payoff, and does not alter audio behavior.  
If determinism rule C7 applies strictly to output bytes, a log line won’t affect audio bytes.

If you want zero code changes, you can still inspect existing logs, but given BUG-LOCAL-046, I would not trust current observability enough.

---

## 4. If you fire and get stubs back, is the fix still just `from_single_file`, or should you also close BUG-LOCAL-046?

## Task #72 fix
Yes, from what you described, the primary functional fix still sounds like:

- replace the broken `from_pretrained(_LTX_PATH, ...)`
- with the verified hybrid:
  - build `T5Config` from local snapshot,
  - instantiate/load `T5EncoderModel` from local safetensors,
  - build tokenizer from snapshot,
  - call `LTXImageToVideoPipeline.from_single_file(...)` with injected `text_encoder` and `tokenizer`,
  - use `torch_dtype=bfloat16` if that’s what your verification proved.

That is the core fix.

## BUG-LOCAL-046
I would **absolutely** address this at the same time, or at minimum partially address it.

Why:
- Silent fallback defeats the purpose of testing.
- It creates false confidence.
- It wastes operator time.
- It is probably a tiny code change compared to the loader patch.

### Minimal acceptable change
At minimum, change silent fallback to a **loud warning** with the exception reason and the exact fallback path taken.

Example:
```python
logger.error("LTX motion backend failed to load; falling back to stub mode: %s", exc)
```

### Better change
Make fallback behavior explicit and controllable:
- `allow_stub_fallback=True/False`
- in TEST mode, maybe default to **False** so failures surface immediately.

That would be my preferred behavior for validation workflows.

If Jeffrey wants the smallest possible change:
- **Do the loader patch**
- **Add a loud warning**
- Don’t redesign fallback policy yet

That gives most of the value with minimal surface area.

---

## Grounded recommendation

## If the goal is “safe to try once?”
**Yes, with caveats.**  
It is reasonable to fire **one exploratory run** as-is **only if everyone understands that a “successful” result may be fake because of silent stub fallback**.

I would not call that “ready,” but I would call it “safe enough for reconnaissance.”

## If the goal is “ready for meaningful validation?”
**No. Patch Task #72 first, unless pre-flight proves this TEST workflow never touches `ltx_motion`.**

That’s my actual recommendation.

---

## Practical decision tree

### Option A: fastest path, lowest effort
1. Inspect `OTR_VisualRenderer` dispatch or add one backend-selection log line.
2. If TEST workflow cannot hit `ltx_motion`:
   - **Fire now**
3. If it can hit `ltx_motion`:
   - **Patch Task #72 first**
   - ideally also make fallback noisy

### Option B: if you don’t want to inspect anything
Patch Task #72 now.  
Given you already have a verified local hybrid loader, this is the cleanest route.

---

## My candid call

Because BUG-LOCAL-046 means failure is **silent**, I would **not** rely on the current regression-green signal as sufficient justification to fire this workflow for anything beyond a smoke test.

So my answer is:

- **Reasonable to fire as a smoke test?** Yes.
- **Reasonable to declare “ready to fire” for actual validation?** No.
- **Best next step?** Pre-flight backend dispatch; if ambiguous, patch #72 before running.
- **Should #046 be fixed too?** Yes, at least make fallback loud in the same edit.

If you want, I can help you design the exact pre-flight grep/check sequence for `OTR_VisualRenderer` and the minimal patch shape for `visual/backends/ltx_motion.py` without touching anything audio-related.

---

## Gemini (gemini-3-pro-preview)

**1. Agreement**
PARTIALLY AGREE: The pre-flight check and patching Task #72 are the right moves, but the ChatGPT answer dangerously underestimates the VRAM/OOM risk of switching to `bfloat16` on a 16GB laptop.

**2. Factual Errors in the ChatGPT Answer**
*   **Hardware Ignorance:** It calls `torch.float8_e4m3fn` "weird experimental" loading. You are on an RTX 5080 Laptop (Blackwell sm_120 architecture). Blackwell natively supports and is highly optimized for FP8. It was almost certainly chosen specifically to keep this pipeline under your 14.5 GB VRAM ceiling.
*   **OOM Risk Dismissal:** It claims OOM is "not my primary concern." T5-XXL in FP16 is ~9.4 GB. LTX-Video in BF16 is ~4 GB. That’s 13.4 GB just for weights. Video generation requires significant VRAM for spatial-temporal attention context. If you run this in BF16 on a 16 GB card, you will almost certainly OOM during generation.

**3. Important Omissions**
*   **Diffusers vs. ComfyUI Memory Management:** `visual/backends/ltx_motion.py` is instantiating a raw HuggingFace Diffusers pipeline (`LTXImageToVideoPipeline`). Doing this inside a ComfyUI custom node bypasses ComfyUI’s native VRAM management (model pacing, offloading). This makes the 14.5 GB VRAM ceiling a hard, brittle wall.
*   **Process Crash Risk to Audio:** If the GPU OOMs hard during the video render, it throws a CUDA Out of Memory exception. If unhandled, this crashes the entire ComfyUI python process. If the audio nodes haven't finished and saved to disk yet, your audio generation dies with it.
*   **Loading vs. Generating:** You noted the hybrid script "loads cleanly end-to-end". Loading weights into VRAM is not the same as surviving the VRAM spike of a forward pass during video generation.

**4. My Short Recommendation**
*   **Do the 30-second pre-flight:** Open `workflows/otr_scifi_16gb_TEST.json` in a text editor. Look at the inputs for `OTR_VisualRenderer`. If it's hardcoded to offline assets/still frames, fire it right now.
*   **Patch Task #72 but keep FP8:** Update `visual/backends/ltx_motion.py` line 226 to use the hybrid `from_single_file` loader, but *revert* the dtype back to `torch_dtype=torch.float8_e4m3fn` for the LTX model. Do not use `bfloat16` as the verify script did, or you will likely OOM.
*   **Fix BUG-LOCAL-046 minimally:** In that same file, change the silent `except` block to a `logger.error("LTX load failed, falling back to stubs: %s", exc)`. Smallest change, massive visibility payoff.
*   **Fire the workflow:** Once patched, run it. If it OOMs, you know the raw Diffusers pipeline is too fat for 16 GB without ComfyUI's native memory management, and you'll need to rely on the stubs for now.

**5. Uncertainties I would want to verify**
*   Did `scripts/verify_ltx_hybrid.py` actually execute a *video generation* pass successfully, or did it just *load the weights* into VRAM without crashing?
*   Is `models/text_encoders/t5xxl_fp16.safetensors` actually a 9.4 GB FP16 file, or is it an FP8 quantized version misnamed? (If it's full FP16, your VRAM math is terrifyingly tight).
*   Graph execution order: Does ComfyUI execute the audio generation nodes *before* `OTR_VisualRenderer`? If audio runs and saves first, a video OOM is harmless to rule C7. If they run in parallel or video runs first, an OOM takes down the whole ship.

---

## To decide (Claude / human)

- [ ] Agree:
- [ ] Disagree:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
