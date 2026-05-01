# Round C -- NVIDIA (mistralai/mistral-nemotron) elapsed=41.9s

# Analysis and Recommendations

## Q1: Sanity check BUG-121 (filesystem fallback for radio still resolver)

**Files/lines:**
- `nodes/batch_humo_render.py` (lines 1694-1756)
- `nodes/batch_flux_render.py` (lines 455-502)

**Analysis:**
Your layered resolver pattern is correct and follows good defense-in-depth principles. The three-tier fallback (ledger path → meta path → filesystem deterministic path) with existence checks at each layer is robust.

**Potential failure modes you're not catching:**
1. **Path traversal in `ledger.episode_id`:**
   - Add validation at the start of `_resolve_radio_still_path`:
     ```python
     if not isinstance(ledger.episode_id, str) or '/' in ledger.episode_id or '\\' in ledger.episode_id:
         logger.warning(f"Invalid episode_id: {ledger.episode_id}")
         return None
     ```
   - Line 1696 in `batch_humo_render.py`

2. **`otr_stills_dir()` resolution failure:**
   - Wrap the call in a try/except:
     ```python
     try:
         stills_dir = otr_stills_dir()
     except Exception as e:
         logger.warning(f"Failed to resolve stills dir: {e}")
         return None
     ```
   - Line 1702 in `batch_humo_render.py`

3. **Corrupt PNG handling:**
   - The current code will let `Image.open()` fail downstream. Consider adding a lightweight check:
     ```python
     if os.path.getsize(final_path) < 100:  # Minimum plausible PNG size
         logger.warning(f"Possible corrupt PNG at {final_path}")
         return None
     ```
   - Line 1748 in `batch_humo_render.py`

**Recommendation:**
Add these three checks. The changes are small (3-4 lines each) and provide significant robustness.

## Q2: Sanity check BUG-123 (end-of-run VRAM cleanup in VideoComposite)

**Files/lines:**
- `nodes/video_composite.py` (lines 120-145)

**Analysis:**
Your cleanup pattern is correct in principle but could be improved:

1. **`unload_all_models()` placement:**
   - Calling it from a consumer node is acceptable but not ideal. The better pattern would be:
     - Create a custom ComfyUI hook that runs after queue completion
     - Or use the `comfy.queue` module's `after_run` callback
   - However, since you're avoiding low-level changes, your current approach is fine for now.

2. **`cuda.synchronize()` necessity:**
   - It's redundant with `empty_cache()`. Remove it to save ~10ms per run.

3. **Additional caches to clear:**
   - ComfyUI's clip vision encoder cache is cleared by `unload_all_models()`
   - IP-Adapter image embeddings are also cleared
   - No need to clear anything else

**Recommendation:**
- Keep the current pattern but remove `cuda.synchronize()`
- Add a comment explaining this is a temporary workaround until proper queue hooks are implemented

## Q3: Root causes of open symptoms

**Symptom 1: No scene-boundary cuts in composited mp4**

**Files/lines:**
- `nodes/video_composite.py` (lines 250-300)
- `nodes/batch_humo_render.py` (lines 800-850)

**Analysis:**
The most likely cause is missing or incorrect `start_s`/`dur_s` values in `ledger.lines[]`. Even though BUG-106 added `dialogue_positions` tracking, there might be:

1. **Race condition in ledger updates:**
   - The audio processing might complete after the video composition starts
   - Add a debug log showing `ledger.lines[0].start_s` and `dur_s` values right before composition

2. **Mode-specific issue:**
   - The `humo_concat` mode might not respect scene boundaries
   - Check if `master_mix_per_clip_mux` mode produces correct cuts

**Recommendation:**
Add debug logging for `ledger.lines` values and test with `master_mix_per_clip_mux` mode.

**Symptom 2: Radio bookend FLUX render not happening**

**Files/lines:**
- `nodes/batch_flux_render.py` (lines 455-502)

**Analysis:**
Most likely cause is (c) - `episode_id` mismatch between render and HuMo phases. To confirm:

1. **Add debug logging:**
   - Log `episode_id` at FLUX render time
   - Log `episode_id` at HuMo render time
   - Compare them

2. **Check ledger stamping:**
   - Add a log showing the ledger state right after FLUX render

**Recommendation:**
Add these debug logs to confirm the hypothesis. If it's an `episode_id` mismatch, you'll need to investigate why the ledger is being modified mid-run.

## Q4: BUG-LOCAL-125 (scene_manifest_json)

**Files/lines:**
- `nodes/scene_sequencer.py` (lines 689-890)

**Analysis:**
The `scene_manifest_json` output is indeed unused and can be deprecated. The ledger already contains all this information.

**Recommendation:**
Mark it as deprecated in the next release and remove it in v2.0. This is a zero-cost change with no downside.

## Q5: Preventing BUG-124 false positives

**Analysis:**
Option (c) - "read the consumer code" - is the most effective for a one-developer project. It's lightweight and prevents exactly this class of error.

**Recommendation:**
Implement a mandatory "read the consumer code" rule before applying any graph wiring changes. This is the smallest change with the largest payoff.

## Summary of Recommendations

1. **BUG-121:**
   - Add path traversal validation
   - Add `otr_stills_dir()` resolution failure handling
   - Add corrupt PNG check

2. **BUG-123:**
   - Remove `cuda.synchronize()`
   - Add comment about temporary workaround

3. **Open symptoms:**
   - Add debug logging for ledger values
   - Test with `master_mix_per_clip_mux` mode
   - Add debug logging for `episode_id` values

4. **BUG-125:**
   - Deprecate `scene_manifest_json`

5. **Process:**
   - Mandatory "read the consumer code" rule

These changes are all small (1-5 lines each) but provide significant robustness improvements.
