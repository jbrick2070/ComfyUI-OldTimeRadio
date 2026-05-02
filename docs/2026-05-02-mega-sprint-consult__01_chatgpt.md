# Round A -- ChatGPT (gpt-5.5) elapsed=140.8s

## 1. VRAM: HuMo teardown → LTX bundled load

- **Risk: real, but mostly about lingering references / ComfyUI cache state, not the LTX 2B file size itself.**
- The LTX 2B bundled checkpoint at ~8.7 GB should not by itself be the danger after HuMo is truly unloaded. The dangerous shape is:
  - HuMo model still resident,
  - HuMo intermediate tensors retained by a Python reference,
  - ComfyUI model manager still holding something “loaded,”
  - CUDA context / compiled kernels / allocator fragmentation leaving less free VRAM than expected,
  - or the LTX loader causing a temporary fp32/fp16 duplication during load.

- The strict teardown in `batch_humo_render.py` is the right defense:
  - `unload_all_models`
  - `gc.collect`
  - `torch.cuda.empty_cache`
  - `torch.cuda.synchronize`

- **What I would watch before node 54 / LowVRAMCheckpointLoader starts:**
  - Add a log immediately after HuMo teardown and before LTX load:
    ```python
    free, total = torch.cuda.mem_get_info()
    print(f"[OTR_VRAM] after HuMo teardown: free={free/2**30:.2f} GiB total={total/2**30:.2f} GiB "
          f"allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB "
          f"reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB")
    ```
  - On a 16 GB laptop GPU under WDDM, I would want to see roughly **13.5–14.5 GiB free** after teardown. Less than ~12 GiB means something is still resident or the desktop/driver load is high.

- **Symptoms / grep targets:**
  - `CUDA out of memory`
  - `torch.OutOfMemoryError`
  - `Allocation on device`
  - `Tried to allocate`
  - `reserved by PyTorch but unallocated`
  - `Requested to load`
  - `loaded completely`
  - `unload_all_models`

- **Highest-probability VRAM failure mode introduced by this wiring:**
  - `LowVRAMCheckpointLoader` is sequenced correctly, but ComfyUI still has a cached model object or prior loaded model state from a previous graph execution.
  - This is especially worth checking on second and third queue runs without restarting ComfyUI.

- **Smallest high-payoff hardening:**
  - Add `[OTR_VRAM]` logs at:
    - end of `BatchHumoRender` teardown,
    - start of `LowVRAMCheckpointLoader`,
    - after LTX checkpoint load,
    - after `OTR_BatchLTXRender` teardown,
    - before `OTR_RTXUpscale`.
  - Do not do low-level VRAM optimization yet. First prove whether memory is actually being released.

- **File references from your description:**
  - `workflows/otr_scifi_16gb_full.json`: node 54, links 86–89.
  - `batch_humo_render.py`: the `finally` teardown block.
  - `low_vram_checkpoint_loader.py`: the `dependencies` input and loader call.
  - I cannot give exact line numbers without the branch/diff in front of me.

---

## 2. Audio path / C7 byte-identical preservation through RTX upscale

- **The video upscale path should not re-encode audio if the mux command really uses `-c:a copy`.**
- So, **audio waveform drift from decode/re-encode should not happen.**
- However, there are two different meanings of “byte-identical”:

  1. **AAC/audio packet payload byte-identical**  
     - High confidence if using stream copy.
     - This is the meaningful C7 interpretation for audio determinism.

  2. **Whole MP4 file byte-identical**  
     - Do not expect this after remuxing/upscaling.
     - MP4 container atom ordering, metadata, `moov` placement, timescale/edit-list behavior, and video stream bytes will differ.

- **Potential real failure modes:**
  - Audio gets truncated if the final mux uses `-shortest` and the upscaled silent video is even slightly shorter than the original composite.
  - Audio timestamps get re-anchored if the mux command changes timestamp policy.
  - Metadata or edit lists change, making container-level comparison fail.
  - If the source audio were ADTS AAC instead of MP4 AAC, an AAC bitstream filter could matter, but because your source is already an MP4 from `VideoComposite`, this is unlikely.

- **Mux command shape I would want:**
  ```bash
  ffmpeg -y \
    -i upscaled_silent.mp4 \
    -i original_832x480.mp4 \
    -map 0:v:0 \
    -map 1:a:0? \
    -c:v copy \
    -c:a copy \
    -map_metadata -1 \
    output_1080p.mp4
  ```

- **I would avoid `-shortest` unless you have proven video duration is always >= audio duration.**
  - If you need deterministic full audio preservation, let the audio stream run to its natural end.
  - A slightly longer/shorter video track is preferable to dropping final AAC packets.

- **Validation command for C7:**
  - Before upscale:
    ```bash
    ffmpeg -v error -i original_832x480.mp4 -map 0:a:0 -c copy -f md5 -
    ```
  - After upscale:
    ```bash
    ffmpeg -v error -i output_1080p.mp4 -map 0:a:0 -c copy -f md5 -
    ```
  - Those MD5s should match.

- **Packet timestamp validation if you care about timing, not just payload:**
  ```bash
  ffprobe -v error -select_streams a \
    -show_entries packet=pts_time,dts_time,duration_time,size,flags \
    -of compact original_832x480.mp4 > audio_packets_before.txt

  ffprobe -v error -select_streams a \
    -show_entries packet=pts_time,dts_time,duration_time,size,flags \
    -of compact output_1080p.mp4 > audio_packets_after.txt
  ```

- **Symptoms / grep targets in ffmpeg logs:**
  - `Non-monotonous DTS`
  - `Queue input is backward in time`
  - `Application provided invalid`
  - `Invalid DTS`
  - `Estimating duration from bitrate`
  - `Could not find tag for codec`
  - `aac`
  - `bitstream filter`

- **Bottom line:**  
  - Audio decode/re-encode drift: **no, should not happen.**
  - Container-level byte identity: **no, do not rely on it.**
  - Audio packet payload identity: **yes, should hold if `-c:a copy`, no `-shortest` truncation, and stable mapping are used.**

- **File references:**
  - `rtx_upscale.py` / `OTR_RTXUpscale`: final mux stage.
  - `workflows/otr_scifi_16gb_full.json`: node 56, link 93.

---

## 3. Ledger / clips_dir union and possible clobbering

- **Risk: real, cheap to eliminate.**
- The role filter is probably sufficient for normal clean ledger data:
  - `_NEVER_HUMO_ROLES = {announcer, music_open, music_close, music_inter, sfx}`
  - character roles go to HuMo,
  - non-character roles go to LTX.

- But I would still add a defensive “do not clobber existing clip” check in `OTR_BatchLTXRender`.

- Why:
  - duplicated `line_id`,
  - malformed ledger role,
  - stale partial rerun,
  - future role taxonomy drift,
  - typo like `announcer ` with whitespace,
  - prior failed run leaving a valid-looking file,
  - or a bug in one renderer stamping/writing the wrong `line_id`.

- **Recommended behavior in `batch_ltx_render.py`:**
  ```python
  if out_mp4.exists():
      print(f"[OTR_LTX] skip existing clip for line_id={line_id}: {out_mp4}")
      # stamp ledger source_kind as existing/ltx only if this line truly belongs to LTX
      continue
  ```

- Even better:
  - Validate unique `line_id`s before rendering.
  - Fail hard on duplicate line IDs:
    ```python
    raise ValueError(f"Duplicate ledger line_id: {line_id}")
    ```

- **Do not silently overwrite.**
  - In this architecture, clobbering a HuMo character clip with an LTX non-character clip would be catastrophic and hard to spot until final composite.
  - Skipping existing output is the smallest change with high payoff.

- **Symptoms / grep targets:**
  - Add explicit logs:
    - `[OTR_LTX] skip existing clip`
    - `[OTR_LTX] duplicate line_id`
    - `[OTR_LTX] refusing to overwrite`
  - Also grep for:
    - `source_kind`
    - `line_id`
    - `role=`

- **Bottom line:**  
  - Is `is_never_humo_role()` sufficient for the intended clean case? **Yes.**
  - Should LTX also skip/refuse existing files? **Yes. Do it.**

- **File references:**
  - `batch_ltx_render.py`: render loop and output path write.
  - shared role utility defining `_NEVER_HUMO_ROLES`.
  - `workflows/otr_scifi_16gb_full.json`: node 55, link 91/92.

---

## 4. DAG sequencing with wildcard `dependencies` input

- **For a normal ComfyUI linked input, yes, the edge should enforce execution order.**
- ComfyUI’s scheduler follows graph links, not merely semantic type names. A `STRING` output linked into a wildcard input should still create a real dependency edge.

- So this chain should order correctly:
  ```text
  BatchHumoRender.clips_dir
      -> LowVRAMCheckpointLoader.dependencies
      -> OTR_BatchLTXRender
      -> VideoComposite
      -> OTR_RTXUpscale
  ```

- **The bigger risk is not wildcard ordering.**
- The bigger risk is **caching / reuse across repeated runs**:
  - If node 54 has the same inputs as a previous run, ComfyUI may reuse the loader output or keep model patchers around.
  - The dependency link gives ordering for execution, but if the loader node is considered cached, you need to verify actual runtime behavior.
  - In practice, `BatchHumoRender` teardown may still unload models, but I would not assume this blindly.

- **What to verify in logs:**
  - You want to see:
    ```text
    [OTR_HUMO] finished/rendered clips
    [OTR_HUMO] teardown complete
    [OTR_VRAM] after HuMo teardown ...
    [OTR_LTX_LOADER] loading ltx-video-2b-v0.9.safetensors
    ```
  - In that order.

- **Symptoms / grep targets:**
  - `got an unexpected keyword argument 'dependencies'`
  - `missing required positional argument`
  - `Cannot connect`
  - `type mismatch`
  - `Execution cached`
  - `Using cached`
  - `Requested to load`
  - `loaded completely`
  - `unload_all_models`

- **Specific failure modes:**
  - `dependencies` declared in `INPUT_TYPES` but not accepted by the function signature:
    - symptom:
      ```text
      TypeError: ... got an unexpected keyword argument 'dependencies'
      ```
  - Wildcard type not accepted by your ComfyUI version:
    - usually caught at workflow load/connect time.
  - Loader node cached from previous run:
    - sequencing may look right in the graph but not produce a fresh “load after teardown” log.

- **Bottom line:**  
  - Wildcard edge itself: **high confidence okay.**
  - Repeated-run cache behavior: **watch carefully.**
  - Add explicit loader start log and VRAM log before the loader claims memory.

- **File references:**
  - `workflows/otr_scifi_16gb_full.json`: node 54, link 86.
  - `low_vram_checkpoint_loader.py`: `dependencies` input and function signature.

---

## 5. RTX VSR / `nvvfx` memory behavior over long episodes

- **Risk: plausible, but I would not preemptively recreate the VSR context per 64-frame chunk.**
- Creating one `nvvfx.VideoSuperRes` context per run is the right starting point.
- Recreating it every chunk may:
  - increase overhead,
  - increase allocator churn,
  - make fragmentation worse,
  - and hide rather than fix leaks.

- **What I would do instead:**
  - Keep one VSR context per full upscale run.
  - Log GPU memory after every chunk.
  - If memory climbs monotonically, then introduce periodic teardown, e.g. every 16 or 32 chunks, not every chunk.

- **For a 5 min / 7,500 frame / 117 chunk run, watch for:**
  - flat memory after warmup: good,
  - small one-time growth in first few chunks: normal,
  - monotonic growth per chunk: leak or retained frame buffers,
  - sudden failure near chunk N: context/resource leak or ffmpeg pipe backpressure.

- **Add log after each chunk:**
  ```python
  free, total = torch.cuda.mem_get_info()
  print(f"[OTR_RTX] chunk={chunk_idx}/{num_chunks} "
        f"free={free/2**30:.2f} GiB "
        f"allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB "
        f"reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB")
  ```

- **Also make sure stale LTX state is gone before RTX upscale.**
  - Since `OTR_RTXUpscale` runs after `VideoComposite`, it should not need any diffusion model resident.
  - If `OTR_BatchLTXRender` already performs strict teardown after its loop, good.
  - I would still log `[OTR_VRAM] before RTX upscale`.

- **Symptoms / grep targets:**
  - `NvVFX`
  - `nvvfx`
  - `VideoSuperRes`
  - `CUDA_ERROR_OUT_OF_MEMORY`
  - `OUT_OF_MEMORY`
  - `NVCV`
  - `Failed to allocate`
  - `BrokenPipeError`
  - `pipe`
  - `ffmpeg exited`
  - `Error while decoding stream`
  - `Error submitting video frame`

- **If memory climbs:**
  - First check that decoded input frame arrays and output frame arrays are not retained in Python lists across chunks.
  - Make sure each chunk’s frame buffers go out of scope.
  - Explicitly delete chunk arrays after encode write.
  - Then consider recreating VSR every N chunks.

- **Bottom line:**  
  - One context per run: **yes, good default.**
  - Recreate per chunk: **not my first move.**
  - Add per-chunk VRAM logging and only add periodic context recycle if the graph shows monotonic growth.

- **File references:**
  - `rtx_upscale.py` / `OTR_RTXUpscale`: chunk loop, VSR context lifetime, ffmpeg subprocess handling.
  - `workflows/otr_scifi_16gb_full.json`: node 56.

---

## 6. LTX prompt risk: “no people in frame” with CFG 1.0 distilled schedule

- **Risk: real.**
- Do not treat the negative prompt as a hard safety guarantee.
- With distilled LTX-style low-step schedules and CFG around `1.0`, negative prompt influence can be weak or effectively minimal depending on the implementation path.

- **Likely behavior:**
  - The radio bookend guide image helps a lot.
  - Strong “no people / no faces” negative text helps some.
  - But LTX can still hallucinate:
    - silhouettes,
    - portraits,
    - faces in radio dials/posters,
    - ghostly human shapes,
    - hands,
    - audience/studio figures,
    - announcer-like figures if the positive prompt says “announcer,” “broadcast studio,” “radio host,” etc.

- **Smallest prompt-side hardening:**
  - Avoid human-implying positive terms for non-character roles.
  - For `announcer`, do not prompt “announcer at microphone.”
  - Prefer:
    - `empty radio transmitter room`
    - `vintage microphone on desk`
    - `glowing vacuum tubes`
    - `radio station logo card`
    - `abstract broadcast waves`
    - `empty 1940s control room`
  - Keep “no people” in both positive and negative phrasing:
    - positive: `empty room, no people present`
    - negative: `person, people, face, portrait, human, man, woman, hands, body, silhouette`

- **CFG question:**
  - Raising CFG from `1.0` to something like `1.3–2.0` may improve prompt adherence.
  - But for distilled LTX schedules, higher CFG can also produce worse motion/texture or overcooked results.
  - It should not meaningfully change VRAM.
  - It will change video output, but deterministically if seed and settings are fixed.

- **If “no people” is a requirement rather than preference:**
  - Prompting is not sufficient.
  - Add a deterministic local rejection pass:
    - local face/person detector,
    - scan generated LTX clips,
    - fail the run or regenerate with alternate seed/prompt.
  - This remains 100% local and deterministic if the detector/model/version/threshold are pinned.
  - But it is more wiring, so I would not add it before the smoke unless faces are absolutely disallowed.

- **No useful log grep exists for this.**
  - This is visual/semantic failure.
  - Add explicit clip review artifacts or deterministic QC thumbnails if needed.

- **Bottom line:**  
  - Will the negative prompt always suppress people/faces at CFG 1.0? **No.**
  - Is this likely acceptable for radio bookend-style non-character clips? **Probably, if the positive prompts avoid human concepts.**
  - Should you raise CFG immediately? **Only after one smoke batch. Test 1.0 vs 1.5/2.0 on the same seeds.**

- **File references:**
  - `batch_ltx_render.py`: `_PROMPT_BY_ROLE`, negative prompt, LTX sampler settings.

---

## My pre-smoke verdict

- **Most likely to break first:** VRAM sequencing/caching visibility, not the JSON topology itself.
- **Most important cheap fix before smoke:** add “refuse/skip existing clip” in `OTR_BatchLTXRender`.
- **Most important cheap instrumentation:** `[OTR_VRAM]` logs around HuMo teardown, LTX load/render teardown, and RTX upscale.
- **Audio C7 is probably okay** if `-c:a copy` is used correctly and you validate audio stream MD5 before/after.
- **Wildcard dependency edge should work**, but verify actual log order and beware repeated-run caching.
- **VSR one-context-per-run is the right starting point.**
- **Prompt-only “no people” is not guaranteed.** Use non-human positive prompts; add deterministic local QC only if required.
