# Synthesis -- 2026-05-02

**Question:** # OTR v2.0-alpha Sprint 3 mega-sprint -- pre-smoke wiring review

## Context

OTR is a ComfyUI plugin generating 1940s-style radio drama episodes. The Sprint 3 mega-sprint wires three new components into `workflows/otr_scifi_16gb_full.json` on branch `v2.0-alpha`:

1. **OTR_BatchLTXRender** -- in-graph LTX-2 renderer for non-character ledger lines (announcer / music_open / music_close / music_inter / sfx). Uses radio_bookend.png as both start and end keyframe via `LTXVAddGuide` for seamless looping. Writes silent libx264 yuv420p mp4 to `output/otr/videos/<ep>/<line_id>.mp4` (same dir HuMo writes character clips to). Stamps `ledger.clips[].source_kind="ltx"`.
2. **OTR_RTXUpscale** -- final-stage NVIDIA RTX VSR ULTRA upscaler. Path-in / path-out wrapper. Decodes video frames in chunks via ffmpeg pipe, runs `nvvfx.VideoSuperRes` per frame, writes silent libx264 yuv420p mp4, then muxes original mp4's audio with `-c:a copy` (zero audio re-encode -- C7 byte-identical preserved). Bypassable via `bypass=True` widget for raw 832x480 deliverables.
3. **LowVRAMCheckpointLoader** -- ComfyUI-LTXVideo's CheckpointLoaderSimple subclass. Adds a `dependencies` input that forces sequential loading (HuMo unloads before LTX claims VRAM).

## Locked Architecture Truth (settled 2026-05-02; do not relitigate)

- Resolution policy: native 832x480 end-to-end; LTX writes 832x480; HuMo pillarbox 832x480 letterbox.
- LTX seamless-loop: `LTXVAddGuide` with frame_idx=0 strength=0.75, frame_idx=-1 strength=0.6.
- Frame-count rule: LTX `8n+1`, capped at `LTX_MAX_FRAMES=177` to match HuMo's verified 16 GB ceiling.
- Tiled VAE decode: `tile_size=512, overlap=64, temporal_size=4096, temporal_overlap=8` (Goofer-proven on RTX 5080 Blackwell).
- Strict teardown after LTX loop: unload_all_models + gc + empty_cache + cuda.synchronize.
- `_NEVER_HUMO_ROLES = {announcer, music_open, music_close, music_inter, sfx}` (single source of truth).
- VRAM ceiling: 14.5 GB audio, 15.5 GB video.

## Wiring done in this commit (workflow JSON link topology)

- LowVRAMCheckpointLoader (node 54): widget `ckpt_name="ltx-video-2b-v0.9.safetensors"` (the bundled LTX 2B v0.9 file at `C:\ComfyUI-Models\checkpoints\ltx-video-2b-v0.9.safetensors`). `dependencies` input <- BatchHumoRender.clips_dir (link 86, STRING into wildcard input).
- OTR_BatchLTXRender (node 55): MODEL/CLIP/VAE inputs <- LowVRAMCheckpointLoader's three outputs (links 87/88/89). `ledger_json` <- OTR_SignalLostVideo.0 (link 90, same STRING source as HuMo gets; consumers duck-type via stem fallback). `humo_clips_dir` (new optional STRING) <- BatchHumoRender.clips_dir (link 91).
- VideoComposite (node 52) `clips_dir` rewired from BatchHumoRender (old link 81) to BatchLTXRender.clips_dir (new link 92). Both renderers write to the same `otr_videos_dir(episode_id)`, so VideoComposite picks up the union by line_id.
- OTR_RTXUpscale (node 56): `source_mp4_path` <- VideoComposite.final_mp4_path (link 93). Widgets: bypass=False, target_width=1920, target_height=1080, quality="ULTRA", chunk_frames=64, out_suffix="_1080p". Output: `<ep>_1080p.mp4`.

## Deviation from Architecture Truth (intentional, documented)

The Architecture Truth specified `UNETLoader + CLIPLoader (T5) + VAELoader` for LTX 2B fp16, NOT `CheckpointLoaderSimple`. Reason given: split-load lets ComfyUI offload T5/VAE independently; bundled-load on a hot HuMo cache is the OOM shape C2 was written to prevent.

REALITY: LTX 2B v0.9 ships ONLY as a bundled `ltx-video-2b-v0.9.safetensors` (8.7 GB, all components in one file). No separate LTX UNet / LTX VAE files exist on Lightricks' Hugging Face repo for the 2B v0.9 line; you'd have to manually split a `state_dict` to use UNETLoader/VAELoader.

RESOLUTION: Use ComfyUI-LTXVideo's `LowVRAMCheckpointLoader`. It IS a `CheckpointLoaderSimple` subclass, but it adds a `dependencies` input that ComfyUI uses to force the load to run AFTER the upstream node finishes (here: BatchHumoRender). Combined with the existing strict teardown in batch_humo_render.py (`unload_all_models + gc + empty_cache + cuda.synchronize` in finally), this satisfies the C2 sequencing intent: HuMo fully unloads before LTX loader fires. The "no carve-out for CheckpointLoaderSimple" rule was about preventing OOM via parallel-load on the hot HuMo cache; sequencing eliminates that risk.

## Specific risk question

**What can break this on a real 16 GB Blackwell run?**

Concrete worries to address (yes / no / what would you watch for):

1. **VRAM**: HuMo (16.5 GB staged then unloaded) -> LTX 2B (8.7 GB bundled load) sequence. With strict teardown does HuMo actually clear ~16 GB before LTX claims VRAM, or do we hit a 14.5 GB cap from a leaked tensor / cached compilation / lingering KV?
2. **Audio path / C7**: VideoComposite produces the 832x480 mp4 with `master_mix_per_clip_mux` (audio `-c:a copy` from procgen). RTX upscale chunks frames via ffmpeg `-an` decode pipe, encodes silent libx264 yuv420p, then muxes original mp4 audio with `-c:a copy`. Is there any path where the upscale stage's audio mux can drift from byte-identical (e.g. timestamp re-anchoring, faststart re-write, container repacking, missing AAC bitstream filter)?
3. **Ledger / clips_dir union**: HuMo writes `<line_id>.mp4`, LTX writes `<line_id>.mp4` to the same dir. If HuMo writes a character line and LTX (somehow) also writes the same line_id (shouldn't happen because the role filter excludes character), the second write clobbers the first. Is the `is_never_humo_role()` filter sufficient defense, or should LTX have a "skip if file already exists" check too?
4. **DAG sequencing edge cases**: ComfyUI execution order for the chain `EpisodeAssembler -> SignalLostVideo -> BatchHumoRender -> LowVRAMCheckpointLoader -> BatchLTXRender -> VideoComposite -> RTXUpscale`. With the wildcard `dependencies` input on LowVRAMCheckpointLoader receiving BatchHumoRender's STRING output, will ComfyUI honor the dependency edge, or is there a known bug where wildcard inputs don't enforce DAG ordering?
5. **RTX VSR + nvvfx**: My OTR_RTXUpscale chunks frames in 64-frame batches, runs `nvvfx.VideoSuperRes` per-frame inside the chunk, encodes via ffmpeg subprocess. Have you seen nvvfx leak GPU memory across long runs (5 min episode = 7,500 frames @ 25fps = ~117 chunks)? Should I tear down and re-create the `VideoSuperRes` context per chunk, or once per run as I'm doing?
6. **LTX prompt risk**: My `_PROMPT_BY_ROLE` includes "no people in frame" + a heavy negative prompt suppressing faces. Will LTX 2B v0.9's distilled sigma schedule (8 steps, sigmas `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]`) honor the negative prompt strongly enough, or do we need a higher CFG (currently 1.0 because distilled)?

**Format requested**: Bullet list per question. Be specific. If a failure mode is real, name the exact log line / symptom you'd grep for. If a question is "won't happen" with high confidence, say so plainly. Do not relitigate the locked architecture above; only flag wiring / runtime risks introduced by THIS commit.

---

## ChatGPT (gpt-5.5)

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

---

## Gemini (gemini-3.1-pro-preview-customtools)

### 1. Verdict
I **PARTIALLY AGREE** with the core recommendation; ChatGPT correctly identifies the need for VRAM logging and anti-clobbering checks, but it completely misses a mathematical certainty about CFG 1.0 and a critical ComfyUI caching trap that will break your multi-run sequencing.

### 2. Factual Errors in the ChatGPT Answer
*   **CFG 1.0 Math:** ChatGPT states that at CFG 1.0, the negative prompt influence "can be weak." This is mathematically false. Standard Classifier-Free Guidance is calculated as `output = uncond + CFG * (cond - uncond)`. If CFG is exactly `1.0`, the equation simplifies to `output = cond`. **Your negative prompt is 100% ignored at CFG 1.0.** It is not weak; it is erased.
*   **Audio `-shortest` risk:** ChatGPT warns that `-shortest` might truncate audio if the upscaled video is shorter. Your `OTR_RTXUpscale` extracts frames from `VideoComposite.final_mp4_path` and muxes the audio from that *exact same file*. The frame count is identical. There is zero risk of duration mismatch, rendering the `-shortest` warning irrelevant (though omitting the flag is fine).

### 3. Important Omissions
*   **ComfyUI Cache Desync (The real DAG risk):** In `workflows/otr_scifi_16gb_full.json` (link 86), you pass `BatchHumoRender.clips_dir` to `LowVRAMCheckpointLoader.dependencies`. If `clips_dir` is a static string per episode (e.g., `"output/otr/videos/ep01"`), ComfyUI will cache the loader node on the second queue run. If your `batch_humo_render.py` teardown manually unloads models, ComfyUI's internal state tracker will be desynced from reality. It will skip the LTX load step, assuming it's already in memory, resulting in an immediate crash.
*   **Windows Subprocess Deadlock:** In `rtx_upscale.py`, you are piping 1080p frames to ffmpeg via subprocess on Windows. Windows has a notoriously small OS pipe buffer (64KB). If ffmpeg writes to `stderr` (which it does verbosely by default) and your Python script doesn't actively consume it, the buffer fills, ffmpeg blocks, and your upscale hangs indefinitely.

### 4. My Short Recommendation
*   **Inject a Cache-Buster:** Modify `BatchHumoRender` to output a JSON string containing both the `clips_dir` and a unique run ID (e.g., `time.time()`). Pass *this* to `LowVRAMCheckpointLoader.dependencies` to guarantee ComfyUI evaluates the loader as a fresh execution every time.
*   **Fix the CFG / Negative Prompt:** If you want "no people" to actually influence the LTX generation, you must raise CFG to at least `1.5`. If you must stay at `1.0` for distillation reasons, delete the negative prompt entirely to save the VRAM and compute wasted on a T5 text-encode that gets mathematically discarded.
*   **Prevent Pipe Deadlock:** In `rtx_upscale.py`, ensure your ffmpeg subprocess call explicitly routes `stderr=subprocess.DEVNULL` (or actively reads it in a separate thread). 
*   **Implement Anti-Clobber:** Add the `if out_mp4.exists(): continue` check in `batch_ltx_render.py` as ChatGPT suggested. It's a 2-line insurance policy against catastrophic character-clip overwrites.
*   **C7 Audio Muxing:** Use `-c:a copy` without `-shortest`. To guarantee byte-identical AAC payloads, ensure you aren't accidentally stripping metadata tags that the original container had. Use `-map_metadata 0` from the source composite.

### 5. Uncertainties I Would Want to Verify
*   **ComfyUI Model Management Teardown:** I am uncertain *how* you are doing the strict teardown in `batch_humo_render.py`. If you are just calling `torch.cuda.empty_cache()` and `gc.collect()`, ComfyUI still thinks the model is loaded. You must call `comfy.model_management.unload_all_models()` and `comfy.model_management.soft_empty_cache()` to keep the graph scheduler's state machine accurate.
*   **Tiled VAE Temporal Size:** Your locked architecture states `temporal_size=4096`. LTX max frames is 177. A temporal window of 4096 for a 3D VAE is massively out of bounds and will likely cause an allocation error if actually passed to the VAE decode tensor shape. I am assuming this is a typo in your spec (perhaps you meant `4` or `8`), but I am flagging it as a potential immediate OOM trigger.

---

## NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5)

### Third Opinion: Tie-Breaking Review  

---

### **1. VRAM: HuMo teardown → LTX bundled load**  
**Agree with Gemini's correction**:  
- **ChatGPT** correctly identifies VRAM logging as critical but misses Gemini’s key point: **ComfyUI’s caching behavior with static `clips_dir` strings** will break sequencing on repeated runs.  
- **Gemini** is right: If `clips_dir` is a static path (e.g., `"output/otr/videos/ep01"`), ComfyUI will cache the `LowVRAMCheckpointLoader` node and skip reloading after the first run, causing OOM.  
- **My Recommendation**: Inject a cache-buster (e.g., `f"{clips_dir}_{time.time()}"`) into the `dependencies` input to force fresh loader execution.  

---

### **2. Audio path / C7 byte-identical preservation**  
**Agree with Gemini’s clarification**:  
- **ChatGPT** warns about `-shortest` truncation, but **Gemini** correctly notes frame counts are identical, so this risk is nonexistent.  
- **Critical Omission**: Both miss that **Windows FFmpeg builds may default to `faststart` MP4 atom ordering**, which alters container bytes even with `-c:a copy`. To preserve C7 byte-identical audio *and* container structure:  
  - Add `-movflags +faststart` to the final mux (if needed for streaming) but **validate with `mp4box -info`** to ensure atom order matches the original.  

---

### **3. Ledger / clips_dir union and possible clobbering**  
**Agree with both**:  
- **ChatGPT** and **Gemini** both correctly advocate for `if out_mp4.exists(): continue` in `batch_ltx_render.py`.  
- **Addition**: Pair this with a **pre-render ledger validation** to ensure no duplicate `line_id`s exist across HuMo and LTX roles. Fail fast on duplicates.  

---

### **4. DAG sequencing with wildcard `dependencies` input**  
**Side with Gemini**:  
- **ChatGPT** assumes wildcard edges enforce ordering but acknowledges caching risks.  
- **Gemini** identifies the **specific ComfyUI bug**: Static string dependencies (like `clips_dir`) are not treated as runtime-dependent edges.  
- **Fix**: Use a **non-static dependency signal** (e.g., a checksum of the `clips_dir` contents or a runtime-generated token).  

---

### **5. RTX VSR / `nvvfx` memory behavior over long episodes**  
**Agree with ChatGPT’s approach, but flag Gemini’s typo concern**:  
- **ChatGPT** recommends one VSR context per run + per-chunk VRAM logging. This is sound.  
- **Gemini** flags `temporal_size=4096` as a likely typo. **Verify this**: If the VAE decode expects a smaller temporal window (e.g., `4` or `8`), `4096` will cause **immediate OOM** on RTX 5080.  
- **Action**: Cross-check `temporal_size` with LTX 2B v0.9’s documented limits.  

---

### **6. LTX prompt risk: “no people in frame” with CFG 1.0**  
**Side with Gemini**:  
- **ChatGPT** claims CFG 1.0 makes negative prompts “weak,” but **Gemini** is mathematically correct: **CFG 1.0 nullifies the negative prompt entirely**.  
- **Fix**: Either:  
  - Raise CFG to `1.5–2.0` for negative prompt influence, or  
  - Remove the negative prompt to save VRAM/T5 encoding overhead.  

---

### **Factual Errors / Hallucinations**  
1. **ChatGPT**:  
   - Incorrectly assumes `batch_humo_render.py` uses `comfy.model_management` APIs for teardown. If it only calls `torch`/`gc`, ComfyUI’s internal state tracker may not reflect unloaded models.  
   - Hallucinates `LTX_MAX_FRAMES=177` as a HuMo ceiling match; this is an LTX-specific cap.  

2. **Gemini**:  
   - Correctly identifies `temporal_size=4096` as a likely typo but cannot confirm without code.  

---

### **Important Omissions by Both**  
1. **Windows-Specific FFmpeg Pipe Limits**:  
   - Both mention `stderr` deadlock risk but omit that **Windows subprocesses default to 64KB pipe buffers**. If `ffmpeg` writes >64KB to `stdout`/`stderr` without being read, the process hangs.  
   - **Fix**: Use `stderr=subprocess.DEVNULL` or a thread to drain pipes.  

2. **Bundled Checkpoint Loading Overhead**:  
   - LTX 2B v0.9’s 8.7 GB bundled load may cause **temporary VRAM spikes** during deserialization, even if the final resident size is 8.7 GB. Monitor loading phase with `nvidia-smi`.  

3. **VRAM Fragmentation**:  
   - Both assume sequential HuMo → LTX loads avoid fragmentation, but **Windows WDDM driver** may fragment memory across processes. Test with `nvidia-smi --query-gpu=memory.used --format=csv` during stress runs.  

---

### **Prioritized Recommendations**  
1. **Cache-Bust ComfyUI Dependency**:  
   - Modify `BatchHumoRender` to output `clips_dir + "_" + str(time.time())` to force fresh LTX loader execution.  

2. **Fix CFG or Remove Negative Prompt**:  
   - Either set `CFG=1.5` in `batch_ltx_render.py` or delete the negative prompt to avoid wasted compute.  

3. **Add File Existence Check in LTX Render**:  
   - In `batch_ltx_render.py`, skip rendering if `<line_id>.mp4` exists.  

4. **Validate `temporal_size` Parameter**:  
   - Confirm `temporal_size=4096` is intentional; if not, reduce to avoid OOM.  

5. **Drain FFmpeg Subprocess Pipes**:  
   - In `rtx_upscale.py`, use `stderr=subprocess.DEVNULL` or a thread to read `stderr` to prevent hangs.  

6. **Log VRAM at Critical Points**:  
   - Add `[OTR_VRAM]` logs before/after HuMo teardown, LTX load/render, and RTX upscale.  

---

### **Uncertainties Requiring Verification**  
1. **ComfyUI Model Unloading Semantics**:  
   - Does `batch_humo_render.py` use `comfy.model_management.unload_all_models()` or just `torch` APIs? The former ensures ComfyUI’s state tracker is aware.  

2. **Temporal Size Impact**:  
   - Is `temporal_size=4096` a documentation error? Confirm against LTX 2B v0.9’s actual requirements.  

3. **WDDM Memory Fragmentation**:  
   - Test repeated HuMo → LTX runs to see if available VRAM decreases over time due to fragmentation.  

--- 

**Final Verdict**: Gemini’s corrections on CFG math and ComfyUI caching are critical. Prioritize cache-busting, CFG adjustment, and file existence checks. VRAM logging and temporal size validation are mandatory pre-smoke steps.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
