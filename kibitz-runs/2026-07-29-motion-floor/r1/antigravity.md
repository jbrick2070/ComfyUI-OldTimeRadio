VERDICT: build-ready as-is? no. The plan attempts to satisfy an operator directive for "real moving video" by substituting short video loops and heavy over-rendering, introducing new architectural inconsistencies, VRAM bloat, and fragile video-demux dependencies into the pipeline.

MUST-FIX BEFORE BUILD:
1. [WHERE IT BITES #1 -- the credits backdrop (this is the live regression)] Proposing to slice and loop the last N seconds of the body video under the credits console directly violates the core directive ("every beat covered by real moving video") by introducing repetitive 2-4s looping artifacts over a 52s credits roll. Additionally, it assumes the body video duration is strictly greater than N seconds [ASSUMPTION]; if an episode or test clip is under N seconds, tail extraction will fail.
   Fix: Retain the static final frame hold (`extract_final_frame` in [nodes/otr_credits_roll.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_credits_roll.py#L30)) as an explicit, bounded UI presentation exception, or generate a single dedicated moving backdrop clip via `nodes/otr_video_render_batch.py` when motion is required.

2. [WHERE IT BITES #2 -- the still floor on a beat] Applying "render min_frames and trim" (`allow_tail_trim`) to cover beats shorter than an engine's minimum render length assumes that over-rendering large discrete frame menus (e.g., Veo/Pixverse 100-frame / 4s minimums for a 0.5s beat) is computationally affordable [ASSUMPTION]. Discarding 75%+ of generated frames wastes GPU time and VRAM on the 16 GiB review budget (`.kibitz/comfyui.local.md`).
   Fix: Enforce a beat floor rule in `coverage_plan.partition_beat` during timeline staging to adjust short beat durations to match engine `min_frames`, or restrict `allow_tail_trim` to low-overhead engines (e.g. `ltx_8gb` with 9 frames or `wan_i2v` with 33 frames).

3. [WHERE IT BITES #2 -- the still floor on a beat] Ambiguity surrounding ping-pong extension in `eng_wan_ti2v` (`PBUG-20260723-02`). Ping-ponging short renders back and forth creates severe visual oscillation and violates the operator requirement for forward temporal motion.
   Fix: Explicitly forbid ping-pong temporal fill in `coverage_plan` and `nodes/_otr_video_engines/render_driver.py`. Force `coverage_plan` to partition beats into valid sequential multi-clip chains or adjust engine frame/FPS parameters to cover beat durations natively.

SHOULD-FIX:
1. [WHERE IT BITES #1 -- the credits backdrop (this is the live regression)] Demuxing and slicing MP4 files within `OTR_CreditsRoll` introduces FFMPEG processing into a terminal presentation node, risking unhandled video decoding crashes at step 21 of 22.
   Fix: Maintain strict separation between presentation rendering and video processing in `nodes/otr_credits_roll.py`: wrap all video/frame extractions in a non-raising try/except that degrades gracefully to a solid darkened canvas rather than throwing `CreditsDataError`.

2. [WHERE IT BITES #2 -- the still floor on a beat] Lack of explicit classification for audio-driven speech avatar stills ("mouth stills") in `nodes/announcer_voice.py` and `nodes/batch_character_voices.py`.
   Fix: Formally document in `coverage_plan` that static init images for audio-conditioned video engines (`audio_conditioned_video` in [nodes/otr_credits_roll.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_credits_roll.py#L182)) serve as seed states for video generation and are not unconditioned static stills.

OPTIONAL / NICE-TO-HAVE:
1. Add an explicit `credits_backdrop_mode` widget parameter (options: `held_frame`, `solid_dark`, `generated_loop`) in [nodes/otr_credits_roll.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_credits_roll.py#L5) to allow operator toggling without code changes.

CUT THESE (scope / over-engineering):
1. [WHERE IT BITES #1 -- the credits backdrop (this is the live regression)] Complex tail-looping schemes (cross-fading last N seconds to slow drift, ping-pong reverse-and-forward cuts). Safe to cut because adding multi-pass FFMPEG slicing to a static UI console background over-engineers a terminal display element for zero narrative gain.
2. [WHERE IT BITES #2 -- the still floor on a beat] Multi-slice cross-fading and multi-still partitioning for beats exceeding engine ceilings. Safe to cut because standard multi-clip sequential chaining in `coverage_plan` already handles long beats without custom cross-fade machinery.
