# pass02 (inputs/outputs) judgment -- Claude, judge + panelist

Panel: GPT-5.5, Gemini 3.1 Pro, DeepSeek v4 + Claude panelist. Grounded vs
HEAD 56caa5b.

## ACCEPTED (grounded)

- EXTRACTION CONTRACT (all 4): audio = humo-style tolerant `_ref_path` on the
  TOP-LEVEL `request.audio_ref` (AudioRef {"path":...} | str | obj;
  eng_humo.py:366-383); init_image = `asset_refs.get("init_image","")`;
  `conditioning_refs` NEVER satisfies required inputs
  (schemas._present_input_tokens, GPT SC3 test adopted); base_clip_ref
  explicitly ignored by both adapters.
- FRAMES (Gemini MF2 CONFIRMED at eng_ltx_video.py:281 -- the legacy formula
  snaps DOWN; do not copy): T = timing.target_frame_count is the AUTHORITY;
  render_frames = min(next_8n1(T), LTX_AV_MAX_FRAMES) with
  next_8n1(n) = ((n + 6) // 8) * 8 + 1; canonicalize TRIMS to exactly T when
  render > T and PADS BY REPEATING THE LAST FRAME when render < T (the cap
  case), LOUD log when padding exceeds 2s. Gemini's "the compositor holds
  the last frame" is UNVERIFIED in the repo -- the adapter therefore owns
  padding; no compositor assumption. GPT's "never truncate below audio"
  satisfied by construction.
- >20s POLICY (Q9 CLOSED): LTX_AV_MAX_FRAMES is an M0-MEASURED constant
  (initial conservative 497 = largest 8n+1 <= 20s*25fps; Gemini's 257 and
  the exact ceiling are VERIFY-AT-BUILD; GPT's 497/505 question goes to the
  M0 sheet). Beats longer than the cap render to the cap and pad-freeze the
  tail; b000 head-gap opens are >=2s and typically short, so this path is
  rare and LOUD.
- AUDIO NORMALIZATION (GPT MF3 + SC6): the sliced-master path guarantees
  WAV PCM s16le / 44100 / mono (_slice_master_audio ffmpeg args), but other
  per-line sources (audio_wav_path / clip_path / video_clip_path variants)
  are NOT normalized -- the shared core ALWAYS normalizes the incoming
  audio_ref to s16le/44.1k/mono WAV in the episode temp dir via the existing
  ffmpeg path before staging; the LTX node's accepted format is
  VERIFY-AT-BUILD (resample again only if the node demands).
- OUTPUT CONTRACT (Gemini SF2 + GPT MF6 + DeepSeek MF4): the graph
  TERMINATES at the video VAEDecode -> IMAGE batch ->
  `wrapper_bridge.encode_frames_to_silent_mp4()` (CONFIRMED:
  wrapper_bridge.py:512; `-an` at :446 "V-1: only the mux adds audio").
  An audio-bearing container never exists on disk. canonicalize returns the
  CanonicalClip shape (has_audio=False, yuv420p, bt709, fps 25, integer
  frame_count) per the eng_humo `_clip_from_raw` precedent, ffprobe-asserts
  ZERO audio streams, and a unit test feeds a fake AV mp4 through the
  fallback strip path (`-map 0:v:0 -an`) to prove the guard.
- CANVAS (GPT MF7 CONFIRMED at render_driver.py:387): the landscape
  override tuple `("ltx_video","wan_i2v")` gains `"ltx_av_talk",
  "ltx_av_music"` -- render_driver.py JOINS THE TOUCH LIST (additive line).
  Adapter reads request.canvas.w/h; av_dims validates IN assert_usable via
  request_template (Gemini MF5 -- fail on CPU BEFORE the AS-3 lease) and
  again in prepare.
- INIT IMAGE (Gemini MF4 + DeepSeek MF6 + Claude MF5, merged): the adapter
  preprocesses IN-GRAPH with core nodes (ImageScale + crop) from the
  resolve_aspect_transform math; v1 policy = uniform-scale COVER +
  center-crop to canvas (no pad bars -- padding would CONDITION bars into
  the generation; the upstream person guard keeps faces centered). The
  pad-then-outpaint idea (Gemini) is recorded as an M0 experiment cell, not
  v1. VERIFY-AT-BUILD: whether the IA2V template ships its own resize
  convention; FLUX portrait true dimensions.
- role_compat MUSIC_VISUAL += audio_ref becomes UNCONDITIONAL in M1
  (DeepSeek MF7); the DRIVER attach for music beats remains pass04 work.
- ltx_av_talk fails closed pre-render when audio_ref or init_image missing
  (DeepSeek SF2; humo :320-323 precedent).

## REJECTED / MISREADS (discarded before they poison the plan)

- GPT MF2 "render_driver writes schema-invalid timing.dur_s": MISREAD.
  build_request writes `"timing": {"target_frame_count": int(frame_count)}`
  (render_driver.py:197); `dur_s` at :360 is read from the LEDGER line, not
  written into Timing. No schema violation exists; production renders parse.
- GPT MF8 strict env parse for OTR_VIDEO_LANDSCAPE_CANVAS: REJECTED as a
  mutation of existing driver behavior; av_dims on the REQUEST canvas
  already fails 1450x832 loudly. Malformed env falling back to 1472x832 is
  valid behavior.
- Gemini MF3's MAX_FRAMES=257: number unsupported; superseded by the
  M0-measured cap (above).
- Claude panelist's "length-gate raises -> fallback to ltx_video for long
  music beats": superseded by cap+pad (simpler, no fallback chain coupling
  for a duration edge).

## CARRIED

- GPT SC1 slice-cache key (add master mtime+size): pass07 candidate, low
  risk (master is frozen per episode).
- M0 sheet additions: LTX-AV node IO shapes (IMAGE batch vs file vs joint
  AV), exact temporal ceiling, accepted audio formats, IA2V resize
  convention, FLUX portrait dims.
