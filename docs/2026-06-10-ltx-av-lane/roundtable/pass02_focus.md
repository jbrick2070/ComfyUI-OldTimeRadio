# PASS 02 REVIEW FOCUS: INPUTS / OUTPUTS

You are one panelist in an adversarial review of the plan below. THIS pass
is the INPUTS/OUTPUTS pass. The architecture (two thin adapters ltx_av_talk
/ ltx_av_music over a shared core, families, fallback chains) is LOCKED by
pass01 -- do not relitigate it; flag only fatal contradictions in one line.

Pressure-test exactly these, against the grounding files:

1. REQUEST SIDE: how does a VideoRequest carry audio_ref and init_image
   (asset_refs?), and what EXACT extraction pattern should the shared core
   use (eng_humo's _ref_path / _init_image_ref precedent)? What does the
   plan need to say so the coder window cannot guess wrong? Cite
   schemas.py field names.
2. AUDIO SLICE CONTRACT: the per-beat audio_ref is sliced from the FROZEN
   master upstream. What format/assumptions can the adapter rely on (wav?
   sample rate? mono/stereo? duration == beat duration?) -- find what
   render_driver.py / the request assembly actually guarantees, and what
   must be VERIFY-AT-BUILD.
3. FRAMES DERIVATION: target_fps 25, frames must be 8n+1, and the clip must
   COVER the audio slice (mux-LAST drops the master mix on top; a video
   shorter than the slice truncates visible action). Propose the exact
   formula (e.g. frames = next_8n1(ceil(duration_s * fps)) with a cap) and
   the clamp policy for >20s beats (b000 music opens) -- pad? chunk? cap
   with kenburns tail? Judge what the LTX-2.3 20s ceiling actually implies
   per the plan's claims ledger.
4. OUTPUT SIDE: what must canonicalize produce so the compositor/mux path
   treats the clip exactly like humo/ltx_video output today (bt709 /
   yuv420p, fps, has_audio=False, ffprobe asserts -- cite motion_common /
   schemas contracts). Where exactly is the audio track dropped if the node
   API forces a joint decode? Name the precise check that proves "no LTX
   audio leaked" (e.g. ffprobe stream count == video-only) and where it
   runs (canonicalize? test? both?).
5. CANVAS: OTR_VIDEO_LANDSCAPE_CANVAS (default 1472x832) plumbing -- how
   does the adapter read it, what happens on a non-/32 custom value (the
   av_dims validator must catch operator overrides), and does ltx_av_talk
   use the FULL landscape frame for a talking head (composition
   implications are pass03's problem; here: the dims/canvas mechanics
   only).
6. INIT IMAGE: the FLUX portrait arrives at what size/aspect? What
   scaling/letterboxing does the I2V conditioning need before LTX accepts
   it, and is that the adapter's job or the upstream image chain's?

Rules: cite grounding or mark VERIFY-AT-BUILD; the V-1 invariant
(has_audio=False, mux-only audio, byte-identical test) is non-negotiable;
no edits to eng_ltx_video.py / eng_humo.py / motion_common.py behavior.
Output: numbered MUST-FIX (file + what), SHOULD-CONSIDER, OPEN-QUESTIONS.
Terse and specific.
