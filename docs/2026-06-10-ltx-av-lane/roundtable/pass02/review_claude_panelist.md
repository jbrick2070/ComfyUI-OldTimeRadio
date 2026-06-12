# pass02 inputs/outputs -- Claude panelist review (written before reading the panel)

MUST-FIX

1. (plan, request side) The plan must state the EXACT field shapes so the
   coder cannot guess: `audio_ref` is a TYPED top-level VideoRequest field
   (Optional[AudioRef]; the driver builds {"path": wav}) -- NOT an
   asset_refs entry; `init_image` IS an asset_refs entry
   (schemas.py VideoRequest: asset_refs dict; render_driver.py:194-195
   builds {"init_image": portrait} / audio_ref={"path": audio}). The shared
   core must reuse eng_humo's tolerant extraction (_ref_path handles
   dict-or-str, eng_humo.py:366-383) verbatim.
2. (plan, frames) `timing.target_frame_count` is the INTEGER timing
   authority (schemas.py VideoRequest docstring) -- the adapter must NOT
   derive frames from audio duration. Formula to encode: render_frames =
   smallest (8n+1) >= target_frame_count (av_dims helper provides
   next_8n1()); canonicalize then TRIMS to exactly target_frame_count.
   This keeps the compositor contract (frame counts, not float seconds)
   and the 8n+1 model rule simultaneously. The audio slice ALREADY matches
   the beat window because the driver slices [start_s, start_s+dur_s] from
   the master (render_driver.py:650) -- the video just has to cover
   target_frame_count.
3. (plan, >20s clamp Q9) Per-REQUEST length gate, not an engine-level
   block: in ltx_av_music.render_clip, if target_frame_count >
   LTX_AV_MAX_FRAMES (20s * 25fps -> the largest 8n+1 <= 500 = 497, i.e.
   ~19.9s) raise a named render error -> the standard LOUD fallback
   degrades that beat to ltx_video (role-valid, no length ceiling, aspect
   -stable). b000 music opens longer than ~20s therefore render exactly as
   they do today. Same gate on ltx_av_talk (talking beats are short; the
   gate is cheap insurance).
4. (plan, output side) Canonicalize reuses the PROVEN silent path: decode
   VIDEO frames from the LTX graph, hand them to wrapper_bridge's silent
   bt709/yuv420p encode (eng_ltx_video docstring: "the silent bt709
   encode ... proven in wrapper_bridge"); CanonicalClip is ALWAYS-SILENT
   by type (schemas.py). LTX's audio side is therefore discarded BEFORE
   any container exists -- "no audio leak" is then proven twice: (a)
   canonicalize ffprobe-asserts exactly one video stream / zero audio
   streams on the emitted clip; (b) a unit test renders the contract on a
   stub. Do NOT mux-strip after the fact; never let an audio-bearing
   container exist on disk.
5. (plan, init_image fit) The FLUX portrait is portrait-aspect; naive
   uniform-FIT into 1472x832 would CONDITION the generation on side bars
   (the model will happily keep them). v1 policy: uniform-scale COVER +
   center crop to canvas inside the shared core (the upstream person
   guard already ensures a face); record the crop in the ledger row.
   v2 (note only): ImageDirector could render landscape in-character
   portraits when the role's engine is ltx_av_talk (it already reads
   video policy for 3D roles -- otr_image_director.py).

SHOULD-CONSIDER

6. Audio slice format: the driver guarantees a WAV path sliced from the
   master; sample rate / channels are VERIFY-AT-BUILD -- the adapter must
   pass the file to the LTX audio-conditioning node unmodified and let it
   resample; if the node demands a specific rate, resample via the
   existing ffmpeg path in a temp file (no new deps), never numpy/librosa.
7. Canvas plumbing: read OTR_VIDEO_LANDSCAPE_CANVAS exactly like
   eng_ltx_video/eng_wan do today (a5f4763 pattern); av_dims raises on
   non-/32 operator overrides BEFORE any model load (prepare), naming the
   nearest valid pair.
8. base_clip_ref is irrelevant to both adapters -- explicitly document
   ignoring it (prevents a future "use the procgen base" scope creep).

OPEN-QUESTIONS

9. Does the LTX-2.3 audio-conditioning node accept an arbitrary-length WAV
   and trim internally, or must the input EXACTLY match render duration?
   (M0 P0 records both behaviors; affects formula 2's padding edge.)
10. Does the IA2V graph expose separate VIDEO-only decode, or always a
    joint AV decode? (Decides where canonicalize taps frames; M0.)
11. Music beats: target_frame_count for b000 opening beats comes from the
    gap slicing (cdbba34) -- confirm those rows carry sane integer frame
    counts for the >20s gate (pass04 grounding).
