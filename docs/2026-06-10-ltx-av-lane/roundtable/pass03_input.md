# PASS 03 REVIEW FOCUS: PROMPTS

You are one panelist in an adversarial review of the plan below. THIS pass
is the PROMPTS pass. Architecture and I/O contracts are LOCKED (pass01/02)
-- do not relitigate; one-line flags only for fatal contradictions.

The repo just shipped a roundtable-hardened brief-to-downstream prompt
restoration (finish_visual_prompt + get_era_tail in
_otr_story_brief_helpers; brief-grounded LTX scene opens; in-character FLUX
portraits via character_description; person guard; no-text clause; 240-char
cap; OTR_LTX_RADIO_PROMPT verbatim lane; stage-direction and self-vocative
scrubs). The new lane must JOIN that pipeline, not fork it.

Pressure-test exactly these against the grounding:

1. PROMPT SOURCES per role: for ltx_av_talk (announcer/character) and
   ltx_av_music, where does text_prompt come from TODAY for the equivalent
   beats (brief-grounded scene opens for ltx_video; portrait/M4 appearance
   chains for character imagery)? Specify the exact helper calls /
   request fields the new adapters should rely on so prompts arrive
   ALREADY finished (finish_visual_prompt'd) -- the adapter should NOT
   compose prompts itself. True or false in the grounding?
2. TALKING-HEAD PROMPT CONTENT: the audio drives the lips; what should the
   text_prompt say (framing, period styling, "speaking" verbs?) and what
   must it NOT say (stage directions, caption text, character names that
   trigger self-vocative issues, anything the no-text clause exists to
   prevent)? Propose a 1-2 sentence TEMPLATE SHAPE (not literal prose) and
   the negative-prompt baseline, citing the existing defaults
   (_LTX_DEFAULT_NEGATIVE in eng_ltx_video.py).
3. MUSIC PROMPT CONTENT: for "visuals breathe with the track", does the
   prompt need motion/rhythm vocabulary, or is the audio conditioning
   expected to carry it (the model "hears")? Should the music prompt path
   reuse get_story_brief_ltx scene composition verbatim (240-char cap,
   no-text clause), and what (small) additive tail is justified?
4. PROMPT BUDGET: LTX-2.3 uses a large text encoder; is the 240-char brief
   cap appropriate for it, harmful, or irrelevant? Recommend whether the
   cap stays (consistency) or the lane gets its own cap, and why.
5. NEGATIVE PROMPT: eng_ltx_video ships a default negative; should the AV
   lane reuse it verbatim, extend it (e.g. "static, frozen, still image"
   pressure against audio-ignoring renders?), and does anything in it
   conflict with talking-head content ("text" exclusion vs captions are
   burned later -- fine)?
6. PERSON-GUARD INTERPLAY: portraits feeding ltx_av_talk pass the person
   guard upstream; does ANY prompt content here need to repeat that
   protection, or is repeating it scope creep?
7. OTR_LTX_RADIO_PROMPT: the verbatim operator override exists for the
   radio open. Should the AV lane honor the same env on announcer beats,
   a sibling env (OTR_LTX_AV_PROMPT?), or none? Pick one and defend.

Rules: cite grounding or VERIFY-AT-BUILD; LOCKED items stay locked; the
adapter must not import or duplicate brief logic (V-12-adjacent: prompt
composition stays upstream). Output: numbered MUST-FIX (file + what),
SHOULD-CONSIDER, OPEN-QUESTIONS. Terse.


# LTX-AV lane -- sprint plan after pass02 (architecture + I/O LOCKED)

> Campaign docs/2026-06-10-ltx-av-lane/. Pass01 locked architecture; pass02
> locked the I/O contracts (judgments in pass01_judgment.md /
> pass02_judgment.md). Remaining passes: prompts, wiring, testing, hardware,
> pre-mortem, finishing. NO production code from the planner window.

## Mission

Additive audio-conditioned LTX-2.3 lane: `ltx_av_talk` (announcer_visual,
character_video -- lip-sync from the FLUX still, I2V + per-beat audio) and
`ltx_av_music` (music_visual -- audio-reactive scene motion). `ltx_video`
(2B) and every other shipped engine stay untouched.

## ARCHITECTURE (LOCKED pass01)

ONE new file `nodes/_otr_video_engines/eng_ltx_av.py`: private shared core +
two thin MotionEngineBase adapters.

- ltx_av_talk: roles (announcer_visual, character_video); family
  audio_driven_face (REUSED; schemas requires (audio_ref, init_image));
  required_inputs (text_prompt, audio_ref, init_image); fallback "humo" ->
  humo -> humo_1.7B -> latentsync -> still_kenburns; degrade aspect change
  landscape -> pillarbox is a LOUD restamped policy.
- ltx_av_music: roles (music_visual,); family audio_conditioned_video (NEW);
  required_inputs (text_prompt, audio_ref); fallback "ltx_video" ->
  still_kenburns (role-valid, aspect-stable, zero ltx_video edits).
- Both: default_roles () dark; ONE flag OTR_ENABLE_LTX_AV;
  ISOLATION_IN_PROCESS; target_fps 25; engine_version "1"; BUG-070
  assert_sage_not_patched; AS-3 lease; BUG-291 reclaim_idle_models; V-12
  lazy imports; executor-thread forward.
- assert_usable order: flag -> Sage -> NODE-AVAILABILITY (all graph node
  classes resolve in NODE_CLASS_MAPPINGS; missing named) -> weights on disk
  -> av_dims on the request_template canvas (CPU-fails BEFORE the lease).
- Isolation STOP rule: in-process IFF pip freeze identical before/after M0
  AND nodes resolve; else ISOLATION_SIDECAR_REQUIRED + STOP + finding.

## I/O CONTRACTS (LOCKED pass02)

REQUEST EXTRACTION (the coder copies this, no guessing):
- audio: tolerant `_ref_path(request.audio_ref)` -- accepts AudioRef
  {"path":...} / bare str / .path obj (eng_humo.py:366-383 pattern).
  audio_ref is a TOP-LEVEL VideoRequest field, never asset_refs.
- init image: `asset_refs.get("init_image", "")`. conditioning_refs NEVER
  satisfies required inputs (schemas._present_input_tokens); base_clip_ref
  ignored by both adapters (documented).
- ltx_av_talk fails closed pre-render if either path is empty (humo
  :320-323 precedent, named error).

AUDIO NORMALIZATION: the sliced-master path guarantees WAV PCM s16le /
44100 / mono (_slice_master_audio); other per-line sources are NOT
normalized -> the shared core ALWAYS ffmpeg-normalizes the incoming
audio_ref to s16le/44.1k/mono WAV under the episode temp dir before
staging. LTX node's accepted format VERIFY-AT-BUILD (M0 sheet).

FRAMES (timing authority preserved):
- T = timing.target_frame_count (INTEGER authority; never derive from
  audio duration). next_8n1(n) = ((n + 6) // 8) * 8 + 1  [snap UP --
  the legacy eng_ltx_video :281 formula snaps DOWN; do not copy].
- render_frames = min(next_8n1(T), LTX_AV_MAX_FRAMES); LTX_AV_MAX_FRAMES
  is M0-measured (initial conservative 497 = largest 8n+1 <= 20s*25fps).
- canonicalize TRIMS to exactly T (render > T) or PADS BY REPEATING THE
  LAST FRAME to T (cap case), LOUD log when padding > 2s. No compositor
  hold is assumed (unverified in repo).

OUTPUT (V-1 by construction):
- Graph terminates at the VIDEO VAEDecode -> IMAGE batch ->
  wrapper_bridge.encode_frames_to_silent_mp4() (`-an`, :446/:512). An
  audio-bearing container NEVER exists on disk. If a node variant forces a
  joint AV file, the strip path is `-map 0:v:0 -an` + re-encode -- guarded
  by a unit test that feeds a fake AV mp4.
- canonicalize returns the CanonicalClip shape (has_audio=False, yuv420p,
  bt709 fields, fps 25, integer frame_count) per eng_humo._clip_from_raw,
  and ffprobe-asserts ZERO audio streams on the emitted clip.

CANVAS:
- request.canvas.w/h is what renders; render_driver.py:387's landscape
  tuple ("ltx_video","wan_i2v") gains "ltx_av_talk","ltx_av_music"
  (additive edit; default 1472x832 both /32-valid).
- av_dims.assert_ltx_dims(W,H,frames): W%32==0, H%32==0, frames%8==1;
  RAISES naming nearest valid values; called in assert_usable (template)
  and prepare. (Upstream silently rounds; "+1 on W/H" is a doc error --
  Lightricks/ComfyUI-LTXVideo #347.)

INIT IMAGE (talk):
- In-graph preprocessing with core nodes (ImageScale + crop) from
  resolve_aspect_transform math; v1 = uniform-scale COVER + center-crop to
  canvas (NO pad bars -- padding conditions bars into the generation).
  Pad+outpaint recorded as an M0 experiment only. VERIFY-AT-BUILD: IA2V
  template's own resize convention; true FLUX portrait dims.

## Additive touch list (updated pass02)

- NEW  nodes/_otr_video_engines/eng_ltx_av.py (core + 2 adapters)
- NEW  nodes/_otr_shared/av_dims.py (+ next_8n1 helper)
- EDIT nodes/_otr_video_engines/__init__.py (guarded import)
- EDIT nodes/_otr_video_engines/schemas.py (FAMILIES + FAMILY_REQUIRED_
       INPUTS: "audio_conditioned_video" -> ("text_prompt","audio_ref"))
- EDIT nodes/_otr_video_engines/registry.py (docstring family list only)
- EDIT nodes/_otr_shared/role_compat.py (MUSIC_VISUAL += "audio_ref";
       UNCONDITIONAL, lands with M1)
- EDIT nodes/_otr_video_engines/render_driver.py (:387 landscape tuple +=
       the two new names; music-beat audio attach lands per pass04)
- NEW  tests (pass05 enumerates)
- Docs/tracker at ship.

## Claims ledger (delta only; pass01 ledger carries)

CONFIRMED pass02: legacy frame snap-DOWN (eng_ltx_video.py:281); landscape
tuple gate (render_driver.py:387); slice format s16le/44.1k/mono;
encode_frames_to_silent_mp4 + -an (wrapper_bridge.py:446,512); Timing
fields = source_line_ids/target_duration_s/target_frame_count/start_s.
MISREAD discarded: "driver writes schema-invalid timing.dur_s" (it writes
target_frame_count; dur_s is a ledger-line read).
UNVERIFIED -> M0 sheet: LTX-AV node IO shapes (IMAGE batch vs file vs joint
AV); exact temporal ceiling (497/505/20.0s); accepted audio formats; IA2V
resize convention; FLUX portrait dims; compositor tail behavior.

## Milestones (M0 sheet grew)

- M0 PROBE: disk inventory; node presence in BOTH Desktop + headless
  builds; pip-freeze sandwich (STOP rule); scratch IA2V render with a real
  slice; hash output audio (probe); NVML peak + wall time per lane L1
  (fp8_scaled 23.5GB block-swap) / L2 (GGUF Q4_K_M) / L3 (NVFP4 stretch);
  RECORD: node IO shapes, temporal ceiling, accepted audio rates, resize
  convention, portrait dims, pad-vs-crop experiment cell; P1 eyeball
  matrix -> LIPSYNC | STYLIZED | INERT per role-shape; INERT everywhere =
  close the lane with a finding.
- M1 ADAPTERS (CPU): eng_ltx_av.py skeleton dark + av_dims + schemas/
  role_compat/__init__/registry/render_driver-tuple deltas + unit tests;
  suite + Bug Bible green; byte-identical untouched.
- M2 GRAPH + LANE: winning-lane graph; node pre-flight; lease; silent
  encode; trim/pad; chain registration + termination tests.
- M3 WIRING: Director pick-through proof; music-beat audio_ref attach
  (pass04); ledger engine-identity stamps; OTR_FORCE_ENGINE_MAP entries.
- M4 GATES: full suite + Bug Bible + byte-identical + live 30-word smoke
  per role forced; acceptance greps (identity lines, LOUD restamps incl.
  aspect-change + pad-tail reasons, NVML <= 14.5); obs playable AAC only.
- M5 LOOK-QA + DOCS.

## Appendix: cut lanes

Yvann-Nodes: CUT (pass01, 4/4); revisit only on INERT-for-music.

## Open questions (assigned)

- pass03 PROMPTS: how the three roles' text_prompts compose -- story-brief
  finish_visual_prompt / get_story_brief_ltx (240-char cap, no-text
  clause), era tails, OTR_LTX_RADIO_PROMPT verbatim lane, person guard
  interplay for portraits, character_description chain; what (if anything)
  the A2V prompt needs to say about SPEECH vs the audio doing the driving;
  music_visual prompt language for "breathes with the track".
- pass04 WIRING: music-beat audio_ref attach point in render_driver;
  ShotLock execution-group/provider effects; restamp wording incl. aspect
  change; FORCE map; dropdown policy tests.
- pass05 TESTING: full test list; Desktop-vs-headless gate mechanics; the
  fake-AV-mp4 strip test; cold-import; chain termination; engine-count
  updates.
- pass06 HARDWARE: gemma encoder artifact + size + placement; per-clip
  wall time vs the ~6 min LTX opens; L1/L2/L3 decision numbers; weight
  streaming fallback; co-residency with FLUX portraits.
- pass07 PRE-MORTEM: OOM mid-episode; fallback storms; partial downloads;
  Comfy restart staleness; zombie VRAM on cancel; slice-cache key (master
  mtime+size); caption/credits interplay.
- pass08 FINISHING: convergence check + coder-window tickets.
