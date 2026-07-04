# ideo_word razzle-vid -- CLOUD video animation of the word card

**STATUS: CODE-READY for PHASE 0 (pin audit -- runnable any time). PHASE 1 (build) is
conditional on the audit verdict + S1.** Supersedes the Blender 3D razzle doc (operator-
rejected). CLOUD ONLY (operator ruling): ideo is cloud, so the razzle is cloud razzle.

Candidate plan -- ideas window 2026-07-02. Hardened r1 by roundtable pass01 (Grok/DeepSeek/GLM,
~$0.0243) + kibitz r1 (claude + codex). Artifacts: docs/2026-07-02-ideo-word-razzle-vid/
roundtable/, kibitz-runs/2026-07-02-razzle-vid/.

## ACCEPTANCE BAR (operator, non-negotiable)

**The words stay READABLE for the WHOLE render -- every frame.** Any i2v pass that melts
letterforms mid-clip FAILS regardless of beauty. Spikes are judged frame-by-frame (manual
review v1; OCR automation later, never a v1 dependency). Spike pass checklist (per clip):
every-frame legibility AND at least two of: particle motion (smoke/rain), light variation
(flicker/breathing), parallax/depth movement. Fail path: retry once with modified prompt/seed,
then FAIL LOUD -- no silent fallback (standing directive).

## Problem / goal

Feed an ideo_word card into a CLOUD image-to-video pass so the card becomes a living period
world -- smoke through the letters, neon flicker, rain on the marquee -- words at the core,
readable throughout.

**Scope is ROLE-AGNOSTIC by design (operator 2026-07-02):** the engine serves ANY video role's
worded card -- character, announcer, or music beats -- because every beat carries its own audio
(spoken line or music bed), which satisfies kling_avatar's required-audio input naturally.
PROOF ORDER still starts at the music_open/music_close bookends (2 clips, bounded spend,
title cards); character/announcer beats turn on after the spike passes -- gated by cost
(~$0.25-1.00/clip x spoken beats is the dominant-spend class; per-episode cap knob mandatory)
and by a speech-audio-specific check: avatar conditioned on SPEECH may push lip-sync-shaped
motion harder than on music -- the no-face/no-mouth prompt directive and motion-class rejection
apply per role. Unique clip per beat always (worded = never pooled).

## Grounded cloud-i2v reality (r1-corrected)

- `cloud_kling_avatar` HAS an optional prompt STRING (partner_nodes.yaml:172) which the engine
  forwards (eng_cloud_video.py:229), takes an IMAGE init, and requires AUDIO -- which music
  bookends HAVE. It IS a legitimate spike candidate, with a named rejection criterion: if its
  motion class is face/lip-sync-shaped (talking-poster artifacts) rather than world animation,
  it fails the checklist and is out. (Earlier "avatar = wrong, period" was too strong.)
- `cloud_wan_i2v` is NOT invocable today without hacks: no prompt forwarded, requires
  OTR_CLOUD_WAN_MODEL env (raises without it), model is DYNAMICCOMBO_V3 -- gated on the S1
  V3-expansion, same as `cloud_seedance_2` (honest dark row). ALL V3 video rows are blocked on
  V3-expansion work, not on pinning.
- No other i2v rows pinned. BUT the live-core catalog is large (~91 video nodes per the
  cloud-engines pass00 survey: Kling i2v-class, Vidu, Luma Ray, Runway, PixVerse, Sora,
  Veo/Gemini-class, ByteDance) -- a promptable non-V3 i2v row plausibly exists. That is what
  Phase 0 answers.

## PHASE 0 -- pin audit (first deliverable, runnable now; r2-specified)

The audit is CODE, not a vibe: extend `scripts/otr_pin_partner_nodes.py` with an `--audit-i2v`
mode that WALKS the live core's api-node modules (the existing `_iter_api_node_modules` seam --
CURATED_ROWS alone cannot discover anything) and applies CODE-CHECKABLE filters per class:
RETURN_TYPES contains VIDEO; INPUT_TYPES has an IMAGE(-ref) input AND a STRING prompt input;
NO required AUDIO input; static (non-DYNAMICCOMBO_V3) media/prompt inputs; seed input present
(flag if absent); duration/fps params recorded. Output = a machine-readable JSON report
(row, pass/fail per filter, reasons) + the BLOCKED verdict emitted explicitly when zero rows
pass. Pricing stamps come from the PRICING.md process (credits->USD), NOT the yaml (it has no
pricing). EXPECTED OUTCOME IS HONESTLY UNCERTAIN: the pinned set has zero passing rows; the
bet is on the unpinned catalog (~91 video nodes) -- "only V3" or "none" are likely results and
the decision tree treats them as first-class outcomes, not failures of the plan.

Per surviving row: manual CURATED_ROWS entry + pin + pricing stamp + prompt profile entry
(PROMPT_PROFILES.md -- world-motion phrase template + era tail, defined there) + STATIC schema
conformance test (pinned kwargs vs yaml) + ONE gated live smoke leg (S0 smoke-harness pattern,
env-gated -- runtime conformance without unmetered spend) + dark/fail-closed registration,
EMPTY defaults, new Engine subclass + register call in the SAME change (rows don't self-wire).

**Decision tree (the audit's output):**
- Non-V3 promptable i2v row found -> Phase 1A (worded-card spike).
- Only V3 rows -> tier WAITS on S1 V3-expansion; revisit then.
- Nothing usable -> worded razzle is BLOCKED, stated loudly. A wordless-plate ambience spike
  (wan post-V3-expansion) is NOT a fallback for the word goal -- it is only ever the cloud half
  of the animate-then-overlay hybrid, which lives in the ideo_word_vid doc behind its own
  local-overlay ruling. This plan stays cloud-pure.

## PHASE 1 -- build (conditional; after S1 + ideo_word land)

- **1A worded-card spike:** one bookend card through the pinned row; prompt = world-motion
  phrases + era tail + text-preservation directive; motion-strength param ONLY where the row
  exposes one. Judged by the acceptance checklist frame-by-frame. Two candidate rows max (best
  filter scores). `kling_avatar` = LAST-RESORT experiment only (operator-requested): it exposes
  only mode/seed/prompt (NO motion-strength), it is a facial-animation model so face-shaped
  motion on typography is the EXPECTED failure mode (r2 deepseek), and the no-face prompt is a
  hope, not a control -- run it as one cheap clip with eyes open, never as the plan's backbone.
- Prompt-shaped mint: CUT from this plan (r2 unanimous) -- animation-friendly card composition
  lands with the ideo_word / ideo_word_vid docs, not in this build's checklist.
- **Duration fit is REQUIRED, not polish -- and it is NEW code:** the timing authority is
  `timing.target_frame_count` (schemas.py:97-99). Cloud canonicalization must exact-fit to it
  (today eng_cloud_video.py:163-184 computes frame_count from PROVIDER duration and never
  trims/loops). Rule: LONGER -> trim from tail; SHORTER -> loop w/ short crossfade;
  pathological (>2x off) -> FAIL LOUD. Test: the final manifest delivers target frames. Mux-
  LAST frozen audio untouched. Bumps Phase 1A productizing above "small".
- **Adapter reality (r2-corrected -- NO new reactivity class):** `mute_only` already models a
  promptable no-audio i2v shape (CloudWanI2VEngine declares required_inputs=(init_image,
  text_prompt), eng_cloud_video.py:290) -- a new row is a second mute_only-pattern subclass
  whose `_partner_inputs` actually FORWARDS request.text_prompt (wan's drops it -- the wiring
  step is explicit, the base class won't do it). Verify-at-build: nothing dispatches on the
  reactivity string vs required_inputs.
- **OWNED HERE (was mis-cited to S3-full): the init_image request-shape fix.** Real requests
  carry `asset_refs["init_image"]` (render_driver.py:255-257, schemas.py:156-158) but
  `_CloudVideoBase._init_image_input()` reads top-level (eng_cloud_video.py:197-202). Fix the
  base to resolve asset_refs like eng_humo.py:421-427 + an integration test feeding
  `render_driver.build_request()` output into `_partner_inputs()`. Without this NO cloud i2v
  works on any minted card.
- **Cost enforcement is code, not prose:** replace the single global `OTR_CLOUD_VIDEO_EST_USD`
  with a per-row pricing table (node_key x duration x resolution) feeding
  `estimate_cost(request) -> CostQuote` reserved before invoke (cloud_media_backend.py:224-232
  already defines CostQuote; adapters don't use it). Per-episode cap rides that.
- Per-row STATIC conformance test (every emitted kwarg exists in the row's pinned
  required/optional inputs -- generalize tests/test_cloud_video_adapters.py) + gated live
  smoke leg. kling_avatar polish: omit optional `prompt` when empty (today it sends "").
- Spike procedure is MANUAL with a recorded run trace (attempt #, seed, prompt rev, row,
  verdict); retry-once rule lives in the procedure, not code. Frame-extraction helper (ffmpeg
  stills every ~0.5s) so "every frame readable" reviews leave repeatable artifacts.
- Candidate ranking (two rows max): cheapest passing row WITH seed support first; tie-break on
  prompt-interface richness.
- Workflow JSON: no new widgets expected for cloud approaches (V-11, verify saved-JSON selector
  exposure); the hybrid's overlay wiring belongs to ideo_word_vid, not here.

## Risks

- Text-warp even with preservation prompts -- THE risk; hence the frame-by-frame bar + retry-
  once-then-fail rule.
- Audit may find nothing usable (honest BLOCKED outcome is acceptable; the stills lane still
  ships value).
- "Razzle vs slightly-moving poster" -- operator eyeball on the first passing spike decides if
  the tier is worth productizing.

## Rough size (complexity, not time)

Phase 0 audit: SMALL-MEDIUM (S0 pin flow exists end-to-end; V3 rows excluded from thin
pinning). Phase 1A spike: SMALL once a row pins. Productizing: MEDIUM (new reactivity class +
duration fit + tests). Sequencing: Phase 0 any time; Phase 1 after S1 stills + ideo_word.
