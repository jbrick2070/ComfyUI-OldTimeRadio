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

## PHASE 0 -- pin audit (first deliverable, runnable now)

Audit the LIVE core's partner-node catalog against ACCEPTANCE FILTERS (not a named shortlist):
image/reference input + text-prompt input + NO required audio (kling_avatar exempted as the
audio-available candidate) + usable duration/fps vs bookend lengths + seed behavior + output
VIDEO + price. Deliverables per surviving row: pin via scripts/otr_pin_partner_nodes.py
(CURATED_ROWS is manual -- each new row is an explicit entry) + pricing stamp
(CANDIDATE_ROWS_ADDENDUM checklist) + prompt profile + conformance test + dark/fail-closed
registration, EMPTY defaults. Note: V3 dynamic rows need V3-expansion, not thin pinning.

**Decision tree (the audit's output):**
- Non-V3 promptable i2v row found -> Phase 1A (worded-card spike).
- Only V3 rows -> tier WAITS on S1 V3-expansion; revisit then.
- Nothing usable -> worded razzle is BLOCKED, stated loudly. A wordless-plate ambience spike
  (wan post-V3-expansion) is NOT a fallback for the word goal -- it is only ever the cloud half
  of the animate-then-overlay hybrid, which lives in the ideo_word_vid doc behind its own
  local-overlay ruling. This plan stays cloud-pure.

## PHASE 1 -- build (conditional; after S1 + ideo_word land)

- **1A worded-card spike:** one bookend card through the pinned row, low motion strength,
  prompt = world-motion phrases + era tail + text-preservation directive. Judged by the
  acceptance checklist frame-by-frame. Two candidate rows max (best filter scores);
  kling_avatar included if nothing better pins.
- **Prompt-shaped mint (design note, not a deliverable here):** the stills lane's card prompts
  gain animation-friendly composition guidance -- TWO contracts: C-worded (silhouette
  letterforms + atmospheric motion sources, for 1A) and C-plate (extends word_video_plate, for
  the hybrid). Lands as notes in the ideo_word / ideo_word_vid docs.
- **Duration fit is REQUIRED, not polish:** provider clip length vs ledger bookend duration
  needs trim-or-loop at build (mux-LAST frozen audio untouched). Boomerang aesthetics = post-
  polish, cut from first build.
- **Adapter reality:** a prompt-conditioned, non-audio i2v row is a NEW reactivity class in
  _CloudVideoBase (current subclasses are audio-required or mute) -- expect base-class work,
  not just a row entry. Per-row pinned-kwargs conformance test mandatory (the S1 generic
  profile->schema guard).
- Workflow JSON: no new widgets expected for cloud approaches (V-11, verify saved-JSON selector
  exposure); the hybrid's overlay wiring belongs to ideo_word_vid, not here.
- Cost: bookends = 2 clips/episode; i2v class ~$0.25-1.00+/clip by provider -- audit's pricing
  stamps drive the pick; S0 budget machine caps it.

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
