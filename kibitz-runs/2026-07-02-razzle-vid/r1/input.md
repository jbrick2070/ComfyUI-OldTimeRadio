# ideo_word razzle-vid -- CLOUD video animation of the word card (high-level, r1)

**STATUS: DRAFT (pre-panel)**
Candidate plan -- ideas window 2026-07-02. SUPERSEDES the Blender 3D razzle doc
(operator-rejected: static extruded text under a camera is not a LIVING world).
OPERATOR RULING: this whole family is CLOUD -- ideo is cloud, so the razzle is CLOUD razzle.
NO local LTX/Wan lanes in this tier.

Family: `ideo_word` stills (CODE-READY/S1-gated) -> `ideo_word_vid` 2D kinetic overlay
(NEEDS-DECISION) -> THIS: the razzle tier -- a CLOUD video engine takes the ideo_word card as
INPUT and animates the WORLD of the card: smoke curling through the letters, neon flicker, rain
on the marquee, the poster world breathing. Words at the core; the world moves around them.

## Problem / goal

Feed an ideo_word card (worded or wordless) into a CLOUD image-to-video pass so the card becomes
an animated period world. Living motion, not camera moves on frozen art. Target slots: the
music_open/music_close bookends first (unique clip per beat -- no pooling with worded cards);
per-beat lyric cards later only if text survives animation.

## Grounded cloud-i2v reality (the gap this plan must close)

The pinned roster has NO promptable i2v today:
- cloud_kling_avatar / cloud_kling_lipsync only (no plain Kling i2v row pinned); avatar on a
  text card = talking poster (wrong).
- cloud_wan_i2v: mute row -- forwards NO text prompt (prompt_extend boolean only) + the
  init_image request-shape bug (top-level vs asset_refs) is unfixed.
- cloud_seedance_2: dark until its V3 DYNAMICCOMBO expansion pin.

**Therefore the razzle tier's FIRST deliverable is a PIN EXPANSION, not an engine:** audit the
LIVE core's partner-node catalog for promptable image-to-video rows and pin the best 1-2 via
the S0 pin flow (scripts/otr_pin_partner_nodes.py + partner_nodes.yaml + drift test + pricing
stamp + prompt profile -- the full row checklist from CANDIDATE_ROWS_ADDENDUM). Candidates to
verify in the live core (verify-at-audit, do not assume): Kling ImageToVideo-class nodes, Veo
i2v, Seedance V3 expansion, ByteDance/other i2v rows. If NO promptable i2v row exists in the
live core, this tier is BLOCKED and says so loudly.

## Candidate approaches (panel to rank)

- **A. New promptable cloud i2v row on the worded card (low motion strength).** The direct
  path: prompt drives the world motion ("neon flicker, drifting smoke, rain streaks" + era
  tail) while asking for text preservation. Text-warp is THE risk -- every i2v model is free to
  repaint letterforms. Spike: one card through the newly-pinned row; legibility eyeball.
- **B. cloud_wan_i2v (existing mute row) on the WORDLESS plate.** No prompt control, black-box
  motion -- but on a wordless word_video_plate there is no text to destroy. Needs the
  request-shape fix (asset_refs.init_image) already owed in the cloud S3-full work. Zero new
  pins.
- **C. Prompt-shaped mint for animation.** Mint the card knowing it will move: Ideogram prompt
  requests animation-friendly composition (strong silhouette letterforms + atmospheric motion
  sources: smoke, rain, neon, crowds). Free quality boost to A/B.
- **D. Animate-then-overlay (warp-proof).** Cloud-animate the WORDLESS plate (B, or A on the
  plate), then the ideo_word_vid 2D kinetic overlay writes the exact words ON TOP of the living
  world. Words always perfect, world always moving. Inherits the ideo_word_vid local-procgen
  NEEDS-DECISION (the overlay is the one local step); if the operator wants zero-local, D is
  out and A carries the worded razzle alone.
- **E. Loop/boomerang polish** on whatever animates (bookends want seamless loops). Small,
  additive, local ffmpeg post (presumed ruling-clean -- it is muxing, not generation; confirm).

## Working preference (to pressure-test)

Pin audit FIRST (it decides everything). Then: A on worded cards if a promptable row pins and
text survives; D as the ceiling if the overlay ruling lands; B as the zero-new-pin wordless
fallback; C always.

## Constraints / notes

- CLOUD ONLY (operator ruling 2026-07-02). The only local steps permitted: ffmpeg loop/mux
  polish (E) and -- pending the standing NEEDS-DECISION -- the D overlay.
- Cost: cloud i2v is the dominant-spend class (~$0.25-1.00+/clip by provider). Bookends only =
  2 clips/episode -- bounded. Per-beat extension = a real budget conversation; hard cap knob
  from S0 budget machine either way.
- Unique clip per beat (no pooling -- worded cards are beat-pinned; family invariant in the
  ideo_word doc). Wordless animated plates ARE pool-safe if pooling ever returns.
- New rows ride the FULL S0 row checklist: pin + pricing stamp + prompt profile + conformance
  test + dark/fail-closed registration + EMPTY defaults.
- Adapter shape: new rows slot into the existing eng_cloud_video adapter family
  (canonicalize_video is real; per-row pinned kwargs, conformance test-locked).
- Workflow JSON: no new widgets expected (V-11, verify); same-change rule if wiring changes.
- Determinism: seed param where the provider exposes one (record resolved behavior per row);
  audio byte-identical untouched.

## Risks / open questions (high level)

- Does the live core even expose a promptable i2v partner node? (Pin audit answers; if not,
  BLOCKED -- and the honest fallback is B on wordless plates only.)
- Text-warp on worded cards even with preservation prompts -- every provider differs; spike per
  pinned row.
- Razzle vs "slightly moving poster": operator eyeball on the first spike clip.
- Provider motion length/fps vs bookend beat duration -- may need loop (E) or trim; mux-LAST
  frozen audio untouched.

## Rough size (complexity, not time)

Pin audit + 1-2 new rows: SMALL-MEDIUM (S0 flow exists end-to-end). A spike: SMALL once pinned.
B: rides the owed request-shape fix. D: MEDIUM integration across two plans. Sequencing: after
S1 stills (needs cards to animate); pin audit can run any time.
