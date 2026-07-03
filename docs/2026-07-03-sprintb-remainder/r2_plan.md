# still_word / word_razzle -- model-agnostic word-driven still engine (kibitz r2 input)

Operator-CONFIRMED architecture 2026-07-03 (supersedes the ideo_word doc AND my
earlier overlay misread). Build `still_word` before B6. Hard rules: NO fallbacks /
dropdown-only defaults; workflow-JSON same-change rule (BUG-LOCAL-097 positional
widgets); UTF-8 no BOM; SFW; suite + Bug Bible + push per green chunk; prod/main
GATED. Panel = codex + antigravity; Claude anchor + judge.

## What still_word IS (operator-confirmed, not my guess)

`still_word` is a NEW model-agnostic VIDEO engine, "a lot like still_flat"
(cheap_families.py sibling; flat hold of a still). It is NOT a text overlay and
NOT literal typography burned on the card. Selected per-role in the VIDEO model
dropdown (OTR_VideoDirector video slots); the IMAGE model that renders the still
is chosen INDEPENDENTLY in the image dropdown -- the image model is NOT coupled
into the video options ("we shouldn't fix the image model into the video
options"). Model-agnostic: ANY image model in the list can render it.

The DIFFERENCE from still_flat is the PROMPT the still is generated from, which
branches by role/mode:

- **character / announcer beats (unique, non-pool):** the still's image prompt is
  GENERATED FROM THE BEAT's SCRIPT WORDS -- the spoken line drives what the image
  depicts. Per-beat unique. NEVER pooled.
- **music beats:** an ABSTRACT picture of the EPISODE TITLE (NO words) built from
  meta["episode_title"] + Meta brief + episode visual style.
- **character beats in POOL mode:** a SET of X non-spoiler mood images conveying
  the sense of the story (meta brief + visual style) WITHOUT spoiling it -- a
  poolable abstract set (consistent with "words forbid pooling": only the
  word-driven per-beat stills are unpoolable).

`word_razzle` = the ANIMATED variant of the word still (razzle-vid). Naming locked
now; full build Phase-gated per docs/GO_FORWARD_NEXT/2026-07-02-ideo-word-razzle-vid.md
(Phase 0 audit is safe filler; Phase 1 gated). Do NOT build the animated path in
this v1 unless in scope.

## Grounded facts (verify-at-build flagged)

- still_flat / still_pan / still_motion are VIDEO engines in cheap_families.py
  (StillFlatFamily :227, StillPanFamily :204, StillMotionFamily :177) -- they take
  a PROVIDED still (asset_refs) and hold/pan it. still_word is a new sibling with
  the SAME hold behavior; the delta is the PROMPT the base still is generated from.
- The prompt composer (nodes/otr_meta_brief_image_prompt.py, generate() +
  object-build loop ~:1130-1400) builds each object's prompt via
  compose_still_prompt(meta, kind, role, beat_id) and does NOT currently see the
  VIDEO engine selection. still_word REQUIRES the composer to know a role's video
  engine == still_word so it builds the word-driven / abstract-title / mood-set
  prompt instead of the normal scene/portrait prompt. VERIFY where the video
  policy (video_models per role) is available to the composer.
- Beats carry their spoken line text (beats have `text`); the word-driven mode
  reads the beat line. Episode title = meta["episode_title"] (writer-stamped).
  Empty title in music mode -> fail LOUD.
- Image-model dropdowns are on OTR_VideoDirector (node 87), combo dynamic from
  all_engine_names(); still_word joins the VIDEO combo (also dynamic). Registering
  it makes it selectable; VERIFY + add a test it is in the video combo; defaults
  unchanged (no JSON widget change unless a value must move).

## Build steps (still_word v1)

1. **still_word VIDEO engine** in cheap_families.py: name "still_word", static-image
   family, still_flat-like flat hold, EMPTY default_roles, model-agnostic (it does
   NOT mint -- it consumes the base still minted by the role's chosen image model).
   Registration + CAPABILITIES + video-combo exposure per the video-engine pattern.
2. **Composer prompt-mode branch (the real work).** Thread the per-role VIDEO engine
   selection into OTRMetaBriefImagePromptGen.generate(); when a role's video engine
   == still_word, build the beat's still prompt by mode:
   - char/announcer unique -> word-driven prompt from the beat script line (via a
     pure, deterministic helper that turns the line into an image prompt; tests).
   - music -> abstract episode-title picture (no words), meta brief + visual style.
   - pooled char -> a non-spoiler mood SET (X images) from meta brief + visual style.
   Determinism preserved (prompt_hash over the composed prompt). Minimal-diff seam
   into the existing kind branches.
3. **Decoupled image model.** The composed prompt is rendered by whatever image
   engine the role's image slot selects -- no change to image-engine resolution.
4. **word_razzle** -- named + stubbed; animated build Phase-gated (not v1).
5. **Family invariant** -- word-driven stills per-beat unique, NEVER pooled;
   music-title + pool-mood sets are the poolable abstract modes.

## Verify-at-build

- Prompt-mode helper(s) PURE + deterministic + edge-case tests (empty line,
  quotes/ellipsis/em-dash/stage-direction; empty episode title fails LOUD).
- Model-agnostic proof: the same still_word beat renders via >=2 different image
  models (no engine-specific coupling).
- No fallbacks; missing beat line (word mode) / missing title (music mode) fail LOUD.
- Audio spine untouched (byte-identical); single resident heavy <= 14.5 GB.
- Suite + Bug Bible + B7 green; push per green chunk; HEAD==origin.

## Open for the panel (r2 coding / r3 wiring)

- Where is the per-role VIDEO engine selection available to the composer
  (generate() inputs -- does it get the video policy, or only the image policy)?
  This is the load-bearing wiring question.
- Is "pool mode" for char beats a live concept post rip-sfx-broll (pooling was
  deleted 2026-07-01)? If pooling is currently OFF, scope the pool-mood SET as a
  DEFERRED mode (named, not built) and ship word-driven + music-title in v1.
- Minimal-diff seam in the object-build loop so still_word does not regress the
  existing kind branches (portrait/scene_character/scene/plate/mesh_*).
- still_word as a video engine that CONSUMES a base still: does it need the image
  slot to have minted a still first (ordering), and how does its flat-hold reuse
  the still_flat ffmpeg path?
