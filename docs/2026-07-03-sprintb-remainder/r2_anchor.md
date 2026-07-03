# R2 anchor (Cowork Claude, code-grounded) -- coding plan / implementability

VERDICT: BUILDABLE. still_word is implementable with a MINIMAL, PRECEDENTED diff.
The load-bearing wiring question (does the composer see the per-role video engine?)
is RESOLVED by an existing pattern.

## Grounded: the composer-plumbing seam already exists (de-risks the build)

- CONFIRMED: OTRMetaBriefImagePromptGen.generate() takes script_json +
  image_policy_json (nodes/otr_meta_brief_image_prompt.py:1333-1391). It does NOT
  take the raw video policy -- BUT image_policy_json ALREADY carries per-role flags
  DERIVED from the video-engine selection: `_still_aspects_from_policy`,
  `_mesh_fodder_roles_from_policy`, `_talking_roles_from_policy` (:1388-1390).
- Therefore still_word needs NO novel plumbing: add a `still_word_roles` (roles
  whose selected video engine == still_word) derived flag to the policy, carried
  through image_policy_json exactly like mesh_fodder_roles/talking_roles, and a
  `_still_word_roles_from_policy` reader passed into derive_image_prompts(). The
  object-build loop then branches the PROMPT when the beat's role is a still_word
  role. This is the SAME mechanism the mesh-fodder fork already uses (:1240-1284),
  so it is a precedented, minimal-diff change -- not a new architecture.

## Grounded: still_word as a video engine

- CONFIRMED: still_flat/still_pan/still_motion are cheap_families.py video engines
  that hold/pan a PROVIDED still. still_word is a sibling with the same flat-hold;
  its ONLY delta is upstream (the base still is generated from a word-driven /
  abstract prompt). The video engine itself is nearly a still_flat clone -> tiny.

## MUST-FIX (fold in)

1. Pool-mode reality check: pooling was DELETED 2026-07-01 (rip-sfx-broll). The
   "pooled char beats -> non-spoiler mood SET" mode targets a mode that may not
   exist in the live pipeline. v1 SCOPE: ship word-driven (char/announcer) +
   abstract-episode-title (music); mark the pool-mood SET as a DEFERRED mode
   (named, fail-LOUD-or-skip if pooling is off) until/unless pooling returns.
   CONFIRM against the current beat model before coding the pool branch.
2. Determinism: the composed prompt now depends on the video-engine selection, so
   prompt_hash changes when still_word is toggled -- expected, but assert the
   still_word prompt is DETERMINISTIC for a fixed beat+seed (no wall-clock/RNG).
3. Model-agnostic proof is a REQUIRED test: same still_word beat, two image models,
   both render a valid still (no engine coupling).

## SHOULD-FIX

- Reuse the mesh_fodder precedent's shape EXACTLY (derived role set in policy +
  a *_from_policy reader) so the diff is reviewable and consistent.
- Keep the still_word video engine a thin still_flat subclass (share the ffmpeg
  hold path) rather than a fresh renderer.

## Open (r3 wiring)

- Exact policy field name + where OTR_VideoDirector/ImageDirector computes the
  derived role sets (mirror mesh_fodder_roles' computation site).
- Ordering: still_word (video) consumes the base still minted by the image slot --
  confirm the image mint precedes the video hold (it already does for still_flat).
