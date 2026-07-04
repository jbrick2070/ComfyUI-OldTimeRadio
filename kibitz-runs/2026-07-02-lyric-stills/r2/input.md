# Ideogram Lyric-Stills + ideogram_lyric_vid Lane (typographic beat cards)

**STATUS: DRAFT (post-roundtable r1, kibitz r1 in flight)**
Candidate plan -- ideas window 2026-07-02. Not in the forward order; coder window pulls when ready.
Hardened by: roundtable pass01 (Grok 4.3 + DeepSeek v4-pro + GLM-4.6, ~$0.03, all claims
grounded) + kibitz codex r1. Artifacts: docs/2026-07-02-lyric-stills/roundtable/pass01/,
kibitz-runs/2026-07-02-lyric-stills/.

> OPERATOR RULINGS (2026-07-02): (1) "lyrics" = CHARACTER + ANNOUNCER beat lines (spoken script
> text). (2) Cloud only -- Comfy Cloud rows, NO local engines in this lane. (3) An `ideogram_vid`
> variant to animate the cards -- "make those ideograms shine." (4) MUSIC beats: use the procgen
> EPISODE TITLE to mint a title card WITHOUT any letters or words -- a wordless visual
> interpretation of the title (title-mood card), so `music_visual` joins the lane too.

## Problem / goal

Ideogram (cloud_ideogram_v4 -- pinned at partner_nodes.yaml:126, priced $0.043-0.13, prompt
profile with the 8-word text cliff in PROMPT_PROFILES.md) is the one engine in the catalog where
in-image TEXT is the point. Lane: every CHARACTER and ANNOUNCER beat gets a UNIQUE designed
typographic card rendering that beat's spoken words -- lyric-video look, period-styled (1940s
radio poster / title-card typography), one card per beat, cut on existing beat boundaries.
A second-tier `ideogram_lyric_vid` engine animates hero/bookend cards with a cloud i2v pass.

Text-first -- the card IS the words plus era styling; no characters, no scene.

## Proposed approach (v1 = STILL lane only; vid variant = v1.5)

- **`ideogram_lyric_still`** -- follows the EXISTING flux_still engine pattern (video-role engine
  that mints a still, then the still_kenburns path gives it beat-length duration). Registered as
  a selectable option for `character_video` and `announcer_visual`. EMPTY default_roles (S3 rule
  -- never automatic). Output contract identical to flux_still (still -> clip inside the
  established path), so no new output-type architecture. [roundtable/codex ambiguity resolved:
  it is a stills-mint video-role engine, NOT a composite video adapter.]
- Per beat: take the beat's script line and reduce it to ONE excerpt card by a DETERMINISTIC
  rule -- first clause boundary (punctuation split) capped at 8 words; no semantic "strongest
  fragment" scoring (GLM: "strongest" implies undefined semantics). v1 explicitly accepts
  excerpt-not-verbatim as a documented limitation; multi-card chunked progression is CUT from
  v1 (cost multiplier, edit complexity).
- Prompt: cloud_ideogram_v4 profile + a NEW lyric-card style tail (period typography) added to
  PROMPT_PROFILES.md -- the profile currently has only the shared PERIOD TAIL + 8-word rule; the
  lyric-card tail is a required schema/profile addition in this plan, not assumed existing.
- **Music beats -- wordless TITLE-MOOD card:** for `music_visual`, the text source is the procgen
  EPISODE TITLE (already minted by the pipeline -- zero writer/story-LLM changes). The prompt is
  a visual interpretation of the title with an explicit NO-TEXT directive ("no letters, no words,
  no typography"). Composition anchor (operator 2026-07-02): Ideogram's rendition of an OLD-STYLE
  RADIO as the centerpiece, mixed with the story's Meta brief (finish_visual_prompt / era tail
  chain -- the same brief-grounding seam the other visual engines use) + the episode visual
  style. Same engine, second prompt mode (`card_mode: lyric_text | title_mood`, keyed by role). Note this inverts Ideogram's usual strength (text is
  its specialty), but its composition/poster aesthetic is the draw; negative-text compliance is
  an eyeball item on the live smoke -- stray glyphs = reject card.
- Resolution: `resolution` is a COMBO whose options are excluded from the pin -- S1 must capture
  allowed values; quantize-then-pad rule rides that (verify-at-build).
- **`ideogram_lyric_vid` (v1.5, hero/bookend beats ONLY -- never per-beat):** grounded reality:
  there is NO plain Kling i2v row (only cloud_kling_avatar / cloud_kling_lipsync -- avatar on a
  text card would mint a "talking poster", wrong); `cloud_wan_i2v` is the ONLY valid mute i2v
  row and takes NO text prompt (prompt_extend boolean only) -- it is a black-box motion pass, so
  "ink bleeding / letters catching light" CANNOT be prompted, only hoped for and eyeballed;
  `cloud_seedance_2` is dark until its V3 pin expansion. Therefore: v1.5 = cloud_wan_i2v on the
  minted card, hero/bookend only, hard per-episode budget cap, text-legibility eyeball bar --
  if text warps, the lane stays stills-only. No A/B needed (there is exactly one candidate).
- Caching: RequestCacheKey makes RETRIES of the identical request cheap; it does NOT dedupe
  across beats (key includes seed/params -- per-beat seed defeats cross-beat hits). Stated
  narrowly on purpose.

### Still-to-clip lifecycle (build contract -- kibitz claude r1)

Three-stage chain, each stage owned by a NAMED existing seam:
1. `invoke_partner_node("cloud_ideogram_v4", ...)` -> PartnerResult IMAGE. The adapter passes
   `estimated_usd` read from the rendering_speed selection ($0.043 TURBO / $0.086 DEFAULT /
   $0.13 QUALITY) into the S0 budget machine -- never a hardcoded constant.
2. `canonicalize_image` (S1 -- TODAY a stub raising `_not_built_yet("image", "S1")`,
   cloud_media_canonical.py:106-109; hard prerequisite). Resolution quantize-to-preset + pad to
   role canvas is canonicalize_image's job per the S1 contract, NOT the adapter's -- the adapter
   only picks the nearest `resolution` COMBO preset (allowed values captured at S1).
3. Existing ffmpeg still->clip Ken Burns pass exactly as the local stills families do
   (_CheapFamilyBase pattern) -> beat-length silent clip dict. Ken Burns params (direction,
   speed) seed-keyed from day one.
Adapter class: follow the _CheapFamilyBase (stills-mint) shape with the cloud fetch swapped in
-- NOT _CloudVideoBase (that family canonicalizes VIDEO). First cloud IMAGE-sourced row in the
video registry = new pattern; confirm no-JSON-change via the V-6 rule (COMBO is the full static
registry; a registered engine auto-appears in the dropdown) at build.

## What it touches

- New engine in the stills-mint family (flux_still pattern), registered dark/fail-closed.
- Engine registry (`nodes/_otr_image_engines/registry.py` CAPABILITIES -- which today has ZERO
  cloud rows; see prerequisite) + "which model" dropdown label.
- PROMPT_PROFILES.md: lyric-card tail EXTENDS the existing cloud_ideogram_v4 profile entry
  (which already holds the quoted-text shape + 8-word rule) -- one entry, no duplicated rule
  (drift vector otherwise).
- v1.5 only: invoke_partner_node chain still -> cloud_wan_i2v; budget-cap knob.
- Workflow JSON: NO NEW WIDGETS expected (V-11); MUST verify at build that the existing selector
  exposes the new registered row in `workflows/otr_scifi_16gb_full.json` -- if it does not, the
  JSON changes in the SAME commit per hard rule 0.

## Hard prerequisite (do not front-run)

S1 stills lane LANDED: cloud stills rows registered in CAPABILITIES, canonicalize_image, COMBO
resolution capture, profile->schema conformance test. None of this exists in code today
(registry.py:107-139 has no cloud rows). This plan is S1+1, not an S1 task.

## Risks

- Ideogram text fidelity beyond ~8 words degrades -- the excerpt rule is load-bearing. Add an
  OCR/legibility spot check to the live smoke before calling the lane production-ready.
- Cost: stills ~$0.043/card (TURBO) x spoken beats -- cheap. Vid variant is the dominant spend
  (~$0.50-0.70 per 6s i2v clip => ~$10-14/episode if per-beat -- hence hero/bookend-only + cap).
- Repetitive look across an episode: per-beat seed + palette rotation from the style inventory;
  record chosen palette/layout per card (repeat guard) -- acceptance-test optional.
- SFW: script lines pass through as rendered text -- already SFW-gated upstream.

## Open questions (NEEDS-DECISION at pull time)

1. TURBO vs DEFAULT rendering_speed default (expose as engine param; TURBO likely fine).
2. Accept excerpt-only cards as the v1 contract? (Alternative -- multi-card progression -- is
   cut; reopening it changes cost and edit timing.)
3. v1.5 vid variant: ship gated with v1, or hold until a promptable cloud i2v row (Seedance V3
   expansion) exists?

## Rough size

Small-medium: one stills-mint engine + prompt-profile tail + excerpt rule + tests (~1 session)
AFTER S1 lands. v1.5 vid pass +1 small session. No collision with the active S0/S1 window.
