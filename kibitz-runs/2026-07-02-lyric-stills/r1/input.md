# Ideogram Lyric-Stills Lane (typographic beat cards)

**STATUS: DRAFT (pre-kibitz)**
Candidate plan -- ideas window 2026-07-02. Not in the forward order; coder window pulls when ready.

## Problem / goal

Music and announcer beats currently render as abstract/visualizer or portrait footage. Ideogram
(cloud_ideogram_v4, already pinned + priced + prompt-profiled) is the one engine in the catalog
where in-image TEXT is the point. Idea: a "lyric-stills" visual lane -- every beat in a music
(or optionally announcer) segment gets a UNIQUE designed typographic still rendering the beat's
words: a lyric-video look, period-styled (1940s radio poster / title-card typography), one card
per beat, cut on the beat boundaries the pipeline already has.

Text-only ("pure 100% lyrics") -- the card IS the text plus era styling; no characters, no scene.

## Proposed approach

- New selectable still engine `ideogram_lyric_still` (working name) registered as a VIDEO-role
  option for `music_visual` (and possibly `announcer_visual`), riding the S1 stills-lane
  infrastructure (canonicalize_image + portrait-mint-style gates). Selectable only, EMPTY
  default_roles -- consistent with the S3 rule (never automatic).
- Per beat: take the beat's text, chunk it to <= 8 words per rendered surface (the known
  Ideogram quoted-text length cliff in PROMPT_PROFILES.md), build the prompt from the
  cloud_ideogram_v4 profile + a lyric-card style tail derived from the story brief / era tail
  (finish_visual_prompt chain), request the role canvas via the `resolution` COMBO (quantize to
  nearest allowed value, then pad -- the COMBO options are excluded from the pin, so allowed
  values must be captured at S1).
- Still -> clip: reuse the existing still_kenburns / still_motion path to give each card its
  beat-length duration; audio-reactivity (S-C C1 profile, default-ON per operator amendment)
  can drive subtle scale/pulse later -- v1 is Ken Burns only.
- Text SOURCE per beat type:
  - character/announcer beats: the actual script line (verbatim, possibly truncated to the
    strongest <= 8-word fragment -- deterministic rule, seed-keyed).
  - music beats (instrumental -- "obviously it won't work for the music" literally): NO lyrics
    exist. Options: (a) episode title / tagline card, (b) writer emits a one-line "lyric card"
    string per music beat via the Meta brief protocol (the same seam _otr_music_prompt.py uses),
    (c) skip -- lane only valid for spoken beats. Leaning (b): small writer-brief addition, keeps
    cards unique per beat. OPEN QUESTION below.
- Caching: RequestCacheKey already covers pre-submit; identical beat text + style = cache hit,
  so retries are cheap.

## What it touches

- New adapter in the eng_ cloud stills family (S1-pattern), registered dark/fail-closed.
- Engine registry + "which model" dropdown label.
- Prompt profile addition in docs/2026-07-02-cloud-engines/PROMPT_PROFILES.md (lyric-card tail).
- IF option (b) for music beats: a small writer-brief field (Meta brief protocol) -- story-LLM
  changes are currently PARKED (UpstreamStoryLab refactor), so (b) may have to wait or ride the
  existing music-prompt seam without touching the writer core.
- Workflow JSON: NONE in v1 if it registers as a selectable engine through the existing selector
  (no new widgets -- V-11). Verify at build time.

## Risks

- Ideogram text fidelity beyond ~8 words degrades -- chunking rule is load-bearing.
- Per-beat cost: $0.043 (TURBO) x ~beats/episode; a 20-beat episode ~ $0.86-2.60 by speed tier.
  Under the flat budget machine (S0) -- fine, but worth a per-episode estimate stamp.
- Repetitive look across an episode: mitigate with per-beat seed + palette rotation from the
  style inventory.
- SFW: script lines pass through as rendered text -- lines are already SFW-gated upstream.
- Depends on S1 stills lane landing first (canonicalize_image, COMBO resolution capture,
  conformance test). This plan should NOT front-run S1.

## Open questions

1. Music-beat text source: writer-brief lyric-card line (b) vs title/tagline (a) vs spoken-beats
   only (c)? (b) touches the parked story-LLM area -- needs a "seam-only" ruling.
2. Also offer for `character_video` beats (word-card cutaways during dialogue) or keep it to
   music/announcer roles in v1?
3. TURBO vs DEFAULT rendering_speed as the lane default?
4. One card per beat vs multiple cards per long beat (chunked lyric progression)?

## Rough size

Small-medium: one adapter + prompt profile + chunking rule + tests (~1 session), riding S1
infrastructure. +1 small session if the writer-brief lyric-card field is approved.
Sequencing: AFTER cloud S1 stills lane; no collision with the active S0/S1 coding window.
