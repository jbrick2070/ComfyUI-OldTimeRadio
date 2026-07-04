# R1 Codex Anchor Review

- VERDICT: yes-with-fixes. The arc is coherent: source brains, story models,
  and visual styles should fill one production ledger through an upstream
  writing spec. The plan still needs sharper boundaries around adapter
  ownership, media-archive story shape, source input shape, and legacy prompt
  surfaces before it is build-ready.

## MUST-FIX BEFORE BUILD

1. [Operator Intent / Ledger-Writing Spec] CONFIRMED: the plan says both "reuse
   the existing multi-stage story/ledger writer" and "build the upstream
   architecture from scratch." Those can coexist only if the boundary is named:
   fresh code builds `StoryInputPacket`, `StoryPromptProfile`,
   `LedgerWritingSpec`, and source-specific brief material, then an adapter feeds
   existing writer stages. The fix is to make `_otr_ledger_input_adapter.py` the
   explicit line between new source brains and existing outline/casting/script
   machinery. Do not create a second independent story-writing pipeline.

2. [Story Model / Tone Layer] CONFIRMED from the user correction: media archive
   is not "sci-fi anthology plot with archive nouns." The architecture needs a
   story-shape layer in or near the prompt profile: `media_restoration_adventure`,
   `cinematic_humorous`, `happy_archive_mystery`, `gentle_thriller`, and
   `broadcast_history_comedy`. Add forbidden plot-pattern guidance for
   Star-Trek/Amazing-Stories drift: no spaceship/lab/twist-of-the-week default
   unless a profile explicitly asks for that.

3. [Prompt Profile Rule] CONFIRMED: the listed prompt inventory is incomplete.
   `rg` finds additional sci-fi-specific prompt language in
   `nodes/_otr_style_picker.py` at the showrunner/descriptor prompt surface
   (`sci-fi radio drama`). Add that file to the prompt audit and source-profile
   scope, or explicitly classify it as dead/test-only. Otherwise non-science
   banks can still inherit sci-fi framing after the main ledger prompts are
   parameterized.

4. [Visual Style Architecture] CONFIRMED: visual style must not be just a final
   suffix. The audit confirms style assumptions in
   `nodes/_otr_story_brief_helpers.py`, `nodes/otr_meta_brief_image_prompt.py`,
   `nodes/otr_shot_lock.py`, and
   `nodes/_otr_video_engines/render_driver.py`. The fix is to require one
   parsed `VisualStylePolicy` object to drive ledger stamping, prompt finishers,
   bookend subjects, image-grade tails, and render-driver fallback prompts. Keep
   role-safety constraints separate from style terms.

5. [Source Banks] [ASSUMPTION] The plan names source banks but does not yet
   define the user-facing source input contract. `media_archive` and
   `public_domain_story` need fixture-first packet shapes before any node UI is
   added: raw title, author/creator, URL or file ref, rights status, source
   summary, source text/hash, key terms, and adaptation trace. Without this, the
   "same ledger" claim can collapse into one-off prompt strings per bank.

6. [No Silent Fallbacks] CONFIRMED: the no-fallback rule is right but must
   distinguish legacy compatibility from fallback. Existing code has many
   `meta.news` and `news_close_brief` consumers. The fix is a tested
   compatibility mirror populated from the active packet, plus hard failures
   when a non-science bank would enter a science-only prompt path.

7. [Hard Workflow Rule] CONFIRMED: deferring
   `workflows/otr_scifi_16gb_full.json` is correct for this phase. R3 must make
   the transplant chunk explicit: append-only widget updates, live input/socket
   checks, workflow validator, JSON round-trip, link audit, and widget/input
   audit. Any R2 plan that edits the workflow before pure contracts are green
   should be rejected.

## SHOULD-FIX

1. [Ledger-Writing Spec] Add a source coda contract now: `real_news_report`,
   `archive_source_note`, and `source_attribution_or_adaptation_note`. The coda
   is one of the most visibly source-specific downstream behaviors.

2. [Public Domain Story] [ASSUMPTION] Public-domain adaptation needs a fidelity
   rule distinct from science/news grounding: preserve named characters, story
   turns, ending, and attribution, while allowing radio-drama compression.

3. [Media Archive] [ASSUMPTION] Treat archive/RSS ingest and archive story
   interpretation as separate stages. Fetching a feed item should not be the
   same operation as turning it into a `StoryInputPacket`.

4. [Visual Style Architecture] Require style-specific negative/forbidden terms
   tests for `anime`, `cartoon`, and `paper_origami`, especially against the
   confirmed hardcoded terms `cinematic`, `35mm`, `film grain`, `1940s radio`,
   and `dramatic film lighting`.

5. [Expected Contracts] Keep canonical new contracts strict, but do not promise
   Pydantic everywhere until existing repo dependency/version facts are
   checked. If Pydantic v2 is already in use, this is CONFIRMED; otherwise mark
   it verify-at-build.

## OPTIONAL / NICE-TO-HAVE

1. Add a small Mermaid diagram to the final plan showing the split:
   source bank -> source packet -> ledger-writing spec -> production ledger, and
   visual style -> policy -> ledger/meta -> still/video prompt seams.

2. Define a one-page "prompt variable inventory" table as the acceptance gate
   for R2.

## CUT THESE

1. Cut live external archive vendor selection from the first build. Fixture-first
   packets are enough to prove architecture and avoid network behavior during
   contract tests.

2. Cut public-domain file path UI from the first transplant unless the adapter is
   already complete. An inert widget creates positional workflow risk without
   usable behavior.

3. Cut multiple visual-style node variants. One catalog-backed
   `OTR_VisualStyleDirector` is enough; styles are data/policy, not separate
   nodes.

4. Cut any hidden "legacy default" compatibility mode for new banks. Legacy
   behavior is allowed only when `source_bank == "science_news"` or when an old
   unwired workflow intentionally omits new sockets.
