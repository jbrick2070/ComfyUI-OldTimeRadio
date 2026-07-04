# R4 Codex Anchor Review

- VERDICT: yes-with-fixes. The plan has converged architecturally. Remaining
  blockers are acceptance gates: exact prompt-surgery inventory, no-fallback
  tests, and staged transplant boundaries.

## MUST-FIX BEFORE BUILD

1. [Prompt surgery inventory] CONFIRMED: the build needs a table that maps every
   active prompt site to one action: shared, profile variable, story-model
   guardrail, or deferred/dead. The existing audits are the source material, but
   the final coding plan must turn them into an implementation checklist.

2. [No fallback] CONFIRMED: the plan must require tests that fail if
   `media_archive` calls `_fetch_rss_seed_or_die`, uses `_INVENTOR_SYSTEM`, or
   emits the hardcoded "science-fiction audio drama" prompt strings. This is
   the user's root concern.

3. [Visual transplant boundary] CONFIRMED: do not include deep render-driver
   prompt edits in the first source/story transplant. They belong to a separate
   visual transplant stage after seam policy tests are green.

4. [Workflow touch gate] CONFIRMED: no canonical workflow edit until there is a
   manifest listing code changes, widget additions, JSON edits, validation
   commands, and expected widget-vector deltas.

## SHOULD-FIX

1. [upstream_story_lab] Add a tiny pure scaffold after kibitz: package files,
   schema docs, fixture JSON, and validation script are fine; production imports
   are not.

2. [Story model auto] Make deterministic `auto` choice explicit in the final
   plan: first registered model per source bank unless a seeded selector is
   implemented.

3. [Visual style names] Lock `archival_documentary` now to prevent future
   `media_archive` id collision.

## OPTIONAL / NICE-TO-HAVE

1. Add a single README diagram in `upstream_story_lab`.

2. Add a transplant manifest section for "files not to touch yet."

## CUT THESE

1. Cut public-domain workflow exposure from the first build.

2. Cut live archive RSS/network fetching from the first build.

3. Cut any runtime read from `upstream_story_lab`.

## VERIFY-AT-BUILD CHECKLIST

- Pydantic v2 available in test venv.
- `OutlineRequest` defaulted additions preserve old science calls.
- `_otr_style_picker.pick_style()` override kwargs preserve existing tests.
- coda helper returns `_otr_line_composer.LineResult`.
- writer optional widget count changes only in transplant chunk.
- API/workflow whitelists include new writer widgets in the same chunk.
- visual-style policy tests prove non-cinematic styles do not leak hardcoded
  35mm/film-grain/radio tails.

