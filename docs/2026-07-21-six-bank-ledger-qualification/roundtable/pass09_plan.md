# Pass 09 -- live word-count ownership R2/R3 audit

## Trigger

The canonical `scifi_news` run `1435f170-78fa-45ec-81a7-779b44533eb7`
reached freeze with a warning that `l003.word_count=21` disagreed with a
whitespace count of 20. Sol stopped the run before TTS qualification and reset
the owned processes/GPU state.

## Review lane

- reviewer: Antigravity, review-only
- exact model: `gemini-3.5-flash-high` (Gemini 3.5 Flash High)
- rounds: scoped R2 hotspot review, then R3 implementation/wiring convergence
- clean worktree: detached throwaway worktree at
  `90e2f85db9721ec73ff603d385630c4ca3827e65`
- driver/coder/judge: Sol only

The first R2 file handoff exceeded its print window. The same Antigravity
conversation was resumed with the exact model and tool use disabled; it wrote a
review artifact rather than modifying the worktree. R3 reused that conversation
and remained review-only.

## Required scope

Inspect the shared tails of `media_archive`, `original`, `public_domain`,
`shakespeare`, `scifi_news`, and `scifi_news_pro` for:

- row-local `text`/`char_count`/`word_count` ownership;
- root, cast, scene, and character/announcer aggregate ownership;
- Phase-0 versus Phase-10 freeze behavior;
- readiness and spoken-hygiene mutations;
- content-owned canonical text versus `text_for_tts` delivery projection;
- hash/seal implications and cross-bank bypasses.

Sol must ground every reviewer claim against the real Windows files, reject
misreads, implement all confirmed sibling fixes, and complete the project,
Bible, workflow, and live qualification gates before resuming the next bank.
