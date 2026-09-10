# Sprint 4 finished-diff review anchor

Driver: Codex. Baseline: c7cbf45b on v2.0-alpha. Scope: finish Claude's interrupted
My Story implementation from docs/2026-09-10-my-story-d1-design.md; do not redesign
the source banks. User's current emphasis: reuse the existing machinery and prove
the user's input produces a clean ledger for the established downstream path.

VERDICT: implementation ready for independent finished-diff review and full
regression comparison; live publication is not yet qualified.

CONFIRMED against the real Windows files:

- Pure input admission and durable draft coordinator are shared by queued
  validation and evaluated writer execution. Linked fields defer persistence;
  known conflicts still fail before downloads. Prompt/node identity is retained.
- The resolver selects user fields before snapshot/fetch/original-premise routes.
  Raw character request survives the legacy clamp. The pre-roll request owns
  the draft digest, including an automatic visual-style selection.
- New runner uses the existing structured-call ladder, deterministic voice
  picker, production ledger setters, authorship/TTS/word-delivery helpers and
  shared writer tail. Exclusive cast takes precedence; uncastable accepted
  plans are terminal before creative calls. Names, genders and ages are kept.
- Canonical node 1 has 37 widgets; appended link 291 carries script_json to
  mux input 10. Original slots and links remain aligned.
- Required delivery checks wire shape, bank, per-run token and video stem
  before media writes, then published-file existence and persisted OBS path.
  Legacy banks keep optional publication; required-file disappearance affects cache.
- Canned-model tests drive the REAL writer through input resolution, real
  ledger assembly and shared-tail handoff; existing content-owned structural
  validation and authorship verification pass. This is not a live model proof.

Inspect all changed/untracked production files plus tests/test_my_story_*.py.
Concentrate on actual ledger compatibility, source authority, ownership,
imports, errors and delivery/caching. Do not re-open prose-quality work or the
settled D1 design. Local tests cover 135 new cases; full suite comparison pending.

Interpretation of the standing no-length-chasing rule: act/frame row counts
remain prompt guidance rather than arbitrary hard rejection bounds. Nonempty
spoken text, cast coverage, identity and exact interstitial topology are enforced.

MUST-FIX currently known: none unaddressed. Two read-only internal audits found
the cast/input/draft/mux gaps listed above; the driver verified and fixed them.
An independent CLI review is still required before push.
