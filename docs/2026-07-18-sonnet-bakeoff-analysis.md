# Sonnet-4.5 Bake-off -- Bank Ranking & Version-Evolution Analysis

**Date:** 2026-07-18. Companion to `docs/2026-07-17-model-bakeoff-scoreboard.md`.
**Panel:** Codex (`gpt-5.5`, file-grounded, auto) + Antigravity (Gemini 3.1 Pro, two file-grounded
ranking passes, run manually) + Claude anchor/judge. Codex reviewed the METHOD and forced the
**three-lane split** below (story quality vs story-only reliability vs full-media suitability)
instead of one blended "newer is better" score. Antigravity then independently ranked the
transcripts and **converged** with the anchor on the top 2, the bottom 3, and every version
verdict -- the only disagreements are two adjacent middle pairs (noted inline), and the two agy
passes even split against each other there, which is what a genuine near-tie looks like. Run:
`kibitz-runs/2026-07-18-sonnet-bakeoff-ranking/`.

## Evidence manifest (what is actually rankable)

Every cell is one of: **rankable** (real frozen transcript), **failed-with-log** (leg raised;
partial or no transcript -> reliability data only), **stub** (`words=0`).

| Bank | 420 | 720 |
|---|---|---|
| media_archive | rankable | rankable |
| original_radio | failed-with-log (framing gate) | rankable |
| scifi_fable2 | rankable | rankable |
| scifi_codex | rankable | rankable |
| media_archive_v3 | rankable | rankable |
| public_domain_story_v3 | rankable | rankable |
| shakespeare_v3 | rankable | rankable |
| scifi_fable2_v3 | failed-with-log (rules_id bug) | **stub** (words=0) |
| scifi_codex_v3 | rankable | rankable |
| scifi_sonnet_v3 | rankable | rankable |
| scifi_codex_v4 | failed-with-log (all-caps gate) | rankable |

Ranking uses the **720 rung** (the only rung graded blind on uniform code across all banks).

## Overall leaderboard -- all reviewers combined (the headline)

Every renderable bank, top-first, under Sonnet-4.5. Combines Fable (blind /25), Antigravity pass 1
and pass 2 (each an independent file-grounded rank), and the Claude anchor. "Renders" = which tiers
went green (the reliability lane -- a top script that only survives one tier is flagged).

| # | Bank | Fable /25 | agy-1 | agy-2 | Renders | Verdict |
|---:|---|---:|---:|---:|---|---|
| 1 | **scifi_codex_v4** | 24 | 1 | 1 | 720 only (fragile) | Best script (earned reversal); production-fragile |
| 2 | **scifi_fable2** | 24 | 2 | 2 | both (robust) | Best prose; the shippable pick |
| 3= | scifi_codex (base) | 21 | 4 | 3 | both | Sharp institutional turn |
| 3= | scifi_codex_v3 | 21 | 3 | 4 | both | Deeper dilemma, ends on indecision |
| 5 | media_archive | 20 | 5 | 5 | both | Standoff + confession; repeats a beat |
| 6 | original_radio | 18 | 6 | 6 | 720 only | Strong reveal; announcer/tag drift |
| 7 | media_archive_v3 | 14 | 7 | 7 | both | Deadline conflict; film-stock self-contradiction |
| 8 | public_domain_story_v3 | 13 | 8 | 8 | both | Placeholder names spoken aloud |
| 9 | shakespeare_v3 | 13 | 9 | 9 | both | Attribution collapse |
| 10 | scifi_sonnet_v3 | 12 | 10 | 10 | both | An essay, not a drama |
| -- | scifi_fable2_v3 | DNF | -- | -- | neither (bug) | Cannot run: `rules_id` bug (any writer) |

**Consensus is near-total.** Every reviewer agrees on the top 2 and the bottom 3, and on the exact
5-6-7-8-9-10 spine. The only wobble is the **3=/3= tie** (`scifi_codex` base vs `scifi_codex_v3`,
both 21/25 -- the two agy passes literally split against each other, base has a real turn, v3 has
the deeper line but ends on indecision) and a **soft 6/7 contest** (agy pass 2 argues
`media_archive_v3` sustains a cleaner two-hander than `original_radio`'s drift-prone reveal). Treat
3-4 and 6-7 as tied pairs, not firm ranks.

## LANE 1 -- 720 story quality (blind Fable grade, /25): the winner/loser ranking

1. **scifi_codex_v4 -- 24.** "The Halicin Gamble": complete 3-hander, earned reversal (the AI's
   reward function buried the renal-toxicity file), a real consequence. Best *play*.
2. **scifi_fable2 -- 24.** Best line-level prose in the set; loses the tie only because at 720w
   Sonnet writes it as a near-monologue (787 words across 13 lines).
3. **scifi_codex -- 21.** Three opposed institutional wants; a sharp turn (nobody in the room
   gets picked).
4. **scifi_codex_v3 -- 21.** Moving dilemma, three clear voices, but ends on deliberate
   indecision -- no turn.
5. **media_archive -- 20.** Real standoff + a 20-year-secret confession; one beat repeats.
6. **original_radio -- 18.** Genuine gothic reveal, undercut by scrambled speaker tags + name drift.
7. **media_archive_v3 -- 14.** Self-contradicts (print stock vs nitrate); garbled tag.
8. **public_domain_story_v3 -- 13.** Placeholder names spoken aloud ("Explain this, THE TIME
   TRAVELER"); broken final line.
9. **shakespeare_v3 -- 13.** Attribution collapse -- a character speaks the bait she's meant to
   overhear; truncated.
10. **scifi_sonnet_v3 -- 12.** An essay in three voices; states its own inertness ("the stakes
    remain hermeneutic, not practical"). The clear cut.

**Winners:** the codex circuit (v4/base/v3 take 3 of the top 4) and fable2. **Losers:** the
`scifi_sonnet_v3` monologue-essay and the `legacy_many_pass_v3` lanes (shakespeare/public_domain/
media_archive `_v3`), which share a failure signature under Sonnet: **speaker-tag scrambling and
placeholder-name leakage** the codex/fable2 lanes don't exhibit.

## LANE 2 -- story-only reliability (does the leg render green?)

Green both tiers: media_archive, scifi_fable2, scifi_codex, media_archive_v3,
public_domain_story_v3, shakespeare_v3, scifi_codex_v3, scifi_sonnet_v3 (8/11).
- **original_radio:** FAIL@420 (news_source_framing gate -- Sonnet framed it as testimony),
  GREEN@720. Draw/length-dependent, not a hard block.
- **scifi_codex_v4:** FAIL@420 (codex all-caps-word gate), GREEN@720. Draw-fragile.
- **scifi_fable2_v3:** FAIL both tiers -- `rules_id` bug, model-independent, cannot render with
  ANY writer until fixed.

## LANE 3 -- full-media production suitability (survives TTS+video+obs?)

- **scifi_codex_v4 @720 canonical: FAILED** on a codex 240-char `string_too_long` contract on a
  fresh source (a *different* gate than its 420 all-caps fail) -> the top-scoring script is the
  least reliable producer.
- **scifi_fable2 @720 canonical: SUCCESS** -> obs_publish OK, 406 MB episode ("The Stone
  Frequency"). The shippable Sonnet pairing.

## Version-evolution -- each family's latest vs predecessor (the "v4 vs predecessor" ask)

Verdict values: newer-better / newer-worse / mixed / unjudgeable-code-fail.

- **codex: base -> v3 -> v4 = MIXED (better script, worse reliability).**
  On 720 story quality, v4 (24) > v3=base (21): the v4 pack's proof-pressure delta -- a stated
  *want*, a *gating proof*, a *mandatory cost beat*, and *one reversal*
  (`nodes/story_packs/scifi_codex_v4/scifi_codex_v4.json`) -- is exactly what converts the codex
  lane from "sharp turn" / "indecision" into a fully-arced tragedy. BUT the same tightened
  contracts that make v4 rigorous are what Sonnet's verbose, emphasis-caps style trips: v4 is the
  ONLY codex version that failed 420 (all-caps) and the ONLY one whose full-media leg failed
  (string cap). Net: **v4 is a genuine story upgrade and a reliability downgrade** on this writer.
- **media_archive: base -> v3 = story NEWER-WORSE; reliability + full-media MIXED** (panel-refined).
  Story: base (20) is a coherent standoff+confession; v3 (14) regressed -- it self-contradicts on
  the film stock ("standard theatrical print stock" then "this nitrate stock's volatile") + garbled
  tags. Reliability: MIXED -- BOTH versions render green at 420 and 720, so the `legacy_many_pass_v3`
  seam degrades the SCRIPT without changing the green rate. Full-media: MIXED -- neither was run
  through the canonical pipeline. Net: the version bump hurt the writing, not the rendering.
- **fable2: base -> v3 = story + full-media UNJUDGEABLE-CODE-FAIL; reliability NEWER-WORSE**
  (panel-refined). base (24) is the set's best prose and passes end-to-end; v3 has no transcript and
  no render to grade (unjudgeable). But on the reliability lane the bump is demonstrably NEWER-WORSE:
  it BROKE rendering -- `nodes/_otr_scifi_fable2.py:2307` hardcodes `rules_id == 'scifi_fable2'` and
  rejects `scifi_fable2_v3`, so v3 fails both tiers with ANY writer. Fix that first
  (`docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md`), then re-run for a real story verdict.

## Ideas / implications

1. **A version bump is not automatically an upgrade under a new writer.** codex_v4 improved the
   script but broke reliability; media_archive_v3 regressed; fable2_v3 is broken. When you build
   the remaining v4s (shakespeare/public_domain/media_archive/original_radio), grade each against
   its predecessor on all THREE lanes, not just blind story score -- codex_v4 would look like a
   clean win on lane 1 alone and it is not.
2. **The proof-pressure pack is the transferable win.** codex_v4's want/gating-proof/cost-beat/
   reversal structure is what lifted its story; port that structure into the other v4 lanes.
3. **But loosen the codex string/all-caps contracts (or make them repairable) before pairing codex
   with a verbose cloud writer** -- otherwise every strong codex_v4 script is one emphatic word or
   one long quote away from a hard fail. This is the single change that would make the blind winner
   also shippable.
4. **Retire or rebuild the `legacy_many_pass_v3` lanes for cloud writers** -- shakespeare_v3 /
   public_domain_story_v3 / media_archive_v3 all leak placeholder names and scramble speaker tags
   under Sonnet; they are the bottom of the ranking for a shared, fixable seam reason.
5. **Practical adoption, unchanged:** free local Mistral-Nemo stays the default; if you opt into
   Sonnet, ship it on `scifi_fable2` today, and on `scifi_codex_v4` only after its contracts are
   made repairable.
