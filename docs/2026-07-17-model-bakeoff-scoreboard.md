# Sonnet-4.5 Cross-Bank Bake-off -- Scoreboard & Verdict

**Date:** 2026-07-18 (campaign 2026-07-17 night -> 2026-07-18 morning). **Type:** RENDER-window
campaign, executing GO_FORWARD "Then, in order" item 3 (the creative-writer model question).
**Baseline HEAD:** `60c73618`, branch `v2.0-alpha`. **Status:** COMPLETE.

## Scope (operator-directed)

Run **every runnable bank** with one held-constant pair -- creative =
**`anthropic/claude-sonnet-4.5`** (OpenRouter remote, slot A) / technical =
**`mistralai/Mistral-Nemo-Instruct-2407`** (local) -- at **420w and 720w**, grade the transcripts
blind, and name the new winner bank. Bank is the only variable. This is a **model swap** of the prior
720 bake-off (whose pinned creative was `aion-labs/aion-3.0-mini`, same local Mistral technical), so the
two are directly comparable. Harness: story-only (`otr_story_only.json`), fresh source per leg,
selective reset per leg, `frozen`* = green. Length recorded, never gated; content-fails recorded with
reason, never re-rolled.

## Result matrix -- RUN STATUS (11 banks x 2 tiers = 22 legs; 18 SUCCESS / 4 FAIL)

| Bank | 420w | 720w |
|---|---|---|
| media_archive | SUCCESS (404w, warns) | SUCCESS (406w, doctor-edits) |
| original_radio | **FAIL** -- deterministic `original_qa` news_source_framing gate (Sonnet framed it as courtroom testimony) | SUCCESS (464w, doctor-edits) |
| scifi_fable2 | SUCCESS (521w / 47 lines, warns) | SUCCESS (787w / 13 lines, warns) |
| scifi_codex | SUCCESS (391w, warns) | SUCCESS (391w, clean) |
| media_archive_v3 | SUCCESS (359w, warns) | SUCCESS (431w, doctor-edits) |
| public_domain_story_v3 | SUCCESS (334w, doctor-edits) | SUCCESS (429w, doctor-edits) |
| shakespeare_v3 | SUCCESS (320w, doctor-edits) | SUCCESS (258w, doctor-edits) |
| scifi_fable2_v3 | **FAIL** -- NEWBUG: fable2 `revision_contract` hardcodes `rules_id=='scifi_fable2'`, rejects the v3 id (model-independent) | **FAIL** -- same NEWBUG |
| scifi_codex_v3 | SUCCESS (607w, clean) | SUCCESS (352w, warns) |
| scifi_sonnet_v3 | SUCCESS (858w / 11 lines, warns) | SUCCESS (840w / 13 lines, warns) |
| scifi_codex_v4 | **FAIL** -- deterministic codex P5 gate: `l003 spoken text contains an all-caps lexical word` (2 repair attempts) | SUCCESS (597w, clean) |

No cell is DISQUALIFIED or NOT-RUN; all 22 ran. The three FAIL classes: one real bug
(`scifi_fable2_v3`, both tiers -- see `docs/2026-07-18-NEWBUG-fable2-v3-rules-id.md`) and two
deterministic-gate content-fails that Sonnet tripped at 420 but cleared at 720 (`original_radio`
news-framing, `scifi_codex_v4` all-caps) -- length/draft-dependent, recorded not re-rolled.

## Blind quality grade -- 720w rung (the decisive rung), Fable pass, de-anonymized

Ten 720-SUCCESS transcripts graded blind (labels hidden) on STORY / CHARACTER / CRAFT / IDIOM_FIT /
SFW, 1-5 each (/25):

| Rank | Bank | /25 | Note |
|---:|---|---:|---|
| 1 | **scifi_codex_v4** | 24 | "The Halicin Gamble" -- complete 3-hander; the AI-buried-the-tox-file reversal is earned and lands. |
| 2 | scifi_fable2 | 24 | Best line-level prose in the set; monologue form caps the character axis. |
| 3 | scifi_codex | 21 | Three opposed institutional wants; sharp turn -- nobody in the room gets picked. |
| 4 | scifi_codex_v3 | 21 | Moving dilemma, three clear voices; ends on deliberate indecision (no turn). |
| 5 | media_archive | 20 | Real standoff + 20-year-secret confession; one beat repeats 3x. |
| 6 | original_radio | 18 | Genuine gothic reveal; scrambled speaker tags + name drift at the end. |
| 7 | media_archive_v3 | 14 | Good core conflict but self-contradicts (print stock vs nitrate); garbled tag. |
| 8 | public_domain_story_v3 | 13 | Placeholder names spoken as dialogue ("Explain this, THE TIME TRAVELER"); broken final line. |
| 9 | shakespeare_v3 | 13 | Attribution collapse -- a character speaks the bait she's meant to overhear; truncated fragment. |
| 10 | scifi_sonnet_v3 | 12 | An essay in three voices; literally states "the stakes remain hermeneutic, not practical." |

## Verdict

**New winner under Sonnet-4.5: `scifi_codex_v4`**, edging `scifi_fable2` on the same top score by being
a full dialogic play (reversal + consequence) where fable2 -- superb on the page -- came out as a long
monologue (787w across only 13 lines). The **codex circuit sweeps the podium** (v4 #1, base #3, v3 #4):
its want/gating-proof/mandatory-cost-beat/reversal structure is the best match for Sonnet's strengths.

**This shifts the crown.** Under the prior aion-3.0-mini writer, `scifi_fable2` won (it converted to an
ensemble only at 720). Under Sonnet-4.5, `scifi_codex_v4` wins and fable2 goes monologue-ward -- so the
answer to "best bank" genuinely depends on the writer model.

**Weak on Sonnet:** `scifi_sonnet_v3` (essayistic, airless -- the clear cut), and the `legacy_many_pass_v3`
lanes (`shakespeare_v3`, `public_domain_story_v3`, `media_archive_v3`) which suffered speaker-tag
scrambling / placeholder-name leakage that the codex/fable2 lanes did not. `scifi_fable2_v3` is unjudgeable
(the rules_id NEWBUG).

**Caveats on the winner:** `scifi_codex_v4` FAILED at 420w (all-caps gate) -- it is 720-robust but
420-fragile with Sonnet's emphasis-caps habit. `scifi_fable2` is the **most robust** strong bank (clean
SUCCESS both tiers) and the runner-up on quality.

## Cost

Creative-slot (Sonnet-4.5) tokens across all 22 legs: **~3.07M** (technical Mistral is local/free; 0 GPU
VRAM for the creative slot). Priced at Claude Sonnet-4.5 rates (~$3/M in, ~$15/M out; accounting does not
split in/out, so assuming ~80/20 -> ~$5.4/M blended): **~$16, order-of-magnitude $15-20.** Per-leg the
heavy legacy/codex lanes ran 200-290k tokens; the light lanes 20-100k.

## Recommendation

Sonnet-4.5 clearly raises the craft ceiling (the codex tragedies and fable2 prose are the strongest
scripts this project has graded), but keep it **opt-in**, not a default swap: it costs money, sends prompts
off-machine, trips two lane gates the local default never hits (codex all-caps, original_radio framing),
and degrades the v3 legacy lanes. If a user opts into a cloud creative, pair it with **`scifi_codex_v4` at
720w** for the best result, or **`scifi_fable2`** for the most robust one. The free local Mistral-Nemo
remains the correct default. Fix `scifi_fable2_v3` (the NEWBUG) so its lane is judgeable; consider whether
the codex all-caps gate is too strict for an emphasis-prone writer.

## Full-media confirmation

One canonical (full-pipeline) leg on the winner `scifi_codex_v4` @ 720w with this config, to prove the
remote-creative winner survives TTS + video + obs_publish. Result recorded below.

**PENDING** -- the confirmation leg is rendering; its RESULT / obs_publish / asset path is appended here
on completion (follow-up commit).
