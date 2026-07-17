# OTR Source-Bank Variant Scoreboard (v1/v2/v3 x 420/720)

**Date:** 2026-07-17. **Writer model (held constant):** aion-3.0-mini creative +
Mistral-Nemo technical. **Method:** story-only harness (writer -> freeze ledger,
no media). Grading: Fable narrative pass over the frozen transcripts -- the same
signal a blind read uses; video carries no cross-bank signal. Scores are OVERALL
1-10, no curve.

Differences here are BANK/VERSION effects only (one writer model). A separate
**aion-vs-Sonnet** column is rendering to test whether the writer model is the
quality ceiling (see bottom).

## Coverage

42 of 48 cells graded. The 6 gaps are lawful/known content-fails, DISQUALIFIED
pending canonical root-fixes (task 7), NOT graded:
`scifi_codex` v2_420 / v2_720 / v3_720 (P3 unstated-contract bug),
`scifi_fable2` v3_720 (SCENE_WORD_GROSS scene-length gate),
`scifi_sonnet` v2_420 (reroll runaway), `original_radio` v2_720 (weapons SFW gate).

## TOP-8 KEEPERS (best version per bank)

| Bank | KEEP | Overall | Why |
|---|---|---:|---|
| scifi_fable2 | **v1** | 8.5 | Flagship. Best bank+version in the matrix; strong at BOTH rungs; zero tag defects. |
| public_domain_story | **v3** | 7.0 | Only version clean at both rungs; the Wells source gives it a real spine. |
| original_radio | **v1** | 6.0 | Best atmosphere-per-word; steadiest (its v2 is the matrix's worst file). |
| scifi_codex | **v1** | 5.75 | Only rung-honest 720 anywhere; clean + SFW -- but inert (a debate, not a drama). |
| media_archive | **v1** | 5.75 | Carried by v1_720 ("The Card Marked Destroyed"); rest is plot mush. |
| shakespeare | **v3** | 5.75 | Best prose of the non-fable2 banks, but only v3 survives 720 without broken speaker tags. |
| scifi_sonnet | **v3** | 5.5 | Tightest execution of a weak format (an authentication liturgy, not a drama). |
| science_news | **v3** | 4.5 | Weakest bank; v3 is merely mediocre where v1/v2 are broken. Weak keep. |

## Overall bank ranking

1. **scifi_fable2** -- the only bank with real dramaturgy (specificity, reversals, endings that cost something).
2. **public_domain_story** -- Wells spine; v3 a real play at both rungs.
3. **original_radio** -- best atmosphere, worst variance (its floor is the matrix floor).
4. **scifi_codex** -- clean, complete, dull; 3 missing cells is itself a mark against it.
5. **shakespeare** -- strong prose repeatedly killed by broken speaker tags at 720.
6. **media_archive** -- one good episode, a bank-wide truncated outro.
7. **scifi_sonnet** -- a format, not a story; no stakes, broken length control both ways.
8. **science_news** -- attribution/monologue collapse in 4 of 6 files; nothing shippable as-is.

Best at 420: fable2 (9) > public_domain_story (7) ~ shakespeare-v2 (7) > original_radio (6.5) ~ codex (6.5) > sonnet (6) > science_news (5) ~ media_archive (5).
Best at 720: fable2 (8) > public_domain_story (7) > media_archive (6.5) > shakespeare-v3 (5.5) ~ original_radio (5.5) > codex (5) ~ sonnet (5) > science_news (4).

## The real headline: pipeline bugs > bank problems

Most of what drags scores down is NOT the bank's writing -- it's **systemic
pipeline defects** hitting many banks at once. Fixing these lifts EVERY bank and
matters more than per-bank tuning. These are code/prompt bugs the writer model
cannot fix (they'll persist under Sonnet too):

1. **Speaker-attribution collapse** (dominant ship-blocker) -- one speaker absorbs
   the other's lines, or roles swap wholesale. Hits 5 banks; **5 of 7 cases are v2
   cells** -- the v2 pipeline has an attribution bias. Investigate before any
   per-bank fix.
2. **Speaker-name token splices into dialogue** -- "what EDNA FROST've done",
   "MINA HUDSONr authorization's", "I demand to know, THE TIME TRAVELER". Looks like
   a name-normalization / anti-ventriloquism pass doing a blind string-replace on
   "you/your". One root cause, 3 banks.
3. **Literal placeholder tokens** -- `'X'` / `'Y'` as character references
   (original_radio v2_420). Hard ship-stop.
4. **Phantom outro characters** -- announcer closes on people not in the play
   (Ruth/Elijah; Silas + a lighthouse). The outro pass isn't reading the final cast.
5. **Truncated template outro** -- media_archive: 5 of 6 files end mid-sentence
   ("...preserve and restore a lost.").
6. **Contract-vocabulary bleed** -- negotiation-engine words (authorization,
   signature, deadline, "the record," pension allocation, loan covenant) surface as
   literal dialogue in every bank regardless of fit. The beat contract is leaking
   into surface text pipeline-wide.
7. **Rung-length knob barely steers** -- at the 720 rung, only codex-v1, fable2, and
   sonnet-v1 exceed 700 words; every other 720 lands ~370-550w, indistinguishable
   from its 420 cell. sonnet is inverted (420->~700w, 720->~215w). Length is
   recorded-not-gated, but the 420-vs-720 comparison is confounded because the knob
   isn't moving most banks.
8. **Stage-direction / audit-machinery leaks** -- parentheticals read aloud by TTS;
   sonnet's grounding pass performing itself on air ("I call for a re-check").
9. **Header/script scene mismatch** -- SETTING header contradicts the announcer's
   scene in ~6 files.

## Length effect (does 720 make a better story?)

Only **public_domain_story v3** clearly becomes a fuller play at 720. **fable2**
uses 720 but changes mode (its 420 duels are the stronger form). Everyone else:
720 is neutral-to-worse (original_radio, shakespeare, science_news get worse; codex
gets looser; media_archive/sonnet don't lengthen). The rung knob (defect 7) is why.

## Pending

- **aion-vs-Sonnet column** -- 8 base banks @420 on `claude-3.5-sonnet` rendering
  now; tests whether the writer model is the quality ceiling (it will lift craft,
  NOT the pipeline bugs above). Fill this section when it lands + cost.
- **Parked canonical root-fixes (task 7)** -- codex P3, fable2 scene-gate,
  original_radio weapons; plus the shared-fix candidates above (attribution
  collapse, name-splice, contract-vocab bleed, outro/cast pass, rung-length knob).
- The 6 DISQUALIFIED cells re-render after their root-fix, then fold into the grid.
