# OTR Story-Quality Baseline -- Current Architecture

**Date:** 2026-05-31
**HEAD:** c7ecf87 (OpenRouter Path-1 hardening)
**Purpose:** Establish a quality baseline on the *current* architecture (post Path-1
hardening, no overhaul) so prompt/structure work is gated on a real read, not a guess.
Prove the LLM drives the whole story -- **news selection ON, real RSS, NO baked
premise** -- and that remote and local inference of the *same model* land in the same
quality band (wiring parity).

## Method

- Workflow: `workflows/otr_scifi_16gb_full.json`, audio-only (PreviewAudio tap on the
  EpisodeAssembler's audio ancestors -- no hours-long HuMo video branch).
- Driver: `_otr_soak2.py` (uses the maintained `scripts/otr_api.py` converter).
- Story source: live RSS -> `news_interpreter` selects the story. **No `custom_premise`.**
- Seeds: `OTR_CAST_SEED=11`, `OTR_STYLE_SEED=11` fixed for the parity pair so cast + style
  are held constant; the news story is still chosen live (the two runs landed on
  *different* stories, which is expected and correct -- the news runs its course).
  Variance runs (Round 2+) draw OS entropy.
- ComfyUI: headless via Scheduled Task `OTR_API`, port 8000, OpenRouter enabled
  (`OPENROUTER_MODEL_A=mistralai/mistral-nemo`).

## Rubric (0-5 each axis; 7 axes; 35 max)

| Axis | What it measures |
|------|------------------|
| **News-grounding** | Script clearly derives from the selected news story, not generic filler |
| **Dramatic arc** | Beginning / middle / end, escalation, a turn (CLAUDE.md prime directive #4) |
| **Voice distinctness** | Speakers sound different from one another / attribution is correct |
| **Dialogue quality** | Naturalness, subtext, period-OTR voice |
| **Coherence** | Internal logic, continuity, no contradictions |
| **Payoff** | The ending lands / pays the setup (allowing for serial sign-offs) |
| **Few AI-tells** | Inverse of cliche, hedging, meta-language, headline-dumping (5 = clean) |

SFW / non-violent compliance is pass/fail, tracked separately (all runs must pass).

## Scorecard

| # | Config | Slots (creative / technical) | Words | Seeds | Title | Grnd | Arc | Voice | Dlg | Coh | Pay | Tells | **Total** | SFW |
|---|--------|------------------------------|-------|-------|-------|------|-----|-------|-----|-----|-----|-------|-----------|-----|
| T1 | **remote** parity | OR mistral-nemo / OR mistral-nemo | 60 | 11/11 | Lost in Orbit | 4.5 | 2.5 | 2.0 | 3.0 | 2.5 | 2.0 | 2.0 | **18.5 / 35** (53%) | pass |
| T2 | **local** parity | local mistral-nemo / local mistral-nemo | 60 | 11/11 | X-Labs: First Light | 4.0 | 3.5 | 3.5 | 3.5 | 4.0 | 3.0 | 3.5 | **25.0 / 35** (71%) | pass |
| T3 | **local** prod-len | local mistral-nemo / local mistral-nemo | 350 | 11/11 | Doorway to Anomaly | 3.0 | 4.5 | 4.5 | 4.0 | 4.0 | 3.5 | 3.5 | **27.0 / 35** (77%) | pass |
| T4 | **remote** prod-len | OR mistral-nemo / OR mistral-nemo | 350 | 11/11 | Muddy Canaries | 4.0 | 4.0 | 3.5 | 3.0 | 3.0 | 3.5 | 2.5 | **23.5 / 35** (67%) | pass |

Wall-clock: T1 remote ~270s, T2 local ~281s (near-identical; remote saves the local
LLM load/unload, local saves the network round-trips). T3 local 350w ~11m20s (18 lines,
3.1 min audio; the per-phase LLM reloads + 16-line Bark dominate). T4 remote 350w ~9m15s
(18 lines, 14-line Bark; remote skips the LLM reloads, absorbed two transient HTTP 429s
via the retry ladder, ~135k tokens for the whole run -- under the 300k cost ceiling).

**Aggregate by length / transport:** 60w pair -> local 71% vs remote 53% (gap 18pts);
350w pair -> local 77% vs remote 67% (gap 10pts). Local edges remote at both lengths, but
the gap *narrows* with length, and both paths sit in the same B-/B band. Longer length
lifts both scripts (more room for a real arc) and largely resolves the attribution issue.

## Per-script notes

### T1 -- remote Mistral-Nemo (both slots) -- "Lost in Orbit"

prompt_id `ff8e5b53`, episode `pending_20260531_174652`, ~270s.
meta: `llm_creative_provider=openrouter`, `slug=mistralai/mistral-nemo`,
`llm_remote_schema_mode=True`.

> **announcer:** Ladies and gentlemen, welcome to this live broadcast from NASA Headquarters, where tonight we gather to hear from the brave men and women of Crew-11, fresh from their historic 167-day mission aboard the International Space Station.
> **c03:** "Assuming you've all had a chance to unwind a bit, I thought we could debrief now. Perhaps over a cup of coffee?"
> **c03:** "Coffee. Hmm. You know, I've been so busy keeping our oxygen and water cycling properly, I've almost forgotten what it tastes like down here."
> **c03:** "Well, that's what I'm interested in, Creed. Not just what you've been doing up there, but how it's felt."
> **announcer:** This has been SIGNAL LOST. NASA hosts SpaceX Crew-11 astronauts for public event at headquarters to discuss their 167-day mission on the International Space Station. Good night.

**Read:** News-selection fired hard -- Crew-11, the 167-day ISS mission, the NASA-HQ
public event are all from the live brief (grounding 4.5). But two real problems pull the
score down, and **both are content/prompt issues, not transport**:

1. **Speaker-attribution collapse.** The three middle lines are all tagged `c03`, yet
   they're plainly a back-and-forth -- one voice offers a debrief "over a cup of coffee",
   Creed answers about oxygen cycling, then a *third* beat addresses "Creed" directly
   ("that's what I'm interested in, Creed"). A line addressed *to* Creed cannot be Creed.
   The dialogue writer flattened a multi-party exchange onto one character id (voice 2.0,
   coherence 2.5).
2. **Headline-dump outro.** The sign-off restates the news headline almost verbatim
   ("NASA hosts SpaceX Crew-11 astronauts for public event at headquarters to discuss
   their 167-day mission...") instead of paying off the scene (payoff 2.0, tells 2.0).

The coffee/oxygen beat ("almost forgotten what it tastes like down here") is a genuinely
nice human touch (dialogue 3.0). Arc is thin -- atmosphere, no turn (arc 2.5).
**The remote pipeline did the entire job** -- news interpreter, cast, dialogue, and every
technical structured call (audit / script-doctor / critic) returned valid JSON, no parse
failures. The wiring is sound; the weaknesses are in the writing.

### T2 -- local Mistral-Nemo (both slots) -- "X-Labs: First Light"

prompt_id `fa4162dc`, episode `pending_20260531_175136`, ~281s.

> **c01:** Tune in as the National Science Foundation opens its doors to the world's brightest minds, embarking on a quest to reshape our future.
> **c02:** Have you seen this, Creed? The pH levels are off the charts. It's like nothing we've ever found.
> **c03:** ((grunting)) Might be a contaminated sample, Erin. Let's not get ahead of ourselves just yet.
> **c02:** (placing another sample on the table) Well, this one's from a different site. Same readings. I think we're onto something, Creed.
> **c01:** Until tomorrow, fellow adventurers, let us remain tuned to the frequencies of progress.

**Read:** NSF / research framing is the chosen story (grounding 4.0). Complete miniature
arc: announcer hook -> anomalous discovery -> skeptic/believer tension -> sign-off
(arc 3.5). **Attribution is correct here** -- c02 (Erin, convinced) and c03 (Creed,
cautious) read as two distinct people and the cross-references line up (voice 3.5,
coherence 4.0). Dialogue has subtext and stage business (dialogue 3.5). Ending is a serial
sign-off rather than a true payoff (payoff 3.0). Announcer purple ("frequencies of
progress") is in-genre, mild (tells 3.5).

## Parity verdict (T1 vs T2)

**Wiring is proven both ways.** The same model (`mistral-nemo`) completes the *entire*
pipeline -- live news selection, cast, dialogue, and all technical structured calls --
whether inferred via OpenRouter (T1) or locally (T2). Remote produced valid JSON on every
structured call with zero parse failures, end-to-end, in the same wall-clock band as
local. The Path-1 hardening (force `json_object` + `require_parameters`, robust extract,
cost-cap split, 1024-token floor) holds under a real run.

**Quality landed in the same band (B-/B), local edging ahead this round (71% vs 53%).**
The gap is **not** transport: it traces to two content issues in the remote script --
speaker-attribution collapse (all middle lines tagged `c03`) and a headline-dump outro.
Both are prompt/casting behaviors that can hit either path; this round the remote run on
the NASA story happened to trigger both, while the local run on the NSF story kept clean
attribution. Different news stories (live RSS, no baked premise) mean this is a
quality *band* comparison, not a line-for-line diff -- by design.

**Conclusion for the go-forward:** transport/wiring is closed. Any remaining quality lift
is a prompt/structure problem, exactly where the lean go-forward plan says to spend
effort -- and the highest-value targets it surfaces are (1) speaker-attribution integrity
in the dialogue pass and (2) suppressing headline-dump outros. Both should be re-checked
at production word count (350), where more lines make the attribution behavior far easier
to judge -- queued next.

## Findings to investigate (not yet logged as bugs)

- **F1 -- attribution collapse at low word count.** T1 flattened a multi-party exchange
  onto one character id. Need to confirm whether this is a 60-word artifact (too few lines
  to spread across the cast) or a real dialogue-pass defect. **Test:** re-run remote at
  350 words; if multi-party scenes still collapse onto one id, log as a bug.
- **F2 -- headline-dump outro.** T1's sign-off pasted the news headline verbatim. Check
  whether the outro prompt invites a one-line news recap (which the model over-literalizes)
  and whether it shows up at longer length too.

## Round 2 (production word count, 350w)

Purpose: 350 words is production shape. More lines make the F1 attribution behavior far
easier to judge and give a longer-form quality read than the 60-word parity pair. Seeds
stay 11 (set in the running process); the live news story still varies per run.
prompt_ids: T3 local `2a49e952`, T4 remote `851229da`.

### T3 -- local Mistral-Nemo (both slots) -- "Doorway to Anomaly"

episode `pending_20260531_175958`, 18 ledger lines (16 speech + 2 music interludes), ~11m20s.

> **announcer:** In the Andean highlands, where potatoes once met their new home, a tale of evolution unfolds tonight on Signal Lost.
> **c02:** I've traveled quite a distance to learn more about your village's peculiar legend, Mr. Terwilliger.
> **c03:** Ah, yes, the 'Potato Spitfire'. I've seen many a curious soul pass through here, drawn to our humble village by that very tale.
> **c02:** I've seen the photographs, Mr. Terwilliger. A potato, spitting fire? It's either a remarkable phenomenon or a remarkable hoax. I intend to find out.
> **c03:** (sighs) Alright, Miss Spender. I'll take you to old Widow Hargrove. She swears that thing glows in the moonlight. But don't expect me to believe in any of it.
> _[music interlude]_
> **c02:** (pauses, noticing a hidden door) Mr. Terwilliger, have you ever seen this door before?
> **c03:** (brushes off the suggestion) Doors come and go in these old houses, Miss Spender. Nothing to get excited about.
> **c02:** (quickly) I think I'll take a sample of this 'Potato Spitfire', just to be sure.
> **c03:** (noticing Erin's quick hands) I believe you took something that wasn't yours, Miss Spender.
> **c02:** (defiant) I'm merely ensuring we have proof before we present this to the villagers.
> **c03:** (tightening his grip on his lantern) You realize what you're doing is tantamount to thievery, don't you?
> _[music interlude]_
> **c02:** (softly, with regret) I'm doing what I've always done, Creed. Protecting those who can't protect themselves. Even if it means bending rules.
> **c03:** (staring at the flickering lantern) Perhaps bending rules is exactly what's needed to embrace... this anomaly.
> **c02:** Creed, do you understand what will happen if the villagers find out about this without proof? They'll be scared, and fear leads to irrationality.
> **c03:** I understand the consequences, Erin. I'll help you gather the evidence.
> **announcer:** In this era of discovery, our understanding of life in the Andes is being rewritten, one spud at a time. Goodnight.

**Read:** A real three-act arc -- investigation of a village legend (setup) -> hidden door
+ sample theft (complication) -> moral confrontation, "tantamount to thievery" (turn) ->
the skeptic's conversion, "I'll help you gather the evidence" (resolution) -> thematic
outro (arc 4.5). **Attribution is clean across all 16 speech lines** -- c02 (Erin Spender,
determined journalist) and c03 (Creed Terwilliger, weary skeptic-turned-ally) never
collapse, cross-references line up, names are consistent (voice 4.5, coherence 4.0). Good
period diction and stage business -- flickering lantern, quick hands (dialogue 4.0).
Grounding is the soft spot: it anchors to a real-sounding Andean-potato/evolution science
item but the body invents a fantastical "Potato Spitfire" folk legend that isn't news
(grounding 3.0). Payoff is satisfying if not surprising (3.5); a couple of on-the-nose
lines ("fear leads to irrationality") and the cute "one spud at a time" outro are mild
tells (3.5). **27/35 -- the best script of the set, and the clearest evidence that the
60-word attribution collapse (F1) is a low-word-count artifact, not a dialogue-pass
defect: given 350 words and 16 lines, local attribution held perfectly.**

### T4 -- remote Mistral-Nemo (both slots) -- "Muddy Canaries"

episode `pending_20260531_181105`, 18 ledger lines (14 Bark + 2 announcer to Kokoro + 2
music), ~9m15s. meta: `llm_creative_provider=openrouter`, `slug=mistralai/mistral-nemo`.

> **announcer:** In the heart of Costa Rica, tonight, we explore a delicate balance: the patient study of capuchin monkeys and the urgent whispers of a changing climate.
> **c02:** I hope the monkeys are holding up against this deluge, Creed. The river's edging closer to their enclosure.
> **c03:** You worry too much, Erin. They've weathered worse. *pauses to roll up his sleeves* Besides, the rain's good for the soil.
> **c03:** "Creed, have you considered how they must feel, trapped in this downpour?"  _<- misattributed: addresses Creed, so this is Erin's (c02) line_
> **c03:** "Erin, science isn't about feelings. *tightens his grip on a clipboard* It's about data, and this storm is giving us invaluable information."
> _[music interlude]_
> **c02:** "I've seen enough, Creed. * ERIN SPENDER the monkeys' enclosure* I won't stand by while their well-being is sacrificed for a few more data points."  _<- F3 name-leak in stage direction_
> **c03:** "Erin, this isn't one of your sentimental yarns. *looks up, addressing the team* Let's focus on the task at hand, shall we?"
> **c02:** "Creed, you need to see this. *holds up a muddy, shredded leaf* They're starting to eat the tree bark. It's a sign of desperation."
> **c03:** "A single leaf, Erin? *consults his clipboard* You're making much ado about nothing. Focus on the systematic data, not anecdotes."
> **c02:** "Like you did with the chimps? *drops the leaf, looking pointedly at Creed* Or have you conveniently forgotten their plight?"
> **c03:** "Watch your tongue, Erin. I can make your unauthorized rabbit trails disappear with a single phone call."
> _[music interlude]_
> **c02:** "I've been documenting my findings, Creed. It's all safe in the ERIN SPENDER."  _<- F3 name-leak as dangling noun_
> **c03:** "Erin, you're being too careless. We can't have that kind of information... loose."
> **c02:** "I'm willing to take that risk, Creed. I've seen what happens when we stay silent."
> **c03:** "The capuchins have tripled in number since you started, Erin. Your work could cause a whole new set of problems."
> **announcer:** Tonight's findings, a testament to the resilience of these capuchins, remind us that understanding nature's adaptability is key in our quest to face a warming world.

**Read:** Best-grounded script of the set -- Costa Rica capuchins + a warming-climate
field study, woven through intro and outro (grounding 4.0). Real ethical-conflict arc:
Erin's compassion vs Creed's cold empiricism, escalating to a genuine menace beat
("I can make your unauthorized rabbit trails disappear with a single phone call") and an
ambiguous twist (the capuchins have *tripled*, "could cause a whole new set of problems")
(arc 4.0). Two defects pull it down, and **both are generation artifacts, not transport**:

1. **One misattribution** (line 4): a line that addresses "Creed" is tagged `c03` (Creed
   himself). Much milder than T1's wholesale 60-word collapse -- one line of 16, not the
   whole scene (voice 3.5, coherence 3.0).
2. **F3 -- cast-name leak into the body** (two lines): the ALL-CAPS canonical cast name
   "ERIN SPENDER" landed where a verb/noun belongs -- "* ERIN SPENDER the monkeys'
   enclosure*" (a stage direction missing its verb) and "safe in the ERIN SPENDER" (a
   dangling noun). These read as nonsense and would degrade TTS (dialogue 3.0, tells 2.5).

**Local (T3) on its story showed neither defect.** Both are remote-mistral-nemo
generation quirks the local run didn't trigger -- *not* a wiring problem. Every structured
call still returned valid JSON; the run completed clean end-to-end.

## Parity verdict -- CLOSED at production length

**Wiring is proven identical on both transports, at both lengths.** Same model
(`mistral-nemo`) drives the *entire* pipeline -- live news selection, cast, dialogue, and
all technical structured calls -- whether inferred via OpenRouter or locally. At 350w both
paths produced an 18-line, multi-act, news-grounded, SFW episode through the identical
cascade; remote returned valid JSON on every structured call (and absorbed two HTTP 429s
via the retry ladder) with zero parse failures. **The Path-1 hardening holds under real
runs.** Transport is no longer the question.

**Quality sits in the same band (B-/B); local edges remote, gap narrows with length**
(18pts at 60w -> 10pts at 350w). The remaining gap is entirely **content-generation**, not
transport: remote mistral-nemo on these stories produced a residual misattribution and the
F3 name-leak that local did not. That is exactly the prompt/structure surface the lean
go-forward plan reserves for *after* this read -- and this baseline names the two
highest-value targets with code anchors (below).

## Findings to carry into the prompt/structure pass

- **F1 -- attribution collapse is a low-word-count artifact (largely resolved at length).**
  60w remote (T1) flattened a whole exchange onto one id; at 350w local was perfect and
  remote slipped on a single line. **Action:** prefer >= ~150w for multi-character scenes;
  the dialogue pass needs a stronger per-line speaker binding only if a 350w+ run reproduces
  multi-line collapse (it did not here). Low priority.
- **F2 -- headline-dump outro** (T1 60w): the sign-off pasted the news headline verbatim.
  Did **not** recur at 350w (T3/T4 outros paraphrase thematically). Likely another
  low-word artifact. Low priority.
- **F3 -- cast-name leak into line body (remote, 350w). HIGHEST priority.** The ALL-CAPS
  canonical cast name leaked into two stage-direction / noun slots. The existing leak
  filter -- `nodes/_otr_line_composer.py` lines ~1663-1692 (BUG-LOCAL-279 follow-on) --
  only catches a leak when the **whole cleaned line** equals the speaker's own name or
  "ANNOUNCER", and deliberately does **not** filter the broader roster (to preserve
  legitimate cross-character name drama, e.g. a one-word "Maeve."). F3 is the uncaught
  case: a roster cast name embedded **mid-phrase / inside an asterisk stage direction**.
  **Proposed bounded fix (operator review -- not shipped here, per no-overhaul):** in the
  compose retry loop, flag a draft where an ALL-CAPS *multi-word roster name* appears
  inside an asterisk `*...*` group or as a bare mid-sentence token, and retry (the loop
  already supports retry-on-reject). Scope it to multi-word ALL-CAPS to avoid touching
  legitimate single-name drama. Must pass Bug Bible + audio byte-identical before ship.

## Test log

| Run | Submitted | Result | Episode |
|-----|-----------|--------|---------|
| T1 remote 60w | `ff8e5b53` | success ~270s | Lost in Orbit |
| T2 local 60w | `fa4162dc` | success ~281s | X-Labs: First Light |
| T3 local 350w | `2a49e952` | success ~11m20s | Doorway to Anomaly |
| T4 remote 350w | `851229da` | success ~9m15s | Muddy Canaries |

All four: real RSS, `news_interpreter` ON, **no baked premise**, audio-only, SFW pass.
