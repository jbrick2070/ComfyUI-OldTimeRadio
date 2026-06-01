# OTR Story-Quality Comparison -- All Scored Runs (incl. Opus)

**Date:** 2026-05-31
**Rubric:** 7 axes x 0-5 = 35 (same as `docs/2026-05-31-otr-story-quality-baseline.md`):
news-grounding, dramatic arc, voice distinctness, dialogue quality, coherence, payoff,
few-AI-tells. All runs: real RSS + `news_interpreter` ON, **no baked premise**, fixed
cast/style seeds (11/11). Scripts graded from the writer/cascade ledger (pre-audio).

## Consolidated scorecard (sorted by score % desc)

| Run | Date | Model (creative / technical) | Transport | Target w | Actual w | Score | % | Notes |
|-----|------|------------------------------|-----------|----------|----------|-------|---|-------|
| **Opus** "The Green Book of Nights" | 05-31 | **claude-opus-4.8:nitro** / local mistral-nemo | remote+local | 350 | **829** | **31.0/35** | **89%** | 30 creative calls -> Opus (0 fallback); 5 technical -> local; ~43k tok ~$0.47; audio skipped per operator |
| T3 "Doorway to Anomaly" | 05-31 | local mistral-nemo / local | local | 350 | 326 | 27.0/35 | 77% | best local; clean attribution |
| T6 "Dancing Plague" | 05-31 | local mistral-nemo / local | local | 350 | ~330 | 25.5/35 | 73% | local consistency run |
| T2 "X-Labs: First Light" | 05-31 | local mistral-nemo / local | local | 60 | ~70 | 25.0/35 | 71% | short; arc cramped |
| T8 "Dust of Uncertainty" | 05-31 | local mistral-nemo / local | local | 500 | 334 | 24.0/35 | 69% | critic flagged needs_full_rerun; "damn" x1 |
| T4 "Muddy Canaries" | 05-31 | remote mistral-nemo / remote | remote | 350 | ~330 | 23.5/35 | 67% | F3 name-leak; 1 misattribution |
| T1 "Lost in Orbit" | 05-31 | remote mistral-nemo / remote | remote | 60 | ~70 | 18.5/35 | 53% | 60w attribution collapse |

**Failed runs (not scored -- both fail-closed/safe, no broken episode shipped):**

| Run | Model | Why | Class |
|-----|-------|-----|-------|
| T5 "(remote 350 F3 repro)" | remote mistral-nemo | crashed at Bark Gate 3 (char_id='announcer' no v2/* preset) | BUG-276/271 cast-routing family (known-open) |
| T7 "(local 150)" | local mistral-nemo | writer "inventor" pass parse-failed 3x | F4 (news/inventor key-term validation, BUG-264 family) |

All 7 scored runs are on the **same rubric + same critic** -- directly comparable. No
incompatible-rubric historical runs were found in `docs/` or `BUG_LOG.md` (prior sessions
logged build/bugfix work, not rubric-scored story reads), so there is no separate
non-comparable table to add.

## Per-axis breakdown -- Opus vs best local Mistral (T3, 77%)

| Axis | Opus | T3 local | Delta | Read |
|------|------|----------|-------|------|
| News-grounding | 5.0 | 3.0 | **+2.0** | Opus dramatizes the actual news (grant cancellations -> records-destruction resistance); Mistral drifted to a "potato that spits fire" tangent |
| Dramatic arc | 4.5 | 4.5 | 0 | both land a real 3-act turn |
| Voice distinctness | 4.5 | 4.5 | 0 | both keep clean, distinct attribution |
| Dialogue quality | 4.5 | 4.0 | +0.5 | Opus is denser/subtextual; Mistral cleaner/plainer |
| Coherence | 4.0 | 4.0 | 0 | both internally consistent |
| Payoff | 4.5 | 3.5 | **+1.0** | Opus's complicity reveal + "three drawers in three houses" lands harder than Mistral's sign-off |
| Few AI-tells | 4.0 | 3.5 | +0.5 | Opus avoids cliche/headline-dump; mild deduction for ornate density |
| **Total** | **31.0** | **27.0** | **+4.0** | Opus wins decisively on grounding + payoff, ties on arc/voice/coherence |

The Opus advantage is concentrated where it matters most for an *original* drama:
**making the news meaningful** (grounding +2) and **sticking the ending** (payoff +1).
On the mechanical axes (arc, voice, coherence) the small local model already matched it.

## Routing verification (Opus run)

Parsed from the run log (`_otr_routing_audit.py`):

| Slot | Decisions | Destination | Fallbacks |
|------|-----------|-------------|-----------|
| creative | 30 | **OpenRouter -> anthropic/claude-opus-4.8, route=throughput** | **0 to local** |
| technical | 5 | **local mistral-nemo** | 0 leaked to remote |

- All 30 remote calls carried slug `anthropic/claude-opus-4.8` (the `:nitro` suffix
  resolved to `route=throughput`); **no non-Opus slug** appeared.
- C2 no-evict held: "resident local model left in place" on every creative call (the local
  technical model was never evicted to make room for the remote call).
- BUG-296 confirmed healthy live: run-total **~43,118 tokens** for the whole episode --
  one episode's spend, well under the 300k per-run ceiling (no spurious cost-abort).

## Cost

OpenRouter `anthropic/claude-opus-4.8`: **$5/M prompt, $25/M completion**. The run
accounted ~43,118 tokens (a conservative OTR-side over-estimate = prompt-chars/4 + a
1024-token output floor per call). Rough cost **~$0.47** (70/30 prompt/completion split);
actual billed is at or below that. So a full Opus creative episode is **well under $1**.

## Verdict

**Opus is materially better, not marginally.** 89% vs the local sweet-spot's 77% (+12
points), and the gain is on the axes that separate a *memorable* story from a competent
one: it turned a dry policy headline into a tense, le-Carre-style two-hander about
archivists smuggling doomed research out of a records vault, with a real reversal and a
defiant close. The cheap local model is genuinely good at the *mechanics* (arc, distinct
voices, clean attribution) -- Opus adds *meaning and craft* on top.

**One real caveat (production, not quality):** Opus overshot 350 words -> **829**, and its
prose is literary to a fault -- every line is a long, comma-spliced paragraph of
indirection. That reads beautifully on the page but is **dense for the ear**; an OTR
*listener* (vs a reader) could lose the cat-and-mouse thread. This is a length/pacing
tuning problem, not a writing-quality problem.

**One-line recommendation:** Make **OpenRouter + Opus the default creative slot** (technical
stays local, fail-closed) -- it's a clear quality win for <$1/episode -- and add a short
length/clarity tuning pass on the Opus creative prompts so the richer writing still plays
cleanly as *radio* (plan: `docs/openrouter-llm-call-improvement-plan.md`).

## The Opus script in full -- "The Green Book of Nights"

News seed: *"Proposed new US funding rules: We can cancel any grant at any time"* (Ars
Technica, Science). Cast: Erin (c02) + Creed (c03) + announcer. Setting: a research
records vault on the eve of a politically-ordered document purge.

> **ANNOUNCER:** Tonight, in the quiet halls where discovery is born, the Trump administration redraws the rules of the grant, and the scientists wonder what may no longer be asked.
> **ERIN:** Log starts now -- Project Halloway, the soil-contamination survey they shelved in March... I'm reading the reel numbers aloud, every one, because come Monday this drawer is supposed to be empty and I'd rather my voice outlast the paper.
> **CREED:** Lights on in the vault past nine, Erin -- you keep that up and the building thinks somebody's afraid of the dark... funny, that reel in your hand, I had it down on the disposal sheet myself.
> **ERIN:** Just tidying -- somebody crossed half these numbers off twice, so I'm matching the index to the cans before they get carted off... your hand on the disposal sheet, was it? I'd have sworn the ink was newer than that.
> **CREED:** Newer, older -- they don't pay me to date the ink, Erin, only to have the shelves clear by the eight o'clock truck Monday, and you've got, what, two evenings to read to yourself... I'd not waste them on a drawer that's already spoken for.
> *-- musical interlude --*
> **CREED:** They've gone and pinned my name to it now -- sign-off on every can, every shred, my initials on the dust before it's swept... so I'll be down here as much as you'll be, only I'll have the keys to the lights.
> **ERIN:** Well -- good, somebody steady ought to mind it, and you've always had the head for numbers... will you be doing the walk-throughs yourself, or has the front desk got the log of who comes and goes? I'd hate to trip a wire just reaching for a stapler.
> **CREED:** Front desk has it all, every door and the minute it opened -- and you'd be surprised, Erin, the times that turn up. Somebody was in here past one Tuesday last, no name beside it, just the lamp left warm... I touched the bulb myself.
> **ERIN:** Tuesday -- that'd be quarter-end, wouldn't it, I'm forever chasing a column that won't sit still, so I likely came back for the reconciliation and lost the hour... left a lamp on, that's me all over, my mother said I'd burn the house down round my ears.
> **ERIN:** That little red eye on the case there -- no, that's only the battery telling me it's full, I keep it for dictating my own figures, you know how I lose the thread aloud... I ought to switch it, the catch sticks, give me a moment --
> **CREED:** Leave it run, Erin -- I've heard the tape before, the one you slid behind the third drawer, and I didn't say a word to the front desk... so here's how it sits: my initials are on every can that goes out, which means whatever's on that little reel goes out with my name pinned to it now, same as yours -- so we'll keep talking, and you'll tell me what you're really down here listening for.
> *-- musical interlude --*
> **CREED:** Price is this -- you keep the catch oiled, and you keep what I couldn't burn... I lifted three cans off the trolley the night they had me feeding the rest into the incinerator, told myself I'd lose them, never did, and here's the green book that says which night is which, the warm lamps and the cold -- I'm putting it under your blotter now, see, my hand's flat on it, and once I lift it off it's yours to carry, not mine.
> **ERIN:** All right -- there, your hand's away, and mine's on it... I'll have a column for these that won't burn down round anybody's ears, the catch oiled, the reels where the front desk never thinks to look, and when somebody comes asking after a Tuesday past one, it'll be me touching the bulb and finding it cold.
> **CREED:** They'll match the initials to the missing cans soon enough -- a man with a clipboard always does -- and I'll be the one walking out past the front desk with my coat over my arm, but that's the trade I made when I left them on the trolley instead of in the fire... the green book doesn't know my face, only the nights, and the nights'll keep telling true after they've shown me the door.
> **ERIN:** It's already gone further than this blotter -- my sister carries a copy out in the laundry hamper every Thursday, folded under the linens, and there's a second reel sitting in a deposit box across town with a name on it that isn't yours or mine... so let them come with the clipboard, Creed; you can't unsay a thing once it's lying in three drawers in three different houses.
> **ANNOUNCER:** And so the laboratories wait in uncertain light, as the Trump administration's rules advance through formal rulemaking, leaving the nation's researchers to wonder which questions tomorrow will still permit them to ask.

(18 ledger lines, 829 words; em-dashes are clean UTF-8 in the ledger -- a console display
artifact only.)
