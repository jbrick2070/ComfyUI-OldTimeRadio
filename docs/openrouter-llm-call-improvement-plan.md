# OpenRouter / LLM-Call Improvement Plan (per-slot)

**Date:** 2026-05-31
**Status:** PLAN ONLY -- nothing here is implemented. Gate every change on the regression
suite + audio byte-identical + a scored A/B vs the 2026-05-31 baseline.
**Evidence base:** `docs/2026-05-31-otr-story-quality-baseline.md` (6 scored local/remote
mistral-nemo runs) + `docs/2026-05-31-otr-story-quality-comparison.md` (Opus 89% run).

## What the scored runs told us

- **Opus is the strongest writer (89%)** -- wins on news-grounding (+2) and payoff (+1) vs
  the best local mistral-nemo (77%). Recommend Opus as the **default creative slot**
  (technical stays local, fail-closed). Cost ~$0.47/episode.
- **But Opus overshoots and over-writes for radio:** 829 words on a 350 target, every line
  a long comma-spliced paragraph of indirection. Beautiful on the page, **dense for the
  ear** -- the #1 thing to tune.
- **Local mistral-nemo is mechanically solid (77/73%)** but (a) can drift off the news into
  whimsy ("potato that spits fire"), (b) collapses speaker attribution at <=60w, (c) leaks
  the cast name into stage directions on the remote path (F3/BUG-295).
- **Two fail-closed aborts** came from LLM-call *validation*, not the story: news key-term
  strictness (T7, BUG-264 family) and the inventor pass (F4).

The gaps are per-slot. Below, each change names the slot, the call site, and the gate.

## C1 -- Creative length + radio-legibility control (HIGHEST value)

**Slot:** creative (dialogue composer + line composer). **Problem:** Opus writes 2.5x the
target and sprawls; a listener can't track it aurally.
**Changes (prompt-side, no schema change):**
1. Add an explicit **per-line spoken ceiling** to the dialogue/line-composer user prompt:
   e.g. "Each line is spoken aloud on radio -- at most ~2 sentences / ~35 words, one breath.
   Favor a single sharp image over a stacked clause." Anchor: `nodes/_otr_line_composer.py`
   `_build_user_prompt` / `_word_bands`, and the exchange composer `nodes/_otr_stage2_prompt.py`.
2. Add a **whole-episode word-budget reminder** in the outline->dialogue handoff so the model
   spreads the target, not overshoots it (the writer already computes target_words/beat --
   surface it in the prompt).
3. Optional **post-generation trim pass** (creative slot): if a composed line exceeds the
   per-line ceiling by >50%, one targeted "tighten to N words, keep the strongest image"
   regenerate (reuses the existing compose retry ladder -- no new call site).
**Gate:** A/B at 350w -- Opus actual-words should land ~350-450 (not 800+); score must hold
>= 85% and read cleanly aloud.

## C2 -- Model-specific creative system prompts

**Slot:** creative. **Problem:** the same prompt is fed to a 12B local model and to Opus;
they need opposite nudges.
**Changes:** a thin per-model creative-prompt variant selected off the resolved slug
(creative writer already knows the slug via the broadcast output):
- **Opus / frontier:** "restraint" nudge -- concision, plain radio cadence, don't out-clever
  the listener; trust subtext but keep one clear throughline.
- **mistral-nemo / small local:** "stay anchored" nudge -- the story must remain about the
  actual news item; no fantastical tangents; name the real subject in the announcer intro.
**Anchor:** the creative prompt builders in `OTR_LedgerScriptWriter.py` (outline/cast/
dialogue) + `_otr_line_composer.py`. Keep ONE shared base; append a model-class rider.
**Gate:** mistral A/B grounding axis should rise (the potato-fire drift is the target);
Opus density axis should improve without losing grounding/payoff.

## C3 -- Per-slot temperature review

**Slot:** creative + technical. Today temps are global per call-type. Opus likely benefits
from a slightly **lower creative temp** (it's already inventive; lower = more control/less
sprawl); the small local model benefits from its current higher temp for color. Make the
creative temperature a function of model-class (not a new widget -- derive from slug).
**Gate:** part of the C1/C2 A/B; watch arc + coherence don't drop.

## C4 -- news_interpreter key-term validation is too brittle (BUG-264 family)

**Slot:** technical (news_interpreter `build_news_briefs`, `news_interpreter.py:803`).
**Problem:** the strict "every key_term must appear verbatim in source" check (+ LLM-judge)
fail-closed two good stories (T7 'human-like errors'/'UCLA researchers'; the BUG-296 verify
run). The model paraphrases correctly but the term isn't a substring.
**Change:** loosen the gate -- accept a key_term if the LLM-judge says it is *supported by*
the source (semantic), not only if it is a verbatim/word-boundary substring; keep a hard
reject only for terms with no semantic support (true hallucinations). Lower the attempt
penalty so one bad term doesn't burn the whole 3-attempt budget.
**Gate:** re-run the stories that failed (T7-style) -- they should pass; add a regression
that a paraphrased-but-supported key_term is accepted and a hallucinated one still rejected.

## C5 -- F3 cast-name leak retry (already specced as BUG-295)

**Slot:** creative (line composer). Extend the BUG-LOCAL-279 leak filter
(`_otr_line_composer.py` ~L1663-1692) to also retry when an ALL-CAPS **multi-word roster
name** appears mid-phrase / inside a `*...*` stage direction (the case it misses today).
Scope strictly to multi-word ALL-CAPS so single-name drama is untouched. **Gate:** Bug Bible
+ audio byte-identical + the remote-mistral 350w case that produced "ERIN SPENDER".

## C6 -- Low-word attribution binding (F1) -- LOW priority

**Slot:** creative (dialogue). Only matters at <=60w (collapses to one speaker id). Either
enforce a stronger per-line speaker binding in the prompt, or simply **prefer >=150w** for
multi-character scenes (the baseline shows 350w is the sweet spot anyway). Likely no code
change -- a default/guidance fix.

## Sequencing

1. **C4** first (cheap, unblocks fail-closed runs -- highest hit rate on "runs that should
   have produced a story").
2. **C1 + C2 + C3** together (the creative-quality core -- one A/B cycle: Opus + mistral,
   350w, scored vs this baseline).
3. **C5** (bounded, gated).
4. **C6** (guidance/default).

Each step: implement -> Bug Bible + core + audio byte-identical green -> one scored A/B run
(local mistral + Opus) vs the 2026-05-31 baseline -> commit only if score holds/improves and
audio stays byte-identical. **No step ships on a content opinion alone; it ships on a scored
read + green gates.**

## Not in scope (deliberately)

- No overhaul of the writer architecture (local already earns B/B+; this is tuning).
- No change to the two-model slot contract (creative/technical) or the cost guard.
- No audio-path change (C1 is king -- byte-identity must hold).
