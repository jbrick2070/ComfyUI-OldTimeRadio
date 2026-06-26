# LEAKING-WORDS STRATEGY -- problem statement for the roundtable

The operator's standing pain: across MANY sprints we have built deterministic
scrub/reroll passes to stop "leaking words" from reaching the SHIPPED spoken
dialogue, and they STILL leak. The operator is tired of whack-a-mole and wants a
STRATEGY, not another regex pass. Decide the right architecture.

## The leak classes (observed in TODAY's shipped local-model episodes, 2026-06-25)

1. **Stage-direction leak.** `Gasping, "We're running out of time..."` shipped --
   "Gasping," reached the spoken line + got SPOKEN by TTS + shown in captions.
2. **News-bleed.** Raw news facts dropped into fictional dialogue, often
   incoherently: "we're losing the funding!" inside a CRYOGENIC-revival scene;
   "President Trump's orders are clear" / "It's Trump's legacy" named straight into
   a fictional NASA drama. (The writer is TOLD to ground in the news -- the weak
   model over-literally pastes the raw nouns/names.)
3. **Caps-name vocative.** "YUKI MARTIN, no!", "MINA ECKELS" -- a character's full
   name shouted in ALL CAPS mid-dialogue.
4. **Malformed quotes.** Unclosed `"` (gemma).

Frontier writers (GPT) shipped ZERO of these today; the leaks are concentrated in
the LOCAL/weak models (mistral-nemo, gemma).

## What has ALREADY been tried (do NOT re-propose these)

A large deterministic stack in `_otr_line_hygiene.py` + `_otr_ledger_scrub.py` +
`_otr_line_composer.py` + `_otr_story_quality_l12.py` + the freeze cascade:
- **Tier-1 leading scrub** (`scrub_leading_stage_direction`): strips a bare leading
  stage direction, gated on a `_NARRATION_VERBS` WHITELIST + ~7 guards.
- **Tier-2 broad reroll detector** (`detect_stage_business_for_reroll`):
  leading / trailing-after-quote / embedded-between-quotes / undelimited-action,
  also `_NARRATION_VERBS`-gated; on hit -> ONE composer reroll.
- **Tier-3 freeze floor** (`strip_quote_anchored_stage_direction` +
  `_otr_ledger_scrub` `_strip_stage_directions`): deterministic last-resort strip
  of the balanced-quote outside-span class, stamps CODE_STAGE_DIRECTION.
- **split_stage_business** (L7): split a line into dialogue | extracted action.
- **Body-output gate** (KILL-1): `validate_composed_grounding` rerolls a line whose
  crisis nouns are ungrounded / missing the conflict object.
- **narration_leak_repair**: a typed-repair directive for narration in a brief.
- A **2026-06-22 stage-direction-leak sprint** (Tier 1-4, a precision gate over 489
  ledgers: 20 would_mutate, 0 false positives) + multiple story-quality sprints
  (R2/R3, C0-C5).
- **G1 (this session):** measured the abstain residual of the two detectors over
  638 shipped ledgers -- the action-clause residual was ~0, which is WHY a binary
  classifier lane was DROPPED. But "Gasping," is a LEADING bare direction whose
  lead verb ("gasping") is NOT in `_NARRATION_VERBS` -> the whitelist misses it.

## Why it still leaks (the grounded root, my anchor read)

- **The detectors are WHITELIST/REGEX-bound and therefore inherently incomplete.**
  Every deterministic gate keys on `_NARRATION_VERBS` (a finite set) + structural
  shape. A leak led by an un-listed verb ("gasping", "wheezing", "trembling") slips
  every time. Broadening the list is the whack-a-mole the operator is sick of -- it
  is provably always one verb behind.
- **News-bleed is a DIFFERENT problem and is NOT a stage direction at all** -- it is
  semantically wrong CONTENT (a real-world name/fact in fictional mouths). No
  shape/whitelist detector can catch it; it needs a meaning-level check.
- **Caps-vocative** is a cheap, fully-deterministic catch (a token in ALL CAPS that
  matches a cast name) -- arguably just not yet wired as its own scrub.
- **The deepest root is the LOCAL model ceiling:** a weak model imitating
  script format emits directions, raw facts, and shouted names. The frontier
  writers do not. The scrubs are a downstream mop for an upstream generation
  problem.

## The strategic question for the panel

Given MANY deterministic shape/whitelist passes have been built and leaks persist,
what is the RIGHT architecture to actually stop the leaks -- one that ESCAPES the
whack-a-mole? Evaluate (and add your own):

A. **A single FINAL LLM-cleaner pass over the frozen spoken text** -- ask a capable
   model (could be the technical slot, or the frontier slot when enabled) one
   bounded question per line: "Return ONLY the spoken words; remove any stage
   direction, any real-world proper noun that doesn't belong in this fiction, and
   any shouted character name." Deterministic-fallback to the current scrubs.
   (This is the binary-decision idea re-pointed as a CLEANER, not a classifier.)
   Pros/cons vs byte-identity, cost, determinism, model-agnosticism?
B. **Constrained generation / output contract** -- make the line composer
   physically unable to emit a stage direction (e.g. a grammar/format that yields
   only spoken words; strip at decode). Feasible on the local lanes (Ollama/in-
   process/llama.cpp GBNF)?
C. **Push the fix UPSTREAM into the prompt** -- a stronger per-line instruction +
   few-shot of the exact failure modes, so the weak model stops emitting them.
   (Has this been under-tried vs the downstream scrubs?)
D. **Accept the local ceiling; make the frontier writer the recommended default**
   for "clean" output (GPT shipped zero leaks). Keep the local lane as the free
   tier with best-effort scrubs.
E. **Targeted cheap wins regardless:** (1) widen the leak detector beyond a verb
   whitelist (e.g. an `-ing`/`-s` 3rd-person lead + no 1st/2nd pronoun, the relaxed
   heuristic from G1, but as a SCRUB not a measure); (2) a deterministic
   caps-name-vocative scrub; (3) a news-proper-noun guard on body lines.

Which combination is the minimal, durable, model-agnostic answer -- and which of
the above is a trap?

## Invariants to respect

Content-only (ledger schema `l3-2026-05-14` frozen, audio spine byte-identical, NO
workflow-JSON change); deterministic + offline-capable; model/transport-agnostic;
must not degrade the frontier lane (which is already clean); UTF-8 no BOM; SFW.
