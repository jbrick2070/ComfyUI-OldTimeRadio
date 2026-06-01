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

## Three-pass story architecture (operator direction, 2026-05-31)

Jeffrey's direction after seeing the Opus run: a **three-pass** generate -> edit -> judge
pipeline. "Opus writes it, a more tactical pass cleans it to the right word count and makes
the dialogue fit, and a final pass checks the dialogue is actually creative/interesting."
(Decoding note: "31" = Opus, the model that scored 31/35.) This section is the spine; C1-C6
below are the component changes it composes.

### The three passes

| Pass | Role | What it does | Maps to OTR | Model role |
|------|------|--------------|-------------|------------|
| **1 -- Writer** | creative generation | outline + cast + dialogue from organic news | the existing **creative slot** (Slot A) in `OTR_LedgerScriptWriter` | **Opus 4.8** |
| **2 -- Tactical editor** | editorial cleanup | trim the overshoot to the radio word target; break comma-splice density into spoken cadence; confirm each beat's dialogue is complete + fits | a **new editorial call** after the dialogue stage (kin to, but distinct from, the existing structural Script Doctor) | a cheap, strong instruction-follower |
| **3 -- Creative QA** | taste judgment | is the dialogue actually creative + interesting (not merely structurally valid)? verdict + targets; can send work back | **upgrade of the existing `run_story_critic`** + its reroll/escalation loop | a premium, non-Opus model |

Pass 2 is **C1 made concrete as its own LLM call.** Pass 3 is the existing Sprint-5B story
critic, re-aimed from "arc/flat-line structural check" to "is this *good*," with authority
to trigger a re-edit or re-write (it already can set `needs_full_rerun`).

### Model candidates + cost (per-pass, measured OpenRouter pricing 2026-05-31)

Passes 2+3 are tiny next to Opus Pass 1, so pick them for **fit, not price**:

| Pass | Candidate | $/Mtok in / out | Est. tokens/ep | Est. $/ep | Note |
|------|-----------|-----------------|----------------|-----------|------|
| 1 | **anthropic/claude-opus-4.8** | 5 / 25 | ~43k | **~$0.47** | measured; dominant cost. Could drop if prompted tighter (C2) |
| 2 | openai/gpt-4o-mini | 0.15 / 0.60 | ~4k | ~$0.002 | cheapest capable editor |
| 2 | mistralai/mistral-small-3.2-24b | 0.075 / 0.20 | ~4k | ~$0.001 | cheapest overall |
| 2 | anthropic/claude-haiku-4.5 | 1 / 5 | ~4k | ~$0.008 | same family as Opus -> stylistic consistency (recommended) |
| 2 | local mistral-nemo | 0 (local) | ~4k | $0 | free; the cascade already edits decently, but adds local VRAM/time |
| 3 | **anthropic/claude-sonnet-4.6** | 3 / 15 | ~4k | **~$0.02** | strong literary taste, distinct from Opus (recommended) |
| 3 | anthropic/claude-haiku-4.5 | 1 / 5 | ~4k | ~$0.008 | cheaper QA if Sonnet is overkill |
| 3 | anthropic/claude-opus-4.8 | 5 / 25 | ~4k | ~$0.04 | **discouraged** -- model judging itself |

**Full 3-pass episode cost:** ~**$0.50** with no retry (Pass 1 ~$0.47 + Pass 2 ~$0.005 +
Pass 3 Sonnet ~$0.02). Worst case with **one** Opus re-write on a Pass-3 fail: ~**$1.00**.
Passes 2+3 are rounding error; the cost lever is "how many times does Pass 1 (Opus) run."

### Control flow

Synchronous **1 -> 2 -> 3**, with a **bounded, escalation-scoped** loop on Pass-3 failure
(reuse the existing `_otr_reroll_escalation.py` `EscalationScope` LINE/BEAT/EPISODE):

```
Pass 1 (Opus writes)  ->  Pass 2 (editor trims/fits)  ->  Pass 3 (creative QA verdict)
                                                              |
                          pass -> ship                        | fail
                                                              v
                              EscalationScope decides:
                              LINE/BEAT  -> back to Pass 2 (cheap re-edit; no new Opus tokens)
                              EPISODE    -> back to Pass 1 (Opus re-write; capped at 1 retry for cost)
                              exhausted  -> ship best draft + WARN (today's bypass behavior)
```

Prefer routing Pass-3 failures to **Pass 2 re-edit** first (free/cheap) and only escalate to
a Pass-1 Opus re-write for true structural misses -- this keeps the expected cost at one
Opus pass. Hard cap Opus re-writes at 1 (cost guard already enforces the per-run ceiling).

### Reconciliation with the existing creative/technical (Slot A / Slot B) split

Today: Slot A = creative writer, Slot B = technical (local JSON/structured: cast contract,
script doctor, critic). The 3-pass design **extends, not replaces** this -- it adds a third
*model role*:

- **Slot A (creative writer)** = Pass 1 = Opus. Unchanged in concept.
- **Slot B (technical)** = the structured validators (cast contract, JSON normalization)
  stay local + fail-closed. Unchanged.
- **NEW: editor + critic model roles.** Per PD6 ("only the writer exposes a model_id; every
  other node receives it via a STRING input from the writer's broadcast outputs"), this
  means adding **`editor_model` and `critic_model` broadcast outputs** on
  `OTR_LedgerScriptWriter`, wired to the Pass-2 node and the Pass-3 critic. That is the
  PD6-compliant shape; it is the main new surface.
- Open question (below): is "editor" a third creative-family slot, or just the technical
  slot with a different model? Is "critic" worth a premium slot, or does it stay local?

### Implementation effort (estimate -- PLAN ONLY)

- **Slot plumbing (moderate):** 2 new writer widgets/broadcast outputs (`editor_model`,
  `critic_model`) defaulting to the technical model (so the default path is byte-identical
  until opted in); wire them through `_SlotScheduler.request_slot` to the editor + critic
  call sites; update the workflow JSON (PD3 re-wire) + the two-model-selector routing table.
- **Pass 2 (moderate):** author the editorial prompt (length target + spoken-cadence rules +
  beat-completeness check); add the call site after the dialogue stage (a new node, or a new
  mode on the Script Doctor). Tag `# LLM slot: creative` or a new editorial tag per PD6.
- **Pass 3 (light):** re-aim the existing `run_story_critic` prompt from structural to
  creative-interest judgment; point it at `critic_model`; the reroll/escalation loop already
  exists. Define the **pass/fail gate** (what verdict ships vs loops -- the new design
  question Pass 3 introduces).
- **Gates (every step):** Bug Bible + core + audio byte-identical green, plus a scored A/B
  vs the 2026-05-31 baseline; the 3-pass output must beat single-pass Opus (89%) on the
  rubric AND read cleanly aloud, or the pass isn't worth its tokens.

### Open design questions for Jeffrey (decide before any code)

1. **Slot count:** stay 2 model-slots (editor+critic reuse technical/local) or go to **4
   roles** (writer / editor / critic / technical-JSON)? More slots = more control + more cost
   + more wiring.
2. **Pass 2 model:** local mistral-nemo (free, decent, +VRAM) vs Haiku 4.5 (~$0.008, stronger
   editor, Anthropic-consistent) vs GPT-4o-mini (~$0.002)?
3. **Pass 3 model:** Sonnet 4.6 (~$0.02, recommended) vs Haiku (~$0.008) vs local? (Opus
   self-judging is discouraged.)
4. **Loop policy:** cap Opus re-writes at 1? Prefer Pass-2 re-edit over Pass-1 re-write on
   failure? What's the ship-anyway fallback after exhaustion (keep today's bypass behavior)?
5. **Pass 2 vs Script Doctor:** new dedicated editor node, or extend the existing Script
   Doctor (which already edits, but structurally)? Risk of two passes fighting over the text.
6. **Radio word target:** what is it? (Baseline sweet spot ~350. Pass 2 trims Opus's ~829 to
   this.) Per-line spoken ceiling (~35 words / one breath)?
7. **Is single-pass Opus already good enough?** 89% single-pass is strong. The 3-pass design
   buys *radio legibility* + a *taste gate*, at +~$0.03 and real wiring. Worth it now, or ship
   single-pass Opus + the C1 prompt nudge first and measure?

## C1 -- Creative length + radio-legibility control (= Pass 2 editor)

**This is Pass 2 above, made concrete.** **Slot:** creative/editor. **Problem:** Opus writes
2.5x the target and sprawls; a listener can't track it aurally.
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
