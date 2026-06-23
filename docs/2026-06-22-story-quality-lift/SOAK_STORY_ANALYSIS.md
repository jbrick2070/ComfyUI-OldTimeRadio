# No-bypass writer-rotation soak -- story-quality analysis (2026-06-22)

Source: the 3h diagnostic soak (620w target, all-visualizer FLOOR, indextts2, NO
freeze-bypass) on the canonical `otr_scifi_16gb_full.json`, rotating writers
mistral / gemma-4-E2B / gemma-4-E4B / gemma-4-12b / grok (technical slot = local
mistral throughout, to isolate the creative writer). Ledgers read directly
(ground-truth `meta.creative_writing_model`), so attribution is correct despite
the earlier overlap.

## A. The D1/D3/D2 fixes hold on EVERY writer (no-bypass, live)
Across all 12 captured episodes: **0 D1 stage-direction leaks** in character text,
**0 D3 role coercions / 0 announcer-on-cast-id rows**, **0 critic stance issues**,
freeze gate shipped every completed episode (`frozen_with_warns` /
`frozen_with_doctor_edits`). The morning's work is solid regardless of writer
strength. Story craft is the remaining variable.

## B. Per-writer craft read (arc verdict / flat-line count / notes)
- **gemma-4-E2B (smallest)** -- "Scalpel's Cold Measure": arc=**strong, 0 flat**.
  Concrete stakes ("Our last test subject coded in forty-eight hours. It was
  Jane's daughter."). Punched above its size on this one.
- **gemma-4-12b** -- best PROSE ("my signature is baked into the core's marrow";
  "watch those servers go into a white void while the floor melts"). BUT leaks
  full **third-person NARRATION into character lines** ("A jagged, guttural cry
  tears from her throat as she claws at the panel..."; "The screen flickers red
  as she watches the progress bar..."). This is a NEW leak class D1 does not
  catch (D1 targets verb-led action clauses after a quote; these are complete
  narrative sentences). arc=uneven, 3 + 7 flat.
- **gemma-4-E4B** -- mid. arc=uneven, 4 flat. Artifacts: a CAPS cast-name leaked
  mid-dialogue ("...weeping again, PHYLLIS FLANDERS"); one run's
  `dramatic_state_source=fallback` (the LLM dramatic-state derivation failed ->
  degraded A/B wants "see researchers fully revealed" / "keep them hidden").
- **mistral (Mistral-Nemo)** -- generic technobabble + arc sag. "Blaze's Echo"
  arc=**mid_collapse, 7/18 flat**; one run was only **61 words** (vs 620 target).
- **grok (OpenRouter, reasoning_effort=none)** -- "Whispers from the Dredged
  Deep" arc=**mid_collapse, 14/18 flat** -- the WORST flatness in the batch.
  Short, repetitive command-shouting ("Transmit now", "Override this, Dale",
  "Stop her, Dale"). The `reasoning_effort=none` boot setting (needed so
  reasoning models don't 0-line) may be flattening grok's dialogue variety.

## C. The #1 universal craft weakness -> the next lever
EVERY writer collapses to terse **imperative command-shouting** under pressure:
"Override the protocols!", "Lockdown now!", "Stop the print!", "Evacuate!",
"Transmit the coordinates!". This is what the critic's flat_lines are flagging
(7 / 14 / 7 / 4 clusters). It reads as a control-room barking match, not drama.

## D. Proposed next campaign (story-quality R3 -- writer craft)
1. **Imperative-flatness gate** (the big one): detect a line that is a bare
   command/instruction with no subtext/stakes shift ("Override X", "Stop Y",
   "Transmit Z") and reroll with a "play the pressure indirectly" hint. Pair
   with the existing on-the-nose/cliche gates. Model-agnostic; only lifts the
   weak end (E2B's strong arc shows a good line survives).
2. **Extend D1 to narrative-sentence-in-dialogue** (gemma-12b class): a character
   line that is a complete third-person narration of the speaker's own action
   ("A cry tears from her throat as she claws...") -> reroll. Distinct from the
   verb-led-after-quote class D1 already covers.
3. **Length adherence**: writers compress hard (61-344w vs 620 target). Check the
   per-beat word allocation + the writer's length rider; mistral's 61w run is the
   outlier to chase.
4. **grok reasoning tuning**: try `reasoning_effort=low` (vs none) for grok and
   re-measure flatness -- the no-reasoning flooring may be the cause.
5. **E4B robustness**: the `dramatic_state_source=fallback` + caps-name-in-line
   are two small reliability bugs worth a look.

## E. Casualties of the earlier too-tight wall (context, not bugs)
Two episodes were overlap/kill casualties from the 32-min wall (since fixed to
75-min + queue-idle serialization): one all-empty ledger (`pending_..._173900`,
words=0) and "Meltdown Machine" (`freeze=None`). Not pipeline failures.
