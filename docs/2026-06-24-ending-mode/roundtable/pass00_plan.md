# Ending-mode design -- break the doomsday-button climax (roundtable problem statement)

## Grounded problem (verified live, 2026-06-24)

The OTR pipeline now diversifies the PREMISE (pitch room) and the STYLE label
(auto picker), proven across a soak. But the local writer (mistral-nemo /
gemma-12b) collapses EVERY premise + style into the same climax: a control room,
a gauge in the red, someone reaching for a kill-switch / self-destruct, "blow
everything up." Evidence from real ledgers:

- A black-market GENETICS premise ("A Mist of Frost") composed as: *"touch that
  lever again and the vats release every ounce of neurotoxin... the keys are
  sitting on the console... the whole floor ended up as ash."*
- A NEUROSCIENCE seed ("Backwards Time") composed as: *"Activate the AI, now." /
  "Not without the council's approval."*
- Critic verdict on every soak episode: `arc_verdict=uneven`,
  `failing_axes=['emotional_arc']`. Grades: gemma ~65, mistral ~42 -- none reach B(75).

## Root cause (the climax)

The beat planner assigns a fixed dramatic spine to the voiced character beats
(`nodes/_otr_story_quality_l12.assign_beat_roles`):
`setup -> pressure -> personal_stake -> irreversible_choice` (the climax is
ALWAYS `irreversible_choice`, forced onto the LAST voiced beat). A weak local
model renders "an irreversible choice under rising stakes" as the doomsday
button. The spine is sound in theory but, with a small model, it is the trope
generator.

## What we just built (context for the panel)

- `nodes/_otr_style_catalog.py`: 100 curated radio-drama styles, each a
  repeatable GRAMMAR -- `sound_world` + `story_engine` + `ending_mode`. Most
  ending_modes are deliberately NON-explosive (revelation, reversal, quiet
  reconciliation, unresolved final sound, bittersweet parting, ironic twist).
- Decision LOCKED by the operator: switch the style stage from "LLM invents a
  fresh style name" to "SELECT the best-fit style from the 100-catalog," then
  inject that style's grammar into the prompts.
- T4 staging penalty (on-mic climax) + T2 critic adapter already exist
  (default-OFF). L1/L2 crisis-noun grounding (swaps console/lever/gauge for
  premise-specific objects) is built but default-OFF.

## The question for the panel (4-round campaign)

**How should `ending_mode` be defined and WIRED so it actually overrides the
`irreversible_choice` climax -- so the weak local model stops collapsing every
premise into the kill-switch -- while staying deterministic, default-safe
(byte-identical when off), and not breaking audio?**

Specifically rank/harden these alternatives and converge on one:

1. **Ending taxonomy.** What is the right CLOSED set of ending archetypes (e.g.
   revelation / reversal / unresolved-final-sound / reconciliation / bittersweet
   parting / ironic-twist / quiet-acceptance / confession / ambiguous-fade)?
   How many? Must they be mutually exclusive? Each catalog style maps to one.

2. **Where to enforce it.** Options:
   (a) Replace the climax `beat_role` itself -- make the last beat's role
       style-driven (`ending_mode`) instead of always `irreversible_choice`.
   (b) Keep `irreversible_choice` as the role but REFRAME it per ending_mode in
       the beat/line prompt (the choice need not be a doomsday button -- it can
       be a confession, a refusal, a quiet decision).
   (c) Inject the ending_mode only at the line composer for the final beat(s).
   (d) Some combination + a negative constraint banning the GENERIC_CRISIS_NOUNS
       vocabulary at the climax.
   Which placement most reliably moves a WEAK local model off the trope?

3. **Anti-trope constraint.** Should we add an explicit negative instruction
   (no countdown / self-destruct / "blow everything up" / kill-switch unless the
   premise literally requires it)? Risk: weak models obey negatives poorly.
   Better to PULL toward a concrete alternative than to PUSH away from the trope?

4. **The style SELECT mechanic.** Replacing the inventor: should the picker (a)
   classify the article to a domain then pick the best-fit style, (b) score all
   100 and pick top-1, or (c) keep a small LLM "chooser" over a shortlist? How
   to keep the emergency-tagged styles from dominating (only ~when the article
   demands one)? Determinism + the C7 byte-identity seed path must hold.

5. **Validation.** How do we measure success cheaply? (e.g. crisis-noun density
   at the climax, distinct ending archetypes across a soak, critic arc_verdict
   distribution, a small graded A/B.)

## Hard constraints (non-negotiable)

- 100% local writer is the default lane (frontier is opt-in, separate decision).
- Ships DARK / default-OFF / byte-identical when the flag is off.
- Deterministic, seed-keyed; the C7 audio byte-identity gate must hold.
- Edit the canonical `workflows/otr_scifi_16gb_full.json` in the SAME change as
  any node/widget change; full suite + Bug Bible green; UTF-8 no BOM; SFW.
- No new heavy model; the style SELECT must be cheap (no extra paid call).
