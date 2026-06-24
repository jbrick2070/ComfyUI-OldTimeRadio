# R1 anchor review (Claude, code-grounded) — ending-mode design

Focus: high-level arc / creative coherence. Claims labeled CONFIRMED / MISREAD /
UNVERIFIABLE against the real files.

## VERDICT

Direction is right, ONE creative assumption is dangerous. The defect is a
COMPOSITION-LAYER collapse, not a premise or style-label problem — confirmed
live: a genetics premise and a neuroscience seed both composed into the same
console/kill-switch climax (CONFIRMED from ledgers). `ending_mode` is the correct
lever. BUT the plan's implicit assumption — that a weak local model will honor an
"ending mode" instruction — is the same assumption that already FAILED: the
composer ignores the pitch's `final_20_seconds` today (CONFIRMED — the smoke's
pitched `final_20_seconds` named an on-stage choice, yet episodes still went
console). A relabel or a soft directive will be ignored the same way. The design
must PULL toward a concrete, pre-seeded final beat, not just name a mode.

## MUST-FIX

1. **The climax `beat_role` is the trope generator — neutralize it at the role,
   not just the prompt.** CONFIRMED: `assign_beat_roles` (nodes/_otr_story_quality_l12.py
   ~L447-476) forces `irreversible_choice` onto the LAST voiced beat, always. A
   weak model reads "irreversible choice + rising stakes" as the doomsday button.
   Option 2b (keep the role, reframe in the prompt) is the weakest — the model
   still sees the role. Favor 2a (the last beat's dramatic function becomes
   style-driven by `ending_mode`) and feed the composer a CONCRETE final-beat
   intent derived from the style, not an abstract label.

2. **Negative constraints under-perform on small local models — PULL, don't
   PUSH.** CONFIRMED by the prior R3 soak (docs note): L2's withhold/deflect
   directive was eligible everywhere yet the 12B/grok writers IGNORED it and kept
   command-shouting. "Don't use countdown/self-destruct/kill-switch" alone will
   be ignored. Pair any ban with a positive, concrete alternative the model can
   execute (a named ending beat).

3. **Taxonomy must be SMALL and CONCRETE.** A 9-archetype abstract list invites
   the model to map everything back to "the dramatic one." Recommend 5-7 mutually
   exclusive archetypes, each carrying a one-line CONCRETE final-beat template
   (what literally happens / what the last sound is), not just a mood word.

## SHOULD-FIX

4. **Style SELECT over the 100-catalog: classify-then-pick, deterministic.**
   CONFIRMED the catalog exists (nodes/_otr_style_catalog.py, 100 entries +
   helpers + EMERGENCY_TAG). Replacing the inventor with a cheap LLM chooser over
   a SHORTLIST (filter out emergency-tagged unless the article demands one, then
   pick top-1) is sound. Must keep the C7 seed path (OTR_CAST_SEED/style) so the
   byte-identity gate holds — UNVERIFIABLE until wired; flag as verify-at-build.

5. **Sound_world is a free render win** — thread it into the visualizer/LTX
   prompt so episodes stop LOOKING like one control room too. Low risk.

6. **L1/L2 crisis-noun grounding is the cheap complement** — it swaps
   console/lever/gauge for premise-specific objects (CONFIRMED built, default
   OFF). Turning it on alongside the ending-mode lever attacks the trope
   vocabulary directly. Cheap to test.

## Cheapest validation (answer Q5)

Crisis-noun density AT THE CLIMAX BEAT is the single best cheap metric — we
already have `count_ungrounded_crisis` over GENERIC_CRISIS_NOUNS (CONFIRMED in
l12). Measure it on the final beat's spoken text, plus the distribution of
distinct ending archetypes across a soak and the critic `arc_verdict` mix. A
small graded A/B (lever off vs on) over ~6 episodes settles it.

## UNVERIFIABLE (verify-at-build)

- Whether a weak local model actually shifts off the trope with a concrete
  pre-seeded ending — needs the live A/B above. This is the whole bet.
- Byte-identity of the C7 path after swapping inventor→select — wire then assert.
