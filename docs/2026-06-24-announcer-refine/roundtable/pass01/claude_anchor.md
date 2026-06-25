# R1 CLAUDE ANCHOR -- high-level arc / creative coherence (code-grounded)

VERDICT: yes-with-fixes. The thesis (drama delivers, news is the payload) is
coherent and most seams already exist, but two arc-level contradictions and one
overstated lever need to be settled before build.

## MUST-FIX BEFORE BUILD
1. [§2 Job3 / §5 ask1] **The current outro VOICE contradicts the news coda.**
   CONFIRMED: `_ANNOUNCER_OUTRO_SYSTEM` (_otr_line_composer.py:2536-2555)
   explicitly forbids a news-summary/"lesson" ("Do NOT state a moral, lesson, or
   news-summary ... 'reminding us', 'tonight's revelation', 'this shows'") and
   demands a CONCRETE FINAL IMAGE. The 3-jobs design wants an explicit "here's
   the real story" teaching coda -- a direct contradiction. FIX: decide the
   announcer END architecture and write it into the plan: either (a) a SEPARATE
   news-coda segment/beat AFTER the reflective close, with its own teaching-
   register prompt, or (b) fold the coda into the outro and REWRITE the outro
   voice to permit the explicit news read. You cannot keep the "no news-summary"
   rule and also ship a news coda. Recommend (a): keep the reflective concrete-
   image close as the character-protecting beat, add a distinct news coda.
2. [§2 Job1 / §5 ask2] **No-spoiler is not deterministic while the open is built
   from `script_brief`.** CONFIRMED: `compose_announcer_intro` (2709) reads only
   `script_brief`; the deterministic `fallback_announcer_intro` (2614) echoes
   `script_brief` verbatim. `script_brief` can contain the outcome -> BOTH paths
   can spoil, and the existing soft instruction ("hint, do not summarize",
   2532) is the exact single-prior trap KILL 1 disproved. FIX: build the open
   from STRUCTURED, outcome-free fields (time/place/cast/opening-situation from
   outline + contract), NOT free `script_brief`; add a deterministic post-gate
   (reject outcome/twist tokens derived from the known ending) with
   reroll-once-else-structured-fallback. Mirror KILL 1.
3. [§1 / §5 ask3] **KILL 2's acceptance overstates an instruction-only lever.**
   CONFIRMED the mechanism is prompt injection of `render_style_grammar`-class
   text (style/sound_world/story_engine, _otr_style_catalog.py:678) into
   macro/phase/beat + every body line. But "two different styles produce visibly
   different stories" is NOT deterministically enforceable by a prompt block on a
   weak local writer -- the same failure KILL 1 had to gate. FIX: scope KILL 2
   honestly as a STEER (it sets sound_world + premise-specific conflict objects),
   measured by re-soak; put the deterministic teeth where they CAN bite (the
   premise-specific `conflict_object` slot, already deterministic via
   `assign_conflict_slot`, + the KILL 1 body gate). Do NOT promise visibly
   different stories from the grammar block alone; "style register" is not
   gate-able the way crisis nouns are.

## SHOULD-FIX
1. [§2 Job2 / §4] **"Protect the character climax" vs. the outro's "state the
   outcome plainly" branch are in tension.** CONFIRMED: `compose_announcer_outro`
   takes `final_character_line` + `ending_change` and, when resolved, injects
   "State this outcome plainly" (2854-2858). If the character beat already landed
   the climax, having the announcer restate the FICTIONAL resolution IS
   pre-empting it. Resolve by `ending_tag`: when the climax landed in-character,
   the announcer pivots to the NEWS coda (real fact), not the fictional outcome.
2. [§5 ask1] **A modern "The real story:" tag breaks the OTR fiction.** An
   in-world period radio host saying "the real story" is meta-commentary out of
   register. Keep teach-ability via CONSISTENCY + POSITION (always the final
   beat) using a period-appropriate recurring lead-in ("From tonight's
   headlines:", "The true account:", "What the record shows:"). Recognizable and
   in-voice beats a modern label.
3. [§0/§7] **Verify the outline exposes the open's structured inputs.** The
   design assumes `outline.time_of_day` / `outline.setting` / era + a usable cast
   roster. [ASSUMPTION] -- verify at build; if absent, the open's determinism
   collapses back to free text.

## OPTIONAL / NICE-TO-HAVE
- Telemetry symmetry: stamp `meta.story_quality.{open_spoiler_rerolls,
  open_gate_failed, news_coda_emitted}` so the open gate + coda are provable by
  the 3-test "baked in" check, same as the body gate.

## CUT THESE
- Nothing at arc level. KILL 3 is already correctly DEFERRED (§4); do not pull it
  forward. KILL 6 cosmetics (rename select_style, DOMAIN_PALETTE scoring) are not
  in this doc's scope -- keep them out of this build.

## ASSUMPTIONS
- [ASSUMPTION] `news_close_brief` is reliably the REAL news (news_interpreter
  authors it as an "era-neutral closing news read"), distinct from the FICTIONAL
  `ending_change`. The coda's whole premise rests on this distinction -- verify
  they are sourced differently at build.
- [ASSUMPTION] the open and the StoryContract can be built before the outline
  without a circular dependency (contract needs news interpretation; open needs
  outline.setting). Confirm the ordering: contract pre-outline, open post-outline.
