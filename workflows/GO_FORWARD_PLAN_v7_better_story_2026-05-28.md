# Go-Forward Plan v7 — Breaking the 3.7 Plateau (Claude's thesis)

**Date:** 2026-05-28
**Author:** Claude (challenged to stake a real position, not hedge)
**Companion doc:** `docs/2026-05-28-better-story-problem-statement.md` (the round-robin seed)
**Status:** opinion / proposal — not yet wired, not committed to ROADMAP

---

## Thesis (one sentence)

The 3.7 plateau is **not a prompt problem and not a model problem — it is a composition-unit problem**: the pipeline writes and scores one line at a time, and tension, subtext, and reversal are properties that only exist *across* lines, so the system is structurally incapable of producing them no matter how good each line is.

Everything below follows from that claim. If you only read one section, read §2.

---

## 1. What the evidence actually says

From "Mars Heartbeat Wait" (mean 3.7, ship):

- The **structural** axes scored 4: premise_clarity, pacing, resolution, character_distinctiveness.
- The **cross-line craft** axes scored 3: naturalness, emotional_arc, specificity (continuity 3 too).
- SFW 5, audio_readiness 4.

That split is the whole story. The pipeline is *good at structure* (it nails "is there a premise, do beats follow, does it end") and *bad at texture* (does it feel alive, does tension build, is anything specific). The critic even named the two inert lines (b002 exposition, b004 "dramatically inert") — both are cases where a line, judged alone, is fine, but in sequence it carries no charge.

**Conclusion:** the low axes are exactly the ones that require lines to *know about and build on each other*. The high axes are the ones a line can satisfy in isolation. The architecture optimizes line-in-isolation. QED on the thesis.

## 2. Root-cause ranking (my opinion, most → least load-bearing)

1. **Per-beat, best-of-N-in-isolation dialogue composer.** Each beat generates N=4 candidate *lines* and picks a local winner. Local optimization of each line pushes toward self-contained competence — every line "explains itself," which *is* told-not-shown. Subtext (withholding), setups/payoffs, and escalation are cross-line; they cannot emerge from a greedy per-line argmax. **This is the plateau's source.**

2. **Episodes are too short and have no middle.** 8 voiced lines, 2 acts (setup → resolution), 137 words. The premise was a *wait* — and there is no act for the wait to go wrong in. You cannot dramatize tension with no runway. Structural, and trivially cheap to change.

3. **The dramatic-state object is decorative.** `costly_choice_beat=d007` is stamped, but nothing verifies d007 *contains* a costly choice. Labels without enforcement. The "ending changes" flag is asserted, never checked against the actual lines.

4. **The editor optimizes the wrong target.** It scores against the *same* 10-axis ship-critic, which rewards competence and SFW. A competent-flat draft passes. There is no axis for subtext, reversal, or a paid-off choice, so the editor literally cannot push toward engagement — only toward not-broken.

5. **No de-exposition discipline.** "intriguing," "game-changing," "unbelievable," "transformative" are emotion-naming tells. Cheap to ban; real but shallow.

## 3. The plan (sequenced, cheapest-highest-leverage first)

### Phase 0 — Baseline before touching anything (this week, ~1 evening)
The heartbeat is shipped (`2b3e708`). Now run **N=3 episodes on the current build** and record per-axis means + variance. We do not change a single parameter until we have a real baseline, or every later "improvement" is indistinguishable from the ±0.2–0.4 critic jitter. **Gate: N=3 mean recorded.**

### Phase 1 — Give the story a middle (low risk, high leverage)
- `act_count` 2 → **3** (setup / **complication** / resolution).
- `target_words` ~220 → **~400**; per-phase beat budget so the complication act gets a **setback beat** (signal drop, contradictory reading, a broken protocol — something that makes the wait *go wrong* before it resolves).
- Mostly outline-budget params; minimal new code surface.
- **Why first:** emotional_arc and pacing physically have no room to move at 8 lines. This buys runway for everything else. **Test:** does emotional_arc come off 3 across N=3? Watch VRAM (longer script, more compose passes) against the 14.5 GB ceiling.

### Phase 2 — Scene-level composition (THE change; medium-high risk)
Replace per-beat line-in-isolation with **scene composition**: generate a *cluster of 3–5 lines* for one beat-group in a single pass, with the already-committed lines in context, and run best-of-N **at the scene level**, scored for build / subtext / turn — not per line.
- This is the change I believe actually breaks the plateau. It makes cross-line properties *expressible and selectable*.
- Risk: longer context per call (VRAM + latency), and it touches the multiturn composer surface + the commit path (rows must still map to dialogue slots cleanly — re-use the clean commit we already have: 8/8, no fallback).
- **Do it after Phase 1** proves the runway exists. **Test:** naturalness + specificity off 3; the critic's "flat_lines" list shrinks; N=3 mean ≥ 4.0.

### Phase 3 — A craft rubric for the editor (medium risk)
Add three editor-only axes, distinct from the ship-critic: **subtext** (does a line imply rather than state), **reversal** (does the scene turn), **costly_choice_realized** (does the tagged beat cost the character something on the page). Wire them as the editor's failing-axis vocabulary so revisions optimize toward engagement, and *verify* the dramatic-state object's claims against the actual lines instead of trusting the stamp.
- **Test:** editor revisions measurably change the flagged lines (use the new heartbeat + a transcript spot-check); ship-critic mean rises without SFW/continuity regressing.

### Phase 4 — Cheap discipline layer (low risk, drop in anytime)
- Banned-phrase list injected into the composer's negative constraints: *intriguing, game-changing, unbelievable, transformative,* + a standing "do not name your own emotion" rule.
- One **de-exposition punch-up pass** over the committed draft.
- **Test:** fastest single-axis nudge — naturalness/specificity should tick up immediately; cheapest thing to ship if you want a visible win this week.

## 4. If you do only one thing
**Phase 2 (scene composition).** It's the only item that attacks the root cause. Phases 1 and 4 make the number prettier; Phase 2 is what makes the *story* better. But sequence matters — Phase 2 without Phase 1's runway will still be cramped, and without Phase 0 you won't be able to prove it worked.

**Recommended order:** 0 → 1 → 4 → 2 → 3. (1 and 4 are quick wins that also de-risk reading Phase 2's signal.)

## 5. Risks & non-negotiables (carried from project rules)
- **Audio is king** — every phase is text-side; if any change destabilizes the commit→audio path, revert. Re-run the audio byte-identity gate.
- **14.5 GB ceiling** — Phases 1 and 2 lengthen context; profile VRAM, truncate against `context_cap`, never `force_vram_offload()` between LLM phases.
- **Latency is real** — more/longer passes cost minutes per episode. Phase 2 especially. Budget it.
- **Run the Bug Bible + core + audio regression after every change** (per CLAUDE.md), and N=3 critic means before declaring any phase a win — single runs are noise.
- **SFW, no profanity, full arc** — unchanged.

## 6. Loose threads found today (separate from this plan — log, don't bundle)
- **Stage-1 shadow plan validation bug:** the constrained plan emits `tension=0` but the schema floor is 1, so the shadow pass exhausts its 2 retries every run (`stage1_shadow_plan_present: false`). Either widen the schema floor to 0 or clamp the generator. Costs time, throws away the shadow plan. Candidate for `BUG_LOG.md`.
- **Critic key discrepancy:** the ledger carries `stage7_shadow_critic.mean_score` (3.7) but **not** `whole_episode_critic` — the key the 3.70 baseline was defined against. Confirm whether these are the same critic renamed or two passes, so the baseline comparison is provably apples-to-apples.

---

### Bottom line
The system already writes *correct* radio dramas. It does not yet write *alive* ones, because it never lets lines build on each other. Lengthen the runway (Phase 1), then change the composition unit from the line to the scene (Phase 2). That is the path off 3.7. Everything else is polish on top of those two moves.
