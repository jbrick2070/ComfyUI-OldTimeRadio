ROUND 1 -- CLAUDE ANCHOR REVIEW (grounded against the real code + frozen ledger)

VERDICT: yes-with-fixes. The four defects are real and grounded, but the plan treats three of them
as post-hoc symptom-scrubbing when the root cause is upstream (generation + outline), and it bundles
one likely non-defect (DEFECT 4) as a peer.

MUST-FIX BEFORE BUILD:

1. [§2 DEFECT 1 -- goal/method divergence] The plan's goal is "spoken radio dialogue," but the method
   is "detect + strip stage directions after the fact." The writer is emitting prose-fiction narration
   ("adjusts dials on the console", "tightens her scarf, a nervous gesture") -- a GENERATION defect.
   A pure scrub/floor is a detector arms race against an unbounded space of bare action clauses.
   FIX: make the GENERATION-side constraint the primary lever (the line composer / Stage-3 beat prompt
   must instruct pure spoken words, first/second person, no third-person action narration), with the
   detector + freeze floor as the SAFETY NET, not the only line of defense. Grounded: the compose-time
   reroll already exists (`_otr_line_composer.compose_line` 2015-2060) -- strengthen its hint + add the
   trailing/embedded detector there; keep the deterministic floor conservative.

2. [§2 DEFECT 2 -- wrong altitude] A stance reversal (Manfred: supportive b003 -> dismissive b008 ->
   leaks to press b011/b014 -> defends her life's work b017) is an ARC/OUTLINE defect spanning 5 beats,
   not a single bad line. The plan proposes routing "the flagged line(s) through the existing scoped
   reroll" -- but rerolling b017 in isolation cannot repair an incoherent through-line; it would just
   reword the contradiction. FIX: locate this at the BEAT/OUTLINE planning stage where the antagonist's
   want is assigned (`_otr_dramatic_state.DramaticState.character_b_wants` exists and has a
   `_wants_must_oppose` validator; `_otr_outline` assigns beat intents), so Manfred has ONE coherent
   objective the beats execute. A line-critic axis can DETECT the reversal, but the FIX must be upstream.

3. [§0/§2 -- missing concept-level piece: no central-spine coherence check] DEFECT 2, DEFECT 4 (UN
   jump), and the random "Sherlock" AI appearing at b016/b017 are three faces of ONE root cause: the
   outline does not commit to who wants what and how stakes escalate proportionally. The plan attacks
   the faces separately and misses the spine. FIX: add ONE outline-coherence lever (antagonist gets a
   binding want + a setup-before-escalation rule) that SUBSUMES DEFECT 2 and DEFECT 4, instead of two
   separate gates. This is the highest-leverage creative fix and it is currently absent from the vision.

SHOULD-FIX:

1. [§2 DEFECT 1 -- false-positive principle] The plan lists the FP risk as an "open question." At the
   creative level it must be a committed design principle, because b017 has NO delimiters at all
   ("Sherlock, stop this at once! overrides systems... I won't let you..."). Commit now: strip a
   non-leading segment as a stage direction ONLY when it is third-person + present-tense physical
   action AND the surrounding segments are first/second-person speech. Anything ambiguous -> reroll,
   never deterministic strip. This keeps the floor from eating real dialogue.

2. [§3 -- well-formedness in acceptance] "Stripped from the frozen text" is insufficient: removing
   b015's embedded clause leaves an orphan close-quote; removing b005's leaves a clean sentence. Add to
   acceptance: the stripped line is still well-formed (balanced quotes, no orphan punctuation, non-empty).

3. [§0 -- prove the no-op] The model-agnostic claim needs a concrete check: run every new gate over a
   known-good opus/frontier ledger and assert ZERO strips/rerolls fired. Without this the lift could
   silently degrade strong scripts. Make it a campaign acceptance item.

OPTIONAL / NICE-TO-HAVE:
- A `story_quality_scan.py` signal counting trailing/embedded leaks (not just leading) for soak telemetry.

CUT THESE (scope / over-engineering):

1. [§2 DEFECT 4 -- the UN-escalation GATE] Cut the dedicated proportion/setup GATE. It is almost
   certainly a symptom of DEFECT 2 (incoherent outline) + the weak local model, not an independent
   defect. A new semantic-scope gate is the most likely to flake on good scripts and the least grounded
   (no existing seam). Safe to cut the gate; KEEP a measurement signal in the scan and let the
   outline-coherence lever (MUST-FIX 3) address the root. Re-open only if a no-bypass frontier re-smoke
   still shows abrupt jumps after DEFECT 2 ships.

[ASSUMPTION] The plan's "retest lever = re-smoke without OTR_BYPASS_FREEZE_HALT to see the critic gate
+ reroll" assumes the gate is trustworthy for all four defects. Grounded reality: the critic (SEAM E)
has NO arc/stance axis today, so DEFECT 2 will NOT gate until that axis ships, and DEFECT 4 has no gate
at all. The no-bypass re-smoke is a valid acceptance ONLY for DEFECT 1 (after the floor extension) and
DEFECT 3 (after the consistency assert). State this per-defect so a green re-smoke is not misread as
covering DEFECT 2/4.

[ASSUMPTION] c02=Manfred / c03=Mali / c04=skeptic is inferred from line text; the exact cast names live
in the ledger `cast` block -- verify-at-build, does not change the structural point.
