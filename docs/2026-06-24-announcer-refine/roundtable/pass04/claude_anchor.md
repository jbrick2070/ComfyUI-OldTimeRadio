# R4 CLAUDE ANCHOR -- convergence / residual defects (code-grounded)

VERDICT: yes-with-fixes, where the only "fixes" left are build-time VERIFY steps
and one operator wording choice -- no architecture or interface remains open. The
design has converged across R1->R3.

## MUST-FIX BEFORE BUILD
None from the anchor -- the architecture, signatures, sequence, and flag plumb are
all grounded and consistent. Remaining items are verify-at-build (below).

## FIX-INTRODUCED REGRESSION CHECK (do the cuts undercut the goals?)
1. [Scope cut: per-line register] **No goal regression, but state the honest
   scope.** KILL 2's goal is "the style shapes the body." The first build delivers
   STRUCTURAL style (outline `story_engine`/grammar injection) + deterministic
   `conflict_object` grounding -- two different slugs WILL produce different
   structure/objects. What is deferred is TONAL register inside dialogue lines.
   This matches the operator thesis (structure + grounding now; raw line-craft is
   the deferred model-ceiling question). Acceptance must read "structurally
   different", not "tonally different", for the first soak.
2. [Scope defer: spoiler belt] **Input-starvation is deterministic; one residual.**
   Severing `script_brief` makes it impossible for the open prompt to carry the
   news/outcome. The ONLY residual spoiler path is `opening_status_quo` itself (the
   setup-beat intent) if the outline LLM front-loads the ending into the setup
   beat -- structurally unlikely (setup = status quo), and the deferred belt is the
   insurance. Honest claim: "deterministically cannot spoil from script_brief/news;
   setup-beat self-spoil is low-risk, belt-covered later."

## SHOULD-FIX
1. [STEP C macro inject] Confirm adding the `style_grammar` block to the MACRO
   prompt does NOT confuse the macro's STRUCTURED-OUTPUT parse (it is extra prompt
   text, not a schema field -- should be safe, but the macro uses a strict JSON
   schema; verify the added instruction does not induce the LLM to emit extra keys).
2. [STEP F validator] Pin the exact `_ANNOUNCER_OUTRO_MIN/MAX_CHARS` vs the new
   18-45-WORD coda band -- one validator must own the bound (don't leave both the
   char-band and the word-band live).

## CUT THESE
- Nothing further. The plan is at minimum viable scope (the two R3 cuts already
  trimmed the risky wiring).

## VERIFY-AT-BUILD CHECKLIST (consolidated; each has a home in pass03)
1. `era` source for SafeOpenBrief (meta/period; "" acceptable). [STEP D]
2. ledger line `beat_id` present on the climax row at outro time (doc'd
   `_otr_ledger.py:96`). [STEP F]
3. `news_close_brief` never empty (guarded) + distinct from `ending_change`. [STEP F]
4. `OutlineRequest` asdict/repr snapshot fixtures (one new default-"" key). [BYTE-IDENTITY]
5. `_build_phase/beat_user_prompt` new param compiles + all call sites updated. [STEP C]
6. both `_ANNOUNCER_*_SYSTEM` rewrites emit new text ONLY under flag (off byte-identical). [STEP A/E/F]
7. macro structured-output parse unaffected by the grammar-block injection. [SHOULD-FIX 1]
8. run()-level OFF-flag golden tests (open line, outro line, ledger meta) green +
   `test_audio_byte_identical` green. [BUILD CHUNKS]

## OPERATOR DECISION (not a build blocker)
The coda lead-in WORDING is the operator's creative call. Build uses ONE fixed
lead-in constant; "The real story:" works mechanically but reads modern -- the
in-voice options ("From tonight's headlines:", "The true account:") protect the
OTR fiction. Coder wires whatever constant the operator picks.

## CONVERGENCE
From the anchor: CONVERGED. No new assumptions, no open interface. Pending the
panel's final sweep for anything that survived R1-R3.
