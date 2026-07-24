# Problem statement -- push the top performer (scifi_fable2) EVEN further at 720

## Framing (what this panel is actually for)
This is NOT a bug-triage doc. The story-only 720 re-run is in and the winner is clear:
**scifi_fable2 is our top performer.** I want the panel to (1) confirm how fable2's pipeline
actually works, (2) enumerate its real known bugs / current failure surface, and (3) tell me the
highest-leverage ways to make it *even better* -- richer stories, higher yield at 720, without
crossing the operator law. The other banks' failures below are supporting evidence, not the target.

## Evidence fable2 is the top performer
- **720 bakeoff verdict (docs/2026-07-15-720-bakeoff-verdict.md): fable2 ranked #1, marked KEEP.**
- **This sweep's 720 rung, 12/16 SUCCESS.** fable2_v2 produced the single **richest** leg at
  **826 words** (RESULT SUCCESS, 2026-07-17T06:48:07) -- by a wide margin the longest coherent
  720 episode in the matrix (next nearest successes: public_domain_story 501/547,
  media_archive 459/406, shakespeare 375/435, science_news 420/371, scifi_sonnet 220/213).
- fable2 is also the most *architecturally disciplined* lane (see below), which is why it is the
  right vehicle to invest further in.

## How the top performer works (grounded, `nodes/_otr_scifi_fable2.py` + arch doc)
Architecture doc: `docs/2026-07-10-scifi-fable2-architecture.md` (ss 3/5/7/8/9/11/13).
Operator law, stated in the module docstring: **"Python judges; the LLM writes. This module never
writes, trims, or repairs a spoken word."** Every spoken ledger row traces to a named LLM artifact
(per-constituent proof gate).

S2 full loop, LLM-first multipass:
P0 dossier -> deal -> P1 pitch (1 pitch < 120 words; 3 pitches at 120-900) -> P2a selection ->
P2b treatment -> **P3 script** (markup ladder + budget gate + truncation retry) ->
**P4 critic + P5 revision** -> deterministic keep-better judge -> P6 casting/voices ->
P7 pure-python assembly (proof gates + incremental saves) -> P8 ledger audit (audit-only, fail loud).
Compact mode (< 120 words) stamps P2a/P4/P5 as explicit skips. Requests above the 900-word ceiling
(`_SUPPORTED_WORD_CEILING = 900`) fail LOUD -- no silent degrade to compact. Pure module: stdlib +
pydantic + shared structured_call ladder + fable2 markup parser; no ComfyUI/GPU imports; never
imports OTR_LedgerScriptWriter (acyclic graph, pinned by the pure-import test). Every failure raises
a `Fable2Error` subclass naming the pass. No fallback to legacy_many_pass, ever.

## fable2's known bugs / current failure surface (verify each against the file)
1. **SCENE_WORD_GROSS fatal (this sweep: scifi_fable2_v3, 720, scene 4 = 162 character words).**
   `_SCENE_WORD_GROSS_BAND = 0.50` (`nodes/_otr_scifi_fable2.py:202`). The +/-30% per-scene band is
   ADVISORY (prompt vector + `_draft_score` steer); only GROSS imbalance (>+/-50% of a scene's
   target, e.g. ~103-word target -> band ~[52,155]) is FATAL. The comment records this guard was
   *itself* a kibitz 2026-07-14 refinement -- a tighter band had killed complete, correct-total
   episodes. So v3's scene-4=162 is a genuine >50% overage the P5 revision pass could not rebalance
   in 4 attempts. OPEN QUESTION for the panel: is the right lever a smarter revision prompt, a
   deterministic scene-rebalance that does NOT rewrite spoken words, or is the fatal guard still
   slightly too aggressive at high scene counts?
2. **Transient non-determinism at 720.** fable2_v2 logged a FAIL then a SUCCESS (826w) on re-run in
   the same sweep -- the winning path is not yet reliably first-try green at 720.
3. **Unstated-contract hazard (`:483`).** fable2 explicitly documents that
   `schema_shape_instruction` emits field NAMES only, so any pydantic/post-hoc rule is
   model-invisible unless taught in the seam. fable2 guards this better than its peers -- an asset
   to preserve and extend, not regress.
4. **All-caps speaker/title collapse (`:410`).** a title-cased name collapsing to "VOSS" mid-script
   yields UNKNOWN_SPEAKER; the one-word/no-titles normalization is a live sharp edge.

## Cross-bank context (why fable2, not the others -- supporting only)
Same sweep, the non-fable2 fails were all the unstated-contract shape fable2 already defends against:
scifi_codex P5 `PostValidationError: l005 spoken text contains an all-caps lexical word`
(`:2153`) and P3 `4 validation errors for RadioScoreDraftV4` incl. `unused_shot: every declared
shot must own >=1 beat`; original_radio `original_qa unrepairable ['weapons_smoking']`. These banks
lack fable2's proof-gate discipline. They are NOT the target here; they show the class fable2 leads on.

## Hard operator constraints any improvement MUST respect
- **Python judges; the LLM writes. No Python authoring/trimming/rewriting of spoken text.** A
  deterministic step may reorder/prune/coerce *structure and metadata* but may never write a word.
- **A validator may improve a story; it may never fail one.** Green = RESULT SUCCESS + obs_publish +
  asset on disk. Word count is a recorded property, never a pass/fail gate. Do not add length gates.
- **Every ledger field keeps exactly one owner.** No unowned field after any change.
- Root fixes only, in the seam/pack/pass contract; wire any workflow-JSON change in the same change.

## What I want from the panel (one prompt, full arc)
**R1 (high-level arc):** Given fable2 is our best lane, what are the 2-3 highest-leverage moves to
make it *even better* at 720 and toward the 900 ceiling -- richer, more coherent stories AND higher
first-try yield -- without weakening the "Python judges / LLM writes" discipline? Weigh at least:
teaching more of the post-hoc contract into the seam up front; a non-text-authoring scene-rebalance
path for P5; tightening the P4 critic -> P5 revision loop; and whether the 900 ceiling / act-chunk
mode is the next unlock. Name what is genuinely worth doing vs gold-plating.
**R2 (coding plan):** Concrete plan for the chosen moves -- which functions (`_pass_script`,
`_pass_critic`, `_pass_revision`, `_pass_treatment`, the `structured_call` seam) change, how the
SCENE_WORD_GROSS revision recovers without rewriting spoken words, and the regression coverage that
reproduces the v3 scene-4 overage and the v2 first-try flake.

I (Cowork) am the code-grounded judge; ground every claim against the real files and I will discard
misreads. r3 wiring and r4 convergence follow the standard arc.
