# OTR QA SWEEP #2 -- paste into agy AND into codex

Both of you get this. Answer independently. REVIEWER ONLY: read anything, but do NOT
edit source, do NOT git add/commit/push. Write to `qa2_<yourname>.md` in the repo root.
Label every claim CONFIRMED (you opened it / ran the number) or [ASSUMPTION]. Retract
anything you previously got wrong -- I would rather have a reviewer who corrects himself
than one who defends a table.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha (pull first -- HEAD moves every few minutes)

## Credit where it is due, and what it taught us

You BOTH independently found the same thing, and you were both right:

> `gemini_scene_draft` told the model to "return its fact_ids" and showed
> `{"fact_ids": ["F01"]}` -- but `DraftLineV4` declares `fact_uses` and forbids extras.

That is worse than a typo. The model obeyed the seam, strict mode rejected the artifact,
and my own extra-key repair then DELETED the model's fact attribution to force it to
validate. The critic then correctly reported "F01 is missing from line_fact_ids" and the
run died. **A deterministic repair was silently destroying authored work to satisfy a
contract that contradicted itself.** That is the most dangerous class we have found.

Fixed: the draft and rewrite seams now ask for `fact_uses` (fact_id + spoken_claim) and
explicitly forbid `fact_ids`; the outline seam no longer asks the model for `tts_model` /
`voice_preset` (CastV4 forbids them and Python assigns the voices anyway); eight Sonnet
seams that demanded "exact target words" are now advisory. A guard test fails the build
if a seam ever again shows a field its strict schema forbids.

**The lesson, and your standing instruction: a seam and its schema are ONE contract.
When they disagree, the model is not wrong -- we are.**

## THE LAW (unchanged; a fix that breaks it is an automatic reject)

Python judges. The LLM writes. Python may never author, rewrite, trim, pad, or template
story text. Deterministic repair is for MECHANICAL metadata only (ids, ordering, enums,
a fixed role label, a parent reference, keys the contract never asked for) -- and if it
is ambiguous, FAIL CLOSED. Word count is an advisory scale request and a post-hoc
statistic: it never causes a trim, a pad, a cull, or a rewrite.

## JOB 1 -- finish the seam/schema reconciliation, ALL THREE LANES

I fixed Gemini and Sonnet. Nobody has done Codex, and agy already flagged it:
`codex_radio_score_system`, `codex_play_system`, `codex_final_audit_system` and the
repair seam reportedly demand the model "reproduce the advisory plan exactly", "Count
ordinary spoken words exactly", and "correct invalid word totals."

For EVERY seam in `nodes/story_packs/scifi_*/**.json`, produce a row:
`<pack>::<seam>` | the exact offending sentence | which schema field it contradicts (or
"unsatisfiable") | the corrected sentence you would ship.

Check specifically:
- a field the seam asks for that the strict model FORBIDS (the fact_ids class)
- a required field the seam never asks for (does the model have to guess?)
- any word-count quota, trim, pad, or "exactly" instruction (the LAW)
- a rule enforcing an episode-level property at scene/line level (the F01 class)
- a seam that asks the model for something Python already knows and will overwrite

## JOB 2 -- the Sonnet code bugs, as an executable plan

You both confirmed these. Sonnet is next to run and has NEVER completed. Give me the
patch, the test, and the MECHANICAL/CREATIVE call for each:
1. The P5 rewrite loop never writes corrected lines back into `events`
   (`nodes/_otr_scifi_sonnet.py` ~513), so the re-audit re-reads the same text and the
   loop can only exhaust. Show the exact missing write-back. Merge by index, by
   line_ref, or by beat? What happens to a line the rewrite did NOT touch?
2. `AttestationV4.attestation_cites` allows 4 but `DraftLineV4.cites` allows 3 -- a
   4-cite reply raises on construction. Which limit is the REAL contract, and why? Do
   not just widen the smaller one.
3. `fact_0` hardcoded vs the dossier's `fact_N` indexing. agy says the spec is
   0-indexed; codex says the P0 rules mandate 1-indexed. **You disagree with each
   other.** Settle it against the actual dossier validator and the P0 seam, quote both,
   and tell me which is authoritative. If a line legitimately cites NOTHING, what should
   it carry?

## JOB 3 -- the 720-word gate (this is the next milestone after the 30w publishes)

You agree at 30w everything fits, and that at 720w the whole-script passes are guaranteed
to silently truncate (Codex P5/P7/P9; agy adds Gemini P4/P6 and Sonnet P5). Now design
the fix, concretely:
- The effective `context_cap` is 8192 for the local writer. Trace who ACTUALLY sets it
  (`resolve_context_cap` vs the per-row `CuratedModel.context_window` -- one of these is
  the live path and one may be dead code; agy claimed to settle this, codex should check
  it independently). Give the exact minimal edit set to raise it, and the tests that pin
  the current value.
- What must the cap be at 720w so that prompt + reservation fit for P5/P7/P9? Show the
  arithmetic. What is the KV-cache VRAM cost at that cap for Mistral-Nemo-12B on a 16 GB
  RTX 5080, given how the loader ACTUALLY loads it (quantization? device_map?)?
- Which passes must set `prompt_must_fit=True` so a miss fails LOUD? Is silent
  truncation EVER acceptable? Argue it if you think so.
- The default 8192 reportedly pins an audio byte-identity baseline. Does an env-var
  opt-in preserve it, or does something persist the cap into a ledger/receipt where a
  changed value drifts?

## JOB 4 -- attack the repairs again

Same standing order: try to break `repair_outline_metadata`,
`repair_forbidden_extra_keys`, `repair_script_artifact_metadata`, the CastLock
content-owned VERIFY-not-REPLAY branch, and the `episode_seed` receipt. You found one
real destruction bug (fact attribution). **Find the next one.** Where else can a
deterministic repair silently discard something a human meant, or accept something it
should have failed closed on? Prune-drops are the prime suspect: enumerate every key
the pruner would currently delete, and tell me which of them carry MEANING.

## Output (`qa2_<yourname>.md`)

JOB 1 SEAM/SCHEMA TABLE (all three lanes, Codex especially)
JOB 2 SONNET PLAN (3 fixes; settle the fact_0 disagreement)
JOB 3 720W PLAN (arithmetic, cap, VRAM, prompt_must_fit, baseline safety)
JOB 4 ATTACKS (which prune-drops carry meaning)
CONFIDENCE on every line. Five things you are sure of beat twenty guesses.
