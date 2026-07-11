# OTR LOOK-AHEAD #5 -- paste into agy AND into codex

REVIEWER ONLY. Read anything; do NOT edit source, do NOT git add/commit/push.
Write to `qa5_<yourname>.md` and stop. Pull first. CONFIRMED or [ASSUMPTION] on every
claim. Five things you are sure of beat twenty guesses.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha

## Both of you were right, and it changed the plan

You independently agreed that Sonnet's `c01` gap and its new empty `cites` are
downstream-safe -- so the ceremonial lines now cite nothing HONESTLY instead of citing a
`fact_0` that can never exist, and I shipped it without fear. That is what a good review
buys.

Codex then called the next kill before it happened: **the Sonnet finalizer treats any
ledger WARNING as fatal.** If that is right, it is the sixth instance of the single defect
class that has dominated this entire build:

> **A gate that blocks production may only block on something that is (a) objectively
> checkable and (b) actually fixable. Everything else is a note.**

The five before it: a word-count quota nobody could satisfy; an episode-level fact rule
enforced per scene; a seam demanding a field its schema forbids; a rewrite told to
preserve the wording it was summoned to replace; and an auditor failing an episode because
a line "adds no new information" in a 30-word script with a two-fact dossier.

## JOB 1 -- prove or kill the warning-is-fatal finding, then sweep for the whole class

1. `_SonnetTailFinalizer.before_save` (and Gemini's) raise when
   `pre.errors or pre.warnings or post.errors or post.warnings` or the freeze verdict is
   not `frozen_clean`. Quote it. **What warnings can `phase_0_gap_audit_pre` /
   `phase_10_gap_audit_post_and_freeze` actually emit?** Enumerate them. Which are real
   defects, and which are notes a human would shrug at? Which can a content-owned lane
   trip in normal operation?
2. Is `frozen_clean` achievable for a content-owned lane at all, or does the cascade emit
   a benign warning by construction? If it does, this gate can never pass and Sonnet is
   unshippable by design.
3. Now sweep the WHOLE codebase for the class: every place that BLOCKS (raises, fails a
   run, exhausts a loop) on something that is a note, a preference, an advisory, a
   warning, or an unsatisfiable exactness demand. All four lanes, the writer tail, the
   freeze cascade, the render/media path. For each: file:line, what it blocks on, is it
   objectively checkable, is it actually fixable by the thing being asked to fix it.

## JOB 2 -- Sonnet's remaining tail

Sonnet has never reached the tail. Assume it clears the audit on this roll. Walk
`_assemble` -> `_SonnetTailFinalizer` -> shared writer tail -> CastLock -> freeze -> media
-> credits -> `obs_publish` and rank what kills it. Specifically:
- `_SonnetTailFinalizer._proof` compares a text receipt (`line_text_sha256`) against the
  saved ledger. Does anything in the shared tail MUTATE line text after that receipt is
  taken (delivery stamping, hygiene, vocative stripping, cliche repair)? If so the proof
  fails by construction. **This is my prime suspect after the warning gate.**
- Sonnet's cast is `announcer, c02, c03, c04` with hardcoded kokoro/bark presets. Confirm
  Gate 1 (`_assert_unique_bark_voices`, `_assert_voice_preset_invariant`) passes on it.
- Does Sonnet stamp everything the shared tail expects of a content-owned lane
  (delivery text, episode seed, freeze policy)?

## JOB 3 -- the Python-that-authors rip

agy traced the four spoken-line fallbacks in `_otr_line_composer.py`
(`fallback_announcer_intro`, `fallback_safe_open`, `fallback_announcer_outro`,
`_resolved_outro_fallback`) and classified some as dead. Settle it jointly:
- For EACH: can a content-owned sci-fi lane reach it? Quote the call path or state
  plainly that there is none.
- Which are reachable by the LEGACY lanes (science_news, public_domain, shakespeare)?
  Those still ship episodes, so a reachable fallback there is Python speaking in a real
  broadcast.
- Recommend: RIP or KEEP for each, with the model field that should supply the line.
An AST guard now rejects a literal assigned to a spoken field in the three sci-fi lanes.
Where else should that guard run? Name the files.

## JOB 4 -- the 720w gate is now the last thing standing

All three lanes are about to publish at 30 words. Then the bake-off. You agree: 16k cap is
a GO (+1.25 GiB KV over 8k), `resolve_context_cap` is live, `compute_effective_context_limit`
is dead. Hand me the patch, jointly and precisely:
1. Every file:line to change to make the effective writer cap 16384, and every test that
   pins 8192 (runtime behavior vs bare constant).
2. The reservation formula for each whole-script pass at 720w such that prompt + output
   fit 16384 -- Codex P5/P7/P9, Gemini P4/P6, Sonnet P5. Show the arithmetic.
3. Which passes set `prompt_must_fit=True`.
4. The proof the default (env unset) stays byte-identical at 8192.
5. What ELSE breaks at 720w that is not token budget? Beat counts, per-line ladders (Sonnet
   makes N model calls per line -- what is N and the wall clock at 720w?), media/render
   scaling, caption timing, the 60s MCP ceiling on any single call.

## Output (`qa5_<yourname>.md`)

JOB 1 WARNING-AS-FATAL + the blocking-on-notes sweep (this is the big one)
JOB 2 SONNET TAIL KILL LIST (receipt-vs-mutation first)
JOB 3 FALLBACK RIP LIST (reachable vs dead, per lane)
JOB 4 720W PATCH (edit set, formulas, non-token risks)
