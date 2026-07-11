# AGY ROUND 3 -- the 720w execution plan + the dead-lever audit (paste into agy)

Round 2 (agy_review2.md) was strong. Status update, then three jobs.

RULES (you broke one last time): you are a REVIEWER. Do NOT git add, git commit,
or git push. Do NOT edit any file except your own review. Last round you committed
and pushed to v2.0-alpha -- it was harmless (only your two review files), but I run
the git. Write this review to `agy_review3.md` and stop there.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha  HEAD: ccaa869d
Read the real files. Label every claim CONFIRMED (you opened it) or [ASSUMPTION].

## Where we actually are

The canonical 30-word Codex lane now clears every gate that has ever killed it:
P5 deterministic metadata repair (no LLM repair call), the voice/delivery gate,
and CastLock -- which was the blocker you reviewed. It is in the media tail now.
Gemini and Sonnet 30w run next, then the 720-word bake-off across the media packs.

Two of your round-2 findings are already ACCEPTED and in the tree (bcaf446c):
- accepted-line-count-aware `_script_output_token_budget`
- the whole-script root contract now forbids echoing the request envelope

## JOB 1 -- settle the dead-lever question with PROOF (I do not trust either answer yet)

You claim `resolve_context_cap` is the LIVE path and `compute_effective_context_limit`
is DEAD (tests only). But `tests/test_effective_context_limit.py` describes ITSELF as
a "Mirror of the legacy CURATED_CONTEXT_OVERRIDES / resolve_context_cap path" that
"reads the NEW row field" -- i.e. the test thinks `resolve_context_cap` is the legacy
one and the row field is the new one. Exactly one of you is right, and I will not
patch a dead lever.

Settle it with an unbroken call chain, file:line at EVERY hop, from the ComfyUI node
execution down to the value that lands in `cache_entry["context_cap"]` which
`OTR_LedgerScriptWriter._build_truncating_generate_fn` reads at line 647:

  OTR_LedgerScriptWriter.<entry> -> ... -> request_slot -> ... -> load_llm -> cache_entry

For each hop quote the actual line. Then answer:
1. Is `resolve_context_cap` called at RUNTIME on the transformers_safetensors path
   for `mistralai/Mistral-Nemo-Instruct-2407`? Quote the call site.
2. Is `compute_effective_context_limit` called by ANY non-test caller? Quote it, or
   state plainly that it has none.
3. Does the per-row `CuratedModel.context_window=8192` (catalog line 127) affect the
   runtime cap, or only a precondition check? Which function consumes it?
4. FINAL: the exact minimal edit set to make the effective cap 16384 for the writer
   LLM, and the exact set of tests that must change with it.

If your round-2 answer was wrong, say so plainly. I would rather you retract than
have me patch the wrong table.

## JOB 2 -- the 720w execution plan (this is the deliverable)

You found the ceiling: `_script_output_token_budget` clamps at 5400, and you estimate
a 720w/48-line ScriptArtifactV4 needs ~5.0-5.5k+ output tokens. So the ceiling and the
context cap must move TOGETHER or 720w truncates. Give me the plan, concretely:

1. What must the output ceiling become at 720w/48 lines? Show the token arithmetic for
   the artifact (per-line metadata x lines + dialogue + envelope), not a guess.
2. What must `context_cap` then be to hold prompt + that output for P5, P7, AND P9?
   (P7/P9 are the worst case: they carry the previous full script AS INPUT and re-emit
   it AS OUTPUT.) Give the number, then the env var value to set.
3. You flagged P6/P8 lack `prompt_must_fit`. But P6 and P8 take the FULL SCRIPT as
   input and their `max_new_tokens` are fixed literals (2200 / 2400). At 720w does the
   P6/P8 *input* blow the cap, or the *output*, or both? Fixed literal budgets for a
   review of a 6x longer script look wrong to me -- do they need to scale too, and on
   what dimension?
4. `beat_ids = [f"b{i:03d}" for i in range(max(3, min(12, len(p2.cast) * 3)))]`
   (_otr_scifi_codex, run_scifi_codex_episode). The beat count is CAPPED AT 12 no matter
   the word count. At 720 words that is ~60 words per beat, vs ~2.5 at 30w. Is that
   coherent with `make_advisory_word_blueprint`, the per-beat word plan, the music-cue
   anchors, and the score/line contract? Or does 720w need more beats -- and if so, what
   else assumes <=12 beats? This is the question I most expect to bite us.
5. Ordered execution list: what to change, in what order, with the test to write for
   each, such that each step is independently green.

## JOB 3 -- the dead-code rip inventory (new standing operator directive)

The operator has ordered a rip of legacy code that nothing calls, because a dead lever
costs us live rolls -- a reviewer patches the dead one (this exact context-cap confusion
is the case study; it is now logged as GO_FORWARD_PLAN section 5).

Give me a first inventory for the context/limit/model-loading family ONLY (do not boil
the ocean):
- Every function, constant, or table in `_otr_model_catalog.py`, `_otr_loader_backends.py`,
  `_otr_model_loader.py`, `_otr_model_runtime.py` that has NO non-test caller.
- For each: file:line, what it was for, what superseded it, and whether any test pins it
  (and whether that test asserts RUNTIME behavior or merely the constant's value -- a test
  that only pins a dead constant dies with it).
- Flag anything where TWO levers exist for one behavior and neither is marked as live.

## Output format (agy_review3.md)

JOB 1 CALL CHAIN: the hops with quoted file:line; the 4 answers; an explicit retraction
  if your round-2 answer was wrong.
JOB 2 720W PLAN: the arithmetic, the numbers, the beat-cap verdict, the ordered list.
JOB 3 DEAD-LEVER INVENTORY: the table.
CONFIDENCE: CONFIRMED / [ASSUMPTION] on every claim.
