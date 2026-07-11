# AGY FOLLOW-UP PROMPT -- round 2 (paste this whole file into agy)

Your first review (agy_review.md) was good. Two of your findings were verified
against the real files and ACCEPTED:

- CONFIRMED + ACCEPTED: `resolve_context_cap` returns `min(override, limit)`
  (nodes/_otr_model_catalog.py:1277-1284) and Mistral-Nemo's curated override is
  8192 (line 1227), while `HARD_VRAM_CONTEXT_LIMIT` defaults to 8192 (line 1214).
  So raising OTR_HARD_VRAM_CONTEXT_LIMIT alone CANNOT raise Nemo's cap. That is a
  double lock, and I had it wrong in my brief. Good catch.
- CONFIRMED + ACCEPTED: P5/P7/P9 omit `prompt_must_fit`, so an over-budget
  whole-script prompt silently left-truncates its own schema prefix.

Same rules as before: REVIEWER ONLY. Do not edit/create/delete anything in the
repo except your review. Write it to `agy_review2.md` in the repo root.
Read the real files. Label every claim CONFIRMED or [ASSUMPTION].

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha (working tree has the Defect A + Defect B fixes described in
the first brief; suite 7581 passed / 31 skipped / 1 xfailed, Bug Bible 17 passed)

## Q1 -- WHICH CONTEXT LEVER IS ACTUALLY LIVE? (highest priority; I think your patch may aim at a dead table)

You proposed patching `CURATED_CONTEXT_OVERRIDES` (nodes/_otr_model_catalog.py:1226).
But `CuratedModel` also carries a per-row `context_window` field, and the
Mistral-Nemo row sets `context_window=8192` (nodes/_otr_model_catalog.py:127).
And `tests/test_effective_context_limit.py` documents itself as a "Mirror of the
legacy CURATED_CONTEXT_OVERRIDES / resolve_context_cap path; reads the NEW row
field so per-row [values win]".

That smells like `resolve_context_cap` / `CURATED_CONTEXT_OVERRIDES` may be a
LEGACY path, and the value the running writer actually gets may come from the row
field via a different function (an "effective context limit" resolver) per loader
backend.

Trace the REAL runtime path end to end and tell me, with file:line at each hop:
1. What actually populates `cache_entry["context_cap"]` that
   `OTR_LedgerScriptWriter._build_truncating_generate_fn` reads
   (nodes/OTR_LedgerScriptWriter.py:647)? Follow the transformers_safetensors
   loader path for `mistralai/Mistral-Nemo-Instruct-2407`.
2. Is `resolve_context_cap` still called on that path, or is it dead code
   superseded by the per-row `context_window` + an effective-limit resolver?
3. Therefore: WHICH value(s) must change to raise the effective cap to 16384 --
   the row field, the override table, the hard limit, or more than one? Give me
   the exact minimal edit set. If your original patch targets a dead path, say so
   plainly.
4. Which tests pin any of those values, and what do they assert?

## Q2 -- does raising the cap break the "C7 audio byte-identity" baseline?

The comment at nodes/_otr_model_catalog.py:1220-1225 says Nemo's 8192 pin exists
to keep "audio byte-identity across B1b" (the C7 regression baseline). I believe
the DEFAULT must stay exactly 8192 so that baseline is untouched, and the 720w
run should opt in via env var. Confirm or refute:
- Does any byte-identity / soak / parity test depend on the effective cap being
  exactly 8192?
- Is an env-opt-in (default unchanged) genuinely safe, or does something cache or
  persist the cap into a ledger/receipt where a changed value would drift?

## Q3 -- VRAM reality at 16k on this box

Your KV math assumed 4-bit weights (~8 GB). Verify against the real loader:
- How is Mistral-Nemo actually loaded (dtype / quantization / device_map)? The
  curated row says `approx_safetensors_gb=24.0`, which is bf16-sized and does NOT
  fit 16 GB, so something must be quantizing or offloading -- find it and name it.
- Recompute the KV-cache cost at 8k / 16k / 24k for the ACTUAL load config, on a
  16 GB RTX 5080 laptop, alongside the rest of the OTR pipeline (the writer LLM is
  unloaded before the media stages, but confirm that).
- Give me a go/no-go for 16384 and for 24576.

## Q4 -- LOOK AHEAD: what else breaks at 720 words, before I burn an hour finding out?

This is the main event. The 30-word canonical lane now publishes. Next is a
720-word bake-off across the media packs, and every previous roll has died on the
NEXT unlogged defect (this is the 12th roll; each one exposed exactly one new
producer-contract gap). Get ahead of it. Read the code and predict the failures.

Specifically hunt for anything that is fine at 13 lines / 30 words and breaks at
~40-48 lines / 720 words:
- Any other fixed/hardcoded `max_new_tokens` in the ladder (P0/P1/P2/P3/P4/P6/P8
  all pass literal values -- which of those inputs GROW with the script and will
  cap out? P6 takes the full script; P8 takes the full script + fact index).
- `_script_artifact_context`, `_script_artifact_inputs`, `_assemble_ledger`,
  `validate_spoken_text_and_roster`, `_validate_script_post` -- anything O(lines)
  that has an implicit ceiling, an advisory-plan assumption, or a beat-count cap.
- `make_advisory_word_blueprint` and the beat_ids formula
  `[f"b{i:03d}" for i in range(max(3, min(12, len(p2.cast) * 3)))]` -- the beat
  count is CAPPED AT 12 regardless of word count. At 720 words that is 60 words
  per beat. Is that coherent with the score/line contract, the music-cue anchors,
  and the per-beat word plan? Does anything downstream assume beats*lines fits a
  bound?
- The media/render tail: frame budgets, clip counts, caption burn, the mux --
  anything that scales with line count or runtime and has a fixed ceiling.
- The Gemini and Sonnet lanes (nodes/_otr_scifi_gemini.py, _otr_scifi_sonnet.py):
  they share the shared writer tail and the content-owned family but have their
  own pass ladders. Do they carry the SAME class of defect that Codex just hit
  (envelope echo, output cap, cast/seed producer gaps)? Name the specific ones.

## Output format (write to agy_review2.md)

Q1 RUNTIME CONTEXT PATH: the traced hops with file:line, and the minimal edit set.
Q2 BASELINE SAFETY: confirm/refute, with the tests involved.
Q3 VRAM GO/NO-GO: the real load config, recomputed KV, go/no-go at 16k and 24k.
Q4 PREDICTED 720W FAILURES: ranked, each as
   <file:line> -- <what breaks at 720w> -- <why it survives at 30w> -- <fix sketch>
   Rank by likelihood of being the NEXT thing that kills the run.
CONFIDENCE: mark each item CONFIRMED (you read it) or [ASSUMPTION].
