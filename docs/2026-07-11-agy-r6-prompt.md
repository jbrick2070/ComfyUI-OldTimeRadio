# AGY R6 -- the truncation sweep (paste this whole file into agy)

REVIEWER ONLY. Do not edit source, do not git add/commit/push. Write to
`agy_review6.md` and stop. Read the real files. Label every claim CONFIRMED (you
opened it) or [ASSUMPTION]. Show your arithmetic.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha  HEAD: bda9186e

## What actually killed Gemini P3 -- and why all of us missed it

Three rolls in a row, Gemini P3 came back missing the same nested fields. We kept
concluding the model was ignoring instructions. We were all wrong, including you and
including me. From the server log:

    PROMPT_GUARD: Truncated 5408 -> 4592 tokens (context_cap=8192, max_new_tokens=3600)

P3 reserved a FLAT 3600 output tokens against an 8192 cap. That leaves
8192 - 3600 = 4592 tokens of input. The typed-repair prompt (failed artifact +
validation error + original request) is 5408 tokens. `_build_truncating_generate_fn`
LEFT-truncates, so the front of the prompt -- the system message, the schema, the
repair rules -- was sliced off every single repair call. The model never received the
instructions we kept "improving." The repair was doomed before it was sent.

Fixed at `bda9186e`: the P3 reservation now scales from the artifact's real cost
(word steer + per-beat metadata), and P3 sets `prompt_must_fit=True` so this fails
LOUD instead of silently.

**The lesson for you: the bug was arithmetic in a log line, not logic in the code.**
You audited my code and said it held -- and it did -- while the actual killer sat in
plain sight in a WARNING. Read the numbers, not just the source.

## JOB 1 -- sweep EVERY structured call in EVERY lane for this class

This is the whole assignment. Nothing else matters as much.

`context_cap` is 8192 (CONFIRM this for the transformers path). For every call site,
`max_input_tokens = 8192 - max_new_tokens`. If the prompt exceeds that, it is
SILENTLY left-truncated and the pass is running blind.

Build me a table. One row per structured-call site across:
- `nodes/_otr_scifi_codex.py` (P0..P9, including P3_rewrite)
- `nodes/_otr_scifi_gemini.py` (P0..P6, including the per-scene P4/P5/P6)
- `nodes/_otr_scifi_sonnet.py` (P0..P6, including per-line P2a/P2b and the warden loop)

Columns:
1. pass_id, file:line
2. `max_new_tokens` (literal or the function that computes it)
3. resulting `max_input_tokens` (8192 - reservation)
4. estimated BASE prompt tokens at 30 words -- and how you estimated it
5. estimated TYPED-REPAIR prompt tokens (ALWAYS bigger: it carries the failed
   artifact AND the validation error AND the original request -- read the actual
   factory for that lane to see what it packs in)
6. VERDICT at 30w: FITS / TRUNCATES
7. VERDICT at 720w: FITS / TRUNCATES
8. does it set `prompt_must_fit=True`? (i.e. would it fail loud or lie to us?)

Rank by "most likely to silently truncate next." I want to fix this class ONCE,
everywhere, rather than discover it one 15-minute render at a time.

Then answer:
- Which passes need a scaled reservation (like Codex's `_script_output_token_budget`
  and Gemini's new `outline_output_token_budget`) rather than a flat literal?
- Which passes should set `prompt_must_fit=True`? Is there any pass where silent
  truncation is genuinely SAFE and desirable? Argue it if so.
- The typed-repair prompt is the fat one everywhere. Should the repair factories stop
  echoing the full `original_request` when the failed artifact already contains the
  content? What is the minimum a repair prompt must carry to be correct?

## JOB 2 -- your Sonnet findings, hardened

From your R5 you claimed, and I want these NAILED DOWN with file:line and exact quotes
before I touch code:
1. The P5 rewrite loop computes corrected lines and never merges them back into
   `events`, so the re-audit re-reads the SAME text. If true this is a serious logic
   bug -- quote the loop and show me the missing write-back.
2. `AttestationV4.attestation_cites` allows max_length 4 but `DraftLineV4.cites`
   allows 3, so a 4-cite reply raises on construction.
3. The lane hardcodes `fact_0` while the P0 contract mandates 1-indexed `fact_1`..N.

For each: is the fix MECHANICAL (derivable -> Python may repair it) or CREATIVE
(authored -> only the model may write it)? Python judges, the LLM writes. A fix where
Python invents story content is an automatic reject.

## Output (agy_review6.md)

JOB 1 TRUNCATION TABLE: the full table, then the ranked fix list.
JOB 2 SONNET: the three findings confirmed or retracted, with quotes.
CONFIDENCE on every row.
