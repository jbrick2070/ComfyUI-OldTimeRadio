# r1 judgment -- decode-guard shape (driver: Claude, sole judge)

**Panel obtained: TWO reviewers, the only genuine two-reviewer round of the day.**
Codex (`gpt-5.6-sol`) and Antigravity via the CLI with `KIBITZ_AGY_MODEL="Gemini
3.7 Flash (High)"` -- the slug the operator supplied, which cleared the quota
wall that had blocked agy in the writer-runaway campaign's r1 and r4.

## The decision

**Candidate A (latched StoppingCriteria) SHIPPED, with its trigger REPLACED.**
Candidate B (`no_repeat_ngram_size`) REJECTED permanently. C, D, E cut.
G (prompting) accepted as a PAIRED prevention layer, not a replacement.

## Reviewer claims, grounded

| Claim | Verdict | Evidence |
|---|---|---|
| **B is unsafe: n-gram ban x lmfe mask can empty the allowed set and crash sampling** | **CONFIRMED, both reviewers independently** | Codex traced it in installed transformers: `NoRepeatNGramLogitsProcessor` is added BEFORE `PrefixConstrainedLogitsProcessor`, both write `-inf`, and prefix validation only checks that lmfe returned a non-empty list -- never that a token SURVIVED the earlier ban. Antigravity independently derived the same `torch.multinomial` crash. Decisive; B is out for good |
| **Candidate A's open-string counter is a per-string LENGTH ceiling mislabeled as non-termination detection** (Codex) | **CONFIRMED, and it killed my implementation** | I argued length was the wrong signal and then shipped a length signal. It cannot separate a long field from a loop, which is the operator's whole constraint |
| **The "long string never halts" test was vacuous** (Codex) | **CONFIRMED** | 50 content tokens against a bound of 64 -- it never approached the threshold, so it asserted nothing. Every no-halt test in the rewrite now generates far beyond any bound |
| **The tracker is JSON-specific but installed on every call; an ordinary quotation mark in dialogue opens its state** (both) | **CONFIRMED** | My amendment generalised the INSTALLATION without generalising the SIGNAL. Radio drama is made of quoted speech |
| **"Construction failure must be loud" was false in the code** (Codex) | **CONFIRMED** | The comment sat above a fallback that set the criterion to `None` and ran on. A comment describing a fix is not a fix |
| **`RepetitionTelemetry._seen` never evicts** (Codex) / cut it entirely (Antigravity) | **CONFIRMED** | Unbounded growth on a long decode. Deleted -- the cycle detector IS the repetition measure now |
| **Nothing stock detects non-termination; `StoppingCriteria` is the designated hook** | **CONFIRMED, both** | `MaxLengthCriteria` / `MaxTimeCriteria` measure length and time only. Settles the operator's "is this best practice" question |
| **Adoption cost is not a real objection** | **ACCEPTED, both** | Internal leaf module, no new dependency, no UI surface, nothing user-visible. Antigravity: "zero user friction while preventing 22-minute GPU stalls on consumer hardware" |
| **Transport parity gap: `_otr_model_loader.make_generate_fn` bypasses the guard** (Antigravity) | **CONFIRMED, and later confirmed again by a six-agent verification sweep** | Recorded as explicitly NOT DONE in `9af0f7e2`; being closed now as a separate change |
| **Keep `OpenStringTracker`, bound its domain to JSON, add a companion raw-text check** (Antigravity) | **REJECTED in favour of Codex's F** | Cycle detection subsumes both: it is format-independent, so one detector covers JSON, markup and prose without a domain gate or a second mechanism |

## What shipped (`9af0f7e2`)

Verbatim-cycle detection over token IDs -- three consecutive repeats of a 48+
token run. Length-independent (a 20,000-token non-repeating field is asserted
untouched), format-independent (covers the unconstrained markup lane the lexer
was blind to), tokenizer-independent (nothing decoded, so byte-fallback and
escape handling stop being risks). Smaller than the version it replaced: no
lexer, no telemetry class, no token cache.

## Not adopted, with reasons

* **Antigravity's short-period companion check** -- deferred, not refused. The
  research report independently specifies the same rule (periods 8-63, six
  repeats) and an external practitioner answer proposed a fixed 32-token chunk
  detector, which a direct comparison showed MISSES our measured 384-token
  cycle entirely while catching tight loops ours does not. Three sources now
  converge on this gap; it is the next addition.
* **Reaching into lmfe parser state** -- rejected by the settled design and by
  Antigravity: coupling a liveness contract to third-party internals is how a
  guard silently stops guarding after an upgrade.

## Process note

This judgment exists because the morning's process audit
(`docs/2026-08-13-process-audit-runaway-fix.md`) named "driver artifacts written
for ONE round of four" as the session's largest process failure. Undocumented
grounding is indistinguishable from no grounding -- so the grounding is written
down here, in the round it belongs to.
