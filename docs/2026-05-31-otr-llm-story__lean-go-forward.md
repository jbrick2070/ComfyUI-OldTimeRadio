# Lean go-forward — a good story via OpenRouter **or** local LLM (defects-first, no overhaul)

**Date:** 2026-05-31 · v2.0-alpha · Synthesis of the two-path remediation doc + the live diagnosis/fix.

## Principle
Local Mistral-Nemo already writes a **B+** story. **Do not overhaul** the call structure, prompts, schemas, or ledger. Make the *remote* path enforce mechanically what the local model gave for free, fix the cheap backend defects, then **prove parity** (local and remote produce consistent, good stories). Rewrite (write-then-extract) **only if reading the output proves** forced-JSON flattens quality — never on theory.

## Already fixed — the real blocker (closed)
Remote output was **truncated**, not prose. `max_tokens` was clamped to the writer's local grammar-era per-call budget (~200), which only works locally because lm-format-enforcer forces a compact bare object; a free-form remote model got cut mid-JSON → `finish_reason=length` → "char 0" abort. **Fix:** 1024-token output floor (`OPENROUTER_MIN_OUTPUT_TOKENS`) + truncation warning. **Verified live:** Opus now writes cast + outline + dialogue end-to-end and reaches TTS. Shipped (BUG-LOCAL-294, full suite green).

## Remaining Path-1 hardening (lean, no rewrite)
Each is a small backend/wiring change; local stays byte-identical.

1. **[HIGH] `provider: {require_parameters: true}`** on the remote payload — so OpenRouter only routes to upstreams that honor `response_format`; otherwise structured-output enforcement can be a silent wire no-op. Pair with #2.
2. **`response_format: {type: json_object}`** on the remote *creative structured* calls (casting / outline / news) — belt-and-suspenders so correctness no longer leans on fence-stripping or the model avoiding a prose preamble. Pydantic + the repair ladder still enforce shape. **Not** strict `json_schema` first (it 400s on unsanitized schemas).
3. **`_extract_text` robustness** — tolerate list-of-parts `content` (join text parts) + fall back to `reasoning`. Turns a silent-abort class into recoverable text. (finish_reason=length warning already added; add an opt-in raw-response debug log for future anomalies.)
4. **Cost-guard defaults** — per-call ceiling must sit above the output cap so a near-cap reply can't spuriously abort (mitigated in the launcher; fix the code default too).
5. **Verbosity vs schema caps** (the only content issue seen) — Opus overran `character_description` (750 chars); the repair ladder recovered but burned a call. Raise the cap *or* add "be concise (≤N)" to the affected prompts. Tuning, not a blocker.

## The parity test (the confirmation you want) — two cheap runs, **read** the output
- **Wiring parity (isolates wiring from model):** run the **same model both ways** — local Mistral-Nemo vs the same model via OpenRouter — identical premise + seed. Consistent story content ⇒ the remote wiring is clean; any later difference is model choice, not a bug.
- **Quality parity:** local Mistral (the B+ baseline) vs remote Opus, identical premise + seed. Read both ledgers/scripts side by side.

## Decision gate
- Remote yields a correct, schema-valid ledger **and** quality ≥ local B+ → **STOP. Path 1 is the whole answer. No rewrite.**
- **Only** if reading shows forced-JSON measurably flattened Opus vs its free-form ability → write-then-extract (remote writes prose, local grammar extracts), gated on that observed evidence.

## Rejected
Preemptive architecture rewrite · strict `json_schema` before `json_object` · prompt overhaul · any change to the B+ local path · change-log churn.

## Open (minor, won't block)
- Which model for the wiring-parity run — suggest **Mistral-Nemo on both** (it's available on OpenRouter), so it's a pure local-vs-remote wiring compare.
- #5: raise the cap vs prompt for brevity — your call.
