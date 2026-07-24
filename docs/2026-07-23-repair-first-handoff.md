# Repair-first LLM handoff hardening plan

## Scope

Harden `PBUG-20260722-02` and the shared structured-call boundary so a
recoverable model-output or semantic validation failure is handed to a
bounded alternate repair owner instead of terminating the workflow after the
first slot's ladder. Claude is explicitly excluded from this review. The
remote first-pass reviewer is Qwen3 Coder on the RTX 4060 at
`http://10.55.0.2:1234/v1`; the RTX 5080 remains reserved for ComfyUI.

## Grounded current state

- `nodes/_otr_structured_call.py` owns the current JSON/schema/post-validation
  ladder and raises `StructuredCallFailedError` when its owners are
  exhausted.
- `PostValidationError` is intentionally recoverable, while provider and
  other non-recoverable exceptions propagate.
- `nodes/_otr_scifi_codex.py` invokes P0 through `invoke_codex_structured`
  with a technical slot and a literal source-span post-validator.
- P0's validator requires each quote to equal the selected source slice and
  preserves the normalized source digest. The current open production defect
  is that the bounded technical repair can still return a non-literal quote.
- `_otr_scifi_p0_contract.py` already provides bounded repair context, but the
  model contract must make `payload[field][start:end] == quote` explicit.

## Proposed design to pressure-test

1. Keep the existing primary ladder and its exception taxonomy intact.
2. Add an optional, caller-supplied repair owner to the shared helper. It is
   invoked only after a recoverable ladder exhaustion and only when the caller
   supplies a bounded repair-ledger builder.
3. The repair ledger contains the failed artifact, validator rejection,
   normalized source evidence/digest, allowed fields, and a fresh repair
   nonce. It is not accepted as output; the original post-validator remains
   authoritative.
4. The alternate owner gets a finite repair budget. It may return a fresh
   schema-valid artifact, which must pass the same post-validator before
   acceptance. Exhausted repair owners produce an explicit terminal receipt.
5. Provider/backend failures, impossible context or schema, safety/rights
   failures, and missing source/assets remain fail-closed. No retry loop may
   recurse into itself.
6. P0 wires the creative slot as its alternate repair owner. If both slots
   resolve to the same backend, the receipt must still identify the distinct
   owner/rung and nonce; no claim of backend diversity is made.

## Acceptance evidence

- Focused tests cover primary post-validation failure -> repair ledger ->
  alternate owner -> accepted literal span; exhausted alternate owners;
  provider exceptions; digest preservation; and no infinite recursion.
- The P0 contract explicitly states literal slice identity and the compact
  repair context contains the failed artifact, rejection, digest, and allowed
  source fields without unbounded prompt growth.
- The canonical workflow remains valid after a JSON round-trip, with intact
  links, widget positions, and live `INPUT_TYPES` names.
- Re-run canonical `scifi_news` at 120 and 320 words. A pass requires an
  explicit terminal `RESULT SUCCESS`, `obs_publish OK`, exact episode/OBS
  assets, literal-span equality, and preserved payload digest.
- The harness must not call a child exit code `PASS` when its log lacks the
  terminal success sentinel.

## Review status

The four-round repair-first KIBITZ review is complete. Codex is the grounded
judge; Qwen3 Coder on the remote RTX 4060 completed all four scoped reviews;
Claude was excluded. Antigravity completed R1 and then hit a confirmed quota
hold before R2. The shared repair-owner path and the P0 literal contract are
now implemented in the working tree. Live 120/320 qualification remains open.

## Converged build gates

- Make the literal P0 identity explicit first:
  `payload[field][start:end] == quote`.
- Add a hard byte/token limit and tagged-data encoding to the P0 repair
  context. Preserve the digest, allowlist, failed artifact, and rejection.
- Add an optional, non-recursive shared repair-owner branch with exactly one
  bounded alternate attempt by default. Reuse the original post-validator.
- Pass repair owner/backend identity from the writer through
  `invoke_codex_structured`; record rung, owner, backend, nonce, hashes, and
  final disposition in the existing journal. Same-backend fallback is visible
  but is not distinct-LLM proof.
- Keep provider, safety, rights, impossible-context/schema, missing
  source/assets, and exhausted-owner cases fail-closed.
- Focused implementation tests (69 passed), canonical workflow/link/widget
  audits (48 passed / 2 skipped), AST compilation, and Bug Bible regression
  are green. The full suite is being rerun after the authored-text contract
  correction.
- Run focused tests, workflow JSON/link/widget audits, then canonical 120-word
  and 320-word proof requiring structured status, `RESULT SUCCESS`,
  `obs_publish OK`, exact assets, literal spans, and digest preservation.
