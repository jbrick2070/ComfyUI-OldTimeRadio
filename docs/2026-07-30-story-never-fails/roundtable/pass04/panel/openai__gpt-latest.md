<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.6-sol -->

VERDICT: build-ready as-is? yes-with-fixes — implementation is blocked by undefined RSS envelope/receipt bounds and nondeterministic preview, window-merge, and receipt semantics.

MUST-FIX BEFORE BUILD:
1. [A0 admission and P0 windows] “Ratified serialized envelope bound” has no numeric value or measurement rule. Define the exact maximum serialized UTF-8 byte count for the normalized seven-field RSS A0 envelope, state that oversize is a loud deterministic source/admission failure, and distinguish it explicitly from the pinned-A0 48,000-byte limit.
2. [Common source acquisition] Source selection and receipts are not fully deterministic. Specify that “longest” means normalized character count; define the exact head/middle/tail allocation and offsets for the 800-character preview; define route enum values; make RSS index zero-based; and define `body_sha256` as SHA-256 of the selected normalized body encoded as UTF-8. Without this, builders can produce different rankings and ledger metadata.
3. [A0 admission and P0 windows] Overlap behavior is underspecified and can stall or produce incompatible offsets when sentence-boundary shortening yields a window no longer than the overlap. Define the advance formula, require positive forward progress, define behavior when `allowance <= overlap_chars`, and state whether the returned offset is always the exact complete-A0 coordinate of `window[0]`.
4. [A0 admission and P0 windows] Local versus complete-A0 validation is ambiguous. Specify that span containment is checked against each window before rebasing, `payload_sha256` remains the complete normalized A0 digest rather than a window digest, only typed `full_text` spans are rebased, and complete-A0 validator state is installed and cleared in `finally` on success, failure, and cancellation.
5. [A0 admission and P0 windows] The merge algorithm does not define one reproducible capped result. Specify exact bundle equality fields, processing order, whether fact IDs are canonicalized before number deduplication, the even-sampling formula and tie handling, and how dependent numbers are dropped/remapped when facts are capped. Require final schema limits of facts ≤6, entities ≤4, numbers ≤4, contiguous IDs, valid number parents, and at least one fact.
6. [Fresh candidate campaign] The “maximum bounded outer-retry receipt” cannot be reserved because no bounds are given. Define exact maximum UTF-8 bytes/characters for collapsed rejection and the complete `writer_retry` mapping, including truncation/collapse rules. Name the exact `StructuredCallFailedError.last_error` classifications treated as JSON, Pydantic, postvalidation, and rerollable `output_limit`; every other exception must remain loud.
7. [Fresh candidate campaign] “Cycle one is byte-identical to the old prompt” conflicts literally with the new windowed P0 inputs. Clarify that cycle one is byte-identical to the newly constructed base prompt for that invocation, with no `writer_retry` mapping; it cannot mean byte-identical to pre-change production P0 prompts.
8. [Final story and ledger] Define the P5 safety gate concretely: project every `ScriptArtifactV4` row to `line_id`, `speaker_role`, `skip`, and `text`; run the existing spoken-ledger scanner; aggregate all hard graph/roster/markup/empty/audibility/spoken-safety findings into structured postvalidation; and retire the candidate on any hard finding. Warnings must remain nonblocking.

SHOULD-FIX:
1. [Common source acquisition] State explicitly that removing `_fetch_full_article`’s 12,000-character slice does not change the HTTPS-only 2 MiB decoded-response security boundary. This prevents an implementor from adding a second, inconsistent 2 MiB character slice.
2. [A0 admission and P0 windows] Replace “field allowlist is computed once” with the concrete authority: derive it from the existing legal span-field set for the normalized seven-string A0 payload and reuse the immutable result across all windows.
3. [Scope and proof] “No … pipeline … changes” is misleading because canonical P0 window/merge execution and candidate retry control flow are changing. Narrow this to no public workflow/node/widget/link/registry/schema/prompt-pack/frozen-artifact change and no separate Pro-runner implementation.
4. [Final story and ledger] State that canonicalization, cleanup, exact-representation validation, assembly, stamping, freeze/save/reopen, and proof are all outside every candidate retry loop. The current ordering implies this but does not explicitly protect against accidental relocation.

OPTIONAL / NICE-TO-HAVE:
- [Scope and proof] Add a compact table mapping each focused test to its invariant and expected failure class; this would make mutation coverage easier to audit.

CUT THESE:
1. None — after clarifying the scope sentence, the remaining mechanisms directly support complete-source coverage, candidate liveness, or final-ledger integrity.

VERIFY-AT-BUILD checklist:
[ASSUMPTION] Earlier review reports were not supplied, so this checklist covers the source-dependent and previously unverified boundaries exposed by the grounding material.

- [ ] [Common source acquisition] Confirm the network seam remains HTTPS-only and capped at 2 MiB decoded bytes.
- [ ] [Common source acquisition] Confirm `_fetch_full_article` no longer clips extracted text at 12,000 characters; test evidence beyond that former boundary.
- [ ] [Common source acquisition] Confirm body resolution still operates only on the existing first five headline-ranked candidates, preserves candidate order, and fetches each linked article even when RSS content exceeds 300 characters.
- [ ] [Common source acquisition] Confirm every mapping-valued RSS `content` row is considered, malformed/non-mapping rows are handled deterministically, and earliest raw index wins equal-length ties.
- [ ] [A0 admission and P0 windows] Confirm the public source payload remains exactly seven strings and the legal span fields come from the existing span-field authority.
- [ ] [A0 admission and P0 windows] Confirm overlap rebasing adjusts both span bounds only for `full_text`, and final validation checks every fact, entity, and number span against complete normalized A0.
- [ ] [A0 admission and P0 windows] Confirm complete-A0 validator state is cleared after success, model failure, deterministic failure, and cancellation.
- [ ] [Fresh candidate campaign] Confirm `StructuredCallFailedError.last_error` actually exposes each classified recoverable error without first being converted to `CodexPassError`.
- [ ] [Fresh candidate campaign] Confirm cancellation is polled at cycle and model-call boundaries, propagates by identity, and no retry layer catches `BaseException`; any lazy Comfy import fallback must catch only `ModuleNotFoundError`.
- [ ] [Fresh candidate campaign] Confirm configuration, prompt-pack, source-security, provider, filesystem, freeze, and proof failures cannot enter the fresh-candidate loop.
- [ ] [Fresh candidate campaign] Confirm each finite ladder writes exactly one journal entry, each later cycle has a unique nonce, and neither rejected raw output nor abandoned prose enters prompts or ledger metadata.
- [ ] [Final story and ledger] Confirm the spoken scanner accepts the projected P5 row mappings and that unsafe output is rejected before assembly.
- [ ] [Final story and ledger] Confirm clean accepted P5 input causes `_apply_script_safety_cleanup` to return without a cleanup-model call.
- [ ] [Final story and ledger] Confirm `_assemble_ledger`, delivery/authorship stamping, freeze/save/reopen, and final hash proof execute exactly once and remain outside all retry loops.
- [ ] [Scope and proof] Confirm `scifi_news_pro_multipass` remains limited by its separate 3,600-character dossier adapter and is not represented as complete-source capable.
- [ ] [Scope and proof] Confirm the canonical workflow file, frozen ledgers, and snapshots remain byte-identical, followed by the stated full Windows suite and HEAD/origin equality checks.