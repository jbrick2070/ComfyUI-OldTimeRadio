# Codex Grounded Anchor - Round 4

## Verdict

Revision 3 is build-ready if the implementation preserves four exact
boundaries.

## Final checks against current code

1. The secure network seam already bounds decoded feed/article documents at
   2 MiB. Removing `_fetch_full_article`'s later 12,000-character slice does
   not remove the security bound.
2. `_fetch_science_news` already limits body work to `pool[:5]` and uses
   `ThreadPoolExecutor.map`, so fetching both RSS and linked text remains a
   bounded, order-preserving five-candidate operation.
3. `SourceSpanV4.field` already exists and `_validate_fact_index` already
   validates all fact, entity, and number spans. The new work is window-local
   validation, typed full-text-only rebasing, deterministic merge, and one
   final complete-A0 validation.
4. `StructuredCallFailedError.last_error` directly carries JSON, Pydantic,
   postvalidation, or capacity errors. A public wrapper can classify it before
   it is converted to `CodexPassError`.
5. Installed Comfy's interruption exception inherits `BaseException`. No
   production retry layer may catch `BaseException`; lazy import catches only
   missing-Comfy `ModuleNotFoundError`.
6. `scan_spoken_ledger` consumes ordinary line mappings with `line_id`,
   `speaker_role`, `skip`, and `text`, which `ScriptArtifactV4` can project
   exactly.
7. `_apply_script_safety_cleanup` returns immediately for clean input. Once P5
   rejects unsafe candidates, the accepted path makes no cleanup-model call.
8. `_assemble_ledger`, delivery/authorship stamps, tail freeze/save/reopen, and
   final hash proof are downstream of P5 today and remain outside every retry.

## No hidden scope

The public source payload remains exactly seven strings. All new source
receipts are additive metadata. Helper defaults remain compatible. No
workflow, widget, node, registry, pack, pipeline, schema, frozen ledger, or
snapshot rebaseline is needed. `scifi_news_pro_multipass` remains explicitly
limited by its separate 3,600-character dossier adapter.

## Convergence question

Report only a concrete correctness or safety blocker that survives these
grounded facts. Do not re-propose a fatal fixed outer model-output ceiling; the
operator explicitly chose candidate persistence until acceptance or
cancellation.
