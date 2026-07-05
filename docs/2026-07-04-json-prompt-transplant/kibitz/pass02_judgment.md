# pass02_judgment.md -- r2 judgment log

Format: per-panelist claim, accept/reject/refine, rationale.

## Codex r2 (raw at ``pass02_raw/codex.md``)

| Codex claim | Status | Rationale |
|---|---|---|
| MF1 not code-ready as r2 | ACCEPT | pass02 delivers the 7-chunk plan. |
| MF2 manifest exists pins d48a9d76 | ACCEPT | Verified verbatim. Chunk 0 refreshes to ``a7bdc42d``. |
| MF3 extractor signature not implementable | ACCEPT | Signature rewritten to 4-tuple keyed. |
| MF4 empty-overrides fail validators | ACCEPT | Restated as omit-vs-populate for required seams. |
| MF5 seam count self-contradictory | ACCEPT | Split ``PRODUCTION_TEMPLATE_SEAMS`` (14 after adds) + ``EXPERIMENTAL_PIPELINE_SEAMS`` (4). |
| MF6 byte-identity test too narrow | ACCEPT | Chunk 5 snapshots ASSEMBLED stage strings + adds identity pytest for outline. |
| MF7 scope surgery not applied | ACCEPT | Chunk 6 rewrites anchor sections. |
| SF1 catalogs.py stale reference | ACCEPT | Chunk 6. |
| SF2 SEAM_RUNTIME_VARIABLES misbinds | ACCEPT | Chunk 1 fixes. |
| SF3 extractor error semantics | ACCEPT | Signature spells them out. |
| SF4 Windows regression command | ACCEPT | Chunk 7 pins exact command. |
| CUT compat mirrors + visual + provenance + cross-product + pipeline sim | ACCEPT | Chunk 6 applies MF-C7. |
| CUT adapter + workflow | ACCEPT | Same. |
| CUT experimental 4-pass from prod table | ACCEPT | Split into ``EXPERIMENTAL_PIPELINE_SEAMS``. |

## Fable r2 (raw at ``pass02_raw/fable.md``)

| Fable claim | Status | Rationale |
|---|---|---|
| MF-R2-1 pick DEFER for MF-C1 | REFINE | Not deferred entirely; MF-C1 downgraded to "outline-only + one identity pytest" (Chunk 5). Extraction still happens but with the identity guarantee locked. |
| MF-R2-2 line_grounding breaks empty-override twice | ACCEPT | line_grounding extraction DEFERRED to Phase B (r2 decision). |
| MF-R2-3 extractor must go through resolve_profile | ACCEPT | Chunk 3 wraps resolve_profile, does not bypass. |
| MF-R2-4 production_mirror empty | REFINE | Fable's glob returned empty for the SUBDIR (correct); Codex found the MANIFEST file at the sibling top level. Both correct; manifest confirms drift. |
| SF-R2-1 macro/phase/beat extraction-safe (no identity check at consumers) | ACCEPT | Documented in seam-name map. |
| SF-R2-2 outline_system maps to LEGACY :532 (superseded by staged) | ACCEPT | Chunk 5 pins identity for the legacy site specifically. |
| SF-R2-3 count language: 14 vs 18 | ACCEPT | ``PRODUCTION_TEMPLATE_SEAMS`` becomes 14 (10 + 4 adds); ``EXPERIMENTAL_PIPELINE_SEAMS`` stays 4 separate. |

## Sonnet r2 (raw at ``pass02_raw/sonnet.md``)

| Sonnet claim | Status | Rationale |
|---|---|---|
| #1 MF-C1 factually wrong for line_composer | ACCEPT | Corrected in pass02 MF-C1 section. |
| #2 extractor signature vs None-vs-""" | ACCEPT | Signature pins ``None`` reserved for intentional empty override. |
| #3 MF-C6 not universally safe (required_seams) | ACCEPT | Corrected in pass02 MF-C6 section. |
| #4 real constant names invalidate invented seam names | ACCEPT | Corrected in pass02 MF-C3 seam-name map. |
| #5 router docstring stale (4 phases claim) | ACCEPT | Chunk 6 doc-hygiene note. |
| MF-R2-1 correct MF-C1 premise | ACCEPT | Done in pass02. |
| MF-R2-2 restate MF-C6 with required-seams awareness | ACCEPT | Done in pass02. |
| MF-R2-3 replace invented outline sub-seam names | ACCEPT | Done in pass02. |
| MF-R2-4 re-pin MF-C5 against actual HEAD | ACCEPT | Chunk 0 refresh. |
| SF pin extractor return contract to None not "" | ACCEPT | Signature explicit. |
| SF flag stale docstring | ACCEPT | Chunk 6 note. |

## Cross-panel refinement summary

**Load-bearing r1-to-r2 corrections:**

1. MF-C1 identity check is OUTLINE-only (Sonnet + Fable + Codex all
   confirmed line_composer does direct-assign, not identity check).
   Downgrade from "load-bearing" to "one pytest".
2. MF-C3 seam names must be real constants (``_MACRO/_PHASE/_BEAT_SYSTEM_PROMPT``),
   not invented ``outline_macro_system`` etc. lab seam names are OK
   for lab schema; production constant names are what tests grep for.
3. MF-C4 signature is 4-tuple keyed, wraps ``resolve_profile()``, and
   distinguishes ``None`` (intentional passthrough) from ``""``
   (populated content that happens to be empty -- currently disallowed
   for required seams).
4. MF-C5 baseline manifest exists (Codex) but subdir empty (Fable) --
   both correct; manifest pins d48a9d76 (2 days behind OTR HEAD
   a7bdc42d). Chunk 0 refreshes.
5. MF-C6 empty-override rule is required-seams-aware, not universal.
   Restated with omit-vs-populate distinction.
6. MF-C7 scope surgery must actually rewrite anchor sections in Chunk
   6, not just annotate.

**Panel diversity dividend:**

- **Codex** grounded the manifest existence and the 3-tuple pack
  keying; delivered the strongest signature critique.
- **Fable** grounded the line_grounding f-string blocker and the
  consumer-side non-identity for macro/phase/beat; "DEFER-is-free" insight.
- **Sonnet** grounded the OUTLINE-only identity fact (single most
  important r1-to-r2 correction) and the required_seams empty-string
  fail-loud pattern.

## Rejected / deferred to Phase B

- ``line_grounding`` extraction: DEFERRED to Phase B (Fable MF-R2-2).
- ``_otr_creative_prompt_router.py:15-19`` 4-phase docstring cleanup:
  doc-hygiene, noted for coder window, NOT in Phase A code.
- Compat mirrors, visual policy, provenance stamping, cross-product
  matrix, pipeline sim, adapter, workflow edits, runtime widgets: all
  Phase B (MF-C7 applied via Chunk 6 rewrites).

Zero panel claims rejected. Every correction is ground-truthed.
