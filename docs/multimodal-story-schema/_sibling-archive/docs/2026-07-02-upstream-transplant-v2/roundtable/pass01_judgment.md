# Roundtable pass01 judgment (gap-hunt; modified single pass by request)

Panel: x-ai/grok-4.3, moonshotai/kimi-k2.6, deepseek/deepseek-v4-pro (retry
with max-tokens 8000 after reasoning-token exhaustion). New families vs the
local kibitz panel, as requested. Spend: ~$0.0465 (main) + ~$0.02 (retry).
Reviews: pass01/ + pass01_deepseek_retry/. Anchor: pass01/claude_anchor.md.

## ACCEPTED and APPLIED (verified against the tree; 50 tests green)

1. Runner context vacuum + no output chaining (DeepSeek M2, Kimi F3,
   Grok critique - CONFIRMED in runner.py): run_pipeline now injects
   SOURCE MATERIAL + TONE GUARDRAILS + LEDGER SCHEMA and chains each pass's
   output into the next; tests pin both. simple_4 pass prompts rewritten
   with hard limits and explicit schema references.
2. Executable pipeline seam coverage never cross-validated (DeepSeek F3 -
   CONFIRMED): registry now hard-errors when a pack does not cover an
   executable pipeline's seam_refs; descriptive pipelines exempt by
   documented contract. Test added.
3. Motion prompts noun-heavy/static (all three + driver anchor -
   CONFIRMED): archival_documentary motion prompts rewritten verb-forward
   (steadicam push, rack focus, tracking drift, rotation dynamics).
4. Mirror-hash verification absent (Kimi F5 - HALF-CONFIRMED: the manifest
   already stores per-file hashes, Kimi missed that; verification was
   indeed missing): drift suite now refuses a tampered or partially
   refreshed mirror (test_mirror_files_match_manifest_hashes).
5. Pack prose fixes (Grok F4 + Kimi + DeepSeek critiques - CONFIRMED prose
   defects): faithful_radio_adaptation pitch contradiction fixed
   (compression variants, identical premise) + compression recipe;
   gentle_thriller non-violence constraint moved to the front of the first
   creative pass; media_restoration human-choice resolution line.
6. Operator tooling absent (Grok F2 + Kimi absent#3 + driver anchor -
   CONFIRMED): tools/new_pack.py (refuses duplicates, forces bank/pipeline,
   clone mode warns about stale leakage terms) +
   docs/PACK_AUTHOR_CHECKLIST.md incl. the compat re-pin recovery guide
   (Grok F3 / DeepSeek absent#1).

## REJECTED, with reasons

- Inject forbidden_plot_patterns into prompts (Kimi F1 fix, DeepSeek F1
  fix): violates the locked negation-copy rule (models copy negated terms;
  the rule is production-proven). Tone guardrails ARE injected (positive
  constraints); forbidden patterns stay post-generation scan metadata. A
  test now pins that they never enter runner prompts. Also PARTIAL MISREAD:
  the transplant path already threads forbidden_plot_patterns to
  OutlineRequest for production to use as it sees fit.
- Scene/beat-count hard limits in outline_system for legacy_many_pass packs
  (Kimi critique): production's outline machinery owns structure
  (phases/beats); pack-imposed counts would fight it. Limits WERE added to
  simple_4 pass_1, where the lab runner owns structure.
- Require all motion keys per style (Kimi F6): empty/missing = "production
  table stays" is the documented default-passthrough contract;
  anime/cartoon/origami intentionally declare none.
- Shared media_archive "dramatic core" label (Grok critique): the five
  models are deliberately distinct lanes; homogenizing their registers
  reverses the design (and contradicts Kimi's own samey-episode worry).
- Retire simple_4's executable flag (DeepSeek do-not-do): with context
  injection + chaining + schema-aware prompts it is now a real experiment;
  it remains lab-only and never a fallback.

## VERIFY-AT-BUILD (carried to GO_FORWARD/T1)

- Attach a real production ledger example as ledger_schema_text when first
  running simple_4 against a real local model (fixture lives production-
  side; copy one generated episode ledger during T1).
- A/B the rewritten motion prompts against the production table style at
  the visual stage before adopting them for renders.
- One real-LLM smoke of simple_4 (Grok/DeepSeek absent-item) once T1 lands.

## Deviation from the skill's 4-round arc

Single gap-hunt pass by explicit user request ("modified roundtable...
another view... pick new models"); the 4-round hardening already ran
tonight via the local kibitz arc. Custom gap-hunt prompt used; findings
ledger included in the briefing so the panel hunted NEW ground only.
