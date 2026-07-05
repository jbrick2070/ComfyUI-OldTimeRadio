R1 ANCHOR REVIEW (Claude, code-grounded) -- STAGE1_SUBPLAN.md

VERDICT: yes-with-fixes. The arc is coherent and correctly minimal -- dormant
foundation + passthrough-default + AST byte-identity is the right spine for
"the sci-fi prompts survive." Risk is concept-level gaps calcifying, not the
core approach.

MUST-FIX BEFORE BUILD:
1. [section 3 contract vs section 2 scope] The StoryPack contract carries
   `story_pipeline_id` and the pack references a `visual_style`/pipeline id, but
   Stage 1 ships NO banks/pipelines/visual_styles loader. CONFIRMED against
   `_sibling-archive/src/upstream_story_lab/registry.py` `_cross_validate`
   (lines 142-195) -- that code RESOLVES pipeline/style ids; Stage 1 deliberately
   omits it. Fix: state that in Stage 1 `story_pipeline_id`/`visual_style` are
   OPAQUE validated-as-string fields (no id resolution) so the contract does not
   imply a resolver that isn't built. Otherwise a reviewer/coder will wire the
   cross-validation early and break the "one pack, structure-only" boundary.

2. [section 5 Chunk 1 vs Chunk 2] The pack extracts 7 seams but Chunk 2 wires
   only the `_otr_creative_prompt_router._MODERN_BY_PHASE` seam (outline +
   line_composer_system). CONFIRMED: router covers 2 of the 7. The other 5
   (coda/announcer intro+outro/style inventor+chooser) would sit in JSON with NO
   consumer -- "unwired = dead" (CLAUDE.md rule 0 spirit). Fix: EXTRACT-AS-YOU-WIRE
   -- each chunk adds only the seams it also consumes, so no pack seam is ever
   dead. Grow the pack chunk-by-chunk.

3. [section 4] The BUILD_PLAN "first pack" reuses the `schema-examples`
   `science_news_default.json`, but those `prompt_stages` are PLACEHOLDER text
   ("Preserve the current science/news behavior"), NOT the live bytes. CONFIRMED
   by diff against `_otr_outline.py:532` `_SYSTEM_PROMPT`. The plan already says
   "extract from live constants" -- make it a hard directive: the shipped pack is
   authored from AST-extracted live bytes; the schema-example JSON is SHAPE-ONLY
   and must not be copied as content.

SHOULD-FIX:
1. [section 4 / runtime selection] With no banks.json in Stage 1, what supplies
   the (bank, model, pipeline) triple to `get_pack_prompt_or_none` at the call
   site? It must be a hardcoded science triple at the ONE consuming site until
   Stage 2 routing exists. State this as the single sanctioned temporary constant
   (else the design looks like it silently assumes a router).

2. [section 0 / operator expectation] Operator says "run the NEW workflow when
   all is done." Stage 1 leaves `otr_scifi_16gb_full.json` untouched (correct per
   R4). Clarify in the doc that the NEW workflow surface (source_bank/visual_style
   dropdowns) is Stage 2+; Stage 1 is invisible in the JSON. Manage the expectation.

CUT (scope):
1. [section 3] `forbidden_plot_patterns` / `forbidden_leakage_terms` /
   `tone_guardrails` / `examples` do nothing in Stage 1 (leakage scanning is the
   Stage 2/3 negative-test lane). Keep the FIELDS (cheap schema-completeness) but
   cut any Stage-1 validation beyond "is a list of str." Do not build the leakage
   scanner in Stage 1.

[ASSUMPTION] pydantic v2 import-time cost in the ComfyUI node package is
negligible (news_interpreter already imports it) -- flagged as the plan's own
open Q3; not a blocker.
