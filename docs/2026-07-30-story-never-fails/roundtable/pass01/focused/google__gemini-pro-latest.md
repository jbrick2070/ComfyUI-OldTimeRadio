<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview -->

VERDICT: no. The plan conflates the validation pass with the authoring pass, breaking the pipeline's core architectural boundaries, and introduces unbounded subjective loops.

MUST-FIX BEFORE BUILD:

1. [Section 4 & 5] **P5 vs P3 responsibility mismatch.**
   *Defect:* The plan states "P5 retries P5", "P5 safety or spoken-surface rejection returns to P5", and requests a "P5 rewrite for concrete craft defects." The grounding explicitly defines P3 as the writer ("The final P3 writer receives compact P0 facts") and P5 as the "compact draft validator." P5 cannot rewrite the story to fix spoken-surface or safety defects; it only reports them.
   *Fix:* Route story-level rejections (safety, spoken-surface) back to P3. P3 owns the prose artifact and must execute the repair cycle.

2. [Section 3] **`scifi_news_pro` merge compatibility assumption.**
   *Defect:* The plan mandates "Give scifi_news_pro the same complete-source window/merge discipline for its dossier". The grounding explicitly warns that `scifi_news_pro` "shares the common RSS fetcher but not the canonical Codex FactIndex pipeline." You cannot blindly apply a P0 FactIndex merge strategy to an architecture that does not use that schema.
   *Fix:* Specify a bespoke map-reduce/merge strategy tailored to `scifi_news_pro`'s specific dossier schema, or explicitly exclude it from the multi-window merge and only provide it the full text to handle internally.

3. [Section 3] **Duplicate entity/fact accumulation during merge.**
   *Defect:* "A merge model returns candidate IDs only; Python copies and renumbers already accepted rows." If adjacent windows extract the same entity (e.g., "NASA"), blindly copying accepted rows will result in duplicate entities, violating the `max_length=4` bound.
   *Fix:* Define a deduplication strategy for entities and facts during the Python merge phase before enforcing the final schema bounds.

4. [Section 3] **Missing foreign-key remapping during merge.**
   *Defect:* The plan states Python "renumbers already accepted rows" and relies on tests to catch "broken number-to-fact references", but fails to specify the mechanism.
   *Fix:* Explicitly mandate that when facts are renumbered during the merge, the `fact_id` references inside the `NumberV4` objects must be programmatically remapped to match the new IDs.

SHOULD-FIX:

1. [Section 3] **2 MiB payload [ASSUMPTION].**
   *Defect:* "Raise the 48,000-byte payload admission ceiling only to the common bounded fetch envelope". The grounding states the fetch envelope is 2 MiB. This assumes the ComfyUI graph, JSON serialization, and downstream nodes can handle a 40x increase in payload size without memory exhaustion or database bloat.
   *Fix:* Cap the A0 payload at a safer intermediate bound (e.g., 256 KB) or explicitly verify ComfyUI/ledger memory limits for 2 MiB payloads.

2. [Section 5] **Contradictory ledger rebuild logic.**
   *Defect:* "Once a story candidate is structurally valid, assemble a fresh ledger... If a gap audit identifies an authored field defect, route the complete finding to the pass that owns that field". If the candidate is already structurally valid, a gap audit shouldn't be finding authored field defects at the ledger assembly stage.
   *Fix:* Clarify that the gap audit occurs *before* final ledger assembly, during the validation passes, ensuring the ledger is only assembled from a fully verified artifact.

OPTIONAL / NICE-TO-HAVE:
- In Section 2, when choosing the "richest clean body", explicitly define the heuristic for "richest" (e.g., highest character count after HTML stripping) to avoid non-deterministic selection.

CUT THESE (scope / over-engineering):

1. [Section 5] "Add a story-quality review that can request a source-grounded P5 rewrite for concrete craft defects."
   *Why:* Subjective "craft defect" review is an unbounded LLM critique loop that has nothing to do with the stated goal of "Ledger-Until-Valid" (which is about structural/technical validity). It introduces massive scope bloat and risks infinite subjective retry loops. Cut entirely.
