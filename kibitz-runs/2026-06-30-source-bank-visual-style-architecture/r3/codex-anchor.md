VERDICT: yes-with-fixes. The r2 coding plan is implementable, but the workflow wiring must be specified by node/link deltas before code lands.

MUST-FIX BEFORE BUILD:
1. [C6: Wire Visual Style Into Prompt Producers] [CONFIRMED] The canonical graph has `OTR_MetaBriefImagePromptGen` at order 6 and `OTR_ShotLock` at order 14. Style policy must fan out to both; it cannot be chained through MetaBrief or ShotLock without starving one side. Concrete fix: add `OTR_VisualStyleDirector` as an independent policy node and link its output to both appended `visual_style_policy_json` inputs.
2. [C6: Workflow] [CONFIRMED] The workflow currently links `VideoDirector` output to `ImageDirector` and `ShotLock`, and `ImageDirector` output to `MetaBrief` and `ImageGenDispatcher`. Adding a style policy is two new links, not a replacement of existing policy links. Concrete fix: leave links 251/254/257/270 intact; add only style fan-out links.
3. [C1: Writer Source Bridge] [CONFIRMED] `OTR_LedgerScriptWriter` currently has 25 widget values. Appending `source_bank` means the canonical workflow must append one widget value, probably `"science_news"`, and widget-vector tests must be updated. Concrete fix: no insertion before existing writer optional widgets.
4. [C6: Optional Input Append] [CONFIRMED] Existing workflow input order for `OTR_MetaBriefImagePromptGen` and `OTR_ShotLock` is not a thing to hand-edit casually; the order is serialized in the workflow and must match live `INPUT_TYPES`. Concrete fix: use the repo's workflow validator/widget audit after code, and link by input name in any workflow patch helper.
5. [C3: Media Archive Bank] [CONFIRMED] The source-bank selector lives on the writer in v1, so no new upstream source node is wired in C3. Concrete fix: the only workflow change for `media_archive` in v1 is the appended writer widget; runtime branch is inside the writer.
6. [C7: Public Domain Adapter] [CONFIRMED] Public-domain text widgets must not be added until the adapter exists. Concrete fix: C7 is a separate workflow update chunk with its own widget append and tests.

SHOULD-FIX:
1. [C6] `OTR_VisualStyleDirector` should appear near the visual policy cluster, not in the story column. Suggested placement: near `OTR_VideoDirector`/`OTR_ImageDirector`, visually parallel to model policy.
2. [C6] Set `style_id=cinematic_35mm` in the shipped workflow so default behavior is unchanged even when the node is wired.
3. [C6] Since style policy is required for the new style path but should not break old graphs, downstream consumers should treat empty/missing/invalid style JSON as current cinematic behavior plus a warning in report, not a hard failure.

OPTIONAL / NICE-TO-HAVE:
1. [C6] Add a tiny report output to `OTR_VisualStyleDirector` only if useful later; cut it for v1 if it creates extra wiring.

CUT THESE (over-engineering):
1. [C6] Do not wire visual style into `OTR_VideoRenderBatch` in v1 unless R3 confirms render-driver cannot read `meta.visual_style`. The cleaner route is style policy -> prompt producers -> ledger meta -> render driver helper.

