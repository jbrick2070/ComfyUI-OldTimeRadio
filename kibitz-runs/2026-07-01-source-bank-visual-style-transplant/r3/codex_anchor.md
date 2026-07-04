# R3 Codex Anchor Review

- VERDICT: yes-with-fixes. The build order is now plausible, but the transplant
  must be split into source/story wiring and visual/video wiring. Treat visual
  prompt extraction as a staged high-risk integration, not a single clean node.

## MUST-FIX BEFORE BUILD

1. [R2 C7 / upstream_story_lab] CONFIRMED: the isolated lab must stay out of
   production imports. Add a gate that no file under `nodes/` imports
   `upstream_story_lab`. The lab can hold fixture JSON and manifests only until
   transplant.

2. [Transplant sequence] CONFIRMED: source/story integration must precede
   workflow JSON edits. Safe sequence:
   - lab fixtures and pure schemas
   - production pure modules under `nodes/`
   - pure prompt preview tests
   - writer/runtime integration behind default `science_news`
   - only then append workflow widgets and validate the canonical JSON
   Editing `workflows/otr_scifi_16gb_full.json` before runtime code exists is a
   repeat of the "unwired code is dead" failure mode in reverse.

3. [Writer entry point] CONFIRMED: R2 deferred writer signature changes, but R3
   must choose the exact transplant point. Add appended optional inputs to
   `OTR_LedgerScriptWriter.INPUT_TYPES` and matching `run()` / `_resolve_inputs`
   params only in the transplant chunk:
   - `source_bank`
   - `story_model`
   Do not add `source_text_path` until `public_domain_story` has a real adapter.

4. [Headless/API parity] CONFIRMED from repo rules: any new writer widget must
   also be added to workflow apply/API creative whitelists in the same
   transplant chunk. R3 must name the files/tests for that parity, not just the
   workflow JSON.

5. [Visual policy wiring] CONFIRMED: the likely clean wiring point is not a new
   process-global singleton. Use a `visual_style_policy_json` forceInput socket
   only when adding the visual director node. Append/read the policy through:
   - MetaBrief `generate(..., visual_style_policy_json="{}")`
   - ShotLock `lock(..., visual_style_policy_json="{}")`
   - shared helper reads from ledger/meta before prompt finishing
   Bad/missing policy in wired path must fail visibly.

6. [Deep visual prompt risk] CONFIRMED: some prompts are deep in
   `_otr_video_engines/render_driver.py` and current still/video repair code.
   R3 must split visual transplant:
   - V1: policy catalog + seam readers + preserve `sci_fi_radio`
   - V2: MetaBrief/ShotLock stamps and forceInput wiring
   - V3: render-driver motion/fallback prompt replacement
   Do not pretend one node overrides every downstream prompt.

7. [Workflow validation] CONFIRMED: transplant chunk must run:
   - `OTR_WorkflowValidator`
   - JSON round-trip
   - link referential integrity audit
   - widget-count vs live `INPUT_TYPES`
   - forceInput sockets have no `widget` sub-key and consume no slot

## SHOULD-FIX

1. [Node registration] `OTR_VisualStyleDirector` registration must be in root
   `__init__.py` `_NODE_MODULES`, but only in the visual transplant chunk. Pure
   helpers are not registered nodes.

2. [Visual policy id] Workflow default should be `sci_fi_radio` until a byte- or
   behavior-stability test proves another style preserves current output.

3. [Source bank exposure] Workflow dropdown should expose only implemented
   source banks. Initial transplant can expose `science_news` and
   `media_archive`; keep `public_domain_story` reserved but hidden until real.

4. [Story model exposure] If `story_model` is appended as a widget, choices must
   be source-aware. Since ComfyUI static dropdowns cannot dynamically depend on
   another widget, use a conservative combo:
   - `auto`
   - media archive models
   - science/default current model
   The resolver must reject incompatible pairs loudly.

5. [Prompt surgery] Use source/story compatibility tests before touching the
   real writer path. Once prompt modules are edited, run focused tests plus the
   normal suite/Bug Bible per repo rules.

## OPTIONAL / NICE-TO-HAVE

1. Add a temporary preview/debug node only after pure helper tests, not before.
   It should not be wired into the canonical workflow unless the user asks.

2. Add `upstream_story_lab/fixtures/story_profiles/` later for human-readable
   prompt profile drafts.

## CUT THESE

1. Cut `source_text_path` from the first workflow transplant.

2. Cut any hidden bridge that reads fixture files from `upstream_story_lab` at
   runtime. Fixtures are tests/docs, not production config.

3. Cut visual-style edits in deep render-driver fallback code from the first
   source/story transplant. They are a separate visual transplant stage.

