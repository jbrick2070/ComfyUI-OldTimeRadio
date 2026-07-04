VERDICT: yes-with-fixes — Batch 3 is not build-ready until caption/credits order, output path, ledger resolution, and strict-types validation are made explicit.

MUST-FIX BEFORE BUILD:
1. [Batch 3 / 86-owner STAGE ORDER] Defect: “CaptionBurn is the LAST silent-video pass before MasterAudioMux” is now wrong/ambiguous with node 95 in the chain. Correct chain is `84 -> 93 -> 86 -> 95 -> 85`: burn captions after final procgen/scopes composition, before credits, so credits stay caption-free and node 95 still feeds `declared_credits_tail_s` to node 85. Current workflow is `84 -> 86 -> 93 -> 95 -> 85` in `workflows\otr_scifi_16gb_full.json`; tests currently pin 93-owner at `tests\test_workflow_live_passes_validator.py:56-87`. Concrete fix: replace the plan text with the exact chain and required links.

2. [Batch 3 / OUTPUT CONTRACT] Defect: if node 86 becomes the active burner, its default output path writes flat to `otr/episodes/<stem>_captioned.mp4`, not the per-episode folder. See `nodes\otr_caption_burn.py:183-192`. This violates the repo output contract for rendered assets. Concrete fix: require node 86 to default to an output adjacent to its input video, or otherwise under `otr/episodes/<episode_id>/`, before enabling it in production.

3. [Batch 3 / LEDGER RESOLUTION] Defect: node 86 cannot reliably resolve a ledger after node 93 because it strips only `_silent`, `_captioned`, `_final`, `_blend` at `nodes\otr_caption_burn.py:70-86`; node 93 outputs `_procgen_blended` at `nodes\otr_post_upscale_procgen_blend.py:923-930`. Node 93’s legacy resolver strips `_procgen_blended` and has a sibling `audio/` fallback at `nodes\otr_post_upscale_procgen_blend.py:98-115`. Concrete fix: require node 86 to port both the `_procgen_blended` strip and sibling-audio fallback, with a Desktop/no in-flight-ledger regression test.

4. [Batch 3 / ENABLEMENT] Defect: “set canonical true OR wire `OTR_BURN_CAPTIONS=1` through every launch path” leaves two incompatible build paths. Current node 86 default and saved widget are false (`nodes\otr_caption_burn.py:160-198`; workflow node 86 widget 0 is false), while node 93/profile mapping owns captions now (`config\profiles\widget_mapping.json:95-109`). Concrete fix: choose one path. For this repo, make the canonical workflow/widget/profile mapping the source of truth; do not rely on env wiring as the production enablement mechanism.

5. [Validation / strict-types] Defect: `tools\validate_workflow_links.py --strict-types` currently fails nodes 80-83, but the runtime registration is not actually missing. The CLI only parses literal `_NODE_MODULES` keys at `tools\validate_workflow_links.py:61-81`; nodes 80-83 are dynamically merged from `nodes\_otr_class_registry.py:48-92` via `__init__.py:327-345`. Concrete fix: update the CLI to include `new_node_modules_table()` keys without importing heavy node modules, then require strict-types to pass.

SHOULD-FIX:
1. [Line cites] `_otr_voice_node_common.py:183-188` no longer cites the docstring; the relevant docstring is `nodes\_otr_voice_node_common.py:176-182`. Adjust the plan citation.
2. [Line cites] `__init__.py:299-302` registers CaptionBurn/MasterAudioMux, not CreditsRoll; CreditsRoll is at `__init__.py:315-322`. Adjust if the plan cites the credits registration.
3. [Batch 3 / comments] `nodes\otr_post_upscale_procgen_blend.py:823-826` says caption widgets are the tail, but `scopes_mp4_path` and `audio_bars` follow at `:848-868`. Clean this stale positional-widget comment during any owner migration.

OPTIONAL / NICE-TO-HAVE:
- Add a small visual smoke that samples the credits tail and confirms no SDH caption box is burned into credits frames. [ASSUMPTION] This is visual QA rather than a structural build blocker.

CUT THESE:
1. [Batch 3 / ENABLEMENT] Cut the “wire `OTR_BURN_CAPTIONS=1` through every launch path” alternative. It is safe to cut because canonical workflow/profile wiring is already the repo’s source of truth and avoids Desktop/headless drift.
2. [Batch 4] Keep Batch 4 as “CUT” only; do not carry alias-dedupe text into the builder handoff.

VERIFY-AT-BUILD checklist:
1. Run `OTR_WorkflowValidator` against `workflows\otr_scifi_16gb_full.json`.
2. JSON round-trip the canonical workflow and run link referential integrity over `links[]`, input `link` fields, and output `links` arrays.
3. Re-run widget audit against live `INPUT_TYPES`; expected drift must be zero after intentional Batch 1 vector edits.
4. Run `tools\validate_workflow_links.py --workflow workflows\otr_scifi_16gb_full.json --strict-types` after the CLI registry fix.
5. Fresh ComfyUI restart: verify live `/object_info` contains nodes 80, 81, 82, 83, 86, 93, 95, and 85.
6. If 86-owner ships: verify chain `84 -> 93 -> 86 -> 95 -> 85`, node 95 tail still wired to node 85 slot 6, and credits frames remain caption-free.
7. Confirm rendered assets land under `otr\episodes\<episode_id>\` and final published video under `otr\obs\`.
8. Run suite, Bug Bible, `test_audio_byte_identical`, `test_capability_profiles.py`, and `test_workflow_apply.py`.
