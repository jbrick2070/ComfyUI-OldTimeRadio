# R3 Wiring And Integration Plan

Status: wiring-plan synthesis. This is the hardened implementation map for the
final convergence round.

## Verdict

Buildable, with three wiring rules that must be treated as hard constraints:

1. Source-bank selection starts inside `OTR_LedgerScriptWriter`; C3 does not add
   new upstream source nodes to the workflow.
2. Visual style is an independent policy fan-out, not a replacement for existing
   video or image policy links.
3. Visual style must be applied inside `finish_visual_prompt`, so every prompt
   producer and render-driver fallback path inherits the same policy behavior.

## Workflow Grounding

Current canonical workflow:

- File: `workflows/otr_scifi_16gb_full.json`
- `last_node_id`: 94
- `last_link_id`: 273
- Writer node: id 1, type `OTR_LedgerScriptWriter`, order 1
- Video policy node: id 87, type `OTR_VideoDirector`, order 2
- Image policy node: id 88, type `OTR_ImageDirector`, order 4
- MetaBrief prompt node: id 89, type `OTR_MetaBriefImagePromptGen`, order 6
- ShotLock node: id 90, type `OTR_ShotLock`, order 14

Existing policy links to preserve:

- 251: `OTR_VideoDirector` -> `OTR_ShotLock.video_policy_json`
- 270: `OTR_VideoDirector` -> `OTR_ImageDirector`
- 254: `OTR_ImageDirector` -> `OTR_MetaBriefImagePromptGen.image_policy_json`
- 257: `OTR_ImageDirector` -> `OTR_ImageGenDispatcher.image_policy_json`

Do not replace or reroute those links when adding visual style.

## C0 Contract Lock

`VisualStylePolicy` must be fully specified before C4 or C5 are coded.

Minimum schema:

```
style_id: str
label: str = ""
base_tail_strategy: Literal["keep", "replace", "suppress"] = "keep"
positive_tail: str = ""
forbidden_terms: list[str] = []
notes: str = ""
```

Style defaults:

- `cinematic_35mm`: `base_tail_strategy="keep"`, `positive_tail=""`,
  `forbidden_terms=[]`; this preserves current output.
- `archival_mono`: `base_tail_strategy="replace"`, positive monochrome/archive
  tail, forbidden modern glossy/color terms.
- `anime`: `base_tail_strategy="replace"`, positive animation/cel-shaded tail,
  forbidden photoreal/35mm/film/photo terms.

`StoryInputPacket.adaptation_trace` can remain an untyped dict in V1, but it
must be documented as implementation-defined until the public-domain adapter
locks its schema.

## C1 Writer Source Selector

Append one optional widget to `OTR_LedgerScriptWriter.INPUT_TYPES`:

```
source_bank: combo = ["science_news", "media_archive", "public_domain_story"]
default: "science_news"
```

Workflow delta:

- Append `"science_news"` to node 1 `widgets_values`.
- Do not insert into the existing vector.
- Do not rename outputs.
- Do not add public-domain text widgets in C1.

Code/test deltas:

- Update the writer self-test optional-widget assertion from 16 to 17.
- Extend the assertion comment to include `source_bank` appended in C1.
- Verify `source_bank=science_news` leaves default science path output stable.
- Verify `meta.source` does not break any strict downstream meta consumer before
  enabling it broadly.

## C2 Prompt Profile Ownership

The hardcoded science framing must be owned by C2, not left as an audit note.

Modify these sites:

- `_otr_outline.py`
- `_otr_pitch_room.py`
- `_otr_story_select.py`

Rules:

- Keep the existing `_SYSTEM_PROMPT` constant for the default `science_news`
  path.
- Add a builder that returns the constant unchanged for the default path and
  constructs non-science wording only for alternate source banks.
- Add `story_form_label` or equivalent to `grade_story` and pass it from the
  writer refine loop.
- Parameterize pitch-room wording, or explicitly bypass pitch-room/story-select
  for non-science source banks in C3. Do not leave a mixed archive/science
  prompt path.

## C3 Media Archive Sequencing

`_resolve_inputs` runs before model load, so it must not run LLM archive
interpretation.

Correct split:

1. `_resolve_inputs`
   - Read `source_bank`.
   - For `science_news`, keep the current RSS/custom-premise setup.
   - For `media_archive`, fetch/select a raw archive item only.
   - For `public_domain_story`, raise a clear not-implemented error until C7.

2. Main writer `run()` after model/generate functions exist
   - Convert archive item into `StoryInputPacket`.
   - Run archive interpretation with `technical_generate_fn`.
   - Build legacy `meta.news` mirror.
   - Populate `key_terms_tuple` from `StoryInputPacket.key_terms` or mirrored
     `meta["news"]["key_terms"]`.
   - Fail closed if `script_brief` is empty.

Branch scope:

- The media archive branch must cover the full science-news path, not only the
  `_fetch_rss_seed_or_die` call.
- It must not accidentally run archive material through science-framed
  pitch-room or story-select prompts.

## C4 Visual Style Prompt Seam

Modify `nodes/_otr_story_brief_helpers.py`, specifically
`finish_visual_prompt`.

Binding behavior:

- Normalize metadata with `_meta(meta)` and read style with
  `_meta(meta).get("visual_style")`.
- If no visual style is present, current behavior remains unchanged.
- If style is present, scrub forbidden terms and apply the policy tail inside
  `finish_visual_prompt`.
- `style_tail=False` suppresses the default `STYLE_TAIL_DEFAULT`, but it must
  not suppress the policy `positive_tail`.
- This is required so `nodes/_otr_video_engines/render_driver.py` fallback scene
  prompts inherit visual style without separate render-driver edits.

## C5 VisualStyleDirector Node

Add `OTR_VisualStyleDirector` with a single output.

Contract:

```
CATEGORY = "OldTimeRadio/v2/video"
FUNCTION = "direct"
RETURN_TYPES = ("STRING",)
RETURN_NAMES = ("visual_style_policy_json",)
```

Inputs:

- `style_id` combo:
  `cinematic_35mm`, `archival_mono`, `anime`

Execution:

- Validate output as `VisualStylePolicy`.
- Return a single-element tuple: `return (visual_style_policy_json,)`.
- No `custom_policy_json` input in V1.

## C6 Visual Style Workflow Wiring

Append a force-input socket to both prompt producers:

- `OTR_MetaBriefImagePromptGen.generate(..., visual_style_policy_json="{}")`
- `OTR_ShotLock.lock(..., visual_style_policy_json="{}")`

Append the new parameter at the end of each execution signature to avoid
shifting positional call sites.

Workflow delta after code is live:

- Add one `OTR_VisualStyleDirector` node near the visual policy cluster.
- Default widget value: `"cinematic_35mm"`.
- Add two links from the node output:
  - to `OTR_MetaBriefImagePromptGen.visual_style_policy_json`
  - to `OTR_ShotLock.visual_style_policy_json`
- Current expectation if the workflow is unchanged at edit time:
  - new node id: 95
  - new links: 274 and 275
- Recompute these from the live workflow immediately before editing; do not
  hardcode stale IDs.

Expected socket indices after code append, to verify live:

- `OTR_MetaBriefImagePromptGen`: `visual_style_policy_json` slot 3
- `OTR_ShotLock`: `visual_style_policy_json` slot 5

Use live `INPUT_TYPES` or a workflow patch helper that links by input name. Do
not guess slot indices by hand.

Downstream behavior:

- Empty/missing/invalid style JSON should fall back to current cinematic
  behavior and add a warning/report where the node already reports warnings.
- Style policy is stamped into ledger/meta as `meta.visual_style`.
- Do not wire style directly into `OTR_VideoRenderBatch` in V1 unless a later
  bug fix proves the ledger/meta path is insufficient.

## C7 Public-Domain Adapter

C7 remains separate from the source-bank selector.

Workflow delta only when C7 exists:

- Append exactly one public-domain source input first: either `source_text` or
  `source_text_path`.
- Prefer one input, not both, until tests show both are needed.
- Update workflow widget/input audits in the same change.

Flow:

```
operator text -> StoryInputPacket -> StoryBlueprint -> OutlineRequest -> ledger
```

Do not bypass the existing outline/ledger path.

## Existing Bug Watch

Antigravity flagged a plausible existing ledger synchronization issue in
`OTR_VideoRenderBatch._stamp_render_engines_meta`, where the node may save via
`production_ledger.get_ledger()` instead of the incoming `patched_ledger_json`.

This is not part of the source-bank or visual-style architecture, but it should
be verified before relying on any render-engine metadata stamp. If confirmed,
fix it as a separate root-cause bug by synchronizing from the incoming parsed
ledger before saving.

## Verification Gates

Per code chunk:

- Targeted tests for the touched modules.
- Writer optional widget count assertion.
- Method signature checks for any new ComfyUI inputs.
- Prompt default-path stability tests.
- `finish_visual_prompt` default unchanged test.
- Visual style policy tests for all three style IDs.

After any workflow edit:

- `OTR_WorkflowValidator`
- JSON round-trip
- link referential integrity audit
- widget-count vs live `INPUT_TYPES`
- every wired input name exists in live `INPUT_TYPES`
- canonical workflow only:
  `workflows/otr_scifi_16gb_full.json`

After code changes:

- focused pytest subset
- regression suite
- Bug Bible from the separate survival-guide repo
- commit and push green chunks to `v2.0-alpha`

## R3 Judgment Log

Accepted:

- Codex anchor: visual style is a fan-out node and preserves all existing
  video/image policy links.
- Codex anchor: media archive v1 is writer-internal, not a new source-node
  wiring change.
- Antigravity: archive LLM interpretation must run after model load, not inside
  `_resolve_inputs`.
- Antigravity: execution signatures must accept appended visual style inputs.
- Antigravity: `OTR_VisualStyleDirector.direct` must return a one-item tuple.
- Antigravity: non-news key terms must come from `StoryInputPacket`.
- Claude: visual style belongs inside `finish_visual_prompt`, and policy
  positive tails still apply when `style_tail=False`.
- Claude: C0 must lock `VisualStylePolicy` fields before C4/C5.
- Claude: update the writer optional-widget assertion in C1.
- Claude: pitch-room/story-select science wording needs an owned change.
- Claude: cut `custom_policy_json` from V1.

Modified:

- Antigravity's `OTR_VideoRenderBatch` overwrite claim is plausible and should
  be verified as an existing bug, but it is not a blocker for the source-bank
  or visual-style contract itself.
- Workflow IDs and slot numbers are recorded from the current graph, but final
  edits must recompute them from the live file immediately before writing.

Rejected:

- Any plan that rewires or replaces existing `VideoDirector`/`ImageDirector`
  policy links.
- Any public-domain source widget in C1.
- Any visual-style implementation limited only to MetaBrief/ShotLock call sites
  while leaving `finish_visual_prompt` unchanged.
