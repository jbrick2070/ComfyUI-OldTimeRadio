VERDICT: no. The proposed validation, lifecycle, and replay contracts cannot work on the current resolver/ledger seams, while the creative schema creates multiple competing visual authorities.

MUST-FIX BEFORE BUILD:

1. [§2 story_binding; §§3-4; §6.1] The stale-story contract is both unimplementable and guaranteed to reject valid production state. `get_visual_style(meta)` and `_resolve_style(meta)` cannot access top-level story rows (nodes/_otr_visual_styles.py:378-390; nodes/_otr_story_brief_helpers.py:330-341). Raw arrays are also mutable after direction: ShotLock adds timing/render fields to lines (nodes/otr_shot_lock.py:169-217, 1036-1042), and CastLock adds voice data to cast (nodes/cast_lock.py:164-251). Add one ledger-aware `resolve_visual_direction(full_ledger)` and a versioned immutable story projection containing only authored content fields and whitelisted story meta. Exclude timing, render, voice, and cache fields; resolve once at each visual entry and thread the validated `VisualStyle` downward.

2. [§§1.5, 5, 6.1] ShotLock suppresses exactly the failure that §6.1 says must abort. `finish_visual_prompt` explicitly forbids a bare catch (nodes/_otr_story_brief_helpers.py:667-678), but ShotLock catches every exception and continues unstyled (nodes/otr_shot_lock.py:621-639). Resolve the dynamic style before the beat loop, outside the fail-soft derivation block, and pass `style=` into the finisher. `VisualStyleError` and visual-direction validation errors must propagate.

3. [§2; §§5-6.3] The plan creates two look authorities. Existing prompt composition prioritizes `meta.story_brief` atmosphere and palette; the pack’s `era_tail` is only a fallback (nodes/_otr_story_brief_helpers.py:357-428). The new artifact separately authors `global.palette`, `medium`, and a `style_pack`, without a precedence or equality rule. Make the executable dynamic pack the sole final-look authority on the dynamic lane, using the story brief only as evidence/input; alternatively retain existing brief precedence and delete the duplicated global look fields.

4. [§2; §6.3; §9-D6] Letting the LLM author the complete v2 pack exposes engine-safety fields as creative prose. The schema includes talking-portrait, mouth, radio-subject, open-subject, and motion fields (nodes/_otr_visual_styles.py:81-123), while validation mostly requires non-empty text, one mouth word, and a motion length cap (nodes/_otr_visual_styles.py:230-292). The talking lane requires a deliberately conservative bright, frontal treatment (nodes/otr_meta_brief_image_prompt.py:150-168). Build the full `VisualStyle` from a vetted safety base plus an LLM-authored whitelist of look-only fields; pin talking-mouth, subject-template, and engine-motion fields. Use a fixed Python-owned geometry vocabulary check, not the LLM-authored `forbidden_terms` list.

5. [§2; §§6.1, 6.5] The executable pack is not evidence-bound. `style_pack` has no field-evidence map, while the example `global.evidence` covers only palette and medium; nevertheless §6.5 claims every pack/global field traces to evidence. ID existence also does not prove that the cited text supports a claim. Require a complete JSON-Pointer-keyed map for every dynamically authored prompt-bearing field, bind the resolved source values into the story hash, whitelist immutable evidence paths, and distinguish factual evidence from creative rationale.

6. [§§3-4, 6.4, 8; §9-D4] Replay, reroll, and determinism contradict the real graph. Writer and FreezeCascade always re-execute (nodes/OTR_LedgerScriptWriter.py:3023-3028; nodes/OTR_LedgerFreezeCascade.py:269-272), Freeze stamps a fresh time (nodes/_otr_ledger_freeze.py:806-811), and `created_utc` sits inside the proposed artifact hash. The new node has no prior-revision source or reroll input. For v1, remove reroll and canonical “unchanged requeue” claims. If replay remains required, define persisted-artifact reuse keyed by immutable story projection + generator/prompt version + requested model, and separate a stable semantic hash from the timestamped provenance-envelope hash.

7. [§2 model_receipt; §4.2; §9-D2/D4] The generation contract is unresolved and contains false premises. The schema says `technical`, D2 leaves the slot open, and the current local seam samples with `do_sample=True`, `top_p=0.92`, and no seed (nodes/_otr_model_loader.py:1061-1127); ShotLock uses 0.1 because temperature zero is rejected (nodes/otr_shot_lock.py:687-692). Choose the creative slot for the taste task, call retries “attempts” unless a real seed changes, and record requested handle, bound backend slug, provider-reported concrete model, effective sampling parameters, output cap, attempts, and cost. Virtual and provider-resolved identities demonstrably differ (nodes/_otr_openrouter_backend.py:500-537, 905-910; nodes/_otr_google_api/llm.py:110-120).

8. [§4.2; §7.1] [ASSUMPTION] The post-freeze LLM can recreate the VRAM collision the existing handoff prevents. FreezeCascade unloads the local LLM specifically before downstream visual models (nodes/OTR_LedgerFreezeCascade.py:376-383, 452-478), but the proposed node reloads through `request_slot` and specifies no teardown. Require `unload_llm_if_local_resident()` in `finally`, expose teardown failure loudly, and verify the 16 GiB peak before image dispatch.

9. [§7.1-7.2] The concrete ComfyUI surface is incomplete. New nodes must enter `_NODE_MODULES`, which populates both class and display mappings (__init__.py:116-122, 278-289, 351-363), and must declare the complete node contract. The fan-out also needs three links—Freeze→Direction, Direction→MetaBrief, Direction→ShotLock—not merely two rewired records in workflows/otr_canonical.json. Add registration, `INPUT_TYPES`, `RETURN_TYPES/NAMES`, `FUNCTION`, `CATEGORY`, all three canonical links/output-link lists, IDs, and validator audits to §7.

SHOULD-FIX:

1. [§8] The tests prove schema validity, not story-derived creative quality. Add a representative multi-genre fixture set and an operator look-QA rubric covering specificity, palette/medium coherence, recurring identity, talking-lane safety, and measurable difference from the fixed-pack control. Valid citations plus shared tokens can still produce a generic visual treatment.

2. [§2; §9-D8] [ASSUMPTION] One response containing a full pack, global prose, evidence, and per-shot rows can exceed realistic structured-output limits. Specify per-field/list caps, maximum rows, output-token budget, and total canonical artifact bytes before implementation.

3. [§8.3 and live-smoke 1] Parameterize the inert-path test over every registered named pack plus absent/default style, not only `sci_fi_radio`. Define byte identity at the serialized ledger, prompt, prompt-hash, and request-key boundaries; verify before requiring bit-identical GPU assets.

4. [§6.2] Remove the claim that merge ownership or the already-completed freeze audit catches a buggy post-freeze mutation. `_MERGE_OWNED_ROW_FIELDS` controls disk copy-forward; it is not a mutation validator (nodes/production_ledger.py:1426-1459, 1492-1512). The immutable projection comparison must provide that protection.

5. [§6.5] Stamp the accepted direction semantic hash and generator version on image/video request observability and durable rows. Otherwise a later replacement cannot reliably connect an asset to the exact artifact that authored its prompt; current cache identity stores prompt/render inputs, not the direction revision (nodes/otr_image_gen_dispatcher.py:117-129).

OPTIONAL / NICE-TO-HAVE:

- A compact pre-render direction preview containing the thesis, palette, base pack, model identity, and evidence coverage.
- Revision history and operator reroll only after the single-artifact path is stable.
- Rename “tamper detection” to checksum/corruption detection unless the digest is anchored outside the mutable ledger.

CUT THESE (scope / over-engineering):

1. [§2; §7.7] Cut `scenes[]` and `shots[]` from vd-1. Their consumption is optional, their scene key is not canonical, and current MetaBrief/ShotLock already author beat prompts independently (nodes/otr_meta_brief_image_prompt.py:1290-1319, 1682-1695; nodes/otr_shot_lock.py:499-579). Stored-but-unconsumed direction is dead scope.

2. [§2; §9-D7] Cut `global.continuity`, wardrobe, environment locks, motifs, `clue_visual`, and free-form `composition_rules`. Existing `meta.continuity` already has an owner (nodes/OTR_LedgerScriptWriter.py:4725-4745), wardrobe has another, and composition rules conflict with the Python-owned geometry law. Retain one short non-authoritative summary derived from the accepted pack if needed for reports.

3. [§§3-4; §9-D4] Cut reroll/revision behavior from v1. It exceeds the one-dropdown scope and currently has no state source, cache trigger, or auditable history.

4. [§9-D1] Cut the placeholder `dynamic_story.json` option. Use the code-side sentinel; a renderable placeholder is a silent fallback hazard.

5. [§§3, 7.1] Cut unused `gate_in`, `done`, credits/dossier integration, and revision display. The patched-ledger data edge already establishes execution order; these surfaces do not serve the first-build goal.
