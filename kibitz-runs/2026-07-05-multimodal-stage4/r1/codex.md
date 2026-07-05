VERDICT: no. The arc is directionally right, but the proposed `rules.json` location breaks the existing story-pack registry, and the inventory misses active story-content vocabularies.

MUST-FIX BEFORE BUILD:
1. [2 / 3 / 5] `nodes/story_packs/<bank>/rules.json` is incompatible with the current router. `_otr_story_routing.py` treats every `*.json` inside a bank dir as a story pack and loads it through `_load_routed_pack`, then expects header coordinates to match the path. See `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_story_routing.py:337` and `:339`; story packs require `story_pipeline_id` etc. in `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_story_pack.py:42`. Concrete fix: put rule packs somewhere not swept as story packs, e.g. `nodes/story_rules/<source_bank_id>.json`, or update the router sweep contract in the same plan before using that path.

2. [2] The module-vs-node story contradicts the parent Stage 4 premise. BUILD_PLAN says “NEW declarative-rule ENFORCER node first,” while this plan says “module, not a graph node” and then claims the existing `OTR_WorkflowValidator` honors the word “node.” The existing validator validates workflow/litegraph/widget structure, not story-content rules; see `docs/multimodal-story-schema/BUILD_PLAN.md:81` and `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_workflow_validator.py:6`. Concrete fix: either explicitly amend Stage 4 to “module enforcer, no graph node” and remove the OTR_WorkflowValidator justification, or add a minimal `OTR_StoryRuleReport` / enforcer node as the parent plan requires.

3. [1 / 5] The inventory is incomplete for active story-content vocabulary. It lists cliche/stage-business/on-the-nose/banned-phrase/exchange words, but omits `_BANNED_THESIS_RES` used by announcer close checks, `_PERSONAL_COST_BOILERPLATE_RES` used by character quality scoring, and `DEFAULT_PROFANITY_TERMS` used by the SFW validator. See `nodes/_otr_line_hygiene.py:600`, `nodes/_otr_line_hygiene.py:1135`, and `nodes/_otr_stage3_validators.py:140`; consumers are visible at `nodes/_otr_line_composer.py:3592`, `nodes/_otr_line_composer.py:2341`, and `nodes/_otr_stage3_validators.py:657`. Concrete fix: include these vocabularies in the v1 schema or explicitly mark each as global policy / structural out-of-scope with a reason.

4. [2 / 4B] “Banks without a rules.json hard-error only when a rule consumer fires” does not hold if rules live under `story_packs/<bank>/rules.json`; registration loads/sweeps before runtime consumers. It also conflicts with 4B’s “3 dormant banks get skeleton rules.json.” Concrete fix: after moving rule packs out of the story-pack sweep, define one rule-loading contract: science must have a pack now; non-runnable banks either have no required rule pack until runnable, or have validated empty packs that are never copied from science.

SHOULD-FIX:
1. [1 / 3] `FORBIDDEN_GENERIC_WORDS` is prompt guidance, not a validation assert. The code says “Soft hygiene nudge ONLY -- NOT a gate” and joins it into prompt text; see `nodes/_otr_compose_exchange.py:197` and `:380`. Concrete fix: remove it from the rule-enforcer slice or move it under story-pack prompt config, not the validation rules schema.

2. [2] The plan creates a second JSON home for story guardrails while story packs already have inert fields like `tone_guardrails`, `forbidden_plot_patterns`, and `forbidden_leakage_terms`; see `nodes/_otr_story_pack.py:81`. Concrete fix: state why rule packs are separate from pack metadata, or merge the validation vocabularies into an explicit `rules` object attached to each bank/model.

3. [6 / Q2] Per-bank full-copy rules are likely to calcify sci-fi taste into unrelated lanes. The plan itself says public-domain should not inherit sci-fi cliche tuning wholesale, then allows “copy of science or curated-lite.” Concrete fix: make dormant-bank packs empty/explicitly uncurated, or defer them until each bank becomes runnable.

OPTIONAL / NICE-TO-HAVE:
- [6 / Q4] Add a minimal regex guard now: max pattern length, compile fail-loud, and a small fixture corpus. Full ReDoS analysis can wait unless packs become user-authored at runtime.

CUT THESE (scope / over-engineering):
1. [3 / 4C] Cut `OTR_StoryRuleReport` from Stage 4 unless a concrete consumer is named. A post-writer report does not move active gates to JSON and can be added later.

2. [3 / 4B] Cut copied skeleton rules for the three non-runnable banks. They do not affect current runtime behavior and risk freezing fake curation.

3. [1 / 3] Cut `forbidden_generic_words` from the rule enforcer. It is safe to cut because it is only prompt guidance today, not a hard validation rule.