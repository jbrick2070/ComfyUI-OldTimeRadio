# R1 Hardened Architecture Plan

Status: round-1 synthesis. This is the input for the coding-plan round.

## Verdict

The two-axis redesign is directionally correct, but the build must narrow the
first implementation. V1 should not create four new ComfyUI nodes at once. It
should first introduce typed source/style contracts, bridge them through the
existing writer and prompt nodes, and preserve the current workflow surface.

## Accepted Architecture Decisions

1. Keep the two axes separate.
   - Story source/writer engine is pre-ledger.
   - Visual style/model prompt policy is post-ledger/pre-render.
   - Visual style never rewrites dialogue or ledger lines.

2. Make `meta.source` canonical for source provenance, but keep `meta.news` as a
   v1 compatibility mirror.
   - Confirmed reason: the current writer emits `news_used`, many code paths
     still read `meta.news.script_brief`, and the workflow wires `news_used`
     downstream.
   - V1 rule: do not rename writer outputs or remove `meta.news`.

3. Keep `meta.visual_plan` as the existing character/scenes visual store.
   - Add a new `meta.visual_style` / `visual_style_policy_json` surface for
     visual prompt transforms.
   - Do not overload `meta.visual_plan.style`; it is already tied to the
     current writer style/aesthetic lineage.

4. Public-domain adaptation must not bypass the existing outline/ledger
   validators directly.
   - V1 public-domain route is:
     `PD source text -> StoryBlueprint -> existing Outline/Beat[] -> ledger`.
   - The adapter carries `adaptation_trace` and source-fidelity checks.

5. `media_archive` cannot be only an RSS feed swap unless it always emits a
   valid `script_brief`.
   - The current outline fallback still says "Science story" when no
     `script_brief` exists.
   - V1 archive mode must fail closed before outline generation if the archive
     interpreter cannot produce a brief, or the outline prompt must gain a
     source-kind/source-label parameter first.

6. Visual style policy must reach prompt composition, not only model selection.
   - V1 route:
     `OTR_VisualStyleDirector -> visual_style_policy_json`
     wired to `OTR_MetaBriefImagePromptGen` and `OTR_ShotLock`.
   - `OTR_ShotLock` stamps `meta.visual_style` and styles shot/video prompts.
   - `OTR_VideoRenderBatch` should not need a new style socket in v1 if the
     shot prompts are already styled; verify this during R3 wiring.

## Rejected Or Deferred Suggestions

1. Do not migrate core `OTR_LedgerScriptWriter` widgets to a new
   `OTR_StoryDirector` in the first chunk.
   - Reason: that is a high-risk widget/order/workflow migration.
   - V1 keeps writer widgets and adds only appended optional controls.

2. Do not implement `IS_CHANGED` as `time.time()`.
   - Reason: it forces reruns and hides source identity bugs.
   - Use source hashes, selected bank id, explicit refresh nonce, and source URL
     or local file mtime/content hash instead.

3. Cut `science_explainer_engine` from the first sprint.
   - It is a useful later engine, but it is a new format, not needed to prove
     news/archive/public-domain source architecture.

4. Cut `graphic_novel`, `custom`, and broad arbitrary Gutenberg search from the
   first visual/source sprint.
   - First source choices: `science_news`, `media_archive`,
     `public_domain_story` from operator-supplied/curated text.
   - First non-default visual choices: `archival_mono` or `anime`, not both
     unless the first one lands cleanly.

## V1 Contract Shape

### StoryInputPacket

Canonical pure schema in a new module such as `nodes/_otr_source_packet.py`.
No ComfyUI imports. Unknown keys rejected.

```
{
  "packet_version": 1,
  "source_bank_id": "science_news|media_archive|public_domain_story|operator_seed",
  "source_mode": "seed|adapt",
  "source_kind": "science_news|archive_media|pd_text|operator_text",
  "source_label": "Science story|Archive/media seed|Public-domain story",
  "rights_status": "unknown|public_domain|licensed|operator_supplied",
  "source_title": "",
  "source_author": "",
  "source_url": "",
  "source_hash": "",
  "source_text_ref": "",
  "source_summary": "",
  "casting_brief": "",
  "script_brief": "",
  "close_brief": "",
  "key_terms": [],
  "adaptation_constraints": {},
  "adaptation_trace": {}
}
```

V1 ledger stamping:

- `meta.source` gets the full packet/provenance subset.
- `meta.news` mirrors `script_brief`, `news_close_brief`/`close_brief`,
  `key_terms`, source title/url/hash, and prompt-version-like fields for legacy
  consumers.
- `news_used` output remains a JSON string with current-compatible fields.

### StoryBlueprint

Pure schema for public-domain adaptations before Outline generation.

```
{
  "blueprint_version": 1,
  "source_packet_hash": "",
  "title": "",
  "premise": "",
  "characters": [],
  "setting": "",
  "required_beats": [],
  "ending": "",
  "fidelity_requirements": {
    "ending": "must",
    "protagonist": "must",
    "antagonist": "must",
    "core_twist_or_moral": "warn",
    "setting": "warn"
  }
}
```

The blueprint adapter feeds the existing outline/ledger path rather than writing
ledger lines directly.

### VisualStylePolicy

Pure schema in a new module such as `nodes/_otr_visual_style_policy.py`.
Remove `allow_story_rewrite`; it is an invariant, not an option.

```
{
  "profile_version": 1,
  "style_id": "cinematic_35mm|archival_mono|anime",
  "prompt_tone": "",
  "positive_tail": "",
  "negative_tail": "",
  "base_tail_strategy": "keep|replace|suppress",
  "color_language": "",
  "camera_language": "",
  "character_design_rules": "",
  "background_rules": "",
  "model_family_hints": [],
  "forbidden_terms": []
}
```

V1 default: `cinematic_35mm` must be byte-identical or semantically identical to
the current prompt behavior when no new style node/input is wired.

## First Coding Plan Shape

1. Add pure schemas and contract tests.
   - `StoryInputPacket`
   - `StoryBlueprint`
   - `VisualStylePolicy`
   - JSON round-trip and unknown-key rejection.

2. Add source-packet bridge inside `OTR_LedgerScriptWriter`.
   - Keep existing widgets and output names.
   - Add default-internal `science_news` packet creation from the current RSS /
     custom-premise path.
   - Stamp `meta.source`.
   - Mirror to `meta.news` and `news_used`.

3. Add prompt source-label plumbing.
   - Add `source_label` / `source_kind` to the outline request or wrapper.
   - Replace raw fallback wording of "Science story" with the source label.
   - Keep current science-news output identical for the default path.

4. Add `media_archive` as the first alternate source bank.
   - Curated LOC/NFPF/ACE feed list.
   - Archive interpreter emits the same packet fields.
   - Fail closed if no `script_brief` is produced.
   - Test that archive prompts do not say "science story."

5. Add visual-style pure policy and prompt transform helpers.
   - Implement tail keep/replace/suppress.
   - Implement forbidden-term scrub for cinematic tails in non-cinematic modes.
   - Unit-test `cinematic_35mm` no-op and one non-default profile.

6. Add `OTR_VisualStyleDirector`.
   - Policy-only ComfyUI node.
   - Register in `__init__.py`.
   - Wire to canonical workflow only when downstream consumers exist.

7. Append optional `visual_style_policy_json` inputs.
   - `OTR_MetaBriefImagePromptGen`: applies style to still prompts.
   - `OTR_ShotLock`: applies style to video/shot prompts and stamps
     `meta.visual_style`.
   - Inputs must be appended in `INPUT_TYPES`; update workflow JSON in the same
     change.

8. Add public-domain adaptation after archive mode proves the packet seam.
   - Start with local/operator-supplied text, not search.
   - Build source extraction -> `StoryBlueprint`.
   - Feed existing outline path.
   - Add fidelity trace and warnings/failures.

## R1 Judgment Log

Accepted:

- Anchor: `meta.source` canonical plus `meta.news` compatibility.
- Anchor/Claude: direct public-domain `Beat[]` bypass is too risky; use
  `StoryBlueprint`.
- Anchor/Antigravity/Claude: visual style needs an explicit prompt-policy route.
- Claude: define relationship between `meta.visual_plan` and visual style.
- Claude: line composer/theme still depends on `meta.news.script_brief`; preserve
  or add a read path before removing `meta.news`.
- Claude: "no-op science_news bank" is a refactor, not a no-op; treat it as a
  regression-sensitive chunk.
- Antigravity: append optional visual-style inputs to avoid widget-position
  drift.
- Antigravity: proposed new nodes must be registered before they are real.

Rejected or modified:

- Antigravity: `IS_CHANGED = time.time()`; replace with source hash / refresh
  nonce / file mtime-content hash policy.
- Antigravity: migrate writer LLM widgets to `OTR_StoryDirector` in v1; defer.
- Antigravity: direct `OTR_VideoRenderBatch` style input in v1; verify in R3,
  but prefer styling prompts before render batch.

Verify in R2/R3:

- Whether render driver ever composes fresh text prompts after ShotLock; if yes,
  it also needs visual-style policy or stamped `meta.visual_style`.
- Exact workflow nodes and links for visual-style insertion.
- Exact list of LLM prompt templates that still hardcode science/scifi wording.

