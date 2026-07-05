# Transplant Manifest

Status: placeholder checklist. Fill this before touching the production
workflow.

## Ready-To-Transplant Gates

- Source packet contracts have pure tests.
- Story model catalog has source-scoped tests.
- Prompt profile rendering has negative tests for forbidden sci-fi/news phrases.
- Media archive fixtures do not call science RSS.
- Visual style policies have negative tests for hardcoded cinematic/radio tails.
- Compatibility mirror to `meta.news.*` is exact and centralized.
- Workflow widget additions are append-only.
- Canonical workflow JSON validation plan is written.

## Production Transplant Targets

Do not begin these until the upstream lab is green:

- `nodes/_otr_source_packets.py`
- `nodes/_otr_story_model_catalog.py`
- `nodes/_otr_story_prompt_profile.py`
- `nodes/_otr_ledger_writing_spec.py`
- `nodes/_otr_ledger_input_adapter.py`
- `nodes/_otr_visual_style_policy.py`
- `nodes/_otr_visual_render_catalog.py`
- prompt-profile edits in story/ledger prompt modules
- visual-policy edits in MetaBrief, ShotLock, and render-driver prompt seams
- canonical workflow JSON changes

## Known Transplant Risk: Deep Visual Prompts

The source/story upstream can be isolated cleanly. The visual/video prompt
transplant is riskier.

Some visual prompts are deep inside current still/video repair and fallback code
that is actively being improved. Treat visual-style extraction as staged:

1. Identify every deep hardcoded prompt before editing.
2. Preserve current shipped behavior under `sci_fi_radio`.
3. Add visual-style policy reads at shared seams first.
4. Only then replace deep fallback prompts one by one.
5. Test each style for forbidden phrase leakage.

Do not assume one new visual-style node can cleanly override every downstream
image/video prompt. R3 must plan this as a transplant with risk checkpoints.

## Future Bridge Strategy

The lab should grow an upstream translator head before it touches the full
production downstream workflow.

Planned boundary:

```text
upstream_story_lab source/story/visual spec
-> translator head
-> bridge node or bridge adapter
-> latest production OTR downstream workflow
```

The bridge is not the first build. The first build proves the upstream
translator can produce a validated production-ledger-shaped payload from fixture
source packets and prompt profiles.

Only after that:

1. add a bridge adapter/node plan
2. generate a dry-run patch plan for production code and workflow JSON
3. review widget deltas and forceInput sockets
4. apply the transplant in one explicit chunk
5. run focused integration tests before any full e2e render

Until the translator head is ready, do not run full e2e tests through the bridge
and do not wire the bridge into `workflows/otr_scifi_16gb_full.json`.

## Future Source-Bank Dropdown

Planned source-bank dropdown labels:

- `science_news`: "Sci-Fi Science News"
- `media_archive`: "Media RSS / Archive"
- `public_domain_story`: "Public Domain"
- `custom_source_bank`: "+ Add Your Own"

Suggested tooltip for `custom_source_bank`:

```text
Build a custom source bank from a schema/profile. See upstream_story_lab/CUSTOM_SOURCE_BANK_GUIDE.md.
```

`custom_source_bank` must not silently run anything. If selected without a valid
custom schema/profile, it raises a clear error pointing to the guide.

## Future Transplant Script

A future Python transplant helper is allowed, but it must be dry-run first and
manifest-driven.

Expected shape:

```text
scripts/plan_transplant.py --dry-run
scripts/plan_transplant.py --apply --manifest <approved-manifest.json>
```

It should be able to propose or apply:

- prompt-profile code edits
- visual-style policy hook edits
- workflow JSON widget appends
- bridge node insertion
- link creation for forceInput policy sockets
- whitelist updates in `scripts/otr_api.py`
- whitelist updates in `nodes/_otr_workflow_apply.py`

Hard rule: the script must refuse to apply unless the manifest names every file,
expected widget-vector delta, and validation command. No silent broad rewrite.

## Workflow JSON Touch Rules

When the transplant reaches workflow wiring:

- edit only `workflows/otr_scifi_16gb_full.json`
- append widgets only
- use forceInput sockets for linked policy JSON
- run the workflow validator
- run JSON round-trip validation
- run link referential integrity audit
- run widget/input audit
