# Custom Source Bank Guide

Status: schema guide for the future `+ Add Your Own` source-bank dropdown item.

This is for building a new source brain without changing the production ledger
schema.

## Dropdown Behavior

Future source-bank dropdown label:

```text
+ Add Your Own
```

Internal id:

```text
custom_source_bank
```

Suggested tooltip:

```text
Build a custom source bank from a schema/profile. See upstream_story_lab/CUSTOM_SOURCE_BANK_GUIDE.md.
```

This is not a silent fallback. If selected without a valid custom schema, it
should raise a clear error.

## Required Idea

A custom source bank must fill the same ledger contract through the same
upstream control plane:

```text
custom source material -> SourceMaterialPacket -> StoryInputPacket
-> StoryPromptProfile -> LedgerWritingSpec -> production ledger
```

It may create different story data and different prompt language, but it must
not create a new downstream ledger type.

## Minimum Schema Fields

Use a unique snake_case source bank id:

```json
{
  "source_bank_id": "my_custom_archive",
  "label": "My Custom Archive",
  "source_kind": "archive_item",
  "rights_status_default": "unknown",
  "story_models": [],
  "prompt_profile": {},
  "forbidden_prompt_terms": []
}
```

Required sections:

- `source_bank_id`: unique snake_case id; never reuse built-in ids.
- `label`: UI label.
- `source_kind`: what source material represents.
- `rights_status_default`: default provenance state.
- `story_models`: one or more source-scoped story models.
- `prompt_profile`: labels and system prompt overrides.
- `forbidden_prompt_terms`: phrases that should not appear in generated prompts.

## Story Model Fields

Each story model should define:

```json
{
  "story_model_id": "warm_restoration_mystery",
  "label": "Warm Restoration Mystery",
  "tone_guardrails": [
    "Use discovery and repair as the central pressure."
  ],
  "forbidden_plot_patterns": [
    "spaceship rescue",
    "laboratory containment breach"
  ],
  "outline_rules_extra": "Build tension from missing context, not violence."
}
```

Forbidden examples are metadata/test rules. Do not blindly paste forbidden terms
into live generative prompts, because models may copy negated terms.

## Prompt Profile Fields

Minimum prompt profile:

```json
{
  "story_form_label": "custom archive-inspired radio drama",
  "source_material_label": "Custom source item",
  "source_develop_verb": "build a fictional radio story from this source material",
  "source_grounding_label": "source material",
  "coda_mode": "archive_source_note",
  "title_form_label": "custom archive radio drama",
  "line_grounding_instruction": "Ground this line in the source material and scene premise.",
  "outline_system_prompt": "You are a radio-drama story editor for this custom source bank."
}
```

## Vibe-Coder Hints

- Clone an existing source-bank schema first.
- Change ids and labels before changing prompt style.
- Keep `source_bank`, `story_model`, and `visual_style` separate.
- Do not rename production ledger fields.
- Do not create hidden fallback to `science_news`.
- Add one fixture source packet before adding live network fetching.
- Add leakage tests before wiring to production.
- Keep custom visual language in `visual_style`, not source-bank prompts.

## Acceptance Checklist

- Custom schema parses.
- At least one fixture source packet parses.
- `StoryInputPacket` can be built without network access.
- `LedgerWritingSpec` can be built.
- Prompt preview contains no forbidden drift terms.
- Unknown source/story/style ids fail loudly.

