# Cloud slug preflight

Queue-time check so a run graph (saved JSON or an unsaved dropdown change)
refuses a dead / inactive cloud model before the first credit moves.

The gate is `OTR_WorkflowValidator` -- already present in every
`workflows/**/*.json`. No graph rebuild is required. Coverage is the
registries (every dropdown-reachable paid engine), not the 21 shipping
pins.

## Where it fires

`nodes/_otr_workflow_validator.py` `_queue_time_readiness_gates`:

1. `ensure_prompt_cloud_slugs(prompt, unique_id)` -- $0 refusal
2. `ensure_prompt_visual_assets(prompt, unique_id)` -- then weight downloads

`validate_anyway=False` still runs both. Replay still checks video / image /
voice / music picks; it skips writer slugs only (no LLM call happens).

## Authority

- Comfy Credits media: in-process partner `define_schema()` / `INPUT_TYPES()`
  Combo + DynamicCombo keys (T1). Not `GET api.comfy.org/v2/models`.
- Writer on Credits or OpenRouter: public OpenRouter catalog (T2).
  Transport fail vs Comfy proxy = warning; vs OpenRouter direct = refuse.
- Google BYO engines and Google writer slots: `models.list` (T2).
  Transport fail = refuse (same host as the paid call).

Adapters own `cloud_selectors()` -- candidate sets, not a second table.
Product rows (Flux Pro, Kling class-import, Sonilo) send `{node_key: {}}`.

No generate. No silent floor. No hiding engines from dropdowns.
