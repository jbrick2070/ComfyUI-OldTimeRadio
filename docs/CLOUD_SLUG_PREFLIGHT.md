# Cloud slug preflight

Queue-time check so a run graph (saved JSON or an unsaved dropdown change)
refuses a dead / inactive cloud model before the first credit moves.

The gate is `OTR_WorkflowValidator` -- already present in every
`workflows/**/*.json`. No graph rebuild is required. Coverage is the
registries (every dropdown-reachable paid engine), not the 21 shipping
pins.

## Where it fires

`nodes/_otr_workflow_validator.py` `_queue_time_readiness_gates`:

1. `ensure_prompt_cloud_slugs(prompt, unique_id)` -- $0 stale-slug refusal
   (Comfy T1 partner schema; Google / OpenRouter T2 catalogs). A `models/`
   prefix on a Google id is stripped before the compare.
2. `ensure_prompt_cloud_balance(prompt, unique_id)` -- wallet vs padded
   estimate. Comfy and OpenRouter refuse when remaining is short or the
   same-host balance GET fails. Google has no remaining-dollar API --
   estimate is logged, never refused on a missing Google balance.
3. `ensure_prompt_visual_assets(prompt, unique_id)` -- then weight downloads

`validate_anyway=False` still runs all three. Replay still checks video /
image / voice / music picks; it skips writer slugs and the writer line of
the wallet estimate (no LLM call happens).

## Authority

- Comfy Credits media: in-process partner `define_schema()` / `INPUT_TYPES()`
  Combo + DynamicCombo keys (T1). Not `GET api.comfy.org/v2/models`.
- Writer on Credits or OpenRouter: public OpenRouter catalog (T2).
  Transport fail vs Comfy proxy = warning; vs OpenRouter direct = refuse.
- Google BYO engines and Google writer slots: `models.list` (T2).
  Transport fail = refuse (same host as the paid call). A `models/`
  prefix on the posted id is stripped before the compare, same as
  writers -- that is the Comfy-style stale-pin check for Google.
  Component smoke (catalog only, no generate):
  `scripts/otr_google_component_smoke.py` writes
  `docs/2026-09-18-google-component-smoke.txt`.

## Writer slugs are the posted value

The writer check does not read the slot widget text. It binds the queued
widgets into the selected backend and asks that backend's own resolver --
`resolve_slug` (OpenRouter, Comfy Credits) or `resolve_model_for_slot`
(Google) -- so the checked slug is the one that will be POSTed. A placeholder
widget on OpenRouter or Comfy still spends via the `OTR_*_SLOT_x_DEFAULT` /
recommended fallback, so that fallback slug is what gets checked. Google has
no fallback: an unbound slot refuses with no request sent. Catalog-down on
OpenRouter direct or Google refuses (same host as the paid call); a Comfy
Credits writer with the OpenRouter catalog down warns and continues, because
the paid host is `api.comfy.org`.

Adapters own `cloud_selectors()` -- candidate sets, not a second table.
Product rows (Flux Pro, Kling class-import, Sonilo) send `{node_key: {}}`.

No generate. No silent floor. No hiding engines from dropdowns.
