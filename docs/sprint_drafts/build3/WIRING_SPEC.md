# Build 3 Wiring Spec -- `slot_drama_contract` + deterministic contract validation

**Plan:** `workflows/GO_FORWARD_PLAN_v10_four_builds_2026-05-28.md`, Build 3
**Status:** DRAFT. Module lives under `docs/sprint_drafts/build3/` and is NOT
auto-imported. This spec is the integration recipe; it does not move any file.
**Integration order:** AFTER Build 2 (slot-formatted output + Tier-A integrity
gate), BEFORE Build 4 (`compose_exchange`).

---

## 1. What this build adds

A **slot drama contract** per voiced dialogue slot: the per-line obligation the
Build 4 exchange writer must honor. The plan's reframed thesis (line 11) is that
the contract -- not the commit gate -- is where the quality lift comes from. The
gate only stops bad output from committing; the contract is what gives each line
a real job to do.

The LLM surface is deliberately tiny (plan critiques 4 & 5): of the eight
contract fields, **six are derived deterministically** and **two are LLM-written**.

---

## 2. Fields: derived vs LLM-written

| Field | Source | How |
| --- | --- | --- |
| `dialogue_slot_id` | **derived** | from the ledger line / beat row (`dialogue_slot_id`, `d###`). |
| `speaker` | **derived** | from the slot row (`speaker`, else `speaker_name`, else `name`). |
| `concrete_detail_required[]` | **derived** | rotated from the candidate pool = `ContinuityState.active_props` UNION `meta["news"]["key_terms"]`. The turning slot gets two, others one; empty pool -> empty list. |
| `state_before` | **derived** | from `DramaticState` + beat position. Turning slot: the opening tension (opposed wants / dramatic_question). |
| `state_after` | **derived** | from `DramaticState` + beat position. Turning slot: `DramaticState.ending_change`. Non-turning slots hold their state (before == after, allowed). |
| `must_turn` | **derived** | `True` iff `slot_id == DramaticState.costly_choice_beat`. |
| `line_job` | **LLM (technical slot)** | one clause: what the line is trying to DO. |
| `hidden_pressure` | **LLM (technical slot)** | one clause: the unspoken force under it. |

The LLM pass is bound to the two-field `SlotJobFields` schema under constrained
decode, so a bad model pass can corrupt at most two short strings -- the rest is
deterministic. (Plan: "derive deterministically wherever possible to limit
garbage-in".)

---

## 3. Where in the pipeline it runs

Inside `OTR_LedgerScriptWriter.run`, the prerequisites for the contract all
already exist before the Story Room writes:

1. `meta["news"]["key_terms"]` -- stamped by the news_interpreter (writer line
   ~2305 `meta["news"] = briefs.model_dump()`).
2. `continuity_state` (`ContinuityState`, has `active_props`) -- built at writer
   line ~3013 `build_continuity_ledger(...)`.
3. `meta["dramatic_state"]` (`DramaticState`) -- stamped at writer line ~2950.
4. Voiced slot rows with `dialogue_slot_id` -- stamped by
   `init_lines_from_outline` (writer line ~2915); the voiced slot ids are
   already collected at line ~2937.

**Insertion point:** after `build_continuity_ledger` and the `dramatic_state`
stamp, and **before the per-beat `LineRequest` render loop** (which begins around
writer line ~3286). At this point all four inputs are live and no line text has
been authored yet -- the contracts must exist before the writer composes.

```
news_interpreter  ->  meta["news"]["key_terms"]
build_continuity_ledger  ->  continuity_state.active_props
derive_dramatic_state_from_meta  ->  meta["dramatic_state"]
init_lines_from_outline  ->  voiced slot rows (dialogue_slot_id)
        |
        v
  [BUILD 3]  build per-slot contracts  ->  validate_episode_contracts
        |
        v
  per-beat LineRequest render loop  /  Build 4 compose_exchange
```

---

## 4. Inputs to pass

For each voiced slot `i` (0-based over the episode's voiced slots, in order):

```python
contract, source = build_slot_drama_contract(
    technical_generate_fn,          # constrained-decode closure, or None
    slot_row=voiced_line_row_i,     # ledger line OR beat row (dialogue_slot_id + speaker)
    slot_index=i,
    dramatic_state=meta["dramatic_state"],          # dict is fine
    beat_intent=(beat.intent or ""),                # the voiced beat's intent text
    active_props=continuity_state.active_props,
    key_terms=(meta.get("news") or {}).get("key_terms") or [],
    temperature=0.6,
)
```

Then validate the episode set once:

```python
ok, reasons = validate_episode_contracts(
    contracts, continuity_state.active_props,
    (meta.get("news") or {}).get("key_terms") or [],
)
```

Store the validated contracts keyed by `dialogue_slot_id` on `meta`
(suggested: `meta["slot_drama_contracts"]`) so Build 4 can look up each slot's
contract when composing the exchange.

---

## 5. The technical-slot LLM wiring (project rule 6)

- The call site is tagged **`# LLM slot: technical`** in
  `_otr_slot_drama_contract.py` (above `_SLOT_JOB_SYSTEM_PROMPT`). The pass is a
  structured JSON pass bound to `SlotJobFields` under constrained decode --
  per rule 6, schema-constrained passes route to the **technical** model slot.
- **No `model_id` widget.** This module exposes no node surface and no widget.
  At integration, build the closure from the writer's existing technical model:

  ```python
  from ._otr_constrained_generate import make_constrained_generate_fn
  technical_generate_fn = make_constrained_generate_fn(
      technical_cache_entry,           # loaded from the technical_model id
      SlotJobFields,
      heartbeat_label="SlotContract",
  )
  ```

  The technical model id is received via the writer's `technical_model` broadcast
  output socket -- consumers take a `STRING` input, never a new widget
  (rule 6). The forbidden-pattern sweep (`docs/_s28_forbidden_sweep.py`) must
  stay green: this module adds no `INPUT_TYPES` block and no `model_id` pick.

---

## 6. Validator rules (deterministic, runs before the contract is trusted)

`validate_contract(contract, active_props, key_terms) -> (ok, reasons)`:

1. **schema-valid** -- coerces to `SlotDramaContract`; catches bad `d###` slot id,
   missing/empty required fields, wrong types.
2. **non-empty** -- `speaker`, `line_job`, `hidden_pressure`, `state_before`,
   `state_after` non-empty after strip.
3. **detail subset** -- every `concrete_detail_required` entry is in
   `active_props` UNION `key_terms` (case-insensitive); no empty entries.
4. **turn must turn** -- when `must_turn`: `state_before != state_after`.
5. **turn must ground** -- when `must_turn`: at least one concrete detail present.

`validate_episode_contracts(...)` runs rules 1-5 per slot PLUS the cross-slot
invariant: **exactly one** slot carries `must_turn == True` (zero -> the episode
never turns; more than one -> ambiguous pivot).

**On fail:** `build_slot_drama_contract` regenerates the LLM pass **once**; if the
second pass still fails (or `generate_fn is None`), it falls back to
`build_minimal_contract` -- a deterministic contract built purely from
`DramaticState` + `beat_intent` that is constructed to pass `validate_contract`
for any well-formed slot row. So **no garbage contract reaches the writer**
(plan Build 3 gate). The `source` return value (`"llm"` / `"llm_regenerate"` /
`"minimal"`) should be logged for auditability and counted per episode.

---

## 7. How it feeds the writer prompt and Build 2's Tier-A manifest

- **Writer prompt (Build 4 `compose_exchange`):** for each slot in a beat group,
  inject `line_job` + `hidden_pressure` + the `concrete_detail_required` list as
  the per-slot obligation block, and the group's `state_before` ->
  `state_after` as the arc the exchange must travel. Build 4's "one concrete
  grounding per *exchange*" rule (plan critique 6) draws its grounding candidates
  from the union of the group's `concrete_detail_required` lists.
- **Build 2 Tier-A manifest:** the contract set is keyed by `dialogue_slot_id`,
  the same slot ids Build 2's Tier-A validator checks for count/order/speaker
  match. The contract's `speaker` is derived from the same slot row Build 2
  validates against, so the two surfaces agree by construction. The Tier-A
  manifest's slot list is the authoritative set of slot ids the contract builder
  iterates -- build one contract per Tier-A slot, no more, no fewer. This keeps
  "one slot = one committed text block" (plan Build 2, critique 8) consistent
  with "one slot = one contract".

---

## 8. Latency note

One technical-slot constrained-decode pass **per voiced slot**, each bounded to
`max_new_tokens=192` (two short strings). For a typical episode of ~12-24 voiced
slots that is 12-24 small passes. Each is far cheaper than a line-composition
pass (192 vs the Editor's 4096-token budget). To keep this off the critical path
it can run during the same model residency as continuity (the technical model is
already resident there). VRAM is unaffected -- no new model load; reuse the
resident technical cache_entry and `_flush_vram_keep_llm()` between phases as
usual. The deterministic-only path (`generate_fn=None`) is effectively free and
is the fallback if the contract pass ever threatens the 14.5 GB ceiling or the
run budget.

---

## 9. Regression gates (run after integration, per project rules)

- `python -m pytest "docs/sprint_drafts/build3/test_slot_drama_contract.py" -v`
  (this build's 24 unit tests; move alongside the suite at integration).
- Bug Bible regression:
  `python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v`
- Core: `pytest tests/test_core.py -v`
- Audio byte-identity: `pytest tests/test_audio_byte_identical.py -v`
  (Build 3 touches no audio path; this must stay byte-identical -- audio is king.)
- Forbidden-pattern sweep `docs/_s28_forbidden_sweep.py` stays green (no new
  `model_id` widget).
- LLM-slot audit: the new call site appears in the Two-Model Selector routing
  table tagged **technical**.

**Build 3 gate (plan line 42):** on N=3, every contract is schema-valid and
passes the deterministic sanity checks; no garbage contract reaches the writer.
The minimal-contract fallback guarantees the second half by construction; the
N=3 run confirms the LLM path produces schema-valid `line_job` /
`hidden_pressure` in practice and logs the `source` distribution.

---

## 10. Open questions for integration

1. **Slot-row source of truth.** The contract builder reads `dialogue_slot_id` +
   speaker from a "slot row". Confirm whether to iterate the **ledger line rows**
   (`led.data["lines"]` filtered to voiced) or the **voiced beat rows**. The
   adapter reads `speaker` / `speaker_name` / `name` and `dialogue_slot_id` off
   either, but the canonical iteration order (and the `slot_index` it implies)
   must match Build 2's Tier-A slot list exactly. Recommend: iterate the Tier-A
   slot manifest so the two builds share one ordering.
2. **`beat_intent` per slot.** A slot maps to one voiced beat; confirm the beat's
   `intent` is the right `beat_intent` to feed (vs `mood` or a composed value).
   The minimal fallback embeds it verbatim into `line_job`, so it should read as
   an action.
3. **State phrasing depth.** `state_before`/`state_after` are derived only from
   `DramaticState` (opposed wants + ending_change) plus position. Non-turning
   slots currently all hold the same generic "tension carried forward" state.
   Open question: does Build 4 need per-beat state granularity (e.g. derived from
   each beat's `intent`/`arc_phase`), or is the turn-vs-hold distinction enough?
   Deferred until Build 4's prompt is drafted; the schema already supports richer
   per-slot states with no change.
4. **Where the contracts are stored on `meta`.** Proposed key
   `meta["slot_drama_contracts"]` (list of `model_dump()` dicts, or dict keyed by
   slot id). Confirm against how Build 4 will look them up and whether they need
   to survive a reroll (`_otr_reroll`) -- if a reroll changes the slot set, the
   contracts must be rebuilt, mirroring `_rebuild_continuity_slice`.
5. **Detail rotation vs. Build 4 anti-spam weighting.** This build assigns one
   detail per non-turning slot (two on the turn). Build 4 weights grounding
   per-exchange, not per-line (critique 6). Confirm the per-slot detail
   requirement is treated by Build 4 as a *candidate pool for the exchange*, not
   a hard per-line mandate, to avoid object spam.
