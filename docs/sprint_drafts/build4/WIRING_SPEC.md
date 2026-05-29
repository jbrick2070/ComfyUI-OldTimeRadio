# Build 4 Wiring Spec -- compose_exchange + repair-by-exchange-group

**Status:** DRAFT (docs/sprint_drafts/build4/). Not imported by ComfyUI.
**Integration order:** LAST. Land Builds 1, 2, 3 and prove their gates
first. Build 4 is the most coupled: it consumes Build 3's contracts and
calls Build 2's Tier-A validator over Build 2's slot format.

This spec describes how the draft `_otr_compose_exchange.py` replaces the
per-beat `compose_line` loop, the interfaces it expects from Builds 2/3,
the commit invariants, and the regression gates.

---

## 1. What replaces what

Today the writer renders one line per beat. Two call sites exist:

* Legacy one-shot: `OTR_LedgerScriptWriter.py` per-beat `_OTRLC.compose_line`.
* Dormant Stage 2 multiturn: `nodes/_otr_stage2_call.compose_line`
  (per-beat best-of-N), dispatched through
  `nodes/_otr_wave0_multiturn.dispatch_compose_line`.

Both render **one slot at a time**. Build 4 introduces **grouped
rendering**: collect 2-3 consecutive **voiced** slots into a beat group,
render them as one exchange, then commit one block per slot.

### Replacement shape (in-loop, pseudo)

```
groups = group_voiced_beats(voiced_beats, size=2..3)   # writer helper, Build 4 wiring
for group in groups:
    result = compose_exchange(
        beat_group = [VoicedSlot(d###, SPEAKER, intent, target_words), ...],
        contracts  = build3_contracts_by_slot_id,        # Build 3
        prior_lines = committed_display_lines[-K:],       # truncated, see sec 5
        cast        = stage1_plan.cast,
        generate_fn = creative_generate_fn,               # writer's creative slot
        tier_a_check = tier_a_adapter,                    # Build 2, adapted (sec 3)
    )
    if result.status in ("ok", "ok_repaired"):
        for slot_id in result.slot_ids:
            commit_one_block(slot_id, result.lines[slot_id])   # one row per slot
    else:  # status == "fail"
        for slot in group:
            legacy_compose_line(slot)   # per-line fallback, PD1: never break audio
```

### Grouping rules

* Group **only voiced** slots (cast speakers). ANNOUNCER / MUSIC slots are
  NOT grouped -- they keep the existing dedicated passes, exactly as
  `_otr_wave0_multiturn.RESERVED_SPEAKERS` already routes them.
* A group is **2 or 3** voiced slots. A trailing single voiced slot
  (group of 1) should route to the existing single-line composer, not to
  `compose_exchange` -- the exchange writer's craft rules assume a back-
  and-forth.
* Do not cross an ANNOUNCER / MUSIC slot when forming a group: a reserved
  slot breaks the group boundary.

---

## 2. Build 3 contract shape consumed (read-only)

`compose_exchange` READS Build 3's `slot_drama_contract` to shape the
prompt. It does **not** validate the contract (Build 3 owns validation).

Fields read per slot (duck-typed; the draft mirrors them as
`SlotContract`):

| Field                       | Use in Build 4 |
|-----------------------------|----------------|
| `dialogue_slot_id`          | key / lookup (matches Build 2 `d###`) |
| `speaker`                   | sanity only (slot speaker is authoritative) |
| `line_job`                  | per-slot obligation line in the prompt |
| `hidden_pressure`           | "do NOT state outright" instruction |
| `concrete_detail_required[]`| pooled across the group -> the ONE grounding candidate list |
| `state_before` / `state_after` | not directly printed in v1; reserved |
| `must_turn: bool`           | adds the "this line must TURN the scene" instruction |

`contracts` may be passed as a **mapping keyed by slot id** (preferred) or
a **sequence of contract objects**; the draft handles both via
`_contract_for`.

**Integration must satisfy:** every voiced slot in a group has a contract
entry. A missing contract degrades the prompt (no line_job / pressure) but
does not crash -- the slot still renders from its beat intent.

---

## 3. Build 2 Tier-A integration

Build 2 ships a deterministic, format/integrity-only validator (slot
count, slot order, speaker match, empty line, per-line word floor --
nothing semantic). Build 4 calls it through the injected `TierACheckFn`:

```
tier_a_check(parsed_exchange: Mapping[slot_id, text],
             expected_slots: Sequence[VoicedSlot]) -> TierAResult
```

`TierAResult(ok: bool, reasons: list[str], failing_slot_ids: list[str])`.

**Adapter to write at wire-up:** Build 2's validator likely takes the
draft rows / slot rows directly. Wrap it:

```
def tier_a_adapter(parsed, slots):
    rows = [SlotRow(dialogue_slot_id=s.dialogue_slot_id,
                    speaker=s.speaker,
                    text=parsed[s.dialogue_slot_id]) for s in slots]
    v = build2_tier_a_validate(rows, expected_slots=slots)
    return TierAResult(ok=v.ok, reasons=v.reasons,
                       failing_slot_ids=v.failing_slot_ids)
```

`reasons` is load-bearing: on failure they are appended verbatim to the
repair prompt (`build_exchange_prompt(..., failure_reasons=...)`). Keep
them short and specific (e.g. `"slot d002 below word floor (4 < 6)"`).

**Note on parse vs Tier-A.** The draft's `parse_exchange` already enforces
slot **count / id** integrity (one block per slot, no missing/extra/dup).
Build 2's Tier-A can therefore focus on the per-line checks (word floor,
speaker match, non-empty). If Tier-A also re-checks count, that is
harmless redundancy.

---

## 4. The one-block-per-slot commit invariant

This is the core Build 4 contract (critique 8):

* **One slot in, one block out.** `result.lines` maps every
  `dialogue_slot_id` in the group to exactly one text block.
* A block **may** contain internal pauses (`...`, em-dashes) but is **one
  ledger row** -- commit it to a single `dialogue_slot_id`.
* **No added, dropped, reordered, merged, or renamed slots.**
  `parse_exchange` raises `ExchangeParseError` on any of these; the
  exchange then routes to repair, and if repair also drifts, to legacy
  fallback. A drifted exchange is **never committed**.
* `result.slot_ids` preserves the group's slot order; commit in that order.

**Only `status in ("ok", "ok_repaired")` is safe to commit.** A `"fail"`
result must trigger the legacy per-line fallback -- do not commit
`result.lines` on `"fail"` (it is partial / diagnostic only).

---

## 5. VRAM and context caution (14.5 GB ceiling)

* The exchange prompt is **longer** than a single-line prompt: it carries
  2-3 slot obligations + prior committed lines + repair reasons. Token
  budget defaults to `DEFAULT_EXCHANGE_MAX_NEW_TOKENS = 320`
  (vs 200 single-line).
* **Truncate `prior_lines` against `context_cap`** before calling
  `compose_exchange`. Pass the last K committed display lines, not the
  whole scene. The writer already computes `context_cap`; reuse it.
  Budget = `context_cap - prompt_overhead - max_new_tokens`.
* **Never `force_vram_offload()` between the first attempt and the repair
  pass.** Use `_flush_vram_keep_llm()` if anything must be reclaimed
  between LLM phases (project rule 2). The repair pass reuses the same
  loaded creative model -- no reload.
* All LLM loaders do a 1-token warmup pass; Build 4 introduces no new
  loader, so no new warmup is needed.
* `compose_exchange` makes **at most 2 generate calls** per group (original
  + one repair). With grouping (2-3 slots per call) the per-episode
  generate-call count DROPS vs per-line rendering, which is favorable for
  the VRAM ceiling.

---

## 6. Model slot (project rule 6)

Every LLM call in `_otr_compose_exchange.py` is tagged
`# LLM slot: creative` -- exchange rendering is creative-axis narrative
work (subtext, refusal, reversal). At wire-up:

* Pass the writer's **creative_writing_model** generate closure as
  `generate_fn`. Do **not** add a `model_id` widget to any node
  (forbidden-pattern sweep `docs/_s28_forbidden_sweep.py`).
* The Build 3 contract pass (if it runs an LLM) is **technical-slot** and
  is Build 3's concern, not Build 4's.

---

## 7. Concrete checklist gate (critique 9 -- no subjective read)

Build 4 is "done" against the plan's measurable gate, on N=3 first then
N=6-10 for the plateau claim:

| Check | Pass condition |
|-------|----------------|
| Slot drift | **Zero.** rows committed == draft slots; no added/skipped/renamed slot across the whole episode. |
| Exposition | **Fewer** exposition hits vs baseline (count the EXPOSITION_DUMP-style flags; no Tier-B promotion yet). |
| Scene movement | **>= 1 interruption / refusal / reversal per exchange.** |
| Commit clean | rows committed == draft rows, no unexpected legacy fallback spike. |
| VRAM | peak <= 14.5 GB. |
| Audio | byte-identical when `use_exchange=False`; audio re-verified each build. |

A high `"fail"` rate (many legacy fallbacks) is itself a red flag: it
means the exchange writer or Tier-A is mis-tuned. Track the
`status` distribution (`ok` / `ok_repaired` / `fail`) per episode in
meta diagnostics.

---

## 8. Regression gates (run after wiring)

```
# Bug Bible regression
python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v
# Core
pytest tests/test_core.py -v
# Audio byte-identity (MUST stay green; exchange OFF by default)
pytest tests/test_audio_byte_identical.py -v
# This draft's own suite
pytest docs/sprint_drafts/build4/test_compose_exchange.py -v
```

Default the exchange path **OFF** (feature widget `use_exchange=False`) so
the legacy byte-identity contract holds out of the box, mirroring the
Wave 0 `use_multiturn_dialogue` default. Flip on only behind the gate run.

---

## 9. Workflow JSON

Build 4 is **internal** (no new node class, no new socket) IF the feature
is gated behind an existing-style boolean widget on the writer. If a new
`use_exchange` widget is added to `OTR_LedgerScriptWriter`, the workflow
JSON must be re-wired to carry the new widget default (project rule 3):
verify widget name + default in the workflow JSON after the node change.
No new `model_id` widget -- the creative model id is already broadcast
from the writer's `creative_writing_model` output socket.
