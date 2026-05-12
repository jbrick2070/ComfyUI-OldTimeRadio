# LFC Cascade Wiring — Clean-Break Go-Forward

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha` / `c1be3f0`
**Workflow:** `workflows/otr_scifi_16gb_full.json`
**Cascade node:** `OTR_LedgerFreezeCascade` (id 62)
**Premise:** v2.0-alpha is a clean break. No legacy workflow JSONs supported. No back-compat surface. Old class name, old output name, old `meta` keys — all gone.

---

## Clean-break protocol (gating step — do this first)

1. **`__init__.py`** — if `_RENAME_ALIASES` exists, delete it (whole dict, not just the cascade entry).
2. **Shim file** — `nodes/OTR_LedgerScriptReviewer.py`: delete if present.
3. **`OTR_LedgerFreezeCascade.py:80`** — `_no_ledger_error_json()` currently emits `"reviewer_verdict": "needs_full_rerun"` into the synthetic error JSON's `meta` block alongside `freeze_verdict`. This is RUNTIME emission of legacy data. Delete the line. Drop the inline comment that justifies it.
4. **`OTR_LedgerFreezeCascade.py:3–6`** — refresh the class docstring. The current text describes a back-compat shim as live. Replace with a clean v2.0-alpha description, no mention of the rename or the old class name.
5. **Workflow audit** — every JSON under `workflows/` and `tests/` either gets regenerated against the current node set OR deleted. Already verified: `otr_scifi_16gb_full.json` has zero hits for `"OTR_LedgerScriptReviewer"` — its S&R property is `"OTR_LedgerFreezeCascade"`, so it's already clean-break-compliant.
6. **Repo-wide rename audit** — apply this table as a single search-and-fix pass across docs, ADRs, fixtures, in-house consumers:

   | Old identifier              | New identifier            | Lookup kind                    | Action          |
   |-----------------------------|---------------------------|--------------------------------|-----------------|
   | `OTR_LedgerScriptReviewer`  | `OTR_LedgerFreezeCascade` | class name (string + import)   | rename in place |
   | `reviewer_verdict`          | `freeze_verdict`          | output port name / meta key    | rename in place |
   | slot index 4                | slot index 4              | positional                     | unchanged       |

   Index-based wiring is unaffected — only string-keyed lookups need touching.

7. **Smoke-check guard** — add to `smoke_check.py`: assert `"OTR_LedgerScriptReviewer"` and `"reviewer_verdict"` are absent from the repo (source files only — exclude `.md` and `# ` / `"""` comment lines, since the rename's history will stay in docstrings unless explicitly purged). Decide once whether comments are in or out and document the choice.
8. **Tag the cutover.** `v2.0-alpha-cleanbreak` annotated tag at the commit that completes steps 1–7.

---

## Bugs

### B1 — Widget count is internally inconsistent in the QA doc
Source review doc §1 says "gained 8 new widgets" but lists nine. Code confirms 10 total widgets (`model_id` + 8 BOOL + `vram_ceiling_gb`). State outright: "9 net-new widgets; `model_id` pre-existed on the prior reviewer node."

### B2 — Acceptance count mismatch
Source doc §6 says "answer YES to all six," §5 has Q1–Q7. Rewrite as "all acceptance criteria" OR add a seventh criterion for the `freeze_verdict` orphan-output decision.

### B3 — "1:1 kwarg" claim is false for `model_id`
Source doc §4 leads with "each widget on the cascade node maps 1:1 to a kwarg on `run_freeze_cascade`" but the table itself shows `model_id` drives `make_generate_fn` + `make_polish_generate_fn` — a factory dep, not a kwarg pass-through. Either drop the 1:1 framing or split `model_id` out with its own note.

### B4 — `last_link_id` semantics misstated, check is too weak
Two issues at acceptance #4. Semantics: `last_link_id` is monotonic — the highest link id EVER assigned, not the highest currently present. After any link deletion, `last_link_id > max(live link ids)`. Coverage: even with correct semantics, `last_link_id ≥ 110` is necessary but not sufficient. Strengthen to: link objects 108 AND 110 both exist in `workflow.links` AND point to exact source/target sockets per §3 (writer[2]→cascade[2], cascade[2]→SignalLostVideo[2]). Both verified in current JSON.

### B5 — `news_used` passthrough: claim needs qualification
`OTR_LedgerFreezeCascade.run()` returns `news_used or ""`. Behavior: non-empty string passes through unchanged (byte-identical); `None` or `""` normalizes to `""`. Rewrite §3's "byte-identical pre/post fix" as "parsed equality always holds; byte-identical iff input is non-empty." Smoke test in C4 below remains the right belt-and-suspenders check.

### B6 — `meta.cleanup_passes` is mis-scoped
`_otr_freeze_cascade.py:171` (`_stamp_phase_record`) appends to `meta.cleanup_passes` for EVERY phase including 7 (audio readiness) and 8 (video readiness). "Cleanup" reads as polish/Phase-3 scope. Phases 7 and 8 are readiness gates, not cleanup. Rename to `meta.phase_passes` OR split into `cleanup_passes` (3 / 4 / 5 / 6) and `readiness_passes` (7 / 8). Clean-break is the right moment.

### B7 — `estimated_minutes` output is orphaned
Slot 3 (`estimated_minutes`) has `links=[]` in the JSON. Input is wired (link 109). Passthrough exists but nothing consumes it. Decide one of: route to a preview/telemetry node, or remove the output port. Stamp the contract.

### B8 — `freeze_verdict` is orphaned
Slot 4 (`freeze_verdict`) has `links=[]`. Operator can only inspect cascade decisions by opening the saved ledger JSON. Covered by C2 below.

### B9 — "Output socket 4" is ambiguous
Source doc §1 says "output socket 4." §2 shows `slot_index: 4`, which is 0-indexed → the FIFTH output. Reviewer reading §1 in isolation counts to slot 3. Clarify once: "slot_index 4 (zero-indexed; the fifth output port)."

### B10 — Q3 can't be falsified from the doc as written
Q3 asserts `writer.news_used.links = [108]` but the source doc never includes the writer node's JSON slice or its class name (verified: writer is `OTR_LedgerScriptWriter`, id 1, `news_used` is output slot 2). Include a writer slice and a SignalLostVideo input slice (~20 lines).

### B11 — Pre-fix bypass-wire deletion is asserted but not shown
§3 describes a pre-fix direct writer → SignalLostVideo wire but never proves the old link was deleted rather than left in parallel. Add: "the pre-fix direct link is absent from `workflow.links`." Verified in current JSON — no parallel wire exists, but the QA doc should still state it explicitly. Promote to acceptance criteria.

### B12 — "BOOL vs BOOL silent bug" overreaches
§5 Q1 claims widget-order mismatch is silent "because the types happen to line up." Slot 0 (STRING) and slot 9 (FLOAT) would NOT load silently if swapped — ComfyUI's widget hydration type-mismatches and resets to default or errors. Silent-misalignment risk is bounded to slots 1–8 (the BOOL block). Tighten the claim.

### B13 — Inline `//` comments in JSON
§2's `widgets_values` block uses `// comment` annotations. Valid as documentation, invalid JSON. Copy-paste into a validator parse-errors. Flag as "annotated, not parseable" or move to a separate index table.

### B14 — Cascade never unloads the LLM at exit
`OTR_LedgerFreezeCascade.run()` returns without calling `_otr_model_loader.unload_llm()`. The model stays cached in VRAM after the cascade finishes. Verified: zero hits for `unload`, `empty_cache`, `del ... model`, or `torch.cuda` anywhere in the cascade orchestrator (`_otr_freeze_cascade.py`). On a 14GB ceiling on a 5080 Laptop, this is the actual OOM risk — `OTR_SignalLostVideo` and HuMo downstream inherit a VRAM-loaded LLM they didn't ask for. Fix: call `unload_llm()` at the end of `OTR_LedgerFreezeCascade.run()` (after the verdict is computed and the ledger serialized). Trade-off: next cascade run pays the model-load cost again — acceptable for soak; profile later if it bites.

---

## Suggested Changes

### C1 — Split the LLM-heavy phases into per-phase nodes
Operator-control and error-isolation case. Current single-node design forces an all-or-nothing run; a Phase 5 LLM failure inside `run()` logs and skips but the operator can't retry just Phase 5 without rerunning the entire cascade. Splitting Phases 4, 5, 6 into individual ComfyUI nodes gets:

- Per-phase skip/rerun from the canvas without rerunning upstream.
- Per-phase failure visibility (red node on the canvas instead of a buried log warning).
- Cleaner per-phase telemetry (each node owns its own `meta.phase_N_record`).

Keep Phases 7 / 8 (cheap, deterministic) bundled in a readiness node. Phase 4.5 is currently regex-only (`_otr_lfc_smart_suggestion.py` runs with `generate_fn=None` by default) and stays bundled with the deterministic phases. Cost: ~+3 nodes of graph clutter. Gain: real operational control.

Note: this is NOT a VRAM argument. The cascade uses one model for all LLM phases — there's no per-phase model swap to enable. The VRAM problem is B14, not the phase structure.

### C2 — Route `freeze_verdict` to a NATIVE preview node
Slot 4 currently has `links=[]`. Operator inspection requires opening the saved ledger JSON. Route slot 4 to a core ComfyUI text-preview node (`ShowText` is core; `PreviewAny` is the generic fallback). Do NOT use `ShowText|pysssss` — third-party dep for a debug surface. Belt-and-suspenders: write `meta.freeze_verdict` into the saved ledger unconditionally so post-hoc soak analysis doesn't need the canvas.

### C3 — Structure `freeze_verdict` payload as compact phase telemetry
`freeze_verdict` is currently a verdict literal (`frozen_clean`, `frozen_with_warns`, etc). Useful but coarse. Extend the saved-ledger companion `meta.freeze_verdict` (NOT the output STRING — keep that as the literal for now) to a JSON array of per-phase records:

```json
[
  {"phase": "phase_3_polish",          "skipped": true,  "changed": false, "warnings": 0},
  {"phase": "phase_7_audio_readiness", "skipped": false, "changed": true,  "warnings": 2}
]
```

Works whether the cascade stays monolithic or splits per C1. Soak telemetry stays grep-friendly.

### C4 — Runtime smoke test for `news_used` passthrough
Acceptance currently only checks wiring shape. Add a runtime test in `smoke_check.py`: load one fixture workflow, execute, assert at three endpoints:

```python
assert writer_output["news_used"] == cascade_output["news_used"]
assert cascade_output["news_used"] == signal_lost_input["news_used"]
```

Parsed equality is sufficient (per B5). Catches any quiet mutation in `run_freeze_cascade`.

### C5 — Automated defaults audit
Don't eyeball widget defaults across two LLM reviewers. Add to `smoke_check.py`:

```python
expected = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    False, False, False, False, False, False,  # phases 3, polish_announcer, 4, 4.5, 5, 6
    True, True,                                  # phases 7, 8
    14.0,                                        # vram_ceiling_gb
]
wf = json.load(open("workflows/otr_scifi_16gb_full.json"))
node = next(n for n in wf["nodes"] if n["id"] == 62)
assert node["widgets_values"] == expected, node["widgets_values"]
```

If C1 lands, expand to cover each split node. Currently passes against `otr_scifi_16gb_full.json`.

### C6 — `RETURN_NAMES` integrity check
Verify Python `RETURN_NAMES` order matches JSON output-slot order exactly:

```
("script_text", "script_json", "news_used", "estimated_minutes", "freeze_verdict")
```

Currently matches. A reorder in Python without a matching JSON update would scramble every downstream wire on next workflow load. Add as a smoke-check assertion alongside C5.

### C7 — Cascade-exit VRAM acceptance
Pair with B14's fix. Add to `smoke_check.py`: after the cascade returns, read GPU memory and assert it's within a small delta of the pre-cascade reading. Catches future regressions where someone adds an LLM-resident pathway without calling `unload_llm()`.

### C8 — Include writer + SignalLostVideo slices in the QA doc
Source doc §7 argues the cascade slice is enough. For Q3 specifically it isn't. Add ~20 lines: writer (`OTR_LedgerScriptWriter`, id 1) `news_used` output port, SignalLostVideo (`OTR_SignalLostVideo`, id 12) `news_used` input port. Q3 becomes falsifiable.

### C9 — Decide the `estimated_minutes` contract
B7 flagged this as orphaned. Pick one:
- **Wire it** to a telemetry/preview node — useful for soak observation of episode-length drift.
- **Remove the output port** if no downstream ever consumes it.

Stamp the decision in §2 of the source doc so reviewers stop tripping on it.

### C10 — Drop Q6 from the round-robin
With C1 adopted, Q6 ("single node vs per-phase") collapses to "split, done." Don't waste a round-robin turn on a settled design call. Delete §5 Q6 from the source QA doc.

---

## Acceptance criteria

1. Clean-break protocol steps 1–8 complete; `v2.0-alpha-cleanbreak` tag exists.
2. `widgets_values` array length == 10 (STRING + 8 BOOL + FLOAT). Auto-checked via C5.
3. `widgets_values` positional values exactly match `INPUT_TYPES` defaults. Auto-checked via C5.
4. Link objects 108 AND 110 both exist in `workflow.links` AND point to the exact source/target sockets specified in §3 (writer[2]→cascade[2], cascade[2]→SignalLostVideo[2]). Pre-fix direct writer→SignalLostVideo link absent.
5. `last_link_id` ≥ max link id currently present in `workflow.links`.
6. Cascade output slot 4 (`freeze_verdict`) has a downstream wire to a preview node AND `meta.freeze_verdict` is written to the saved ledger AND its payload conforms to the per-phase telemetry shape in C3.
7. Cascade output slot 3 (`estimated_minutes`) has a decided contract per C9 — either wired or removed.
8. Python `RETURN_NAMES` order matches JSON output-slot order exactly. Auto-checked via C6.
9. Runtime `news_used` passthrough smoke test (C4) passes.
10. Cascade-exit VRAM check (C7) passes — model unloaded before HuMo loads downstream.
11. Repo-wide grep for `"OTR_LedgerScriptReviewer"` and `"reviewer_verdict"` returns zero hits in source files (scope decision per clean-break step 7) — enforced by smoke check.

---

**End.**
