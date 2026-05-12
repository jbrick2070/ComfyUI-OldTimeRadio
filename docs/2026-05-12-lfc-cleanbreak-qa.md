# LFC Clean-Break — Reviewer QA

**Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
**HEAD:** `332f0e3` (tag `v2.0-alpha-cleanbreak` pending re-tag at HEAD; prior tag at `46322bd` predates the B1 BLOCKER fix + G/D follow-up series)
**Series:** 16 commits (12.3 → 12.18) + annotated tag (pending re-tag)
**Premise:** v2.0-alpha is a clean break. No legacy back-compat
surface. Every cascade phase has its own record; cascade unloads
the LLM at exit (try/finally, even on exception); per-phase
telemetry on `meta.freeze_phase_telemetry` derived from the three
bucket lists; standalone Phase 4/5/6 nodes next to the main
cascade with G1+G2 prerequisite/double-run interlocks. Core
workflow has no third-party UI-extension dep (pysssss previews
dropped in G5).

Paste this whole doc into ChatGPT + Gemini. Disagreements between
them are the signal.

---

## 1. What landed (one-line summary per commit)

| # | Hash | Change |
|---|------|--------|
| 12.3 | `690da2c` | Legacy purge: `_RENAME_ALIASES` dict deleted, `nodes/OTR_LedgerScriptReviewer.py` shim deleted, `reviewer_verdict` field dropped from synthetic error JSON, docstrings refreshed |
| 12.4 | `f4eaac8` | B6: `meta.cleanup_passes` split into three buckets — `audit_passes` / `cleanup_passes` / `readiness_passes` + `all_phase_passes(meta)` helper |
| 12.5 | `5d8225c` | B14 + C7: cascade calls `_otr_model_loader.unload_llm()` at exit so VRAM is released before HuMo / SignalLostVideo load |
| 12.6 | `bc8cb0c` | C3: compact per-phase telemetry on `meta.freeze_phase_telemetry` `[{phase, bucket, skipped, changed, warnings, edits_proposed, edits_applied}]` |
| 12.7 | `890e3f7` | C5 + C6: `scripts/lfc_wiring_smoke.py` runs widget defaults + RETURN_NAMES + legacy-token + news_used chain + last_link_id checks. Pytest wrapper gates it in `pytest tests/` |
| 12.8 | `4a7da85` | C2: `freeze_verdict` slot 4 wired to `ShowText\|pysssss` node 63 via link 111. **Superseded by 12.15 (G5)** -- the wire was intentionally removed; cascade output port `freeze_verdict` remains unwired on the node so operators can self-wire a preview. |
| 12.9 | `ea83e48` | C9: `estimated_minutes` slot 3 wired to `ShowText\|pysssss` node 64 via link 112. **Superseded by 12.15 (G5)** -- same rationale as 12.8. |
| 12.10 | `dd2ec26` | C4: runtime smoke test asserts writer → cascade → SignalLostVideo `news_used` is byte-identical end-to-end. The cascade passes `news_used` through as an unmodified STRING (no re-serialization), so the stronger byte-identical invariant holds and is the one the test pins. |
| 12.11 | `46322bd` | C1: standalone `OTR_LFCPhase4Scene` / `OTR_LFCPhase5Voice` / `OTR_LFCPhase6Arc` nodes — each defaults OFF, peeks the ledger, delegates to its phase module |
| 12.12 | `9a65dba` | B1 BLOCKER: cascade body wrapped in try/finally so `_OTRML.unload_llm()` always runs on exit. Adds `meta.freeze_unload_ok` (True / False) for soak diagnostics; exception still propagates (ComfyUI red-node convention). |
| 12.13 | `eb0f4af` | G3: new `nodes/_otr_lfc_phase_verdicts.py` module exporting a shared `PhaseVerdict` Literal across 10 strings (UNIVERSAL + INTERLOCK + PHASE_SPECIFIC sets). Phase 5 voice-drift report gains `failed` flag + `failure_reason`; standalone node surfaces `failed` verdict before edit-count bucketing. |
| 12.14 | `b15446f` | G1 + G2: `force: BOOLEAN` widget on Phase 4 / 5 / 6 standalone nodes. G2 double-run interlock on all three. G1 prerequisite check (Phase 5 only; Phase 6 has documented synopses fallback). |
| 12.15 | `de8a094` | G5: dropped `ShowText\|pysssss` nodes 63 + 64 and links 111 + 112. v2.0-alpha is a foundational alpha; a third-party UI extension dep for basic telemetry is the wrong shape. Cascade output ports stay on the node (unwired). `last_node_id: 64 -> 62`, `last_link_id: 112 -> 110`. Supersedes 12.8 + 12.9. |
| 12.16 | `df1f779` | G4: regression pin on telemetry derivation -- `meta.freeze_phase_telemetry == build_phase_telemetry(meta)`. Pure function over the three bucket lists; pin prevents future drift between source-of-truth (buckets) and view (telemetry). |
| 12.17 | `e248e7f` | D4: smoke-script hardening. LEGACY_TOKENS extended to 5 entries; new `WORKFLOW_LEGACY_CLASS_TYPES` set + `_scan_workflow_class_types()` walks every JSON under `workflows/` for stale `type` / `class_type` strings. Bumps from 6 to 7 checks. Plus legacy-name purge across 14 tracked files. |
| 12.18 | `332f0e3` | D1 + D2 + D5 + Q6 doc fixes (this section). Tag pointer corrected, acceptance criterion 6 strengthened to equality check, byte-identical wording picked, Q6 marked answered by 12.17. |

---

## 2. Wiring snapshot

### Cascade node id 62 — `OTR_LedgerFreezeCascade`

```json
"widgets_values": [
  "mistralai/Mistral-Nemo-Instruct-2407",  // 0  model_id
  false,                                    // 1  enable_phase_3_polish
  false,                                    // 2  polish_announcer_beats
  false,                                    // 3  enable_phase_4_scene_coherence
  false,                                    // 4  enable_phase_4_5_smart_suggestion
  false,                                    // 5  enable_phase_5_voice_drift
  false,                                    // 6  enable_phase_6_episode_arc
  true,                                     // 7  enable_phase_7_audio_readiness
  true,                                     // 8  enable_phase_8_video_readiness
  14.0                                      // 9  vram_ceiling_gb
],
"outputs": [
  {"name": "script_text",       "links": [1]},
  {"name": "script_json",       "links": [2, 12, 16, 19, 24]},
  {"name": "news_used",         "links": [110]},     // -> SignalLostVideo
  {"name": "estimated_minutes", "links": []},        // unwired (G5, 12.15)
  {"name": "freeze_verdict",    "links": []}         // unwired (G5, 12.15)
]
```

### Preview nodes intentionally absent (commit 12.15, §G5)

The 12.15 commit dropped two `ShowText|pysssss` nodes (previously
ids 63 + 64) and their links (111 + 112). v2.0-alpha is a
foundational alpha; requiring a third-party UI extension for basic
verdict telemetry is the wrong shape. Cascade outputs
`freeze_verdict` (STRING) and `estimated_minutes` (INT) remain on
the node so operators who DO have pysssss installed can wire their
own preview -- the core workflow no longer carries the dep. Data
also lives on `meta.freeze_verdict` / `meta.estimated_minutes` /
the saved ledger JSON.

### news_used link chain (W2 fix in commit 12.1, pinned by smoke)

```
writer(1).news_used --[108]--> cascade(62).news_used
cascade(62).news_used --[110]--> SignalLostVideo(12).news_used
```
Pre-fix bypass link 18 is absent. `last_link_id = 110`,
`last_node_id = 62` (both shrunk after G5 drop in 12.15).

### Three new standalone phase nodes

| Class | Category | Defaults |
|-------|----------|----------|
| `OTR_LFCPhase4Scene` | `OldTimeRadio/v2` | `enable=False`, `model_id=Mistral-Nemo` |
| `OTR_LFCPhase5Voice` | `OldTimeRadio/v2` | same |
| `OTR_LFCPhase6Arc`   | `OldTimeRadio/v2` | same |

Each is an OPTIONAL entry point. The main cascade still owns the
full chain (Phase 0 → 1+2+9 → 3 → 4 → 4.5 → 5 → 6 → 7 → 8 → 10);
the standalone nodes let operators rerun just one phase against
the current ledger without re-running the cascade.

---

## 3. Code snapshot — non-obvious contracts

### Per-phase meta bucket routing (B6 split, commit 12.4)

```python
_PHASE_BUCKETS = {
    "phase_0_gap_audit_pre":              "audit_passes",
    "phase_10_gap_audit_post_and_freeze": "audit_passes",
    "phase_1_2_9_reviewer_composite":     "cleanup_passes",
    "phase_3_per_line_polish":            "cleanup_passes",
    "phase_4_per_scene_coherence":        "cleanup_passes",
    "phase_4_5_smart_suggestion":         "cleanup_passes",
    "phase_5_voice_drift":                "cleanup_passes",
    "phase_6_episode_arc":                "cleanup_passes",
    "phase_7_audio_readiness":            "readiness_passes",
    "phase_8_video_readiness":            "readiness_passes",
}
```

Soak diagnostics call `_otr_freeze_cascade.all_phase_passes(meta)`
to get the chronological concatenation across all three buckets.

### Per-phase telemetry shape (C3, commit 12.6)

```json
"meta": {
  "freeze_phase_telemetry": [
    {
      "phase":          "phase_3_per_line_polish",
      "bucket":         "cleanup_passes",
      "skipped":        false,
      "changed":        false,
      "warnings":       0,
      "edits_proposed": 0,
      "edits_applied":  0
    },
    ...
  ]
}
```

`skipped=True` iff the phase's failures list carries
`stub_bypassed` / `terminal_skipped` / `enable_false`. The output
STRING `freeze_verdict` stays the verdict literal — only meta
carries the per-phase detail.

### Cascade exit unload (B14, commit 12.5)

```python
try:
    _OTRML.unload_llm()
except Exception as exc:
    log.warning("[OTR_LedgerFreezeCascade] unload_llm at cascade exit raised (%s); ...", exc)
```

Placed AFTER `disp = run_freeze_cascade(...)` returns AND
AFTER `json.dumps(led.data)` so the verdict + ledger snapshot
are computed before letting go of the model. Wrapped in
best-effort try/except — unload failure does not break the
cascade return.

### Standalone phase node skeleton (C1, commit 12.11)

```python
def run(self, script_json="", enable=False, model_id=DEFAULT_MODEL_ID):
    if not enable:
        return (script_json or "{}", "skipped")
    cache_entry = _OTRML.load_llm(model_id=model_id)
    generate_fn = _OTRML.make_generate_fn(cache_entry)
    try:
        rep = _LFC_P5.phase_5_voice_drift(generate_fn, led, enable=True)
    finally:
        _OTRML.unload_llm()
    return (new_json, verdict)
```

---

## 4. Validity questions for the round-robin

Answer YES / NO / CONCERN with one sentence of reasoning. Don't
overthink — these target the specific risk surface this series
introduces.

### Q1 — Bucket split semantics
Does the `audit_passes` / `cleanup_passes` / `readiness_passes`
mapping in `_PHASE_BUCKETS` (commit 12.4) match the actual
semantics of each phase? Specifically: is Phase 1+2+9 (the 3-pass
reviewer composite) correctly classified as `cleanup_passes`, or
should it be `audit_passes` since the auditor runs in Pass 1 +
Pass 3?

### Q2 — Cascade-exit unload placement
The `unload_llm()` call runs AFTER `disp = run_freeze_cascade(...)`
returns. The cascade itself (`_otr_freeze_cascade.run_freeze_cascade`)
calls `review_ledger` which calls `generate_fn`. By the time we
unload, the cascade is done with the model. Is there any code
path inside the cascade that needs the model AFTER `run_freeze_cascade`
returns? (e.g. would the `json.dumps(led.data)` or
`assemble_script_text_from_ledger` touch torch tensors?)

### Q3 — Standalone phase node + main cascade contention
A user could enable Phase 5 on the main cascade widget AND drop
an `OTR_LFCPhase5Voice` standalone node downstream with its own
`enable=True`. Phase 5 would run TWICE in one workflow execution.
Is that operationally acceptable, or do the standalone nodes need
a check that the corresponding cascade widget is OFF?

### Q4 — ShowText|pysssss dependency
The two preview nodes (id 63, 64) use `ShowText|pysssss` from
the pysssss extension. The plan recommended core-only. Operators
without pysssss installed see two missing-node warnings on
workflow load. Is the trade-off acceptable, or should these be
swapped to a core fallback (or removed in favor of inspecting
the saved ledger JSON directly)?

### Q5 — Standalone node verdict literals
Each standalone phase node returns a `phase_N_verdict` STRING
output (e.g. `"completed_with_edits"`, `"completed_no_edits"`,
`"failed"`, `"skipped"`, `"no_ledger"`). These literals are
NOT in any shared enum — they're per-node strings. Downstream
nodes that branch on the verdict would need to know each node's
literal set. Is that acceptable as-is, or should the literals
share a common typing (e.g. a `Literal` type alias) so future
consumers have one source of truth?

### Q6 — Legacy purge completeness
**ANSWERED by commit 12.17 (D4 hardening).** `scripts/lfc_wiring_smoke.py`
LEGACY_TOKENS now scans for: `OTR_LedgerScriptReviewer`,
`OTR_Gemma4Director`, `OTR_LLMScriptWriter`, `OTR_Gemma4ScriptWriter`,
`_RENAME_ALIASES`. Plus a new workflow-JSON scan
(`WORKFLOW_LEGACY_CLASS_TYPES` against every saved workflow's
`type` / `class_type` field) catches the case where a stale
workflow file references a renamed class id. `script_parse_json`
stays under `test_legacy_contract_retired.py` where it lives now.

### Q7 — `meta.freeze_phase_telemetry` cost
The telemetry array contains one entry per phase (10 entries on
a clean run). Plus `audit_passes`, `cleanup_passes`,
`readiness_passes` buckets carry the full phase records. So the
same per-phase data lives in two places on `meta` (buckets +
telemetry array). Is the duplication worth the soak-grep
convenience, or should the telemetry array reference into the
bucket records by index?

---

## 5. Acceptance criteria

For the series to be accepted, both reviewers should agree:

1. `scripts/lfc_wiring_smoke.py` exits 0 (7 checks pass, after the
   12.17 hardening commit -- workflow-JSON class-type scan added).
2. `pytest tests/test_lfc_*.py tests/test_phase3_ledger_reviewer.py
   tests/test_workflow_json_guardrails.py tests/test_legacy_contract_retired.py`
   returns 452 passed / 5 skipped / 0 failed.
3. Bug Bible regression holds 23 passed / 1 skipped / 2 xfailed
   (skip/xfail split may shift by environment; the only firm
   contract is "0 failed").
4. `grep -rn "OTR_LedgerScriptReviewer" nodes/ tests/ __init__.py`
   returns ZERO hits outside the allow-list (acceptance test file +
   smoke script + their pytest wrapper).
5. Workflow JSON loads in ComfyUI Desktop without missing-node
   warnings on a clean ComfyUI core install (no third-party UI
   extensions installed). 12.15 dropped the pysssss previews;
   cascade outputs `freeze_verdict` + `estimated_minutes` remain
   on the cascade node (unwired) so operators with pysssss can
   self-wire previews -- the core workflow no longer requires it.
   The new 7th smoke check (12.17 `_scan_workflow_class_types`)
   also verifies no saved workflow node carries a legacy
   `type` / `class_type` string.
6. `v2.0-alpha-cleanbreak` tag resolves to the **current HEAD**
   (`332f0e3` after the 12.12-12.18 follow-up series; the
   pre-existing `46322bd` tag predates the B1 BLOCKER fix and
   the G/D fixes and MUST be moved before reviewers verify).
   Check with an explicit equality, not a list-only existence
   test:
   ```bash
   test "$(git rev-list -n 1 v2.0-alpha-cleanbreak)" = "$(git rev-parse 332f0e3)"
   ```
   `git tag --list v2.0-alpha-cleanbreak` only proves the tag
   exists; the equality above proves it points at the right
   commit.

---

## 6. What to send

Just this doc. ~250 lines. The workflow JSON is 47 kB and most of
it is unchanged HuMo/LTX/FLUX loader nodes that have nothing to do
with the clean-break series.

If a reviewer pushes for code-level inspection on a specific commit,
give them the commit hash and one of:
- `nodes/_otr_freeze_cascade.py` (orchestrator + bucket routing +
  telemetry helper)
- `nodes/OTR_LedgerFreezeCascade.py` (cascade ComfyUI node)
- `nodes/OTR_LFCPhase{4Scene,5Voice,6Arc}.py` (standalone nodes)
- `scripts/lfc_wiring_smoke.py` (CI gate)

---

**End.**
