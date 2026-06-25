# KILL 1 -- LIVE 309w OBS SMOKE (2026-06-24)

**Commit under test:** `adb47483` (KILL 1 body-output gate). Branch `v2.0-alpha`, HEAD == origin.
**Workflow:** the canonical `workflows/otr_scifi_16gb_full.json`, loaded fresh, patched by name
(`target_words=309`, `num_characters=2`, `act_count=3`).
**Boot:** fresh headless server on :8000 via `scripts/_otr_soak_server_launch.cmd <log> LTX`
(LTX lane: LTX-AV bookends + visualizer body + flux portraits; HuMo not reached -- the director routes
all roles to ltx_av / visualizer). Box reset + verified clean before boot (:8000 free, VRAM ~1.6 GB).

## Result: PASS -- full episode end-to-end, OBS final published

- `Prompt executed in 00:31:09`, zero tracebacks.
- OBS final: `otr/obs/signal_lost_corks_dance_20260624_170035_silent_procgen_blended_final.mp4`
  (70.3 MB; archival PCM-byte-identical twin under the episode dir).
- `audio_byte_identical OK (02e30c5cb09d)` -- the frozen audio spine invariant held.
- Episode: "Cork's Dance" (a wine-aging / disputed-record premise), 18 beats, 16 voiced character beats,
  111 s master audio, freeze verdict `frozen_with_warns` (reviewer clean), critic arc `mid_collapse`.

## The body gate fired LIVE (the point of KILL 1)

From the shipped ledger `meta.story_quality`:

| field | value |
| --- | --- |
| style_slug / ending_tag | `overnight_jazz_host_mystery` / `revelation` |
| body_gate_rerolls (validated + shipped) | 1 |
| body_gate_failed (reroll invalid -> kept original, LOUD) | 10 |
| **body_gate_ungrounded_crisis (SHIPPED-body density)** | **2** |
| grounded palette size | 50 tokens |
| distinct conflict objects | 5 |

The gate validated every character line against the grounded premise palette, attempted ONE targeted
reroll when a line leaned on generic machinery or skipped its conflict object, and shipped the reroll only
when it validated (else kept the original, logged LOUD). Live log examples:

```
body-gate reroll did not validate for beat b011; keeping original (ungrounded_crisis:button)
body-gate reroll did not validate for beat b014; keeping original (ungrounded_crisis:button,missing_conflict_object:the authorization everyone needs)
body-gate reroll did not validate for beat b015; keeping original (missing_conflict_object:the one signature that settles it)
```

## Reading

- **The console standoff is effectively gone in the SHIPPED body**: only 2 ungrounded crisis tokens
  ("button" x2) across 16 character beats. That is the metric KILL 1 targets, and it is near-zero here.
- **10 of 11 flagged beats kept the original** because the weak local writer (mistral-nemo) could not
  produce a grounded rewrite even with the split hint. Most of those are `missing_conflict_object` (the
  line is premise-grounded but does not echo the exact seed-keyed object) -- a softer miss than machinery,
  and the model-ceiling reality the plan flagged. The deferred **model-capability gate** /
  frontier-writer is the belt-and-suspenders for that prose ceiling.
- KILL 1 does its job model-agnostically: it CATCHES + measures + reduces ungrounded machinery, and never
  ships nothing (deterministic keep-original). It is not a prose-quality lift -- that is downstream.

## Next (operator-gated)

The KILL-1 **LIVE RE-SOAK** (gemma + mistral, 320 w) measuring SHIPPED-body crisis-noun density ON vs OFF
-- the change that targets gemma's console standoff specifically. Do NOT start KILL 2 (StoryContract)
until the re-soak is clean. The ON/OFF comparison is exactly the kill-switch:
`OTR_ENABLE_STYLE_GRAMMAR=0` = grammar + gate OFF (the AI's own story, byte-identical to pre-grammar);
default = ON (the scaffolded story).

Box reset after the run (selective CIM kill of the ComfyUI server), :8000 FREE, VRAM ~1.4 GB.
