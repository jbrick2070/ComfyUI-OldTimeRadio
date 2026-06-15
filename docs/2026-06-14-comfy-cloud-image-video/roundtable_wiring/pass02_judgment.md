# Roundtable pass 02 (wiring plan — polish) — judgment log

Panel: GPT-5.5, Gemini 3.1 Pro, DeepSeek-v4-pro. Spend ≈ $0.25.
Grounding (richer than pass01): `role_compat.py`, `otr_image_gen_dispatcher.py`,
`eng_humo.py`, **`motion_common.py`** (the `MotionEngineBase` lease contract),
**`otr_video_director.py`** (the Director COMBO).
Verdicts: Gemini **yes-with-fixes**; GPT + DeepSeek **no** — driven by ONE real
grounded bug + several stale internal contradictions I introduced in pass01.

## THE BUG (all three, grounded) → folded

**Video GPU lease lives in `MotionEngineBase.prepare()`, not a dispatcher
"video render lease site."** `motion_common.py` `prepare()` unconditionally
calls `_GR.acquire(...)` then `self.load()`; `teardown()` releases and only
`wait_until_below_mb` when `had_lease`. So a cloud video engine that subclasses
`MotionEngineBase` (as planned) **acquires the local 16 GB lease and blocks it
for the whole network call.** My §4.2 "apply the same branch at the video render
lease site" pointed at the wrong layer.

Fix (grounded): cloud video adapters override `prepare()` — when
`is_network`, skip `acquire` + `load` and return
`{"engine_id": self.name, "lease": None, "patchers": self._patchers}`.
`teardown()` already handles `lease is None` safely (no release-wait). No
dispatcher video-lease change needed.

## Other CONFIRMED → folded

1. **Cost reservation leaks on a failed render (Gemini).** Reserve happens
   before render; on a network/5xx failure the dispatcher falls to the radio
   floor but the reservation stays deducted, spuriously tripping the ceiling. →
   `reserve_cloud_cost` returns the amount; the dispatcher **releases/refunds it
   on render failure** (reserve → commit-on-success / release-on-failure). Only
   committed spend counts.
2. **`declared_isolation="network"` in §8 S1 contradicts `is_network` in §4.1
   (all 3).** → Deleted from S1; `is_network = True` is the only marker.
3. **Stale §7 tests say `assert_usable` fails "over budget" (GPT, DeepSeek).** →
   `assert_usable` tests cover flag-off / missing-key ONLY; over-budget is tested
   via `reserve_cloud_cost` at the dispatcher.
4. **§6 widget-vs-env contradiction (GPT, DeepSeek).** → Ceiling is env-only
   (`OTR_CLOUD_CREDIT_CEILING`); the Director-widget mention is removed (no JSON
   edit, avoids positional `widgets_values` risk).
5. **Cloud image `prepare()` must be a no-op (GPT).** `_inprocess_gen_fn` calls
   `eng.prepare(None,None,None)` before `render_image`. `flux_gen1.prepare`
   returns `{"engine_id": name}` with no GPU — `cloud_flux_pro.prepare` must
   likewise do no GPU work / take no lease.
6. **Silent-MP4 is not free for a cloud clip (GPT).** HuMo guarantees silence by
   encoding frames itself; a cloud avatar may return an MP4 **with** audio. →
   the Kling adapter strips provider audio and `canonicalize` asserts
   `has_audio=False`; test with an audio-bearing input MP4.
7. **Cloud adapter decodes the provider result itself (DeepSeek).** It has NO
   `wrapper_bridge.images_to_uint8` (that's in-process Comfy tensors). → decode
   provider bytes to uint8 via PIL/numpy, or return the `.png` path; never
   assume the in-process helper.
8. **WAV duration probe (DeepSeek).** `cloud_kling_avatar` duration via stdlib
   `wave` (fallback `ffprobe`); resolve `audio_ref` as str-or-dict-with-`path`
   exactly like `eng_humo`'s `_ref_path`.
9. **Billing schema + staleness threshold (GPT, DeepSeek).** Pin
   `ledger["billing"] = {"currency","price_table_date","ceiling",
   "reserved_total","committed_total","requests":{request_id:{engine_id,units,
   unit_price,estimated_cost,status}}}`; price table older than 30 days or
   missing validity fails closed.
10. **Language: "offered" → "compatible at execute time" (GPT).** The Director
    COMBO is the full static registry (`otr_video_director.py` confirms); role
    fit is annotated/enforced at execute, not by filtering the widget. Wording
    fixed so a coder doesn't try to dynamically filter the dropdown.
11. **Provider job-id before polling (GPT should-fix).** Persist the provider job
    id in the billing/request record before polling; a retry after submission
    resumes/polls, never re-submits (no double charge).

## RESOLVED by the richer grounding (no longer "verify-at-build")

- `MotionEngineBase` contract is now grounded (the lease/ teardown bug above).
- `otr_video_director.py` confirms the COMBO is the full static registry → the
  "no JSON edit to make engines selectable" claim holds.

## STILL OPEN / gating

- §5 invocation-seam S0 spike (all 3, again): the actual Partner-node call
  mechanism + key + output discovery + idempotency must be resolved and folded
  BEFORE adapter coding. Make S0 a hard stop with a checked-in fixture; if no
  option works, stop before S1.

## CUTs accepted

- `declared_isolation="network"` (redundant with `is_network`). GPT.
- Director-widget ceiling in v1 (env-only). GPT.
- In-process Partner-node path is NOT mandatory — let the S0 spike pick the
  proven seam (POST or HTTP are fine). GPT.

## Convergence

Pass02 found ONE new material bug (the prepare() lease layer) + a cluster of
stale-text contradictions, all now folded. The architecture is unchanged and
sound. Recommend treating the plan as build-ready **after the S0 spike**; a
pass03 would only echo "looks good" until S0 resolves.
