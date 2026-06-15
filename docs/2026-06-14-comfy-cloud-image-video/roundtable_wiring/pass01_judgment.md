# Roundtable pass 01 (wiring plan) — judgment log

Panel: GPT-5.5, Gemini 3.1 Pro, DeepSeek-v4-pro. Spend ≈ $0.25.
Grounding: `role_compat.py`, `otr_image_gen_dispatcher.py`, `flux_gen1.py`,
`eng_humo.py`, `_otr_video_engines/registry.py`.
Verdicts: Gemini **yes-with-fixes**; GPT + DeepSeek **no** — but their "no" is
under-specification, not an architectural error. The adapter/role_compat/
CAPABILITIES/lease-skip architecture is sound; the gaps are spec precision.

## CONFIRMED (grounded) → folded into the plan

1. **`render_image` signature.** `flux_gen1.render_image(self, request,
   prepared=None)` and `_inprocess_gen_fn` calls `render_image(request,
   prepared)`. The plan's `render_image(request)` would `TypeError`. → Spec
   `render_image(self, request, prepared=None)`. GPT.
2. **`cloud_ltx2` role contradiction.** Plan said "all video roles" but declared
   `required_inputs=("init_image",)`, which `role_compat` excludes from
   `background_abstract` (text-only). → Re-declare `cloud_ltx2` as
   `required_inputs=("text_prompt",)` (fits ALL five roles) and use `init_image`
   opportunistically when the beat supplies it (i2v) else t2v. Matches operator
   intent AND the rule. GPT.
3. **Cost guard must NOT be in `assert_usable`.** Dispatcher calls
   `assert_usable(engine_id, role)` before the request exists. Plan
   self-contradicted (§2a/§7 vs §4.2). → `assert_usable` checks only flag + key;
   a dispatcher-level `reserve_cloud_cost(engine_id, request, episode_id,
   request_id)` runs after request assembly, idempotent by `request_id`, cache
   hits free, unknown/stale price fails closed. GPT + DeepSeek.
4. **NVML probe also stalls.** After the lease `finally`, `dispatch_images`
   unconditionally calls `_lease.wait_until_below_mb(15000)`. Skipping the lease
   for a network engine must ALSO skip this probe or it polls the local GPU
   after every cloud render. Gemini (grounded-confirmed).
5. **Episode budget is cross-phase.** `dispatch_images` (images) runs before the
   video dispatcher and only sees stills. An episode ceiling needs a running
   `ledger["billing"]` spent total both phases read/write. Gemini.
6. **Image has no duration.** The image request carries no duration/fps →
   image cost is strictly **per-run/per-image**; **per-second** applies only to
   the video engines. Gemini.
7. **One network marker.** Plan was inconsistent (`declared_isolation="network"`
   for LTX only; `is_network` alt). → Single source: a class attribute
   `is_network = True` on all three adapters, read at BOTH lease sites via
   `getattr(engine, "is_network", False)`. Avoids touching the capability-decl
   validator (keeps CAPABILITIES rows standard). GPT + DeepSeek.
8. **Output: prefer in-memory numpy.** `flux_gen1` returns a numpy uint8 array
   and `_coerce_pixels` accepts it; forcing a disk PNG adds `wait_for_file_ready`
   risk. → cloud image adapter returns uint8 numpy (convert the API node's
   IMAGE tensor) when possible, else a `.png` path; never a raw torch tensor.
   Gemini, DeepSeek.
9. **Auth source.** v1 = `OTR_COMFY_API_KEY` env ONLY; drop the un-grounded
   "then Comfy account" fallback. GPT, DeepSeek.
10. **Guarded import hides breakage.** The `try/except` in `__init__.py` would
    silently drop a broken cloud adapter. → log LOUD in the except; a test
    asserts the three rows are in `all_engine_names()` with the flag off. GPT.

## CORRECTED (plan was wrong about the UX mechanism)

- **Dropdown rows can't be "disabled."** The COMBO is the full static registry
  (V-6 forbids dynamic widget mutation). Cloud rows are always **visible**;
  usability is communicated only via `assert_usable`/report at execute time —
  there is no greyed-out widget. → Drop the "visible-disabled vs hidden"
  decision; document the static-visible + fail-closed reality. GPT cut #1.

## ACCEPTED should-fix (folded as spec)

- **`other_beats_video_model` + Kling Avatar is per-beat partial.** That slot
  spans character/scene/background; Kling fits only character. → Recommend Kling
  on `announcer_video_model`; if set on other-beats, character beats use it and
  scene/background **fail closed to the fallback, LOUD** (add a test). Optionally
  a dedicated `character_video_model` selector later. GPT.
- **Pin durations:** `cloud_ltx2` from `timing.target_frame_count / target_fps`
  (clamped min/max, cost-bounded); `cloud_kling_avatar` = `audio_ref` duration
  (probe WAV header), clamped + cost-bounded. GPT, DeepSeek.
- **Pin the HTTP client + lazy import** (e.g. `httpx`/`requests` inside
  `render_*` only) so cold-import stays clean. DeepSeek.
- **Add adapter metadata:** `provider`, `provider_model`, `pricing_version`,
  `terms_url`, `date_checked` for reports + commercial-clean review. GPT.

## ACCEPTED cut / trim

- Direct provider HTTP is the LAST fallback, not in the initial adapters. GPT.
- Cost accounting = ONE idempotent reserve per `request_id` + unknown-price
  fail-closed; no full billing subsystem in v1. GPT.
- Cost ceiling = env/config (`OTR_CLOUD_CREDIT_CEILING`), NOT a Director widget,
  in v1 (no JSON edit). DeepSeek.

## RECOMMENDED DEFAULTS (so the coder isn't blocked)

- Per-episode ceiling default ≈ **$5.00** (configurable); over-ceiling ⇒ skip
  that beat to the radio floor, LOUD.
- commercial_clean = **False** for all three until ToS confirmed (dated URL in
  metadata).

## OPEN / verify-at-build

- §5 invocation seam (all 3 reiterate): HARD-STOP S0 spike — pin invocation
  mode, key source, request schema, polling/cancel, output-file discovery — and
  update the plan with the pinned details BEFORE coding adapters.
- `MotionEngineBase.render_clip`/`canonicalize` exact return shape (not in the
  grounding set) — the cloud video adapters subclass it like
  eng_humo/eng_ltx_video; confirm the return contract when coding S3/S4.
- Confirm the video Director `INPUT_TYPES` builds its COMBO from the registry
  (assumed from the image dispatcher + V-6; verify before relying on
  "no JSON edit").

## Convergence

One grounded pass (as requested). Remaining items are pinned build constraints,
not research unknowns. A pass02 is optional after the S0 spike resolves the
invocation seam.
