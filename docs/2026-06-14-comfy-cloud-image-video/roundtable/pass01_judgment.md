# Roundtable pass 01 — judgment log

Panel: GPT-5.5 (`openai/gpt-5.5-20260423`), Gemini 3.1 Pro
(`google/gemini-3.1-pro-preview-20260219`), DeepSeek-v4-pro
(`deepseek/deepseek-v4-pro-20260423`). Spend ≈ $0.19.
Judge: Claude (grounded every claim against the real files).
Grounding: `otr_image_gen_dispatcher.py`, `_otr_video_engines/registry.py`,
`_otr_image_engines/registry.py`, plus judge-side checks of
`_otr_video_engines/schemas.py`, `_otr_image_engines/registry.py` CAPABILITIES,
and `_otr_shared/fallback.py`.

All three returned VERDICT: no — for the **same** core reason: the research doc
overclaimed "zero dispatcher change / per-adapter behavior" in places the
grounded dispatcher contradicts. High-confidence convergence; folded.

## CONFIRMED (grounded against code) → folded into the doc

1. **GPU lease is unconditional.** `dispatch_images` calls `_lease.acquire()`
   immediately before `gen_fn(request)` with no engine check. A cloud call
   would hold the local GPU lease for the whole network wait. → The dispatcher
   (not the adapter alone) must skip the lease for a network engine. All 3.

2. **Cost guard cannot live in `assert_usable`.** Dispatcher calls
   `_ireg.assert_usable(engine_id, role)` — stateless, no request/budget/dims/
   duration context. → Cost guard needs the full request + a budget-accounting
   step, not `assert_usable`. All 3.

3. **CAPABILITIES row required.** Both registries have a `CAPABILITIES` dict
   keyed by engine, consumed by `capability_profiles.py`; no network class
   exists. → Every cloud engine needs a row (`vram_class="cpu"`,
   `vram_estimate_mb=0`, `cpu_ok=True`, `requires_sidecar=False`). All 3.

4. **Image fallback ladder overclaimed.** `dispatch_images` catches render
   exceptions and *skips the object* (→ radio floor); it does NOT try a local
   engine. `_otr_shared/fallback.py` is real but wired into the **video**
   render path, not the image dispatcher. → Correct the doc: image cloud
   failure degrades straight to radio floor (accept this for v1); only video
   can use the fallback resolver. All 3.

5. **Partner-Node invocation seam underspecified.** "Calls an API node under
   the hood" is not buildable as written. Must specify: direct provider HTTP
   call vs. importing the partner-node class vs. sub-graph execution; the
   API-key injection point; polling; output-file discovery; cold-import
   safety. GPT + DeepSeek. (UNVERIFIABLE against OTR code — it's a ComfyUI
   internals question → recorded as a verify-at-build spike.)

6. **Output type coercion.** `_coerce_pixels` accepts a `.png` path OR a numpy
   array (`.tobytes`). A raw ComfyUI IMAGE tensor would hit its `TypeError`. →
   The cloud image adapter must return a path or convert tensor→uint8 numpy.
   Gemini, grounded-confirmed.

7. **cloud_auto cache-key collision.** Cache key includes `engine_id` +
   `engine_version` but not the resolved underlying model. A single
   `cloud_auto` that switches Flux↔Nano↔Luma would reuse stale results under
   one key. → Resolve to a concrete model BEFORE the key and fold the resolved
   model id into `engine_version`/key. GPT.

8. **`cloud_auto` slot routing.** `slot` is local to `dispatch_images` and not
   in `request`; but `request["role"]` IS present (confirmed in code) and the
   video `VideoRequest` carries `role` + `audio_ref` + `init_image`. →
   role-based routing works; per-slot auto engine IDs are the simpler,
   more-testable alternative. GPT + Gemini + DeepSeek.

## RESOLVED BY GROUNDING (panel assumption corrected, no doc change needed)

- GPT should-fix #7 "audio_ref not grounded for video": `VideoRequest`
  (`schemas.py`) carries `audio_ref` + `init_image`; `audio_driven_face`
  requires both. Kling Avatar gets what it needs. Announcer LTX-2 fallback
  (`image_to_video`) needs only `init_image`, which the still pipeline
  produces (resolves DeepSeek should-fix 2b.5).
- Gemini #4 "cache key misunderstanding": correct — the doc's open question
  "hash-the-result vs key-on-request" is already answered by `request_cache_key`
  (keys on request; output hash is only the filename). → Open question removed;
  replaced with the cloud_auto resolved-model nuance (#7 above).

## ACCEPTED should-fix (folded as build constraints)

- Auth: name the env var, define headless key readability, surface missing key
  as `EngineUnusable` in the dropdown/report (GPT, Gemini).
- Per-cloud-adapter network timeouts: connect + wall-clock + poll interval +
  429/5xx retry + cancel (GPT, DeepSeek).
- `commercial_clean`: hardcoded **dated** table, unknown ⇒ False, pin a source
  URL in adapter metadata (all 3).
- Dated price table; stale/unknown price ⇒ fail closed (GPT).
- Budget accounting (reserve/spent, idempotent across retries), not just a
  pre-estimate (GPT, DeepSeek).
- Dropdown UX: decide visible-but-disabled vs hidden when `OTR_ENABLE_CLOUD`
  off (GPT).

## ACCEPTED cut / scope-trim (panel consensus)

- **Curated v1 set only** — do not surface all 60+ models. One image, one
  motion, one talking-face, plus the easy option. (Doc already recommended;
  reinforced.) All 3.
- **No dynamic ToS fetching** — hardcoded dated table. GPT, DeepSeek.
- **No full OpenRouter-style guard** — minimal per-episode ceiling with
  deterministic reserve/spent. GPT, DeepSeek.
- **Simplify `cloud_auto`** — implement as per-slot auto engine IDs
  (`cloud_auto_image` / `cloud_auto_motion` / `cloud_auto_talking`) instead of
  one universal slot-inference engine. Satisfies the operator's "easy option"
  while removing hidden dependence on un-passed request fields. All 3
  (reconciled: kept the feature, changed the implementation).

## REJECTED

- "Cut cloud_auto entirely from v1" (DeepSeek/GPT alt) — REJECTED: the operator
  explicitly asked for an easy one-click cloud option. Kept, but implemented as
  per-slot auto IDs (above).

## OPEN / verify-at-build

- Can ComfyUI Partner/API nodes be invoked programmatically from
  `_inprocess_gen_fn` / `render_image` outside the graph executor, honoring the
  Comfy API key + credit billing? (All 3 flagged as an assumption.) Spike
  before committing to the import-the-node-class path; the direct-HTTP path is
  the fallback.
- Registration/import bootstrap: confirm the existing adapter import path and
  add cloud modules to it; add a cold-import test that `all_engine_names()`
  lists the cloud rows with `OTR_ENABLE_CLOUD` off.

## Convergence

One grounded pass (as requested). Not run to formal convergence — the
must-fixes are design-spec gaps, now resolved into build constraints, not
research-blocking unknowns. A pass02 is optional and only worthwhile once a
draft sprint plan exists to harden.
