# nano_banana_2 + seedream_2 render fixes (2026-07-05)

## Root causes (grounded in the LIVE partner-node source)
Both are the SAME class: the V3 `model` input is a DICT the node destructures,
and our adapters (written against an older node rev) under-populated it.

1. **cloud_nano_banana_2** -- CONFIRMED live failure:
   `GeminiNanoBanana2V2.execute` (comfy_api_nodes/nodes_gemini.py:1528+) reads
   `model["model"]`, `model["resolution"]`, `model["aspect_ratio"]`. The adapter
   sent `{"model": <id>}` only -> `KeyError: 'resolution'` -> surfaced as
   `provider_rejected -- 'resolution'` (the bake-off failure). resolution options
   = ["1K","2K","4K"]; aspect_ratio "auto".
   FIX: `"model": {"model": <id>, "resolution": "1K", "aspect_ratio": "auto"}`
   (both env-overridable: OTR_CLOUD_NANO_RESOLUTION / OTR_CLOUD_NANO_ASPECT).

2. **cloud_seedream_2** -- LATENT (never ran; bake-off died earlier in my harness
   at node-1 widgets, unrelated). `ByteDanceSeedreamNodeV2.execute`
   (nodes_bytedance.py:790+) reads `model["model"]`; `size_preset`/`width`/`height`
   use `model.get(..., default)` (safe). The adapter sent `model` as a BARE
   STRING -> `model["model"]` would raise "string indices must be integers".
   FIX: `"model": {"model": <id>}` (only the "model" key is required).

## Why the conformance guard missed it (CORRECTED per codex r3)
nano/seedream are NOT in KNOWN_NONBUILDABLE -- `test_emitted_kwargs_are_declared`
DOES build them, but it only checks TOP-LEVEL emitted kwargs against the pinned
top-level schema; it cannot see missing NESTED keys under the `model` dict. So the
V3 model-dict under-population slipped through. FOLLOW-UP (queued): add a V3
model-dict shape assertion to the conformance guard (cross-check the nested keys
each partner node destructures) so this class of bug is caught at build time.
The dedicated `tests/test_cloud_image_adapters.py` nano/seedream tests now pin the
full model dicts (updated this commit).

## Open (separate from the adapters)
- The bake-off DRIVER (scripts/_otr_anime_bakeoff.py, scratch) hit a node-1
  widgets mismatch (OTR_LedgerScriptWriter len(wv)=25 vs 27) on the seedream leg
  ONLY -- earlier legs patched node 1 fine. Harness-side; investigate whether it
  is a real saved-JSON-vs-live-schema drift or a driver-only glitch (Sonnet wiring
  check). Does NOT touch production adapters.

## Sonnet fan-out findings (folded in)
- nano needed a 4TH key `model["thinking_level"]` (nodes_gemini.py:812) that the
  first fix missed -> ADDED (options ["LOW","HIGH"], default LOW; env-overridable).
  Sonnet recalled the options as MINIMAL/HIGH -- VERIFIED against source = LOW/HIGH,
  used LOW.
- `response_modalities` is compared `== "IMAGE"` (:1024) -> default corrected
  "Image" -> "IMAGE" (was silently requesting IMAGE+TEXT).
- SAME latent bug found in `cloud_wan_i2v` (VIDEO): `Wan2ImageToVideoApi.execute`
  (nodes_wan.py:1090+) destructures model["model"/"prompt"/"negative_prompt"/
  "resolution"/"duration"]; the adapter sends `model` as a bare string + none of
  those. It is a KNOWN-DARK row (comment: awaits the S1 V3-expansion pin) -> NOT
  fixed here; QUEUED as a separate item. cloud_seedance_2 self-disables (dark).
- All other cloud image + video adapters (recraft/flux_pro/ideo/kling*/word_razzle)
  VERIFIED clean (scalar params, no dict destructure).

## Verify
- Sonnet fan-out: confirm the two model dicts match exactly what the live nodes
  destructure; scan the OTHER cloud adapters (recraft/flux_pro/ideo) + the cloud
  VIDEO adapters for the same bare-string-vs-dict V3 landmine.
- Live re-run nano_banana_2 + seedream_2 -> obs episode, no KeyError/TypeError.
- Full suite + Bug Bible + B7 green; NOBOM; push.

## Live proof follow-up (Codex continuation)
- 2026-07-05 quick probe with `cloud_nano_banana_2 + viz_green + 30 words`
  succeeded and published to OBS, but `viz_green` ignores init images, so it did
  NOT exercise the cloud still adapter.
- 2026-07-05 quick probe with `cloud_nano_banana_2 + word_razzle + 30 words`
  DID exercise stills and found the real remaining provider issue:
  `thinking_level` reached Vertex as `thinkingConfig.thinkingLevel`, and Vertex
  rejected it for the pinned Nano Banana image model:
  "thinking_level is not supported by this model."
- Root fix: OTR's `invoke_partner_node` bridge now has a scoped
  `GeminiNanoBanana2V2` path that preserves the pinned Comfy auth/session/sync
  machinery but builds the Gemini image request without `thinkingConfig`. The
  adapter no longer emits `thinking_level`; it emits the required
  model/resolution/aspect_ratio dict plus uppercase `IMAGE`.
- The scoped bridge path also mirrors the partner node by NOT forwarding
  `seed` into `generationConfig`; Vertex treats that field as INT32 while OTR
  request seeds are 64-bit, causing rejects such as
  `Invalid value at 'generation_config.seed' (TYPE_INT32), 2573885473`.
- 2026-07-05 style smoke with `cloud_seedream_2 + still_word + cartoon + 30
  words` found the same provider-contract class in the ByteDance row: the
  pinned Seedream request schema rejects seeds above signed INT32. Root fix:
  Seedream still seeds are folded deterministically into signed INT32 range
  before partner invocation. Live rerun minted all Seedream stills and published
  `signal_lost_signals_echo_20260705_192453_silent_procgen_blended_captioned_with_credits_final.mp4`
  to OBS.
