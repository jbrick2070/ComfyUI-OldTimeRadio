# Roundtable — S0 spike + sprint — judgment log

Panel: GPT-5.5, Gemini 3.1 Pro, DeepSeek-v4-pro. Spend ≈ $0.21.
Grounding: `otr_image_gen_dispatcher.py`, `motion_common.py`, `flux_gen1.py`,
`role_compat.py`.
Verdicts: Gemini + DeepSeek **yes-with-fixes**; GPT **no** (spike rigor, not
architecture). The seam (call `comfy_api_nodes.util.client` directly with an
explicit key) is validated; fixes are about proving it in the right context and
pinning billing/error contracts.

## CONFIRMED → folded into S0 doc

1. **Spike must prove the EXECUTOR-THREAD / headless `/prompt` path, not just a
   standalone script (GPT, DeepSeek; Gemini refine).** A venv script proves
   imports/signatures only. → S0 adds an in-graph `/prompt` smoke (throwaway
   debug node/hook calling the same util-client code inside a real prompt exec
   with `OTR_COMFY_API_KEY`), and tests inside a plain `threading.Thread` (no
   event loop) to mirror `PromptExecutor`. The standalone script stays as the
   import/signature probe only.
2. **Avoid pulling torch into the image network path (Gemini cut).** Use
   `download_url_to_bytesio` → `PIL.Image` → uint8 numpy (or save `.png` + return
   the path), NOT `download_url_to_image_tensor`. Lighter and matches
   `_coerce_pixels` (path or numpy). Folded into §2.
3. **`assert_usable` must also check `comfy_api_nodes` is importable (Gemini).**
   Key present + package missing ⇒ hard crash at render. → cold `assert_usable`
   = `OTR_ENABLE_CLOUD` on + `OTR_COMFY_API_KEY` non-empty +
   `importlib.util.find_spec("comfy_api_nodes")` present (find_spec, no execute);
   live key validation at smoke; render handles 401/403 fail-closed (GPT split).
4. **Billing reservation ordering, per object/clip (GPT, DeepSeek).** Reserve
   AFTER cache-hit check + `assert_usable`, BEFORE the first billed util call;
   commit on success; **release on ANY failure** (auth/upload/submit/poll-
   timeout/download/canonicalize). Folded into §5 S1.
5. **Cost guard = price-table ESTIMATE (primary); observed billed cost =
   optional telemetry (reconcile GPT cut vs Gemini channel).** The deterministic
   gate uses the dated price table (GPT: don't depend on one observed charge).
   Recording the real billed cost needs `gen_fn -> (result, meta)` — kept as a
   SHOULD, not required for v1 (avoids a dispatcher signature change). Folded.
6. **Error-handling contract (DeepSeek).** Define adapter exceptions for
   auth / timeout / HTTP / rate-limit, mapped to the existing fail-closed
   contract (dispatcher warns + skips → radio floor). Folded into §5.
7. **Executor-thread async (GPT, DeepSeek).** If `sync_op`/`poll_op` are async,
   the spike documents the exact sync wrapper (e.g. `asyncio.run` vs a loop
   already running) that doesn't clash with Comfy's loop. Folded into §3/§4.
8. **Secrets hygiene (GPT, DeepSeek).** `S0_RESULTS.md` redacts the key, auth
   headers, signed URLs, task ids, local user paths — only module paths,
   signatures, redacted result shape, cost source. The key is NEVER logged or
   put in error messages. Folded into §3.
9. **Spike live-spend gate (GPT).** The spike aborts unless
   `OTR_RUN_LIVE_CLOUD_SPIKE=1` and prints the planned billed calls first.
   Folded into §3.
10. **Atomic video write (GPT).** temp file → flush/fsync → atomic rename →
    verify nonzero before returning the path. Folded into §2.
11. **`cloud_ltx2.required_inputs = ("text_prompt",)` only (GPT, Gemini).**
    init_image optional at render time, else `background_abstract` is excluded.
    Already in the wiring plan; reaffirmed in §5 S3.
12. **`extra_data` auth path is NOT equivalent (GPT).** The direct util-client
    seam doesn't read `/prompt` `extra_data`. → demoted from "equivalent
    alternative" to a documented FALLBACK to verify in the spike only if the
    explicit-key route is fragile (DeepSeek agrees). Folded into §2.
13. **Return-value validation (GPT should).** `cloud_flux_pro.render_image`
    asserts ndarray `(H,W,3)` uint8, nonzero dims, before returning. Folded.

## CONFIRMED already-present (no change, reaffirmed)

- Lease-skip in `dispatch_images` (image) + `MotionEngineBase.prepare()` override
  (video) — already specced in the wiring plan; S0 sprint S1 carries the test
  "fails if `_lease.acquire` is called for a network engine."

## ACCEPTED cut / defer

- Separate "confirm `/proxy/`" gate — unnecessary once the util-layer signatures
  are pinned (GPT).
- `commercial_clean` dated-table polish — defer to after the three adapters pass
  live smoke (GPT). Stays S5.
- Per-call observed credit as an adapter-coding input — use the price table (GPT;
  reconciled in #5).

## OPEN / verify-in-spike

- Exact `util.client` module path + `sync_op`/`poll_op` signatures + auth arg
  name on the INSTALLED Comfy Desktop build (the spike pins these).
- Whether the response exposes a billed-cost field (drives the optional
  telemetry channel #5).
- The §2 headless-`None`-auth claim is from ComfyUI issues #13222/#8344/#11481 —
  the spike confirms it on the installed build.

## Convergence

The seam is validated; remaining items are pinned spec + the spike itself. After
S0 runs green and its results are folded, the plan is build-ready. No further
roundtable needed pre-spike.
